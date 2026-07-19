"""Batched ONNX encoder inference (LLD §8).

This is the layer between a verified :class:`~gatekeeper.models.loader.Artifact` and the
gate decision functions: it turns ``(premise, hypothesis)`` text pairs into the
probabilities the thresholds read. It runs ONNX Runtime and the Rust ``tokenizers``
library directly — no torch, no transformers, none of the model-prep extra.

Two things here are deliberately strict rather than convenient.

**Label mapping is resolved from ``config.json``, and refuses to guess.** An NLI head's
three logits are only meaningful once you know which column is contradiction, and that
order is not standardised across checkpoints — MoritzLaurer, tasksource and cross-encoder
all differ. A wrong mapping does not crash; it silently swaps entailment for
contradiction, which in this cascade means contradictions discarded as NEUTRAL at G1.
That is the single most expensive failure the gatekeeper can have, so an unmappable
``id2label`` is a hard error rather than a positional fallback.

**Truncation is reported, not swallowed.** G1 counts ``truncated`` and the LLD has a
``TRUNCATED`` escalation reason; a pair whose evidence was cut off has not really been
judged.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gatekeeper.config import Config, GateBinding
from gatekeeper.enums import Gate
from gatekeeper.errors import ErrorCode, GatekeeperError
from gatekeeper.logging import get_logger
from gatekeeper.models.loader import Artifact
from gatekeeper.models.roster import ModelTask

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EncoderScorer",
    "GroundingScore",
    "NliScores",
    "ScorerError",
    "scorer_for_binding",
    "scorer_for_gate",
]

log = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32
"""LLD §8 G1: 'Run NLI forward + backward, batch 32.'"""

_ENTAILMENT_PREFIXES = ("entail", "support")
_NEUTRAL_PREFIXES = ("neutral", "not_enough_info", "nei")
_CONTRADICTION_PREFIXES = ("contradict", "refut", "not_entail")
"""Three label vocabularies appear on the roster and they do not agree.

MNLI checkpoints say entailment/neutral/contradiction; FEVER-trained ones (VitaminC) say
SUPPORTS/REFUTES/NOT ENOUGH INFO; two-way heads spell the negative class
``not_entailment``. All three are recognised by prefix. ``not_enough_info`` is neutral and
``not_entailment`` is contradiction, which is why the neutral spelling is matched in full
rather than on ``not``.
"""

_THREE_WAY_COLUMNS = 3


class ScorerError(GatekeeperError):
    """A model whose outputs cannot be interpreted — same class of failure as a bad fetch.

    An artifact that loads but whose head cannot be mapped is not usable, and pretending
    otherwise would produce confident nonsense. It reuses ``GK_E_MODEL_FETCH`` because the
    operator action is identical: fix the artifact, retrigger FROM_GATE.
    """

    code = ErrorCode.GK_E_MODEL_FETCH


@dataclass(frozen=True, slots=True)
class NliScores:
    """Three-way NLI probabilities for one ordered pair."""

    entailment: float
    neutral: float
    contradiction: float
    truncated: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "entailment": self.entailment,
            "neutral": self.neutral,
            "contradiction": self.contradiction,
        }


@dataclass(frozen=True, slots=True)
class GroundingScore:
    """A single support probability for one ordered (document, claim) pair."""

    support: float
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _LabelMap:
    """Which output column is which class."""

    entailment: int
    neutral: int | None
    contradiction: int


def _resolve_label_map(id2label: dict[str, str], columns: int) -> _LabelMap:
    """Map an ``id2label`` onto the three NLI classes.

    Raises:
        ScorerError: if any class is missing or ambiguous. Guessing here would invert
            entailment and contradiction on some checkpoints, and nothing downstream
            would notice.
    """
    found: dict[str, int] = {}
    for raw_index, raw_label in id2label.items():
        label = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
        for name, prefixes in (
            ("entailment", _ENTAILMENT_PREFIXES),
            ("neutral", _NEUTRAL_PREFIXES),
            ("contradiction", _CONTRADICTION_PREFIXES),
        ):
            if label.startswith(prefixes):
                if name in found:
                    raise ScorerError(
                        f"id2label maps two columns to {name} ({found[name]} and {raw_index}); "
                        "the head cannot be interpreted"
                    )
                found[name] = int(raw_index)

    missing = [name for name in ("entailment", "contradiction") if name not in found]
    if missing:
        raise ScorerError(
            f"id2label {id2label!r} does not name {', '.join(missing)}; refusing to guess "
            "the column order — a wrong mapping silently discards contradictions as NEUTRAL"
        )
    if columns >= _THREE_WAY_COLUMNS and "neutral" not in found:
        raise ScorerError(f"id2label {id2label!r} has {columns} columns but names no neutral class")
    return _LabelMap(
        entailment=found["entailment"],
        neutral=found.get("neutral"),
        contradiction=found["contradiction"],
    )


class EncoderScorer:
    """One loaded artifact, ready to score text pairs.

    Construction is the expensive part (an ONNX session over a few hundred megabytes), so
    a scorer is built once per gate per worker and reused across batches.
    """

    __slots__ = (
        "_artifact",
        "_batch_size",
        "_input_names",
        "_label_map",
        "_max_seq_tokens",
        "_session",
        "_task",
        "_tokenizer",
    )

    def __init__(
        self,
        artifact: Artifact,
        task: ModelTask,
        *,
        max_seq_tokens: int = 512,
        batch_size: int = DEFAULT_BATCH_SIZE,
        intra_op_threads: int = 0,
        label_order: Sequence[str] = (),
    ) -> None:
        # ONNX Runtime and `tokenizers` are hard runtime dependencies, but importing them
        # costs about a second, and most of this package's importers (the decision
        # functions, the metrics, every test that never touches a model) never build a
        # scorer at all. Paying it here means paying it only when a model is loaded.
        import onnxruntime
        from tokenizers import Tokenizer

        self._artifact = artifact
        self._task = task
        self._max_seq_tokens = max_seq_tokens
        self._batch_size = batch_size

        options = onnxruntime.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self._session = onnxruntime.InferenceSession(
            str(artifact.model_path), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {tensor.name for tensor in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(str(artifact.tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_seq_tokens)
        self._tokenizer.enable_padding()

        config = json.loads(artifact.config_path.read_text(encoding="utf-8"))
        id2label = dict(enumerate(label_order)) if label_order else (config.get("id2label") or {})
        id2label = {str(index): label for index, label in id2label.items()}
        columns = len(id2label)
        self._label_map = (
            _resolve_label_map(id2label, columns) if task is ModelTask.NLI_3WAY else None
        )

        log.info(
            "scorer ready",
            fields={
                "artifact": artifact.ref.ref,
                "precision": str(artifact.precision),
                "task": str(task),
                "columns": columns,
                "maxSeqTokens": max_seq_tokens,
            },
        )

    @property
    def artifact(self) -> Artifact:
        return self._artifact

    # --- scoring ----------------------------------------------------------------------

    def score_nli(self, pairs: Sequence[tuple[str, str]]) -> list[NliScores]:
        """Three-way probabilities for ``(premise, hypothesis)`` pairs, in order."""
        if self._label_map is None:
            raise ScorerError(f"{self._artifact.ref.ref} is not a 3-way NLI head")

        results: list[NliScores] = []
        for probabilities, truncated in self._run(pairs):
            mapping = self._label_map
            for row, was_truncated in zip(probabilities, truncated, strict=True):
                results.append(
                    NliScores(
                        entailment=float(row[mapping.entailment]),
                        neutral=float(row[mapping.neutral]) if mapping.neutral is not None else 0.0,
                        contradiction=float(row[mapping.contradiction]),
                        truncated=bool(was_truncated),
                    )
                )
        return results

    def score_grounding(self, pairs: Sequence[tuple[str, str]]) -> list[GroundingScore]:
        """Support probabilities for ``(document, claim)`` pairs, in order.

        Grounding checkpoints come in two shapes — a single support logit, or a two-way
        unsupported/supported head — and which one is in front of us is read off the graph
        rather than configured.

        For the two-way shape the *last* column is the supported one. That is MiniCheck's
        documented convention ("1 for supported, 0 for unsupported") and it was confirmed
        for both mirrored G2 candidates on 2026-07-19 by scoring known supported and
        unsupported pairs; neither ships a usable ``id2label`` to read it from.
        """
        results: list[GroundingScore] = []
        for probabilities, truncated in self._run(pairs):
            for row, was_truncated in zip(probabilities, truncated, strict=True):
                if row.shape[-1] == 1:
                    support = float(row[0])
                elif self._label_map is not None:
                    support = float(row[self._label_map.entailment])
                else:
                    support = float(row[-1])
                results.append(GroundingScore(support=support, truncated=bool(was_truncated)))
        return results

    # --- internals --------------------------------------------------------------------

    def _run(self, pairs: Sequence[tuple[str, str]]) -> Iterator[tuple[np.ndarray, list[bool]]]:
        """Yield ``(probabilities, truncated)`` per batch."""
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            encodings = self._tokenizer.encode_batch([(first, second) for first, second in batch])

            feeds: dict[str, np.ndarray] = {
                "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
            }
            # ModernBERT and other RoBERTa-lineage graphs take no segment ids at all;
            # feeding an input the graph does not declare is an ORT error, not a no-op.
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.asarray(
                    [e.type_ids for e in encodings], dtype=np.int64
                )
            feeds = {name: value for name, value in feeds.items() if name in self._input_names}

            logits = self._session.run(None, feeds)[0]
            yield (
                self._to_probabilities(logits),
                [len(encoding.ids) >= self._max_seq_tokens for encoding in encodings],
            )

    def _to_probabilities(self, logits: np.ndarray) -> np.ndarray:
        if logits.shape[-1] == 1:
            return 1.0 / (1.0 + np.exp(-logits))
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


# --- wiring ------------------------------------------------------------------------------


def roster_label_order(name: str) -> tuple[str, ...]:
    """The declared column order for an artifact name, or ``()`` to read ``id2label``.

    The roster is consulted for this and nothing else. An artifact the catalogue has never
    heard of still loads and still scores — a bake-off winner must not need a code change
    to run (see :func:`~gatekeeper.models.roster.entry_for`) — it just has to name its own
    classes in ``config.json``.
    """
    from gatekeeper.models.roster import ALTERNATE_BY_NAME, V1_ROSTER

    known = {entry.name: entry for entry in V1_ROSTER} | ALTERNATE_BY_NAME
    entry = known.get(name)
    return entry.label_order if entry else ()


def scorer_for_artifact(
    artifact: Artifact,
    *,
    max_seq_tokens: int = 512,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EncoderScorer:
    """Build a scorer for a loaded artifact, task and label order resolved for you.

    Prefer this over constructing :class:`EncoderScorer` directly: forgetting the label
    order for a checkpoint that ships ``LABEL_0/1/2`` is a silent correctness bug, and the
    only way to not forget it is to not have to remember.
    """
    return EncoderScorer(
        artifact,
        ModelTask(artifact.manifest.task),
        max_seq_tokens=max_seq_tokens,
        batch_size=batch_size,
        label_order=roster_label_order(artifact.ref.name),
    )


def scorer_for_binding(
    artifact: Artifact, binding: GateBinding, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> EncoderScorer:
    """Build the scorer a run's frozen binding calls for.

    The token budget is frozen alongside the model: truncation changes scores, so a run
    that started at 512 tokens must not finish at 256 because config moved underneath it.
    """
    return scorer_for_artifact(
        artifact,
        max_seq_tokens=binding.max_seq_tokens,
        batch_size=batch_size,
    )


def scorer_for_gate(
    artifact: Artifact, gate: Gate, config: Config, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> EncoderScorer:
    """Build the scorer a gate needs from *live* config — the bake-off/tooling path.

    Resolves through :meth:`Config.snapshot` so it agrees with what a real run freezes.
    """
    return scorer_for_binding(
        artifact,
        GateBinding.from_snapshot(config.snapshot(), gate),
        batch_size=batch_size,
    )
