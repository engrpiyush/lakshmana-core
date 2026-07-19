"""The model roster (LLD §8).

Every gate slot is model-agnostic: the artifact and the thresholds both come from
``configSnapshot``, so LK-5's bake-off can swap any row without touching gate code. This
module is the catalogue that maps a config value like ``modernbert-base-nli@v1`` onto the
upstream checkpoint an owner-run mirror pulls from.

The v1 rows are the LLD's proposal; ``ALTERNATES`` are the candidates the replay
bake-off (VA-97) sweeps against them. Both are mirrored *before* session03 so a NO-GO can
re-sweep without waiting on another download.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ALTERNATES",
    "ALTERNATE_BY_NAME",
    "V1_ROSTER",
    "ArtifactRef",
    "ExportKind",
    "ModelTask",
    "RosterEntry",
    "entry_for",
    "known_names",
    "parse_ref",
]

_REF_PATTERN = re.compile(r"^(?P<name>[a-z0-9]+(?:-[a-z0-9]+)*)@(?P<version>v[0-9]+)$")


class ModelTask(StrEnum):
    """What the head produces, which is what the gate code has to interpret."""

    NLI_3WAY = "NLI_3WAY"
    """entailment / neutral / contradiction logits — G1 and G3."""

    GROUNDING = "GROUNDING"
    """a single support score for (document, claim) — G2's native input shape."""


class ExportKind(StrEnum):
    """How the checkpoint becomes ONNX."""

    SEQUENCE_CLASSIFICATION = "SEQUENCE_CLASSIFICATION"
    """Plain ``optimum`` export. Everything on the roster that is a standard encoder."""

    CUSTOM = "CUSTOM"
    """Needs ``trust_remote_code`` and a hand-written export path.

    The prep script refuses these rather than exporting something that loads but scores
    nonsense — a quantized wrong-architecture model is the worst failure mode here,
    because it produces plausible numbers.
    """


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A parsed ``name@version``, independent of whether the roster knows it."""

    name: str
    version: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def prefix(self) -> str:
        return f"{self.name}/{self.version}"


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One mirrorable artifact."""

    name: str
    """Artifact name — the GCS prefix and the left half of a config ref."""

    version: str
    """Artifact version — the right half of a config ref, and the GCS subdirectory."""

    checkpoint: str
    """Upstream Hugging Face repo id. Read by the prep script only, never at runtime."""

    task: ModelTask
    role: str
    approx_int8_mb: int
    export: ExportKind = ExportKind.SEQUENCE_CLASSIFICATION
    notes: str = ""

    label_order: tuple[str, ...] = ()
    """Class name per output column, for checkpoints whose ``config.json`` does not say.

    Empty means "read ``id2label``", which is the normal case and the one to prefer: the
    artifact then carries its own meaning. Some checkpoints ship the transformers default
    ``LABEL_0/1/2``, and for those the order has to come from the model card — recorded
    here, and verified against the sanity fixture by ``scripts/probe_label_order.py``
    rather than trusted. Column order genuinely differs across the roster (this file's
    own G1 and G3 rows disagree), so a positional guess is not available.
    """

    @property
    def ref(self) -> str:
        """The ``name@version`` string a config value carries."""
        return f"{self.name}@{self.version}"

    @property
    def prefix(self) -> str:
        """GCS object prefix: ``<name>/<version>``."""
        return f"{self.name}/{self.version}"


# --- v1: the LLD §8 proposal -----------------------------------------------------------

V1_ROSTER: tuple[RosterEntry, ...] = (
    RosterEntry(
        name="modernbert-base-nli",
        version="v1",
        checkpoint="tasksource/ModernBERT-base-nli",
        task=ModelTask.NLI_3WAY,
        role="G1 — NEUTRAL gate, REPEATS, and the contradiction pre-signal",
        approx_int8_mb=150,
    ),
    RosterEntry(
        name="minicheck-deberta-l",
        version="v1",
        checkpoint="lytang/MiniCheck-DeBERTa-v3-Large",
        task=ModelTask.GROUNDING,
        role="G2 — grounding/support checker producing CORROBORATES",
        approx_int8_mb=435,
    ),
    RosterEntry(
        name="deberta-mnli-fever-anli",
        version="v1",
        checkpoint="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        task=ModelTask.NLI_3WAY,
        role="G3 — second-family contradiction cross-check",
        approx_int8_mb=185,
        notes="Deliberately a different family from G1: cross-architecture agreement is "
        "what replaces the old 5-vote self-consistency.",
    ),
)


# --- bake-off alternates ---------------------------------------------------------------

ALTERNATES: tuple[RosterEntry, ...] = (
    RosterEntry(
        name="nli-deberta-v3-small",
        version="v1",
        checkpoint="cross-encoder/nli-deberta-v3-small",
        task=ModelTask.NLI_3WAY,
        role="G1/G3 alternate — the cheap end of the curve",
        approx_int8_mb=70,
    ),
    RosterEntry(
        name="ettinx-nli-s",
        version="v1",
        checkpoint="dleemiller/EttinX-nli-s",
        task=ModelTask.NLI_3WAY,
        role="G1/G3 alternate — the LLD's 'EttinX-s' row",
        approx_int8_mb=140,
        label_order=("contradiction", "entailment", "neutral"),
        notes="Ships the transformers default `LABEL_0/1/2`, so the column order cannot "
        "be read off the artifact. Taken from the model card's `label_mapping` and "
        "confirmed on the sanity fixture by scripts/probe_label_order.py (2026-07-19): "
        "98% agreement, 62-point margin over the runner-up permutation.",
    ),
    RosterEntry(
        name="vitaminc-mnli",
        version="v1",
        checkpoint="tals/albert-xlarge-vitaminc-mnli",
        task=ModelTask.NLI_3WAY,
        role="G3 alternate — trained on contrastive evidence, so contradiction recall is "
        "its whole point",
        approx_int8_mb=210,
    ),
    RosterEntry(
        name="factcg-deberta-l",
        version="v1",
        checkpoint="yaxili96/FactCG-DeBERTa-v3-Large",
        task=ModelTask.GROUNDING,
        role="G2 alternate — the LLD's conditional 'if the checkpoint verifies' row",
        approx_int8_mb=435,
        notes="Checkpoint verified to exist on the Hub (2026-07-19); its scoring "
        "behaviour is unproven here and that is exactly what VA-97 measures.",
    ),
    RosterEntry(
        name="hhem",
        version="v1",
        checkpoint="vectara/hallucination_evaluation_model",
        task=ModelTask.GROUNDING,
        export=ExportKind.CUSTOM,
        role="G2 alternate — HHEM",
        approx_int8_mb=0,
        notes="Ships as a custom `trust_remote_code` architecture wrapping its own "
        "backbone, so `optimum` cannot export it from the standard "
        "sequence-classification path. Mirroring it needs a hand-written export; the "
        "prep script refuses it until that exists rather than quantizing the wrong "
        "graph. Not a blocker for VA-97 — there are two other G2 candidates.",
    ),
)


ALTERNATE_BY_NAME: dict[str, RosterEntry] = {entry.name: entry for entry in ALTERNATES}
_ALL: dict[str, RosterEntry] = {entry.ref: entry for entry in (*V1_ROSTER, *ALTERNATES)}


def parse_ref(ref: str) -> ArtifactRef:
    """Split ``name@version``.

    Raises:
        ValueError: on anything that is not exactly ``kebab-name@vN``. Config carries
            these strings, and a typo must fail at load rather than resolve to a GCS
            prefix that happens not to exist.
    """
    match = _REF_PATTERN.match(ref)
    if match is None:
        raise ValueError(f"{ref!r} is not a valid artifact ref; expected 'name@vN'")
    return ArtifactRef(name=match["name"], version=match["version"])


def entry_for(ref: str) -> RosterEntry:
    """Look up a roster entry by ``name@version``.

    Raises:
        KeyError: if the ref is not on the roster. The loader does *not* go through
            here — an artifact in the bucket is loadable whether or not this catalogue
            lists it, because a bake-off winner should not need a code change to run.
            This is for the prep script, which does need to know where to pull from.
    """
    parse_ref(ref)
    try:
        return _ALL[ref]
    except KeyError:
        raise KeyError(
            f"{ref!r} is not on the roster; known refs: {', '.join(sorted(_ALL))}"
        ) from None


def known_names() -> tuple[str, ...]:
    """Every ref the catalogue knows, v1 and alternates alike."""
    return tuple(sorted(_ALL))
