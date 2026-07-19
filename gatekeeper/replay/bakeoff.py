"""The replay bake-off: score once, sweep thresholds many times (LLD §15, VA-97).

The expensive thing is inference and the cheap thing is a comparison, so the two are
separated absolutely. :func:`score_corpus` runs every candidate artifact over every pair
exactly once and caches the raw probabilities; :func:`replay` then re-runs the *production
decision functions* over those cached numbers for as many threshold combinations as the
sweep wants, at roughly the speed of a loop. A hundred-point sweep costs one inference
pass, which is what makes a real sweep affordable on a corpus this size.

Because thresholds move the boundaries between gates, every pair is scored on all three
slots rather than only on the gates it would reach under one particular setting. A sweep
that re-derived the G2 population from the G1 thresholds it was measuring would be unable
to answer the question it was asked.

**Two readings of §8 this file had to fix**, both recorded here because they change
numbers:

* A contradiction-flagged pair always reaches G3. §8's G3 heading says it sees
  "contradiction-flagged + G2 leftovers", so a flag set at G1 survives G2 rather than
  being overtaken by a CORROBORATES there. The alternative — letting G2 finalize a pair
  G1 flagged — would put a contradiction signal beyond the reach of the cross-check whose
  entire job is to adjudicate it.
* The contexted variant of a pair is the claim text followed by its explanation text.
  §11.9 says to "fetch the explanation text and run a contexted variant" without fixing
  the composition; concatenation is the reading that keeps the pair's own text intact and
  therefore keeps the bare and contexted scores comparable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any

from gatekeeper.gates.decisions import (
    Disposition,
    G1Scores,
    G2Scores,
    G3Scores,
    Thresholds,
    decide_g1,
    decide_g2,
    decide_g3,
)
from gatekeeper.logging import get_logger
from gatekeeper.replay.corpus import PairRecord
from gatekeeper.replay.metrics import (
    GoBar,
    GoBarResult,
    Outcome,
    RosterReport,
    contradicts_recall,
    coverage_curve,
    evaluate_go_bar,
    expected_calibration_error,
    neutral_summary,
    per_verdict_scores,
    volumes,
)
from gatekeeper.scoring.encoder import (
    EncoderScorer,
    GroundingScore,
    NliScores,
    scorer_for_artifact,
)

__all__ = [
    "GridCell",
    "GroundingPass",
    "NliPass",
    "PairScores",
    "RosterCandidate",
    "ScoreCache",
    "best_cell",
    "build_report",
    "corpus_fingerprint",
    "grid_search",
    "replay",
    "run_cascade",
    "score_corpus",
    "sweep",
]

log = get_logger(__name__)

SCORE_CACHE_VERSION = 2
"""Bumped for the artifact-keyed layout; a v1 roster-keyed cache is simply a miss."""


@dataclass(frozen=True, slots=True)
class RosterCandidate:
    """One combination of artifacts to measure — a row of the bake-off table."""

    name: str
    g1: str
    g2: str
    g3: str

    @property
    def artifacts(self) -> dict[str, str]:
        return {"g1": self.g1, "g2": self.g2, "g3": self.g3}

    @property
    def refs(self) -> tuple[str, ...]:
        """Distinct artifacts to load — a roster may reuse one model in two slots."""
        return tuple(dict.fromkeys((self.g1, self.g2, self.g3)))


@dataclass(frozen=True, slots=True)
class PairScores:
    """Every number a cascade replay needs for one pair, for one candidate roster."""

    g1: G1Scores
    g2: G2Scores
    g3: G3Scores

    def to_json(self) -> dict[str, Any]:
        def nli(scores: NliScores) -> list[float | bool]:
            return [scores.entailment, scores.neutral, scores.contradiction, scores.truncated]

        def grounding(score: GroundingScore | None) -> list[float | bool] | None:
            return None if score is None else [score.support, score.truncated]

        return {
            "g1": {
                "forward": nli(self.g1.forward),
                "backward": nli(self.g1.backward),
                "contextForward": nli(self.g1.context_forward) if self.g1.context_forward else None,
                "contextBackward": (
                    nli(self.g1.context_backward) if self.g1.context_backward else None
                ),
            },
            "g2": {
                "forward": grounding(self.g2.forward),
                "backward": grounding(self.g2.backward),
                "grounded": grounding(self.g2.grounded),
            },
            "g3": {
                "forward": nli(self.g3.family_b_forward),
                "backward": nli(self.g3.family_b_backward),
            },
        }

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> PairScores:
        def nli(raw: list[Any] | None) -> NliScores | None:
            if raw is None:
                return None
            return NliScores(
                entailment=float(raw[0]),
                neutral=float(raw[1]),
                contradiction=float(raw[2]),
                truncated=bool(raw[3]),
            )

        def grounding(raw: list[Any] | None) -> GroundingScore | None:
            if raw is None:
                return None
            return GroundingScore(support=float(raw[0]), truncated=bool(raw[1]))

        g1, g2, g3 = document["g1"], document["g2"], document["g3"]
        forward, backward = nli(g1["forward"]), nli(g1["backward"])
        assert forward is not None and backward is not None
        g2_forward, g2_backward = grounding(g2["forward"]), grounding(g2["backward"])
        assert g2_forward is not None and g2_backward is not None
        family_forward, family_backward = nli(g3["forward"]), nli(g3["backward"])
        assert family_forward is not None and family_backward is not None

        return cls(
            g1=G1Scores(
                forward=forward,
                backward=backward,
                context_forward=nli(g1.get("contextForward")),
                context_backward=nli(g1.get("contextBackward")),
            ),
            g2=G2Scores(
                forward=g2_forward, backward=g2_backward, grounded=grounding(g2.get("grounded"))
            ),
            g3=G3Scores(family_b_forward=family_forward, family_b_backward=family_backward),
        )


@dataclass(frozen=True, slots=True)
class NliPass:
    """One artifact's NLI scores over the whole corpus, both directions."""

    forward: list[NliScores]
    backward: list[NliScores]
    context_forward: dict[int, NliScores] = field(default_factory=dict)
    context_backward: dict[int, NliScores] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "forward": [_nli_json(scores) for scores in self.forward],
            "backward": [_nli_json(scores) for scores in self.backward],
            "contextForward": {str(i): _nli_json(s) for i, s in self.context_forward.items()},
            "contextBackward": {str(i): _nli_json(s) for i, s in self.context_backward.items()},
        }

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> NliPass:
        return cls(
            forward=[_nli_from(raw) for raw in document["forward"]],
            backward=[_nli_from(raw) for raw in document["backward"]],
            context_forward={
                int(i): _nli_from(raw) for i, raw in document["contextForward"].items()
            },
            context_backward={
                int(i): _nli_from(raw) for i, raw in document["contextBackward"].items()
            },
        )


@dataclass(frozen=True, slots=True)
class GroundingPass:
    """One artifact's support scores over the whole corpus, both directions."""

    forward: list[GroundingScore]
    backward: list[GroundingScore]
    grounded: dict[int, GroundingScore] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "forward": [[s.support, s.truncated] for s in self.forward],
            "backward": [[s.support, s.truncated] for s in self.backward],
            "grounded": {str(i): [s.support, s.truncated] for i, s in self.grounded.items()},
        }

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> GroundingPass:
        def one(raw: list[Any]) -> GroundingScore:
            return GroundingScore(support=float(raw[0]), truncated=bool(raw[1]))

        return cls(
            forward=[one(raw) for raw in document["forward"]],
            backward=[one(raw) for raw in document["backward"]],
            grounded={int(i): one(raw) for i, raw in document["grounded"].items()},
        )


def _nli_json(scores: NliScores) -> list[float | bool]:
    return [scores.entailment, scores.neutral, scores.contradiction, scores.truncated]


def _nli_from(raw: list[Any]) -> NliScores:
    return NliScores(
        entailment=float(raw[0]),
        neutral=float(raw[1]),
        contradiction=float(raw[2]),
        truncated=bool(raw[3]),
    )


def corpus_fingerprint(records: Sequence[PairRecord]) -> str:
    """Identifies the exact pair sequence a cached pass was computed over.

    Without it a ``--limit 64`` smoke run leaves behind a cache that a later full run would
    load and silently report 64 pairs from, because a pass is stored positionally and a
    missing pair is indistinguishable from an unscored one.
    """
    digest = sha256("\n".join(record.pair_id for record in records).encode("utf-8"))
    return f"{len(records)}:{digest.hexdigest()[:16]}"


class ScoreCache:
    """Raw probabilities on disk, keyed by **artifact and task**, not by roster.

    Inference over the corpus is measured in hours on CPU at fp32; a sweep, a re-report or
    a crashed run should never pay for it twice. Keying on the artifact rather than the
    roster matters just as much: the rosters overlap heavily — five of session03's six
    share a G2 — so a roster-keyed cache pays for the same weights up to five times. The
    bake-off's cost is then the number of distinct models (7), not the number of table
    rows (18 slot-passes).

    An NLI pass always includes the contexted variant, even for an artifact currently sat
    in the G3 slot where nothing reads it. That keeps one cache entry per artifact instead
    of one per (artifact, slot), and costs nothing on a corpus whose pairs carry no
    explanation text.
    """

    def __init__(self, directory: Path | None, fingerprint: str = "") -> None:
        self.directory = directory
        self.fingerprint = fingerprint

    def _path(self, ref: str, task: str) -> Path | None:
        if self.directory is None:
            return None
        return self.directory / f"{ref.replace('/', '_')}--{task}.json"

    def _read(self, ref: str, task: str) -> dict[str, Any] | None:
        path = self._path(ref, task)
        if path is None or not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("version") != SCORE_CACHE_VERSION:
            return None
        if document.get("fingerprint") != self.fingerprint:
            # A cache over a different corpus (or a --limit subset of it) is worse than none.
            log.warning(
                "score cache is for a different corpus; ignoring it",
                fields={
                    "artifact": ref,
                    "task": task,
                    "cached": document.get("fingerprint"),
                    "wanted": self.fingerprint,
                },
            )
            return None
        return document

    def _write(self, ref: str, task: str, payload: dict[str, Any]) -> None:
        path = self._path(ref, task)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": SCORE_CACHE_VERSION,
                    "artifact": ref,
                    "task": task,
                    "fingerprint": self.fingerprint,
                    "scores": payload,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def nli(self, ref: str, compute: Callable[[], NliPass]) -> NliPass:
        document = self._read(ref, "nli")
        if document is not None:
            log.info("score cache hit", fields={"artifact": ref, "task": "nli"})
            return NliPass.from_json(document["scores"])
        result = compute()
        self._write(ref, "nli", result.to_json())
        return result

    def grounding(self, ref: str, compute: Callable[[], GroundingPass]) -> GroundingPass:
        document = self._read(ref, "grounding")
        if document is not None:
            log.info("score cache hit", fields={"artifact": ref, "task": "grounding"})
            return GroundingPass.from_json(document["scores"])
        result = compute()
        self._write(ref, "grounding", result.to_json())
        return result


# --- scoring ------------------------------------------------------------------------------


def _contexted(text: str, context: str) -> str:
    return f"{text}\n\n{context}" if context else text


def score_corpus(
    records: Sequence[PairRecord],
    candidate: RosterCandidate,
    scorer_for: Callable[[str], EncoderScorer],
    *,
    cache: ScoreCache | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, PairScores]:
    """Run every slot's artifact over every pair, once.

    Args:
        scorer_for: artifact ref → scorer. Injected so the harness can be driven from a
            local mirror, a bucket, or a stub, without knowing which.
    """
    cache = cache or ScoreCache(None, corpus_fingerprint(records))
    forward_pairs = [(r.claim_a_text, r.claim_b_text) for r in records]
    backward_pairs = [(r.claim_b_text, r.claim_a_text) for r in records]

    def nli_pass(ref: str, slot: str) -> NliPass:
        def compute() -> NliPass:
            if progress:
                progress(slot, 0, len(records))
            scorer = scorer_for(ref)
            # The contexted variant only exists for the pairs that carry an explanation, so
            # it is scored as its own dense batch rather than as a mostly-empty pass over
            # everything.
            contexted_index = [i for i, r in enumerate(records) if r.with_context]
            subset = [records[i] for i in contexted_index]
            context_forward = scorer.score_nli(
                [
                    (
                        _contexted(r.claim_a_text, r.context_a),
                        _contexted(r.claim_b_text, r.context_b),
                    )
                    for r in subset
                ]
            )
            context_backward = scorer.score_nli(
                [
                    (
                        _contexted(r.claim_b_text, r.context_b),
                        _contexted(r.claim_a_text, r.context_a),
                    )
                    for r in subset
                ]
            )
            return NliPass(
                forward=scorer.score_nli(forward_pairs),
                backward=scorer.score_nli(backward_pairs),
                context_forward=dict(zip(contexted_index, context_forward, strict=True)),
                context_backward=dict(zip(contexted_index, context_backward, strict=True)),
            )

        return cache.nli(ref, compute)

    def grounding_pass(ref: str, slot: str) -> GroundingPass:
        def compute() -> GroundingPass:
            if progress:
                progress(slot, 0, len(records))
            scorer = scorer_for(ref)
            grounded_index = [i for i, r in enumerate(records) if r.source_snippet]
            grounded = scorer.score_grounding(
                [(records[i].source_snippet, records[i].claim_b_text) for i in grounded_index]
            )
            return GroundingPass(
                forward=scorer.score_grounding(forward_pairs),
                backward=scorer.score_grounding(backward_pairs),
                grounded=dict(zip(grounded_index, grounded, strict=True)),
            )

        return cache.grounding(ref, compute)

    g1 = nli_pass(candidate.g1, "g1")
    g2 = grounding_pass(candidate.g2, "g2")
    g3 = nli_pass(candidate.g3, "g3")

    return {
        record.pair_id: PairScores(
            g1=G1Scores(
                forward=g1.forward[index],
                backward=g1.backward[index],
                context_forward=g1.context_forward.get(index),
                context_backward=g1.context_backward.get(index),
            ),
            g2=G2Scores(
                forward=g2.forward[index],
                backward=g2.backward[index],
                grounded=g2.grounded.get(index),
            ),
            g3=G3Scores(family_b_forward=g3.forward[index], family_b_backward=g3.backward[index]),
        )
        for index, record in enumerate(records)
    }


# --- replay -------------------------------------------------------------------------------


def run_cascade(record: PairRecord, scores: PairScores, thresholds: Thresholds) -> Outcome:
    """Walk one pair through G1 → G2 → G3 using the production decision functions."""
    first = decide_g1(scores.g1, thresholds)
    if first.disposition is not Disposition.FORWARD:
        return _outcome(record, first, "G1_NEUTRAL")

    if not first.contradiction_flag:
        # A flagged pair skips G2's chance to finalize it: §8 puts contradiction-flagged
        # pairs in front of G3 whatever else happens to them.
        second = decide_g2(scores.g2, thresholds)
        if second.disposition is not Disposition.FORWARD:
            return _outcome(record, second, "G2_CORROBORATION")

    third = decide_g3(scores.g1, scores.g3, thresholds)
    return _outcome(record, third, "G3_CONTRADICTION", flag=first.contradiction_flag)


def _outcome(record: PairRecord, decision: Any, gate: str, *, flag: bool = False) -> Outcome:
    return Outcome(
        pair_id=record.pair_id,
        disposition=decision.disposition,
        gate=gate,
        verdict=decision.verdict,
        ensemble_verdict=record.ensemble_verdict,
        golden=record.golden,
        confidence=decision.confidence,
        contradiction_flag=decision.contradiction_flag or flag,
    )


def replay(
    records: Sequence[PairRecord],
    scores: dict[str, PairScores],
    thresholds: Thresholds,
) -> list[Outcome]:
    """Re-run the whole corpus through the cascade at one threshold setting."""
    return [
        run_cascade(record, scores[record.pair_id], thresholds)
        for record in records
        if record.pair_id in scores
    ]


def sweep(
    records: Sequence[PairRecord],
    scores: dict[str, PairScores],
    base: Thresholds,
    axis: str,
    values: Iterable[float],
) -> list[tuple[float, list[Outcome]]]:
    """Replay the corpus once per value of one threshold."""
    return [(value, replay(records, scores, replace(base, **{axis: value}))) for value in values]


# --- report -------------------------------------------------------------------------------


SWEEPABLE: dict[str, str] = {
    "neutralMin": "neutral_min",
    "repeatMin": "repeat_min",
    "contraEscape": "contra_escape",
    "supportMin": "support_min",
    "contraMin": "contra_min",
    "neutralConsensus": "neutral_consensus",
}
"""Threshold names (as they appear in ``configSnapshot``) → the field that carries them."""


@dataclass(frozen=True, slots=True)
class GridCell:
    """One threshold combination and how it scored."""

    thresholds: Thresholds
    values: dict[str, float]
    neutral: dict[str, float]
    funnel: dict[str, float]
    contradictions: dict[str, float]
    go_bar: GoBarResult

    def to_json(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "neutral": self.neutral,
            "funnel": self.funnel,
            "contradictions": self.contradictions,
            "passed": self.go_bar.passed,
        }


def grid_search(
    records: Sequence[PairRecord],
    scores: dict[str, PairScores],
    base: Thresholds,
    axes: dict[str, Sequence[float]],
    bar: GoBar | None = None,
) -> list[GridCell]:
    """Score every combination of the given threshold values.

    Independent one-axis sweeps cannot find this cascade's operating point, because the
    thresholds gate each other: while ``neutralMin`` is being swept, a ``contraEscape``
    left at its default may already have flagged the pairs the sweep is about, so the
    curve comes back flat and says nothing. The combinations are what §15 means by "the
    best passing roster **and thresholds**".

    Replaying cached scores is cheap — a full corpus pass is a loop over probabilities —
    so a few hundred cells cost seconds, against hours for the inference behind them.
    """
    names = list(axes)
    cells: list[GridCell] = []
    for combination in product(*(axes[name] for name in names)):
        values = dict(zip(names, combination, strict=True))
        thresholds = replace(base, **{SWEEPABLE[name]: value for name, value in values.items()})
        outcomes = replay(records, scores, thresholds)
        cells.append(
            GridCell(
                thresholds=thresholds,
                values=values,
                neutral=neutral_summary(outcomes),
                funnel=volumes(outcomes),
                contradictions=contradicts_recall(outcomes),
                go_bar=evaluate_go_bar(outcomes, bar),
            )
        )
    return cells


def best_cell(cells: Sequence[GridCell]) -> GridCell | None:
    """The passing combination with the most NEUTRAL coverage, then the least G4."""
    passing = [cell for cell in cells if cell.go_bar.passed]
    if not passing:
        return None
    return max(passing, key=lambda cell: (cell.neutral["coverage"], -cell.funnel["escalatedToG4"]))


def build_report(
    candidate: RosterCandidate,
    records: Sequence[PairRecord],
    scores: dict[str, PairScores],
    thresholds: Thresholds,
    *,
    sweeps: dict[str, Sequence[float]] | None = None,
    grid: dict[str, Sequence[float]] | None = None,
    bar: GoBar | None = None,
    wall_clock_seconds: float = 0.0,
    precisions: dict[str, str] | None = None,
) -> RosterReport:
    """Everything §15 asks a candidate to report.

    When a ``grid`` is given, the reported thresholds are the best *passing* combination
    rather than the ones passed in — that is the mechanical application of §15's bar the
    session is meant to perform. With no passing cell the base thresholds are reported, so
    a NO-GO still shows what was actually measured.
    """
    cells = grid_search(records, scores, thresholds, grid, bar) if grid else []
    winner = best_cell(cells)
    if winner is not None:
        thresholds = winner.thresholds

    outcomes = replay(records, scores, thresholds)
    curves = {
        axis: coverage_curve(sweep(records, scores, thresholds, SWEEPABLE[axis], values))
        for axis, values in (sweeps or {}).items()
    }

    artifacts = dict(candidate.artifacts)
    if precisions:
        artifacts = {slot: f"{ref} ({precisions.get(ref, '?')})" for slot, ref in artifacts.items()}

    return RosterReport(
        roster=candidate.name,
        pairs=len(outcomes),
        artifacts=artifacts,
        thresholds={
            "neutralMin": thresholds.neutral_min,
            "repeatMin": thresholds.repeat_min,
            "contraEscape": thresholds.contra_escape,
            "supportMin": thresholds.support_min,
            "contraMin": thresholds.contra_min,
            "neutralConsensus": thresholds.neutral_consensus,
        },
        per_verdict=per_verdict_scores(outcomes),
        neutral=neutral_summary(outcomes),
        sweeps=curves,
        grid=[cell.to_json() for cell in cells],
        contradictions=contradicts_recall(outcomes),
        calibration=expected_calibration_error(outcomes),
        funnel=volumes(outcomes),
        go_bar=evaluate_go_bar(outcomes, bar),
        wall_clock_seconds=wall_clock_seconds,
    )


def timed(work: Callable[[], Any]) -> tuple[Any, float]:
    started = time.monotonic()
    return work(), time.monotonic() - started


def best_passing(reports: Sequence[RosterReport]) -> RosterReport | None:
    """The roster to adopt: passing bars first, then NEUTRAL coverage, then G4 share.

    §15 leaves "best" to judgement; this ranks by the thing the cascade exists to do —
    remove judging load — and breaks ties on the thing it costs, so the choice is
    mechanical and reproducible rather than a matter of taste.
    """
    passing = [report for report in reports if report.go_bar.passed]
    if not passing:
        return None
    return max(
        passing,
        key=lambda report: (report.neutral["coverage"], -report.funnel["escalatedToG4"]),
    )


def scorer_factory(
    loader: Any, *, batch_size: int = 32, max_live: int = 1
) -> Callable[[str], EncoderScorer]:
    """Build a ``ref -> scorer`` over a production loader, holding ``max_live`` at a time.

    The bake-off scores one artifact over the whole corpus before it touches the next, so
    it never needs two resident at once. That matters at fp32: the seven mirrored rosters
    total ~5.6 GB of weights, and an unbounded cache would hold every one of them — plus
    its ONNX arena — for the length of a multi-hour run. Keeping one live trades a reload
    (seconds) for gigabytes, and a reload only ever happens on a cache miss anyway.

    Raise ``max_live`` for a caller that really does interleave artifacts; the worker does
    not use this path at all — it loads lazily per gate through the loader itself.
    """
    built: dict[str, EncoderScorer] = {}

    def build(ref: str) -> EncoderScorer:
        if ref not in built:
            while len(built) >= max_live:
                evicted = next(iter(built))  # oldest first — dicts keep insertion order
                del built[evicted]
                log.info("releasing scorer", fields={"artifact": evicted})
            artifact = loader.load(ref, expected_manifest_sha256="")
            built[ref] = scorer_for_artifact(artifact, batch_size=batch_size)
        return built[ref]

    return build
