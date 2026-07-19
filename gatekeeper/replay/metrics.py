"""Bake-off metrics and the GO bar (LLD §15 step 2).

Everything here is a pure function over cascade outcomes and corpus labels, so the numbers
that decide a go/no-go can be unit-tested against hand-built cases rather than only ever
observed on a corpus nobody can diff.

Three definitions carry the weight, and each resolves an ambiguity in §15's one-line
statement of the bar:

* **Escalating is not a wrong answer.** A pair sent to G4 or to a human has not been
  judged wrongly, it has been deferred. So escalations never count against *precision* —
  only against recall, which is exactly the trade the cascade is meant to expose.
* **NEUTRAL coverage is over the whole corpus**, not over the pairs G1 happened to look
  at. "0.95 precision at 60% coverage" is a claim about how much of the judging load the
  gate actually removes, and measuring it against a subset would flatter it.
* **CONTRADICTS recall is "did not discard"**, not "said CONTRADICTS". The cascade never
  finalizes CONTRADICTS by design (§8, G3), so the only failure that matters is a golden
  contradiction that got silently decided as NEUTRAL, REPEATS or CORROBORATES instead of
  reaching a human or the LLM tail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from gatekeeper.enums import Verdict
from gatekeeper.gates.decisions import Disposition

__all__ = [
    "GoBar",
    "GoBarResult",
    "Outcome",
    "PrecisionRecall",
    "RosterReport",
    "coverage_curve",
    "expected_calibration_error",
    "neutral_summary",
    "per_verdict_scores",
]

DECIDED_VERDICTS: tuple[Verdict, ...] = (Verdict.NEUTRAL, Verdict.REPEATS, Verdict.CORROBORATES)
"""What the encoder cascade can actually finalize. CONTRADICTS is never one of them."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the cascade did with one pair, next to what the ensemble said about it."""

    pair_id: str
    disposition: Disposition
    gate: str
    verdict: Verdict | None
    ensemble_verdict: str
    golden: bool = False
    confidence: float | None = None
    contradiction_flag: bool = False

    @property
    def decided(self) -> bool:
        return self.disposition is Disposition.DECIDED

    @property
    def escalated(self) -> bool:
        return self.disposition in (Disposition.ESCALATE_G4, Disposition.ROUTE_HUMAN)


@dataclass(frozen=True, slots=True)
class PrecisionRecall:
    label: str
    true_positives: int
    predicted: int
    actual: int

    @property
    def precision(self) -> float:
        return self.true_positives / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.actual if self.actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "truePositives": self.true_positives,
            "predicted": self.predicted,
            "actual": self.actual,
        }


def per_verdict_scores(outcomes: Sequence[Outcome]) -> list[PrecisionRecall]:
    """Precision and recall per verdict the cascade can finalize."""
    scores: list[PrecisionRecall] = []
    for verdict in DECIDED_VERDICTS:
        predicted = [o for o in outcomes if o.verdict is verdict]
        actual = [o for o in outcomes if o.ensemble_verdict == verdict.value]
        hits = sum(1 for o in predicted if o.ensemble_verdict == verdict.value)
        scores.append(
            PrecisionRecall(
                label=verdict.value,
                true_positives=hits,
                predicted=len(predicted),
                actual=len(actual),
            )
        )
    return scores


def neutral_summary(outcomes: Sequence[Outcome]) -> dict[str, float]:
    """NEUTRAL precision and coverage for one cascade replay.

    Coverage is over the whole corpus — see this module's docstring for why.
    """
    neutral = [o for o in outcomes if o.verdict is Verdict.NEUTRAL]
    correct = sum(1 for o in neutral if o.ensemble_verdict == Verdict.NEUTRAL.value)
    total = len(outcomes)
    return {
        "coverage": len(neutral) / total if total else 0.0,
        "precision": correct / len(neutral) if neutral else 0.0,
        "decided": float(len(neutral)),
    }


def coverage_curve(
    curves: Sequence[tuple[float, Sequence[Outcome]]],
) -> list[dict[str, float]]:
    """NEUTRAL precision/coverage and G4 share at each swept value of one threshold.

    ``curves`` is ``(value, outcomes)`` — one full cascade replay per value, which is
    cheap because the inference behind it was done once and cached.

    G4 share rides along because the axis that moves coverage most is not always
    ``neutralMin``: the contradiction escape hatch reroutes pairs wholesale, and reading
    its coverage without its cost would be reading half the trade.
    """
    points: list[dict[str, float]] = []
    for value, outcomes in curves:
        summary = neutral_summary(outcomes)
        funnel = volumes(outcomes)
        points.append(
            {
                "value": value,
                "coverage": summary["coverage"],
                "precision": summary["precision"],
                "decided": summary["decided"],
                "g4Share": funnel["escalatedToG4"],
                "humanShare": funnel["routedToHuman"],
            }
        )
    return points


def contradicts_recall(outcomes: Sequence[Outcome]) -> dict[str, float]:
    """How well the cascade preserves the ensemble's contradictions on golden pairs.

    The denominator is the golden pairs the *ensemble* called CONTRADICTS — that is "the
    ensemble's own recall" in §15's phrasing, the bar the cascade is measured relative to.
    A pair is preserved if it reached a human or the LLM tail rather than being decided.
    """
    golden = [o for o in outcomes if o.golden and o.ensemble_verdict == Verdict.CONTRADICTS.value]
    if not golden:
        return {"goldenContradictions": 0.0, "preserved": 0.0, "recall": 0.0, "discarded": 0.0}

    preserved = sum(1 for o in golden if not o.decided)
    return {
        "goldenContradictions": float(len(golden)),
        "preserved": float(preserved),
        "recall": preserved / len(golden),
        "discarded": float(len(golden) - preserved),
    }


def expected_calibration_error(outcomes: Sequence[Outcome], bins: int = 10) -> dict[str, float]:
    """ECE over the pairs that carry a confidence (G2's CORROBORATES decisions).

    Correctness is agreement with the ensemble's verdict for the same pair.
    """
    scored = [o for o in outcomes if o.confidence is not None and o.verdict is not None]
    if not scored:
        return {"ece": 0.0, "samples": 0.0, "bins": float(bins)}

    total_error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            o
            for o in scored
            if (low <= (o.confidence or 0.0) < high)
            or (index == bins - 1 and (o.confidence or 0.0) == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(
            1 for o in members if o.ensemble_verdict == (o.verdict.value if o.verdict else "")
        ) / len(members)
        confidence = sum(o.confidence or 0.0 for o in members) / len(members)
        total_error += (len(members) / len(scored)) * abs(accuracy - confidence)

    return {"ece": total_error, "samples": float(len(scored)), "bins": float(bins)}


def volumes(outcomes: Sequence[Outcome]) -> dict[str, float]:
    """Where the corpus ended up — the funnel §16 draws, as fractions."""
    total = len(outcomes) or 1
    g4 = sum(1 for o in outcomes if o.disposition is Disposition.ESCALATE_G4)
    human = sum(1 for o in outcomes if o.disposition is Disposition.ROUTE_HUMAN)
    decided = sum(1 for o in outcomes if o.decided)
    return {
        "total": float(len(outcomes)),
        "decidedByEncoders": decided / total,
        "escalatedToG4": g4 / total,
        "routedToHuman": human / total,
        "undecided": (len(outcomes) - decided - g4 - human) / total,
    }


# --- the GO bar ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoBar:
    """LLD §15 step 2, as numbers."""

    min_neutral_precision: float = 0.95
    min_neutral_coverage: float = 0.60
    min_contradicts_recall_ratio: float = 0.95
    max_g4_share: float = 0.05


@dataclass(frozen=True, slots=True)
class GoBarResult:
    passed: bool
    criteria: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def unmeasurable(self) -> tuple[str, ...]:
        """Criteria this corpus cannot answer either way."""
        return tuple(name for name, detail in self.criteria.items() if not detail["measurable"])

    def describe(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if self.passed and self.unmeasurable:
            # Never let a criterion that was never tested read as one that was met.
            verdict = "PASS (PARTIAL — see n/a below)"
        lines = [verdict]
        for name, detail in self.criteria.items():
            if not detail["measurable"]:
                lines.append(f"  n/a  {name}: not measurable on this corpus — {detail['reason']}")
                continue
            mark = "ok  " if detail["passed"] else "MISS"
            lines.append(
                f"  {mark} {name}: {detail['observed']:.4f} "
                f"(bar {detail['comparison']} {detail['required']})"
            )
        return "\n".join(lines)


def evaluate_go_bar(outcomes: Sequence[Outcome], bar: GoBar | None = None) -> GoBarResult:
    """Apply §15's pre-agreed criteria mechanically.

    A criterion whose denominator is empty is reported as **unmeasurable**, not as passed
    and not as failed. Session03's corpus is the case that forces the distinction: it
    carries no golden pairs and no CONTRADICTS verdicts at all, so the contradiction-recall
    criterion has nothing to measure. Scoring that as a pass would manufacture evidence for
    the one property §15 most wants proven; scoring it as a fail would reject a roster for
    a gap in the corpus rather than a fault in the cascade. Both are wrong, so it is
    neither, and :attr:`GoBarResult.unmeasurable` carries it into the report where the
    owner will see it.
    """
    bar = bar or GoBar()
    summary = neutral_summary(outcomes)
    contradictions = contradicts_recall(outcomes)
    funnel = volumes(outcomes)

    golden_contradictions = contradictions["goldenContradictions"]
    criteria = {
        "neutralPrecision": {
            "observed": summary["precision"],
            "required": bar.min_neutral_precision,
            "comparison": ">=",
            "measurable": True,
            "passed": summary["precision"] >= bar.min_neutral_precision,
        },
        "neutralCoverage": {
            "observed": summary["coverage"],
            "required": bar.min_neutral_coverage,
            "comparison": ">=",
            "measurable": True,
            "passed": summary["coverage"] >= bar.min_neutral_coverage,
        },
        "contradictsRecall": {
            "observed": contradictions["recall"],
            "required": bar.min_contradicts_recall_ratio,
            "comparison": ">=",
            "measurable": golden_contradictions > 0,
            "reason": "the corpus holds no golden pair the ensemble called CONTRADICTS",
            "passed": contradictions["recall"] >= bar.min_contradicts_recall_ratio,
        },
        "g4Share": {
            "observed": funnel["escalatedToG4"],
            "required": bar.max_g4_share,
            "comparison": "<=",
            "measurable": True,
            "passed": funnel["escalatedToG4"] <= bar.max_g4_share,
        },
    }
    return GoBarResult(
        passed=all(detail["passed"] for detail in criteria.values() if detail["measurable"]),
        criteria=criteria,
    )


@dataclass
class RosterReport:
    """Everything measured for one roster + threshold combination."""

    roster: str
    thresholds: dict[str, float]
    artifacts: dict[str, str]
    per_verdict: list[PrecisionRecall]
    neutral: dict[str, float]
    """Precision and coverage at the reported thresholds — what the GO bar reads."""

    sweeps: dict[str, list[dict[str, float]]]
    """One curve per swept axis, keyed by threshold name."""

    contradictions: dict[str, float]
    calibration: dict[str, float]
    funnel: dict[str, float]
    go_bar: GoBarResult

    grid: list[dict[str, Any]] = field(default_factory=list)
    """Every threshold combination tried, when a joint grid search was run."""

    wall_clock_seconds: float = 0.0
    pairs: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "roster": self.roster,
            "pairs": self.pairs,
            "artifacts": self.artifacts,
            "thresholds": self.thresholds,
            "perVerdict": [score.to_json() for score in self.per_verdict],
            "neutral": self.neutral,
            "sweeps": self.sweeps,
            "grid": self.grid,
            "contradictions": self.contradictions,
            "calibration": self.calibration,
            "funnel": self.funnel,
            "goBar": {
                "passed": self.go_bar.passed,
                "unmeasurable": list(self.go_bar.unmeasurable),
                "criteria": self.go_bar.criteria,
            },
            "wallClockSeconds": self.wall_clock_seconds,
        }
