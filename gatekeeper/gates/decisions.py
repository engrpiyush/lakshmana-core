"""The gate decision rules, as pure functions (LLD §8).

Each function is a direct transcription of the LLD's pseudocode: scores and thresholds in,
a :class:`Decision` out, no I/O, no clock, no Firestore. That shape is what lets the
replay bake-off (VA-97) claim it measured the production gates rather than a model of
them — it calls exactly these functions, with thresholds from exactly the same
``configSnapshot`` shape the worker freezes onto a run doc.

Thresholds are always passed in. Reading live config here would let a mid-run edit split
a run's calibration, which §5 forbids, and would make a sweep's results depend on the
environment it ran in.

**Two places where the LLD's pseudocode needed a decision and this file made one**, both
resolved toward escalation rather than toward silently deciding a pair:

* ``neutralConsensus`` (G3) is named in the pseudocode but has no config key or proposed
  value in §8. It is added as ``gatekeeper.gates.g3.neutral-consensus``, a proposal like
  every other number there, for the bake-off to set.
* ``both_neutral_lean`` (G3) is not defined at all. It is read here as *neutral is the
  argmax in both directions of both families* — the strictest available reading. A looser
  one would let a pair the two families disagree about be discarded as NEUTRAL, and the
  whole point of the cross-check is that it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from gatekeeper.enums import EscalationReason, Method, Verdict
from gatekeeper.scoring.encoder import GroundingScore, NliScores

__all__ = [
    "Decision",
    "Direction",
    "Disposition",
    "G1Scores",
    "G2Scores",
    "G3Scores",
    "Thresholds",
    "decide_g1",
    "decide_g2",
    "decide_g3",
]


class Disposition(StrEnum):
    """What happens to the pair after a gate has looked at it."""

    DECIDED = "DECIDED"
    """A verdict was reached here; the pair leaves the cascade."""

    FORWARD = "FORWARD"
    """Undecided; the next gate sees it."""

    ESCALATE_G4 = "ESCALATE_G4"
    """Straight to the LLM tail, skipping any gate in between."""

    ROUTE_HUMAN = "ROUTE_HUMAN"
    """To the human queue — the cascade never finalizes CONTRADICTS itself."""


class Direction(StrEnum):
    """Which way round a directional verdict was reached."""

    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    GROUNDED = "GROUNDED"
    """Claim scored against the paired claim's source snippet — G2's evidence mode."""


@dataclass(frozen=True, slots=True)
class Decision:
    """One gate's outcome for one pair."""

    disposition: Disposition
    verdict: Verdict | None = None
    method: Method | None = None
    escalation_reason: EscalationReason | None = None

    contradiction_flag: bool = False
    """G1's escape hatch. Once set, the pair may never be neutral-discarded downstream."""

    direction: Direction | None = None
    confidence: float | None = None
    truncated: bool = False
    notes: dict[str, Any] = field(default_factory=dict)
    """Diagnostics for the replay report — never part of the stored verdict."""


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Every number the gates read, straight off ``configSnapshot`` (LLD §7.1)."""

    neutral_min: float = 0.95
    repeat_min: float = 0.90
    contra_escape: float = 0.02
    support_min: float = 0.90
    support_floor: float = 0.50
    contra_min: float = 0.85
    neutral_consensus: float = 0.10

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> Thresholds:
        """Build from a ``configSnapshot`` mapping, falling back to the LLD proposals."""
        g1, g2, g3 = snapshot.get("g1", {}), snapshot.get("g2", {}), snapshot.get("g3", {})
        defaults = cls()
        return cls(
            neutral_min=float(g1.get("neutralMin", defaults.neutral_min)),
            repeat_min=float(g1.get("repeatMin", defaults.repeat_min)),
            contra_escape=float(g1.get("contraEscape", defaults.contra_escape)),
            support_min=float(g2.get("supportMin", defaults.support_min)),
            support_floor=float(g2.get("supportFloor", defaults.support_floor)),
            contra_min=float(g3.get("contraMin", defaults.contra_min)),
            neutral_consensus=float(g3.get("neutralConsensus", defaults.neutral_consensus)),
        )


# --- G1_NEUTRAL ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class G1Scores:
    """Both directions of the G1 NLI pass, plus the contexted variant when there is one."""

    forward: NliScores
    backward: NliScores
    context_forward: NliScores | None = None
    context_backward: NliScores | None = None

    @property
    def with_context(self) -> bool:
        return self.context_forward is not None and self.context_backward is not None

    @property
    def max_contradiction(self) -> float:
        return max(self.forward.contradiction, self.backward.contradiction)

    @property
    def truncated(self) -> bool:
        return self.forward.truncated or self.backward.truncated


def _bare_outcome(
    forward: NliScores, backward: NliScores, thresholds: Thresholds
) -> Verdict | None:
    """What G1 would decide from one direction pair, ignoring the escape hatch.

    ``None`` means "would forward" — which is itself an outcome the context comparison
    has to be able to observe, since bare-forwards-but-contexted-decides is exactly the
    disagreement §11.9 is looking for.
    """
    if forward.neutral >= thresholds.neutral_min and backward.neutral >= thresholds.neutral_min:
        return Verdict.NEUTRAL
    if forward.entailment >= thresholds.repeat_min and backward.entailment >= thresholds.repeat_min:
        return Verdict.REPEATS
    return None


def decide_g1(scores: G1Scores, thresholds: Thresholds) -> Decision:
    """G1_NEUTRAL — sees 100% of capped pairs (LLD §8)."""
    # THE escape hatch, and it runs first for a reason: a pair with any contradiction
    # signal at all must reach G3, whatever the neutral probabilities say.
    if scores.max_contradiction > thresholds.contra_escape:
        return Decision(
            disposition=Disposition.FORWARD,
            contradiction_flag=True,
            truncated=scores.truncated,
            notes={"maxContradiction": scores.max_contradiction},
        )

    bare = _bare_outcome(scores.forward, scores.backward, thresholds)

    if scores.with_context:
        assert scores.context_forward is not None and scores.context_backward is not None
        contexted = _bare_outcome(scores.context_forward, scores.context_backward, thresholds)
        if bare is not contexted:
            return Decision(
                disposition=Disposition.ESCALATE_G4,
                escalation_reason=EscalationReason.CONTEXT_DISAGREEMENT,
                truncated=scores.truncated,
                notes={
                    "bare": bare.value if bare else None,
                    "contexted": contexted.value if contexted else None,
                },
            )

    if bare is not None:
        return Decision(
            disposition=Disposition.DECIDED,
            verdict=bare,
            method=Method.GK_G1_NLI,
            truncated=scores.truncated,
        )

    return Decision(disposition=Disposition.FORWARD, truncated=scores.truncated)


# --- G2_CORROBORATION ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class G2Scores:
    """Support in both directions, and against the source snippet where one exists."""

    forward: GroundingScore
    backward: GroundingScore
    grounded: GroundingScore | None = None

    @property
    def truncated(self) -> bool:
        return (
            self.forward.truncated
            or self.backward.truncated
            or bool(self.grounded and self.grounded.truncated)
        )


def decide_g2(scores: G2Scores, thresholds: Thresholds) -> Decision:
    """G2_CORROBORATION — sees the G1 leftovers (LLD §8)."""
    candidates: list[tuple[float, Direction]] = [
        (scores.forward.support, Direction.FORWARD),
        (scores.backward.support, Direction.BACKWARD),
    ]
    if scores.grounded is not None:
        candidates.append((scores.grounded.support, Direction.GROUNDED))

    best, direction = max(candidates)

    if best >= thresholds.support_min:
        return Decision(
            disposition=Disposition.DECIDED,
            verdict=Verdict.CORROBORATES,
            method=Method.GK_G2_GROUNDING,
            direction=direction,
            # The LLD says `calibrated(score)`. No calibrator is fitted yet — fitting one
            # is what this bake-off's ECE numbers are for — and mapping the raw score
            # through an invented curve would make the confidence look earned when it is
            # not. So it is the raw score, and the report says how well calibrated it is.
            confidence=best,
            truncated=scores.truncated,
        )

    return Decision(
        disposition=Disposition.FORWARD,
        truncated=scores.truncated,
        notes={"bestSupport": best, "belowFloor": best < thresholds.support_floor},
    )


# --- G3_CONTRADICTION ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class G3Scores:
    """The second family's pass. G1's logits come along as family A (LLD §8)."""

    family_b_forward: NliScores
    family_b_backward: NliScores

    @property
    def max_contradiction(self) -> float:
        return max(self.family_b_forward.contradiction, self.family_b_backward.contradiction)

    @property
    def truncated(self) -> bool:
        return self.family_b_forward.truncated or self.family_b_backward.truncated


def _leans_neutral(scores: NliScores) -> bool:
    return scores.neutral >= scores.entailment and scores.neutral >= scores.contradiction


def decide_g3(g1: G1Scores, g3: G3Scores, thresholds: Thresholds) -> Decision:
    """G3_CONTRADICTION — cross-architecture agreement replaces self-consistency (LLD §8)."""
    family_a = g1.max_contradiction
    family_b = g3.max_contradiction
    truncated = g1.truncated or g3.truncated

    if family_a >= thresholds.contra_min and family_b >= thresholds.contra_min:
        # The cascade never finalizes CONTRADICTS; the human queue confirms, as today.
        return Decision(
            disposition=Disposition.ROUTE_HUMAN,
            method=Method.GK_G3_XCHECK,
            escalation_reason=EscalationReason.CONTRADICTION_SIGNAL,
            contradiction_flag=True,
            truncated=truncated,
            notes={"familyA": family_a, "familyB": family_b, "candidate": "CONTRADICTS"},
        )

    both_neutral_lean = all(
        _leans_neutral(scores)
        for scores in (g1.forward, g1.backward, g3.family_b_forward, g3.family_b_backward)
    )
    if (
        family_a < thresholds.neutral_consensus
        and family_b < thresholds.neutral_consensus
        and both_neutral_lean
    ):
        return Decision(
            disposition=Disposition.DECIDED,
            verdict=Verdict.NEUTRAL,
            method=Method.GK_G3_XCHECK,
            truncated=truncated,
            notes={"familyA": family_a, "familyB": family_b},
        )

    return Decision(
        disposition=Disposition.ESCALATE_G4,
        escalation_reason=EscalationReason.STAGE_DISAGREEMENT,
        truncated=truncated,
        notes={"familyA": family_a, "familyB": family_b},
    )
