"""The gate decision rules (VA-97 / LLD §8).

These are the functions the bake-off scores through and sessions 04-06 wrap in worker
plumbing, so they are tested against the LLD's pseudocode branch by branch — especially
the branches whose failure mode is silence rather than an exception.
"""

from __future__ import annotations

import pytest

from gatekeeper.config import load_config
from gatekeeper.enums import EscalationReason, Method, Verdict
from gatekeeper.gates import (
    Direction,
    Disposition,
    G1Scores,
    G2Scores,
    G3Scores,
    Thresholds,
    decide_g1,
    decide_g2,
    decide_g3,
)
from gatekeeper.scoring.encoder import GroundingScore, NliScores

T = Thresholds()


def nli(entailment: float, neutral: float, contradiction: float, truncated: bool = False):
    return NliScores(
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
        truncated=truncated,
    )


NEUTRAL_BOTH = G1Scores(forward=nli(0.02, 0.97, 0.01), backward=nli(0.02, 0.97, 0.01))


# --- thresholds ---------------------------------------------------------------------------


def test_thresholds_default_to_the_lld_proposals() -> None:
    assert (T.neutral_min, T.repeat_min, T.contra_escape) == (0.95, 0.90, 0.02)
    assert (T.support_min, T.contra_min) == (0.90, 0.85)


def test_thresholds_round_trip_through_a_config_snapshot() -> None:
    """The bake-off sweeps by editing a snapshot; it must reach every number."""
    snapshot = load_config(
        env={
            "GATEKEEPER_G1_NEUTRAL_MIN": "0.80",
            "GATEKEEPER_G2_SUPPORT_MIN": "0.70",
            "GATEKEEPER_G3_CONTRA_MIN": "0.60",
            "GATEKEEPER_G3_NEUTRAL_CONSENSUS": "0.05",
        }
    ).snapshot()

    thresholds = Thresholds.from_snapshot(snapshot)

    assert thresholds.neutral_min == 0.80
    assert thresholds.support_min == 0.70
    assert thresholds.contra_min == 0.60
    assert thresholds.neutral_consensus == 0.05


def test_an_empty_snapshot_falls_back_to_the_proposals() -> None:
    assert Thresholds.from_snapshot({}) == Thresholds()


# --- G1_NEUTRAL ---------------------------------------------------------------------------


def test_g1_decides_neutral_when_both_directions_agree() -> None:
    decision = decide_g1(NEUTRAL_BOTH, T)
    assert decision.disposition is Disposition.DECIDED
    assert decision.verdict is Verdict.NEUTRAL
    assert decision.method is Method.GK_G1_NLI


def test_g1_requires_both_directions_for_neutral() -> None:
    """One-directional neutrality is not neutrality; it forwards instead."""
    scores = G1Scores(forward=nli(0.02, 0.97, 0.01), backward=nli(0.60, 0.39, 0.01))
    assert decide_g1(scores, T).disposition is Disposition.FORWARD


def test_g1_decides_repeats_when_both_directions_entail() -> None:
    scores = G1Scores(forward=nli(0.95, 0.04, 0.01), backward=nli(0.93, 0.06, 0.01))
    decision = decide_g1(scores, T)
    assert decision.verdict is Verdict.REPEATS
    assert decision.method is Method.GK_G1_NLI


def test_the_contradiction_escape_hatch_beats_an_overwhelming_neutral() -> None:
    """The non-negotiable rule: a flagged pair may NEVER be neutral-discarded."""
    scores = G1Scores(forward=nli(0.00, 0.97, 0.03), backward=nli(0.00, 0.99, 0.00))

    decision = decide_g1(scores, T)

    assert decision.disposition is Disposition.FORWARD
    assert decision.contradiction_flag is True
    assert decision.verdict is None


def test_the_escape_hatch_is_strictly_greater_than_the_threshold() -> None:
    """`> contraEscape` in the pseudocode, not `>=` — exactly at it is not a flag."""
    at = G1Scores(forward=nli(0.01, 0.97, 0.02), backward=nli(0.01, 0.97, 0.02))
    just_over = G1Scores(forward=nli(0.01, 0.97, 0.021), backward=nli(0.01, 0.97, 0.02))

    assert decide_g1(at, T).contradiction_flag is False
    assert decide_g1(just_over, T).contradiction_flag is True


def test_the_escape_hatch_fires_on_either_direction() -> None:
    backward_only = G1Scores(forward=nli(0.01, 0.98, 0.00), backward=nli(0.01, 0.90, 0.09))
    assert decide_g1(backward_only, T).contradiction_flag is True


def test_g1_escalates_when_context_changes_the_verdict() -> None:
    """§11.9 dual-eval: the bare pair reads neutral, the contexted one does not."""
    scores = G1Scores(
        forward=nli(0.02, 0.97, 0.01),
        backward=nli(0.02, 0.97, 0.01),
        context_forward=nli(0.55, 0.44, 0.01),
        context_backward=nli(0.55, 0.44, 0.01),
    )

    decision = decide_g1(scores, T)

    assert decision.disposition is Disposition.ESCALATE_G4
    assert decision.escalation_reason is EscalationReason.CONTEXT_DISAGREEMENT


def test_g1_does_not_escalate_when_context_agrees() -> None:
    scores = G1Scores(
        forward=nli(0.02, 0.97, 0.01),
        backward=nli(0.02, 0.97, 0.01),
        context_forward=nli(0.01, 0.98, 0.01),
        context_backward=nli(0.01, 0.98, 0.01),
    )
    assert decide_g1(scores, T).verdict is Verdict.NEUTRAL


def test_a_contradiction_flag_outranks_a_context_disagreement() -> None:
    """Both apply; the escape hatch runs first so the pair still reaches G3."""
    scores = G1Scores(
        forward=nli(0.00, 0.95, 0.05),
        backward=nli(0.00, 0.99, 0.01),
        context_forward=nli(0.99, 0.01, 0.00),
        context_backward=nli(0.99, 0.01, 0.00),
    )

    decision = decide_g1(scores, T)

    assert decision.disposition is Disposition.FORWARD
    assert decision.contradiction_flag is True


def test_g1_carries_truncation_forward() -> None:
    scores = G1Scores(forward=nli(0.02, 0.97, 0.01, truncated=True), backward=nli(0.02, 0.97, 0.01))
    assert decide_g1(scores, T).truncated is True


# --- G2_CORROBORATION ---------------------------------------------------------------------


def test_g2_corroborates_on_the_strongest_direction() -> None:
    scores = G2Scores(forward=GroundingScore(0.40), backward=GroundingScore(0.94))

    decision = decide_g2(scores, T)

    assert decision.verdict is Verdict.CORROBORATES
    assert decision.method is Method.GK_G2_GROUNDING
    assert decision.direction is Direction.BACKWARD
    assert decision.confidence == 0.94


def test_g2_prefers_the_grounded_score_when_it_is_the_strongest() -> None:
    """The evidence mode today's judge does not have (LLD §8)."""
    scores = G2Scores(
        forward=GroundingScore(0.40),
        backward=GroundingScore(0.50),
        grounded=GroundingScore(0.97),
    )
    assert decide_g2(scores, T).direction is Direction.GROUNDED


def test_g2_forwards_everything_undecided() -> None:
    scores = G2Scores(forward=GroundingScore(0.60), backward=GroundingScore(0.70))
    decision = decide_g2(scores, T)
    assert decision.disposition is Disposition.FORWARD
    assert decision.verdict is None
    assert decision.notes["bestSupport"] == 0.70


def test_g2_marks_pairs_below_the_support_floor() -> None:
    scores = G2Scores(forward=GroundingScore(0.10), backward=GroundingScore(0.20))
    assert decide_g2(scores, T).notes["belowFloor"] is True


def test_g2_support_min_is_inclusive() -> None:
    scores = G2Scores(forward=GroundingScore(0.90), backward=GroundingScore(0.10))
    assert decide_g2(scores, T).verdict is Verdict.CORROBORATES


# --- G3_CONTRADICTION ---------------------------------------------------------------------


def _g1_with_contradiction(value: float) -> G1Scores:
    return G1Scores(forward=nli(0.0, 1 - value, value), backward=nli(0.0, 1 - value, value))


def _g3_with_contradiction(value: float) -> G3Scores:
    return G3Scores(
        family_b_forward=nli(0.0, 1 - value, value),
        family_b_backward=nli(0.0, 1 - value, value),
    )


def test_g3_routes_a_two_family_contradiction_to_a_human() -> None:
    """The cascade never finalizes CONTRADICTS itself."""
    decision = decide_g3(_g1_with_contradiction(0.90), _g3_with_contradiction(0.88), T)

    assert decision.disposition is Disposition.ROUTE_HUMAN
    assert decision.verdict is None
    assert decision.contradiction_flag is True
    assert decision.escalation_reason is EscalationReason.CONTRADICTION_SIGNAL


def test_one_family_alone_is_not_enough_to_route_a_human() -> None:
    """Cross-architecture agreement is the whole point; one family is an escalation."""
    decision = decide_g3(_g1_with_contradiction(0.95), _g3_with_contradiction(0.20), T)

    assert decision.disposition is Disposition.ESCALATE_G4
    assert decision.escalation_reason is EscalationReason.STAGE_DISAGREEMENT


def test_g3_decides_neutral_only_on_two_family_consensus() -> None:
    decision = decide_g3(_g1_with_contradiction(0.01), _g3_with_contradiction(0.02), T)

    assert decision.disposition is Disposition.DECIDED
    assert decision.verdict is Verdict.NEUTRAL
    assert decision.method is Method.GK_G3_XCHECK


def test_g3_will_not_call_neutral_when_a_direction_leans_entailment() -> None:
    """`both_neutral_lean` is read strictly: every direction of both families."""
    g1 = G1Scores(forward=nli(0.80, 0.19, 0.01), backward=nli(0.01, 0.98, 0.01))
    decision = decide_g3(g1, _g3_with_contradiction(0.01), T)

    assert decision.disposition is Disposition.ESCALATE_G4


def test_a_pair_between_the_two_bars_escalates() -> None:
    """Neither confident enough to route nor quiet enough to discard — that is G4's tail."""
    decision = decide_g3(_g1_with_contradiction(0.50), _g3_with_contradiction(0.50), T)
    assert decision.disposition is Disposition.ESCALATE_G4


@pytest.mark.parametrize("value", [0.10, 0.11, 0.5])
def test_neutral_consensus_is_strict(value: float) -> None:
    """At or above the consensus bar is not consensus; it escalates."""
    decision = decide_g3(_g1_with_contradiction(value), _g3_with_contradiction(0.0), T)
    assert decision.disposition is Disposition.ESCALATE_G4


def test_g3_never_decides_contradicts_as_a_verdict() -> None:
    """The one invariant the human queue depends on, across the whole threshold space."""
    for family_a in (0.0, 0.3, 0.6, 0.9, 1.0):
        for family_b in (0.0, 0.3, 0.6, 0.9, 1.0):
            decision = decide_g3(
                _g1_with_contradiction(family_a), _g3_with_contradiction(family_b), T
            )
            assert decision.verdict is not Verdict.CONTRADICTS
