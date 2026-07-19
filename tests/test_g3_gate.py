"""G3_CONTRADICTION against the emulator (LLD §8, VA-101).

The gate with the most load-bearing property in the cascade: it finds contradiction
candidates and it never finalizes one. Both halves of that are tested here — the routing
that keeps `decidedBy` unset, and the edge row that still reaches the human queue.
"""

from __future__ import annotations

import random
import uuid

import pytest

from gatekeeper.config import load_config
from gatekeeper.enums import Gate, JudgeMode, Method, QueueTier, Verdict
from gatekeeper.errors import Neo4jUnavailableError
from gatekeeper.gates.decisions import (
    Disposition,
    G1Scores,
    G3Scores,
    Thresholds,
    decide_g3,
)
from gatekeeper.gates.g3 import run_g3
from gatekeeper.scoring.encoder import NliScores
from gatekeeper.worker.edges import edge_key, g1_stage_scores
from gatekeeper.worker.gates import GateContext
from gatekeeper.worker.queue import PairQueue, QueuedPair, pair_key
from tests.test_g1_gate import FakeGraph, _pairs, _texts

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"


def _nli(contradiction: float, *, neutral: float | None = None) -> NliScores:
    """An NLI row with the contradiction mass the test cares about."""
    if neutral is None:
        neutral = 1.0 - contradiction - 0.005
    entailment = max(0.0, 1.0 - contradiction - neutral)
    return NliScores(entailment=entailment, neutral=neutral, contradiction=contradiction)


CONTRADICTORY = _nli(0.92)
NEUTRAL_LEAN = _nli(0.01)
AMBIGUOUS = _nli(0.50)


# --- fixtures ------------------------------------------------------------------------------


@pytest.fixture
def collections():
    suffix = uuid.uuid4().hex[:12]
    return {
        "queue": f"gatekeeper_pairs_test_{suffix}",
        "edges": f"stage3_edges_test_{suffix}",
        "runs": f"stage3_runs_test_{suffix}",
    }


@pytest.fixture
def gate_config(collections):
    return load_config(
        {
            "GATEKEEPER_QUEUE_COLLECTION": collections["queue"],
            "GATEKEEPER_STAGE3_EDGES_COLLECTION": collections["edges"],
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": collections["runs"],
            "GATEKEEPER_QUEUE_BATCH_SIZE": "4",
        }
    )


@pytest.fixture
def subject(emulator_client, collections, claimed):
    subject_id = f"subject-{uuid.uuid4().hex[:8]}"
    emulator_client.collection(collections["runs"]).document(claimed.stage3_run_id).set(
        {"subjectId": subject_id, "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )
    return subject_id


class StubNliScorer:
    """Family B's pass, keyed by premise text."""

    def __init__(self, by_premise, default=NEUTRAL_LEAN):
        self.by_premise = by_premise
        self.default = default
        self.calls = 0

    def score_nli(self, pairs):
        self.calls += 1
        return [self.by_premise.get(premise, self.default) for premise, _ in pairs]


def _seed_at_g3(emulator_client, collections, claimed, subject_id, pairs, family_a):
    """Put pairs where G1/G2 would have left them: at G3, with G1's scores on the row.

    Written through the same ``stageScores.g1`` shape G1 writes, because that shape is
    exactly what this gate has to be able to read back.
    """
    queue = emulator_client.collection(collections["queue"])
    edges = emulator_client.collection(collections["edges"])
    for pair in pairs:
        key = pair_key(claimed.stage3_run_id, pair.claim_a_id, pair.claim_b_id)
        queue.document(key).set(
            QueuedPair(
                pair_id=key,
                stage3_run_id=claimed.stage3_run_id,
                intake_id=claimed.intake_id,
                claim_a_id=pair.claim_a_id,
                claim_b_id=pair.claim_b_id,
                rank=pair.rank,
                gate=Gate.G3_CONTRADICTION,
                contradiction_flag=True,
            ).to_firestore()
        )
        scores = family_a[pair.claim_a_id]
        edges.document(edge_key(pair.claim_a_id, pair.claim_b_id)).set(
            {
                "subjectId": subject_id,
                "claimIdLow": min(pair.claim_a_id, pair.claim_b_id),
                "claimIdHigh": max(pair.claim_a_id, pair.claim_b_id),
                "stage3RunId": claimed.stage3_run_id,
                "method": Method.GK_G1_NLI.value,
                "stageScores": {"g1": g1_stage_scores(scores.forward, scores.backward)},
            }
        )


def _context(emulator_client, gate_config, run, graph, scorer, judge_mode=JudgeMode.GATEKEEPER):
    run.judge_mode = judge_mode
    return GateContext(
        run=run,
        gate=Gate.G3_CONTRADICTION,
        config=gate_config,
        client=emulator_client,
        lease_owner=OWNER,
        reader_factory=lambda: graph,
        scorer_factory=lambda binding: scorer,
    )


def _queue(emulator_client, collections):
    return PairQueue(emulator_client, collection=collections["queue"])


def _uniform(pairs, scores):
    return {pair.claim_a_id: G1Scores(forward=scores, backward=scores) for pair in pairs}


# --- the three outcomes ---------------------------------------------------------------------


def test_two_family_agreement_routes_to_the_human_queue(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(2)
    texts = _texts(pairs)
    _seed_at_g3(
        emulator_client, collections, claimed, subject, pairs, _uniform(pairs, CONTRADICTORY)
    )

    counters = run_g3(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubNliScorer({}, default=CONTRADICTORY),
        )
    )

    assert counters["humanRouted"] == 2
    assert counters["neutral"] == 0
    assert counters["escalated"] == 0

    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert {pair.tier for pair in queued} == {QueueTier.HUMAN}
    assert {pair.gate for pair in queued} == {None}
    assert {pair.decided_by for pair in queued} == {None}


def test_two_family_neutral_consensus_decides_neutral(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(2)
    texts = _texts(pairs)
    _seed_at_g3(
        emulator_client, collections, claimed, subject, pairs, _uniform(pairs, NEUTRAL_LEAN)
    )

    counters = run_g3(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubNliScorer({}, default=NEUTRAL_LEAN),
        )
    )

    assert counters["neutral"] == 2
    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert {pair.decided_by for pair in queued} == {Method.GK_G3_XCHECK.value}
    assert {pair.tier for pair in queued} == {QueueTier.CASCADE}


def test_family_disagreement_escalates_to_g4(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """One family sure, the other not — the cross-check's whole reason for existing."""
    pairs = _pairs(2)
    texts = _texts(pairs)
    _seed_at_g3(
        emulator_client, collections, claimed, subject, pairs, _uniform(pairs, CONTRADICTORY)
    )

    counters = run_g3(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubNliScorer({}, default=AMBIGUOUS),
        )
    )

    assert counters["escalated"] == 2
    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert {pair.gate for pair in queued} == {Gate.G4_ESCALATION}
    assert {pair.tier for pair in queued} == {QueueTier.LLM_TAIL}


# --- the property ----------------------------------------------------------------------------


def test_g3_never_writes_contradicts_as_a_final_verdict(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """The DoD property, swept over the whole score space.

    "Final" is the operative word, and `decidedBy` is the field that asserts it: a pair the
    cascade settled carries the method that settled it. Every contradiction candidate must
    instead leave with `tier = HUMAN` and no `decidedBy` — routed for review, never
    concluded. The edge row's `relation` is a separate question, tested below.
    """
    generator = random.Random(20260719)
    pairs = _pairs(48)
    texts = _texts(pairs)

    family_a = {}
    family_b_by_premise = {}
    for pair in pairs:
        a_contradiction = generator.choice([0.0, 0.05, 0.4, 0.84, 0.86, 0.95, 0.999])
        b_contradiction = generator.choice([0.0, 0.05, 0.4, 0.84, 0.86, 0.95, 0.999])
        family_a[pair.claim_a_id] = G1Scores(
            forward=_nli(a_contradiction), backward=_nli(a_contradiction)
        )
        for claim_id in (pair.claim_a_id, pair.claim_b_id):
            family_b_by_premise[texts[claim_id]] = _nli(b_contradiction)

    _seed_at_g3(emulator_client, collections, claimed, subject, pairs, family_a)
    counters = run_g3(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubNliScorer(family_b_by_premise),
        )
    )

    assert counters["humanRouted"] > 0, "the fixture must actually produce candidates"

    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert len(queued) == len(pairs)
    for pair in queued:
        if pair.tier is QueueTier.HUMAN:
            assert pair.decided_by is None, "a contradiction candidate was finalized"
            assert pair.gate is None
        else:
            assert pair.decided_by in (None, Method.GK_G3_XCHECK.value)

    # And nothing in the cascade lane ever carries a CONTRADICTS relation.
    for snapshot in emulator_client.collection(collections["edges"]).stream():
        document = snapshot.to_dict() or {}
        if document.get("relation") == Verdict.CONTRADICTS.value:
            assert document["escalationReason"] == "CONTRADICTION_SIGNAL"


def test_a_human_routed_pair_is_visible_to_the_contradiction_queue_reader(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """DoD: the row a candidate writes must be one vishwamitra can actually surface.

    The §11.10 queue reads Fact-level `CONTRADICTS` edges stamped `reviewStatus = PROPOSED`,
    and those are lifted by `FactAssembler` from exactly one thing: a `stage3_edges` pair row
    whose `relation` is CONTRADICTS, carrying a `confidence` its floor can filter on and the
    subject/claim ids it joins by. A candidate that wrote no relation would be invisible to
    the review queue built to review it — the failure this asserts against.
    """
    pairs = _pairs(1)
    texts = _texts(pairs)
    _seed_at_g3(emulator_client, collections, claimed, subject, pairs, _uniform(pairs, _nli(0.93)))

    run_g3(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubNliScorer({}, default=_nli(0.88)),
        )
    )

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()

    assert document["relation"] == Verdict.CONTRADICTS.value
    assert document["method"] == Method.GK_G3_XCHECK.value
    assert document["escalationReason"] == "CONTRADICTION_SIGNAL"
    assert document["subjectId"] == subject
    assert document["claimIdLow"] == min(pairs[0].claim_a_id, pairs[0].claim_b_id)
    assert document["claimIdHigh"] == max(pairs[0].claim_a_id, pairs[0].claim_b_id)

    # The weaker family bounds the agreement, and the queue's floor filters on it.
    assert document["confidence"] == pytest.approx(0.88)


# --- family A comes off the row, never from a second inference pass ---------------------------


def test_family_a_is_read_back_from_stage_scores_and_never_re_inferred(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§8: G1's logits "are already in stageScores".

    Two calls to the scorer — forward and backward for family B — and not one more. A third
    would mean the cross-check had quietly re-derived family A and become a self-check.
    """
    pairs = _pairs(2)
    texts = _texts(pairs)
    _seed_at_g3(
        emulator_client, collections, claimed, subject, pairs, _uniform(pairs, CONTRADICTORY)
    )
    scorer = StubNliScorer({}, default=CONTRADICTORY)

    run_g3(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            scorer,
        )
    )

    assert scorer.calls == 2


def test_a_pair_without_g1_scores_fails_rather_than_re_inferring_or_defaulting(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """A missing family A read as "no contradiction" would discard the very pairs G3 exists
    to catch, so the gate stops and the operator retriggers FROM_GATE at G1."""
    pairs = _pairs(1)
    texts = _texts(pairs)
    _seed_at_g3(
        emulator_client, collections, claimed, subject, pairs, _uniform(pairs, CONTRADICTORY)
    )
    # The row is there but its G1 slot is not — a purge that outran a resume.
    emulator_client.collection(collections["edges"]).document(
        edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    ).set({"stage3RunId": claimed.stage3_run_id, "method": Method.GK_G1_NLI.value})

    with pytest.raises(Neo4jUnavailableError, match=r"stageScores\.g1"):
        run_g3(
            _context(
                emulator_client,
                gate_config,
                store.get(claimed.run_request_id),
                FakeGraph(pairs, texts),
                StubNliScorer({}, default=CONTRADICTORY),
            )
        )


# --- SHADOW ------------------------------------------------------------------------------------


def test_shadow_mode_records_the_candidate_without_asserting_it(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§9: in SHADOW the ensemble is still the verdict writer — but the disagreement report
    still needs to know the cascade would have flagged this pair."""
    pairs = _pairs(1)
    texts = _texts(pairs)
    _seed_at_g3(
        emulator_client, collections, claimed, subject, pairs, _uniform(pairs, CONTRADICTORY)
    )

    run_g3(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubNliScorer({}, default=CONTRADICTORY),
            judge_mode=JudgeMode.SHADOW,
        )
    )

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()
    assert document["shadow"]["verdict"] == Verdict.CONTRADICTS.value
    assert "relation" not in document, "SHADOW must not assert a verdict on the row"


# --- replay parity -------------------------------------------------------------------------------


def test_the_gate_loop_agrees_with_the_replay_harness(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """VA-97's harness and the production gate must reach the same verdict (DoD)."""
    profiles = [
        (CONTRADICTORY, CONTRADICTORY),
        (NEUTRAL_LEAN, NEUTRAL_LEAN),
        (CONTRADICTORY, AMBIGUOUS),
        (AMBIGUOUS, AMBIGUOUS),
    ]
    pairs = _pairs(len(profiles))
    texts = _texts(pairs)

    family_a = {}
    family_b_by_premise = {}
    for pair, (a_scores, b_scores) in zip(pairs, profiles, strict=True):
        family_a[pair.claim_a_id] = G1Scores(forward=a_scores, backward=a_scores)
        for claim_id in (pair.claim_a_id, pair.claim_b_id):
            family_b_by_premise[texts[claim_id]] = b_scores

    _seed_at_g3(emulator_client, collections, claimed, subject, pairs, family_a)
    run = store.get(claimed.run_request_id)
    thresholds = Thresholds.from_snapshot(run.config_snapshot)

    run_g3(
        _context(
            emulator_client,
            gate_config,
            run,
            FakeGraph(pairs, texts),
            StubNliScorer(family_b_by_premise),
        )
    )

    queued = {
        pair.claim_a_id: pair
        for pair in _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    }
    for pair, (a_scores, b_scores) in zip(pairs, profiles, strict=True):
        expected = decide_g3(
            G1Scores(forward=a_scores, backward=a_scores),
            G3Scores(family_b_forward=b_scores, family_b_backward=b_scores),
            thresholds,
        )
        actual = queued[pair.claim_a_id]

        if expected.disposition is Disposition.ROUTE_HUMAN:
            assert actual.tier is QueueTier.HUMAN
            assert actual.decided_by is None
        elif expected.disposition is Disposition.DECIDED:
            assert actual.decided_by == Method.GK_G3_XCHECK.value
            assert actual.gate is None
        else:
            assert actual.gate is Gate.G4_ESCALATION
