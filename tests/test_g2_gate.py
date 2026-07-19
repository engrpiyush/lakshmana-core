"""G2_CORROBORATION against the emulator (LLD §8, VA-100).

Same doubles as `test_g1_gate.py` and for the same reasons: the graph and the ONNX session
are stubbed, everything between them is the code that ships. The properties under test are
the four the DoD names for this gate — the flagged pass-through, the grounding mode, replay
parity, and the confidence that lands on the row.
"""

from __future__ import annotations

import uuid

import pytest

from gatekeeper.config import load_config
from gatekeeper.enums import Gate, JudgeMode, Method, QueueTier, Verdict
from gatekeeper.errors import Neo4jUnavailableError
from gatekeeper.gates.decisions import Direction, Disposition, G2Scores, Thresholds, decide_g2
from gatekeeper.gates.g2 import run_g2
from gatekeeper.scoring.encoder import GroundingScore
from gatekeeper.worker.edges import edge_key
from gatekeeper.worker.gates import GateContext
from gatekeeper.worker.queue import PairQueue, QueuedPair, pair_key
from tests.test_g1_gate import FakeGraph, _pairs, _texts

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"

STRONG = GroundingScore(support=0.97)
WEAK = GroundingScore(support=0.20)
MIDDLING = GroundingScore(support=0.60)


class StubGroundingScorer:
    """Returns support keyed by the ``(document, claim)`` premise/hypothesis pair."""

    def __init__(self, by_pair, default=WEAK):
        self.by_pair = by_pair
        self.default = default
        self.seen: list[tuple[str, str]] = []

    def score_grounding(self, pairs):
        self.seen.extend(pairs)
        return [self.by_pair.get((document, claim), self.default) for document, claim in pairs]


# --- fixtures ------------------------------------------------------------------------------


@pytest.fixture
def collections():
    suffix = uuid.uuid4().hex[:12]
    return {
        "queue": f"gatekeeper_pairs_test_{suffix}",
        "edges": f"stage3_edges_test_{suffix}",
        "runs": f"stage3_runs_test_{suffix}",
        "claims": f"claims_test_{suffix}",
    }


@pytest.fixture
def gate_config(collections):
    return load_config(
        {
            "GATEKEEPER_QUEUE_COLLECTION": collections["queue"],
            "GATEKEEPER_STAGE3_EDGES_COLLECTION": collections["edges"],
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": collections["runs"],
            "GATEKEEPER_CLAIMS_COLLECTION": collections["claims"],
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


def _seed_at_g2(emulator_client, collections, claimed, pairs, *, flagged=()):
    """Put pairs where G1 would have left them: waiting at G2."""
    queue = emulator_client.collection(collections["queue"])
    for pair in pairs:
        key = pair_key(claimed.stage3_run_id, pair.claim_a_id, pair.claim_b_id)
        queued = QueuedPair(
            pair_id=key,
            stage3_run_id=claimed.stage3_run_id,
            intake_id=claimed.intake_id,
            claim_a_id=pair.claim_a_id,
            claim_b_id=pair.claim_b_id,
            rank=pair.rank,
            with_context=pair.with_context,
            gate=Gate.G2_CORROBORATION,
            contradiction_flag=pair.claim_a_id in flagged,
        )
        queue.document(key).set(queued.to_firestore())


def _context(emulator_client, gate_config, run, graph, scorer, judge_mode=JudgeMode.GATEKEEPER):
    run.judge_mode = judge_mode
    return GateContext(
        run=run,
        gate=Gate.G2_CORROBORATION,
        config=gate_config,
        client=emulator_client,
        lease_owner=OWNER,
        reader_factory=lambda: graph,
        scorer_factory=lambda binding: scorer,
    )


def _queue(emulator_client, collections):
    return PairQueue(emulator_client, collection=collections["queue"])


# --- the happy path ------------------------------------------------------------------------


def test_g2_decides_corroborates_above_support_min(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(2)
    texts = _texts(pairs)
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    counters = run_g2(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubGroundingScorer({}, default=STRONG),
        )
    )

    assert counters["seen"] == 2
    assert counters["corroborates"] == 2
    assert counters["forwarded"] == 0

    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert {pair.gate for pair in queued} == {None}
    assert {pair.decided_by for pair in queued} == {Method.GK_G2_GROUNDING.value}


def test_g2_forwards_undecided_pairs_to_g3(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(2)
    texts = _texts(pairs)
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    counters = run_g2(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubGroundingScorer({}, default=MIDDLING),
        )
    )

    assert counters["forwarded"] == 2
    assert counters["corroborates"] == 0
    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert {pair.gate for pair in queued} == {Gate.G3_CONTRADICTION}
    assert {pair.tier for pair in queued} == {QueueTier.CASCADE}


# --- the flagged pass-through ----------------------------------------------------------------


def test_a_contradiction_flagged_pair_reaches_g3_without_being_scored(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§8's ordering note: G2 may not finalize CORROBORATES on a flagged pair.

    The scores here would decide CORROBORATES for every pair, so if the flag were ignored
    the flagged one would leave the cascade and never reach the cross-check built to
    adjudicate it. It must arrive at G3 *and* cost no inference on the way.
    """
    pairs = _pairs(3)
    texts = _texts(pairs)
    flagged = {pairs[1].claim_a_id}
    _seed_at_g2(emulator_client, collections, claimed, pairs, flagged=flagged)

    scorer = StubGroundingScorer({}, default=STRONG)
    counters = run_g2(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            scorer,
        )
    )

    assert counters["flaggedPassThrough"] == 1
    assert counters["seen"] == 2, "the flagged pair is not judged here"
    assert counters["corroborates"] == 2

    queued = {
        pair.claim_a_id: pair
        for pair in _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    }
    flagged_pair = queued[pairs[1].claim_a_id]
    assert flagged_pair.gate is Gate.G3_CONTRADICTION
    assert flagged_pair.decided_by is None
    assert flagged_pair.contradiction_flag is True

    # No inference was spent on it, and no row claims this gate judged it.
    scored_texts = {text for pair in scorer.seen for text in pair}
    assert texts[pairs[1].claim_a_id] not in scored_texts

    key = edge_key(pairs[1].claim_a_id, pairs[1].claim_b_id)
    assert not emulator_client.collection(collections["edges"]).document(key).get().exists


# --- grounding mode --------------------------------------------------------------------------


def test_grounding_mode_scores_the_claim_against_the_other_claims_source_excerpt(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """The evidence mode: MiniCheck's native ``(document, claim)`` shape (§8).

    Claim-vs-claim is deliberately weak here and only the excerpt supports the claim, so a
    CORROBORATES verdict is proof the grounded score was both computed and consulted.
    """
    pairs = _pairs(1)
    texts = _texts(pairs)
    excerpt = f"verbatim source behind {pairs[0].claim_b_id}"
    emulator_client.collection(collections["claims"]).document(pairs[0].claim_b_id).set(
        {"sourceExcerpt": excerpt}
    )
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    scorer = StubGroundingScorer({(excerpt, texts[pairs[0].claim_a_id]): STRONG}, default=WEAK)
    counters = run_g2(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            scorer,
        )
    )

    assert counters["grounded"] == 1
    assert counters["corroborates"] == 1

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()
    scores = document["stageScores"]["g2"]
    assert scores["supportGroundedFwd"] == pytest.approx(STRONG.support)
    assert scores["supportGrounded"] == pytest.approx(STRONG.support)
    assert scores["supportFwd"] == pytest.approx(WEAK.support)
    assert document["relation"] == Verdict.CORROBORATES.value


def test_a_pair_with_no_source_excerpt_is_still_judged(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """A claim extracted from audio has no printed excerpt; that is normal, not an error."""
    pairs = _pairs(1)
    texts = _texts(pairs)
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    counters = run_g2(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubGroundingScorer({}, default=STRONG),
        )
    )

    assert counters["grounded"] == 0
    assert counters["corroborates"] == 1

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()
    assert "supportGrounded" not in document["stageScores"]["g2"]


def test_grounding_mode_off_never_reads_an_excerpt(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """``groundingMode = OFF`` is what a bake-off needs to measure grounding's worth."""
    pairs = _pairs(1)
    texts = _texts(pairs)
    excerpt = f"verbatim source behind {pairs[0].claim_b_id}"
    emulator_client.collection(collections["claims"]).document(pairs[0].claim_b_id).set(
        {"sourceExcerpt": excerpt}
    )
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    run = store.get(claimed.run_request_id)
    run.config_snapshot["g2"]["groundingMode"] = "OFF"
    scorer = StubGroundingScorer({}, default=MIDDLING)

    counters = run_g2(_context(emulator_client, gate_config, run, FakeGraph(pairs, texts), scorer))

    assert counters["grounded"] == 0
    assert excerpt not in {text for pair in scorer.seen for text in pair}


# --- confidence ------------------------------------------------------------------------------


def test_the_stored_confidence_is_the_score_the_decision_argmaxed(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """DoD: the confidence on the row is spot-checked against the calibration mapping.

    No calibrator is fitted yet (DEFERRED-LIVE 19), so `decide_g2` documents the mapping as
    the identity — the raw support score, honestly labelled, rather than an invented curve.
    This pins that: whatever the mapping becomes, the row must carry what the decision
    produced, unrounded, and never a number the gate computed for itself.
    """
    pairs = _pairs(1)
    texts = _texts(pairs)
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    forward = GroundingScore(support=0.9312345678)
    backward = GroundingScore(support=0.9112345678)
    scorer = StubGroundingScorer(
        {
            (texts[pairs[0].claim_b_id], texts[pairs[0].claim_a_id]): forward,
            (texts[pairs[0].claim_a_id], texts[pairs[0].claim_b_id]): backward,
        }
    )

    run = store.get(claimed.run_request_id)
    thresholds = Thresholds.from_snapshot(run.config_snapshot)
    run_g2(_context(emulator_client, gate_config, run, FakeGraph(pairs, texts), scorer))

    expected = decide_g2(G2Scores(forward=forward, backward=backward), thresholds)
    assert expected.direction is Direction.FORWARD

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()
    assert document["confidence"] == pytest.approx(expected.confidence, abs=0)
    assert document["confidence"] == pytest.approx(forward.support, abs=0)


# --- SHADOW ------------------------------------------------------------------------------------


def test_shadow_mode_writes_only_the_shadow_block(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(1)
    texts = _texts(pairs)
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    run_g2(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubGroundingScorer({}, default=STRONG),
            judge_mode=JudgeMode.SHADOW,
        )
    )

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()
    assert document["shadow"]["verdict"] == Verdict.CORROBORATES.value
    assert "relation" not in document


# --- replay parity -----------------------------------------------------------------------------


def test_the_gate_loop_agrees_with_the_replay_harness(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """VA-97's harness and the production gate must reach the same verdict (DoD).

    Both call `decide_g2`, so this proves the loop feeds it the directions it thinks it
    does and routes the answer it got back rather than one of its own.
    """
    profiles = [STRONG, WEAK, MIDDLING, GroundingScore(support=0.90)]
    pairs = _pairs(len(profiles))
    texts = _texts(pairs)
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    by_pair = {}
    for pair, score in zip(pairs, profiles, strict=True):
        by_pair[(texts[pair.claim_b_id], texts[pair.claim_a_id])] = score
        by_pair[(texts[pair.claim_a_id], texts[pair.claim_b_id])] = score

    run = store.get(claimed.run_request_id)
    thresholds = Thresholds.from_snapshot(run.config_snapshot)
    run_g2(
        _context(
            emulator_client, gate_config, run, FakeGraph(pairs, texts), StubGroundingScorer(by_pair)
        )
    )

    queued = {
        pair.claim_a_id: pair
        for pair in _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    }
    for pair, score in zip(pairs, profiles, strict=True):
        expected = decide_g2(G2Scores(forward=score, backward=score), thresholds)
        actual = queued[pair.claim_a_id]
        if expected.disposition is Disposition.DECIDED:
            assert actual.decided_by == Method.GK_G2_GROUNDING.value
            assert actual.gate is None
        else:
            assert actual.gate is Gate.G3_CONTRADICTION
            assert actual.decided_by is None


# --- failure modes -----------------------------------------------------------------------------


def test_a_queued_pair_with_no_claim_text_fails_rather_than_guessing(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(1)
    _seed_at_g2(emulator_client, collections, claimed, pairs)

    with pytest.raises(Neo4jUnavailableError, match="no claim text"):
        run_g2(
            _context(
                emulator_client,
                gate_config,
                store.get(claimed.run_request_id),
                FakeGraph(pairs, {}),
                StubGroundingScorer({}),
            )
        )
