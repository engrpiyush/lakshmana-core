"""G1_NEUTRAL end to end against the emulator (LLD §8, §10, §11.9).

Real Firestore transactions, real queue, real edge writes, real
:func:`~gatekeeper.gates.decisions.decide_g1` — with the graph and the ONNX session stubbed,
because those two are the parts a laptop cannot stand up cheaply and the parts this file is
*not* about. Everything between them is the code that ships.

The four properties under test, in the order the DoD asks for them:

* the escape hatch holds through the wiring, not just inside the decision function;
* a crash mid-batch resumes without losing or double-counting a pair;
* the gate loop and the VA-97 replay harness reach the same verdict from the same numbers;
* the FROM_START purge deletes the gatekeeper's rows and only the gatekeeper's rows.
"""

from __future__ import annotations

import random
import uuid

import pytest
from google.cloud.firestore_v1.base_query import FieldFilter

from gatekeeper.config import load_config
from gatekeeper.enums import Gate, JudgeMode, Method, QueueTier, Verdict
from gatekeeper.errors import Neo4jUnavailableError
from gatekeeper.gates.decisions import Disposition, G1Scores, Thresholds, decide_g1
from gatekeeper.gates.g1 import run_g1
from gatekeeper.scoring.encoder import NliScores
from gatekeeper.worker.edges import edge_key
from gatekeeper.worker.gates import GateContext
from gatekeeper.worker.hydrate import GraphPair
from gatekeeper.worker.queue import PairQueue

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"

NEUTRAL = NliScores(entailment=0.02, neutral=0.97, contradiction=0.01)
REPEATS = NliScores(entailment=0.95, neutral=0.04, contradiction=0.01)
CONTRADICTORY = NliScores(entailment=0.02, neutral=0.10, contradiction=0.88)
UNDECIDED = NliScores(entailment=0.50, neutral=0.49, contradiction=0.01)


# --- doubles -------------------------------------------------------------------------------


class FakeGraph:
    """A Neo4j reader that answers the four cyphers `hydrate` issues, and nothing else."""

    def __init__(self, pairs, texts, explanations=None):
        self.pairs = pairs
        self.texts = texts
        self.explanations = explanations or {}
        self.closed = False

    def run(self, cypher: str, **parameters):
        if "JUDGE_QUEUED" in cypher:
            return [
                {
                    "aId": pair.claim_a_id,
                    "bId": pair.claim_b_id,
                    "rank": pair.rank,
                    "withContext": pair.with_context,
                    "humanAsserted": pair.human_asserted,
                }
                for pair in self.pairs
            ]
        # The card query is checked before the explanation one: it *contains* the
        # explanation sub-pattern, so the looser test would swallow it and hand G4 rows
        # with no card fields on them.
        if "c.speakerRole" in cypher:
            return [
                {
                    "claimId": claim_id,
                    "text": self.texts[claim_id],
                    "type": "EMPLOYMENT",
                    "sourceClass": "RESUME",
                    "claimedDate": None,
                    "relationship": None,
                    "speakerRole": None,
                    "explanationText": self.explanations.get(claim_id),
                }
                for claim_id in parameters["claimIds"]
                if claim_id in self.texts
            ]
        source = self.explanations if "Explanation" in cypher else self.texts
        return [
            {"claimId": claim_id, "text": source[claim_id]}
            for claim_id in parameters["claimIds"]
            if claim_id in source
        ]

    def close(self) -> None:
        self.closed = True


class StubScorer:
    """Returns the scores a test asked for, keyed by the premise text."""

    def __init__(self, by_premise, fail_after: int | None = None):
        self.by_premise = by_premise
        self.fail_after = fail_after
        self.calls = 0

    def score_nli(self, pairs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise Neo4jUnavailableError("simulated crash mid-batch")
        return [self.by_premise[premise] for premise, _ in pairs]


# --- fixtures ------------------------------------------------------------------------------


@pytest.fixture
def collections():
    """Isolated collection names, so tests never see each other's rows."""
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
    """The `stage3_runs` doc G1 reads its ``subjectId`` from."""
    subject_id = f"subject-{uuid.uuid4().hex[:8]}"
    emulator_client.collection(collections["runs"]).document(claimed.stage3_run_id).set(
        {"subjectId": subject_id, "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )
    return subject_id


def _pairs(count: int, *, with_context: bool = False):
    return [
        GraphPair(
            claim_a_id=f"claim-a{index}",
            claim_b_id=f"claim-b{index}",
            rank=index,
            with_context=with_context,
        )
        for index in range(count)
    ]


def _texts(pairs):
    texts = {}
    for pair in pairs:
        texts[pair.claim_a_id] = f"text of {pair.claim_a_id}"
        texts[pair.claim_b_id] = f"text of {pair.claim_b_id}"
    return texts


def _context(emulator_client, gate_config, run, graph, scorer, judge_mode=JudgeMode.GATEKEEPER):
    run.judge_mode = judge_mode
    return GateContext(
        run=run,
        gate=Gate.G1_NEUTRAL,
        config=gate_config,
        client=emulator_client,
        lease_owner=OWNER,
        reader_factory=lambda: graph,
        scorer_factory=lambda binding: scorer,
    )


def _queue(emulator_client, collections):
    return PairQueue(emulator_client, collection=collections["queue"])


# --- the happy path ------------------------------------------------------------------------


def test_g1_decides_neutral_pairs_and_takes_them_out_of_the_cascade(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(3)
    texts = _texts(pairs)
    graph = FakeGraph(pairs, texts)
    scorer = StubScorer(dict.fromkeys(texts.values(), NEUTRAL))

    counters = run_g1(
        _context(emulator_client, gate_config, store.get(claimed.run_request_id), graph, scorer)
    )

    assert counters["seen"] == 3
    assert counters["neutral"] == 3
    assert counters["forwarded"] == 0

    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert {pair.gate for pair in queued} == {None}
    assert {pair.decided_by for pair in queued} == {Method.GK_G1_NLI.value}


def test_g1_forwards_undecided_pairs_to_g2(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    pairs = _pairs(2)
    texts = _texts(pairs)
    graph = FakeGraph(pairs, texts)
    scorer = StubScorer(dict.fromkeys(texts.values(), UNDECIDED))

    counters = run_g1(
        _context(emulator_client, gate_config, store.get(claimed.run_request_id), graph, scorer)
    )

    assert counters["forwarded"] == 2
    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert {pair.gate for pair in queued} == {Gate.G2_CORROBORATION}
    assert {pair.tier for pair in queued} == {QueueTier.CASCADE}


def test_g1_writes_full_precision_stage_scores(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§7.2: the raw probabilities are what make offline recalibration possible."""
    pairs = _pairs(1)
    texts = _texts(pairs)
    graph = FakeGraph(pairs, texts)
    scorer = StubScorer(dict.fromkeys(texts.values(), NEUTRAL))

    run_g1(_context(emulator_client, gate_config, store.get(claimed.run_request_id), graph, scorer))

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()

    assert document["method"] == Method.GK_G1_NLI.value
    assert document["relation"] == Verdict.NEUTRAL.value
    assert document["gkRunRequestId"] == claimed.run_request_id
    assert document["stageScores"]["g1"]["neuFwd"] == pytest.approx(NEUTRAL.neutral)
    assert document["stageScores"]["g1"]["conBwd"] == pytest.approx(NEUTRAL.contradiction)


# --- the escape hatch ----------------------------------------------------------------------


def test_a_contradiction_signal_is_never_neutral_discarded_by_the_gate_loop(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """The property, through the wiring: not one flagged pair leaves as NEUTRAL.

    `test_gate_decisions.py` proves this of `decide_g1`; this proves the gate loop does not
    then route it somewhere the decision never said. It is the single most expensive bug
    this system could have — a discarded contradiction is invisible.
    """
    generator = random.Random(20260719)
    thresholds = Thresholds.from_snapshot(store.get(claimed.run_request_id).config_snapshot)

    pairs = _pairs(60)
    texts = _texts(pairs)
    by_premise = {}
    contradictory_texts = set()
    for pair in pairs:
        # Deliberately hostile: an overwhelming neutral alongside a small contradiction, the
        # exact shape the escape hatch exists to survive.
        contradiction = generator.choice([0.0, 0.001, 0.019, 0.021, 0.05, 0.4, 0.99])
        neutral = 1.0 - contradiction - 0.005
        scores = NliScores(entailment=0.005, neutral=neutral, contradiction=contradiction)
        for claim_id in (pair.claim_a_id, pair.claim_b_id):
            by_premise[texts[claim_id]] = scores
        if contradiction > thresholds.contra_escape:
            contradictory_texts.add(pair.claim_a_id)

    graph = FakeGraph(pairs, texts)
    counters = run_g1(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            graph,
            StubScorer(by_premise),
        )
    )

    assert contradictory_texts, "the fixture must actually produce flagged pairs"
    queued = {
        pair.claim_a_id: pair
        for pair in _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    }
    for claim_id in contradictory_texts:
        pair = queued[claim_id]
        assert pair.contradiction_flag is True
        assert pair.decided_by is None, "a flagged pair was decided at G1"
        assert pair.gate is Gate.G2_CORROBORATION
    assert counters["contraFlagged"] == len(contradictory_texts)


# --- context disagreement ------------------------------------------------------------------


def test_bare_versus_contexted_disagreement_goes_straight_to_g4(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§11.9's dual evaluation: G2 and G3 cannot adjudicate this, so it skips them."""
    pairs = _pairs(1, with_context=True)
    texts = _texts(pairs)
    explanations = {claim_id: f"explanation for {claim_id}" for claim_id in texts}
    # Bare reads NEUTRAL; with the explanation appended it no longer does.
    by_premise = dict.fromkeys(texts.values(), NEUTRAL)
    for claim_id, text in texts.items():
        by_premise[f"{text}\n\n{explanations[claim_id]}"] = UNDECIDED

    graph = FakeGraph(pairs, texts, explanations)
    counters = run_g1(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            graph,
            StubScorer(by_premise),
        )
    )

    assert counters["ctxDisagreed"] == 1
    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert queued[0].gate is Gate.G4_ESCALATION
    assert queued[0].tier is QueueTier.LLM_TAIL


# --- SHADOW --------------------------------------------------------------------------------


def test_shadow_mode_writes_only_the_shadow_block(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§9: in SHADOW the ensemble is still the verdict writer."""
    pairs = _pairs(1)
    texts = _texts(pairs)
    graph = FakeGraph(pairs, texts)
    scorer = StubScorer(dict.fromkeys(texts.values(), NEUTRAL))

    run_g1(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            graph,
            scorer,
            judge_mode=JudgeMode.SHADOW,
        )
    )

    key = edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id)
    document = emulator_client.collection(collections["edges"]).document(key).get().to_dict()

    assert document["shadow"]["verdict"] == Verdict.NEUTRAL.value
    assert "relation" not in document, "SHADOW must not assert a verdict on the row"


# --- the purge -----------------------------------------------------------------------------


def test_from_start_purges_gatekeeper_rows_and_spares_the_ensemble(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """D-5: the ensemble's history is not ours to delete."""
    edges = emulator_client.collection(collections["edges"])
    edges.document("survivor").set(
        {"stage3RunId": claimed.stage3_run_id, "method": "ENSEMBLE", "relation": "CORROBORATES"}
    )
    edges.document("stale-gk").set(
        {"stage3RunId": claimed.stage3_run_id, "method": Method.GK_G1_NLI.value}
    )

    pairs = _pairs(1)
    texts = _texts(pairs)
    counters = run_g1(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            FakeGraph(pairs, texts),
            StubScorer(dict.fromkeys(texts.values(), NEUTRAL)),
        )
    )

    assert counters["purgedEdges"] == 1
    assert edges.document("survivor").get().exists
    assert not edges.document("stale-gk").get().exists


# --- crash and resume ----------------------------------------------------------------------


def test_a_crash_mid_batch_resumes_without_losing_or_double_counting_a_pair(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """The recovery contract: re-running a gate re-decides only what it never finished."""
    pairs = _pairs(10)
    texts = _texts(pairs)
    by_premise = dict.fromkeys(texts.values(), NEUTRAL)
    run = store.get(claimed.run_request_id)

    # Batch size is 4 and each batch costs two score_nli calls, so failing on the third
    # call lands the crash inside the second batch.
    crashing = StubScorer(by_premise, fail_after=2)
    with pytest.raises(Neo4jUnavailableError):
        run_g1(_context(emulator_client, gate_config, run, FakeGraph(pairs, texts), crashing))

    decided_before = [
        pair
        for pair in _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
        if pair.gate is None
    ]
    assert len(decided_before) == 4, "the first batch's work must be durable"

    # The resumed attempt is a rescue, not a new run: bump the attempt so the purge does
    # not fire and wipe what the first attempt committed.
    run.gate(Gate.G1_NEUTRAL).attempt = 2
    # Pairs the crashed worker leased are still leased; a real resume waits them out.
    _release_leases(emulator_client, collections, claimed.stage3_run_id)

    counters = run_g1(
        _context(emulator_client, gate_config, run, FakeGraph(pairs, texts), StubScorer(by_premise))
    )

    assert counters["seen"] == 6, "only the undecided remainder is re-scored"
    assert counters.get("purgedEdges") is None, "a resume must not purge"

    queued = _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    assert len(queued) == 10, "no pair was duplicated"
    assert all(pair.gate is None for pair in queued), "every pair ended up decided"

    # One row per pair, upserted by a deterministic key — never one per attempt.
    written = list(emulator_client.collection(collections["edges"]).stream())
    assert len(written) == 10


def _release_leases(emulator_client, collections, stage3_run_id) -> None:
    for snapshot in (
        emulator_client.collection(collections["queue"])
        .where(filter=FieldFilter("stage3RunId", "==", stage3_run_id))
        .stream()
    ):
        snapshot.reference.update({"leaseOwner": None, "leaseExpiresAt": None})


# --- replay parity -------------------------------------------------------------------------


def test_the_gate_loop_agrees_with_the_replay_harness(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """VA-97's harness and the production gate must reach the same verdict (DoD).

    Both call ``decide_g1``, so this is a **wiring** check: it proves the gate loop feeds
    that function the scores it thinks it does — same directions, same order, same
    thresholds off the same ``configSnapshot`` — and routes the answer it got back rather
    than one of its own.
    """
    profiles = [NEUTRAL, REPEATS, CONTRADICTORY, UNDECIDED]
    pairs = _pairs(len(profiles))
    texts = _texts(pairs)
    by_premise = {}
    for pair, scores in zip(pairs, profiles, strict=True):
        by_premise[texts[pair.claim_a_id]] = scores
        by_premise[texts[pair.claim_b_id]] = scores

    run = store.get(claimed.run_request_id)
    thresholds = Thresholds.from_snapshot(run.config_snapshot)

    run_g1(
        _context(emulator_client, gate_config, run, FakeGraph(pairs, texts), StubScorer(by_premise))
    )

    queued = {
        pair.claim_a_id: pair
        for pair in _queue(emulator_client, collections).all_pairs(claimed.stage3_run_id)
    }

    for pair, scores in zip(pairs, profiles, strict=True):
        # What the harness would compute, from the same numbers and the same thresholds.
        expected = decide_g1(G1Scores(forward=scores, backward=scores), thresholds)
        actual = queued[pair.claim_a_id]

        if expected.disposition is Disposition.DECIDED:
            assert actual.decided_by == Method.GK_G1_NLI.value
            assert actual.gate is None
        elif expected.disposition is Disposition.ESCALATE_G4:
            assert actual.gate is Gate.G4_ESCALATION
        else:
            assert actual.gate is Gate.G2_CORROBORATION
            assert actual.decided_by is None
        assert actual.contradiction_flag is expected.contradiction_flag


# --- failure modes -------------------------------------------------------------------------


def test_a_run_without_a_subject_id_fails_the_gate(
    store, claimed, emulator_client, gate_config
) -> None:
    """No subject means no way to find the pairs; guessing is not an option."""
    with pytest.raises(Neo4jUnavailableError, match="subjectId"):
        run_g1(
            _context(
                emulator_client,
                gate_config,
                store.get(claimed.run_request_id),
                FakeGraph([], {}),
                StubScorer({}),
            )
        )


def test_a_queued_pair_with_no_claim_text_fails_rather_than_guessing(
    store, claimed, emulator_client, gate_config, subject
) -> None:
    pairs = _pairs(1)
    graph = FakeGraph(pairs, {})  # the graph has the queue but not the text

    with pytest.raises(Neo4jUnavailableError, match="no claim text"):
        run_g1(
            _context(
                emulator_client,
                gate_config,
                store.get(claimed.run_request_id),
                graph,
                StubScorer({}),
            )
        )
