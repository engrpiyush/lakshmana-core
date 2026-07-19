"""Purge & retrigger semantics — VA-103 (LLD §6 rules 3-4, §10, D-5).

`tests/test_run_state_machine.py` proves the transitions in isolation. This file proves the
three *operator-facing* stories §10 describes, each one end to end against the emulator:

* **FROM_START** twice — a new ``runRequestId`` supersedes its predecessor, and the purge
  that opens it is idempotent under redelivery;
* **FROM_GATE** on a gate that is not broken — refused, in every shape "healthy" takes;
* **redelivery** of a retrigger — the second copy is a no-op, not a second run.

And the invariant underneath all of them: **D-5, the ensemble cache is immutable.** That one
gets the most attention here, because it is the only thing in this codebase that deletes
another service's data, and the corpus it could destroy was paid for per token.
"""

from __future__ import annotations

import uuid

import pytest

from gatekeeper.config import load_config
from gatekeeper.enums import Gate, GateState, JudgeMode, Method, RunMode, RunState
from gatekeeper.errors import ErrorCode
from gatekeeper.gates.g1 import run_g1
from gatekeeper.runs.store import ClaimRejection
from gatekeeper.worker.edges import EdgeWriter, edge_key
from gatekeeper.worker.gates import GateContext
from gatekeeper.worker.queue import PairQueue
from tests.test_g1_gate import NEUTRAL, FakeGraph, StubScorer, _pairs, _texts

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"


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


def _stage3_run(emulator_client, collections, stage3_run_id):
    subject_id = f"subject-{uuid.uuid4().hex[:8]}"
    emulator_client.collection(collections["runs"]).document(stage3_run_id).set(
        {"subjectId": subject_id, "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )
    return subject_id


def _run_g1(store, run_request_id, emulator_client, gate_config, pairs, texts):
    run = store.get(run_request_id)
    run.judge_mode = JudgeMode.GATEKEEPER
    return run_g1(
        GateContext(
            run=run,
            gate=Gate.G1_NEUTRAL,
            config=gate_config,
            client=emulator_client,
            lease_owner=OWNER,
            reader_factory=lambda: FakeGraph(pairs, texts),
            scorer_factory=lambda binding: StubScorer(dict.fromkeys(texts.values(), NEUTRAL)),
        )
    )


# --- FROM_START ------------------------------------------------------------------------------


def test_a_second_from_start_supersedes_the_first_and_purges_its_output(
    store, emulator_client, gate_config, collections, make_request, config_snapshot
) -> None:
    """§10: a new runRequestId is a new run, and G1's first act is the purge.

    The first run's verdicts must not survive into the second — they were produced by a
    calibration the operator has just chosen to replace.
    """
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    _stage3_run(emulator_client, collections, stage3_run_id)
    pairs = _pairs(3)
    texts = _texts(pairs)

    first = make_request(stage3_run_id=stage3_run_id)
    assert store.claim_gate(
        first,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    ).claimed
    _run_g1(store, first.run_request_id, emulator_client, gate_config, pairs, texts)

    edges = emulator_client.collection(collections["edges"])
    assert len(list(edges.stream())) == 3
    assert {doc.to_dict()["gkRunRequestId"] for doc in edges.stream()} == {first.run_request_id}

    # The operator retriggers FROM_START: a brand-new runRequestId for the same stage3 run.
    second = make_request(stage3_run_id=stage3_run_id)
    result = store.claim_gate(
        second,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    assert result.claimed
    assert result.superseded == (first.run_request_id,)
    assert store.get(first.run_request_id).state is RunState.SUPERSEDED

    counters = _run_g1(store, second.run_request_id, emulator_client, gate_config, pairs, texts)

    assert counters["purgedEdges"] == 3
    assert counters["purgedQueue"] == 3
    # Same three pairs, re-judged, now attributed to the run that judged them.
    assert {doc.to_dict()["gkRunRequestId"] for doc in edges.stream()} == {second.run_request_id}


def test_the_purge_is_idempotent_under_redelivery(
    store, emulator_client, gate_config, collections, make_request, config_snapshot
) -> None:
    """A purge that ran once and runs again deletes nothing new (§10).

    Called directly rather than through two G1 executions, because the second execution of a
    *run* is not a purge at all — `attempt > 1` skips it, which is a different guarantee
    (and the one `test_g1_gate.py` pins). This is the purge's own idempotency.
    """
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    _stage3_run(emulator_client, collections, stage3_run_id)
    pairs = _pairs(2)
    texts = _texts(pairs)

    request = make_request(stage3_run_id=stage3_run_id)
    store.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    _run_g1(store, request.run_request_id, emulator_client, gate_config, pairs, texts)

    writer = EdgeWriter(emulator_client, collection=collections["edges"])
    assert writer.purge(stage3_run_id) == 2
    assert writer.purge(stage3_run_id) == 0
    assert writer.purge(stage3_run_id) == 0


def test_resetting_the_queue_twice_leaves_the_same_state(
    store, emulator_client, gate_config, collections, make_request, config_snapshot
) -> None:
    """The purge's queue half, likewise idempotent — that is what makes it redelivery-safe."""
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    _stage3_run(emulator_client, collections, stage3_run_id)
    pairs = _pairs(3)
    texts = _texts(pairs)

    request = make_request(stage3_run_id=stage3_run_id)
    store.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    _run_g1(store, request.run_request_id, emulator_client, gate_config, pairs, texts)

    queue = PairQueue(emulator_client, collection=collections["queue"])
    queue.reset(stage3_run_id)
    once = sorted(
        (pair.pair_id, pair.gate, pair.tier, pair.decided_by)
        for pair in queue.all_pairs(stage3_run_id)
    )
    queue.reset(stage3_run_id)
    twice = sorted(
        (pair.pair_id, pair.gate, pair.tier, pair.decided_by)
        for pair in queue.all_pairs(stage3_run_id)
    )

    assert once == twice
    assert {pair.gate for pair in queue.all_pairs(stage3_run_id)} == {Gate.G1_NEUTRAL}


# --- FROM_GATE on a healthy gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("gate", "rejection", "why"),
    [
        (
            Gate.G2_CORROBORATION,
            ClaimRejection.GATE_NOT_RETRIGGERABLE,
            "a gate that has never run is PENDING, not broken",
        ),
        (
            Gate.G3_CONTRADICTION,
            ClaimRejection.PRIOR_GATE_NOT_SUCCEEDED,
            "further down the chain the ordering guard refuses it first, which is stricter",
        ),
    ],
)
def test_from_gate_on_a_pending_gate_is_refused(
    store, make_request, config_snapshot, gate, rejection, why
) -> None:
    """§10: FROM_GATE is valid only on FAILED, or on RUNNING with an expired lease.

    Two guards refuse a healthy gate, and which one fires depends on how far down the chain
    the operator aimed. Both are asserted rather than collapsed into "not claimed": a change
    that removed the ordering guard would still leave this test green if it only checked the
    boolean, and the ordering guard is the one keeping a retrigger from skipping a gate.
    """
    request = make_request()
    store.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    store.commit_gate(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)

    result = store.claim_gate(
        request.for_gate(gate, mode=RunMode.FROM_GATE),
        lease_owner="operator/retrigger",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    assert not result.claimed, why
    assert result.rejection is rejection, why


def test_from_gate_on_a_gate_a_live_worker_holds_is_refused(
    store, make_request, config_snapshot
) -> None:
    """RUNNING with an *unexpired* lease is a worker doing its job, not a stuck gate.

    Admitting this would put two workers on one gate — the exact thing the lease exists to
    prevent — and the operator's mental model ("it looks stuck") is precisely the case where
    the lease is the authority rather than the impression.
    """
    request = make_request()
    store.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    result = store.claim_gate(
        request.for_gate(Gate.G1_NEUTRAL, mode=RunMode.FROM_GATE),
        lease_owner="operator/retrigger",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    assert not result.claimed
    assert result.rejection is ClaimRejection.GATE_LEASED


def test_from_gate_on_a_lease_expired_gate_is_allowed(
    emulator_client, runs_collection, make_request, config_snapshot
) -> None:
    """The other half of §10's precondition: a crashed worker's gate is retriggerable."""
    from datetime import UTC, datetime, timedelta

    from gatekeeper.runs.store import GatekeeperRunStore

    now = datetime.now(UTC)
    store = GatekeeperRunStore(
        emulator_client, collection=runs_collection, clock=lambda: now, lease_minutes=90
    )
    request = make_request()
    store.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    later = GatekeeperRunStore(
        emulator_client,
        collection=runs_collection,
        clock=lambda: now + timedelta(minutes=91),
        lease_minutes=90,
    )
    result = later.claim_gate(
        request.for_gate(Gate.G1_NEUTRAL, mode=RunMode.FROM_GATE),
        lease_owner="operator/retrigger",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    assert result.claimed
    assert result.run.gate(Gate.G1_NEUTRAL).attempt == 2


def test_a_redelivered_retrigger_does_not_run_twice(store, make_request, config_snapshot) -> None:
    """Pub/Sub is at-least-once, and a retrigger is a message like any other (§6 rule 2)."""
    request = make_request()
    store.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    store.fail_gate(
        request.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_NEO4J_UNAVAILABLE,
        error_detail="graph was down",
    )

    retrigger = request.for_gate(Gate.G1_NEUTRAL, mode=RunMode.FROM_GATE)
    first = store.claim_gate(
        retrigger,
        lease_owner="operator/retrigger-1",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    second = store.claim_gate(
        retrigger,
        lease_owner="operator/retrigger-2",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    assert first.claimed
    assert not second.claimed, "the redelivered copy must not start a second worker"
    assert second.rejection is ClaimRejection.GATE_LEASED
    assert store.get(request.run_request_id).gate(Gate.G1_NEUTRAL).lease_owner == (
        "operator/retrigger-1"
    )


def test_a_superseded_run_refuses_every_retrigger(store, make_request, config_snapshot) -> None:
    """§6 rule 4: superseded runRequestIds refuse all further transitions."""
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    first = make_request(stage3_run_id=stage3_run_id)
    store.claim_gate(
        first,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    store.claim_gate(
        make_request(stage3_run_id=stage3_run_id),
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    for mode in (RunMode.FULL, RunMode.FROM_GATE):
        result = store.claim_gate(
            first.for_gate(Gate.G1_NEUTRAL, mode=mode),
            lease_owner="operator/retrigger",
            judge_mode=JudgeMode.GATEKEEPER,
            config_snapshot=config_snapshot,
        )
        assert not result.claimed, f"{mode.value} revived a superseded run"
        assert result.rejection is ClaimRejection.RUN_SUPERSEDED


# --- D-5: the ensemble cache is immutable -------------------------------------------------------


def test_the_purge_cannot_reach_a_real_ensemble_row(emulator_client, collections) -> None:
    """D-5, in the shape the ensemble actually writes.

    Vishwamitra's ``Stage3EdgeVerdict`` carries ``subjectId``/``claimIdLow``/``claimIdHigh``/
    ``withContext``/``promptStamp``/``relation``/``votes``/``judgeModel`` — and **no
    ``stage3RunId`` and no ``method`` at all**. So the purge's query, which filters on
    ``stage3RunId``, cannot return one even in principle: the field it filters on does not
    exist on those documents.

    That is a stronger guarantee than the ``method`` re-check, and it is the one that holds
    in production. The re-check is the belt for the case below.
    """
    edges = emulator_client.collection(collections["edges"])
    ensemble_id = "claim-a0|claim-b0|bare|judge:v3:9f2c1a"
    ensemble_row = {
        "subjectId": "subject-1",
        "claimIdLow": "claim-a0",
        "claimIdHigh": "claim-b0",
        "withContext": False,
        "promptStamp": "judge:v3:9f2c1a",
        "relation": "CORROBORATES",
        "confidence": 0.83,
        "votes": {"CORROBORATES": 3},
        "rationale": "Paid for, per token, and never to be regenerated.",
        "judgeModel": "gemini-2.5-flash",
    }
    edges.document(ensemble_id).set(ensemble_row)

    deleted = EdgeWriter(emulator_client, collection=collections["edges"]).purge("any-run-id")

    assert deleted == 0
    assert edges.document(ensemble_id).get().to_dict() == ensemble_row


def test_the_purge_spares_a_foreign_row_that_carries_our_run_id(
    emulator_client, collections
) -> None:
    """The belt: a row inside the query's reach still survives if its method is not ours.

    Nothing writes this row today. It is here because the ``method`` re-check is the
    difference between "the query happens not to match" and "deleting someone else's verdict
    is impossible", and only the second of those survives a schema change upstream.
    """
    edges = emulator_client.collection(collections["edges"])
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"

    edges.document("foreign").set(
        {"stage3RunId": stage3_run_id, "method": Method.ENSEMBLE.value, "relation": "REPEATS"}
    )
    edges.document("ours").set(
        {"stage3RunId": stage3_run_id, "method": Method.GK_G1_NLI.value, "relation": "NEUTRAL"}
    )

    deleted = EdgeWriter(emulator_client, collection=collections["edges"]).purge(stage3_run_id)

    assert deleted == 1
    assert edges.document("foreign").get().exists
    assert not edges.document("ours").get().exists


def test_a_gatekeeper_row_and_an_ensemble_row_can_never_collide(emulator_client) -> None:
    """The doc-id shapes are disjoint, so neither judge can overwrite the other's verdict.

    Ours ends in the literal ``GK``; the ensemble's last segment is a real
    ``<impl>:<version>:<hash>`` prompt stamp, which is never that token.
    """
    ours = edge_key("claim-a0", "claim-b0")
    theirs = "claim-a0|claim-b0|bare|judge:v3:9f2c1a"

    assert ours != theirs
    assert ours.endswith("|GK")
    assert not theirs.endswith("|GK")


def test_a_from_start_purge_leaves_the_ensemble_cache_intact_end_to_end(
    store, emulator_client, gate_config, collections, make_request, config_snapshot
) -> None:
    """The invariant through the real gate, not through `EdgeWriter` directly.

    A full G1 run over a collection that already holds ensemble verdicts: the gatekeeper's
    own rows are purged and rewritten, and the paid-for corpus is byte-for-byte untouched.
    """
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    _stage3_run(emulator_client, collections, stage3_run_id)
    pairs = _pairs(2)
    texts = _texts(pairs)

    edges = emulator_client.collection(collections["edges"])
    cache = {
        f"{pair.claim_a_id}|{pair.claim_b_id}|bare|judge:v3:9f2c1a": {
            "subjectId": "subject-1",
            "claimIdLow": pair.claim_a_id,
            "claimIdHigh": pair.claim_b_id,
            "promptStamp": "judge:v3:9f2c1a",
            "relation": "NEUTRAL",
            "votes": {"NEUTRAL": 3},
            "judgeModel": "gemini-2.5-flash",
        }
        for pair in pairs
    }
    for doc_id, row in cache.items():
        edges.document(doc_id).set(row)

    first = make_request(stage3_run_id=stage3_run_id)
    store.claim_gate(
        first,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    _run_g1(store, first.run_request_id, emulator_client, gate_config, pairs, texts)

    second = make_request(stage3_run_id=stage3_run_id)
    store.claim_gate(
        second,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    counters = _run_g1(store, second.run_request_id, emulator_client, gate_config, pairs, texts)

    assert counters["purgedEdges"] == 2, "the first run's own rows were purged"
    for doc_id, row in cache.items():
        assert edges.document(doc_id).get().to_dict() == row, f"{doc_id} was modified"


def test_the_run_doc_of_a_superseded_run_is_never_deleted(
    store, make_request, config_snapshot
) -> None:
    """D-5's other half: ``gatekeeper_runs`` docs are immutable history (§7.1).

    "Which run produced this verdict" has to stay answerable after the run that produced it
    was replaced, so a supersede changes the state and nothing else.
    """
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    first = make_request(stage3_run_id=stage3_run_id)
    store.claim_gate(
        first,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    store.claim_gate(
        make_request(stage3_run_id=stage3_run_id),
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )

    superseded = store.get(first.run_request_id)
    assert superseded is not None
    assert superseded.state is RunState.SUPERSEDED
    assert superseded.gate(Gate.G1_NEUTRAL).state is GateState.RUNNING, "history is preserved"
    assert superseded.config_snapshot, "the calibration it ran under is still readable"
