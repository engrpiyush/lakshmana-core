"""The run state machine against the real emulator (LLD §6).

These are the tests VA-95 exists for. Pub/Sub gives us duplicates, reordering and
redelivery, and the claim transaction is the only thing standing between that and a
double-judged intake — so it is proven here against real Firestore transaction
semantics, not a mock that would happily agree with whatever the code does.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from gatekeeper.enums import Gate, GateState, JudgeMode, RunMode, RunState
from gatekeeper.errors import ErrorCode, FirestoreTxnError
from gatekeeper.runs.model import RunTotals
from gatekeeper.runs.store import ClaimRejection, GatekeeperRunStore

pytestmark = pytest.mark.emulator

OWNER = "dispatcher/test/owner-a"
OTHER_OWNER = "dispatcher/test/owner-b"


def _retrigger(request, gate=None):
    """The operator's FROM_GATE re-entry for one gate, same runRequestId."""
    return replace(request, gate=gate or request.gate, mode=RunMode.FROM_GATE)


def _claim(store, request, *, owner=OWNER, snapshot=None):
    return store.claim_gate(
        request,
        lease_owner=owner,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=snapshot if snapshot is not None else {},
    )


# -- creation and the frozen snapshot -----------------------------------------


def test_first_claim_creates_the_run_and_freezes_config(store, make_request, config_snapshot):
    request = make_request()

    result = _claim(store, request, snapshot=config_snapshot)

    assert result.claimed
    run = store.get(request.run_request_id)
    assert run is not None
    assert run.state is RunState.RUNNING
    assert run.config_snapshot == config_snapshot
    assert run.judge_mode is JudgeMode.GATEKEEPER
    assert run.gate(Gate.G1_NEUTRAL).state is GateState.RUNNING
    assert run.gate(Gate.G1_NEUTRAL).attempt == 1
    assert run.gate(Gate.G2_CORROBORATION).state is GateState.PENDING


def test_claim_result_reports_the_post_claim_attempt(store, make_request):
    request = make_request()

    result = _claim(store, request)

    assert result.run is not None
    assert result.run.gate(Gate.G1_NEUTRAL).attempt == 1


def test_config_edits_do_not_leak_into_an_in_flight_run(store, make_request, config_snapshot):
    """A mid-run config change must not produce a mixed-calibration run (LLD §5)."""
    request = make_request()
    assert _claim(store, request, snapshot=config_snapshot).claimed
    assert store.commit_gate(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)

    edited = config_snapshot | {"rosterVersion": "v2-EDITED-MID-RUN"}
    assert _claim(store, request.for_gate(Gate.G2_CORROBORATION), snapshot=edited).claimed

    run = store.get(request.run_request_id)
    assert run.config_snapshot["rosterVersion"] == config_snapshot["rosterVersion"]


def test_lease_expiry_is_ninety_minutes_by_default(store, make_request):
    request = make_request()
    before = datetime.now(UTC)

    _claim(store, request)

    lease = store.get(request.run_request_id).gate(Gate.G1_NEUTRAL).lease_expires_at
    assert timedelta(minutes=89) < lease - before < timedelta(minutes=91)


# -- duplicates ---------------------------------------------------------------


def test_duplicate_delivery_is_a_no_op(store, make_request):
    """At-least-once delivery: the second copy must change nothing."""
    request = make_request()
    assert _claim(store, request).claimed

    duplicate = _claim(store, request, owner=OTHER_OWNER)

    assert not duplicate.claimed
    assert duplicate.rejection is ClaimRejection.GATE_LEASED
    entry = store.get(request.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.attempt == 1
    assert entry.lease_owner == OWNER


def test_redelivery_after_commit_is_a_no_op(store, make_request):
    request = make_request()
    assert _claim(store, request).claimed
    assert store.commit_gate(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)

    redelivered = _claim(store, request)

    assert not redelivered.claimed
    assert redelivered.rejection is ClaimRejection.GATE_TERMINAL
    assert store.get(request.run_request_id).gate(Gate.G1_NEUTRAL).attempt == 1


# -- stale / out-of-order messages --------------------------------------------


def test_gate_message_before_its_predecessor_succeeds_is_dropped(store, make_request):
    """No ordering guarantees: a G2 message may arrive before G1 has committed."""
    request = make_request()
    assert _claim(store, request).claimed  # G1 running, not committed

    early = _claim(store, request.for_gate(Gate.G2_CORROBORATION))

    assert not early.claimed
    assert early.rejection is ClaimRejection.PRIOR_GATE_NOT_SUCCEEDED
    assert store.get(request.run_request_id).gate(Gate.G2_CORROBORATION).state is GateState.PENDING


def test_chained_message_for_an_unknown_run_is_dropped(store, make_request):
    """A non-G1 message with no run doc has lost its predecessor; it cannot open one."""
    orphan = make_request(gate=Gate.G2_CORROBORATION)

    result = _claim(store, orphan)

    assert not result.claimed
    assert result.rejection is ClaimRejection.INVALID_ENTRY
    assert store.get(orphan.run_request_id) is None


def test_from_gate_message_for_an_unknown_run_is_dropped(store, make_request):
    orphan = make_request(gate=Gate.G1_NEUTRAL, mode=RunMode.FROM_GATE)

    result = _claim(store, orphan)

    assert not result.claimed
    assert result.rejection is ClaimRejection.INVALID_ENTRY


def test_messages_for_a_superseded_run_are_refused(store, make_request):
    """Superseded runRequestIds refuse all further transitions (LLD §6 rule 4)."""
    first = make_request()
    assert _claim(store, first).claimed
    second = make_request(stage3_run_id=first.stage3_run_id)
    assert _claim(store, second).claimed

    late = _claim(store, first)

    assert not late.claimed
    assert late.rejection is ClaimRejection.RUN_SUPERSEDED


def test_messages_for_a_succeeded_run_are_refused(store, make_request):
    request = make_request()
    _drive_to_success(store, request)

    replay = _claim(store, request.for_gate(Gate.G1_NEUTRAL))

    assert not replay.claimed
    assert replay.rejection is ClaimRejection.RUN_TERMINAL


# -- concurrency --------------------------------------------------------------


def test_concurrent_claims_produce_exactly_one_winner(
    emulator_client, runs_collection, make_request
):
    """Two dispatcher instances, one message, one worker job.

    A loser can lose in either of two ways: a clean rejection, or an exhausted retry
    budget surfacing as ``GK_E_FIRESTORE_TXN`` (see ``_commit`` — the dispatcher turns
    that into a redelivery). Both mean "did not claim", so both are tolerated here; the
    property under test is that exactly one caller was handed the gate.
    """
    request = make_request()
    stores = [GatekeeperRunStore(emulator_client, collection=runs_collection) for _ in range(6)]

    def _attempt(pair):
        index, store = pair
        try:
            return _claim(store, request, owner=f"owner-{index}")
        except FirestoreTxnError:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_attempt, enumerate(stores)))

    assert sum(1 for result in results if result is not None and result.claimed) == 1
    entry = stores[0].get(request.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.attempt == 1


def test_concurrent_from_start_publishes_leave_one_live_run(
    emulator_client, runs_collection, make_request
):
    """Racing FROM_START runs must never leave two live runs on one stage3RunId.

    Under heavy contention some claims exhaust their retries and raise
    ``GK_E_FIRESTORE_TXN``. That is an acceptable outcome — nothing was written, the
    operator sees the error and retriggers — so the property under test is the safety
    one: at most one RUNNING, and every doc that *was* created is RUNNING or SUPERSEDED.
    """
    stage3_run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    requests = [make_request(stage3_run_id=stage3_run_id) for _ in range(4)]
    store = GatekeeperRunStore(emulator_client, collection=runs_collection)

    def _attempt(request):
        try:
            return _claim(GatekeeperRunStore(emulator_client, collection=runs_collection), request)
        except FirestoreTxnError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(_attempt, requests))

    assert any(outcome is not None and outcome.claimed for outcome in outcomes), (
        "every concurrent claim lost to contention; at least one must win"
    )
    states = [
        run.state
        for run in (store.get(request.run_request_id) for request in requests)
        if run is not None
    ]
    assert states.count(RunState.RUNNING) == 1
    assert set(states) <= {RunState.RUNNING, RunState.SUPERSEDED}


# -- supersede ----------------------------------------------------------------


def test_from_start_supersedes_the_predecessor_atomically(store, make_request):
    first = make_request()
    assert _claim(store, first).claimed

    second = make_request(stage3_run_id=first.stage3_run_id)
    result = _claim(store, second)

    assert result.claimed
    assert result.superseded == (first.run_request_id,)
    predecessor = store.get(first.run_request_id)
    assert predecessor.state is RunState.SUPERSEDED
    assert predecessor.superseded_by == second.run_request_id
    assert store.get(second.run_request_id).state is RunState.RUNNING


def test_supersede_only_touches_the_same_stage3_run(store, make_request):
    unrelated = make_request()
    assert _claim(store, unrelated).claimed

    fresh = make_request()
    assert _claim(store, fresh).claimed

    assert store.get(unrelated.run_request_id).state is RunState.RUNNING


def test_a_failed_predecessor_keeps_its_failed_state(store, make_request):
    """FAILED is the audit record of what went wrong; the diagram never overwrites it."""
    first = make_request()
    assert _claim(store, first).claimed
    assert store.fail_gate(
        first.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_NEO4J_UNAVAILABLE,
        error_detail="hydrate budget exhausted",
    )

    second = make_request(stage3_run_id=first.stage3_run_id)
    assert _claim(store, second).claimed

    assert store.get(first.run_request_id).state is RunState.FAILED


# -- commit, fail, and the lease ----------------------------------------------


def test_commit_records_counters_and_clears_the_lease(store, make_request):
    request = make_request()
    assert _claim(store, request).claimed

    assert store.commit_gate(
        request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER, counters={"seen": 12208}
    )

    entry = store.get(request.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.SUCCEEDED
    assert entry.counters == {"seen": 12208}
    assert entry.lease_owner is None
    assert entry.ended_at is not None


def test_a_worker_that_lost_its_lease_cannot_commit(store, make_request):
    """The sweeper may have rescued this gate; the stale worker must not overwrite it."""
    request = make_request()
    assert _claim(store, request).claimed

    assert not store.commit_gate(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OTHER_OWNER)
    assert store.get(request.run_request_id).gate(Gate.G1_NEUTRAL).state is GateState.RUNNING


def test_failure_marks_the_gate_and_the_run(store, make_request):
    request = make_request()
    assert _claim(store, request).claimed

    assert store.fail_gate(
        request.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_MODEL_FETCH,
        error_detail="sha256 mismatch on modernbert-base-nli@v1",
    )

    run = store.get(request.run_request_id)
    assert run.state is RunState.FAILED
    entry = run.gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.FAILED
    assert entry.error_code is ErrorCode.GK_E_MODEL_FETCH
    assert "sha256" in entry.error_detail


def test_a_redelivery_cannot_restart_failed_work(store, make_request):
    """The sweeper rescues crashes, never failures — only the operator retriggers."""
    request = make_request()
    assert _claim(store, request).claimed
    assert store.fail_gate(
        request.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_NEO4J_UNAVAILABLE,
        error_detail="neo4j down",
    )

    redelivered = _claim(store, request)

    assert not redelivered.claimed
    assert redelivered.rejection is ClaimRejection.RUN_TERMINAL


# -- retrigger ----------------------------------------------------------------


def test_from_gate_revives_a_failed_gate_and_bumps_the_attempt(store, make_request):
    request = make_request()
    assert _claim(store, request).claimed
    assert store.fail_gate(
        request.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_NEO4J_UNAVAILABLE,
        error_detail="neo4j down",
    )

    retrigger = _claim(store, _retrigger(request, Gate.G1_NEUTRAL))

    assert retrigger.claimed
    run = store.get(request.run_request_id)
    assert run.state is RunState.RUNNING
    entry = run.gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.RUNNING
    assert entry.attempt == 2
    assert entry.error_code is None


def test_from_gate_retains_earlier_gates(store, make_request):
    request = make_request()
    assert _claim(store, request).claimed
    assert store.commit_gate(
        request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER, counters={"seen": 12208}
    )
    g2 = request.for_gate(Gate.G2_CORROBORATION)
    assert _claim(store, g2).claimed
    assert store.fail_gate(
        request.run_request_id,
        Gate.G2_CORROBORATION,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_MODEL_FETCH,
        error_detail="artifact missing",
    )

    assert _claim(store, _retrigger(g2)).claimed

    run = store.get(request.run_request_id)
    assert run.gate(Gate.G1_NEUTRAL).state is GateState.SUCCEEDED
    assert run.gate(Gate.G1_NEUTRAL).counters == {"seen": 12208}


def test_from_gate_on_a_succeeded_gate_is_refused(store, make_request):
    request = make_request()
    assert _claim(store, request).claimed
    assert store.commit_gate(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)

    result = _claim(store, _retrigger(request))

    assert not result.claimed
    assert result.rejection is ClaimRejection.GATE_TERMINAL


# -- expired leases (the sweeper's rescue path) -------------------------------


def test_an_expired_lease_can_be_reclaimed(emulator_client, runs_collection, make_request):
    """A crashed worker's gate is claimable again once its lease lapses (LLD §11)."""
    request = make_request()
    now = datetime(2026, 7, 19, 4, 0, tzinfo=UTC)
    clock = {"now": now}
    store = GatekeeperRunStore(
        emulator_client, collection=runs_collection, clock=lambda: clock["now"]
    )
    assert _claim(store, request).claimed

    clock["now"] = now + timedelta(minutes=91)
    rescued = _claim(store, request, owner=OTHER_OWNER)

    assert rescued.claimed
    entry = store.get(request.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.attempt == 2
    assert entry.lease_owner == OTHER_OWNER


def test_an_unexpired_lease_blocks_reclaim(emulator_client, runs_collection, make_request):
    request = make_request()
    now = datetime(2026, 7, 19, 4, 0, tzinfo=UTC)
    clock = {"now": now}
    store = GatekeeperRunStore(
        emulator_client, collection=runs_collection, clock=lambda: clock["now"]
    )
    assert _claim(store, request).claimed

    clock["now"] = now + timedelta(minutes=89)

    assert not _claim(store, request, owner=OTHER_OWNER).claimed


# -- finalize -----------------------------------------------------------------


def _drive_to_success(store, request) -> None:
    for gate in Gate:
        assert _claim(store, request.for_gate(gate)).claimed
        assert store.commit_gate(request.run_request_id, gate, lease_owner=OWNER)
    assert store.finalize_run(request.run_request_id, RunTotals(pairs_seen=12208))


def test_finalize_sets_totals_and_succeeds_the_run(store, make_request):
    request = make_request()

    _drive_to_success(store, request)

    run = store.get(request.run_request_id)
    assert run.state is RunState.SUCCEEDED
    assert run.totals.pairs_seen == 12208


def test_finalize_is_refused_while_a_gate_is_unsettled(store, make_request):
    request = make_request()
    assert _claim(store, request).claimed
    assert store.commit_gate(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)

    assert not store.finalize_run(request.run_request_id, RunTotals())
    assert store.get(request.run_request_id).state is RunState.RUNNING


def test_finalize_is_idempotent(store, make_request):
    request = make_request()
    _drive_to_success(store, request)

    assert not store.finalize_run(request.run_request_id, RunTotals(pairs_seen=999))
    assert store.get(request.run_request_id).totals.pairs_seen == 12208


def test_get_returns_none_for_an_unknown_run(store):
    assert store.get(str(uuid.uuid4())) is None


def test_run_doc_id_is_the_run_request_id(store, make_request, runs_collection, emulator_client):
    request = make_request()
    _claim(store, request)

    snapshot = emulator_client.collection(runs_collection).document(request.run_request_id).get()

    assert snapshot.exists
    assert snapshot.to_dict()["runRequestId"] == request.run_request_id
    # createdAt is a resolved server timestamp, not the sentinel we wrote.
    assert isinstance(snapshot.to_dict()["createdAt"], datetime)
