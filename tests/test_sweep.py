"""The sweeper against the emulator (LLD §11).

The property that matters most here is **exactly once**. A sweep that rescued a gate twice
would put two messages on the topic for one gate; a sweep that rescued nothing would leave
a crashed run stuck until someone noticed by hand. Both are tested directly, through real
Firestore transactions, because the transaction is the thing doing the work.

The other invariant under test is the one the LLD states twice and owner requirement #7
states again: **the sweeper rescues crashes, never failures.**
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gatekeeper.clients.pubsub import RecordingPublisher
from gatekeeper.dispatcher.sweep import RUN_OVERDUE_HOURS, Sweeper, actionable_gate
from gatekeeper.enums import Gate, GateState, JudgeMode, RunState, TriggeredBy
from gatekeeper.errors import ErrorCode
from gatekeeper.runs.store import GatekeeperRunStore

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"


def _at(offset_minutes: int):
    """A clock pinned ``offset_minutes`` from now — how a lease is aged deterministically."""
    moment = datetime.now(UTC) + timedelta(minutes=offset_minutes)
    return lambda: moment


@pytest.fixture
def stale_run(emulator_client, runs_collection, make_request, config_snapshot):
    """A run whose G1 lease was taken two hours ago and has therefore expired."""
    past = GatekeeperRunStore(
        emulator_client, collection=runs_collection, clock=_at(-120), lease_minutes=90
    )
    request = make_request()
    assert past.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    ).claimed
    return request


def _sweeper(store, publisher=None, **kwargs):
    return Sweeper(store=store, publisher=publisher or RecordingPublisher(), **kwargs)


# -- rescue -------------------------------------------------------------------


def test_an_expired_lease_is_rescued_and_republished(store, stale_run) -> None:
    publisher = RecordingPublisher()

    report = _sweeper(store, publisher).sweep()

    assert report.rescued == [f"{stale_run.run_request_id}:{Gate.G1_NEUTRAL.value}"]
    entry = store.get(stale_run.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.PENDING
    assert entry.sweep_attempt == 1
    assert entry.lease_owner is None

    assert [message.gate for message in publisher.published] == [Gate.G1_NEUTRAL]
    assert publisher.published[0].triggered_by is TriggeredBy.SWEEPER


def test_a_gate_is_rescued_exactly_once(store, stale_run) -> None:
    """Two sweepers racing the same expired lease: one rescues, the other finds nothing."""
    first, second = RecordingPublisher(), RecordingPublisher()

    _sweeper(store, first).sweep()
    second_report = _sweeper(store, second).sweep()

    assert second_report.rescued == []
    assert second.published == []
    assert store.get(stale_run.run_request_id).gate(Gate.G1_NEUTRAL).sweep_attempt == 1


def test_a_live_lease_is_left_alone(store, claimed) -> None:
    """The worker is still inside its 90 minutes; rescuing would race it."""
    report = _sweeper(store).sweep()

    assert report.rescued == []
    assert store.get(claimed.run_request_id).gate(Gate.G1_NEUTRAL).state is GateState.RUNNING


# -- the sweep budget ---------------------------------------------------------


def test_sweeps_exhausted_fails_the_run_with_gk_e_stuck(
    store, stale_run, emulator_client, runs_collection
) -> None:
    for _ in range(3):
        _sweeper(store).sweep()
        # Each rescue returns the gate to PENDING, so re-age it to look crashed again.
        _reclaim_and_expire(emulator_client, runs_collection, stale_run)

    report = _sweeper(store).sweep()

    assert report.stuck == [f"{stale_run.run_request_id}:{Gate.G1_NEUTRAL.value}"]
    run = store.get(stale_run.run_request_id)
    assert run.state is RunState.FAILED
    assert run.gate(Gate.G1_NEUTRAL).error_code is ErrorCode.GK_E_STUCK


def _reclaim_and_expire(emulator_client, runs_collection, request) -> None:
    """Put the gate back into RUNNING with an already-expired lease.

    Stands in for the worker the dispatcher would have started, dying again.
    """
    prefix = f"gates.{Gate.G1_NEUTRAL.value}"
    emulator_client.collection(runs_collection).document(request.run_request_id).update(
        {
            f"{prefix}.state": GateState.RUNNING.value,
            f"{prefix}.leaseOwner": OWNER,
            f"{prefix}.leaseExpiresAt": datetime.now(UTC) - timedelta(minutes=5),
        }
    )


# -- failures are not the sweeper's -------------------------------------------


def test_a_failed_gate_is_never_rescued(store, claimed) -> None:
    """Owner requirement #7: failures wait for an operator, whatever the lease says."""
    store.fail_gate(
        claimed.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_NEO4J_UNAVAILABLE,
        error_detail="graph down",
    )
    publisher = RecordingPublisher()

    report = _sweeper(store, publisher).sweep()

    assert report.rescued == []
    assert report.republished == []
    assert publisher.published == []
    assert store.get(claimed.run_request_id).gate(Gate.G1_NEUTRAL).state is GateState.FAILED


# -- the lost-message pass ----------------------------------------------------


def test_a_run_idle_with_an_unclaimed_gate_is_republished(store, claimed) -> None:
    """The commit-then-publish crash window: G1 succeeded, G2's message never arrived."""
    store.commit_gate(claimed.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER, counters={})
    publisher = RecordingPublisher()

    report = _sweeper(store, publisher, clock=_at(60)).sweep()

    assert report.republished == [f"{claimed.run_request_id}:{Gate.G2_CORROBORATION.value}"]
    assert [message.gate for message in publisher.published] == [Gate.G2_CORROBORATION]


def test_a_run_inside_the_grace_period_is_left_alone(store, claimed) -> None:
    store.commit_gate(claimed.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER, counters={})

    report = _sweeper(store).sweep()

    assert report.republished == []


def test_a_rescued_gate_is_not_also_republished_as_stalled(store, stale_run) -> None:
    """One gate, one message: the rescue already republishes it."""
    publisher = RecordingPublisher()

    _sweeper(store, publisher, clock=_at(60)).sweep()

    assert len(publisher.published) == 1


# -- overdue ------------------------------------------------------------------


def test_a_long_running_run_emits_the_overdue_event(store, claimed) -> None:
    """LK-11's log metric filters on ``event="run_overdue"`` (LLD §12)."""
    report = _sweeper(store, clock=_at(RUN_OVERDUE_HOURS * 60 + 1)).sweep()

    assert claimed.run_request_id in report.overdue


# -- the actionable-gate rule -------------------------------------------------


def test_actionable_gate_is_the_first_pending_gate_whose_predecessor_succeeded(
    store, claimed
) -> None:
    store.commit_gate(claimed.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER, counters={})

    assert actionable_gate(store.get(claimed.run_request_id)) is Gate.G2_CORROBORATION


def test_a_run_with_a_failed_gate_has_no_actionable_gate(store, claimed) -> None:
    store.fail_gate(
        claimed.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_NEO4J_UNAVAILABLE,
        error_detail="graph down",
    )

    assert actionable_gate(store.get(claimed.run_request_id)) is None
