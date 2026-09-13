"""Gate-lease renewal and conditional routing (VA-159 / DEFERRED-LIVE B2, B3).

Two coupled defects. The gate lease was written once at claim and never renewed, so a gate
that outlived its 90-minute lease — which a 40k-pair G1 pass does by 6-7x — looked dead to
the sweeper while it was still running, and a second worker was started on it (B2). And
``route_all`` overwrote pairs unconditionally, so that straggler could strand a pair at a
gate already marked SUCCEEDED, or regress one a later gate had settled (B3).

The renewal heartbeat closes B2 and the conditional route closes B3. These prove both, and
the four behaviours the ticket's acceptance criteria name:

* a worker that loses its lease mid-run aborts without writing — ``test_a_gate_aborts_...``;
* ``route_all`` cannot overwrite a pair that advanced past this gate — ``test_route_all_skips_...``;
* a gate legitimately exceeding the lease is not rescued — ``test_a_renewed_gate_is_not_rescued``
  (with ``test_an_unrenewed_gate_is_rescued`` proving the test has teeth);
* reconciliation holds across a rescue, because the straggler's stale route is refused
  rather than stranding the pair — ``test_route_all_skips_a_pair_that_advanced_past_the_gate``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from gatekeeper.clients.pubsub import RecordingPublisher
from gatekeeper.config import load_config
from gatekeeper.dispatcher.sweep import Sweeper
from gatekeeper.enums import Gate, GateState, JudgeMode, QueueTier, RunState
from gatekeeper.errors import ErrorCode
from gatekeeper.runs.store import GatekeeperRunStore
from gatekeeper.timeutil import utc_now
from gatekeeper.worker.queue import PairQueue, QueuedPair, pair_key

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"


class _Clock:
    """A hand-cranked clock, so a lease's expiry is a fact this test controls."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def __call__(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


# --- renewal, at the store ---------------------------------------------------------------


def test_renew_extends_the_lease_for_the_holder(
    emulator_client, runs_collection, make_request, config_snapshot
) -> None:
    """A renewal pushes ``leaseExpiresAt`` out by the lease length, from the new now."""
    clock = _Clock(utc_now())
    store = GatekeeperRunStore(emulator_client, collection=runs_collection, clock=clock)
    request = make_request()
    assert store.claim_gate(
        request, lease_owner=OWNER, judge_mode=JudgeMode.GATEKEEPER, config_snapshot=config_snapshot
    ).claimed

    before = store.get(request.run_request_id).gate(Gate.G1_NEUTRAL).lease_expires_at
    clock.advance(timedelta(minutes=30))
    assert store.renew_gate_lease(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)

    after = store.get(request.run_request_id).gate(Gate.G1_NEUTRAL).lease_expires_at
    # 90-minute lease claimed at T0, renewed at T0+30, so the new expiry is exactly 30
    # minutes past the old one — the renewal took, and took from the current clock.
    assert after == before + timedelta(minutes=30)


def test_renew_refuses_a_worker_whose_lease_moved(store, claimed) -> None:
    """Only the current holder renews; a stranger's renewal is the swept-worker's stop sign."""
    assert (
        store.renew_gate_lease(claimed.run_request_id, Gate.G1_NEUTRAL, lease_owner="worker/other")
        is False
    )
    # The genuine holder still can — the refusal is about ownership, not a broken gate.
    assert store.renew_gate_lease(claimed.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)


def test_renew_refuses_once_the_run_is_no_longer_running(store, claimed) -> None:
    """A terminal run's straggler learns to stop here too, not only at commit."""
    store.fail_gate(
        claimed.run_request_id,
        Gate.G1_NEUTRAL,
        lease_owner=OWNER,
        error_code=ErrorCode.GK_E_NEO4J_UNAVAILABLE,
        error_detail="boom",
    )
    assert store.get(claimed.run_request_id).state is RunState.FAILED
    assert (
        store.renew_gate_lease(claimed.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER) is False
    )


# --- renewal vs the sweeper --------------------------------------------------------------


def _long_running_gate(emulator_client, runs_collection, make_request, config_snapshot):
    """Claim a gate, then jump the clock past its lease — a gate that has outrun 90 minutes."""
    clock = _Clock(utc_now())
    store = GatekeeperRunStore(emulator_client, collection=runs_collection, clock=clock)
    request = make_request()
    assert store.claim_gate(
        request, lease_owner=OWNER, judge_mode=JudgeMode.GATEKEEPER, config_snapshot=config_snapshot
    ).claimed
    clock.advance(timedelta(minutes=100))
    return store, clock, request


def test_a_renewed_gate_is_not_rescued(
    emulator_client, runs_collection, make_request, config_snapshot
) -> None:
    """Acceptance #3: a live worker renews, so the sweeper does not start a second execution."""
    store, clock, request = _long_running_gate(
        emulator_client, runs_collection, make_request, config_snapshot
    )
    # The live worker's heartbeat, once — enough to push the lease back ahead of the sweeper.
    assert store.renew_gate_lease(request.run_request_id, Gate.G1_NEUTRAL, lease_owner=OWNER)

    report = Sweeper(
        store=store, publisher=RecordingPublisher(), max_sweeps=3, grace_minutes=30, clock=clock
    ).sweep()

    assert report.rescued == []
    assert report.stuck == []
    entry = store.get(request.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.RUNNING
    assert entry.sweep_attempt == 0


def test_an_unrenewed_gate_is_rescued(
    emulator_client, runs_collection, make_request, config_snapshot
) -> None:
    """The teeth behind the test above: with no renewal the same clock triggers a rescue.

    This is the B2 pre-fix behaviour, now reached only by a genuinely dead worker — which is
    exactly what the sweeper is for.
    """
    store, clock, request = _long_running_gate(
        emulator_client, runs_collection, make_request, config_snapshot
    )
    report = Sweeper(
        store=store, publisher=RecordingPublisher(), max_sweeps=3, grace_minutes=30, clock=clock
    ).sweep()

    assert report.rescued == [f"{request.run_request_id}:{Gate.G1_NEUTRAL.value}"]
    assert store.get(request.run_request_id).gate(Gate.G1_NEUTRAL).state is GateState.PENDING


# --- conditional routing -----------------------------------------------------------------


def _queue(emulator_client) -> tuple[PairQueue, str, str]:
    collection = f"gatekeeper_pairs_test_{uuid.uuid4().hex[:12]}"
    stage3 = f"s3-{uuid.uuid4().hex[:8]}"
    return PairQueue(emulator_client, collection=collection), collection, stage3


def _write(emulator_client, collection: str, pair: QueuedPair) -> None:
    emulator_client.collection(collection).document(pair.pair_id).set(pair.to_firestore())


def test_route_all_skips_a_pair_that_advanced_past_the_gate(emulator_client) -> None:
    """Acceptance #2: a straggler cannot move a pair a later gate already advanced.

    A rescuer took the pair from G1 to G2 while the straggler was still scoring; the
    straggler's late route must be refused, or the pair would be dragged back and — in the
    case that bites — stranded at a gate that has since been marked SUCCEEDED.
    """
    queue, collection, stage3 = _queue(emulator_client)
    key = pair_key(stage3, "a", "b")
    # The row as the rescuer left it: advanced to G2, lease cleared.
    _write(
        emulator_client,
        collection,
        QueuedPair(
            pair_id=key,
            stage3_run_id=stage3,
            intake_id="i",
            claim_a_id="a",
            claim_b_id="b",
            gate=Gate.G2_CORROBORATION,
        ),
    )
    # The straggler still believes it holds this pair at G1.
    stale = QueuedPair(
        pair_id=key,
        stage3_run_id=stage3,
        intake_id="i",
        claim_a_id="a",
        claim_b_id="b",
        gate=Gate.G1_NEUTRAL,
        lease_owner="straggler",
    )

    written = queue.route_all(
        [(stale, {"gate": None, "tier": QueueTier.CASCADE.value, "decidedBy": "GK_G1_NLI"})]
    )

    assert written == 0
    assert queue.all_pairs(stage3)[0].gate is Gate.G2_CORROBORATION


def test_route_all_skips_a_pair_whose_lease_changed_hands(emulator_client) -> None:
    """Same gate, different owner: the rescuer holds the lease, so the straggler is refused."""
    queue, collection, stage3 = _queue(emulator_client)
    key = pair_key(stage3, "a", "b")
    _write(
        emulator_client,
        collection,
        QueuedPair(
            pair_id=key,
            stage3_run_id=stage3,
            intake_id="i",
            claim_a_id="a",
            claim_b_id="b",
            gate=Gate.G1_NEUTRAL,
            lease_owner="rescuer",
            lease_expires_at=utc_now() + timedelta(minutes=15),
        ),
    )
    stale = QueuedPair(
        pair_id=key,
        stage3_run_id=stage3,
        intake_id="i",
        claim_a_id="a",
        claim_b_id="b",
        gate=Gate.G1_NEUTRAL,
        lease_owner="straggler",
    )

    written = queue.route_all(
        [(stale, {"gate": Gate.G2_CORROBORATION.value, "tier": QueueTier.CASCADE.value})]
    )

    assert written == 0
    assert queue.all_pairs(stage3)[0].lease_owner == "rescuer"


def test_route_all_moves_a_pair_the_worker_still_holds(emulator_client) -> None:
    """The happy path is unchanged: the lease holder routes onward and the lease is cleared."""
    queue, collection, stage3 = _queue(emulator_client)
    key = pair_key(stage3, "a", "b")
    _write(
        emulator_client,
        collection,
        QueuedPair(
            pair_id=key,
            stage3_run_id=stage3,
            intake_id="i",
            claim_a_id="a",
            claim_b_id="b",
            gate=Gate.G1_NEUTRAL,
            lease_owner=OWNER,
            lease_expires_at=utc_now() + timedelta(minutes=15),
        ),
    )
    held = QueuedPair(
        pair_id=key,
        stage3_run_id=stage3,
        intake_id="i",
        claim_a_id="a",
        claim_b_id="b",
        gate=Gate.G1_NEUTRAL,
        lease_owner=OWNER,
    )

    written = queue.route_all(
        [(held, {"gate": Gate.G2_CORROBORATION.value, "tier": QueueTier.CASCADE.value})]
    )

    assert written == 1
    moved = queue.all_pairs(stage3)[0]
    assert moved.gate is Gate.G2_CORROBORATION
    assert moved.lease_owner is None


# --- the gate loop aborts on a revoked lease ---------------------------------------------


def test_a_gate_aborts_when_its_lease_is_revoked_mid_run(store, claimed, emulator_client) -> None:
    """Acceptance #1: a worker that has lost its lease writes no verdicts and moves no pairs.

    Driven through the real G1 gate with its lease guard forced to report the lease gone.
    The gate does its setup (purge + seed) and then, finding the lease revoked, stops before
    scoring a single batch: no edge rows, and every pair still sitting at G1 for the rescuer.
    """
    from gatekeeper.gates.g1 import run_g1
    from gatekeeper.worker.gates import GateContext
    from tests.test_g1_gate import NEUTRAL, FakeGraph, StubScorer, _pairs, _texts

    suffix = uuid.uuid4().hex[:12]
    collections = {
        "queue": f"gatekeeper_pairs_test_{suffix}",
        "edges": f"stage3_edges_test_{suffix}",
        "runs": f"stage3_runs_test_{suffix}",
    }
    config = load_config(
        {
            "GATEKEEPER_QUEUE_COLLECTION": collections["queue"],
            "GATEKEEPER_STAGE3_EDGES_COLLECTION": collections["edges"],
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": collections["runs"],
            "GATEKEEPER_QUEUE_BATCH_SIZE": "4",
        }
    )
    emulator_client.collection(collections["runs"]).document(claimed.stage3_run_id).set(
        {"subjectId": f"subject-{suffix}", "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )

    pairs = _pairs(3)
    texts = _texts(pairs)
    by_premise = {}
    for pair in pairs:
        by_premise[texts[pair.claim_a_id]] = NEUTRAL
        by_premise[texts[pair.claim_b_id]] = NEUTRAL

    context = GateContext(
        run=store.get(claimed.run_request_id),
        gate=Gate.G1_NEUTRAL,
        config=config,
        client=emulator_client,
        lease_owner=OWNER,
        reader_factory=lambda: FakeGraph(pairs, texts),
        scorer_factory=lambda binding: StubScorer(by_premise),
        lease_guard=lambda: False,  # the lease was revoked before the first batch
    )

    counters = run_g1(context)

    # Setup ran (the queue was seeded), but the loop never claimed a batch.
    assert counters["seeded"] == 3
    assert counters["seen"] == 0
    # Nothing was judged and nothing advanced: no edge rows, every pair still at G1.
    assert list(emulator_client.collection(collections["edges"]).stream()) == []
    queued = PairQueue(emulator_client, collection=collections["queue"]).all_pairs(
        claimed.stage3_run_id
    )
    assert queued and all(pair.gate is Gate.G1_NEUTRAL for pair in queued)


def test_run_gate_wires_the_store_renewal_and_a_false_renewal_aborts(
    store, claimed, emulator_client, monkeypatch
) -> None:
    """The wiring itself is load-bearing (VA-159/B2), so it gets its own test.

    ``run_gate`` is the only thing that connects a real gate's ``renew_lease()`` to
    ``store.renew_gate_lease`` — ``GateContext`` defaults to always-True when unwired, and
    every other renewal test either injects its own guard (bypassing ``run_gate``) or runs a
    gate that always holds its lease, so a deleted wiring line would leave the whole suite
    green while a multi-hour production gate quietly stopped renewing and got swept. This
    drives the real G1 through ``run_gate`` with the store's renewal spied and forced False:
    the spy proves the wiring exists, and the absence of writes proves the False propagated
    to an abort. Delete the wiring and both halves fail.
    """
    from gatekeeper.gates.g1 import run_g1  # noqa: F401  (registers the real G1 runner)
    from gatekeeper.worker.gates import GateContext
    from gatekeeper.worker.main import run_gate
    from tests.test_g1_gate import NEUTRAL, FakeGraph, StubScorer, _pairs, _texts

    suffix = uuid.uuid4().hex[:12]
    collections = {
        "queue": f"gatekeeper_pairs_test_{suffix}",
        "edges": f"stage3_edges_test_{suffix}",
        "runs": f"stage3_runs_test_{suffix}",
    }
    config = load_config(
        {
            "GATEKEEPER_QUEUE_COLLECTION": collections["queue"],
            "GATEKEEPER_STAGE3_EDGES_COLLECTION": collections["edges"],
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": collections["runs"],
            "GATEKEEPER_QUEUE_BATCH_SIZE": "4",
        }
    )
    emulator_client.collection(collections["runs"]).document(claimed.stage3_run_id).set(
        {"subjectId": f"subject-{suffix}", "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )

    pairs = _pairs(3)
    texts = _texts(pairs)
    by_premise = {}
    for pair in pairs:
        by_premise[texts[pair.claim_a_id]] = NEUTRAL
        by_premise[texts[pair.claim_b_id]] = NEUTRAL

    # The sweeper moved the lease mid-run: the store reports the renewal refused.
    calls: list[tuple] = []

    def _refused(*args, **kwargs) -> bool:
        calls.append((args, kwargs))
        return False

    monkeypatch.setattr(store, "renew_gate_lease", _refused)

    run_gate(
        store,
        claimed.run_request_id,
        Gate.G1_NEUTRAL,
        OWNER,
        context_factory=lambda run: GateContext(
            run=run,
            gate=Gate.G1_NEUTRAL,
            config=config,
            client=emulator_client,
            lease_owner=OWNER,
            reader_factory=lambda: FakeGraph(pairs, texts),
            scorer_factory=lambda binding: StubScorer(by_premise),
        ),
        publisher=None,
    )

    # run_gate wired the store's renewal onto the context, and the real G1 invoked it.
    assert calls, "run_gate did not wire store.renew_gate_lease onto the gate context"
    assert calls[0][1] == {"lease_owner": OWNER}
    # And the False renewal aborted G1 before it scored or moved anything.
    assert list(emulator_client.collection(collections["edges"]).stream()) == []
    queued = PairQueue(emulator_client, collection=collections["queue"]).all_pairs(
        claimed.stage3_run_id
    )
    assert queued and all(pair.gate is Gate.G1_NEUTRAL for pair in queued)
