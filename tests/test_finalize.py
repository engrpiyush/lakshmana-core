"""FINALIZE and the totals reconciliation invariant (LLD §8, §7.1, §7.4).

The invariant under test is one line —

    decidedByGates + escalatedLlm + escalatedHuman == pairsSeen

— and everything here is about the ways it can be broken. A run that reports SUCCEEDED with
a pair stranded at a committed gate tells vishwamitra to consume a verdict set with a hole
in it, and nothing downstream would ever notice.
"""

from __future__ import annotations

import uuid

import pytest

from gatekeeper.enums import GATE_ORDER, Gate, GateState, JudgeMode, Method, QueueTier, RunState
from gatekeeper.worker.finalize import ReconciliationError, finalize, tally
from gatekeeper.worker.queue import PairQueue, QueuedPair

pytestmark = pytest.mark.emulator


def _pair(index: int, **overrides) -> QueuedPair:
    """A queue row in whatever end state a test needs."""
    defaults = {
        "pair_id": f"run|claim-a{index}|claim-b{index}",
        "stage3_run_id": "run",
        "intake_id": "intake",
        "claim_a_id": f"claim-a{index}",
        "claim_b_id": f"claim-b{index}",
        "gate": None,
        "tier": QueueTier.CASCADE,
    }
    return QueuedPair(**(defaults | overrides))


def _decided(index: int, method: Method = Method.GK_G1_NLI) -> QueuedPair:
    return _pair(index, tier=QueueTier.CASCADE, decided_by=method.value)


def _llm(index: int) -> QueuedPair:
    return _pair(index, tier=QueueTier.LLM_TAIL, decided_by=Method.GK_G4_LLM.value)


def _human(index: int) -> QueuedPair:
    return _pair(index, tier=QueueTier.HUMAN, decided_by=None)


# --- the tally -----------------------------------------------------------------------------


def test_the_three_buckets_add_up_to_pairs_seen() -> None:
    counted = tally([_decided(0), _decided(1, Method.GK_G2_GROUNDING), _llm(2), _human(3)])

    assert counted.pairs_seen == 4
    assert counted.decided_by_gates == 2
    assert counted.escalated_llm == 1
    assert counted.escalated_human == 1
    assert counted.stranded == 0
    assert counted.reconciles


def test_a_human_routed_pair_is_counted_by_tier_not_by_method() -> None:
    """G3's contradiction candidates and G4's failures both leave `decidedBy` unset.

    Reading `decidedBy` first would put every one of them in no bucket at all, and the
    invariant would fail on a run where nothing had actually gone wrong.
    """
    counted = tally([_human(0), _human(1)])

    assert counted.escalated_human == 2
    assert counted.decided_by_gates == 0
    assert counted.reconciles


def test_a_shadow_tail_pair_counts_as_escalated_llm() -> None:
    """SHADOW makes no calls, so `decidedBy` is unset — but the pair did reach the tail."""
    counted = tally([_pair(0, tier=QueueTier.LLM_TAIL, decided_by=None)])

    assert counted.escalated_llm == 1
    assert counted.reconciles


def test_a_pair_still_sitting_at_a_gate_is_stranded() -> None:
    counted = tally([_decided(0), _pair(1, gate=Gate.G2_CORROBORATION)])

    assert counted.stranded == 1
    assert not counted.reconciles


def test_a_cleared_pair_no_method_claimed_is_stranded_too() -> None:
    """CASCADE tier, gate cleared, nothing claiming it — nothing produces this state."""
    counted = tally([_pair(0, tier=QueueTier.CASCADE, decided_by=None)])

    assert counted.stranded == 1
    assert not counted.reconciles


def test_an_unknown_method_does_not_count_as_a_gate_decision() -> None:
    """`ENSEMBLE` on a gatekeeper queue row would be someone else's verdict leaking in."""
    counted = tally([_pair(0, tier=QueueTier.CASCADE, decided_by=Method.ENSEMBLE.value)])

    assert counted.decided_by_gates == 0
    assert counted.stranded == 1


# --- finalize against the emulator ----------------------------------------------------------


@pytest.fixture
def queue_collection():
    return f"gatekeeper_pairs_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def finalizable(store, claimed, emulator_client, runs_collection):
    """A run whose four gates have all committed — what FINALIZE always finds.

    The later three gates are marked RUNNING directly rather than driven through a real
    cascade: this file is about the reconciliation, and `tests/test_e2e_cascade.py` is where
    the chain that produces this state is exercised for real.
    """
    document = emulator_client.collection(runs_collection).document(claimed.run_request_id)
    for gate in GATE_ORDER:
        if store.get(claimed.run_request_id).gate(gate).state is not GateState.RUNNING:
            document.update(
                {
                    f"gates.{gate.value}.state": GateState.RUNNING.value,
                    f"gates.{gate.value}.leaseOwner": "worker/test/task-0",
                }
            )
        store.commit_gate(
            claimed.run_request_id,
            gate,
            lease_owner="worker/test/task-0",
            counters={"seen": 1},
        )
    return claimed


def _seed_queue(emulator_client, collection, stage3_run_id, pairs):
    queue = PairQueue(emulator_client, collection=collection)
    for index, pair in enumerate(pairs):
        pair.stage3_run_id = stage3_run_id
        pair.pair_id = f"{stage3_run_id}|claim-a{index}|claim-b{index}"
        emulator_client.collection(collection).document(pair.pair_id).set(pair.to_firestore())
    return queue


def test_finalize_writes_the_totals_and_succeeds_the_run(
    store, finalizable, emulator_client, queue_collection
) -> None:
    queue = _seed_queue(
        emulator_client,
        queue_collection,
        finalizable.stage3_run_id,
        [_decided(0), _decided(1), _llm(2), _human(3)],
    )
    run = store.get(finalizable.run_request_id)

    totals = finalize(store, run, queue, spend_usd=0.0042)

    assert totals.pairs_seen == 4
    assert totals.decided_by_gates == 2
    assert totals.escalated_llm == 1
    assert totals.escalated_human == 1
    assert totals.llm_spend_usd == pytest.approx(0.0042)

    stored = store.get(finalizable.run_request_id)
    assert stored.state is RunState.SUCCEEDED
    assert stored.totals.pairs_seen == 4
    assert stored.totals.llm_spend_usd == pytest.approx(0.0042)


def test_finalize_refuses_a_run_with_a_stranded_pair(
    store, finalizable, emulator_client, queue_collection
) -> None:
    """The one thing FINALIZE must never do is bless an incomplete verdict set."""
    queue = _seed_queue(
        emulator_client,
        queue_collection,
        finalizable.stage3_run_id,
        [_decided(0), _pair(1, gate=Gate.G3_CONTRADICTION)],
    )
    run = store.get(finalizable.run_request_id)

    with pytest.raises(ReconciliationError, match="stranded=1"):
        finalize(store, run, queue)

    assert store.get(finalizable.run_request_id).state is not RunState.SUCCEEDED


def test_shadow_disagreed_is_zero_and_says_so(
    store, finalizable, emulator_client, queue_collection
) -> None:
    """VA-105 owns the disagreement report; finalize would only ever race the ensemble."""
    queue = _seed_queue(
        emulator_client, queue_collection, finalizable.stage3_run_id, [_llm(0), _llm(1)]
    )
    run = store.get(finalizable.run_request_id)
    run.judge_mode = JudgeMode.SHADOW

    assert finalize(store, run, queue).shadow_disagreed == 0


def test_finalize_publishes_nothing(store, finalizable, emulator_client, queue_collection) -> None:
    """Owner requirement #6: vishwamitra polls, there is no callback.

    ``finalize`` takes no publisher at all — the strongest form this assertion can take,
    since it makes "publish something here" a change to the signature rather than a change
    to a branch someone could add without noticing.
    """
    import inspect

    assert "publisher" not in inspect.signature(finalize).parameters


def test_finalizing_twice_is_harmless(
    store, finalizable, emulator_client, queue_collection
) -> None:
    """A redelivered G4 runs FINALIZE again; the second pass must not fail the run."""
    queue = _seed_queue(emulator_client, queue_collection, finalizable.stage3_run_id, [_decided(0)])
    run = store.get(finalizable.run_request_id)

    finalize(store, run, queue)
    # Second pass: the run is SUCCEEDED, so `finalize_run` declines and totals stand.
    totals = finalize(store, store.get(finalizable.run_request_id), queue)

    assert totals.pairs_seen == 1
    assert store.get(finalizable.run_request_id).state is RunState.SUCCEEDED
