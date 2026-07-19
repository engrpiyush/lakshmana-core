"""The worker entrypoint against the emulator (LLD §4, §8).

This is the DoD's "worker runs a no-op gate against the emulator": claim, execute,
commit, all through real Firestore transactions. The gates themselves arrive on later
tickets, but the chassis they plug into is proven here.
"""

from __future__ import annotations

import pytest

from gatekeeper.clients.pubsub import RecordingPublisher
from gatekeeper.config import load_config
from gatekeeper.enums import Gate, GateState, JudgeMode, RunState
from gatekeeper.errors import ErrorCode, Neo4jUnavailableError
from gatekeeper.worker import main as main_module
from gatekeeper.worker.gates import GateContext, no_op_gate
from gatekeeper.worker.main import EXIT_BAD_INVOCATION, EXIT_GATE_FAILED, EXIT_OK, run_gate

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"


def _context_factory(client, gate: Gate, lease_owner: str = OWNER):
    """A context with no graph and no models — enough for the no-op gate chassis."""

    def _build(run):
        return GateContext(
            run=run,
            gate=gate,
            config=load_config({}),
            client=client,
            lease_owner=lease_owner,
        )

    return _build


def _run(store, emulator_client, run_request_id, gate, owner=OWNER, publisher=None):
    return run_gate(
        store,
        run_request_id,
        gate,
        owner,
        context_factory=_context_factory(emulator_client, gate, owner),
        publisher=publisher,
    )


def test_every_implemented_gate_resolves_to_its_own_runner() -> None:
    """The registry must not resolve an implemented gate to the no-op (VA-100/VA-101).

    Regression test for a lazy-import guard that read "any gate registered" as "all gates
    loaded". Because each gate module registers itself on import, importing one directly —
    which the replay harness and half these tests do — was enough to make every *other*
    gate resolve to `no_op_gate`. That failure is invisible by construction: the no-op
    commits successfully, so a full cascade would run G1, commit zeroes for G2 and G3, and
    report SUCCEEDED having skipped two thirds of the judging.

    The import below reproduces the trigger; the assertion is that it no longer matters.
    """
    from gatekeeper.enums import GATE_ORDER
    from gatekeeper.gates import g1  # noqa: F401  (the import that used to poison the load)
    from gatekeeper.worker.gates import runner_for

    # All four, as of VA-102 — the cascade has no unimplemented gate left, so the no-op is
    # now a runner nothing should ever resolve to.
    for gate in GATE_ORDER:
        assert runner_for(gate) is not no_op_gate, f"{gate.value} resolved to the no-op gate"


def test_the_no_op_gate_commits_against_the_emulator(
    store, claimed, emulator_client, monkeypatch
) -> None:
    """The chassis: claim → execute → commit, through real Firestore transactions.

    Driven through the no-op rather than the real G1, which needs a graph and a model of
    its own; `tests/test_g1_gate.py` exercises that end to end.
    """
    monkeypatch.setattr(main_module, "runner_for", lambda gate: no_op_gate)

    exit_code = _run(store, emulator_client, claimed.run_request_id, Gate.G1_NEUTRAL)

    assert exit_code == EXIT_OK
    entry = store.get(claimed.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.SUCCEEDED
    assert entry.counters["noOp"] is True
    assert entry.lease_owner is None


def test_the_chain_publishes_the_next_gate(store, claimed, emulator_client, monkeypatch) -> None:
    """Commit, then publish — the worker drives its own successor (LLD §5)."""
    monkeypatch.setattr(main_module, "runner_for", lambda gate: lambda context: {"seen": 0})
    publisher = RecordingPublisher()

    exit_code = _run(
        store, emulator_client, claimed.run_request_id, Gate.G1_NEUTRAL, publisher=publisher
    )

    assert exit_code == EXIT_OK
    assert [message.gate for message in publisher.published] == [Gate.G2_CORROBORATION]
    # Same run, always: the chain is one run and the transactional claim on that id is
    # what keeps redeliveries harmless.
    assert publisher.published[0].run_request_id == claimed.run_request_id


def test_the_last_gate_publishes_nothing(store, claimed, emulator_client, monkeypatch) -> None:
    """After G4 comes FINALIZE, which is internal (LLD §8)."""
    monkeypatch.setattr(main_module, "runner_for", lambda gate: lambda context: {"seen": 0})
    publisher = RecordingPublisher()

    for gate in Gate:
        if gate is not Gate.G1_NEUTRAL:
            store.claim_gate(
                claimed.for_gate(gate),
                lease_owner=OWNER,
                judge_mode=JudgeMode.GATEKEEPER,
                config_snapshot={},
            )
        _run(store, emulator_client, claimed.run_request_id, gate, publisher=publisher)

    assert [message.gate for message in publisher.published] == [
        Gate.G2_CORROBORATION,
        Gate.G3_CONTRADICTION,
        Gate.G4_ESCALATION,
    ]


def test_a_gate_failure_is_recorded_with_its_error_code(
    store, claimed, emulator_client, monkeypatch
) -> None:
    def _explode(context):
        raise Neo4jUnavailableError("hydrate budget exhausted")

    # main imports runner_for by value, so the patch has to land on its binding.
    monkeypatch.setattr(main_module, "runner_for", lambda gate: _explode)

    exit_code = _run(store, emulator_client, claimed.run_request_id, Gate.G1_NEUTRAL)

    assert exit_code == EXIT_GATE_FAILED
    run = store.get(claimed.run_request_id)
    assert run.state is RunState.FAILED
    entry = run.gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.FAILED
    assert entry.error_code is ErrorCode.GK_E_NEO4J_UNAVAILABLE
    assert "hydrate budget" in entry.error_detail


def test_a_worker_whose_lease_moved_on_does_nothing(store, claimed, emulator_client) -> None:
    """The sweeper may have rescued this gate; racing the rescuer would double-run it."""
    exit_code = _run(
        store, emulator_client, claimed.run_request_id, Gate.G1_NEUTRAL, owner="worker/stale/task-9"
    )

    assert exit_code == EXIT_BAD_INVOCATION
    assert store.get(claimed.run_request_id).gate(Gate.G1_NEUTRAL).state is GateState.RUNNING


def test_an_unknown_run_is_not_invented(store, emulator_client) -> None:
    exit_code = _run(
        store, emulator_client, "3f7c2a18-9b4e-4d6a-8c11-5e2f0a7d9b34", Gate.G1_NEUTRAL
    )

    assert exit_code == EXIT_BAD_INVOCATION


def test_the_chain_runs_gate_to_gate(store, claimed, emulator_client, monkeypatch) -> None:
    """G1 through G4, each claimed and committed in turn."""
    monkeypatch.setattr(main_module, "runner_for", lambda gate: no_op_gate)

    for gate in Gate:
        if gate is not Gate.G1_NEUTRAL:
            assert store.claim_gate(
                claimed.for_gate(gate),
                lease_owner=OWNER,
                judge_mode=JudgeMode.GATEKEEPER,
                config_snapshot={},
            ).claimed
        assert _run(store, emulator_client, claimed.run_request_id, gate) == EXIT_OK

    run = store.get(claimed.run_request_id)
    assert all(run.gate(gate).state is GateState.SUCCEEDED for gate in Gate)
