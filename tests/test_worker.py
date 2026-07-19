"""The worker entrypoint against the emulator (LLD §4, §8).

This is the DoD's "worker runs a no-op gate against the emulator": claim, execute,
commit, all through real Firestore transactions. The gates themselves arrive on later
tickets, but the chassis they plug into is proven here.
"""

from __future__ import annotations

import pytest

from gatekeeper.enums import Gate, GateState, JudgeMode, RunState
from gatekeeper.errors import ErrorCode, Neo4jUnavailableError
from gatekeeper.worker import main as main_module
from gatekeeper.worker.main import EXIT_BAD_INVOCATION, EXIT_GATE_FAILED, EXIT_OK, run_gate

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"


@pytest.fixture
def claimed(store, make_request, config_snapshot):
    """A run whose G1 gate is claimed and waiting for a worker."""
    request = make_request()
    result = store.claim_gate(
        request,
        lease_owner=OWNER,
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    )
    assert result.claimed
    return request


def test_the_no_op_gate_commits_against_the_emulator(store, claimed) -> None:
    exit_code = run_gate(store, claimed.run_request_id, Gate.G1_NEUTRAL, OWNER)

    assert exit_code == EXIT_OK
    entry = store.get(claimed.run_request_id).gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.SUCCEEDED
    assert entry.counters["noOp"] is True
    assert entry.lease_owner is None


def test_a_gate_failure_is_recorded_with_its_error_code(store, claimed, monkeypatch) -> None:
    def _explode(run, gate):
        raise Neo4jUnavailableError("hydrate budget exhausted")

    # main imports runner_for by value, so the patch has to land on its binding.
    monkeypatch.setattr(main_module, "runner_for", lambda gate: _explode)

    exit_code = run_gate(store, claimed.run_request_id, Gate.G1_NEUTRAL, OWNER)

    assert exit_code == EXIT_GATE_FAILED
    run = store.get(claimed.run_request_id)
    assert run.state is RunState.FAILED
    entry = run.gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.FAILED
    assert entry.error_code is ErrorCode.GK_E_NEO4J_UNAVAILABLE
    assert "hydrate budget" in entry.error_detail


def test_a_worker_whose_lease_moved_on_does_nothing(store, claimed) -> None:
    """The sweeper may have rescued this gate; racing the rescuer would double-run it."""
    exit_code = run_gate(store, claimed.run_request_id, Gate.G1_NEUTRAL, "worker/stale/task-9")

    assert exit_code == EXIT_BAD_INVOCATION
    assert store.get(claimed.run_request_id).gate(Gate.G1_NEUTRAL).state is GateState.RUNNING


def test_an_unknown_run_is_not_invented(store) -> None:
    exit_code = run_gate(store, "3f7c2a18-9b4e-4d6a-8c11-5e2f0a7d9b34", Gate.G1_NEUTRAL, OWNER)

    assert exit_code == EXIT_BAD_INVOCATION


def test_the_chain_runs_gate_to_gate(store, claimed) -> None:
    """G1 through G4, each claimed and committed in turn."""
    for gate in Gate:
        if gate is not Gate.G1_NEUTRAL:
            assert store.claim_gate(
                claimed.for_gate(gate),
                lease_owner=OWNER,
                judge_mode=JudgeMode.GATEKEEPER,
                config_snapshot={},
            ).claimed
        assert run_gate(store, claimed.run_request_id, gate, OWNER) == EXIT_OK

    run = store.get(claimed.run_request_id)
    assert all(run.gate(gate).state is GateState.SUCCEEDED for gate in Gate)
