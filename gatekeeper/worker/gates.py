"""Gate implementations, registered by gate (LLD §8).

Session01 ships the registry and a no-op so the chain is executable end to end against
the emulator. The real gates arrive on their own tickets — G1 with VA-99, G2/G3 with
VA-100/VA-101, G4 with VA-102 — and each one replaces its entry here without touching
the dispatcher, the store, or the contract.

Every gate has the same shape: take the run and its frozen ``configSnapshot``, do the
work, return the counters that land on the gate entry. Thresholds are read from the
snapshot, never from live config, so a mid-run edit cannot split a run's calibration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gatekeeper.enums import Gate
from gatekeeper.logging import get_logger
from gatekeeper.runs.model import GatekeeperRun

__all__ = ["GateRunner", "no_op_gate", "runner_for"]

log = get_logger(__name__)

GateRunner = Callable[[GatekeeperRun, Gate], dict[str, Any]]


def no_op_gate(run: GatekeeperRun, gate: Gate) -> dict[str, Any]:
    """Do nothing and report zero pairs seen.

    Placeholder for a gate not yet implemented. It commits successfully — that is the
    point, it exercises claim → execute → commit — but its zero ``seen`` counter is the
    signal on the run doc that no judging actually happened.
    """
    log.warning(
        "no-op gate: not implemented yet, committing zero counters",
        fields={"judgeMode": run.judge_mode.value},
    )
    return {"seen": 0, "forwarded": 0, "noOp": True}


_RUNNERS: dict[Gate, GateRunner] = {}


def runner_for(gate: Gate) -> GateRunner:
    """The runner for ``gate``, falling back to the no-op until its ticket lands."""
    return _RUNNERS.get(gate, no_op_gate)


def register(gate: Gate, runner: GateRunner) -> None:
    """Register a real gate implementation."""
    _RUNNERS[gate] = runner
