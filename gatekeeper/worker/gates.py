"""Gate implementations, registered by gate (LLD §8).

Every gate has the same shape: take a :class:`GateContext` — the run with its frozen
``configSnapshot``, plus the clients it may touch — do the work, and return the counters
that land on the gate entry in the run doc.

**Collaborators are built lazily and injected, never imported at the point of use.** A gate
that reached for a Firestore client or an ONNX session directly would be untestable without
both, and the context is what lets `tests/` drive a real gate against the emulator with a
stub scorer. It is also what keeps a 400 MB model off the critical path of a gate that
turns out to have no pairs to judge.

Thresholds and the model are read from ``run.config_snapshot``, never from live config: the
freeze invariant (LLD §8) covers both, and `tests/test_freeze_invariant.py` proves it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.cloud import firestore

from gatekeeper.clients.neo4j import Neo4jReader, neo4j_driver
from gatekeeper.config import Config
from gatekeeper.enums import Gate
from gatekeeper.logging import get_logger
from gatekeeper.models.loader import ModelLoader, loader_from_config
from gatekeeper.runs.model import GatekeeperRun

__all__ = ["GateContext", "GateRunner", "no_op_gate", "register", "runner_for"]

log = get_logger(__name__)


@dataclass(slots=True)
class GateContext:
    """One gate execution's inputs and collaborators."""

    run: GatekeeperRun
    gate: Gate
    config: Config
    client: firestore.Client
    lease_owner: str = ""

    reader_factory: Callable[[], Neo4jReader] | None = None
    loader_factory: Callable[[], ModelLoader] | None = None
    scorer_factory: Callable[[Any], Any] | None = None
    """``GateBinding`` → scorer, bypassing the loader entirely when set.

    The seam tests use to run a real gate loop without real weights. It takes the binding
    rather than a loaded artifact so that overriding it skips the fetch as well as the ONNX
    session — the two things a test has no way to afford.
    """

    _reader: Neo4jReader | None = field(default=None, init=False, repr=False)
    _loader: ModelLoader | None = field(default=None, init=False, repr=False)

    def reader(self) -> Neo4jReader:
        """The read-only graph handle, built once per gate execution."""
        if self._reader is None:
            if self.reader_factory is not None:
                self._reader = self.reader_factory()
            else:
                self._reader = Neo4jReader(
                    neo4j_driver(self.config),
                    database=self.config.get_str("gatekeeper.neo4j.database"),
                )
        return self._reader

    def loader(self) -> ModelLoader:
        if self._loader is None:
            self._loader = (
                self.loader_factory() if self.loader_factory else loader_from_config(self.config)
            )
        return self._loader

    def close(self) -> None:
        """Release the graph driver. Safe to call when nothing was ever built."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None


GateRunner = Callable[[GateContext], dict[str, Any]]


def no_op_gate(context: GateContext) -> dict[str, Any]:
    """Do nothing and report zero pairs seen.

    Placeholder for a gate not yet implemented. It commits successfully — that is the
    point, it exercises claim → execute → commit — but its zero ``seen`` counter is the
    signal on the run doc that no judging actually happened.
    """
    log.warning(
        "no-op gate: not implemented yet, committing zero counters",
        fields={"judgeMode": context.run.judge_mode.value},
    )
    return {"seen": 0, "forwarded": 0, "noOp": True}


_RUNNERS: dict[Gate, GateRunner] = {}


def runner_for(gate: Gate) -> GateRunner:
    """The runner for ``gate``, falling back to the no-op until its ticket lands."""
    _load_implementations()
    return _RUNNERS.get(gate, no_op_gate)


def register(gate: Gate, runner: GateRunner) -> None:
    """Register a real gate implementation."""
    _RUNNERS[gate] = runner


def _load_implementations() -> None:
    """Import the gate modules that register themselves.

    Deferred so that importing this registry does not drag in the scorer — and through it
    ONNX Runtime — for a process that only wants to know a gate's name.
    """
    if _RUNNERS:
        return
    from gatekeeper.gates import g1  # noqa: F401  (imported for its registration)
