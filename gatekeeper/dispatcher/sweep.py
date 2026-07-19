"""The sweeper (LLD §11, §12).

Cloud Scheduler calls ``/sweep`` every 15 minutes. It exists because two things can go
missing without anything failing: a **worker** can die mid-gate, leaving a lease that will
never be settled, and a **message** can be lost between the gate commit and the next-gate
publish — the crash window the commit-then-publish ordering deliberately leaves open
(LLD §5).

**The sweeper rescues crashes, never failures.** A ``FAILED`` gate is a decision: something
about the data or the config was wrong, and re-running it would fail identically while
burning a sweep. Those wait for an operator retrigger (owner requirement #7). The only
thing here that ever moves a run to ``FAILED`` is exhausting the sweep budget, which is
the ``GK_E_STUCK`` row of §11's table.

Three passes, in order of how much they cost:

1. **Expired leases** → gate back to ``PENDING`` and republished (``sweepAttempt`` ≤ 3).
2. **Stalled runs** → a run with no gate running and no lease, idle past the grace period,
   has an actionable ``PENDING`` gate whose message never arrived. Re-send it.
3. **DLQ depth** → surfaced as a log line and a counter; the alert policy itself sits on
   the native Pub/Sub metric (LLD §12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gatekeeper.clients.pubsub import Publisher
from gatekeeper.enums import GATE_ORDER, Gate, GateState, RunState, TriggeredBy, prior_gate
from gatekeeper.logging import get_logger, log_context
from gatekeeper.runs.model import GatekeeperRun
from gatekeeper.runs.store import GatekeeperRunStore, SweepOutcome
from gatekeeper.timeutil import utc_now

__all__ = ["SweepReport", "Sweeper", "actionable_gate"]

log = get_logger(__name__)

RUN_OVERDUE_HOURS = 2
"""LLD §12: a run ``RUNNING`` past this emits ``event="run_overdue"`` for the log metric."""


@dataclass(slots=True)
class SweepReport:
    """What one sweep did — the response body, and what the tests assert on."""

    scanned: int = 0
    rescued: list[str] = field(default_factory=list)
    stuck: list[str] = field(default_factory=list)
    republished: list[str] = field(default_factory=list)
    overdue: list[str] = field(default_factory=list)
    dlq_depth: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "rescued": self.rescued,
            "stuck": self.stuck,
            "republished": self.republished,
            "overdue": self.overdue,
            "dlqDepth": self.dlq_depth,
        }


def actionable_gate(run: GatekeeperRun) -> Gate | None:
    """The gate a stalled run is waiting on, if any.

    That is the first ``PENDING`` gate whose predecessor has ``SUCCEEDED`` — exactly the
    gate a lost chain message would have driven. A run with a ``RUNNING`` gate is not
    stalled (pass 1 owns it), and a run with a ``FAILED`` gate is not the sweeper's.
    """
    for gate in GATE_ORDER:
        entry = run.gate(gate)
        if entry.state is GateState.FAILED:
            return None
        if entry.state is not GateState.PENDING:
            continue
        previous = prior_gate(gate)
        if previous is None or run.gate(previous).state is GateState.SUCCEEDED:
            return gate
        return None
    return None


@dataclass(slots=True)
class Sweeper:
    """Collaborators for one sweep, injectable for tests."""

    store: GatekeeperRunStore
    publisher: Publisher
    max_sweeps: int = 3
    grace_minutes: int = 30
    dlq_reader: Any = None
    clock: Any = utc_now

    def sweep(self) -> SweepReport:
        """Run all three passes and report."""
        now: datetime = self.clock()
        report = SweepReport()

        for run in self.store.active_runs():
            report.scanned += 1
            with log_context(runRequestId=run.run_request_id, stage3RunId=run.stage3_run_id):
                self._sweep_run(run, now, report)

        report.dlq_depth = self.dlq_reader() if self.dlq_reader else None
        if report.dlq_depth:
            log.error(
                "messages are sitting in the DLQ",
                fields={"event": "dlq_backlog", "dlqDepth": report.dlq_depth},
            )

        log.info("sweep complete", fields=report.to_json())
        return report

    def _sweep_run(self, run: GatekeeperRun, now: datetime, report: SweepReport) -> None:
        self._flag_overdue(run, now, report)

        expired = [
            gate
            for gate in GATE_ORDER
            if run.gate(gate).state is GateState.RUNNING and not run.gate(gate).lease_held_at(now)
        ]
        if expired:
            for gate in expired:
                self._rescue(run, gate, report)
            return

        # Only a run with nothing in flight can be stalled. Checking this after the
        # rescue pass matters: a gate rescued a moment ago is PENDING and *will* be
        # republished by the rescue itself, and republishing it twice here would put two
        # messages on the topic for one gate. The claim transaction would make the second
        # a harmless no-op, but a DLQ built out of normal operation is what §5 is at pains
        # to avoid.
        if any(run.gate(gate).state is GateState.RUNNING for gate in GATE_ORDER):
            return
        self._republish_stalled(run, now, report)

    def _flag_overdue(self, run: GatekeeperRun, now: datetime, report: SweepReport) -> None:
        """Emit the ``run_overdue`` line the LK-11 log metric filters on (LLD §12)."""
        if run.state is not RunState.RUNNING or run.created_at is None:
            return
        if now - run.created_at < timedelta(hours=RUN_OVERDUE_HOURS):
            return
        report.overdue.append(run.run_request_id)
        log.error(
            "run has been RUNNING past the overdue threshold",
            fields={
                "event": "run_overdue",
                "thresholdHours": RUN_OVERDUE_HOURS,
                "startedAt": run.created_at.isoformat(),
            },
        )

    def _rescue(self, run: GatekeeperRun, gate: Gate, report: SweepReport) -> None:
        outcome = self.store.rescue_gate(run.run_request_id, gate, max_sweeps=self.max_sweeps)
        label = f"{run.run_request_id}:{gate.value}"

        if outcome is SweepOutcome.STUCK:
            report.stuck.append(label)
            log.error(
                "sweeps exhausted; run FAILED",
                fields={"gate": gate.value, "errorCode": "GK_E_STUCK"},
            )
            return
        if outcome is not SweepOutcome.RESCUED:
            # Another sweeper won, or the worker turned out to be alive.
            return

        report.rescued.append(label)
        # Published only after the doc says PENDING: the reverse order would race a
        # dispatcher into claiming a gate this transaction had not yet released.
        self.publisher.publish(run.to_request(gate, triggered_by=TriggeredBy.SWEEPER))
        log.warning("rescued a crashed gate", fields={"gate": gate.value})

    def _republish_stalled(self, run: GatekeeperRun, now: datetime, report: SweepReport) -> None:
        idle_since = run.updated_at or run.created_at
        if idle_since is None or now - idle_since < timedelta(minutes=self.grace_minutes):
            return

        gate = actionable_gate(run)
        if gate is None:
            return

        report.republished.append(f"{run.run_request_id}:{gate.value}")
        self.publisher.publish(run.to_request(gate, triggered_by=TriggeredBy.SWEEPER))
        log.warning(
            "run has been idle with an unclaimed gate; republishing",
            fields={"gate": gate.value, "idleSince": idle_since.isoformat()},
        )
