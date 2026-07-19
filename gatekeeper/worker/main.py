"""The worker entrypoint — one gate, one job execution (LLD §4, §8).

Framework-free ``main()``: Cloud Run Jobs hand it a task and an environment, not an HTTP
request. The dispatcher has already won the claim, so this process owns the lease it was
given and must present it back to settle the gate.

The commit-then-publish order is the recovery contract: the gate result is durable
before any next-gate message exists, so a crash in the window costs a sweeper rescue
rather than a double-run (LLD §5).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from gatekeeper import __version__
from gatekeeper.clients.firestore import firestore_client
from gatekeeper.clients.pubsub import Publisher, publisher_for
from gatekeeper.config import Config, load_config
from gatekeeper.enums import Gate, TriggeredBy, next_gate
from gatekeeper.errors import GatekeeperError
from gatekeeper.logging import configure_logging, get_logger, log_context
from gatekeeper.runs.model import GatekeeperRun
from gatekeeper.runs.store import GatekeeperRunStore
from gatekeeper.worker.gates import GateContext, runner_for
from gatekeeper.worker.queue import PairQueue

__all__ = ["main", "run_gate"]

log = get_logger(__name__)

EXIT_OK = 0
EXIT_BAD_INVOCATION = 2
EXIT_GATE_FAILED = 1


def run_gate(
    store: GatekeeperRunStore,
    run_request_id: str,
    gate: Gate,
    lease_owner: str,
    *,
    context_factory: Callable[[GatekeeperRun], GateContext],
    publisher: Publisher | None = None,
) -> int:
    """Execute one gate, settle it, and publish the next. Returns a process exit code.

    A gate that raises is recorded FAILED with its ``GK_E_*`` code and the run goes
    FAILED with it — no automatic retry. The operator fixes the data or config and
    retriggers FROM_GATE from the vishwamitra UI (LLD §6 rule 3).
    """
    run = store.get(run_request_id)
    if run is None:
        log.error("run doc not found; nothing to execute")
        return EXIT_BAD_INVOCATION

    entry = run.gate(gate)
    if entry.lease_owner != lease_owner:
        # The lease moved on — almost always a sweeper rescue after this task stalled.
        # Doing the work anyway would race the rescuer for the same gate.
        log.warning(
            "lease is held by another worker; exiting without touching the gate",
            fields={"expectedLeaseOwner": lease_owner, "actualLeaseOwner": entry.lease_owner},
        )
        return EXIT_BAD_INVOCATION

    context = context_factory(run)
    try:
        counters = runner_for(gate)(context)
    except GatekeeperError as exc:
        log.error(
            "gate failed",
            fields={"errorCode": exc.code.value, "detail": exc.detail},
            exc_info=True,
        )
        store.fail_gate(
            run_request_id,
            gate,
            lease_owner=lease_owner,
            error_code=exc.code,
            error_detail=exc.detail,
        )
        return EXIT_GATE_FAILED
    finally:
        context.close()

    if not store.commit_gate(run_request_id, gate, lease_owner=lease_owner, counters=counters):
        log.warning("gate commit was rejected; not publishing the next gate")
        return EXIT_BAD_INVOCATION

    log.info("gate committed", fields={"counters": counters})

    # Commit first, publish second — always. The window between them is a crash the
    # sweeper rescues; the reverse order would let a redelivery double-run a gate whose
    # result was never recorded (LLD §5).
    following = next_gate(gate)
    if following is None:
        # After G4 comes FINALIZE: internal, no message, no lease, publishes nothing
        # (LLD §8). It runs here rather than as a fifth gate because there is nothing for a
        # dispatcher to claim — the work is one transaction over a doc this worker just
        # committed to.
        return _finalize(store, run_request_id, context)
    if publisher is None:
        log.warning(
            "no publisher configured; the chain stops here",
            fields={"nextGate": following.value},
        )
    else:
        publisher.publish(run.to_request(following, triggered_by=TriggeredBy.SYSTEM))

    return EXIT_OK


def _finalize(store: GatekeeperRunStore, run_request_id: str, context: GateContext) -> int:
    """Run FINALIZE after G4's commit (LLD §8).

    The run doc is **re-read** rather than reused: ``context.run`` predates G4's own commit,
    so its gate counters do not carry the tail's spend yet, and ``llmSpendUsd`` has to come
    from what was durably committed rather than from this process's memory.

    A reconciliation failure fails the *run*, not the gate. The gate genuinely succeeded —
    it judged what it was given — and what broke is the run-level invariant: some pair is
    stranded, and marking the run SUCCEEDED over it would tell vishwamitra to consume a
    verdict set with a hole in it.
    """
    from gatekeeper.worker.finalize import ReconciliationError, finalize

    run = store.get(run_request_id)
    if run is None:
        log.error("run doc vanished between commit and finalize")
        return EXIT_BAD_INVOCATION

    queue = PairQueue(
        context.client,
        collection=context.config.get_str("gatekeeper.queue.collection"),
        lease_minutes=context.config.get_int("gatekeeper.queue.lease-minutes"),
    )
    spend = float(run.gate(Gate.G4_ESCALATION).counters.get("llmSpendUsd") or 0.0)

    try:
        finalize(store, run, queue, spend_usd=spend)
    except ReconciliationError as exc:
        log.error(
            "run totals do not reconcile; refusing to mark the run SUCCEEDED",
            fields={"errorCode": exc.code.value, "detail": exc.detail},
        )
        store.fail_run(run_request_id, error_code=exc.code, error_detail=exc.detail)
        return EXIT_GATE_FAILED

    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Read the task's environment, run its gate, and exit.

    Environment:
        GATEKEEPER_RUN_REQUEST_ID: the run to execute.
        GATEKEEPER_GATE: which gate.
        GATEKEEPER_LEASE_OWNER: the lease the dispatcher recorded when it claimed.
    """
    configure_logging()
    config: Config = load_config()

    run_request_id = os.environ.get("GATEKEEPER_RUN_REQUEST_ID", "")
    raw_gate = os.environ.get("GATEKEEPER_GATE", "")
    lease_owner = os.environ.get("GATEKEEPER_LEASE_OWNER", "")

    missing = [
        name
        for name, value in (
            ("GATEKEEPER_RUN_REQUEST_ID", run_request_id),
            ("GATEKEEPER_GATE", raw_gate),
            ("GATEKEEPER_LEASE_OWNER", lease_owner),
        )
        if not value
    ]
    if missing:
        log.error("worker invoked without required environment", fields={"missing": missing})
        return EXIT_BAD_INVOCATION

    try:
        gate = Gate(raw_gate)
    except ValueError:
        log.error("unknown gate", fields={"found": raw_gate})
        return EXIT_BAD_INVOCATION

    with log_context(runRequestId=run_request_id, gate=gate.value):
        log.info("worker starting", fields={"version": __version__, "leaseOwner": lease_owner})
        client = firestore_client(config)
        store = GatekeeperRunStore(
            client, lease_minutes=config.get_int("gatekeeper.lease.gate-minutes")
        )
        return run_gate(
            store,
            run_request_id,
            gate,
            lease_owner,
            context_factory=lambda run: GateContext(
                run=run,
                gate=gate,
                config=config,
                client=client,
                lease_owner=lease_owner,
            ),
            publisher=publisher_for(config),
        )


if __name__ == "__main__":
    sys.exit(main())
