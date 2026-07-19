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

from gatekeeper import __version__
from gatekeeper.clients.firestore import firestore_client
from gatekeeper.config import Config, load_config
from gatekeeper.enums import Gate, next_gate
from gatekeeper.errors import GatekeeperError
from gatekeeper.logging import configure_logging, get_logger, log_context
from gatekeeper.runs.store import GatekeeperRunStore
from gatekeeper.worker.gates import runner_for

__all__ = ["main", "run_gate"]

log = get_logger(__name__)

EXIT_OK = 0
EXIT_BAD_INVOCATION = 2
EXIT_GATE_FAILED = 1


def run_gate(store: GatekeeperRunStore, run_request_id: str, gate: Gate, lease_owner: str) -> int:
    """Execute one gate and settle it. Returns a process exit code.

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

    try:
        counters = runner_for(gate)(run, gate)
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

    if not store.commit_gate(run_request_id, gate, lease_owner=lease_owner, counters=counters):
        log.warning("gate commit was rejected; not publishing the next gate")
        return EXIT_BAD_INVOCATION

    log.info("gate committed", fields={"counters": counters})

    following = next_gate(gate)
    if following is None:
        # After G4 comes FINALIZE, which is internal and publishes nothing (LLD §8).
        log.info("last gate committed; FINALIZE is pending VA-103")
    else:
        # Chain publishing lands with the dispatcher/G1 ticket (VA-98/VA-99).
        log.info(
            "next gate is pending VA-98 chain publishing", fields={"nextGate": following.value}
        )

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
        return run_gate(store, run_request_id, gate, lease_owner)


if __name__ == "__main__":
    sys.exit(main())
