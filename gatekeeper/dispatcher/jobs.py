"""Launching the worker job (LLD §4).

The dispatcher's only side effect beyond the claim transaction is executing the Cloud
Run Job that does the actual work. The real Run Admin API call is VA-98's (session04);
this module fixes the seam now so the claim path can be built and tested against it.
"""

from __future__ import annotations

from typing import Protocol

from gatekeeper.contracts.payload import GatekeeperRunRequest
from gatekeeper.logging import get_logger

__all__ = ["JobLauncher", "LoggingJobLauncher"]

log = get_logger(__name__)


class JobLauncher(Protocol):
    """Executes the worker job for a claimed gate."""

    def execute(self, request: GatekeeperRunRequest, *, lease_owner: str) -> str:
        """Start the job and return an execution identifier."""
        ...


class LoggingJobLauncher:
    """Records the intent to execute without calling the Run API.

    The default until VA-98 wires the real launcher: a claim is still committed, so the
    lease and attempt counter behave exactly as they will in production, and the gate is
    rescued by the sweeper when no worker ever picks it up.
    """

    def execute(self, request: GatekeeperRunRequest, *, lease_owner: str) -> str:
        execution_id = f"pending-va98:{request.run_request_id}:{request.gate.value}"
        log.warning(
            "worker job execution is not wired yet (VA-98); claim committed without a job",
            fields={"executionId": execution_id, "leaseOwner": lease_owner},
        )
        return execution_id
