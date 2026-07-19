"""Launching the worker job (LLD §4).

The dispatcher's only side effect beyond the claim transaction is executing the Cloud Run
Job that does the actual work. The claim is already committed by the time we get here, so
the three environment variables below are the whole handover: **which run, which gate, and
the lease the worker must present back** to settle it. A worker that cannot prove the lease
exits without touching the gate (see ``worker.main.run_gate``), which is what makes a
double-execution harmless.

The execution is started and **not** waited on. A gate takes minutes to hours; the push
subscription wants its ACK in seconds. Failure of the job itself is not the dispatcher's
problem to observe — the lease expires and the sweeper rescues it (LLD §11).
"""

from __future__ import annotations

from typing import Protocol

from gatekeeper.config import Config
from gatekeeper.contracts.payload import GatekeeperRunRequest
from gatekeeper.logging import get_logger

__all__ = ["CloudRunJobLauncher", "JobLauncher", "LoggingJobLauncher", "launcher_for"]

log = get_logger(__name__)


class JobLauncher(Protocol):
    """Executes the worker job for a claimed gate."""

    def execute(self, request: GatekeeperRunRequest, *, lease_owner: str) -> str:
        """Start the job and return an execution identifier."""
        ...


class LoggingJobLauncher:
    """Records the intent to execute without calling the Run API.

    What a local run uses, and what a deployment falls back to when job execution is
    switched off: the claim is still committed, so the lease and attempt counter behave
    exactly as they will in production, and the gate is rescued by the sweeper when no
    worker ever picks it up.
    """

    def execute(self, request: GatekeeperRunRequest, *, lease_owner: str) -> str:
        execution_id = f"not-executed:{request.run_request_id}:{request.gate.value}"
        log.warning(
            "job execution is disabled; claim committed without starting a worker",
            fields={"executionId": execution_id, "leaseOwner": lease_owner},
        )
        return execution_id


class CloudRunJobLauncher:
    """Starts the ``gatekeeper-worker`` Cloud Run Job for one claimed gate.

    The gate is passed as a container override rather than baked into the job, because one
    job definition serves all four gates — which is also why the job's own env must not
    carry these three names: an override replaces the value, and a stale baked-in
    ``GATEKEEPER_GATE`` would otherwise be what a hand-run execution picked up.
    """

    def __init__(self, project_id: str, region: str, job_name: str, client: object | None = None):
        from google.cloud import run_v2

        self._client = client or run_v2.JobsClient()
        self._job_path = f"projects/{project_id}/locations/{region}/jobs/{job_name}"

    def execute(self, request: GatekeeperRunRequest, *, lease_owner: str) -> str:
        from google.cloud import run_v2

        overrides = run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[
                        run_v2.EnvVar(
                            name="GATEKEEPER_RUN_REQUEST_ID", value=request.run_request_id
                        ),
                        run_v2.EnvVar(name="GATEKEEPER_GATE", value=request.gate.value),
                        run_v2.EnvVar(name="GATEKEEPER_LEASE_OWNER", value=lease_owner),
                    ]
                )
            ]
        )
        operation = self._client.run_job(  # type: ignore[attr-defined]
            request=run_v2.RunJobRequest(name=self._job_path, overrides=overrides)
        )
        # The operation name identifies the execution; resolving it would mean waiting for
        # the job to finish, which is precisely what a push handler must not do.
        execution_id = str(getattr(operation, "operation", None) or operation)
        log.info(
            "worker job execution started",
            fields={"job": self._job_path, "executionId": execution_id, "leaseOwner": lease_owner},
        )
        return execution_id


def launcher_for(config: Config) -> JobLauncher:
    """The launcher this environment can use.

    Job execution is opt-in (``gatekeeper.worker.execute-jobs``): a local dispatcher run
    must never reach the Run Admin API by accident, and a deployment turns it on explicitly.
    """
    project_id = config.get_str("gatekeeper.firestore.project-id")
    if not config.get_bool("gatekeeper.worker.execute-jobs") or not project_id:
        return LoggingJobLauncher()
    return CloudRunJobLauncher(
        project_id,
        config.get_str("gatekeeper.worker.job-region"),
        config.get_str("gatekeeper.worker.job-name"),
    )
