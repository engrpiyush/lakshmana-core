"""The dispatcher service (LLD §4, §5, §6).

Thin by design: verify the schema, claim the gate in one transaction, execute the job,
ACK fast. Everything expensive happens in the worker, because a push subscription wants
its answer in seconds and a gate takes minutes.

Response codes are the Pub/Sub contract, so they are chosen deliberately:

* **204** — handled. Includes every duplicate and stale message: those are *expected*
  under at-least-once delivery, and NACKing them would manufacture a DLQ backlog out of
  normal operation (LLD §6 rule 2).
* **400** — ``GK_E_SCHEMA``. Never retried into success; five attempts send it to the
  DLQ, which is exactly where an unparseable message belongs (LLD §11).
* **500** — transient. Pub/Sub redelivers, and the claim transaction makes the retry
  safe.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request, Response, status

from gatekeeper import __version__
from gatekeeper.clients.firestore import firestore_client
from gatekeeper.clients.pubsub import Publisher, dlq_depth, publisher_for
from gatekeeper.config import Config, load_config
from gatekeeper.contracts.pubsub import PushMessage, parse_push_envelope
from gatekeeper.dispatcher.jobs import JobLauncher, launcher_for
from gatekeeper.dispatcher.oidc import OidcVerifier, TokenRejectedError, verifier_for
from gatekeeper.dispatcher.sweep import Sweeper
from gatekeeper.enums import JudgeMode
from gatekeeper.errors import FirestoreTxnError, SchemaError
from gatekeeper.integration import resolve_judge_mode
from gatekeeper.logging import configure_logging, get_logger, log_context
from gatekeeper.runs.store import GatekeeperRunStore

__all__ = ["Dispatcher", "create_app"]

log = get_logger(__name__)


def _default_lease_owner() -> str:
    """Identify this dispatcher instance in the lease, revision and all."""
    revision = os.environ.get("K_REVISION") or socket.gethostname()
    return f"dispatcher/{revision}/{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class Dispatcher:
    """The push handler's collaborators, injectable for tests."""

    store: GatekeeperRunStore
    config: Config
    launcher: JobLauncher
    judge_mode_resolver: Callable[[str], JudgeMode]
    lease_owner_factory: Callable[[], str] = _default_lease_owner
    verifier: OidcVerifier = field(default_factory=OidcVerifier)
    publisher: Publisher | None = None
    dlq_reader: Callable[[], int | None] | None = None

    def sweeper(self) -> Sweeper:
        """Build a sweeper over the same store this dispatcher claims with."""
        if self.publisher is None:
            raise RuntimeError("sweeping needs a publisher; none was configured")
        return Sweeper(
            store=self.store,
            publisher=self.publisher,
            max_sweeps=self.config.get_int("gatekeeper.sweep.max-attempts"),
            grace_minutes=self.config.get_int("gatekeeper.sweep.requested-grace-minutes"),
            dlq_reader=self.dlq_reader,
        )


def _claim_and_execute(current: Dispatcher, message: PushMessage, response: Response) -> Any:
    """Claim the gate this message names and, if we won, start its worker.

    Split out of the route so the handler stays a thin sequence of guards: everything
    above it is about whether the *message* is acceptable, everything here is about
    whether the *transition* is.
    """
    payload = message.request
    with log_context(
        runRequestId=payload.run_request_id,
        gate=payload.gate.value,
        intakeId=payload.intake_id,
        stage3RunId=payload.stage3_run_id,
    ):
        judge_mode = current.judge_mode_resolver(payload.stage3_run_id)
        lease_owner = current.lease_owner_factory()

        try:
            result = current.store.claim_gate(
                payload,
                lease_owner=lease_owner,
                judge_mode=judge_mode,
                config_snapshot=current.config.snapshot(),
            )
        except FirestoreTxnError as exc:
            # Transient. Nothing was written, so a redelivery is safe and is the
            # cheapest fix — 500 asks Pub/Sub for exactly that.
            log.error(
                "claim transaction did not resolve; asking for redelivery",
                fields={"errorCode": exc.code.value, "detail": exc.detail},
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"errorCode": exc.code.value, "detail": exc.detail}

        if not result.claimed:
            log.info(
                "GK_DUP: dropping duplicate or stale message",
                fields={
                    "rejection": result.rejection.value if result.rejection else None,
                    "messageId": message.message_id,
                    "deliveryAttempt": message.delivery_attempt,
                },
            )
            response.status_code = status.HTTP_204_NO_CONTENT
            return None

        if result.superseded:
            log.info(
                "superseded predecessor runs",
                fields={"supersededRunRequestIds": list(result.superseded)},
            )

        execution_id = current.launcher.execute(payload, lease_owner=lease_owner)
        log.info(
            "gate claimed and job executed",
            fields={
                "leaseOwner": lease_owner,
                "executionId": execution_id,
                "judgeMode": judge_mode.value,
                "attempt": result.run.gate(payload.gate).attempt if result.run else None,
            },
        )
        response.status_code = status.HTTP_204_NO_CONTENT
        return None


def create_app(dispatcher: Dispatcher | None = None) -> FastAPI:
    """Build the ASGI app.

    Args:
        dispatcher: pre-built collaborators. When omitted, they are constructed at
            startup from the environment — which is what the container does.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if dispatcher is None:
            configure_logging()
            config = load_config()
            client = firestore_client(config)
            app.state.dispatcher = Dispatcher(
                store=GatekeeperRunStore(
                    client, lease_minutes=config.get_int("gatekeeper.lease.gate-minutes")
                ),
                config=config,
                launcher=launcher_for(config),
                judge_mode_resolver=lambda run_id: resolve_judge_mode(client, config, run_id),
                verifier=verifier_for(config),
                publisher=publisher_for(config),
                dlq_reader=lambda: dlq_depth(config),
            )
            log.info("dispatcher ready", fields={"version": __version__})
        else:
            app.state.dispatcher = dispatcher
        yield

    app = FastAPI(title="gatekeeper-dispatcher", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "service": "gatekeeper-dispatcher", "version": __version__}

    @app.post("/pubsub/push")
    async def pubsub_push(request: Request, response: Response) -> Any:
        current: Dispatcher = request.app.state.dispatcher

        try:
            current.verifier.verify(request.headers.get("authorization"))
        except TokenRejectedError as exc:
            # 401, not 400: the message may be perfectly well-formed and simply not ours
            # to accept. Pub/Sub retries, and a caller that never authenticates ends up in
            # the DLQ, which is where unauthenticated traffic belongs.
            log.error("rejecting an unauthenticated push", fields={"detail": str(exc)})
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"errorCode": "GK_E_UNAUTHENTICATED", "detail": str(exc)}

        try:
            envelope = await request.json()
        except ValueError as exc:
            log.warning("push body is not JSON", fields={"errorCode": SchemaError.code.value})
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"errorCode": SchemaError.code.value, "detail": str(exc)}

        try:
            message = parse_push_envelope(envelope)
        except SchemaError as exc:
            # Straight to the DLQ. Guessing at a payload we do not understand is the one
            # thing the protocol section forbids outright.
            log.error(
                "rejecting unparseable message",
                fields={"errorCode": exc.code.value, "detail": exc.detail},
            )
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"errorCode": exc.code.value, "detail": exc.detail}

        return _claim_and_execute(current, message, response)

    @app.post("/sweep")
    async def sweep(request: Request, response: Response) -> Any:
        """Cloud Scheduler's 15-minute rescue pass (LLD §11).

        Returns the report rather than 204: this endpoint has a human audience — the
        scheduler's execution log is where an operator looks first when a run is stuck.
        """
        current: Dispatcher = request.app.state.dispatcher

        try:
            current.verifier.verify(request.headers.get("authorization"))
        except TokenRejectedError as exc:
            log.error("rejecting an unauthenticated sweep", fields={"detail": str(exc)})
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"errorCode": "GK_E_UNAUTHENTICATED", "detail": str(exc)}

        try:
            return current.sweeper().sweep().to_json()
        except RuntimeError as exc:
            # No publisher: a rescue would move gates back to PENDING and then have no way
            # to re-drive them, which is worse than not sweeping at all.
            log.error("sweep is not configured", fields={"detail": str(exc)})
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"errorCode": "GK_E_SWEEP_UNCONFIGURED", "detail": str(exc)}

    return app


app = create_app()
"""Module-level ASGI app for ``uvicorn gatekeeper.dispatcher.app:app``."""
