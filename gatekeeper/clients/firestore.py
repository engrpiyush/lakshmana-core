"""Firestore client construction (LLD §4, §13).

Production uses ADC with the worker/dispatcher service account. Locally, setting
``FIRESTORE_EMULATOR_HOST`` is enough: the client library routes to the emulator over a
plaintext channel and needs no real credentials, so nothing here ever reaches GCP by
accident during a local test run.
"""

from __future__ import annotations

import os

from google.auth.credentials import AnonymousCredentials
from google.cloud import firestore

from gatekeeper.config import Config
from gatekeeper.logging import get_logger

__all__ = ["EMULATOR_ENV_VAR", "firestore_client", "is_emulator"]

log = get_logger(__name__)

EMULATOR_ENV_VAR = "FIRESTORE_EMULATOR_HOST"

_EMULATOR_PROJECT_FALLBACK = "lakshmana-local"


def is_emulator() -> bool:
    """True when the Firestore emulator is configured for this process."""
    return bool(os.environ.get(EMULATOR_ENV_VAR))


def firestore_client(
    config: Config | None = None, *, project_id: str | None = None
) -> firestore.Client:
    """Build a Firestore client for the current environment.

    Args:
        config: resolved config; supplies ``gatekeeper.firestore.project-id``.
        project_id: explicit override, mostly for tests.

    Returns:
        A client bound to the emulator when ``FIRESTORE_EMULATOR_HOST`` is set,
        otherwise an ADC-authenticated client.
    """
    resolved = project_id or (config.get_str("gatekeeper.firestore.project-id") if config else "")

    if is_emulator():
        resolved = resolved or os.environ.get("GOOGLE_CLOUD_PROJECT") or _EMULATOR_PROJECT_FALLBACK
        log.info(
            "using firestore emulator",
            fields={"host": os.environ[EMULATOR_ENV_VAR], "projectId": resolved},
        )
        return firestore.Client(project=resolved, credentials=AnonymousCredentials())

    if not resolved:
        # ADC can infer a project, but an unset one here usually means a missing env
        # var in a Cloud Run revision — better to say so than to write to a surprise.
        raise ValueError(
            "no Firestore project configured; set GATEKEEPER_FIRESTORE_PROJECT_ID "
            "or GOOGLE_CLOUD_PROJECT"
        )
    return firestore.Client(project=resolved)
