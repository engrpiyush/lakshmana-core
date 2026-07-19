"""Shared fixtures.

Emulator tests are opt-out rather than opt-in: if ``FIRESTORE_EMULATOR_HOST`` is set and
reachable they run, otherwise they skip with a message naming what to start. That keeps
`uv run pytest` honest on a laptop with nothing running while still exercising the real
transaction semantics in CI, which is the only place the state machine can actually be
proven.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from google.auth.credentials import AnonymousCredentials
from google.cloud import firestore

from gatekeeper.config import load_config
from gatekeeper.contracts.payload import SCHEMA_VERSION, GatekeeperRunRequest
from gatekeeper.enums import Gate, RunMode, TriggeredBy
from gatekeeper.runs.store import GatekeeperRunStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "contracts" / "fixtures"

TEST_PROJECT = "lakshmana-test"


def _emulator_reachable() -> bool:
    host = os.environ.get("FIRESTORE_EMULATOR_HOST")
    if not host:
        return False
    name, _, port = host.rpartition(":")
    try:
        with socket.create_connection((name or "127.0.0.1", int(port)), timeout=2):
            return True
    except (OSError, ValueError):
        return False


@pytest.fixture(scope="session")
def emulator_client() -> Iterator[firestore.Client]:
    """A Firestore client bound to the local emulator, or a skip.

    CI sets ``GATEKEEPER_REQUIRE_EMULATOR=1`` so an unreachable emulator is a failure
    there rather than a skip: a green build that quietly skipped the entire state
    machine is worse than a red one.
    """
    if not _emulator_reachable():
        message = (
            "Firestore emulator not reachable; start it and set "
            "FIRESTORE_EMULATOR_HOST=127.0.0.1:8082"
        )
        if os.environ.get("GATEKEEPER_REQUIRE_EMULATOR") == "1":
            pytest.fail(message)
        pytest.skip(message)
    client = firestore.Client(project=TEST_PROJECT, credentials=AnonymousCredentials())
    yield client
    client.close()


@pytest.fixture
def runs_collection() -> str:
    """A collection name unique to one test, so tests never see each other's docs."""
    return f"gatekeeper_runs_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def store(emulator_client: firestore.Client, runs_collection: str) -> GatekeeperRunStore:
    """A store over an isolated collection with the production 90-minute lease."""
    return GatekeeperRunStore(emulator_client, collection=runs_collection)


@pytest.fixture
def config_snapshot() -> dict:
    """The frozen config a run would be created with."""
    return load_config({}).snapshot()


@pytest.fixture
def make_request():
    """Build a payload, defaulting to a fresh FULL/G1 message."""

    def _make(
        *,
        run_request_id: str | None = None,
        gate: Gate = Gate.G1_NEUTRAL,
        mode: RunMode = RunMode.FULL,
        intake_id: str = "intake-test",
        stage3_run_id: str | None = None,
        triggered_by: TriggeredBy = TriggeredBy.OPERATOR,
    ) -> GatekeeperRunRequest:
        return GatekeeperRunRequest(
            schema_version=SCHEMA_VERSION,
            run_request_id=run_request_id or str(uuid.uuid4()),
            intake_id=intake_id,
            stage3_run_id=stage3_run_id or f"stage3run-{uuid.uuid4().hex[:8]}",
            gate=gate,
            mode=mode,
            triggered_by=triggered_by,
            request_timestamp=datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
        )

    return _make
