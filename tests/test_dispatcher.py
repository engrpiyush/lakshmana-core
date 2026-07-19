"""The dispatcher's push handling (LLD §5).

The response code *is* the Pub/Sub contract, so each one is pinned: 204 means handled
(including every duplicate), 400 sends the message toward the DLQ, 500 asks for a
redelivery. Getting these backwards is how a healthy system manufactures a DLQ backlog,
which is why they are tested rather than assumed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gatekeeper.clients.pubsub import RecordingPublisher
from gatekeeper.config import load_config
from gatekeeper.contracts.payload import GatekeeperRunRequest
from gatekeeper.dispatcher.app import Dispatcher, create_app
from gatekeeper.dispatcher.oidc import OidcVerifier, TokenRejectedError
from gatekeeper.enums import Gate, JudgeMode
from gatekeeper.errors import ErrorCode, FirestoreTxnError
from gatekeeper.runs.store import ClaimRejection, ClaimResult

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "contracts" / "fixtures"


class FakeStore:
    """Records claims and returns whatever the test told it to."""

    def __init__(self, result: ClaimResult | Exception) -> None:
        self._result = result
        self.calls: list[tuple[GatekeeperRunRequest, str, JudgeMode]] = []

    def claim_gate(self, request, *, lease_owner, judge_mode, config_snapshot):
        self.calls.append((request, lease_owner, judge_mode))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[GatekeeperRunRequest, str]] = []

    def execute(self, request, *, lease_owner):
        self.calls.append((request, lease_owner))
        return "execution-1"


def _client(
    store,
    launcher=None,
    judge_mode=JudgeMode.GATEKEEPER,
    verifier=None,
    publisher=None,
):
    dispatcher = Dispatcher(
        store=store,
        config=load_config({}),
        launcher=launcher or FakeLauncher(),
        judge_mode_resolver=lambda _: judge_mode,
        lease_owner_factory=lambda: "dispatcher/test/owner",
        # These tests are about the protocol, not the transport's authentication; the
        # OIDC path has its own tests below and in tests/test_oidc.py.
        verifier=verifier or OidcVerifier(required=False),
        publisher=publisher,
    )
    return TestClient(create_app(dispatcher), raise_server_exceptions=False)


def _payload() -> dict:
    return json.loads((FIXTURES_DIR / "gatekeeper_run_request.g1_full.json").read_text())


def _envelope(payload: dict) -> dict:
    return {
        "message": {
            "data": base64.b64encode(json.dumps(payload).encode()).decode(),
            "messageId": "1234567890",
            "publishTime": "2026-07-19T04:15:01.000Z",
        },
        "subscription": "projects/p/subscriptions/gatekeeper-push",
    }


# -- health -------------------------------------------------------------------


def test_health_endpoint_reports_ok() -> None:
    with _client(FakeStore(ClaimResult(True))) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# -- the happy path -----------------------------------------------------------


def test_a_won_claim_executes_the_worker_job() -> None:
    store = FakeStore(ClaimResult(True))
    launcher = FakeLauncher()

    with _client(store, launcher) as client:
        response = client.post("/pubsub/push", json=_envelope(_payload()))

    assert response.status_code == 204
    assert len(launcher.calls) == 1
    request, lease_owner = launcher.calls[0]
    assert request.gate is Gate.G1_NEUTRAL
    assert lease_owner == "dispatcher/test/owner"


def test_the_claim_receives_the_resolved_judge_mode() -> None:
    store = FakeStore(ClaimResult(True))

    with _client(store, judge_mode=JudgeMode.SHADOW) as client:
        client.post("/pubsub/push", json=_envelope(_payload()))

    assert store.calls[0][2] is JudgeMode.SHADOW


def test_the_same_lease_owner_is_claimed_and_handed_to_the_job() -> None:
    """A mismatch here would make every worker fail its own commit."""
    store = FakeStore(ClaimResult(True))
    launcher = FakeLauncher()

    with _client(store, launcher) as client:
        client.post("/pubsub/push", json=_envelope(_payload()))

    assert store.calls[0][1] == launcher.calls[0][1]


# -- duplicates and stale messages --------------------------------------------


@pytest.mark.parametrize(
    "rejection",
    [
        ClaimRejection.GATE_LEASED,
        ClaimRejection.GATE_TERMINAL,
        ClaimRejection.PRIOR_GATE_NOT_SUCCEEDED,
        ClaimRejection.RUN_SUPERSEDED,
        ClaimRejection.RUN_TERMINAL,
        ClaimRejection.INVALID_ENTRY,
    ],
)
def test_a_lost_claim_is_acked_and_dropped(rejection: ClaimRejection) -> None:
    """Duplicates are normal traffic, not failures — they must never reach the DLQ."""
    store = FakeStore(ClaimResult(False, rejection))
    launcher = FakeLauncher()

    with _client(store, launcher) as client:
        response = client.post("/pubsub/push", json=_envelope(_payload()))

    assert response.status_code == 204
    assert launcher.calls == []


# -- schema failures go to the DLQ --------------------------------------------


def test_an_unknown_schema_version_is_rejected_toward_the_dlq() -> None:
    store = FakeStore(ClaimResult(True))

    with _client(store) as client:
        response = client.post("/pubsub/push", json=_envelope(_payload() | {"schemaVersion": 99}))

    assert response.status_code == 400
    assert response.json()["errorCode"] == ErrorCode.GK_E_SCHEMA.value
    assert store.calls == []


def test_a_malformed_envelope_is_rejected_toward_the_dlq() -> None:
    with _client(FakeStore(ClaimResult(True))) as client:
        response = client.post("/pubsub/push", json={"not": "an envelope"})

    assert response.status_code == 400
    assert response.json()["errorCode"] == ErrorCode.GK_E_SCHEMA.value


def test_a_non_json_body_is_rejected() -> None:
    with _client(FakeStore(ClaimResult(True))) as client:
        response = client.post(
            "/pubsub/push",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400


def test_claim_text_on_the_wire_is_rejected() -> None:
    """IDs only — a payload carrying text is a contract violation, not a warning."""
    with _client(FakeStore(ClaimResult(True))) as client:
        response = client.post(
            "/pubsub/push", json=_envelope(_payload() | {"claimText": "some claim"})
        )

    assert response.status_code == 400


# -- transient failures ask for a redelivery ----------------------------------


def test_transaction_contention_asks_for_a_redelivery() -> None:
    store = FakeStore(FirestoreTxnError("contention"))
    launcher = FakeLauncher()

    with _client(store, launcher) as client:
        response = client.post("/pubsub/push", json=_envelope(_payload()))

    assert response.status_code == 500
    assert response.json()["errorCode"] == ErrorCode.GK_E_FIRESTORE_TXN.value
    assert launcher.calls == []


# -- OIDC ---------------------------------------------------------------------


class RejectingVerifier:
    """Stands in for a real verifier that dislikes the token it was given."""

    def verify(self, authorization):
        raise TokenRejectedError("no token")


def test_an_unauthenticated_push_is_rejected_without_claiming() -> None:
    """401 before the transaction: an unauthenticated caller must not move a gate."""
    store = FakeStore(ClaimResult(True))
    launcher = FakeLauncher()

    with _client(store, launcher, verifier=RejectingVerifier()) as client:
        response = client.post("/pubsub/push", json=_envelope(_payload()))

    assert response.status_code == 401
    assert store.calls == []
    assert launcher.calls == []


def test_an_unauthenticated_sweep_is_rejected() -> None:
    with _client(FakeStore(ClaimResult(True)), verifier=RejectingVerifier()) as client:
        response = client.post("/sweep")

    assert response.status_code == 401


# -- sweep --------------------------------------------------------------------


class EmptyStore:
    """A store with nothing to sweep."""

    def active_runs(self):
        return []


def test_sweep_reports_an_empty_pass() -> None:
    with _client(EmptyStore(), publisher=RecordingPublisher()) as client:
        response = client.post("/sweep")

    assert response.status_code == 200
    assert response.json()["scanned"] == 0


def test_sweep_without_a_publisher_refuses_rather_than_stranding_gates() -> None:
    """A rescue that cannot republish would leave a gate PENDING and undriven."""
    with _client(EmptyStore()) as client:
        response = client.post("/sweep")

    assert response.status_code == 503
    assert response.json()["errorCode"] == "GK_E_SWEEP_UNCONFIGURED"
