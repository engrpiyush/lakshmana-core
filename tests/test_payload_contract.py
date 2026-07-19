"""The wire contract (LLD §5, D-6).

Two things are being defended here: the fixtures still round-trip byte-identically, and
the Python codec still agrees with the normative .proto. The second matters because the
.proto is what the Kotlin side reads — if this table and that file drift apart, the two
implementations drift apart silently and the first symptom is a DLQ in production.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

from gatekeeper.contracts.payload import (
    SCHEMA_VERSION,
    WIRE_FIELDS,
    GatekeeperRunRequest,
)
from gatekeeper.contracts.pubsub import parse_push_envelope
from gatekeeper.enums import Gate, RunMode, TriggeredBy
from gatekeeper.errors import ErrorCode, SchemaError
from gatekeeper.runs.model import GatekeeperRun
from tests.conftest import FIXTURES_DIR

PROTO_PATH = Path(__file__).resolve().parent.parent / "contracts" / "gatekeeper_run_request.proto"

PAYLOAD_FIXTURES = [
    "gatekeeper_run_request.g1_full.json",
    "gatekeeper_run_request.g3_from_gate.json",
]


def _valid_payload() -> dict:
    return json.loads((FIXTURES_DIR / "gatekeeper_run_request.g1_full.json").read_text())


# -- golden fixtures ----------------------------------------------------------


@pytest.mark.parametrize("fixture_name", PAYLOAD_FIXTURES)
def test_fixture_round_trips_byte_identical(fixture_name: str) -> None:
    raw = (FIXTURES_DIR / fixture_name).read_text(encoding="utf-8")

    parsed = GatekeeperRunRequest.from_wire(json.loads(raw))

    assert parsed.to_canonical_json() == raw


def test_run_doc_fixture_round_trips_byte_identical() -> None:
    raw = (FIXTURES_DIR / "gatekeeper_runs.doc.json").read_text(encoding="utf-8")

    parsed = GatekeeperRun.from_firestore(json.loads(raw))

    assert parsed.to_canonical_json() == raw


def test_fixture_keys_are_in_proto_field_number_order() -> None:
    raw = (FIXTURES_DIR / "gatekeeper_run_request.g1_full.json").read_text()

    keys = list(json.loads(raw).keys())

    assert keys == [field.json_name for field in WIRE_FIELDS]


# -- agreement with the normative .proto --------------------------------------


def _proto_message_fields() -> list[tuple[str, int]]:
    """Extract (snake_case name, field number) from the GatekeeperRunRequest message."""
    source = PROTO_PATH.read_text()
    body = re.search(r"message GatekeeperRunRequest \{(.*?)\n\}", source, re.DOTALL)
    assert body, "GatekeeperRunRequest message not found in the .proto"
    pattern = re.compile(r"^\s*(?:\w+)\s+(\w+)\s*=\s*(\d+);", re.MULTILINE)
    return [(name, int(number)) for name, number in pattern.findall(body.group(1))]


def test_codec_matches_the_normative_proto() -> None:
    assert _proto_message_fields() == [(field.name, field.number) for field in WIRE_FIELDS]


def test_proto_gate_enum_matches_python_gates() -> None:
    source = PROTO_PATH.read_text()
    body = re.search(r"enum Gate \{(.*?)\n\}", source, re.DOTALL)
    assert body
    names = re.findall(r"^\s*(\w+)\s*=\s*\d+;", body.group(1), re.MULTILINE)

    # The zero value is proto's required "unspecified"; it is not a real gate.
    assert names[0] == "GATE_UNSPECIFIED"
    assert names[1:] == [gate.value for gate in Gate]


def test_proto_mode_enum_matches_python_modes() -> None:
    source = PROTO_PATH.read_text()
    body = re.search(r"enum Mode \{(.*?)\n\}", source, re.DOTALL)
    assert body
    names = re.findall(r"^\s*(\w+)\s*=\s*\d+;", body.group(1), re.MULTILINE)

    assert names[0] == "MODE_UNSPECIFIED"
    assert names[1:] == [mode.value for mode in RunMode]


# -- parsing rejects what it must ---------------------------------------------


def test_unknown_schema_version_is_rejected() -> None:
    payload = _valid_payload() | {"schemaVersion": 99}

    with pytest.raises(SchemaError) as caught:
        GatekeeperRunRequest.from_wire(payload)

    assert caught.value.code is ErrorCode.GK_E_SCHEMA
    assert "99" in str(caught.value)


def test_unknown_schema_version_is_diagnosed_before_unknown_fields() -> None:
    """A future v2 payload carries new keys; the honest diagnosis is the version."""
    payload = _valid_payload() | {"schemaVersion": 2, "brandNewField": "from the future"}

    with pytest.raises(SchemaError, match="schemaVersion"):
        GatekeeperRunRequest.from_wire(payload)


def test_unknown_field_at_a_known_version_is_rejected() -> None:
    payload = _valid_payload() | {"claimText": "never allowed on the wire"}

    with pytest.raises(SchemaError, match="unknown field"):
        GatekeeperRunRequest.from_wire(payload)


def test_missing_field_is_rejected() -> None:
    payload = _valid_payload()
    del payload["intakeId"]

    with pytest.raises(SchemaError, match="intakeId"):
        GatekeeperRunRequest.from_wire(payload)


@pytest.mark.parametrize(
    "run_request_id",
    [
        "not-a-uuid",
        "3F7C2A18-9B4E-4D6A-8C11-5E2F0A7D9B34",  # uppercase is not canonical
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",  # v1, not v4
    ],
)
def test_run_request_id_must_be_a_canonical_uuid4(run_request_id: str) -> None:
    payload = _valid_payload() | {"runRequestId": run_request_id}

    with pytest.raises(SchemaError, match="runRequestId"):
        GatekeeperRunRequest.from_wire(payload)


def test_boolean_is_not_accepted_as_schema_version() -> None:
    """``True == 1`` in Python; the contract still says integer."""
    payload = _valid_payload() | {"schemaVersion": True}

    with pytest.raises(SchemaError, match="integer"):
        GatekeeperRunRequest.from_wire(payload)


@pytest.mark.parametrize("field_name", ["gate", "mode", "triggeredBy"])
def test_unknown_enum_value_is_rejected(field_name: str) -> None:
    payload = _valid_payload() | {field_name: "G9_NONSENSE"}

    with pytest.raises(SchemaError, match=field_name):
        GatekeeperRunRequest.from_wire(payload)


@pytest.mark.parametrize(
    "timestamp",
    ["2026-07-19 04:15:00", "2026-07-19T04:15:00", "yesterday", ""],
)
def test_request_timestamp_must_be_rfc3339_with_an_offset(timestamp: str) -> None:
    payload = _valid_payload() | {"requestTimestamp": timestamp}

    with pytest.raises(SchemaError, match="requestTimestamp"):
        GatekeeperRunRequest.from_wire(payload)


def test_non_object_payload_is_rejected() -> None:
    with pytest.raises(SchemaError, match="JSON object"):
        GatekeeperRunRequest.from_wire([1, 2, 3])


def test_from_bytes_rejects_malformed_json() -> None:
    with pytest.raises(SchemaError, match="JSON"):
        GatekeeperRunRequest.from_bytes(b"{not json")


# -- chaining -----------------------------------------------------------------


def test_for_gate_preserves_the_run_request_id() -> None:
    """The chain is one run: a new id per gate would defeat the idempotency root."""
    request = GatekeeperRunRequest.from_wire(_valid_payload())

    chained = request.for_gate(Gate.G2_CORROBORATION)

    assert chained.run_request_id == request.run_request_id
    assert chained.gate is Gate.G2_CORROBORATION
    assert chained.intake_id == request.intake_id


def test_for_gate_can_restamp_who_triggered_it() -> None:
    request = GatekeeperRunRequest.from_wire(_valid_payload())

    chained = request.for_gate(Gate.G2_CORROBORATION, triggered_by=TriggeredBy.SWEEPER)

    assert chained.triggered_by is TriggeredBy.SWEEPER


def test_bytes_round_trip() -> None:
    request = GatekeeperRunRequest.from_wire(_valid_payload())

    assert GatekeeperRunRequest.from_bytes(request.to_bytes()) == request


def test_schema_version_constant_is_supported() -> None:
    assert GatekeeperRunRequest.from_wire(_valid_payload()).schema_version == SCHEMA_VERSION


# -- push envelope ------------------------------------------------------------


def _envelope(payload: dict, **overrides) -> dict:
    message = {
        "data": base64.b64encode(json.dumps(payload).encode()).decode(),
        "messageId": "9876543210",
        "publishTime": "2026-07-19T04:15:01.000Z",
    } | overrides
    return {"message": message, "subscription": "projects/p/subscriptions/gatekeeper-push"}


def test_push_envelope_is_decoded() -> None:
    message = parse_push_envelope(_envelope(_valid_payload()))

    assert message.request.gate is Gate.G1_NEUTRAL
    assert message.message_id == "9876543210"


def test_push_envelope_carries_the_delivery_attempt() -> None:
    envelope = _envelope(_valid_payload()) | {"deliveryAttempt": 4}

    assert parse_push_envelope(envelope).delivery_attempt == 4


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"message": "not an object"},
        {"message": {}},
        {"message": {"data": "!!!not base64!!!"}},
    ],
)
def test_malformed_push_envelope_is_rejected(envelope: dict) -> None:
    with pytest.raises(SchemaError):
        parse_push_envelope(envelope)
