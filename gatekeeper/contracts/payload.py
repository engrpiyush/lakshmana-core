"""``GatekeeperRunRequest`` — the Pub/Sub payload codec (LLD §5, D-6).

Hand-written on purpose: D-6 rules out a shared code artifact between the repos, so
``contracts/gatekeeper_run_request.proto`` is the normative schema, this module is the
Python mapping, and the golden fixtures in ``contracts/fixtures/`` are what actually
pin the two implementations together. ``tests/test_payload_contract.py`` re-reads the
.proto and fails if this table drifts from it.

Parsing is strict in both directions. Every field is required, unknown keys are refused,
and anything malformed raises :class:`~gatekeeper.errors.SchemaError` — which the
dispatcher turns into a NACK so the message reaches the DLQ instead of being guessed at
(LLD §11, ``GK_E_SCHEMA``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Self
from uuid import UUID

from gatekeeper.enums import Gate, RunMode, TriggeredBy
from gatekeeper.errors import SchemaError
from gatekeeper.timeutil import parse_rfc3339

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "WIRE_FIELDS",
    "GatekeeperRunRequest",
    "WireField",
]

SCHEMA_VERSION = 1
"""The version this build emits."""

SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
"""Versions this build accepts. Anything else is a DLQ, never a guess (LLD §11)."""

_UUID_VERSION = 4
"""``runRequestId`` is a UUIDv4 minted by the publisher (LLD §5)."""


@dataclass(frozen=True, slots=True)
class WireField:
    """One payload field: Python name, proto3 JSON name, and proto field number."""

    name: str
    json_name: str
    number: int


WIRE_FIELDS: tuple[WireField, ...] = (
    WireField("schema_version", "schemaVersion", 1),
    WireField("run_request_id", "runRequestId", 2),
    WireField("intake_id", "intakeId", 3),
    WireField("stage3_run_id", "stage3RunId", 4),
    WireField("gate", "gate", 5),
    WireField("mode", "mode", 6),
    WireField("triggered_by", "triggeredBy", 7),
    WireField("request_timestamp", "requestTimestamp", 8),
)
"""Ordered by proto field number — this order is the canonical JSON key order."""

_JSON_NAMES: tuple[str, ...] = tuple(field.json_name for field in WIRE_FIELDS)


def _require_str(value: Any, json_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{json_name} must be a non-empty string, got {value!r}")
    return value


def _require_uuid4(value: Any, json_name: str) -> str:
    raw = _require_str(value, json_name)
    try:
        parsed = UUID(raw)
    except ValueError as exc:
        raise SchemaError(f"{json_name} must be a UUID, got {raw!r}") from exc
    if parsed.version != _UUID_VERSION:
        raise SchemaError(f"{json_name} must be a UUIDv4, got a v{parsed.version} UUID")
    if str(parsed) != raw:
        raise SchemaError(f"{json_name} must be canonical lowercase UUID text, got {raw!r}")
    return raw


def _require_enum[E: str](enum_cls: type[E], value: Any, json_name: str) -> E:
    raw = _require_str(value, json_name)
    try:
        return enum_cls(raw)
    except ValueError as exc:
        allowed = ", ".join(sorted(member.value for member in enum_cls))  # type: ignore[attr-defined]
        raise SchemaError(f"{json_name} must be one of [{allowed}], got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class GatekeeperRunRequest:
    """An attempted gate transition. IDs only — never claim text (LLD §5)."""

    schema_version: int
    run_request_id: str
    intake_id: str
    stage3_run_id: str
    gate: Gate
    mode: RunMode
    triggered_by: TriggeredBy
    request_timestamp: str

    # -- serialization --------------------------------------------------------

    def to_wire(self) -> dict[str, Any]:
        """proto3 JSON mapping: lowerCamelCase keys in proto field-number order."""
        return {
            "schemaVersion": self.schema_version,
            "runRequestId": self.run_request_id,
            "intakeId": self.intake_id,
            "stage3RunId": self.stage3_run_id,
            "gate": self.gate.value,
            "mode": self.mode.value,
            "triggeredBy": self.triggered_by.value,
            "requestTimestamp": self.request_timestamp,
        }

    def to_canonical_json(self) -> str:
        """The exact byte form stored in ``contracts/fixtures/`` (2-space indent, \\n)."""
        return json.dumps(self.to_wire(), indent=2, ensure_ascii=False) + "\n"

    def to_bytes(self) -> bytes:
        """Compact UTF-8 bytes for the Pub/Sub message body."""
        return json.dumps(self.to_wire(), separators=(",", ":"), ensure_ascii=False).encode()

    # -- parsing --------------------------------------------------------------

    @classmethod
    def from_wire(cls, payload: Any) -> Self:
        """Parse a proto3-JSON object.

        Raises:
            SchemaError: unknown ``schemaVersion``, unknown key, missing field, or any
                value that fails validation. All of these are ``GK_E_SCHEMA`` → DLQ.
        """
        if not isinstance(payload, dict):
            raise SchemaError(f"payload must be a JSON object, got {type(payload).__name__}")

        # Version first: a future v2 payload will also carry unknown keys, and the
        # honest diagnosis is the version, not the keys.
        if "schemaVersion" not in payload:
            raise SchemaError("payload is missing schemaVersion")
        version = payload["schemaVersion"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise SchemaError(f"schemaVersion must be an integer, got {version!r}")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaError(
                f"unsupported schemaVersion {version}; "
                f"this build accepts {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        unknown = sorted(set(payload) - set(_JSON_NAMES))
        if unknown:
            raise SchemaError(f"unknown field(s) for schemaVersion {version}: {unknown}")
        missing = sorted(set(_JSON_NAMES) - set(payload))
        if missing:
            raise SchemaError(f"payload is missing required field(s): {missing}")

        request_timestamp = _require_str(payload["requestTimestamp"], "requestTimestamp")
        try:
            parse_rfc3339(request_timestamp)
        except ValueError as exc:
            raise SchemaError(f"requestTimestamp is not RFC3339: {exc}") from exc

        return cls(
            schema_version=version,
            run_request_id=_require_uuid4(payload["runRequestId"], "runRequestId"),
            intake_id=_require_str(payload["intakeId"], "intakeId"),
            stage3_run_id=_require_str(payload["stage3RunId"], "stage3RunId"),
            gate=_require_enum(Gate, payload["gate"], "gate"),
            mode=_require_enum(RunMode, payload["mode"], "mode"),
            triggered_by=_require_enum(TriggeredBy, payload["triggeredBy"], "triggeredBy"),
            request_timestamp=request_timestamp,
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> Self:
        """Parse the Pub/Sub message body."""
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SchemaError(f"message body is not valid UTF-8 JSON: {exc}") from exc
        return cls.from_wire(decoded)

    # -- chaining -------------------------------------------------------------

    def for_gate(self, gate: Gate, *, triggered_by: TriggeredBy | None = None) -> Self:
        """The same run, aimed at another gate — how the worker chains G1 → G2 → …

        The ``runRequestId`` is deliberately preserved: the chain is one run, and the
        transactional claim on that id is what keeps redeliveries harmless.
        """
        return type(self)(
            schema_version=self.schema_version,
            run_request_id=self.run_request_id,
            intake_id=self.intake_id,
            stage3_run_id=self.stage3_run_id,
            gate=gate,
            mode=self.mode,
            triggered_by=triggered_by or self.triggered_by,
            request_timestamp=self.request_timestamp,
        )
