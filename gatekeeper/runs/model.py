"""The ``gatekeeper_runs`` document model (LLD §7.1).

Immutable history: one doc per ``runRequestId``, never deleted, never purged. It answers
"which run produced this verdict", and its counters are what vishwamitra's gate-progress
panel polls — so the field names here are a UI contract as much as a storage one.

This is the serialization boundary: Python stays snake_case, Firestore gets camelCase,
and the mapping lives in exactly one place (LLD §3).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

from gatekeeper.contracts.payload import GatekeeperRunRequest
from gatekeeper.enums import GATE_ORDER, Gate, GateState, JudgeMode, RunMode, RunState, TriggeredBy
from gatekeeper.errors import ErrorCode
from gatekeeper.timeutil import format_rfc3339, parse_rfc3339

__all__ = ["COLLECTION", "GateEntry", "GatekeeperRun", "RunTotals"]

COLLECTION = "gatekeeper_runs"
"""snake_case collection name, matching ``stage3_edges`` (LLD §3)."""


def _encode_timestamp(value: datetime | None) -> str | None:
    return None if value is None else format_rfc3339(value)


def _decode_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return parse_rfc3339(value)


@dataclass(slots=True)
class GateEntry:
    """One gate's slot in the run doc's ``gates`` map (LLD §6, §7.1)."""

    state: GateState = GateState.PENDING
    attempt: int = 0
    sweep_attempt: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    counters: dict[str, Any] = field(default_factory=dict)
    error_code: ErrorCode | None = None
    error_detail: str | None = None

    def lease_held_at(self, now: datetime) -> bool:
        """True while a worker still owns this gate — an unexpired lease blocks claims."""
        return (
            self.state is GateState.RUNNING
            and self.lease_expires_at is not None
            and self.lease_expires_at > now
        )

    def to_firestore(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "attempt": self.attempt,
            "sweepAttempt": self.sweep_attempt,
            "leaseOwner": self.lease_owner,
            "leaseExpiresAt": self.lease_expires_at,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "counters": dict(self.counters),
            "errorCode": self.error_code.value if self.error_code else None,
            "errorDetail": self.error_detail,
        }

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe form (RFC3339 strings) — the golden-fixture shape."""
        payload = self.to_firestore()
        for key in ("leaseExpiresAt", "startedAt", "endedAt"):
            payload[key] = _encode_timestamp(payload[key])
        return payload

    @classmethod
    def from_firestore(cls, data: Any) -> Self:
        data = data or {}
        raw_code = data.get("errorCode")
        return cls(
            state=GateState(data.get("state", GateState.PENDING.value)),
            attempt=int(data.get("attempt", 0)),
            sweep_attempt=int(data.get("sweepAttempt", 0)),
            lease_owner=data.get("leaseOwner"),
            lease_expires_at=_decode_timestamp(data.get("leaseExpiresAt")),
            started_at=_decode_timestamp(data.get("startedAt")),
            ended_at=_decode_timestamp(data.get("endedAt")),
            counters=dict(data.get("counters") or {}),
            error_code=ErrorCode(raw_code) if raw_code else None,
            error_detail=data.get("errorDetail"),
        )


@dataclass(slots=True)
class RunTotals:
    """Run-level rollup written by FINALIZE (LLD §7.1, §12)."""

    pairs_seen: int = 0
    decided_by_gates: int = 0
    escalated_llm: int = 0
    escalated_human: int = 0
    shadow_disagreed: int = 0
    llm_spend_usd: float = 0.0

    def to_firestore(self) -> dict[str, Any]:
        return {
            "pairsSeen": self.pairs_seen,
            "decidedByGates": self.decided_by_gates,
            "escalatedLlm": self.escalated_llm,
            "escalatedHuman": self.escalated_human,
            "shadowDisagreed": self.shadow_disagreed,
            "llmSpendUsd": self.llm_spend_usd,
        }

    @classmethod
    def from_firestore(cls, data: Any) -> Self:
        data = data or {}
        return cls(
            pairs_seen=int(data.get("pairsSeen", 0)),
            decided_by_gates=int(data.get("decidedByGates", 0)),
            escalated_llm=int(data.get("escalatedLlm", 0)),
            escalated_human=int(data.get("escalatedHuman", 0)),
            shadow_disagreed=int(data.get("shadowDisagreed", 0)),
            llm_spend_usd=float(data.get("llmSpendUsd", 0.0)),
        )


def _default_gates() -> dict[Gate, GateEntry]:
    return {gate: GateEntry() for gate in GATE_ORDER}


@dataclass(slots=True)
class GatekeeperRun:
    """One gatekeeper run — doc id is the ``runRequestId``."""

    schema_version: int
    run_request_id: str
    intake_id: str
    stage3_run_id: str
    request_timestamp: str
    triggered_by: TriggeredBy
    judge_mode: JudgeMode
    mode: RunMode
    state: RunState
    config_snapshot: dict[str, Any]
    superseded_by: str | None = None
    gates: dict[Gate, GateEntry] = field(default_factory=_default_gates)
    totals: RunTotals = field(default_factory=RunTotals)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def gate(self, gate: Gate) -> GateEntry:
        """The entry for ``gate``, defaulting to PENDING if the map predates it."""
        return self.gates.setdefault(gate, GateEntry())

    def to_request(self, gate: Gate, *, triggered_by: TriggeredBy) -> GatekeeperRunRequest:
        """Rebuild the Pub/Sub payload that would drive ``gate`` of this run.

        The sweeper republishes work it never saw published, and the only record of the
        original message is this document — which carries every envelope field, because
        the payload is IDs and the doc stored all of them (LLD §5, §7.1). ``mode`` stays
        ``FULL``: a rescue re-enters a gate the state machine already considers reachable,
        which is not the same thing as an operator's FROM_GATE retrigger of failed work.
        """
        return GatekeeperRunRequest(
            schema_version=self.schema_version,
            run_request_id=self.run_request_id,
            intake_id=self.intake_id,
            stage3_run_id=self.stage3_run_id,
            gate=gate,
            mode=RunMode.FULL,
            triggered_by=triggered_by,
            request_timestamp=self.request_timestamp,
        )

    def to_firestore(self) -> dict[str, Any]:
        """Storage form. ``createdAt``/``updatedAt`` are left to the caller's sentinels."""
        return {
            "schemaVersion": self.schema_version,
            "runRequestId": self.run_request_id,
            "intakeId": self.intake_id,
            "stage3RunId": self.stage3_run_id,
            "requestTimestamp": self.request_timestamp,
            "triggeredBy": self.triggered_by.value,
            "judgeMode": self.judge_mode.value,
            "mode": self.mode.value,
            "state": self.state.value,
            "supersededBy": self.superseded_by,
            "configSnapshot": dict(self.config_snapshot),
            "gates": {gate.value: entry.to_firestore() for gate, entry in self.gates.items()},
            "totals": self.totals.to_firestore(),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe form (RFC3339 strings) — the golden-fixture shape."""
        payload = self.to_firestore()
        payload["gates"] = {gate.value: entry.to_wire() for gate, entry in self.gates.items()}
        payload["createdAt"] = _encode_timestamp(self.created_at)
        payload["updatedAt"] = _encode_timestamp(self.updated_at)
        return payload

    def to_canonical_json(self) -> str:
        """The exact byte form stored in ``contracts/fixtures/``."""
        return json.dumps(self.to_wire(), indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def from_firestore(cls, data: Mapping[str, Any]) -> Self:
        gates = {
            gate: GateEntry.from_firestore((data.get("gates") or {}).get(gate.value))
            for gate in GATE_ORDER
        }
        return cls(
            schema_version=int(data["schemaVersion"]),
            run_request_id=str(data["runRequestId"]),
            intake_id=str(data["intakeId"]),
            stage3_run_id=str(data["stage3RunId"]),
            request_timestamp=str(data["requestTimestamp"]),
            triggered_by=TriggeredBy(data["triggeredBy"]),
            judge_mode=JudgeMode(data["judgeMode"]),
            mode=RunMode(data["mode"]),
            state=RunState(data["state"]),
            superseded_by=data.get("supersededBy"),
            config_snapshot=dict(data.get("configSnapshot") or {}),
            gates=gates,
            totals=RunTotals.from_firestore(data.get("totals")),
            created_at=_decode_timestamp(data.get("createdAt")),
            updated_at=_decode_timestamp(data.get("updatedAt")),
        )
