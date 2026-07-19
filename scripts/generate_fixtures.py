#!/usr/bin/env python
"""Regenerate the golden contract fixtures (LLD D-6).

The fixtures in ``contracts/fixtures/`` are the only thing binding the Python and Kotlin
implementations of the contract together — there is no shared code artifact. They are
generated rather than hand-edited so that "the serializer changed" and "the contract
changed" cannot be confused: run this, read the diff, and decide whether that diff was
intended. VA-106's Kotlin contract test pins the same bytes.

Usage:
    uv run python scripts/generate_fixtures.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gatekeeper.config import load_config
from gatekeeper.contracts.payload import SCHEMA_VERSION, GatekeeperRunRequest
from gatekeeper.enums import Gate, GateState, JudgeMode, RunMode, RunState, TriggeredBy
from gatekeeper.runs.model import GatekeeperRun, RunTotals

FIXTURES = Path(__file__).resolve().parent.parent / "contracts" / "fixtures"

# Fixed identifiers: a golden file must not change because a clock or a UUID moved.
RUN_REQUEST_ID = "3f7c2a18-9b4e-4d6a-8c11-5e2f0a7d9b34"
INTAKE_ID = "intake-2026-07-19-reference"
STAGE3_RUN_ID = "stage3run-209claims-001"
REQUEST_TIMESTAMP = "2026-07-19T04:15:00.000Z"

_CREATED_AT = datetime(2026, 7, 19, 4, 15, 2, 500_000, tzinfo=UTC)
_UPDATED_AT = datetime(2026, 7, 19, 4, 27, 44, 125_000, tzinfo=UTC)


def _g1_request() -> GatekeeperRunRequest:
    """The message vishwamitra publishes to start a run."""
    return GatekeeperRunRequest(
        schema_version=SCHEMA_VERSION,
        run_request_id=RUN_REQUEST_ID,
        intake_id=INTAKE_ID,
        stage3_run_id=STAGE3_RUN_ID,
        gate=Gate.G1_NEUTRAL,
        mode=RunMode.FULL,
        triggered_by=TriggeredBy.OPERATOR,
        request_timestamp=REQUEST_TIMESTAMP,
    )


def _g3_retrigger() -> GatekeeperRunRequest:
    """An operator retrigger of one failed gate, keeping the same runRequestId."""
    return GatekeeperRunRequest(
        schema_version=SCHEMA_VERSION,
        run_request_id=RUN_REQUEST_ID,
        intake_id=INTAKE_ID,
        stage3_run_id=STAGE3_RUN_ID,
        gate=Gate.G3_CONTRADICTION,
        mode=RunMode.FROM_GATE,
        triggered_by=TriggeredBy.OPERATOR,
        request_timestamp=REQUEST_TIMESTAMP,
    )


def _run_doc() -> GatekeeperRun:
    """A run mid-flight: G1 committed, G2 leased and running, G3/G4 untouched.

    Counters are the 209-claim reference subject from LLD §16, so the fixture doubles as
    a readable example of what a real G1 commit looks like.
    """
    run = GatekeeperRun(
        schema_version=SCHEMA_VERSION,
        run_request_id=RUN_REQUEST_ID,
        intake_id=INTAKE_ID,
        stage3_run_id=STAGE3_RUN_ID,
        request_timestamp=REQUEST_TIMESTAMP,
        triggered_by=TriggeredBy.OPERATOR,
        judge_mode=JudgeMode.GATEKEEPER,
        mode=RunMode.FULL,
        state=RunState.RUNNING,
        config_snapshot=load_config({}).snapshot(),
        totals=RunTotals(),
        created_at=_CREATED_AT,
        updated_at=_UPDATED_AT,
    )

    g1 = run.gate(Gate.G1_NEUTRAL)
    g1.state = GateState.SUCCEEDED
    g1.attempt = 1
    g1.started_at = datetime(2026, 7, 19, 4, 15, 3, tzinfo=UTC)
    g1.ended_at = datetime(2026, 7, 19, 4, 25, 41, tzinfo=UTC)
    g1.counters = {
        "seen": 12208,
        "neutral": 10450,
        "repeats": 160,
        "contraFlagged": 140,
        "forwarded": 1598,
        "truncated": 0,
        "ctxDisagreed": 0,
    }

    g2 = run.gate(Gate.G2_CORROBORATION)
    g2.state = GateState.RUNNING
    g2.attempt = 1
    g2.lease_owner = "worker/gatekeeper-worker-00001-abc/task-0"
    g2.lease_expires_at = datetime(2026, 7, 19, 5, 57, 44, tzinfo=UTC)
    g2.started_at = datetime(2026, 7, 19, 4, 27, 44, tzinfo=UTC)

    return run


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    written = {
        "gatekeeper_run_request.g1_full.json": _g1_request().to_canonical_json(),
        "gatekeeper_run_request.g3_from_gate.json": _g3_retrigger().to_canonical_json(),
        "gatekeeper_runs.doc.json": _run_doc().to_canonical_json(),
    }
    for name, content in written.items():
        (FIXTURES / name).write_text(content, encoding="utf-8")
        print(f"wrote {FIXTURES.relative_to(Path.cwd()) / name}")


if __name__ == "__main__":
    main()
