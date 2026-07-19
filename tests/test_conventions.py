"""The conventions from LLD §3, enforced from commit one.

Naming discipline is the kind of thing that erodes quietly and then costs a migration,
so the boundary rules are asserted rather than trusted: camelCase never leaks into
Python, snake_case never leaks into stored fields, and every log line carries the two
fields an incident is triaged by.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from io import StringIO

import pytest

from gatekeeper.config import SETTINGS, load_config
from gatekeeper.enums import GATE_ORDER, Gate, JudgeMode, Method, Verdict, next_gate, prior_gate
from gatekeeper.errors import ErrorCode, GatekeeperError, SchemaError
from gatekeeper.logging import configure_logging, get_logger, log_context
from gatekeeper.runs.model import COLLECTION, GatekeeperRun
from gatekeeper.timeutil import format_rfc3339, parse_rfc3339
from gatekeeper.worker.edges import EdgeVerdict
from gatekeeper.worker.queue import COLLECTION as QUEUE_COLLECTION
from gatekeeper.worker.queue import QueuedPair
from tests.conftest import FIXTURES_DIR

CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
KEBAB_KEY = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
SCREAMING_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")


# -- config -------------------------------------------------------------------


def test_every_config_key_is_kebab_case_and_dotted() -> None:
    for setting in SETTINGS:
        assert KEBAB_KEY.match(setting.key), setting.key
        assert setting.key.startswith("gatekeeper.")


def test_every_env_override_is_screaming_snake_with_the_prefix() -> None:
    for setting in SETTINGS:
        assert SCREAMING_SNAKE.match(setting.env), setting.env
        assert setting.env.startswith("GATEKEEPER_")


def test_env_override_names_are_unique() -> None:
    names = [setting.env for setting in SETTINGS]
    assert len(names) == len(set(names))


def test_lld_documented_override_name_is_honoured() -> None:
    """LLD §3 names this exact variable; it is a documented interface."""
    config = load_config({"GATEKEEPER_G1_NEUTRAL_MIN": "0.97"})

    assert config.get_float("gatekeeper.gates.g1.neutral-min") == 0.97


def test_defaults_apply_when_nothing_is_set() -> None:
    config = load_config({})

    assert config.get_float("gatekeeper.gates.g1.neutral-min") == 0.95
    assert config.get_int("gatekeeper.lease.gate-minutes") == 90
    assert config.get_int("gatekeeper.gates.g4.max-llm-pairs") == 3000


def test_a_malformed_override_fails_at_startup() -> None:
    """A typo in a threshold must not silently fall back to the default."""
    with pytest.raises(ValueError, match="GATEKEEPER_G1_NEUTRAL_MIN"):
        load_config({"GATEKEEPER_G1_NEUTRAL_MIN": "very high"})


def test_project_id_falls_back_to_the_google_variable() -> None:
    config = load_config({"GOOGLE_CLOUD_PROJECT": "vishwakarma-prod"})

    assert config.get_str("gatekeeper.firestore.project-id") == "vishwakarma-prod"


def test_config_snapshot_matches_the_lld_shape() -> None:
    snapshot = load_config({}).snapshot()

    assert set(snapshot) == {"rosterVersion", "g1", "g2", "g3", "g4"}
    assert set(snapshot["g1"]) == {
        "model",
        "sha256",
        "neutralMin",
        "repeatMin",
        "contraEscape",
        "maxSeqTokens",
    }
    assert set(snapshot["g2"]) == {
        "model",
        "sha256",
        "supportMin",
        "supportFloor",
        "groundingMode",
    }
    # `neutralConsensus` is named in §8's decide_g3 pseudocode but was never given a key
    # or a proposed value; VA-97 added it as a proposal like the rest of §8's numbers.
    assert set(snapshot["g3"]) == {
        "model",
        "sha256",
        "contraMin",
        "neutralConsensus",
        "agreementRule",
    }
    assert set(snapshot["g4"]) == {"llmModel", "maxLlmPairs", "thinkingBudget"}


def test_config_snapshot_keys_are_camel_case() -> None:
    snapshot = load_config({}).snapshot()

    for group, value in snapshot.items():
        assert CAMEL_CASE.match(group), group
        if isinstance(value, dict):
            for key in value:
                assert CAMEL_CASE.match(key), f"{group}.{key}"


def test_the_secret_is_masked_when_config_is_logged() -> None:
    config = load_config({"GATEKEEPER_NEO4J_PASSWORD": "hunter2"})

    assert config.redacted()["gatekeeper.neo4j.password"] == "***"


def test_an_unset_secret_is_not_masked_into_looking_set() -> None:
    assert load_config({}).redacted()["gatekeeper.neo4j.password"] == ""


# -- storage naming -----------------------------------------------------------


def test_collection_name_is_snake_case() -> None:
    assert COLLECTION == "gatekeeper_runs"


def test_stored_field_names_are_camel_case() -> None:
    """Every key written to Firestore, at every level of the run doc."""
    document = json.loads((FIXTURES_DIR / "gatekeeper_runs.doc.json").read_text())

    def _assert_camel(node, path=""):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            # Gate map keys are enum values, not field names.
            if key in {gate.value for gate in Gate}:
                _assert_camel(value, f"{path}.{key}")
                continue
            assert CAMEL_CASE.match(key), f"{path}.{key} is not camelCase"
            _assert_camel(value, f"{path}.{key}")

    _assert_camel(document)


def test_python_attributes_are_snake_case() -> None:
    """The other half of the boundary rule: no camelCase in Python identifiers."""
    for name in GatekeeperRun.__slots__:
        assert name == name.lower(), name


def test_queue_field_names_are_camel_case() -> None:
    """The pair queue is a second serialization boundary and obeys the same rule."""
    document = QueuedPair(
        pair_id="run|a|b",
        stage3_run_id="run",
        intake_id="intake",
        claim_a_id="a",
        claim_b_id="b",
    ).to_firestore()

    for key in document:
        assert CAMEL_CASE.match(key), f"{key} is not camelCase"


def test_edge_field_names_are_camel_case() -> None:
    """And so does the ``stage3_edges`` row, which another service reads."""
    document = EdgeVerdict(
        claim_a_id="a",
        claim_b_id="b",
        subject_id="subject",
        intake_id="intake",
        stage3_run_id="run",
        run_request_id="req",
        slot="g1",
        method=Method.GK_G1_NLI,
        judge_model="modernbert-base-nli@v1",
        stage_scores={"neuFwd": 0.9},
        verdict=Verdict.NEUTRAL,
    ).to_firestore(JudgeMode.GATEKEEPER)

    def _assert_camel(node, path=""):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            assert CAMEL_CASE.match(key), f"{path}.{key} is not camelCase"
            _assert_camel(value, f"{path}.{key}")

    _assert_camel(document)


def test_the_queue_collection_name_is_snake_case() -> None:
    assert QUEUE_COLLECTION == "gatekeeper_pairs"


# -- enums --------------------------------------------------------------------


def test_gate_order_matches_the_cascade() -> None:
    assert GATE_ORDER == (
        Gate.G1_NEUTRAL,
        Gate.G2_CORROBORATION,
        Gate.G3_CONTRADICTION,
        Gate.G4_ESCALATION,
    )


def test_finalize_is_not_a_gate() -> None:
    """FINALIZE is internal: not published, not a key in the gates map (LLD §7.1, §8)."""
    assert "FINALIZE" not in {gate.value for gate in Gate}


def test_prior_and_next_gate_bookend_the_cascade() -> None:
    assert prior_gate(Gate.G1_NEUTRAL) is None
    assert prior_gate(Gate.G2_CORROBORATION) is Gate.G1_NEUTRAL
    assert next_gate(Gate.G3_CONTRADICTION) is Gate.G4_ESCALATION
    assert next_gate(Gate.G4_ESCALATION) is None


def test_error_codes_all_carry_the_gk_e_prefix() -> None:
    for code in ErrorCode:
        assert code.value.startswith("GK_E_")


def test_every_error_subclass_pins_a_code() -> None:
    for subclass in GatekeeperError.__subclasses__():
        assert isinstance(subclass.code, ErrorCode)


def test_schema_error_maps_to_the_dlq_code() -> None:
    assert SchemaError("bad").code is ErrorCode.GK_E_SCHEMA


# -- timestamps ---------------------------------------------------------------


def test_timestamps_are_rfc3339_utc_with_a_z_suffix() -> None:
    formatted = format_rfc3339(datetime(2026, 7, 19, 4, 15, 0, 500_000, tzinfo=UTC))

    assert formatted == "2026-07-19T04:15:00.500Z"


def test_a_naive_datetime_is_refused() -> None:
    with pytest.raises(ValueError, match="naive"):
        format_rfc3339(datetime(2026, 7, 19, 4, 15))


def test_timestamps_round_trip() -> None:
    original = datetime(2026, 7, 19, 4, 15, 0, 123_000, tzinfo=UTC)

    assert parse_rfc3339(format_rfc3339(original)) == original


def test_a_non_utc_offset_is_normalized() -> None:
    parsed = parse_rfc3339("2026-07-19T09:45:00+05:30")

    assert format_rfc3339(parsed) == "2026-07-19T04:15:00.000Z"


# -- logging ------------------------------------------------------------------


def _capture_logs(emit) -> list[dict]:
    stream = StringIO()
    configure_logging(logging.INFO, stream=stream)
    try:
        emit()
    finally:
        logging.getLogger().handlers.clear()
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_every_line_carries_run_request_id_and_gate() -> None:
    log = get_logger("test")

    lines = _capture_logs(lambda: log.info("something happened"))

    assert lines[0]["runRequestId"] is None
    assert lines[0]["gate"] is None


def test_bound_context_lands_on_the_line() -> None:
    log = get_logger("test")

    def _emit():
        with log_context(runRequestId="abc-123", gate=Gate.G1_NEUTRAL.value):
            log.info("gate claimed")

    lines = _capture_logs(_emit)

    assert lines[0]["runRequestId"] == "abc-123"
    assert lines[0]["gate"] == "G1_NEUTRAL"
    assert lines[0]["message"] == "gate claimed"
    assert lines[0]["severity"] == "INFO"


def test_nested_context_merges_rather_than_replaces() -> None:
    log = get_logger("test")

    def _emit():
        with log_context(runRequestId="abc-123"), log_context(gate="G2_CORROBORATION"):
            log.info("inner")

    lines = _capture_logs(_emit)

    assert lines[0]["runRequestId"] == "abc-123"
    assert lines[0]["gate"] == "G2_CORROBORATION"


def test_context_is_unbound_on_exit() -> None:
    log = get_logger("test")

    def _emit():
        with log_context(runRequestId="abc-123"):
            log.info("inside")
        log.info("outside")

    lines = _capture_logs(_emit)

    assert lines[0]["runRequestId"] == "abc-123"
    assert lines[1]["runRequestId"] is None


def test_structured_fields_are_emitted_alongside_the_message() -> None:
    log = get_logger("test")

    lines = _capture_logs(lambda: log.info("gate committed", fields={"counters": {"seen": 12208}}))

    assert lines[0]["counters"] == {"seen": 12208}


def test_each_record_is_one_json_line() -> None:
    log = get_logger("test")

    def _emit():
        log.info("first")
        log.warning("second")

    lines = _capture_logs(_emit)

    assert len(lines) == 2
    assert lines[1]["severity"] == "WARNING"
