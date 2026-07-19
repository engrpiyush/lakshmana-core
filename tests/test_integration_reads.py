"""Reading vishwamitra's ``Stage3Run`` for the judge mode (LLD §9).

The shape matters more than it looks: ``paramsSnapshot`` is a JSON **string** field in
``stage3_runs``, not a nested map, so a reader that indexes into it finds nothing and
silently falls back. These tests pin the parse and every degraded path, because the
fallback is the one that would let a run be judged in the wrong mode.
"""

from __future__ import annotations

import json
import uuid

import pytest

from gatekeeper.config import load_config
from gatekeeper.enums import JudgeMode
from gatekeeper.integration import resolve_judge_mode

pytestmark = pytest.mark.emulator


@pytest.fixture
def stage3_collection() -> str:
    return f"stage3_runs_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def config(stage3_collection: str):
    return load_config({"GATEKEEPER_STAGE3_RUNS_COLLECTION": stage3_collection})


def _write_run(client, collection: str, params: object) -> str:
    run_id = f"stage3run-{uuid.uuid4().hex[:8]}"
    document = {} if params is None else {"paramsSnapshot": params}
    client.collection(collection).document(run_id).set(document)
    return run_id


def test_judge_mode_is_read_from_the_params_snapshot_string(
    emulator_client, stage3_collection, config
) -> None:
    run_id = _write_run(
        emulator_client, stage3_collection, json.dumps({"ensembleK": 3, "judgeMode": "SHADOW"})
    )

    assert resolve_judge_mode(emulator_client, config, run_id) is JudgeMode.SHADOW


def test_gatekeeper_mode_is_read_back(emulator_client, stage3_collection, config) -> None:
    run_id = _write_run(emulator_client, stage3_collection, json.dumps({"judgeMode": "GATEKEEPER"}))

    assert resolve_judge_mode(emulator_client, config, run_id) is JudgeMode.GATEKEEPER


def test_a_pre_va106_snapshot_falls_back_to_the_default(
    emulator_client, stage3_collection, config
) -> None:
    """Until VA-106 lands, no Stage 3 run carries judgeMode at all."""
    run_id = _write_run(emulator_client, stage3_collection, json.dumps({"ensembleK": 3}))

    assert resolve_judge_mode(emulator_client, config, run_id) is JudgeMode.GATEKEEPER


def test_a_missing_run_doc_falls_back_to_the_default(
    emulator_client, stage3_collection, config
) -> None:
    assert (
        resolve_judge_mode(emulator_client, config, "stage3run-does-not-exist")
        is JudgeMode.GATEKEEPER
    )


@pytest.mark.parametrize("params", [None, "", "{not json", 42])
def test_an_unreadable_snapshot_falls_back_rather_than_raising(
    emulator_client, stage3_collection, config, params
) -> None:
    """A degraded read must not strand the run with an exception mid-claim."""
    run_id = _write_run(emulator_client, stage3_collection, params)

    assert resolve_judge_mode(emulator_client, config, run_id) is JudgeMode.GATEKEEPER


def test_an_unknown_mode_falls_back_to_the_default(
    emulator_client, stage3_collection, config
) -> None:
    run_id = _write_run(
        emulator_client, stage3_collection, json.dumps({"judgeMode": "SOMETHING_NEW"})
    )

    assert resolve_judge_mode(emulator_client, config, run_id) is JudgeMode.GATEKEEPER


def test_llm_mode_is_reported_as_published(emulator_client, stage3_collection, config) -> None:
    """A gatekeeper message for an LLM-mode run is a misroute, but it is reported, not
    rewritten — the split-brain guard lives on the vishwamitra side (LLD §9)."""
    run_id = _write_run(emulator_client, stage3_collection, json.dumps({"judgeMode": "LLM"}))

    assert resolve_judge_mode(emulator_client, config, run_id) is JudgeMode.LLM


def test_the_default_is_configurable(emulator_client, stage3_collection) -> None:
    config = load_config(
        {
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": stage3_collection,
            "GATEKEEPER_JUDGE_MODE_DEFAULT": "SHADOW",
        }
    )
    run_id = _write_run(emulator_client, stage3_collection, json.dumps({"ensembleK": 3}))

    assert resolve_judge_mode(emulator_client, config, run_id) is JudgeMode.SHADOW
