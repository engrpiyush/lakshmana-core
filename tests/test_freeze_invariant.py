"""The per-run freeze invariant (LLD §5, owner requirement 2026-07-19).

One rule, proven from three sides:

    The model *and* the thresholds are frozen into ``configSnapshot`` when the run doc is
    created, and every gate and every FROM_GATE resume reads that snapshot. Only a
    FROM_START — a new ``runRequestId``, a new run doc — may pick up new config.

The threshold half of this has been true since session01 because
:meth:`Thresholds.from_snapshot` is the only way a gate gets a number. The model half was
not: ``artifact_for_gate`` read live config, so an operator who edited
``GATEKEEPER_G1_MODEL`` between G1 and G3 would have split a run across two checkpoints
while its thresholds stayed put — a mixed-calibration run that looks clean on the doc.
The tests below are what stops that from coming back.

Why it matters more than it sounds: sessions 04-06 build the gates against a roster
nobody has calibrated yet (DEFERRED-LIVE 18-19), so config *will* move under running
code. A run that silently re-binds mid-cascade would make the eventual calibration
unreproducible — you could not tell which artifact produced which verdict.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gatekeeper.config import GateBinding, load_config
from gatekeeper.enums import Gate, JudgeMode, RunMode
from gatekeeper.errors import ErrorCode
from gatekeeper.gates.decisions import Thresholds
from gatekeeper.models.loader import ModelLoader, artifact_for_binding
from tests.test_models import FakeBlobStore, build_artifact_objects

# --- resolution: the snapshot wins over live config --------------------------------------


def test_the_binding_comes_from_the_snapshot_not_the_environment() -> None:
    """The frozen model is used even when the environment names a different one."""
    frozen = load_config(env={}).snapshot()

    # The operator edits config mid-run. The snapshot on the doc is untouched.
    load_config(env={"GATEKEEPER_G1_MODEL": "ettinx-nli-s@v1"})

    binding = GateBinding.from_snapshot(frozen, Gate.G1_NEUTRAL)

    assert binding.model == "modernbert-base-nli@v1"


def test_a_snapshot_frozen_on_an_alternate_keeps_the_alternate() -> None:
    """A run created while config pointed at an alternate stays on it.

    This is the direction that matters after the bake-off resumes: once the owner
    switches the default, in-flight runs must not jump to the new model.
    """
    frozen = load_config(env={"GATEKEEPER_G3_MODEL": "vitaminc-mnli@v1"}).snapshot()

    binding = GateBinding.from_snapshot(frozen, Gate.G3_CONTRADICTION)

    assert binding.model == "vitaminc-mnli@v1"


def test_the_token_budget_is_frozen_alongside_the_model() -> None:
    """Truncation changes scores, so the budget is part of the calibration."""
    frozen = load_config(env={"GATEKEEPER_G1_MAX_SEQ_TOKENS": "256"}).snapshot()

    assert GateBinding.from_snapshot(frozen, Gate.G1_NEUTRAL).max_seq_tokens == 256
    # G2 and G3 do not pin one; they inherit the documented default rather than G1's.
    assert GateBinding.from_snapshot(frozen, Gate.G2_CORROBORATION).max_seq_tokens == 512


def test_the_pin_is_frozen_too() -> None:
    """A sha edited mid-run must not turn a pinned load into an unpinned one."""
    frozen = load_config(env={"GATEKEEPER_G2_SHA256": "a" * 64}).snapshot()

    assert GateBinding.from_snapshot(frozen, Gate.G2_CORROBORATION).sha256 == "a" * 64


def test_every_encoder_gate_resolves_and_g4_refuses() -> None:
    snapshot = load_config(env={}).snapshot()

    for gate in (Gate.G1_NEUTRAL, Gate.G2_CORROBORATION, Gate.G3_CONTRADICTION):
        assert GateBinding.from_snapshot(snapshot, gate).model

    with pytest.raises(ValueError, match="escalates to Vertex"):
        GateBinding.from_snapshot(snapshot, Gate.G4_ESCALATION)


def test_an_older_snapshot_missing_a_key_falls_back_to_the_default() -> None:
    """A run doc frozen before a slot gained a key still resolves.

    Failing mid-cascade on a doc that is merely old would turn a schema addition into an
    outage for every in-flight run.
    """
    binding = GateBinding.from_snapshot({"g1": {}}, Gate.G1_NEUTRAL)

    assert binding.model == "modernbert-base-nli@v1"
    assert binding.max_seq_tokens == 512
    assert binding.sha256 == ""


def test_defaults_are_the_v1_roster_and_the_lld_thresholds() -> None:
    """Closeout item 1: the shipped defaults are the v1 roster + §8 proposals."""
    snapshot = load_config(env={}).snapshot()

    assert GateBinding.from_snapshot(snapshot, Gate.G1_NEUTRAL).model == "modernbert-base-nli@v1"
    assert GateBinding.from_snapshot(snapshot, Gate.G2_CORROBORATION).model == (
        "minicheck-deberta-l@v1"
    )
    assert GateBinding.from_snapshot(snapshot, Gate.G3_CONTRADICTION).model == (
        "deberta-mnli-fever-anli@v1"
    )

    thresholds = Thresholds.from_snapshot(snapshot)
    assert thresholds.neutral_min == 0.95
    assert thresholds.support_min == 0.90
    assert thresholds.contra_min == 0.85


# --- loading: the frozen binding is what actually reaches the loader ----------------------


def test_the_loader_fetches_the_frozen_artifact(tmp_path: Path) -> None:
    """End to end: snapshot → binding → loader, with config pointing somewhere else."""
    objects, manifest = build_artifact_objects()
    loader = ModelLoader(FakeBlobStore(objects), cache_dir=tmp_path)
    frozen = load_config(env={"GATEKEEPER_G1_SHA256": manifest.digest()}).snapshot()

    load_config(env={"GATEKEEPER_G1_MODEL": "nli-deberta-v3-small@v1"})

    artifact = artifact_for_binding(loader, GateBinding.from_snapshot(frozen, Gate.G1_NEUTRAL))

    assert artifact.ref.ref == "modernbert-base-nli@v1"


# --- lifecycle: the snapshot survives a resume --------------------------------------------


@pytest.mark.emulator
def test_a_from_gate_resume_reuses_the_original_snapshot(store, make_request, config_snapshot):
    """The retrigger carries a *different* snapshot; the run keeps the one it was born with."""
    request = make_request()
    assert store.claim_gate(
        request,
        lease_owner="owner-a",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    ).claimed
    assert store.commit_gate(request.run_request_id, Gate.G1_NEUTRAL, lease_owner="owner-a")

    # Between G1 and G2 the owner adopts a different roster.
    rerostered = load_config(env={"GATEKEEPER_G2_MODEL": "factcg-deberta-l@v1"}).snapshot()
    g2 = request.for_gate(Gate.G2_CORROBORATION)
    assert store.claim_gate(
        g2,
        lease_owner="owner-a",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=rerostered,
    ).claimed

    run = store.get(request.run_request_id)
    binding = GateBinding.from_snapshot(run.config_snapshot, Gate.G2_CORROBORATION)
    assert binding.model == "minicheck-deberta-l@v1"

    # And the same holds through a FROM_GATE retrigger of that gate.
    assert store.fail_gate(
        request.run_request_id,
        Gate.G2_CORROBORATION,
        lease_owner="owner-a",
        error_code=ErrorCode.GK_E_MODEL_FETCH,
        error_detail="artifact missing",
    )
    assert store.claim_gate(
        replace(g2, mode=RunMode.FROM_GATE),
        lease_owner="owner-b",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=rerostered,
    ).claimed

    resumed = store.get(request.run_request_id)
    assert resumed.config_snapshot == config_snapshot
    assert (
        GateBinding.from_snapshot(resumed.config_snapshot, Gate.G2_CORROBORATION).model
        == "minicheck-deberta-l@v1"
    )


@pytest.mark.emulator
def test_a_from_start_run_picks_up_the_new_config(store, make_request, config_snapshot):
    """The one door that is open: a full restart from vishwamitra, new runRequestId."""
    first = make_request()
    assert store.claim_gate(
        first,
        lease_owner="owner-a",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=config_snapshot,
    ).claimed

    rerostered = load_config(env={"GATEKEEPER_G1_MODEL": "ettinx-nli-s@v1"}).snapshot()
    restart = make_request(stage3_run_id=first.stage3_run_id)
    assert store.claim_gate(
        restart,
        lease_owner="owner-b",
        judge_mode=JudgeMode.GATEKEEPER,
        config_snapshot=rerostered,
    ).claimed

    fresh = store.get(restart.run_request_id)
    assert GateBinding.from_snapshot(fresh.config_snapshot, Gate.G1_NEUTRAL).model == (
        "ettinx-nli-s@v1"
    )
    # The predecessor is untouched — it keeps the calibration it ran under.
    superseded = store.get(first.run_request_id)
    assert GateBinding.from_snapshot(superseded.config_snapshot, Gate.G1_NEUTRAL).model == (
        "modernbert-base-nli@v1"
    )
