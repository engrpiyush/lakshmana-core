"""The LK-4 mirror script's driver logic (VA-96).

Only the parts that run without the ``model-prep`` extra are exercised here: selection,
argument guards, the sanity-diff verdict, and the manifest the script would write. The
export itself needs torch and a network, so it is the owner's to run — but everything
around it that could silently mirror the wrong thing is testable, and is tested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from gatekeeper.models.manifest import MANIFEST_FILENAME, Manifest
from gatekeeper.models.roster import ALTERNATES, V1_ROSTER

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prepare_models.py"


def _load_script():
    """Import `scripts/prepare_models.py`, which is a script rather than a package module.

    It has to be registered in ``sys.modules`` before execution: ``@dataclass(slots=True)``
    rebuilds the class and looks its module up by name to resolve annotations.
    """
    spec = importlib.util.spec_from_file_location("prepare_models", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = _load_script()


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "refs": [],
        "roster": "v1",
        "bucket": "",
        "local_dir": "var/models",
        "work_dir": "var/model-prep",
        "revision": None,
        "no_upload": True,
        "clean": False,
        "sanity_pairs": prep.SANITY_PAIRS,
        "pairs": 50,
        "max_mean_abs_diff": prep.DEFAULT_MAX_MEAN_ABS_DIFF,
        "max_abs_diff": prep.DEFAULT_MAX_ABS_DIFF,
        "min_label_agreement": prep.DEFAULT_MIN_LABEL_AGREEMENT,
    }
    return argparse.Namespace(**(defaults | overrides))


# --- selection ---------------------------------------------------------------------------


def test_default_selection_is_the_v1_roster() -> None:
    assert [entry.ref for entry in prep.select(_args())] == [e.ref for e in V1_ROSTER]


def test_alternates_can_be_selected_wholesale() -> None:
    assert [e.ref for e in prep.select(_args(roster="alternates"))] == [e.ref for e in ALTERNATES]


def test_all_selects_both_sets() -> None:
    assert len(prep.select(_args(roster="all"))) == len(V1_ROSTER) + len(ALTERNATES)


def test_explicit_refs_beat_the_roster_flag() -> None:
    selected = prep.select(_args(refs=["minicheck-deberta-l@v1"], roster="all"))
    assert [entry.ref for entry in selected] == ["minicheck-deberta-l@v1"]


def test_an_unknown_ref_exits_two() -> None:
    assert prep.main(["not-a-model@v1", "--no-upload"]) == 2


def test_no_bucket_means_local_mirror_only_not_an_error() -> None:
    """Until the model bucket is applied, mirroring locally is the whole job."""
    parsed = prep.build_parser().parse_args(["--roster", "v1"])
    assert parsed.bucket == ""
    assert parsed.local_dir.endswith("var/models")


def test_mirror_locally_reproduces_the_bucket_layout(tmp_path: Path) -> None:
    """The local mirror and the bucket are the same layout, so the loader cannot tell."""
    staging, mirror = tmp_path / "staged", tmp_path / "mirror"
    staging.mkdir()
    for name in (*prep.REQUIRED_FILES, MANIFEST_FILENAME):
        (staging / name).write_bytes(name.encode())

    destination = prep.mirror_locally(V1_ROSTER[0], staging, mirror)

    assert destination == mirror / "modernbert-base-nli" / "v1"
    assert sorted(p.name for p in destination.iterdir()) == sorted(
        [*prep.REQUIRED_FILES, MANIFEST_FILENAME]
    )


# --- the sanity fixture ------------------------------------------------------------------


def test_the_sanity_fixture_supplies_the_fifty_pairs_the_dod_asks_for() -> None:
    pairs = prep.load_sanity_pairs(prep.SANITY_PAIRS, 50)
    assert len(pairs) == 50
    assert all(pair["premise"] and pair["hypothesis"] for pair in pairs)
    assert len({pair["id"] for pair in pairs}) == 50


def test_the_sanity_fixture_spans_all_three_leanings() -> None:
    """A diff over 50 entailments would not notice a collapsed contradiction head."""
    leanings = {pair["lean"] for pair in prep.load_sanity_pairs(prep.SANITY_PAIRS, 50)}
    assert leanings == {"ENTAIL", "NEUTRAL", "CONTRADICT"}


def test_asking_for_more_pairs_than_exist_is_an_error() -> None:
    with pytest.raises(prep.PrepError, match="wants"):
        prep.load_sanity_pairs(prep.SANITY_PAIRS, 500)


# --- the sanity verdict ------------------------------------------------------------------


def test_a_clean_quantization_passes() -> None:
    report = prep.SanityReport(
        pairs=50, mean_abs_diff=0.004, max_abs_diff=0.05, label_agreement=1.0
    )
    assert report.passed(_args())


def test_a_drifted_mean_fails_even_with_perfect_labels() -> None:
    """Label agreement hides recalibration: the thresholds read probabilities, not argmax."""
    report = prep.SanityReport(pairs=50, mean_abs_diff=0.2, max_abs_diff=0.3, label_agreement=1.0)
    assert not report.passed(_args())


def test_a_single_wild_outlier_fails_even_with_a_tiny_mean() -> None:
    report = prep.SanityReport(pairs=50, mean_abs_diff=0.001, max_abs_diff=0.9, label_agreement=1.0)
    assert not report.passed(_args())


def test_flipped_labels_fail() -> None:
    report = prep.SanityReport(pairs=50, mean_abs_diff=0.01, max_abs_diff=0.1, label_agreement=0.9)
    assert not report.passed(_args())


# --- assembly and manifest ---------------------------------------------------------------


def test_assemble_lays_out_the_gcs_layout(tmp_path: Path) -> None:
    int8, fp32, staging = tmp_path / "int8", tmp_path / "fp32", tmp_path / "staged"
    int8.mkdir()
    fp32.mkdir()
    (int8 / "model_quantized.onnx").write_bytes(b"quantized")
    (int8 / "config.json").write_text("{}")
    (fp32 / "tokenizer.json").write_text("{}")  # optimum leaves it beside the fp32 export

    prep.assemble(V1_ROSTER[0], int8, fp32, staging)

    assert sorted(p.name for p in staging.iterdir()) == [
        "config.json",
        "model.onnx",
        "tokenizer.json",
    ]
    assert (staging / "model.onnx").read_bytes() == b"quantized"


def test_assemble_refuses_a_multi_graph_export(tmp_path: Path) -> None:
    """An encoder-decoder split cannot be represented by the single-file layout."""
    int8, fp32, staging = tmp_path / "int8", tmp_path / "fp32", tmp_path / "staged"
    int8.mkdir()
    fp32.mkdir()
    (int8 / "encoder_model.onnx").write_bytes(b"a")
    (int8 / "decoder_model.onnx").write_bytes(b"b")

    with pytest.raises(prep.PrepError, match="ONNX graphs"):
        prep.assemble(V1_ROSTER[0], int8, fp32, staging)


def test_assemble_refuses_an_export_with_no_graph(tmp_path: Path) -> None:
    int8, fp32, staging = tmp_path / "int8", tmp_path / "fp32", tmp_path / "staged"
    int8.mkdir()
    fp32.mkdir()
    with pytest.raises(prep.PrepError, match=r"no \.onnx"):
        prep.assemble(V1_ROSTER[0], int8, fp32, staging)


def test_the_manifest_the_script_writes_is_the_one_the_loader_reads(tmp_path: Path) -> None:
    """The round trip that makes the config pin meaningful."""
    staging = tmp_path / "staged"
    staging.mkdir()
    (staging / "model.onnx").write_bytes(b"quantized")
    (staging / "tokenizer.json").write_text("{}")
    (staging / "config.json").write_text("{}")

    manifest = prep.build_manifest(V1_ROSTER[0], "a" * 40, staging)
    (staging / MANIFEST_FILENAME).write_bytes(manifest.canonical())

    reparsed = Manifest.from_json(json.loads((staging / MANIFEST_FILENAME).read_bytes()))
    assert reparsed == manifest
    assert reparsed.source_revision == "a" * 40
    assert reparsed.source_checkpoint == V1_ROSTER[0].checkpoint
    assert reparsed.files["model.onnx"].bytes_ == len(b"quantized")


def test_a_custom_export_entry_is_refused_before_any_download() -> None:
    hhem = next(entry for entry in ALTERNATES if entry.name == "hhem")
    with pytest.raises(prep.PrepError, match="CUSTOM"):
        prep.prepare_one(hhem, _args())
