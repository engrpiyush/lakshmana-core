#!/usr/bin/env python3
"""Mirror a roster checkpoint into the model bucket (LK-4 / VA-96).

    Hugging Face → ONNX (fp32) → dynamic INT8 → sanity diff → GCS

**This is the only place in the system allowed to talk to Hugging Face**, and it is run
by the owner, never by the runtime (CLAUDE.md rule 5). The worker reads the bucket and
nothing else.

The step that earns its keep is the sanity diff. Dynamic INT8 quantization is not
value-preserving, and a quantized encoder that has quietly lost its calibration does not
crash — it returns confident, plausible, wrong probabilities, which in this system means
pairs silently discarded as NEUTRAL. So every export is scored against its own fp32
parent on a fixed 50-pair fixture before it is allowed near the bucket.

Usage::

    uv run --extra model-prep python scripts/prepare_models.py modernbert-base-nli@v1
    uv run --extra model-prep python scripts/prepare_models.py --roster v1 --bucket X
    uv run --extra model-prep python scripts/prepare_models.py --roster all --no-upload

The heavy dependencies (torch, transformers, optimum, onnxruntime) live in the
``model-prep`` extra so neither the runtime image nor CI carries them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatekeeper.models.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_FILES,
    Manifest,
    ManifestFile,
    sha256_file,
)
from gatekeeper.models.roster import (
    ALTERNATES,
    V1_ROSTER,
    ExportKind,
    ModelTask,
    RosterEntry,
    entry_for,
)

if TYPE_CHECKING:
    import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SANITY_PAIRS = REPO_ROOT / "contracts" / "fixtures" / "model_sanity_pairs.jsonl"

DEFAULT_LOCAL_MIRROR = REPO_ROOT / "var" / "models"
"""Interim source of truth until the bucket exists (`execution-plan/DEFERRED-LIVE.md`).

``var/`` is gitignored, so mirroring here never puts half a gigabyte near a commit.
"""

DEFAULT_MAX_MEAN_ABS_DIFF = 0.02
"""Mean absolute probability drift allowed between fp32 and INT8."""

DEFAULT_MAX_ABS_DIFF = 0.15
"""Worst single-probability drift allowed. A handful of borderline pairs may move; a
systematic recalibration will blow through this long before the mean does."""

DEFAULT_MIN_LABEL_AGREEMENT = 0.98
"""Fraction of fixture pairs whose argmax label must survive quantization."""

ONNX_OPSET = 17

GROUNDING_DECISION_BOUNDARY = 0.5
"""Where a single-logit grounding head flips supported/unsupported.

Only ever used to measure *agreement between fp32 and INT8* — it is not G2's
``supportMin``, which is a calibrated threshold the bake-off sets (LLD §8).
"""


class PrepError(RuntimeError):
    """Anything that should stop a mirror run with a legible message."""


@dataclass(slots=True)
class SanityReport:
    """The fp32-vs-INT8 comparison for one artifact."""

    pairs: int
    mean_abs_diff: float
    max_abs_diff: float
    label_agreement: float

    def passed(self, args: argparse.Namespace) -> bool:
        return (
            self.mean_abs_diff <= args.max_mean_abs_diff
            and self.max_abs_diff <= args.max_abs_diff
            and self.label_agreement >= args.min_label_agreement
        )

    def describe(self) -> str:
        return (
            f"{self.pairs} pairs · mean |Δp| {self.mean_abs_diff:.4f} · "
            f"max |Δp| {self.max_abs_diff:.4f} · label agreement {self.label_agreement:.2%}"
        )


# --- steps ------------------------------------------------------------------------------


def resolve_revision(entry: RosterEntry, requested: str | None) -> str:
    """Pin the upstream commit sha.

    A branch name is not a pin: ``main`` today and ``main`` next month are different
    weights, and the manifest is supposed to make a run reproducible.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(entry.checkpoint, revision=requested or "main")
    if not info.sha:
        raise PrepError(f"could not resolve a commit sha for {entry.checkpoint}")
    return str(info.sha)


def export_fp32(entry: RosterEntry, revision: str, destination: Path) -> None:
    """Export the checkpoint to ONNX at fp32, tokenizer and config alongside."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    print(f"  exporting fp32 ONNX (opset {ONNX_OPSET})…")
    model = ORTModelForSequenceClassification.from_pretrained(
        entry.checkpoint, revision=revision, export=True
    )
    model.save_pretrained(destination)

    tokenizer = AutoTokenizer.from_pretrained(entry.checkpoint, revision=revision, use_fast=True)
    tokenizer.save_pretrained(destination)

    if not (destination / "tokenizer.json").is_file():
        raise PrepError(
            f"{entry.checkpoint} did not produce a fast tokenizer.json. The loader's GCS "
            "layout requires one; a slow-only tokenizer needs a hand-written conversion."
        )


def quantize_int8(source: Path, destination: Path) -> None:
    """Dynamic INT8 quantization — no calibration set, weights only."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    print("  quantizing to dynamic INT8…")
    quantizer = ORTQuantizer.from_pretrained(source)
    # avx512_vnni matches Cloud Run's CPU floor; per-channel keeps the per-output-channel
    # scales that large DeBERTa heads are sensitive to.
    config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=destination, quantization_config=config)


def load_sanity_pairs(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise PrepError(f"sanity fixture not found: {path}")
    pairs: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                pairs.append(json.loads(line))
    if len(pairs) < limit:
        raise PrepError(f"{path} has {len(pairs)} pairs; the sanity diff wants {limit}")
    return pairs[:limit]


def _score(model_dir: Path, pairs: list[dict[str, str]], task: ModelTask) -> np.ndarray:
    """Probabilities for every fixture pair from the ONNX model in ``model_dir``."""
    import numpy as np
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ORTModelForSequenceClassification.from_pretrained(model_dir)

    # Grounding models take (document, claim); NLI takes (premise, hypothesis). Same
    # tensor shape, different semantics — the fixture supplies both halves either way.
    first = [pair["premise"] for pair in pairs]
    second = [pair["hypothesis"] for pair in pairs]

    encoded = tokenizer(
        first, second, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    logits = model(**encoded).logits.detach().numpy()

    if task is ModelTask.GROUNDING and logits.shape[-1] == 1:
        return 1.0 / (1.0 + np.exp(-logits))
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def sanity_diff(
    fp32_dir: Path, int8_dir: Path, pairs: list[dict[str, str]], task: ModelTask
) -> SanityReport:
    """Compare INT8 against its own fp32 parent — the gate on every export."""
    import numpy as np

    print(f"  scoring {len(pairs)} fixture pairs on fp32 and INT8…")
    reference = _score(fp32_dir, pairs, task)
    candidate = _score(int8_dir, pairs, task)

    if reference.shape != candidate.shape:
        raise PrepError(
            f"output shape changed under quantization: {reference.shape} → {candidate.shape}"
        )

    difference = np.abs(reference - candidate)
    agreement = float(
        (reference.argmax(axis=-1) == candidate.argmax(axis=-1)).mean()
        if reference.shape[-1] > 1
        else (
            (reference >= GROUNDING_DECISION_BOUNDARY) == (candidate >= GROUNDING_DECISION_BOUNDARY)
        ).mean()
    )
    return SanityReport(
        pairs=len(pairs),
        mean_abs_diff=float(difference.mean()),
        max_abs_diff=float(difference.max()),
        label_agreement=agreement,
    )


def assemble(entry: RosterEntry, int8_dir: Path, fp32_dir: Path, staging: Path) -> None:
    """Lay out exactly the four files the loader expects (LLD §13)."""
    staging.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(int8_dir.glob("*.onnx"))
    if not onnx_files:
        raise PrepError(f"no .onnx produced in {int8_dir}")
    if len(onnx_files) > 1:
        raise PrepError(
            f"{entry.ref} exported {len(onnx_files)} ONNX graphs "
            f"({', '.join(f.name for f in onnx_files)}); the single-graph layout cannot "
            "represent an encoder-decoder split."
        )
    shutil.copy2(onnx_files[0], staging / "model.onnx")

    for filename in ("tokenizer.json", "config.json"):
        source = int8_dir / filename
        if not source.is_file():
            source = fp32_dir / filename
        if not source.is_file():
            raise PrepError(f"{filename} missing from both {int8_dir} and {fp32_dir}")
        shutil.copy2(source, staging / filename)


def build_manifest(entry: RosterEntry, revision: str, staging: Path) -> Manifest:
    files = {
        name: ManifestFile(
            sha256=sha256_file(staging / name), bytes_=(staging / name).stat().st_size
        )
        for name in REQUIRED_FILES
    }
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=entry.name,
        version=entry.version,
        source_checkpoint=entry.checkpoint,
        source_revision=revision,
        task=entry.task.value,
        quantization="dynamic-int8-avx512-vnni-per-channel",
        exported_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        files=files,
        opset=ONNX_OPSET,
    )


def upload(entry: RosterEntry, staging: Path, bucket: str) -> None:
    """Copy the four files to ``gs://<bucket>/<name>/<version>/``.

    Shells out to ``gcloud storage`` rather than pulling in the SDK: this script already
    carries a heavy optional dependency set, and the owner running it is authenticated.
    The manifest goes **last** — until it exists, the loader treats the prefix as absent,
    so a half-finished upload is invisible rather than corrupt.
    """
    ordered = [name for name in REQUIRED_FILES] + [MANIFEST_FILENAME]
    for name in ordered:
        target = f"gs://{bucket}/{entry.prefix}/{name}"
        print(f"  → {target}")
        result = subprocess.run(
            ["gcloud", "storage", "cp", str(staging / name), target],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise PrepError(f"upload of {name} failed: {result.stderr.strip()}")


def mirror_locally(entry: RosterEntry, staging: Path, local_dir: Path) -> Path:
    """Publish into the local mirror the loader's local-dir mode reads.

    Same layout as the bucket, so ``GATEKEEPER_MODELS_LOCAL_DIR`` is a drop-in for
    ``GATEKEEPER_MODELS_BUCKET`` and the replay bake-off exercises the production loader
    rather than a test seam. The manifest is written last, matching the upload order:
    until it exists the loader treats the prefix as absent.
    """
    destination = local_dir / entry.prefix
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_FILES, MANIFEST_FILENAME):
        shutil.copy2(staging / name, destination / name)
    print(f"  local mirror → {destination}")
    return destination


# --- driver -----------------------------------------------------------------------------


def prepare_one(entry: RosterEntry, args: argparse.Namespace) -> Manifest:
    print(f"\n=== {entry.ref} ← {entry.checkpoint} ===")

    if entry.export is ExportKind.CUSTOM:
        raise PrepError(
            f"{entry.ref} is marked {ExportKind.CUSTOM.value}: {entry.notes} "
            "Refusing to export it through the standard path."
        )

    work = Path(args.work_dir) / entry.name / entry.version
    fp32_dir, int8_dir, staging = work / "fp32", work / "int8", work / "staged"
    if work.exists() and args.clean:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    revision = resolve_revision(entry, args.revision)
    print(f"  pinned revision {revision}")

    export_fp32(entry, revision, fp32_dir)
    quantize_int8(fp32_dir, int8_dir)

    pairs = load_sanity_pairs(args.sanity_pairs, args.pairs)
    report = sanity_diff(fp32_dir, int8_dir, pairs, entry.task)
    print(f"  sanity: {report.describe()}")
    if not report.passed(args):
        raise PrepError(
            f"{entry.ref} failed the INT8 sanity diff ({report.describe()}). "
            "The export is not trustworthy — do not mirror it."
        )

    assemble(entry, int8_dir, fp32_dir, staging)
    manifest = build_manifest(entry, revision, staging)
    (staging / MANIFEST_FILENAME).write_bytes(manifest.canonical())

    size_mb = (staging / "model.onnx").stat().st_size / 1_048_576
    print(f"  model.onnx {size_mb:.1f} MB (roster estimate {entry.approx_int8_mb} MB)")

    mirror_locally(entry, staging, Path(args.local_dir))
    if args.no_upload:
        print("  --no-upload: GCS upload deferred (execution-plan/DEFERRED-LIVE.md item 2)")
    else:
        upload(entry, staging, args.bucket)

    print(f"  manifest sha256 {manifest.digest()}")
    return manifest


def select(args: argparse.Namespace) -> list[RosterEntry]:
    if args.refs:
        return [entry_for(ref) for ref in args.refs]
    if args.roster == "v1":
        return list(V1_ROSTER)
    if args.roster == "alternates":
        return list(ALTERNATES)
    return [*V1_ROSTER, *ALTERNATES]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mirror roster checkpoints to the gatekeeper model bucket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("refs", nargs="*", help="artifact refs, e.g. modernbert-base-nli@v1")
    parser.add_argument(
        "--roster",
        choices=("v1", "alternates", "all"),
        default="v1",
        help="which set to mirror when no explicit refs are given (default: v1)",
    )
    parser.add_argument("--bucket", default="", help="destination GCS bucket")
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_MIRROR),
        help="local mirror, always written (default: var/models). Point "
        "GATEKEEPER_MODELS_LOCAL_DIR at it to run gates without a bucket.",
    )
    parser.add_argument(
        "--work-dir", default="var/model-prep", help="scratch directory (default: var/model-prep)"
    )
    parser.add_argument("--revision", default=None, help="override the upstream revision to pin")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="skip GCS even when --bucket is given (upload is already skipped without one)",
    )
    parser.add_argument("--clean", action="store_true", help="wipe the scratch dir per artifact")
    parser.add_argument("--sanity-pairs", type=Path, default=SANITY_PAIRS)
    parser.add_argument("--pairs", type=int, default=50, help="fixture pairs to diff (default: 50)")
    parser.add_argument("--max-mean-abs-diff", type=float, default=DEFAULT_MAX_MEAN_ABS_DIFF)
    parser.add_argument("--max-abs-diff", type=float, default=DEFAULT_MAX_ABS_DIFF)
    parser.add_argument("--min-label-agreement", type=float, default=DEFAULT_MIN_LABEL_AGREEMENT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The local mirror is unconditional and the bucket is opt-in: until the model bucket
    # is applied, mirroring locally is the whole job (DEFERRED-LIVE.md item 2).
    args.no_upload = args.no_upload or not args.bucket

    try:
        entries = select(args)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mirrored: list[tuple[RosterEntry, Manifest]] = []
    failed: list[tuple[RosterEntry, str]] = []
    for entry in entries:
        try:
            mirrored.append((entry, prepare_one(entry, args)))
        except PrepError as exc:
            # One bad checkpoint should not abandon the rest of the roster — the whole
            # point of mirroring the alternates up front is to have them ready.
            print(f"  SKIPPED: {exc}", file=sys.stderr)
            failed.append((entry, str(exc)))

    print("\n=== summary ===")
    for entry, manifest in mirrored:
        print(f"  OK      {entry.ref:28} {manifest.digest()}")
    for entry, reason in failed:
        print(f"  FAILED  {entry.ref:28} {reason.splitlines()[0]}")

    if mirrored:
        print("\nPin these in gatekeeper/config.py (or as GATEKEEPER_G*_SHA256):")
        for entry, manifest in mirrored:
            print(f"  {entry.ref} → {manifest.digest()}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
