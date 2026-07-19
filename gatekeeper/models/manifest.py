"""The artifact manifest — the pin that makes ``GK_E_MODEL_FETCH`` mean something.

The integrity chain has two links, and the split is deliberate:

1. **Config pins the manifest.** ``gatekeeper.gates.g1.sha256`` is the sha256 of the
   manifest's canonical bytes — one short value per gate, small enough to live in a
   config table and to appear in ``configSnapshot`` on every run doc.
2. **The manifest pins the files.** Each of ``model.onnx`` / ``tokenizer.json`` /
   ``config.json`` carries its own digest and size, per mirrored precision.

So a single value in config transitively covers hundreds of megabytes, and an auditor
reading a six-month-old run doc can prove which bytes produced its verdicts. Verifying
the manifest without pinning it would be theatre: an attacker who can rewrite the model
can rewrite the manifest beside it.

**Precision (schemaVersion 2).** An artifact mirrors fp32 and INT8 side by side and names
one of them as the *shipping* precision — the graph the loader materializes and the gates
actually run. INT8 earns that slot only by passing the export-time sanity diff against its
own fp32 parent; where it does not, fp32 ships and the failing numbers stay in the
manifest rather than being thrown away. That keeps a demotion auditable: the run doc pins
the manifest, and the manifest says both which precision produced the verdicts and what
the rejected alternative measured.

Canonical bytes are ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` + a
trailing newline — fixed here rather than left to the caller, because "the digest of the
manifest" has to mean exactly one byte sequence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "REQUIRED_FILES",
    "Manifest",
    "ManifestFile",
    "Precision",
    "SanityGate",
    "SanityReport",
    "canonical_bytes",
    "sha256_bytes",
    "sha256_file",
]

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 2

REQUIRED_FILES: tuple[str, ...] = ("model.onnx", "tokenizer.json", "config.json")
"""The per-precision layout from LLD §13. A precision missing one of these is malformed."""

_CHUNK = 1024 * 1024


class Precision(StrEnum):
    """Numeric precision of a mirrored ONNX graph.

    Lowercase because these are path segments in the bucket (``<name>/<version>/int8/…``),
    not wire enums.
    """

    FP32 = "fp32"
    INT8 = "int8"


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks so a 435 MB model does not land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Hex sha256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """The one byte sequence a manifest digest is taken over."""
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


@dataclass(frozen=True, slots=True)
class ManifestFile:
    """One mirrored file."""

    sha256: str
    bytes_: int

    def to_json(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes_}

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> ManifestFile:
        return cls(sha256=str(document["sha256"]), bytes_=int(document["bytes"]))


@dataclass(frozen=True, slots=True)
class SanityGate:
    """The thresholds an INT8 export had to clear to ship.

    Recorded alongside the measurement because "it passed" is only meaningful next to
    what it had to pass: a later session that loosens the gate must not be able to make
    an old demotion look like an old promotion.
    """

    max_mean_abs_diff: float
    max_abs_diff: float
    min_label_agreement: float

    def to_json(self) -> dict[str, Any]:
        return {
            "maxMeanAbsDiff": self.max_mean_abs_diff,
            "maxAbsDiff": self.max_abs_diff,
            "minLabelAgreement": self.min_label_agreement,
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> SanityGate:
        return cls(
            max_mean_abs_diff=float(document["maxMeanAbsDiff"]),
            max_abs_diff=float(document["maxAbsDiff"]),
            min_label_agreement=float(document["minLabelAgreement"]),
        )


@dataclass(frozen=True, slots=True)
class SanityReport:
    """The fp32-vs-INT8 comparison that decides which precision ships."""

    pairs: int
    mean_abs_diff: float
    max_abs_diff: float
    label_agreement: float
    gate: SanityGate

    @property
    def passed(self) -> bool:
        return (
            self.mean_abs_diff <= self.gate.max_mean_abs_diff
            and self.max_abs_diff <= self.gate.max_abs_diff
            and self.label_agreement >= self.gate.min_label_agreement
        )

    def failures(self) -> tuple[str, ...]:
        """Which specific criteria the export missed — the message an operator needs."""
        missed: list[str] = []
        if self.mean_abs_diff > self.gate.max_mean_abs_diff:
            missed.append(f"mean |Δp| {self.mean_abs_diff:.4f} > {self.gate.max_mean_abs_diff}")
        if self.max_abs_diff > self.gate.max_abs_diff:
            missed.append(f"max |Δp| {self.max_abs_diff:.4f} > {self.gate.max_abs_diff}")
        if self.label_agreement < self.gate.min_label_agreement:
            missed.append(
                f"label agreement {self.label_agreement:.2%} < {self.gate.min_label_agreement:.2%}"
            )
        return tuple(missed)

    def describe(self) -> str:
        return (
            f"{self.pairs} pairs · mean |Δp| {self.mean_abs_diff:.4f} · "
            f"max |Δp| {self.max_abs_diff:.4f} · label agreement {self.label_agreement:.2%}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "meanAbsDiff": self.mean_abs_diff,
            "maxAbsDiff": self.max_abs_diff,
            "labelAgreement": self.label_agreement,
            "passed": self.passed,
            "gate": self.gate.to_json(),
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> SanityReport:
        return cls(
            pairs=int(document["pairs"]),
            mean_abs_diff=float(document["meanAbsDiff"]),
            max_abs_diff=float(document["maxAbsDiff"]),
            label_agreement=float(document["labelAgreement"]),
            gate=SanityGate.from_json(document["gate"]),
        )


@dataclass(frozen=True, slots=True)
class Manifest:
    """What the prep script writes and the loader verifies.

    Field names are camelCase on the wire like every other stored gatekeeper document
    (LLD §3), snake_case in Python — the conversion happens here and nowhere else.
    """

    schema_version: int
    name: str
    version: str
    source_checkpoint: str
    source_revision: str
    task: str
    quantization: str
    exported_at: str
    precision: Precision
    """The shipping precision: the graph the loader materializes and the gates run."""

    precisions: dict[Precision, dict[str, ManifestFile]]
    """Every mirrored precision, each with its own digests. Always contains ``precision``."""

    sanity: SanityReport | None = None
    """The fp32-vs-INT8 diff. ``None`` when INT8 was never produced at all."""

    opset: int | None = None

    @property
    def shipping_files(self) -> dict[str, ManifestFile]:
        """Digests for the precision the loader will actually fetch."""
        return self.precisions[self.precision]

    def object_path(self, filename: str, precision: Precision | None = None) -> str:
        """Bucket-relative path of a mirrored file, below the artifact prefix."""
        return f"{precision or self.precision}/{filename}"

    def to_json(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schemaVersion": self.schema_version,
            "name": self.name,
            "version": self.version,
            "sourceCheckpoint": self.source_checkpoint,
            "sourceRevision": self.source_revision,
            "task": self.task,
            "quantization": self.quantization,
            "exportedAt": self.exported_at,
            "precision": str(self.precision),
            "precisions": {
                str(precision): {name: entry.to_json() for name, entry in sorted(files.items())}
                for precision, files in sorted(self.precisions.items())
            },
        }
        if self.sanity is not None:
            document["sanity"] = self.sanity.to_json()
        if self.opset is not None:
            document["opset"] = self.opset
        return document

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> Manifest:
        """Parse a manifest.

        Raises:
            ValueError: on an unknown ``schemaVersion``, an unknown precision, or a
                missing required file. An unreadable manifest is a fetch failure, not
                something to work around — the caller turns this into ``GK_E_MODEL_FETCH``.
        """
        try:
            schema_version = int(document["schemaVersion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("manifest has no usable schemaVersion") from exc

        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schemaVersion {schema_version}; "
                f"this build understands {MANIFEST_SCHEMA_VERSION}"
            )

        try:
            precisions = {
                Precision(name): {
                    filename: ManifestFile.from_json(entry)
                    for filename, entry in dict(files).items()
                }
                for name, files in dict(document["precisions"]).items()
            }
            parsed = cls(
                schema_version=schema_version,
                name=str(document["name"]),
                version=str(document["version"]),
                source_checkpoint=str(document["sourceCheckpoint"]),
                source_revision=str(document["sourceRevision"]),
                task=str(document["task"]),
                quantization=str(document["quantization"]),
                exported_at=str(document["exportedAt"]),
                precision=Precision(str(document["precision"])),
                precisions=precisions,
                sanity=(
                    SanityReport.from_json(document["sanity"]) if "sanity" in document else None
                ),
                opset=int(document["opset"]) if "opset" in document else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed manifest: {exc}") from exc

        if parsed.precision not in parsed.precisions:
            raise ValueError(
                f"manifest ships {parsed.precision} but mirrors only "
                f"{', '.join(sorted(str(p) for p in parsed.precisions))}"
            )
        for precision, files in parsed.precisions.items():
            missing = [name for name in REQUIRED_FILES if name not in files]
            if missing:
                raise ValueError(
                    f"manifest precision {precision} is missing required files: "
                    f"{', '.join(missing)}"
                )

        return parsed

    def canonical(self) -> bytes:
        """The exact bytes uploaded as ``manifest.json`` and digested into config."""
        return canonical_bytes(self.to_json())

    def digest(self) -> str:
        """The sha256 a gate's ``sha256`` config value must equal."""
        return sha256_bytes(self.canonical())
