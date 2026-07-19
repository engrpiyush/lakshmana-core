"""The artifact manifest — the pin that makes ``GK_E_MODEL_FETCH`` mean something.

The integrity chain has two links, and the split is deliberate:

1. **Config pins the manifest.** ``gatekeeper.gates.g1.sha256`` is the sha256 of the
   manifest's canonical bytes — one short value per gate, small enough to live in a
   config table and to appear in ``configSnapshot`` on every run doc.
2. **The manifest pins the files.** Each of ``model.onnx`` / ``tokenizer.json`` /
   ``config.json`` carries its own digest and size.

So a single value in config transitively covers hundreds of megabytes, and an auditor
reading a six-month-old run doc can prove which bytes produced its verdicts. Verifying
the manifest without pinning it would be theatre: an attacker who can rewrite the model
can rewrite the manifest beside it.

Canonical bytes are ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` + a
trailing newline — fixed here rather than left to the caller, because "the digest of the
manifest" has to mean exactly one byte sequence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "REQUIRED_FILES",
    "Manifest",
    "ManifestFile",
    "canonical_bytes",
    "sha256_bytes",
    "sha256_file",
]

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

REQUIRED_FILES: tuple[str, ...] = ("model.onnx", "tokenizer.json", "config.json")
"""The GCS layout from LLD §13. A manifest that omits one of these is malformed."""

_CHUNK = 1024 * 1024


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
    files: dict[str, ManifestFile]
    opset: int | None = None

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
            "files": {name: entry.to_json() for name, entry in sorted(self.files.items())},
        }
        if self.opset is not None:
            document["opset"] = self.opset
        return document

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> Manifest:
        """Parse a manifest.

        Raises:
            ValueError: on an unknown ``schemaVersion`` or a missing required file. An
                unreadable manifest is a fetch failure, not something to work around —
                the caller turns this into ``GK_E_MODEL_FETCH``.
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
            files = {
                name: ManifestFile.from_json(entry)
                for name, entry in dict(document["files"]).items()
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
                files=files,
                opset=int(document["opset"]) if "opset" in document else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed manifest: {exc}") from exc

        missing = [name for name in REQUIRED_FILES if name not in parsed.files]
        if missing:
            raise ValueError(f"manifest is missing required files: {', '.join(missing)}")

        return parsed

    def canonical(self) -> bytes:
        """The exact bytes uploaded as ``manifest.json`` and digested into config."""
        return canonical_bytes(self.to_json())

    def digest(self) -> str:
        """The sha256 a gate's ``sha256`` config value must equal."""
        return sha256_bytes(self.canonical())
