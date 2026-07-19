"""Artifact loading (LLD §13).

The loader is the only thing standing between the model bucket and inference, so it is
deliberately unforgiving: every byte it hands a gate has been digested and matched
against a pin that traces back to ``configSnapshot``. Anything that does not match is
``GK_E_MODEL_FETCH`` — a FAILED gate the operator fixes and retriggers FROM_GATE, never
a silent fallback to whatever happens to be on disk.

Three properties the gates depend on:

* **Lazy, per gate.** G2's artifact is ~435 MB. A worker executing G1 has no business
  paying for it, so nothing is fetched until a gate asks by name.
* **Cached on disk.** Cloud Run job tasks get a writable filesystem, and a retried task
  or a second gate in the same container should not re-download.
* **Atomic.** Files land in a temp directory and are renamed in only after every digest
  matches, so a killed task cannot leave a half-written model that looks cached.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gatekeeper.config import Config
from gatekeeper.enums import Gate
from gatekeeper.errors import ModelFetchError
from gatekeeper.logging import get_logger
from gatekeeper.models.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    sha256_bytes,
    sha256_file,
)
from gatekeeper.models.roster import ArtifactRef, parse_ref

__all__ = [
    "Artifact",
    "BlobStore",
    "GcsBlobStore",
    "LocalBlobStore",
    "ModelLoader",
    "artifact_for_gate",
    "loader_from_config",
]

log = get_logger(__name__)

DEFAULT_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "gatekeeper-models"


class BlobStore(Protocol):
    """The slice of object storage the loader needs.

    Narrow on purpose: it keeps ``google-cloud-storage`` out of the test path, and it
    means a replay harness can point the same loader at a local directory.
    """

    def read_bytes(self, object_path: str) -> bytes:
        """Fetch a small object into memory. Raises :class:`FileNotFoundError` if absent."""
        ...

    def download(self, object_path: str, destination: Path) -> None:
        """Stream a large object to a path. Raises :class:`FileNotFoundError` if absent."""
        ...


@dataclass(frozen=True, slots=True)
class Artifact:
    """A verified artifact on local disk."""

    ref: ArtifactRef
    directory: Path
    manifest: Manifest

    @property
    def model_path(self) -> Path:
        return self.directory / "model.onnx"

    @property
    def tokenizer_path(self) -> Path:
        return self.directory / "tokenizer.json"

    @property
    def config_path(self) -> Path:
        return self.directory / "config.json"


class LocalBlobStore:
    """:class:`BlobStore` over a directory laid out exactly like the bucket.

    The bake-off (VA-97) has to run the *production* loader over artifacts that have not
    been mirrored anywhere yet, and pointing it at a local mirror is the honest way to do
    that: same manifest, same digest checks, same ``GK_E_MODEL_FETCH`` on a bad file. A
    mode that skipped verification would be measuring code that never ships.
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _resolve(self, object_path: str) -> Path:
        candidate = (self._root / object_path).resolve()
        # An artifact ref is validated upstream, but a store that can be walked out of
        # its own root is not a store.
        if not candidate.is_relative_to(self._root.resolve()):
            raise FileNotFoundError(object_path)
        return candidate

    def read_bytes(self, object_path: str) -> bytes:
        return self._resolve(object_path).read_bytes()

    def download(self, object_path: str, destination: Path) -> None:
        shutil.copyfile(self._resolve(object_path), destination)


class GcsBlobStore:
    """:class:`BlobStore` over a GCS bucket."""

    __slots__ = ("_bucket", "_bucket_name")

    def __init__(self, bucket_name: str, client: object | None = None) -> None:
        from google.cloud import storage  # imported lazily: tests never need the SDK

        resolved = client if client is not None else storage.Client()
        self._bucket_name = bucket_name
        self._bucket = resolved.bucket(bucket_name)  # type: ignore[attr-defined]

    def read_bytes(self, object_path: str) -> bytes:
        from google.cloud.exceptions import NotFound

        try:
            return bytes(self._bucket.blob(object_path).download_as_bytes())
        except NotFound as exc:
            raise FileNotFoundError(f"gs://{self._bucket_name}/{object_path}") from exc

    def download(self, object_path: str, destination: Path) -> None:
        from google.cloud.exceptions import NotFound

        try:
            self._bucket.blob(object_path).download_to_filename(str(destination))
        except NotFound as exc:
            raise FileNotFoundError(f"gs://{self._bucket_name}/{object_path}") from exc


class ModelLoader:
    """Fetches, verifies and caches artifacts; memoizes per ref within the process."""

    __slots__ = ("_cache_dir", "_loaded", "_store")

    def __init__(self, store: BlobStore, *, cache_dir: Path | None = None) -> None:
        self._store = store
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._loaded: dict[str, Artifact] = {}

    def load(self, ref: str, *, expected_manifest_sha256: str) -> Artifact:
        """Return a verified artifact for ``name@version``.

        Args:
            ref: the config value, e.g. ``modernbert-base-nli@v1``.
            expected_manifest_sha256: the gate's pinned digest. Empty means *unpinned* —
                allowed only because the roster's shas stay blank until LK-4 has actually
                mirrored something, and it is logged loudly every time.

        Raises:
            ModelFetchError: artifact missing, manifest malformed, or any digest
                mismatch — all of it ``GK_E_MODEL_FETCH`` (LLD §11).
        """
        if ref in self._loaded:
            return self._loaded[ref]

        try:
            parsed = parse_ref(ref)
        except ValueError as exc:
            raise ModelFetchError(str(exc)) from exc

        manifest = self._fetch_manifest(parsed, expected_manifest_sha256)
        directory = self._materialize(parsed, manifest)

        artifact = Artifact(ref=parsed, directory=directory, manifest=manifest)
        self._loaded[ref] = artifact
        log.info(
            "artifact ready",
            fields={
                "artifact": parsed.ref,
                "sourceCheckpoint": manifest.source_checkpoint,
                "sourceRevision": manifest.source_revision,
                "quantization": manifest.quantization,
                "directory": str(directory),
            },
        )
        return artifact

    # --- internals --------------------------------------------------------------------

    def _fetch_manifest(self, ref: ArtifactRef, expected_sha256: str) -> Manifest:
        object_path = f"{ref.prefix}/{MANIFEST_FILENAME}"
        try:
            raw = self._store.read_bytes(object_path)
        except FileNotFoundError as exc:
            raise ModelFetchError(f"no manifest at {object_path}") from exc
        except Exception as exc:  # transport failures are fetch failures too
            raise ModelFetchError(f"could not read {object_path}: {exc}") from exc

        found = sha256_bytes(raw)
        if not expected_sha256:
            # Not an error yet: config ships blank shas until the first mirror run, and
            # refusing to start would block session03's own bake-off. It is logged at
            # WARNING with the digest so pinning it is a copy-paste.
            log.warning(
                "artifact is not pinned; loading it unverified",
                fields={"artifact": ref.ref, "observedManifestSha256": found},
            )
        elif found != expected_sha256:
            raise ModelFetchError(
                f"manifest sha256 mismatch for {ref.ref}: "
                f"config pins {expected_sha256}, bucket holds {found}"
            )

        try:
            return Manifest.from_json(json.loads(raw))
        except (ValueError, TypeError) as exc:
            raise ModelFetchError(f"manifest for {ref.ref} is unreadable: {exc}") from exc

    def _materialize(self, ref: ArtifactRef, manifest: Manifest) -> Path:
        target = self._cache_dir / ref.name / ref.version
        if self._is_complete(target, manifest):
            log.info("artifact cache hit", fields={"artifact": ref.ref, "directory": str(target)})
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{ref.name}-{ref.version}-", dir=target.parent))
        try:
            for filename, expected in manifest.files.items():
                destination = staging / filename
                object_path = f"{ref.prefix}/{filename}"
                try:
                    self._store.download(object_path, destination)
                except FileNotFoundError as exc:
                    raise ModelFetchError(
                        f"{object_path} is listed in the manifest but absent from the bucket"
                    ) from exc
                except Exception as exc:
                    raise ModelFetchError(f"could not download {object_path}: {exc}") from exc

                actual = sha256_file(destination)
                if actual != expected.sha256:
                    raise ModelFetchError(
                        f"sha256 mismatch for {object_path}: "
                        f"manifest says {expected.sha256}, downloaded {actual}"
                    )

            # Only now is the directory allowed to exist under its real name. A crash
            # anywhere above leaves a temp directory, not a plausible-looking cache entry.
            if target.exists():
                shutil.rmtree(target)
            staging.replace(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return target

    @staticmethod
    def _is_complete(directory: Path, manifest: Manifest) -> bool:
        """True when every manifest file is present with a matching digest.

        Digests are re-checked on every cache hit rather than trusted from a stamp file.
        Hashing half a gigabyte costs about a second; being wrong about which weights
        produced a verdict costs a re-run of the whole corpus.
        """
        if not directory.is_dir():
            return False
        for filename, expected in manifest.files.items():
            candidate = directory / filename
            if not candidate.is_file() or candidate.stat().st_size != expected.bytes_:
                return False
            if sha256_file(candidate) != expected.sha256:
                return False
        return True


# --- wiring ----------------------------------------------------------------------------

_GATE_SLOT: dict[Gate, str] = {
    Gate.G1_NEUTRAL: "g1",
    Gate.G2_CORROBORATION: "g2",
    Gate.G3_CONTRADICTION: "g3",
}
"""Gate → config slot. G4 is absent by design: it calls Vertex, not an ONNX artifact."""


def loader_from_config(config: Config) -> ModelLoader:
    """Build a loader from ``gatekeeper.models.*``.

    ``gatekeeper.models.local-dir`` wins when set — that is the bake-off and local-dev
    path. Otherwise the bucket is used, which is what Terraform configures.

    Raises:
        ModelFetchError: if neither is configured, or if the local directory is missing.
            A worker that reaches inference with nowhere to fetch from should say so in
            the gate's error code, not fail later with an SDK traceback.
    """
    cache_dir = Path(config.get_str("gatekeeper.models.cache-dir"))

    local_dir = config.get_str("gatekeeper.models.local-dir")
    if local_dir:
        root = Path(local_dir)
        if not root.is_dir():
            raise ModelFetchError(f"GATEKEEPER_MODELS_LOCAL_DIR={local_dir!r} is not a directory")
        log.info("loading artifacts from a local mirror", fields={"localDir": str(root)})
        return ModelLoader(LocalBlobStore(root), cache_dir=cache_dir)

    bucket = config.get_str("gatekeeper.models.bucket")
    if not bucket:
        raise ModelFetchError(
            "no model source configured; set GATEKEEPER_MODELS_BUCKET (Terraform wires "
            "this to the gatekeeper-models bucket) or GATEKEEPER_MODELS_LOCAL_DIR for a "
            "local mirror"
        )
    return ModelLoader(GcsBlobStore(bucket), cache_dir=cache_dir)


def artifact_for_gate(loader: ModelLoader, config: Config, gate: Gate) -> Artifact:
    """Load the artifact a gate's config points at.

    Raises:
        ModelFetchError: for G4, which has no encoder artifact, and for every failure
            :meth:`ModelLoader.load` raises.
    """
    slot = _GATE_SLOT.get(gate)
    if slot is None:
        raise ModelFetchError(f"{gate.value} has no ONNX artifact; it escalates to Vertex")
    return loader.load(
        config.get_str(f"gatekeeper.gates.{slot}.model"),
        expected_manifest_sha256=config.get_str(f"gatekeeper.gates.{slot}.sha256"),
    )
