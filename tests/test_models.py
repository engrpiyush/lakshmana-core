"""Roster, manifest and loader (VA-96).

The loader tests run against an in-memory blob store rather than GCS: what is being
tested is the integrity chain and the cache behaviour, and neither of those is more true
for costing money to exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gatekeeper.config import load_config
from gatekeeper.enums import Gate
from gatekeeper.errors import ErrorCode, ModelFetchError
from gatekeeper.models.loader import (
    Artifact,
    LocalBlobStore,
    ModelLoader,
    artifact_for_gate,
    loader_from_config,
)
from gatekeeper.models.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_FILES,
    Manifest,
    ManifestFile,
    canonical_bytes,
    sha256_bytes,
)
from gatekeeper.models.roster import (
    ALTERNATES,
    V1_ROSTER,
    ExportKind,
    ModelTask,
    entry_for,
    known_names,
    parse_ref,
)

# --- fakes ------------------------------------------------------------------------------


class FakeBlobStore:
    """An in-memory :class:`~gatekeeper.models.loader.BlobStore`."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects: dict[str, bytes] = dict(objects or {})
        self.downloads: list[str] = []

    def read_bytes(self, object_path: str) -> bytes:
        try:
            return self.objects[object_path]
        except KeyError:
            raise FileNotFoundError(object_path) from None

    def download(self, object_path: str, destination: Path) -> None:
        self.downloads.append(object_path)
        destination.write_bytes(self.read_bytes(object_path))


def build_artifact_objects(
    prefix: str = "modernbert-base-nli/v1",
    *,
    payloads: dict[str, bytes] | None = None,
) -> tuple[dict[str, bytes], Manifest]:
    """A complete, self-consistent artifact: three files plus a matching manifest."""
    content = payloads or {
        "model.onnx": b"onnx-graph-bytes",
        "tokenizer.json": b'{"tokenizer": true}',
        "config.json": b'{"config": true}',
    }
    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=prefix.split("/", maxsplit=1)[0],
        version=prefix.split("/")[1],
        source_checkpoint="tasksource/ModernBERT-base-nli",
        source_revision="0" * 40,
        task=ModelTask.NLI_3WAY.value,
        quantization="dynamic-int8-avx512-vnni-per-channel",
        exported_at="2026-07-19T00:00:00Z",
        files={
            name: ManifestFile(sha256=sha256_bytes(body), bytes_=len(body))
            for name, body in content.items()
        },
        opset=17,
    )
    objects = {f"{prefix}/{name}": body for name, body in content.items()}
    objects[f"{prefix}/{MANIFEST_FILENAME}"] = manifest.canonical()
    return objects, manifest


# --- roster -------------------------------------------------------------------------------


def test_v1_roster_covers_the_three_encoder_gates() -> None:
    assert [entry.ref for entry in V1_ROSTER] == [
        "modernbert-base-nli@v1",
        "minicheck-deberta-l@v1",
        "deberta-mnli-fever-anli@v1",
    ]


def test_v1_roster_matches_the_config_defaults() -> None:
    """The roster and the config table must not be able to drift apart."""
    config = load_config(env={})
    for slot, entry in zip(("g1", "g2", "g3"), V1_ROSTER, strict=True):
        assert config.get_str(f"gatekeeper.gates.{slot}.model") == entry.ref


def test_g1_and_g3_are_deliberately_different_families() -> None:
    g1, _, g3 = V1_ROSTER
    assert g1.task is ModelTask.NLI_3WAY and g3.task is ModelTask.NLI_3WAY
    assert g1.checkpoint.split("/")[-1].split("-")[0] != g3.checkpoint.split("/")[-1].split("-")[0]


def test_every_roster_entry_names_an_upstream_checkpoint() -> None:
    for entry in (*V1_ROSTER, *ALTERNATES):
        assert "/" in entry.checkpoint, f"{entry.ref} has no usable Hub id"


def test_alternates_cover_both_gate_shapes() -> None:
    tasks = {entry.task for entry in ALTERNATES}
    assert tasks == {ModelTask.NLI_3WAY, ModelTask.GROUNDING}


def test_hhem_is_flagged_as_needing_a_custom_export() -> None:
    """It ships as a trust_remote_code architecture; the standard path would lie."""
    hhem = next(entry for entry in ALTERNATES if entry.name == "hhem")
    assert hhem.export is ExportKind.CUSTOM
    assert hhem.notes


@pytest.mark.parametrize("ref", ["modernbert-base-nli@v1", "hhem@v1"])
def test_entry_for_round_trips(ref: str) -> None:
    assert entry_for(ref).ref == ref


@pytest.mark.parametrize(
    "bad", ["modernbert-base-nli", "modernbert-base-nli@1", "Modern@v1", "a@v1@v2", ""]
)
def test_parse_ref_refuses_malformed_refs(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_ref(bad)


def test_entry_for_rejects_an_unknown_ref() -> None:
    with pytest.raises(KeyError):
        entry_for("not-a-model@v1")


def test_known_names_lists_v1_and_alternates() -> None:
    assert len(known_names()) == len(V1_ROSTER) + len(ALTERNATES)


# --- manifest -----------------------------------------------------------------------------


def test_manifest_round_trips() -> None:
    _, manifest = build_artifact_objects()
    assert Manifest.from_json(json.loads(manifest.canonical())) == manifest


def test_manifest_canonical_bytes_are_order_independent() -> None:
    """The digest is a pin; it cannot depend on dict insertion order."""
    first = canonical_bytes({"b": 1, "a": {"d": 2, "c": 3}})
    second = canonical_bytes({"a": {"c": 3, "d": 2}, "b": 1})
    assert first == second
    assert first.endswith(b"\n")


def test_manifest_rejects_an_unknown_schema_version() -> None:
    _, manifest = build_artifact_objects()
    document = manifest.to_json() | {"schemaVersion": 99}
    with pytest.raises(ValueError, match="schemaVersion"):
        Manifest.from_json(document)


def test_manifest_rejects_a_missing_required_file() -> None:
    _, manifest = build_artifact_objects()
    document = manifest.to_json()
    del document["files"]["tokenizer.json"]
    with pytest.raises(ValueError, match="missing required files"):
        Manifest.from_json(document)


def test_manifest_digest_changes_when_any_file_digest_changes() -> None:
    _, original = build_artifact_objects()
    _, tampered = build_artifact_objects(
        payloads={
            "model.onnx": b"different-onnx-bytes",
            "tokenizer.json": b'{"tokenizer": true}',
            "config.json": b'{"config": true}',
        }
    )
    assert original.digest() != tampered.digest()


# --- loader -------------------------------------------------------------------------------


def test_load_verifies_and_caches(tmp_path: Path) -> None:
    objects, manifest = build_artifact_objects()
    store = FakeBlobStore(objects)
    loader = ModelLoader(store, cache_dir=tmp_path)

    artifact = loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())

    assert isinstance(artifact, Artifact)
    assert artifact.model_path.read_bytes() == b"onnx-graph-bytes"
    assert artifact.tokenizer_path.is_file() and artifact.config_path.is_file()
    assert artifact.manifest.source_revision == "0" * 40
    assert len(store.downloads) == len(REQUIRED_FILES)


def test_a_second_load_is_memoized_in_process(tmp_path: Path) -> None:
    objects, manifest = build_artifact_objects()
    store = FakeBlobStore(objects)
    loader = ModelLoader(store, cache_dir=tmp_path)

    first = loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())
    second = loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())

    assert first is second
    assert len(store.downloads) == len(REQUIRED_FILES)


def test_a_fresh_loader_reuses_the_disk_cache(tmp_path: Path) -> None:
    objects, manifest = build_artifact_objects()
    store = FakeBlobStore(objects)
    ModelLoader(store, cache_dir=tmp_path).load(
        "modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest()
    )

    store.downloads.clear()
    ModelLoader(store, cache_dir=tmp_path).load(
        "modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest()
    )

    assert store.downloads == []


def test_a_corrupted_cache_entry_is_refetched(tmp_path: Path) -> None:
    """Digests are re-checked on every hit, so a bit-flip on disk cannot survive."""
    objects, manifest = build_artifact_objects()
    store = FakeBlobStore(objects)
    artifact = ModelLoader(store, cache_dir=tmp_path).load(
        "modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest()
    )
    artifact.model_path.write_bytes(b"onnx-graph-bytez")  # same length, different bytes

    store.downloads.clear()
    refetched = ModelLoader(store, cache_dir=tmp_path).load(
        "modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest()
    )

    assert refetched.model_path.read_bytes() == b"onnx-graph-bytes"
    assert len(store.downloads) == len(REQUIRED_FILES)


def test_a_pinned_manifest_that_does_not_match_is_refused(tmp_path: Path) -> None:
    objects, _ = build_artifact_objects()
    loader = ModelLoader(FakeBlobStore(objects), cache_dir=tmp_path)

    with pytest.raises(ModelFetchError) as caught:
        loader.load("modernbert-base-nli@v1", expected_manifest_sha256="0" * 64)

    assert caught.value.code is ErrorCode.GK_E_MODEL_FETCH
    assert "config pins" in caught.value.detail


def test_a_tampered_model_file_is_refused(tmp_path: Path) -> None:
    """The manifest is honest, the object behind it is not — the second link in the chain."""
    objects, manifest = build_artifact_objects()
    objects["modernbert-base-nli/v1/model.onnx"] = b"malicious-graph"
    loader = ModelLoader(FakeBlobStore(objects), cache_dir=tmp_path)

    with pytest.raises(ModelFetchError, match="sha256 mismatch"):
        loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())


def test_a_refused_download_leaves_no_cache_directory(tmp_path: Path) -> None:
    objects, manifest = build_artifact_objects()
    objects["modernbert-base-nli/v1/model.onnx"] = b"malicious-graph"
    loader = ModelLoader(FakeBlobStore(objects), cache_dir=tmp_path)

    with pytest.raises(ModelFetchError):
        loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())

    assert not (tmp_path / "modernbert-base-nli" / "v1").exists()
    assert list(tmp_path.glob("**/model.onnx")) == []


def test_a_missing_manifest_is_a_fetch_error(tmp_path: Path) -> None:
    loader = ModelLoader(FakeBlobStore({}), cache_dir=tmp_path)
    with pytest.raises(ModelFetchError, match="no manifest"):
        loader.load("modernbert-base-nli@v1", expected_manifest_sha256="")


def test_a_file_listed_but_absent_is_a_fetch_error(tmp_path: Path) -> None:
    objects, manifest = build_artifact_objects()
    del objects["modernbert-base-nli/v1/tokenizer.json"]
    loader = ModelLoader(FakeBlobStore(objects), cache_dir=tmp_path)

    with pytest.raises(ModelFetchError, match="absent from the bucket"):
        loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())


def test_an_unreadable_manifest_is_a_fetch_error(tmp_path: Path) -> None:
    store = FakeBlobStore({f"modernbert-base-nli/v1/{MANIFEST_FILENAME}": b"not json at all"})
    loader = ModelLoader(store, cache_dir=tmp_path)

    with pytest.raises(ModelFetchError, match="unreadable"):
        loader.load("modernbert-base-nli@v1", expected_manifest_sha256="")


def test_a_malformed_ref_is_a_fetch_error_not_a_value_error(tmp_path: Path) -> None:
    loader = ModelLoader(FakeBlobStore({}), cache_dir=tmp_path)
    with pytest.raises(ModelFetchError):
        loader.load("modernbert-base-nli", expected_manifest_sha256="")


def test_an_unpinned_artifact_loads_but_warns(tmp_path: Path, caplog) -> None:
    """Config ships blank shas until the first mirror run; that must not block a load."""
    objects, _ = build_artifact_objects()
    loader = ModelLoader(FakeBlobStore(objects), cache_dir=tmp_path)

    with caplog.at_level("WARNING"):
        artifact = loader.load("modernbert-base-nli@v1", expected_manifest_sha256="")

    assert artifact.model_path.is_file()
    assert "not pinned" in caplog.text


# --- gate wiring --------------------------------------------------------------------------


def test_artifact_for_gate_follows_the_config_slot(tmp_path: Path) -> None:
    objects, manifest = build_artifact_objects()
    loader = ModelLoader(FakeBlobStore(objects), cache_dir=tmp_path)
    config = load_config(env={"GATEKEEPER_G1_SHA256": manifest.digest()})

    artifact = artifact_for_gate(loader, config, Gate.G1_NEUTRAL)

    assert artifact.ref.ref == "modernbert-base-nli@v1"


def test_g4_has_no_encoder_artifact(tmp_path: Path) -> None:
    loader = ModelLoader(FakeBlobStore({}), cache_dir=tmp_path)
    with pytest.raises(ModelFetchError, match="escalates to Vertex"):
        artifact_for_gate(loader, load_config(env={}), Gate.G4_ESCALATION)


def test_loader_from_config_requires_a_source() -> None:
    with pytest.raises(ModelFetchError, match="no model source configured"):
        loader_from_config(load_config(env={}))


# --- local-dir mode -----------------------------------------------------------------------


def _write_local_mirror(root: Path, prefix: str = "modernbert-base-nli/v1") -> Manifest:
    objects, manifest = build_artifact_objects(prefix)
    for object_path, body in objects.items():
        destination = root / object_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    return manifest


def test_local_mode_verifies_exactly_like_gcs(tmp_path: Path) -> None:
    """The bake-off must exercise the production loader, not a permissive test seam."""
    mirror, cache = tmp_path / "mirror", tmp_path / "cache"
    manifest = _write_local_mirror(mirror)
    loader = ModelLoader(LocalBlobStore(mirror), cache_dir=cache)

    artifact = loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())

    assert artifact.model_path.read_bytes() == b"onnx-graph-bytes"


def test_local_mode_still_refuses_a_tampered_file(tmp_path: Path) -> None:
    mirror, cache = tmp_path / "mirror", tmp_path / "cache"
    manifest = _write_local_mirror(mirror)
    (mirror / "modernbert-base-nli" / "v1" / "model.onnx").write_bytes(b"tampered-bytes!!")
    loader = ModelLoader(LocalBlobStore(mirror), cache_dir=cache)

    with pytest.raises(ModelFetchError, match="sha256 mismatch"):
        loader.load("modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest())


def test_local_mode_refuses_to_escape_its_root(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")
    store = LocalBlobStore(tmp_path / "mirror")
    (tmp_path / "mirror").mkdir()

    with pytest.raises(FileNotFoundError):
        store.read_bytes("../secret.txt")


def test_loader_from_config_prefers_the_local_dir(tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    manifest = _write_local_mirror(mirror)
    config = load_config(
        env={
            "GATEKEEPER_MODELS_BUCKET": "a-bucket-that-must-not-be-touched",
            "GATEKEEPER_MODELS_LOCAL_DIR": str(mirror),
            "GATEKEEPER_MODELS_CACHE_DIR": str(tmp_path / "cache"),
        }
    )

    artifact = loader_from_config(config).load(
        "modernbert-base-nli@v1", expected_manifest_sha256=manifest.digest()
    )

    assert artifact.model_path.is_file()


def test_a_missing_local_dir_is_a_fetch_error(tmp_path: Path) -> None:
    config = load_config(env={"GATEKEEPER_MODELS_LOCAL_DIR": str(tmp_path / "nope")})
    with pytest.raises(ModelFetchError, match="is not a directory"):
        loader_from_config(config)
