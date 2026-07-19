"""Model artifacts: the roster, the manifest format, and the loader (LLD §8, §13).

The runtime never fetches from Hugging Face. Hugging Face appears exactly once in this
package's vocabulary — in :mod:`gatekeeper.models.roster`, as the *source* an owner-run
prep script mirrors from. What the worker loads is whatever is in the model bucket, and
only after its sha256 matches (``GK_E_MODEL_FETCH`` otherwise).
"""

from gatekeeper.models.loader import (
    Artifact,
    BlobStore,
    GcsBlobStore,
    LocalBlobStore,
    ModelLoader,
    artifact_for_binding,
    artifact_for_gate,
    loader_from_config,
)
from gatekeeper.models.manifest import MANIFEST_FILENAME, Manifest, ManifestFile, sha256_file
from gatekeeper.models.roster import (
    ALTERNATES,
    V1_ROSTER,
    ArtifactRef,
    RosterEntry,
    entry_for,
    parse_ref,
)

__all__ = [
    "ALTERNATES",
    "MANIFEST_FILENAME",
    "V1_ROSTER",
    "Artifact",
    "ArtifactRef",
    "BlobStore",
    "GcsBlobStore",
    "LocalBlobStore",
    "Manifest",
    "ManifestFile",
    "ModelLoader",
    "RosterEntry",
    "artifact_for_binding",
    "artifact_for_gate",
    "entry_for",
    "loader_from_config",
    "parse_ref",
    "sha256_file",
]
