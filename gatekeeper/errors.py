"""Error codes and exceptions (LLD §11).

Every failure the gatekeeper records carries a ``GK_E_*`` code: it lands on the gate
entry in the run doc, in the structured log line, and in the runbook's symptom table.
"""

from enum import StrEnum

__all__ = [
    "CapExceededError",
    "DryRunInGatekeeperError",
    "ErrorCode",
    "FirestoreTxnError",
    "GatekeeperError",
    "ModelFetchError",
    "Neo4jUnavailableError",
    "SchemaError",
    "StuckError",
    "VertexError",
]


class ErrorCode(StrEnum):
    """The ``GK_E_*`` table (LLD §11)."""

    GK_E_SCHEMA = "GK_E_SCHEMA"
    GK_E_MODEL_FETCH = "GK_E_MODEL_FETCH"
    GK_E_NEO4J_UNAVAILABLE = "GK_E_NEO4J_UNAVAILABLE"
    GK_E_FIRESTORE_TXN = "GK_E_FIRESTORE_TXN"
    GK_E_VERTEX = "GK_E_VERTEX"
    GK_E_CAP_EXCEEDED = "GK_E_CAP_EXCEEDED"
    GK_E_STUCK = "GK_E_STUCK"
    GK_E_G4_DRYRUN = "GK_E_G4_DRYRUN"


class GatekeeperError(Exception):
    """Base class; every subclass pins one :class:`ErrorCode`."""

    code: ErrorCode

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class SchemaError(GatekeeperError):
    """Unknown ``schemaVersion`` or malformed payload — NACK to the DLQ, never guessed."""

    code = ErrorCode.GK_E_SCHEMA


class ModelFetchError(GatekeeperError):
    """GCS artifact missing or sha256 mismatch — gate FAILED, operator fixes the manifest."""

    code = ErrorCode.GK_E_MODEL_FETCH


class Neo4jUnavailableError(GatekeeperError):
    """Hydrate failures beyond the retry budget — gate FAILED, usually just retrigger."""

    code = ErrorCode.GK_E_NEO4J_UNAVAILABLE


class FirestoreTxnError(GatekeeperError):
    """Claim/commit contention beyond retries — gate FAILED."""

    code = ErrorCode.GK_E_FIRESTORE_TXN


class VertexError(GatekeeperError):
    """G4 quota/5xx beyond backoff — affected pairs go HUMAN, the gate still SUCCEEDS."""

    code = ErrorCode.GK_E_VERTEX


class CapExceededError(GatekeeperError):
    """G4 ``maxLlmPairs`` hit — remainder goes HUMAN, run SUCCEEDS with a warning counter."""

    code = ErrorCode.GK_E_CAP_EXCEEDED


class StuckError(GatekeeperError):
    """Sweeps exhausted — run FAILED, see the runbook."""

    code = ErrorCode.GK_E_STUCK


class DryRunInGatekeeperError(GatekeeperError):
    """G4 reached a deciding run with the dry-run double — gate FAILED, never fabricated.

    In GATEKEEPER mode the cascade's verdicts are authoritative, so a G4 tail backed by the
    dry-run double would write canned NEUTRAL verdicts into ``stage3_edges`` and let the run
    report SUCCEEDED. The double is a local-testing affordance and must never be reachable in
    a deciding run: the gate refuses instead. The fix is to enable live calls
    (``GATEKEEPER_G4_LIVE_CALLS=true``) or to run in SHADOW. See VA-158 / DEFERRED-LIVE B1.
    """

    code = ErrorCode.GK_E_G4_DRYRUN
