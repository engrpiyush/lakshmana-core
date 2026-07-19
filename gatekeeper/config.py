"""Configuration (LLD §3, §7.1).

Keys are kebab-case dotted paths mirroring the factory's ``app.stage3.*`` style; each
one declares its ``GATEKEEPER_*`` environment override explicitly rather than deriving
it, because the LLD pins names like ``GATEKEEPER_G1_NEUTRAL_MIN`` that drop the
``gates`` segment. The same table builds ``gatekeeper_runs.configSnapshot``, so the
thing gates read at runtime and the thing an auditor reads six months later cannot
disagree.

Every numeric gate threshold below is a **proposal**: LK-5's replay report sets the real
values and the owner signs them (LLD §8, O-1). Model ``sha256`` values stay empty until
LK-4 mirrors the artifacts.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from gatekeeper.enums import Gate

__all__ = ["SETTINGS", "Config", "GateBinding", "Setting", "load_config"]


def _to_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Setting:
    """One configuration key: its kebab-case name, env override, default, and coercion."""

    key: str
    env: str
    default: Any
    cast: Callable[[str], Any]
    snapshot_path: tuple[str, ...] | None = None
    secret: bool = False


SETTINGS: tuple[Setting, ...] = (
    # --- roster ------------------------------------------------------------------
    Setting(
        "gatekeeper.roster.version", "GATEKEEPER_ROSTER_VERSION", "v1", str, ("rosterVersion",)
    ),
    # --- G1_NEUTRAL --------------------------------------------------------------
    Setting(
        "gatekeeper.gates.g1.model",
        "GATEKEEPER_G1_MODEL",
        "modernbert-base-nli@v1",
        str,
        ("g1", "model"),
    ),
    Setting("gatekeeper.gates.g1.sha256", "GATEKEEPER_G1_SHA256", "", str, ("g1", "sha256")),
    Setting(
        "gatekeeper.gates.g1.neutral-min",
        "GATEKEEPER_G1_NEUTRAL_MIN",
        0.95,
        float,
        ("g1", "neutralMin"),
    ),
    Setting(
        "gatekeeper.gates.g1.repeat-min",
        "GATEKEEPER_G1_REPEAT_MIN",
        0.90,
        float,
        ("g1", "repeatMin"),
    ),
    Setting(
        "gatekeeper.gates.g1.contra-escape",
        "GATEKEEPER_G1_CONTRA_ESCAPE",
        0.02,
        float,
        ("g1", "contraEscape"),
    ),
    Setting(
        "gatekeeper.gates.g1.max-seq-tokens",
        "GATEKEEPER_G1_MAX_SEQ_TOKENS",
        512,
        int,
        ("g1", "maxSeqTokens"),
    ),
    # --- G2_CORROBORATION --------------------------------------------------------
    Setting(
        "gatekeeper.gates.g2.model",
        "GATEKEEPER_G2_MODEL",
        "minicheck-deberta-l@v1",
        str,
        ("g2", "model"),
    ),
    Setting("gatekeeper.gates.g2.sha256", "GATEKEEPER_G2_SHA256", "", str, ("g2", "sha256")),
    Setting(
        "gatekeeper.gates.g2.support-min",
        "GATEKEEPER_G2_SUPPORT_MIN",
        0.90,
        float,
        ("g2", "supportMin"),
    ),
    Setting(
        "gatekeeper.gates.g2.support-floor",
        "GATEKEEPER_G2_SUPPORT_FLOOR",
        0.50,
        float,
        ("g2", "supportFloor"),
    ),
    Setting(
        "gatekeeper.gates.g2.grounding-mode",
        "GATEKEEPER_G2_GROUNDING_MODE",
        "AUTO",
        str,
        ("g2", "groundingMode"),
    ),
    # --- G3_CONTRADICTION --------------------------------------------------------
    Setting(
        "gatekeeper.gates.g3.model",
        "GATEKEEPER_G3_MODEL",
        "deberta-mnli-fever-anli@v1",
        str,
        ("g3", "model"),
    ),
    Setting("gatekeeper.gates.g3.sha256", "GATEKEEPER_G3_SHA256", "", str, ("g3", "sha256")),
    Setting(
        "gatekeeper.gates.g3.contra-min",
        "GATEKEEPER_G3_CONTRA_MIN",
        0.85,
        float,
        ("g3", "contraMin"),
    ),
    # Named in LLD §8's decide_g3 pseudocode but never given a key or a proposed value.
    # Added as a proposal like the rest of §8's numbers; the bake-off sets it.
    Setting(
        "gatekeeper.gates.g3.neutral-consensus",
        "GATEKEEPER_G3_NEUTRAL_CONSENSUS",
        0.10,
        float,
        ("g3", "neutralConsensus"),
    ),
    Setting(
        "gatekeeper.gates.g3.agreement-rule",
        "GATEKEEPER_G3_AGREEMENT_RULE",
        "TWO_FAMILY",
        str,
        ("g3", "agreementRule"),
    ),
    # --- G4_ESCALATION -----------------------------------------------------------
    Setting(
        "gatekeeper.gates.g4.llm-model",
        "GATEKEEPER_G4_LLM_MODEL",
        "pending-lite-pin",
        str,
        ("g4", "llmModel"),
    ),
    Setting(
        "gatekeeper.gates.g4.max-llm-pairs",
        "GATEKEEPER_G4_MAX_LLM_PAIRS",
        3000,
        int,
        ("g4", "maxLlmPairs"),
    ),
    Setting(
        "gatekeeper.gates.g4.thinking-budget",
        "GATEKEEPER_G4_THINKING_BUDGET",
        512,
        int,
        ("g4", "thinkingBudget"),
    ),
    # --- runtime (not frozen into the snapshot) ----------------------------------
    Setting("gatekeeper.lease.gate-minutes", "GATEKEEPER_LEASE_GATE_MINUTES", 90, int),
    Setting("gatekeeper.sweep.max-attempts", "GATEKEEPER_SWEEP_MAX_ATTEMPTS", 3, int),
    Setting(
        "gatekeeper.sweep.requested-grace-minutes",
        "GATEKEEPER_SWEEP_REQUESTED_GRACE_MINUTES",
        30,
        int,
    ),
    Setting("gatekeeper.judge-mode-default", "GATEKEEPER_JUDGE_MODE_DEFAULT", "GATEKEEPER", str),
    Setting(
        "gatekeeper.integration.stage3-runs-collection",
        "GATEKEEPER_STAGE3_RUNS_COLLECTION",
        "stage3_runs",
        str,
    ),
    Setting(
        "gatekeeper.integration.stage3-edges-collection",
        "GATEKEEPER_STAGE3_EDGES_COLLECTION",
        "stage3_edges",
        str,
    ),
    Setting("gatekeeper.queue.collection", "GATEKEEPER_QUEUE_COLLECTION", "gatekeeper_pairs", str),
    Setting("gatekeeper.queue.batch-size", "GATEKEEPER_QUEUE_BATCH_SIZE", 32, int),
    Setting("gatekeeper.queue.lease-minutes", "GATEKEEPER_QUEUE_LEASE_MINUTES", 15, int),
    # --- dispatcher ingress ------------------------------------------------------
    # Push deliveries carry an OIDC token; the dispatcher is internal-ingress but that
    # is a network control, not an authentication one, so the token is verified too.
    # Empty audience means "accept the token's own audience" — a deployment that has not
    # been told its URL yet still authenticates the caller's identity.
    Setting(
        "gatekeeper.dispatcher.require-oidc", "GATEKEEPER_DISPATCHER_REQUIRE_OIDC", True, _to_bool
    ),
    Setting("gatekeeper.dispatcher.oidc-audience", "GATEKEEPER_DISPATCHER_OIDC_AUDIENCE", "", str),
    Setting(
        "gatekeeper.dispatcher.allowed-service-accounts",
        "GATEKEEPER_DISPATCHER_ALLOWED_SERVICE_ACCOUNTS",
        "",
        str,
    ),
    Setting("gatekeeper.firestore.project-id", "GATEKEEPER_FIRESTORE_PROJECT_ID", "", str),
    Setting("gatekeeper.firestore.database", "GATEKEEPER_FIRESTORE_DATABASE", "(default)", str),
    Setting("gatekeeper.pubsub.topic", "GATEKEEPER_PUBSUB_TOPIC", "gatekeeper-requests", str),
    Setting(
        "gatekeeper.pubsub.dlq-subscription",
        "GATEKEEPER_PUBSUB_DLQ_SUBSCRIPTION",
        "gatekeeper-requests-dlq-pull",
        str,
    ),
    Setting("gatekeeper.worker.job-name", "GATEKEEPER_WORKER_JOB_NAME", "gatekeeper-worker", str),
    Setting("gatekeeper.worker.job-region", "GATEKEEPER_WORKER_JOB_REGION", "asia-southeast1", str),
    Setting("gatekeeper.worker.execute-jobs", "GATEKEEPER_WORKER_EXECUTE_JOBS", False, _to_bool),
    Setting("gatekeeper.models.bucket", "GATEKEEPER_MODELS_BUCKET", "", str),
    # Local-directory mode. Set, the loader reads artifacts straight off disk instead of
    # GCS — same manifest, same digests, same refusals. It is how the replay bake-off
    # runs before anything has been mirrored to a bucket. Empty (the default, and what
    # Terraform leaves it as) means GCS, so a deployed worker cannot silently fall back
    # to a directory that happens to exist.
    Setting("gatekeeper.models.local-dir", "GATEKEEPER_MODELS_LOCAL_DIR", "", str),
    Setting(
        "gatekeeper.models.cache-dir", "GATEKEEPER_MODELS_CACHE_DIR", "/tmp/gatekeeper-models", str
    ),
    Setting("gatekeeper.neo4j.uri", "GATEKEEPER_NEO4J_URI", "bolt://localhost:7687", str),
    Setting("gatekeeper.neo4j.user", "GATEKEEPER_NEO4J_USER", "neo4j", str),
    Setting("gatekeeper.neo4j.password", "GATEKEEPER_NEO4J_PASSWORD", "", str, secret=True),
    Setting("gatekeeper.neo4j.database", "GATEKEEPER_NEO4J_DATABASE", "neo4j", str),
)

_BY_KEY: dict[str, Setting] = {setting.key: setting for setting in SETTINGS}


class Config(Mapping[str, Any]):
    """Resolved configuration — a read-only mapping of kebab-case key to typed value."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get_str(self, key: str) -> str:
        return str(self._values[key])

    def get_int(self, key: str) -> int:
        return int(self._values[key])

    def get_float(self, key: str) -> float:
        return float(self._values[key])

    def get_bool(self, key: str) -> bool:
        return bool(self._values[key])

    def snapshot(self) -> dict[str, Any]:
        """Build ``gatekeeper_runs.configSnapshot`` (camelCase, LLD §7.1).

        Frozen into the run doc at creation so a mid-run config edit cannot produce a
        mixed-calibration run (LLD §5).
        """
        snapshot: dict[str, Any] = {}
        for setting in SETTINGS:
            if setting.snapshot_path is None:
                continue
            *parents, leaf = setting.snapshot_path
            target = snapshot
            for parent in parents:
                target = target.setdefault(parent, {})
            target[leaf] = self._values[setting.key]
        return snapshot

    def redacted(self) -> dict[str, Any]:
        """All values with secrets masked — safe to log at startup."""
        return {
            setting.key: "***"
            if setting.secret and self._values[setting.key]
            else self._values[setting.key]
            for setting in SETTINGS
        }


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Resolve every setting from ``env`` (defaults to ``os.environ``).

    Raises:
        ValueError: if an override is present but cannot be coerced — a typo in a
            threshold must fail at startup, not silently fall back to the default.
    """
    source = os.environ if env is None else env
    values: dict[str, Any] = {}
    for setting in SETTINGS:
        raw = source.get(setting.env)
        if raw is None:
            values[setting.key] = setting.default
            continue
        try:
            values[setting.key] = setting.cast(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{setting.env}={raw!r} is not a valid value for {setting.key}"
            ) from exc

    if not values["gatekeeper.firestore.project-id"]:
        values["gatekeeper.firestore.project-id"] = source.get("GOOGLE_CLOUD_PROJECT", "")

    return Config(values)


def setting_for(key: str) -> Setting:
    """Look up a setting by its kebab-case key."""
    return _BY_KEY[key]


# --- per-gate binding --------------------------------------------------------------------

GATE_SLOT: dict[Gate, str] = {
    Gate.G1_NEUTRAL: "g1",
    Gate.G2_CORROBORATION: "g2",
    Gate.G3_CONTRADICTION: "g3",
}
"""Gate → snapshot slot. G4 is absent by design: it calls Vertex, not an ONNX artifact."""

DEFAULT_MAX_SEQ_TOKENS = 512
"""What a gate reads when its slot does not pin a budget. Only G1 pins one (LLD §8)."""


@dataclass(frozen=True, slots=True)
class GateBinding:
    """Which artifact a gate runs and how much of a pair it may read.

    The companion to :class:`~gatekeeper.gates.decisions.Thresholds`: that one carries every
    *number* a gate reads off ``configSnapshot``, this one carries every *artifact* choice.
    Both are resolved from the snapshot and never from live config, because the freeze
    invariant (LLD §5) covers the model just as much as the thresholds — a run that started
    on ``modernbert-base-nli@v1`` must finish on it, even if an operator edits the config
    between G1 and G3, and a FROM_GATE resume must pick up the same artifact the earlier
    gates used. Only a FROM_START run, which gets a new ``runRequestId`` and a fresh
    snapshot, may move to a new model.
    """

    gate: Gate
    model: str
    sha256: str
    max_seq_tokens: int

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any], gate: Gate) -> GateBinding:
        """Resolve ``gate``'s artifact from a ``configSnapshot`` mapping.

        Missing keys fall back to the shipped defaults — the same LLD proposals
        :data:`SETTINGS` declares — so a run doc frozen before a slot gained a key still
        resolves instead of failing mid-cascade.

        Raises:
            ValueError: for a gate that has no encoder artifact (G4).
        """
        slot = GATE_SLOT.get(gate)
        if slot is None:
            raise ValueError(f"{gate.value} has no ONNX artifact; it escalates to Vertex")

        row = snapshot.get(slot) or {}
        tokens_setting = _BY_KEY.get(f"gatekeeper.gates.{slot}.max-seq-tokens")
        return cls(
            gate=gate,
            model=str(row.get("model") or _BY_KEY[f"gatekeeper.gates.{slot}.model"].default),
            sha256=str(row.get("sha256") or ""),
            max_seq_tokens=int(
                row.get("maxSeqTokens")
                or (tokens_setting.default if tokens_setting else DEFAULT_MAX_SEQ_TOKENS)
            ),
        )
