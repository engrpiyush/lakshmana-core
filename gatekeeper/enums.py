"""Enumerations shared by the wire contract, the run doc, and the gates (LLD §3, §5-§8).

Every value is a SCREAMING_SNAKE string: proto3 JSON writes enums by name, and
Firestore stores the same token, so one spelling covers wire and storage.
"""

from enum import StrEnum

__all__ = [
    "GATE_ORDER",
    "EscalationReason",
    "Gate",
    "GateState",
    "JudgeMode",
    "Method",
    "QueueTier",
    "RunMode",
    "RunState",
    "TriggeredBy",
    "Verdict",
    "next_gate",
    "prior_gate",
]


class Gate(StrEnum):
    """The four cascade gates that appear on the wire and in the run doc's gate map.

    ``FINALIZE`` is deliberately absent: it is an internal worker step after G4, not
    a published gate and not a key in ``gatekeeper_runs.gates`` (LLD §7.1, §8).
    """

    G1_NEUTRAL = "G1_NEUTRAL"
    G2_CORROBORATION = "G2_CORROBORATION"
    G3_CONTRADICTION = "G3_CONTRADICTION"
    G4_ESCALATION = "G4_ESCALATION"


GATE_ORDER: tuple[Gate, ...] = (
    Gate.G1_NEUTRAL,
    Gate.G2_CORROBORATION,
    Gate.G3_CONTRADICTION,
    Gate.G4_ESCALATION,
)


def prior_gate(gate: Gate) -> Gate | None:
    """The gate that must be ``SUCCEEDED`` before ``gate`` may be claimed (None for G1)."""
    index = GATE_ORDER.index(gate)
    return GATE_ORDER[index - 1] if index else None


def next_gate(gate: Gate) -> Gate | None:
    """The gate the worker publishes after committing ``gate`` (None after G4 → FINALIZE)."""
    index = GATE_ORDER.index(gate)
    return GATE_ORDER[index + 1] if index + 1 < len(GATE_ORDER) else None


class RunState(StrEnum):
    """Run-level lifecycle (LLD §6)."""

    REQUESTED = "REQUESTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


TERMINAL_RUN_STATES: frozenset[RunState] = frozenset({RunState.SUCCEEDED, RunState.SUPERSEDED})
"""States that refuse every further transition.

``FAILED`` is *not* terminal: an operator retrigger (FROM_GATE) revives the run
(LLD §6 figure 3, §10). ``SUPERSEDED`` runs refuse everything (LLD §6 rule 4).
"""


class GateState(StrEnum):
    """Per-gate lifecycle, stored in the run doc's ``gates`` map (LLD §6)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RunMode(StrEnum):
    """Payload ``mode`` — a full chain or a single-gate re-entry (LLD §5, §10)."""

    FULL = "FULL"
    FROM_GATE = "FROM_GATE"


class TriggeredBy(StrEnum):
    """Who minted the triggering request (standard envelope, LLD §3)."""

    OPERATOR = "OPERATOR"
    SYSTEM = "SYSTEM"
    SWEEPER = "SWEEPER"


class JudgeMode(StrEnum):
    """Read from ``Stage3Run.paramsSnapshot`` — the split-brain guard (LLD §9).

    ``LLM`` never reaches a gatekeeper run doc: in that mode vishwamitra's ensemble
    decides and the cascade is not invoked at all.
    """

    LLM = "LLM"
    GATEKEEPER = "GATEKEEPER"
    SHADOW = "SHADOW"


class Verdict(StrEnum):
    """The pair-level verdicts the cascade can reach (LLD §8).

    ``CONTRADICTS`` is deliberately not something a gate *decides*: G3 routes
    contradiction candidates to the human queue and the human confirms, exactly as the
    LLM ensemble does today. It appears here because the replay corpus is labelled with
    the ensemble's own verdicts, which do include it.
    """

    NEUTRAL = "NEUTRAL"
    REPEATS = "REPEATS"
    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"


class Method(StrEnum):
    """``stage3_edges.method`` — who produced a verdict (LLD §7.2).

    ``ENSEMBLE``/``RULE``/``EMBEDDING`` are vishwamitra's existing values, listed here
    because the FROM_START purge must delete only the ``GK_*`` rows (D-5).
    """

    ENSEMBLE = "ENSEMBLE"
    RULE = "RULE"
    EMBEDDING = "EMBEDDING"
    GK_G1_NLI = "GK_G1_NLI"
    GK_G2_GROUNDING = "GK_G2_GROUNDING"
    GK_G3_XCHECK = "GK_G3_XCHECK"
    GK_G4_LLM = "GK_G4_LLM"


GATEKEEPER_METHODS: frozenset[Method] = frozenset(
    {Method.GK_G1_NLI, Method.GK_G2_GROUNDING, Method.GK_G3_XCHECK, Method.GK_G4_LLM}
)
"""The methods a FROM_START purge deletes; everything else is immutable (D-5)."""


class EscalationReason(StrEnum):
    """Why a pair left the cascade for G4 or the human queue (LLD §7.2)."""

    CONTRADICTION_SIGNAL = "CONTRADICTION_SIGNAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STAGE_DISAGREEMENT = "STAGE_DISAGREEMENT"
    CONTEXT_DISAGREEMENT = "CONTEXT_DISAGREEMENT"
    TRUNCATED = "TRUNCATED"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    LLM_ERROR = "LLM_ERROR"


class QueueTier(StrEnum):
    """Judge-queue routing tier (LLD §7.3)."""

    CASCADE = "CASCADE"
    LLM_TAIL = "LLM_TAIL"
    HUMAN = "HUMAN"
