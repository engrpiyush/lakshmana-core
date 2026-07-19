"""Lakshmana gatekeeper — the Stage 3 judge cascade.

An LLM-free cascade of encoder gates (G1_NEUTRAL → G2_CORROBORATION →
G3_CONTRADICTION → G4_ESCALATION → FINALIZE) replacing vishwamitra's Stage 3 LLM
pair-judge. Design authority: LLD Confluence page 255688733, mirrored at
`lakshmana-gatekeeper-lld-wiki.md`.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
