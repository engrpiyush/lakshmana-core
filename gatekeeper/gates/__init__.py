"""Gate decision logic (LLD §8) — the rules, separated from the plumbing.

Sessions 04-06 wrap these in the worker's claim/execute/commit machinery. The replay
bake-off (VA-97) drives them directly. Both must reach the same verdicts from the same
scores, which is only guaranteed if there is one implementation, so this is it.
"""

from gatekeeper.gates.decisions import (
    Decision,
    Direction,
    Disposition,
    G1Scores,
    G2Scores,
    G3Scores,
    Thresholds,
    decide_g1,
    decide_g2,
    decide_g3,
)

__all__ = [
    "Decision",
    "Direction",
    "Disposition",
    "G1Scores",
    "G2Scores",
    "G3Scores",
    "Thresholds",
    "decide_g1",
    "decide_g2",
    "decide_g3",
]
