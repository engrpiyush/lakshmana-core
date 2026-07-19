"""FINALIZE — compute ``totals``, set the run SUCCEEDED, publish nothing (LLD §8, §7.4).

Not a gate. It has no message, no lease and no entry in ``gatekeeper_runs.gates`` — it is
the last thing the G4 worker does before exiting, which is why :class:`~gatekeeper.enums.Gate`
deliberately has no ``FINALIZE`` member. It publishes nothing by owner requirement #6:
vishwamitra's existing poll observes ``state == SUCCEEDED`` and advances its own lifecycle.
No callback, no notification, no second integration surface.

**The totals are recomputed from the queue, never accumulated from the gate counters.**
Counters are per-execution and a rescued gate writes a fresh set, so summing them across a
run that was swept would double-count exactly the pairs a sweep touched. The queue rows are
the durable record of where every pair ended up, and there is one row per pair by
construction (§7.3's deterministic id), so counting them is both cheap and correct.

That is also what makes the reconciliation invariant meaningful rather than circular:

    decidedByGates + escalatedLlm + escalatedHuman == pairsSeen

A failure means a pair is **stranded** — sitting at a gate that has already been marked
SUCCEEDED, where nothing will pick it up again. That is a routing bug, not a rounding
difference, so it raises: a run that quietly reports ``SUCCEEDED`` over a stranded pair is
worse than one that fails loudly with the count.
"""

from __future__ import annotations

from dataclasses import dataclass

from gatekeeper.enums import Method, QueueTier
from gatekeeper.errors import ErrorCode, GatekeeperError
from gatekeeper.logging import get_logger
from gatekeeper.runs.model import GatekeeperRun, RunTotals
from gatekeeper.runs.store import GatekeeperRunStore
from gatekeeper.worker.queue import PairQueue, QueuedPair

__all__ = ["ReconciliationError", "Tally", "finalize", "tally"]

log = get_logger(__name__)

_CASCADE_METHODS = frozenset(
    {Method.GK_G1_NLI.value, Method.GK_G2_GROUNDING.value, Method.GK_G3_XCHECK.value}
)
"""The three encoder gates. ``GK_G4_LLM`` counts as ``escalatedLlm``, not as a gate."""


class ReconciliationError(GatekeeperError):
    """The totals do not add up — a pair was stranded somewhere in the cascade.

    Carries ``GK_E_FIRESTORE_TXN`` rather than a new code: from the runbook's point of view
    this is "the run's stored state is not what the gates should have left behind", the
    operator action is the same (inspect the run doc, retrigger), and §11's code table is a
    contract that a new code would silently extend.
    """

    code = ErrorCode.GK_E_FIRESTORE_TXN


@dataclass(frozen=True, slots=True)
class Tally:
    """Where this run's pairs ended up, counted off the queue."""

    pairs_seen: int = 0
    decided_by_gates: int = 0
    escalated_llm: int = 0
    escalated_human: int = 0
    stranded: int = 0
    """Pairs still sitting at a gate, or cleared without any method claiming them."""

    @property
    def reconciles(self) -> bool:
        return (
            self.decided_by_gates + self.escalated_llm + self.escalated_human == self.pairs_seen
            and self.stranded == 0
        )


def tally(pairs: list[QueuedPair]) -> Tally:
    """Classify every queue row into §7.1's three buckets.

    The classification is by **tier first, method second**, and the order matters. A
    contradiction candidate leaves G3 with ``tier = HUMAN`` and no ``decidedBy``; a pair the
    tail settled leaves G4 with ``tier = LLM_TAIL`` and ``decidedBy = GK_G4_LLM``; a pair
    the tail could not settle leaves with ``tier = HUMAN`` and no ``decidedBy``. Reading
    ``decidedBy`` first would leave that last group in no bucket at all.

    SHADOW is the one case where ``LLM_TAIL`` carries no ``decidedBy``: G4 makes no calls,
    so nothing decided the pair, but it *did* reach the tail and that is what this counts.
    It lands in ``escalatedLlm`` — what a cutover decision needs is how big the tail was,
    not what it cost in a mode where the answer is nothing.
    """
    pairs_seen = decided = llm = human = stranded = 0

    for pair in pairs:
        pairs_seen += 1
        if pair.gate is not None:
            stranded += 1
        elif pair.tier is QueueTier.HUMAN:
            human += 1
        elif pair.tier is QueueTier.LLM_TAIL:
            llm += 1
        elif pair.decided_by in _CASCADE_METHODS:
            decided += 1
        else:
            # CASCADE tier, gate cleared, but no gatekeeper method claimed it. Nothing in
            # the cascade produces this state, so it is counted as stranded rather than
            # quietly absorbed into a bucket it does not belong to.
            stranded += 1

    return Tally(
        pairs_seen=pairs_seen,
        decided_by_gates=decided,
        escalated_llm=llm,
        escalated_human=human,
        stranded=stranded,
    )


def finalize(
    store: GatekeeperRunStore,
    run: GatekeeperRun,
    queue: PairQueue,
    *,
    spend_usd: float = 0.0,
) -> RunTotals:
    """Reconcile, write ``totals``, and mark the run SUCCEEDED.

    ``shadowDisagreed`` is written as **0 in every mode**, and that is a deliberate gap
    rather than an oversight. In GATEKEEPER mode it is simply correct — the cascade's
    verdict *is* the verdict, so there is nothing to disagree with. In SHADOW mode the
    honest number cannot be computed here: vishwamitra's ensemble runs concurrently and may
    not have judged a pair by the time FINALIZE fires, so any count taken at this moment is
    a race with a systematic bias toward zero. The disagreement report is VA-105's offline
    pass over the durable rows, which does not race anybody, and it is the number the
    cutover decision reads.

    Args:
        spend_usd: what the tail cost, read off G4's committed gate counter — durable state,
            not this process's in-memory total, so a resumed G4 reports what was banked.

    Raises:
        ReconciliationError: if the buckets do not sum to ``pairsSeen``, or any pair is
            still sitting at a gate.

    Returns:
        The totals. If the run was not in a finalizable state they are returned but nothing
        was stored — the caller logs it.
    """
    counted = tally(queue.all_pairs(run.stage3_run_id))

    if not counted.reconciles:
        raise ReconciliationError(
            "run totals do not reconcile: "
            f"decidedByGates={counted.decided_by_gates} + escalatedLlm={counted.escalated_llm} "
            f"+ escalatedHuman={counted.escalated_human} != pairsSeen={counted.pairs_seen} "
            f"(stranded={counted.stranded}). A pair is sitting at a gate that has already "
            "been committed; inspect gatekeeper_pairs for this stage3RunId."
        )

    totals = RunTotals(
        pairs_seen=counted.pairs_seen,
        decided_by_gates=counted.decided_by_gates,
        escalated_llm=counted.escalated_llm,
        escalated_human=counted.escalated_human,
        shadow_disagreed=0,
        llm_spend_usd=spend_usd,
    )

    if not store.finalize_run(run.run_request_id, totals):
        # Not an error: a superseded run, and one another worker already finalized, both
        # land here, and both are correct outcomes of a redelivered G4.
        log.info(
            "run was not finalizable; totals were not written",
            fields={"state": run.state.value},
        )
        return totals

    log.info(
        "run finalized",
        fields={
            "totals": totals.to_firestore(),
            "judgeMode": run.judge_mode.value,
            # §8: FINALIZE publishes nothing. Said out loud because "did anything get
            # published?" is the first question asked when vishwamitra looks stuck.
            "published": None,
        },
    )
    return totals
