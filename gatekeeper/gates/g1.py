"""G1_NEUTRAL — the gate that sees 100% of capped pairs (LLD §8, §10, §11.9).

G1 is where the cascade earns its keep: on the reference subject it decides roughly 87% of
pairs outright and hands ~13% on. It is also the only gate that owns the run's *setup* —
the FROM_START purge, and seeding lakshmana's routing state from the graph — because it is
the only gate guaranteed to run first.

The decision itself is not here. It is :func:`~gatekeeper.gates.decisions.decide_g1`, a
pure function over probabilities and thresholds, and this module is the plumbing that feeds
it: hydrate, score both directions, apply, route, count. That split is what lets the VA-97
replay harness claim it measured the production gate rather than a re-implementation of it
— both call the same function with the same ``configSnapshot`` shape.

**Two orderings that are not stylistic.**

*The purge runs before any inference.* §10 says "G1's first act — before any inference".
A purge after scoring would delete rows the same execution had just written.

*Routing is written after the edge.* A crash between them leaves a pair still sitting at
G1 with its verdict already durable; the resumed worker re-scores it and upserts the same
row by the same key. The reverse order would lose the verdict of a pair the queue had
already moved past — the one outcome a resume cannot repair.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from gatekeeper.config import GateBinding
from gatekeeper.enums import EscalationReason, Gate, Method, QueueTier, Verdict
from gatekeeper.errors import Neo4jUnavailableError
from gatekeeper.gates.decisions import Decision, Disposition, G1Scores, Thresholds, decide_g1
from gatekeeper.integration import resolve_subject_id
from gatekeeper.logging import get_logger
from gatekeeper.models.loader import artifact_for_binding
from gatekeeper.scoring.encoder import NliScores, scorer_for_binding
from gatekeeper.worker.edges import EdgeVerdict, EdgeWriter, g1_stage_scores
from gatekeeper.worker.gates import GateContext, register
from gatekeeper.worker.hydrate import claim_texts, explanation_texts, queued_pairs
from gatekeeper.worker.queue import PairQueue, QueuedPair

__all__ = ["G1_COUNTERS", "run_g1"]

log = get_logger(__name__)

G1_COUNTERS: tuple[str, ...] = (
    "seen",
    "neutral",
    "repeats",
    "contraFlagged",
    "forwarded",
    "truncated",
    "ctxDisagreed",
)
"""LLD §8's counter list for G1, verbatim."""

CONTEXT_SEPARATOR = "\n\n"
"""How a claim and its explanation are composed for the contexted variant.

§11.9 says to "fetch the explanation text and run a contexted variant" without fixing the
composition. Concatenation keeps the pair's own text intact and therefore keeps the bare
and contexted scores comparable — the same reading `gatekeeper/replay/bakeoff.py` made, and
the two must agree or replay parity is meaningless.
"""


def _contexted(text: str, context: str) -> str:
    return f"{text}{CONTEXT_SEPARATOR}{context}" if context else text


def run_g1(context: GateContext) -> dict[str, Any]:
    """Execute G1 over every pair still sitting at it. Returns the gate counters."""
    run = context.run
    config = context.config
    binding = GateBinding.from_snapshot(run.config_snapshot, Gate.G1_NEUTRAL)
    thresholds = Thresholds.from_snapshot(run.config_snapshot)

    queue = PairQueue(
        context.client,
        collection=config.get_str("gatekeeper.queue.collection"),
        lease_minutes=config.get_int("gatekeeper.queue.lease-minutes"),
    )
    edges = EdgeWriter(
        context.client,
        collection=config.get_str("gatekeeper.integration.stage3-edges-collection"),
    )
    subject_id = resolve_subject_id(context.client, config, run.stage3_run_id)
    if not subject_id:
        # Without it there is no way to find the pairs in the graph. The operator action is
        # the same as any other hydrate failure — fix the data, retrigger FROM_GATE — so it
        # carries the same code rather than inventing a second one for the same runbook row.
        raise Neo4jUnavailableError(
            f"stage3 run {run.stage3_run_id!r} names no subjectId; cannot hydrate the queue"
        )

    # §8 names G1's counters exactly, and §7.1 makes them a UI contract: the gate-progress
    # panel polls this map, so every key is present at zero rather than absent. A missing
    # key and a zero mean different things to a reader, and only one of them is true.
    counters: Counter[str] = Counter(dict.fromkeys(G1_COUNTERS, 0))
    reader = context.reader()

    # --- setup: purge, then seed -----------------------------------------------------
    #
    # `attempt == 1` is exactly FROM_START. A new runRequestId is the only thing that
    # creates a run doc, and creation claims G1 at attempt 1 (LLD §6 rule 1) — so a first
    # attempt means new run, purge. Every later G1 execution on this run is a redelivery, a
    # sweeper rescue or an operator FROM_GATE, all of which are *resuming* output the purge
    # would destroy. The purge itself stays idempotent regardless (§10).
    if run.gate(Gate.G1_NEUTRAL).attempt <= 1:
        counters["purgedEdges"] = edges.purge(run.stage3_run_id)
        counters["purgedQueue"] = queue.reset(run.stage3_run_id)

    counters["seeded"] = queue.seed(
        run.stage3_run_id, run.intake_id, queued_pairs(reader, subject_id)
    )

    # --- the batch loop ---------------------------------------------------------------
    scorer = None
    batch_size = config.get_int("gatekeeper.queue.batch-size")

    while True:
        # Renew the gate lease before claiming more work, and stop if it has moved on: a
        # gate that outruns its 90-minute lease would otherwise be rescued mid-flight and
        # a second worker started on it (VA-159/B2). Renewing keeps a live worker's lease
        # ahead of the sweeper; a False return means a rescue already happened and this
        # worker must not write another batch.
        if not context.renew_lease():
            log.warning(
                "G1 no longer holds its gate lease; stopping without claiming more work",
                fields={"leaseOwner": context.lease_owner},
            )
            break

        batch = queue.claim_batch(
            run.stage3_run_id,
            Gate.G1_NEUTRAL,
            lease_owner=context.lease_owner,
            limit=batch_size,
        )
        if not batch:
            break

        if scorer is None:
            # Built on first use: a resumed run whose pairs are all decided must not pay
            # for several hundred megabytes of weights to discover it has nothing to do.
            scorer = _build_scorer(context, binding, batch_size)

        decisions = _score_batch(scorer, reader, batch)
        _write_outcomes(
            batch, decisions, thresholds, run, subject_id, binding, edges, queue, counters
        )
        log.info("G1 batch complete", fields={"batch": len(batch), "seen": counters["seen"]})

    return dict(counters)


def _build_scorer(context: GateContext, binding: GateBinding, batch_size: int) -> Any:
    """The scorer for this run's frozen binding.

    An injected factory short-circuits the loader entirely rather than being handed a
    loaded artifact: a caller supplying its own scorer has no use for several hundred
    megabytes of weights being fetched and verified first.
    """
    if context.scorer_factory is not None:
        return context.scorer_factory(binding)
    artifact = artifact_for_binding(context.loader(), binding)
    return scorer_for_binding(artifact, binding, batch_size=batch_size)


def _score_batch(scorer: Any, reader: Any, batch: Sequence[QueuedPair]) -> list[G1Scores]:
    """Both NLI directions for a batch, plus the contexted variant where §11.9 asks for it."""
    claim_ids = [pair.claim_a_id for pair in batch] + [pair.claim_b_id for pair in batch]
    texts = claim_texts(reader, claim_ids)

    missing = [
        pair.pair_id
        for pair in batch
        if not (texts.get(pair.claim_a_id) and texts.get(pair.claim_b_id))
    ]
    if missing:
        # A queued pair whose claim text the graph no longer has cannot be judged, and
        # guessing at it is worse than failing: the gate stops and the operator retriggers.
        raise Neo4jUnavailableError(
            f"{len(missing)} queued pair(s) have no claim text in the graph; first: {missing[0]}"
        )

    forward = scorer.score_nli([(texts[p.claim_a_id], texts[p.claim_b_id]) for p in batch])
    backward = scorer.score_nli([(texts[p.claim_b_id], texts[p.claim_a_id]) for p in batch])

    contexted_index = [index for index, pair in enumerate(batch) if pair.with_context]
    context_forward: dict[int, NliScores] = {}
    context_backward: dict[int, NliScores] = {}
    if contexted_index:
        subset = [batch[index] for index in contexted_index]
        explanations = explanation_texts(
            reader, [p.claim_a_id for p in subset] + [p.claim_b_id for p in subset]
        )
        pairs_a = [
            _contexted(texts[p.claim_a_id], explanations.get(p.claim_a_id, "")) for p in subset
        ]
        pairs_b = [
            _contexted(texts[p.claim_b_id], explanations.get(p.claim_b_id, "")) for p in subset
        ]
        context_forward = dict(
            zip(
                contexted_index,
                scorer.score_nli(list(zip(pairs_a, pairs_b, strict=True))),
                strict=True,
            )
        )
        context_backward = dict(
            zip(
                contexted_index,
                scorer.score_nli(list(zip(pairs_b, pairs_a, strict=True))),
                strict=True,
            )
        )

    return [
        G1Scores(
            forward=forward[index],
            backward=backward[index],
            context_forward=context_forward.get(index),
            context_backward=context_backward.get(index),
        )
        for index in range(len(batch))
    ]


def _write_outcomes(
    batch: Sequence[QueuedPair],
    scores: Sequence[G1Scores],
    thresholds: Thresholds,
    run: Any,
    subject_id: str,
    binding: GateBinding,
    edges: EdgeWriter,
    queue: PairQueue,
    counters: Counter[str],
) -> None:
    """Decide, persist the verdicts, then move the queue — in that order."""
    verdicts: list[EdgeVerdict] = []
    routes: list[tuple[QueuedPair, dict[str, Any]]] = []

    for pair, pair_scores in zip(batch, scores, strict=True):
        decision = decide_g1(pair_scores, thresholds)
        counters["seen"] += 1
        if pair_scores.truncated:
            counters["truncated"] += 1

        verdicts.append(
            EdgeVerdict(
                claim_a_id=pair.claim_a_id,
                claim_b_id=pair.claim_b_id,
                subject_id=subject_id,
                intake_id=run.intake_id,
                stage3_run_id=run.stage3_run_id,
                run_request_id=run.run_request_id,
                slot="g1",
                method=Method.GK_G1_NLI,
                judge_model=binding.model,
                stage_scores=g1_stage_scores(
                    pair_scores.forward,
                    pair_scores.backward,
                    pair_scores.context_forward,
                    pair_scores.context_backward,
                ),
                verdict=decision.verdict,
                escalation_reason=decision.escalation_reason,
                truncated=decision.truncated,
                with_context=pair.with_context,
                attempt=run.gate(Gate.G1_NEUTRAL).attempt,
            )
        )
        routes.append((pair, _routing(decision, run.run_request_id, counters)))

    edges.write_all(verdicts, run.judge_mode)
    queue.route_all(routes)


def _routing(decision: Decision, run_request_id: str, counters: Counter[str]) -> dict[str, Any]:
    """Where a pair goes next, and the counter that records it (LLD §8's counter list)."""
    routing: dict[str, Any] = {
        "contradictionFlag": decision.contradiction_flag,
        "gkRunRequestId": run_request_id,
    }

    if decision.disposition is Disposition.DECIDED:
        counters["neutral" if decision.verdict is Verdict.NEUTRAL else "repeats"] += 1
        return routing | {
            "gate": None,
            "tier": QueueTier.CASCADE.value,
            "decidedBy": Method.GK_G1_NLI.value,
        }

    if decision.disposition is Disposition.ESCALATE_G4:
        # The only G1 escalation is §11.9's bare-vs-contexted disagreement, and it skips
        # G2 and G3 entirely: the two gates in between cannot adjudicate a disagreement
        # about whether the pair's own context changes its meaning.
        if decision.escalation_reason is EscalationReason.CONTEXT_DISAGREEMENT:
            counters["ctxDisagreed"] += 1
        return routing | {
            "gate": Gate.G4_ESCALATION.value,
            "tier": QueueTier.LLM_TAIL.value,
            "decidedBy": None,
        }

    counters["forwarded"] += 1
    if decision.contradiction_flag:
        counters["contraFlagged"] += 1
    return routing | {
        "gate": Gate.G2_CORROBORATION.value,
        "tier": QueueTier.CASCADE.value,
        "decidedBy": None,
    }


register(Gate.G1_NEUTRAL, run_g1)
