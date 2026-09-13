"""G3_CONTRADICTION — the two-family cross-check (LLD §8).

G3 replaces the old judge's 5-vote self-consistency with **cross-architecture agreement**:
one model can be confidently wrong five times, two models from different training lineages
agreeing is a different kind of evidence. Family A is G1's NLI pass, family B is this gate's
own — and family A is *read back off the edge row*, never re-inferred. §8 says its logits
"are already in `stageScores`", and a cross-check that quietly recomputes one of its two
families has silently become a self-check.

**This gate never finalizes CONTRADICTS, and that is a property, not a preference.** The
cascade's job is to find contradiction candidates; a human confirms them, exactly as today.
So a two-family agreement produces `tier = HUMAN` with no `decidedBy` — the pair leaves the
cascade *unjudged* — while the edge row carries `relation = CONTRADICTS` with the
`CONTRADICTION_SIGNAL` escalation reason.

Those two facts look contradictory and are not. `relation` on a `stage3_edges` row is a
*pair-level candidate*, not a verdict anyone acts on: vishwamitra's `FactAssembler` lifts
CORROBORATES/CONTRADICTS pair rows to Fact edges and stamps every contradiction
`reviewStatus = PROPOSED`, which is precisely what the §11.10 human queue reads. Writing
anything else — or nothing — would make a confirmed contradiction *invisible* to the queue
that exists to review it, which is the single most expensive failure this system can have.
Not writing the relation would not make the cascade more cautious; it would make it silent.
`decidedBy` is the field that says "the cascade settled this", and G3 never sets it on a
contradiction candidate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from gatekeeper.config import GateBinding
from gatekeeper.enums import Gate, Method, QueueTier, Verdict
from gatekeeper.errors import Neo4jUnavailableError
from gatekeeper.gates.decisions import (
    Decision,
    Disposition,
    G1Scores,
    G3Scores,
    Thresholds,
    decide_g3,
)
from gatekeeper.integration import resolve_subject_id
from gatekeeper.logging import get_logger
from gatekeeper.models.loader import artifact_for_binding
from gatekeeper.scoring.encoder import scorer_for_binding
from gatekeeper.worker.edges import (
    EdgeVerdict,
    EdgeWriter,
    edge_key,
    g1_scores_from_stage_scores,
    g3_stage_scores,
)
from gatekeeper.worker.gates import GateContext, register
from gatekeeper.worker.hydrate import claim_texts
from gatekeeper.worker.queue import PairQueue, QueuedPair

__all__ = ["G3_COUNTERS", "run_g3"]

log = get_logger(__name__)

G3_COUNTERS: tuple[str, ...] = (
    "seen",
    "neutral",
    "humanRouted",
    "escalated",
    "truncated",
)
"""One total, one per outcome, written at zero rather than omitted (§7.1).

``humanRouted`` is deliberately not called ``contradicts``: nothing here decides that.
"""


def run_g3(context: GateContext) -> dict[str, Any]:
    """Execute G3 over every pair sitting at it. Returns the gate counters."""
    run = context.run
    config = context.config
    binding = GateBinding.from_snapshot(run.config_snapshot, Gate.G3_CONTRADICTION)
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
        raise Neo4jUnavailableError(
            f"stage3 run {run.stage3_run_id!r} names no subjectId; cannot judge its pairs"
        )

    counters: Counter[str] = Counter(dict.fromkeys(G3_COUNTERS, 0))
    reader = context.reader()
    scorer = None
    batch_size = config.get_int("gatekeeper.queue.batch-size")

    while True:
        # Renew-or-stop before claiming more (VA-159/B2): keep the lease ahead of the
        # sweeper while alive, and abort without writing if a rescue has already taken it.
        if not context.renew_lease():
            log.warning(
                "G3 no longer holds its gate lease; stopping without claiming more work",
                fields={"leaseOwner": context.lease_owner},
            )
            break

        batch = queue.claim_batch(
            run.stage3_run_id,
            Gate.G3_CONTRADICTION,
            lease_owner=context.lease_owner,
            limit=batch_size,
        )
        if not batch:
            break

        if scorer is None:
            scorer = _build_scorer(context, binding, batch_size)

        family_a = _read_family_a(edges, batch)
        family_b = _score_batch(scorer, reader, batch)
        _write_outcomes(
            batch,
            family_a,
            family_b,
            thresholds,
            run,
            subject_id,
            binding,
            edges,
            queue,
            counters,
        )
        log.info("G3 batch complete", fields={"batch": len(batch), "seen": counters["seen"]})

    return dict(counters)


def _build_scorer(context: GateContext, binding: GateBinding, batch_size: int) -> Any:
    if context.scorer_factory is not None:
        return context.scorer_factory(binding)
    artifact = artifact_for_binding(context.loader(), binding)
    return scorer_for_binding(artifact, binding, batch_size=batch_size)


def _read_family_a(edges: EdgeWriter, batch: Sequence[QueuedPair]) -> list[G1Scores]:
    """G1's logits, off the rows G1 wrote — §8's "already in ``stageScores``".

    Raises:
        Neo4jUnavailableError: if any pair has no readable ``stageScores.g1``. The gate
            stops rather than re-inferring, because a re-inferred family A would make the
            two-family agreement a single-family one without saying so, and rather than
            defaulting it, because a missing family A read as "no contradiction" would
            discard the exact pairs the cross-check exists to catch. Operationally this is
            a FROM_GATE retrigger at G1, the same runbook row as any other hydrate failure.
    """
    keys = [edge_key(p.claim_a_id, p.claim_b_id, with_context=p.with_context) for p in batch]
    stored = edges.read_stage_scores(keys, slot="g1")

    scores: list[G1Scores] = []
    missing: list[str] = []
    for pair, key in zip(batch, keys, strict=True):
        recovered = g1_scores_from_stage_scores(stored.get(key, {}))
        if recovered is None:
            missing.append(pair.pair_id)
        else:
            scores.append(recovered)

    if missing:
        raise Neo4jUnavailableError(
            f"{len(missing)} pair(s) at G3 have no stageScores.g1 to cross-check against; "
            f"first: {missing[0]}. Retrigger FROM_GATE at G1_NEUTRAL."
        )
    return scores


def _score_batch(scorer: Any, reader: Any, batch: Sequence[QueuedPair]) -> list[G3Scores]:
    """The second family's NLI pass, both directions."""
    claim_ids = [pair.claim_a_id for pair in batch] + [pair.claim_b_id for pair in batch]
    texts = claim_texts(reader, claim_ids)

    missing = [
        pair.pair_id
        for pair in batch
        if not (texts.get(pair.claim_a_id) and texts.get(pair.claim_b_id))
    ]
    if missing:
        raise Neo4jUnavailableError(
            f"{len(missing)} queued pair(s) have no claim text in the graph; first: {missing[0]}"
        )

    forward = scorer.score_nli([(texts[p.claim_a_id], texts[p.claim_b_id]) for p in batch])
    backward = scorer.score_nli([(texts[p.claim_b_id], texts[p.claim_a_id]) for p in batch])

    return [
        G3Scores(family_b_forward=forward[index], family_b_backward=backward[index])
        for index in range(len(batch))
    ]


def _write_outcomes(
    batch: Sequence[QueuedPair],
    family_a: Sequence[G1Scores],
    family_b: Sequence[G3Scores],
    thresholds: Thresholds,
    run: Any,
    subject_id: str,
    binding: GateBinding,
    edges: EdgeWriter,
    queue: PairQueue,
    counters: Counter[str],
) -> None:
    """Decide, persist, then move the queue — G1's ordering, for G1's reason."""
    verdicts: list[EdgeVerdict] = []
    routes: list[tuple[QueuedPair, dict[str, Any]]] = []

    for pair, g1, g3 in zip(batch, family_a, family_b, strict=True):
        decision = decide_g3(g1, g3, thresholds)
        counters["seen"] += 1
        if decision.truncated:
            counters["truncated"] += 1

        verdicts.append(
            EdgeVerdict(
                claim_a_id=pair.claim_a_id,
                claim_b_id=pair.claim_b_id,
                subject_id=subject_id,
                intake_id=run.intake_id,
                stage3_run_id=run.stage3_run_id,
                run_request_id=run.run_request_id,
                slot="g3",
                method=Method.GK_G3_XCHECK,
                judge_model=binding.model,
                stage_scores=g3_stage_scores(g3.family_b_forward, g3.family_b_backward),
                verdict=_edge_relation(decision),
                confidence=decision.confidence,
                escalation_reason=decision.escalation_reason,
                truncated=decision.truncated,
                with_context=pair.with_context,
                attempt=run.gate(Gate.G3_CONTRADICTION).attempt,
            )
        )
        routes.append((pair, _routing(decision, run.run_request_id, counters)))

    edges.write_all(verdicts, run.judge_mode)
    queue.route_all(routes)


def _edge_relation(decision: Decision) -> Verdict | None:
    """What lands in the row's ``relation`` — a candidate for HUMAN, the verdict otherwise.

    ``decide_g3`` deliberately leaves ``verdict`` unset on a ROUTE_HUMAN decision: no
    verdict was reached, and a decision object that claimed one would be lying to the
    replay harness. The *row* still has to say CONTRADICTS, because that is the only thing
    vishwamitra's `FactAssembler` lifts into a `PROPOSED` Fact edge and therefore the only
    way the §11.10 human queue ever sees the pair (see this module's docstring). The
    translation lives here, at the storage boundary, rather than in the shared decision
    function — which is also what keeps `decidedBy` unset on the queue row.
    """
    if decision.disposition is Disposition.ROUTE_HUMAN:
        return Verdict.CONTRADICTS
    return decision.verdict


def _routing(decision: Decision, run_request_id: str, counters: Counter[str]) -> dict[str, Any]:
    """Where a pair goes after the cross-check, and the counter that records it."""
    routing: dict[str, Any] = {"gkRunRequestId": run_request_id}

    if decision.disposition is Disposition.ROUTE_HUMAN:
        counters["humanRouted"] += 1
        return routing | {
            "gate": None,
            "tier": QueueTier.HUMAN.value,
            # Never set. `decidedBy` is what asserts the cascade settled a pair, and this
            # is the one outcome it must not claim: a human confirms the contradiction.
            "decidedBy": None,
            "contradictionFlag": True,
        }

    if decision.disposition is Disposition.DECIDED:
        counters["neutral"] += 1
        return routing | {
            "gate": None,
            "tier": QueueTier.CASCADE.value,
            "decidedBy": Method.GK_G3_XCHECK.value,
        }

    counters["escalated"] += 1
    return routing | {
        "gate": Gate.G4_ESCALATION.value,
        "tier": QueueTier.LLM_TAIL.value,
        "decidedBy": None,
    }


register(Gate.G3_CONTRADICTION, run_g3)
