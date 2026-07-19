"""G4_ESCALATION — the LLM tail (LLD §8, §11, D-1).

Every pair that reaches here is one the encoders could not settle: a bare-vs-contexted
disagreement from G1, or a two-family disagreement from G3. Target is ≤5% of pairs, and the
hard ceiling is `maxLlmPairs` (3,000).

**This is the only gate that spends money, and every design choice here follows from that.**

*The cap is durable, not per-execution.* It is computed from the queue — how many pairs this
run has already settled with `GK_G4_LLM` — rather than from a counter that restarts at zero
on every sweeper rescue. A gate that crashed at 2,999 calls and resumed on a fresh counter
would spend the budget twice, and the run doc would show it having done so only if someone
added the two attempts up by hand.

*A failed call costs the pair, never the gate.* §11's row for `GK_E_VERTEX` is explicit —
quota and 5xx beyond backoff send the **affected pairs** to a human with `LLM_ERROR` while
the gate still SUCCEEDS. That is the partial-tail policy, and it is why the Vertex error is
caught per pair inside the loop rather than propagating out to `run_gate`. A tail that
failed the whole run because the 2,000th of 3,000 pairs hit a quota blip would throw away
1,999 verdicts that were already paid for.

*The over-cap remainder is drained, not abandoned.* Once the cap is hit the gate keeps
claiming batches and routes every remaining pair to a human with `CAP_EXCEEDED` — costing
nothing, because no call is made. Leaving them sitting at G4 would strand them: FINALIZE's
reconciliation would fail, and the next run would find pairs at a gate that had already
been marked SUCCEEDED.

*SHADOW makes zero calls.* §9's table gives SHADOW the ensemble as verdict writer, so
calling an LLM here would spend money to compare an LLM to an LLM. The pairs are recorded
`shadow.verdict = WOULD_ESCALATE_LLM` — a statement about routing, which is the only true
thing the row can say — and the tail's *size* is still measured, which is the number the
cutover decision actually needs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from gatekeeper.clients.vertex import VertexClient, estimate_spend_usd
from gatekeeper.enums import (
    SHADOW_WOULD_ESCALATE_LLM,
    EscalationReason,
    Gate,
    JudgeMode,
    Method,
    QueueTier,
)
from gatekeeper.errors import ErrorCode, Neo4jUnavailableError, VertexError
from gatekeeper.gates.prompts import ClaimCard, g4_judge_prompt, judge_rubric, parse_g4_response
from gatekeeper.integration import resolve_subject_id
from gatekeeper.logging import get_logger
from gatekeeper.worker.edges import EdgeVerdict, EdgeWriter
from gatekeeper.worker.gates import GateContext, register
from gatekeeper.worker.hydrate import claim_cards
from gatekeeper.worker.queue import PairQueue, QueuedPair

__all__ = ["G4_COUNTERS", "run_g4"]

log = get_logger(__name__)

G4_COUNTERS: tuple[str, ...] = (
    "seen",
    "decided",
    "capExceeded",
    "llmError",
    "llmCalls",
    "promptTokens",
    "outputTokens",
    "shadowSuppressed",
)
"""§8 names no counters for G4, so these are chosen; all written at zero per §7.1.

They answer the three questions an operator asks about a tail: what did it settle
(``decided``), what did it cost (``llmCalls`` + the two token counts, which is what
``llmSpendUsd`` is derived from and what makes the derivation checkable against a bill), and
what did it fail to settle and why — ``capExceeded`` and ``llmError`` are counted apart
because they are different problems with different fixes. ``shadowSuppressed`` is the tail's
*size* in a mode where its cost is zero, which is the number a cutover decision needs.

``llmSpendUsd`` is deliberately **not** a gate counter: §7.1 puts it in run-level ``totals``,
and FINALIZE is what writes it.
"""

SPEND_COUNTER = "llmSpendUsd"
"""Carried on the gate entry too, so a mid-run doc shows spend before FINALIZE lands.

§7.1's home for the number is ``totals``, and that stays authoritative; this is the same
value visible one level down while the gate is still running, which is when an operator
watching a bill actually wants it.
"""


def run_g4(context: GateContext) -> dict[str, Any]:
    """Execute the escalation tail over every pair sitting at G4. Returns the gate counters."""
    run = context.run
    config = context.config
    snapshot_g4 = run.config_snapshot.get("g4") or {}

    # From the frozen snapshot, never live config: a run that started on one lite row must
    # finish on it, exactly as the encoder gates' artifacts are frozen (LLD §8).
    model = str(snapshot_g4.get("llmModel") or "")
    max_llm_pairs = int(snapshot_g4.get("maxLlmPairs") or 0)
    thinking_budget = snapshot_g4.get("thinkingBudget")

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

    counters: Counter[str] = Counter(dict.fromkeys(G4_COUNTERS, 0))
    counters[SPEND_COUNTER] = 0.0
    shadow = run.judge_mode is JudgeMode.SHADOW

    # Already spent on this run, by previous attempts of this gate. See the module docstring
    # on why this is read from the queue rather than kept in a counter.
    spent = queue.count_decided_by(run.stage3_run_id, Method.GK_G4_LLM.value)
    if spent:
        log.info(
            "resuming the tail with budget already spent",
            fields={"priorLlmPairs": spent, "maxLlmPairs": max_llm_pairs},
        )

    client: VertexClient | None = None
    rubric: str | None = None
    batch_size = config.get_int("gatekeeper.queue.batch-size")

    while True:
        batch = queue.claim_batch(
            run.stage3_run_id,
            Gate.G4_ESCALATION,
            lease_owner=context.lease_owner,
            limit=batch_size,
        )
        if not batch:
            break

        if shadow:
            _record_shadow(batch, run, subject_id, model, edges, queue, counters)
            continue

        # Built on first use and only when a call is actually going to happen: a resumed
        # gate with nothing left to judge must not construct credentials to discover it.
        if client is None:
            client = context.llm_factory(config, model, thinking_budget)
            rubric = judge_rubric(
                context.client,
                collection=config.get_str("gatekeeper.integration.prompts-collection"),
            )

        cards = _cards_for(context, batch)
        spent = _judge_batch(
            batch,
            cards,
            client,
            rubric or "",
            run=run,
            subject_id=subject_id,
            model=model,
            edges=edges,
            queue=queue,
            counters=counters,
            spent=spent,
            max_llm_pairs=max_llm_pairs,
            config=config,
        )
        log.info(
            "G4 batch complete",
            fields={"batch": len(batch), "seen": counters["seen"], "llmPairs": spent},
        )

    if client is not None and not client.live:
        # Loud, because a run doc showing `decided` on a tail that never called anything is
        # exactly the thing someone would later read as evidence the tail works.
        log.warning(
            "G4 ran against the dry-run double; no verdict here came from a model",
            fields={"decided": counters["decided"]},
        )

    return dict(counters)


def _cards_for(context: GateContext, batch: Sequence[QueuedPair]) -> dict[str, ClaimCard]:
    """The claim cards for a batch, or a hard failure if the graph has lost one.

    A card is what the prompt renders; without it there is nothing to ask about. Same
    posture as G1's missing claim text — stop and let the operator retrigger, rather than
    ask the model to judge a blank.
    """
    claim_ids = [pair.claim_a_id for pair in batch] + [pair.claim_b_id for pair in batch]
    cards = claim_cards(context.reader(), claim_ids)

    missing = [
        pair.pair_id
        for pair in batch
        if pair.claim_a_id not in cards or pair.claim_b_id not in cards
    ]
    if missing:
        raise Neo4jUnavailableError(
            f"{len(missing)} pair(s) at G4 have no claim card in the graph; first: {missing[0]}"
        )
    return cards


def _judge_batch(
    batch: Sequence[QueuedPair],
    cards: dict[str, ClaimCard],
    client: VertexClient,
    rubric: str,
    *,
    run: Any,
    subject_id: str,
    model: str,
    edges: EdgeWriter,
    queue: PairQueue,
    counters: Counter[str],
    spent: int,
    max_llm_pairs: int,
    config: Any,
) -> int:
    """One call per pair, with the cap and the per-pair error policy. Returns the new spend."""
    verdicts: list[EdgeVerdict] = []
    routes: list[tuple[QueuedPair, dict[str, Any]]] = []
    attempt = run.gate(Gate.G4_ESCALATION).attempt

    for pair in batch:
        counters["seen"] += 1

        if max_llm_pairs and spent >= max_llm_pairs:
            # No call. The remainder is drained to a human at zero cost — see the module
            # docstring on why they are drained rather than left sitting at the gate.
            _reject(
                pair,
                EscalationReason.CAP_EXCEEDED,
                run=run,
                subject_id=subject_id,
                model=model,
                attempt=attempt,
                verdicts=verdicts,
                routes=routes,
            )
            counters["capExceeded"] += 1
            if counters["capExceeded"] == 1:
                # §12's `gatekeeper/cap_exceeded` log metric filters on this exact
                # errorCode. Emitted once per execution, on the transition — one line per
                # over-cap pair would turn a 3,000-pair overflow into 3,000 alerts.
                log.error(
                    "G4 hit maxLlmPairs; the remainder goes to the human queue",
                    fields={
                        "errorCode": ErrorCode.GK_E_CAP_EXCEEDED.value,
                        "maxLlmPairs": max_llm_pairs,
                        "llmPairs": spent,
                    },
                )
            continue

        prompt = g4_judge_prompt(
            cards[pair.claim_a_id],
            cards[pair.claim_b_id],
            rubric=rubric,
            with_context=pair.with_context,
        )

        try:
            response = client.generate(prompt)
            verdict = parse_g4_response(response.text)
        except (VertexError, ValueError) as exc:
            # Per pair, never per gate: §11's partial-tail policy. A parse failure and a
            # quota failure land in the same place because the operator's move is the same
            # — the pair is reviewed by a human, and the run still completes.
            log.warning(
                "G4 could not settle a pair; routing it to a human",
                fields={
                    "errorCode": ErrorCode.GK_E_VERTEX.value,
                    "pairId": pair.pair_id,
                    "detail": str(exc),
                },
            )
            _reject(
                pair,
                EscalationReason.LLM_ERROR,
                run=run,
                subject_id=subject_id,
                model=model,
                attempt=attempt,
                verdicts=verdicts,
                routes=routes,
            )
            counters["llmError"] += 1
            # A failed call still consumed a call's worth of quota, so it counts against
            # the cap — the ceiling is about spend, not about successes.
            spent += 1
            counters["llmCalls"] += 1
            continue

        spent += 1
        counters["llmCalls"] += 1
        counters["decided"] += 1
        counters["promptTokens"] += response.prompt_tokens
        counters["outputTokens"] += response.output_tokens
        counters[SPEND_COUNTER] += estimate_spend_usd(
            response.prompt_tokens,
            response.output_tokens,
            input_per_million=config.get_float("gatekeeper.gates.g4.usd-per-million-input"),
            output_per_million=config.get_float("gatekeeper.gates.g4.usd-per-million-output"),
        )

        verdicts.append(
            EdgeVerdict(
                claim_a_id=pair.claim_a_id,
                claim_b_id=pair.claim_b_id,
                subject_id=subject_id,
                intake_id=run.intake_id,
                stage3_run_id=run.stage3_run_id,
                run_request_id=run.run_request_id,
                slot="g4",
                method=Method.GK_G4_LLM,
                judge_model=model,
                # No probabilities: §7.2's `stageScores` is the encoders' full-precision
                # calibration corpus, and an LLM's self-reported confidence is not a
                # comparable number. What the tail produces is prose, and prose has its
                # own fields.
                stage_scores={},
                verdict=verdict.relation,
                confidence=verdict.confidence,
                rationale=verdict.rationale,
                temporal_note=verdict.temporal_note,
                with_context=pair.with_context,
                attempt=attempt,
            )
        )
        routes.append(
            (
                pair,
                {
                    "gate": None,
                    "tier": QueueTier.LLM_TAIL.value,
                    "decidedBy": Method.GK_G4_LLM.value,
                    "gkRunRequestId": run.run_request_id,
                },
            )
        )

    edges.write_all(verdicts, run.judge_mode)
    queue.route_all(routes)
    return spent


def _reject(
    pair: QueuedPair,
    reason: EscalationReason,
    *,
    run: Any,
    subject_id: str,
    model: str,
    attempt: int,
    verdicts: list[EdgeVerdict],
    routes: list[tuple[QueuedPair, dict[str, Any]]],
) -> None:
    """Send one pair to the human queue with the reason it could not be settled.

    The edge row is written even though it carries no verdict: it is the record of *why*
    this pair needs a human, and the §11.10 queue reads `escalationReason` to say so. The
    queue row leaves with `decidedBy` unset — nothing decided it — which is the same
    contract G3 keeps for a contradiction candidate.
    """
    verdicts.append(
        EdgeVerdict(
            claim_a_id=pair.claim_a_id,
            claim_b_id=pair.claim_b_id,
            subject_id=subject_id,
            intake_id=run.intake_id,
            stage3_run_id=run.stage3_run_id,
            run_request_id=run.run_request_id,
            slot="g4",
            method=Method.GK_G4_LLM,
            judge_model=model,
            stage_scores={},
            verdict=None,
            escalation_reason=reason,
            with_context=pair.with_context,
            attempt=attempt,
        )
    )
    routes.append(
        (
            pair,
            {
                "gate": None,
                "tier": QueueTier.HUMAN.value,
                "decidedBy": None,
                "gkRunRequestId": run.run_request_id,
            },
        )
    )


def _record_shadow(
    batch: Sequence[QueuedPair],
    run: Any,
    subject_id: str,
    model: str,
    edges: EdgeWriter,
    queue: PairQueue,
    counters: Counter[str],
) -> None:
    """SHADOW: record what the tail *would* have done, and call nothing (LLD §8, §9)."""
    attempt = run.gate(Gate.G4_ESCALATION).attempt
    verdicts: list[EdgeVerdict] = []
    routes: list[tuple[QueuedPair, dict[str, Any]]] = []

    for pair in batch:
        counters["seen"] += 1
        counters["shadowSuppressed"] += 1
        verdicts.append(
            EdgeVerdict(
                claim_a_id=pair.claim_a_id,
                claim_b_id=pair.claim_b_id,
                subject_id=subject_id,
                intake_id=run.intake_id,
                stage3_run_id=run.stage3_run_id,
                run_request_id=run.run_request_id,
                slot="g4",
                method=Method.GK_G4_LLM,
                judge_model=model,
                stage_scores={},
                shadow_verdict=SHADOW_WOULD_ESCALATE_LLM,
                with_context=pair.with_context,
                attempt=attempt,
            )
        )
        routes.append(
            (
                pair,
                {
                    "gate": None,
                    "tier": QueueTier.LLM_TAIL.value,
                    # Nothing decided this pair — the ensemble is the verdict writer in
                    # SHADOW mode (§9), and claiming otherwise on our own queue row would
                    # be the split-brain this mode exists to avoid.
                    "decidedBy": None,
                    "gkRunRequestId": run.run_request_id,
                },
            )
        )

    edges.write_all(verdicts, run.judge_mode)
    queue.route_all(routes)


register(Gate.G4_ESCALATION, run_g4)
