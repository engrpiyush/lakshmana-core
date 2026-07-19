"""G2_CORROBORATION — the grounding gate, sees the G1 leftovers (LLD §8).

G2 is the only gate with an input the LLM judge never had. G1 and G3 ask a model to compare
two *assertions*; G2 can additionally ask whether a claim is supported by the **evidence the
other claim was extracted from** — MiniCheck's native `(document, claim)` shape. That is
grounding mode, and it is why this gate can decide CORROBORATES at all: two claims agreeing
with each other is weak, one claim being entailed by the other's source text is not.

**Contradiction-flagged pairs are not scored here at all.** §8's own G3 heading says that
gate sees "contradiction-flagged + G2 leftovers", and §8's ordering note is explicit that G2
may not finalize CORROBORATES on a flagged pair — doing so would put a contradiction signal
permanently beyond the reach of the cross-check built to adjudicate it. Since the only thing
G2 can do with such a pair is forward it, it forwards it *without inference*: running a
435 MB model to produce a number no decision may read is pure cost. They arrive at G3 with
their G1 scores untouched, which is exactly what `decide_g3` wants.

**Grounding is best-effort, and its absence is not an error.** A claim extracted from audio
has no printed excerpt behind it, so `groundingMode = AUTO` means "ground the pairs that can
be grounded" rather than "require it". The two claim-vs-claim directions are always scored,
so a pair with no excerpt anywhere is still judged — just without the evidence mode.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from gatekeeper.config import GateBinding
from gatekeeper.enums import EscalationReason, Gate, Method, QueueTier
from gatekeeper.errors import Neo4jUnavailableError
from gatekeeper.gates.decisions import Decision, Disposition, G2Scores, Thresholds, decide_g2
from gatekeeper.integration import resolve_subject_id, source_excerpts
from gatekeeper.logging import get_logger
from gatekeeper.models.loader import artifact_for_binding
from gatekeeper.scoring.encoder import GroundingScore, scorer_for_binding
from gatekeeper.worker.edges import EdgeVerdict, EdgeWriter, g2_stage_scores
from gatekeeper.worker.gates import GateContext, register
from gatekeeper.worker.hydrate import claim_texts
from gatekeeper.worker.queue import PairQueue, QueuedPair

__all__ = ["G2_COUNTERS", "GROUNDING_MODES", "run_g2"]

log = get_logger(__name__)

G2_COUNTERS: tuple[str, ...] = (
    "seen",
    "corroborates",
    "forwarded",
    "grounded",
    "flaggedPassThrough",
    "truncated",
)
"""§8 names G1's counters verbatim and leaves the rest to the build.

These mirror G1's shape — a total, one per outcome, and the diagnostics a reader needs to
trust the total. ``grounded`` and ``flaggedPassThrough`` are the two numbers that explain a
surprising ``corroborates``: how many pairs got the evidence mode, and how many were never
judged here at all. Like G1's, every key is written at zero rather than omitted (§7.1).
"""

GROUNDING_MODES = ("AUTO", "OFF")
"""``configSnapshot.g2.groundingMode``.

``AUTO`` grounds every pair that has an excerpt to ground against; ``OFF`` disables the
evidence mode entirely, which is what a bake-off run needs to measure what grounding is
actually worth. There is deliberately no ``REQUIRED``: it would mean dropping pairs the
graph simply has no printed source for, and those pairs still deserve a verdict.
"""


def run_g2(context: GateContext) -> dict[str, Any]:
    """Execute G2 over every pair sitting at it. Returns the gate counters."""
    run = context.run
    config = context.config
    binding = GateBinding.from_snapshot(run.config_snapshot, Gate.G2_CORROBORATION)
    thresholds = Thresholds.from_snapshot(run.config_snapshot)
    grounding_mode = _grounding_mode(run.config_snapshot)

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

    counters: Counter[str] = Counter(dict.fromkeys(G2_COUNTERS, 0))
    reader = context.reader()
    scorer = None
    batch_size = config.get_int("gatekeeper.queue.batch-size")

    while True:
        batch = queue.claim_batch(
            run.stage3_run_id,
            Gate.G2_CORROBORATION,
            lease_owner=context.lease_owner,
            limit=batch_size,
        )
        if not batch:
            break

        judged = [pair for pair in batch if not pair.contradiction_flag]
        flagged = [pair for pair in batch if pair.contradiction_flag]

        if flagged:
            _pass_through(flagged, run.run_request_id, queue, counters)

        if judged:
            if scorer is None:
                scorer = _build_scorer(context, binding, batch_size)
            scores = _score_batch(
                scorer, reader, context, judged, grounding_mode=grounding_mode, counters=counters
            )
            _write_outcomes(
                judged, scores, thresholds, run, subject_id, binding, edges, queue, counters
            )

        log.info(
            "G2 batch complete",
            fields={"batch": len(batch), "flagged": len(flagged), "seen": counters["seen"]},
        )

    return dict(counters)


def _grounding_mode(snapshot: dict[str, Any]) -> str:
    """The frozen grounding mode, defaulting to ``AUTO`` for anything unrecognised.

    An unknown value degrades to the default rather than failing the gate: it costs the
    evidence mode nothing to be on, and a typo in one config key should not strand a run
    that has already scored its G1 pairs.
    """
    raw = str((snapshot.get("g2") or {}).get("groundingMode") or "AUTO").upper()
    if raw not in GROUNDING_MODES:
        log.warning(
            "unknown groundingMode; falling back to AUTO",
            fields={"found": raw, "known": list(GROUNDING_MODES)},
        )
        return "AUTO"
    return raw


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """One pair's G2 numbers: what the decision reads, and what only the row keeps.

    :class:`~gatekeeper.gates.decisions.G2Scores` is the frozen input `decide_g2` sees and
    is shared verbatim with the replay harness, so it carries exactly the numbers the
    decision consults — one ``grounded``, the stronger orientation. §7.2 wants everything
    that was *computed* on the row, including the orientation that lost, so the two travel
    together to the edge writer instead of being smuggled onto the decision contract.
    """

    scores: G2Scores
    grounded_forward: GroundingScore | None = None
    grounded_backward: GroundingScore | None = None


def _build_scorer(context: GateContext, binding: GateBinding, batch_size: int) -> Any:
    if context.scorer_factory is not None:
        return context.scorer_factory(binding)
    artifact = artifact_for_binding(context.loader(), binding)
    return scorer_for_binding(artifact, binding, batch_size=batch_size)


def _pass_through(
    flagged: Sequence[QueuedPair],
    run_request_id: str,
    queue: PairQueue,
    counters: Counter[str],
) -> None:
    """Move contradiction-flagged pairs to G3 untouched — no inference, no edge write.

    No ``stageScores.g2`` is written for them, and that absence is the honest record: this
    gate did not judge these pairs. Inventing a support score no decision consulted would
    put a number in the calibration corpus that never influenced anything.
    """
    counters["flaggedPassThrough"] += len(flagged)
    counters["forwarded"] += len(flagged)
    queue.route_all(
        (
            pair,
            {
                "gate": Gate.G3_CONTRADICTION.value,
                "tier": QueueTier.CASCADE.value,
                "decidedBy": None,
                "gkRunRequestId": run_request_id,
            },
        )
        for pair in flagged
    )


def _score_batch(
    scorer: Any,
    reader: Any,
    context: GateContext,
    batch: Sequence[QueuedPair],
    *,
    grounding_mode: str,
    counters: Counter[str],
) -> list[ScoredPair]:
    """Support in both claim-vs-claim directions, plus claim-vs-evidence where possible."""
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

    # `(document, claim)`: the premise is the thing doing the supporting. For the plain
    # directions that is the other claim's text, which is the weaker question grounding
    # mode exists to improve on.
    forward = scorer.score_grounding([(texts[p.claim_b_id], texts[p.claim_a_id]) for p in batch])
    backward = scorer.score_grounding([(texts[p.claim_a_id], texts[p.claim_b_id]) for p in batch])

    grounded_forward: dict[int, GroundingScore] = {}
    grounded_backward: dict[int, GroundingScore] = {}
    if grounding_mode == "AUTO":
        grounded_forward, grounded_backward = _score_grounded(scorer, context, batch, texts)

    scored: list[ScoredPair] = []
    for index in range(len(batch)):
        pair_forward = grounded_forward.get(index)
        pair_backward = grounded_backward.get(index)
        available = [score for score in (pair_forward, pair_backward) if score is not None]
        if available:
            counters["grounded"] += 1
        scored.append(
            ScoredPair(
                scores=G2Scores(
                    forward=forward[index],
                    backward=backward[index],
                    # `decide_g2` takes one grounded number and argmaxes over three
                    # candidates. Handing it the stronger orientation keeps that contract
                    # while both orientations still reach `stageScores` for calibration.
                    grounded=max(available, key=lambda score: score.support) if available else None,
                ),
                grounded_forward=pair_forward,
                grounded_backward=pair_backward,
            )
        )
    return scored


def _score_grounded(
    scorer: Any,
    context: GateContext,
    batch: Sequence[QueuedPair],
    texts: dict[str, str],
) -> tuple[dict[int, GroundingScore], dict[int, GroundingScore]]:
    """Score each claim against the *other* claim's source excerpt, where one exists."""
    excerpts = source_excerpts(
        context.client,
        context.config,
        [pair.claim_a_id for pair in batch] + [pair.claim_b_id for pair in batch],
    )
    if not excerpts:
        return {}, {}

    forward_index, forward_pairs = [], []
    backward_index, backward_pairs = [], []
    for index, pair in enumerate(batch):
        # Forward: is claim A supported by the evidence behind claim B?
        if pair.claim_b_id in excerpts:
            forward_index.append(index)
            forward_pairs.append((excerpts[pair.claim_b_id], texts[pair.claim_a_id]))
        if pair.claim_a_id in excerpts:
            backward_index.append(index)
            backward_pairs.append((excerpts[pair.claim_a_id], texts[pair.claim_b_id]))

    forward = (
        dict(zip(forward_index, scorer.score_grounding(forward_pairs), strict=True))
        if forward_pairs
        else {}
    )
    backward = (
        dict(zip(backward_index, scorer.score_grounding(backward_pairs), strict=True))
        if backward_pairs
        else {}
    )
    return forward, backward


def _write_outcomes(
    batch: Sequence[QueuedPair],
    scored: Sequence[ScoredPair],
    thresholds: Thresholds,
    run: Any,
    subject_id: str,
    binding: GateBinding,
    edges: EdgeWriter,
    queue: PairQueue,
    counters: Counter[str],
) -> None:
    """Decide, persist the verdicts, then move the queue — G1's ordering, for G1's reason."""
    verdicts: list[EdgeVerdict] = []
    routes: list[tuple[QueuedPair, dict[str, Any]]] = []

    for pair, entry in zip(batch, scored, strict=True):
        pair_scores = entry.scores
        decision = decide_g2(pair_scores, thresholds)
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
                slot="g2",
                method=Method.GK_G2_GROUNDING,
                judge_model=binding.model,
                stage_scores=g2_stage_scores(
                    pair_scores.forward,
                    pair_scores.backward,
                    entry.grounded_forward,
                    entry.grounded_backward,
                ),
                verdict=decision.verdict,
                confidence=decision.confidence,
                escalation_reason=decision.escalation_reason,
                truncated=decision.truncated,
                with_context=pair.with_context,
                attempt=run.gate(Gate.G2_CORROBORATION).attempt,
            )
        )
        routes.append((pair, _routing(decision, run.run_request_id, counters)))

    edges.write_all(verdicts, run.judge_mode)
    queue.route_all(routes)


def _routing(decision: Decision, run_request_id: str, counters: Counter[str]) -> dict[str, Any]:
    """Where a judged pair goes next, and the counter that records it."""
    routing: dict[str, Any] = {"gkRunRequestId": run_request_id}

    if decision.disposition is Disposition.DECIDED:
        counters["corroborates"] += 1
        return routing | {
            "gate": None,
            "tier": QueueTier.CASCADE.value,
            "decidedBy": Method.GK_G2_GROUNDING.value,
        }

    if decision.disposition is Disposition.ESCALATE_G4:
        # Not reachable from `decide_g2` today, which only decides or forwards. Handled
        # anyway so that adding an escalation to the decision function is a one-line
        # change there rather than a silent misroute here.
        if decision.escalation_reason is not EscalationReason.CONTRADICTION_SIGNAL:
            counters["forwarded"] += 1
        return routing | {
            "gate": Gate.G4_ESCALATION.value,
            "tier": QueueTier.LLM_TAIL.value,
            "decidedBy": None,
        }

    counters["forwarded"] += 1
    return routing | {
        "gate": Gate.G3_CONTRADICTION.value,
        "tier": QueueTier.CASCADE.value,
        "decidedBy": None,
    }


register(Gate.G2_CORROBORATION, run_g2)
