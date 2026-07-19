"""Writing verdicts to ``stage3_edges`` (LLD §7.2, §10).

`stage3_edges` is vishwamitra's collection and lakshmana is an additive writer in it: it
creates rows whose ``method`` is one of the ``GK_*`` values and touches nothing else.
`ENSEMBLE` / `RULE` / `EMBEDDING` rows are another judge's output and another repo's to
manage (D-5), and the purge below is written to make that impossible to get wrong.

**Doc ids are deterministic, so a write is an upsert.** The worker job runs with
`maxRetries 1` precisely because "verdict writes are idempotent upserts by pair key"
(LLD §13) — a redelivered gate, a sweeper rescue, or a mid-batch crash all re-write the
same row rather than duplicating it. The key mirrors vishwamitra's own shape
(`{low}|{high}|{variant}|{stamp}`) with the gatekeeper's gate as the stamp, so lakshmana's
rows sort alongside the ensemble's and can never collide with one.

**`stageScores` keeps full precision.** §7.2 is explicit about why: it is what makes
offline threshold recalibration possible without re-running inference, it powers the SHADOW
disagreement report, and it is the training corpus if VA-79 ever becomes a fine-tune. So the
raw probabilities go in unrounded.

**SHADOW writes `shadow.*` and nothing else** (LLD §9). In that mode vishwamitra's ensemble
is still the verdict writer, and a cascade row that looked authoritative would be a
split-brain bug rather than a report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from gatekeeper.enums import GATEKEEPER_METHODS, EscalationReason, JudgeMode, Method, Verdict
from gatekeeper.logging import get_logger
from gatekeeper.scoring.encoder import NliScores

__all__ = [
    "DEFAULT_COLLECTION",
    "GATEKEEPER_STAMP",
    "EdgeVerdict",
    "EdgeWriter",
    "edge_key",
    "g1_stage_scores",
]

log = get_logger(__name__)

DEFAULT_COLLECTION = "stage3_edges"

_WRITE_BATCH_LIMIT = 400


GATEKEEPER_STAMP = "GK"
"""Our slot in vishwamitra's ``promptStamp`` position.

Not a per-gate value: §7.2 gives one row per pair carrying ``stageScores: {g1, g2, g3}``,
so all four gates write the *same* document and the stamp identifies the judge, not the
gate. It is what keeps a gatekeeper row from ever colliding with an ensemble one, whose
stamp is a real `<impl>:<version>:<hash>` prompt stamp.
"""


def edge_key(claim_a_id: str, claim_b_id: str, *, with_context: bool = False) -> str:
    """The deterministic ``stage3_edges`` doc id for a pair's gatekeeper row.

    Same four-part shape vishwamitra uses — pair, variant, stamp — so the collection stays
    legible to a human reading it.
    """
    low, high = sorted((claim_a_id, claim_b_id))
    variant = "ctx" if with_context else "bare"
    return f"{low}|{high}|{variant}|{GATEKEEPER_STAMP}"


def g1_stage_scores(
    forward: NliScores,
    backward: NliScores,
    context_forward: NliScores | None = None,
    context_backward: NliScores | None = None,
) -> dict[str, float]:
    """``stageScores.g1`` exactly as §7.2 names its fields.

    ``ctxDelta`` is the §7.2 optional: how far the contexted neutral probability moved from
    the bare one. It is the number the dual-eval disagreement is *about*, so storing it
    saves recomputing it from four other fields during a shadow review.
    """
    scores = {
        "entFwd": forward.entailment,
        "entBwd": backward.entailment,
        "neuFwd": forward.neutral,
        "neuBwd": backward.neutral,
        "conFwd": forward.contradiction,
        "conBwd": backward.contradiction,
    }
    if context_forward is not None and context_backward is not None:
        scores["ctxDelta"] = max(
            abs(context_forward.neutral - forward.neutral),
            abs(context_backward.neutral - backward.neutral),
        )
    return scores


@dataclass(frozen=True, slots=True)
class EdgeVerdict:
    """One gate's contribution to one pair's edge row.

    Not "a verdict" but "what this gate learned": a gate that forwards still writes, because
    its ``stageScores`` are the whole point of §7.2's full-precision store, and because the
    next gate's cross-check reads G1's contradiction logits back out of it. ``verdict`` is
    ``None`` for a forwarded pair, and the ``relation`` field is then left untouched rather
    than nulled — a later gate will set it.
    """

    claim_a_id: str
    claim_b_id: str
    subject_id: str
    intake_id: str
    stage3_run_id: str
    run_request_id: str
    slot: str
    """Which ``stageScores`` sub-map this gate owns: ``g1`` / ``g2`` / ``g3``."""

    method: Method
    judge_model: str
    stage_scores: dict[str, Any]
    verdict: Verdict | None = None
    confidence: float | None = None
    escalation_reason: EscalationReason | None = None
    truncated: bool = False
    with_context: bool = False
    attempt: int = 1

    @property
    def key(self) -> str:
        return edge_key(self.claim_a_id, self.claim_b_id, with_context=self.with_context)

    def to_firestore(self, judge_mode: JudgeMode) -> dict[str, Any]:
        """The document body for this mode.

        Written with ``merge=True``, and the nested ``stageScores`` map merges with it, so a
        gate writes only its own slot and the earlier gates' numbers survive. That is what
        makes §7.2's ``{g1, g2, g3}`` shape reachable from four independently-scheduled
        gate executions.

        In GATEKEEPER mode the verdict is authoritative and lands on the row's own fields.
        In SHADOW mode identical content goes under ``shadow`` and the row asserts nothing
        about the pair — §9's "cascade → ``shadow.*`` only".
        """
        low, high = sorted((self.claim_a_id, self.claim_b_id))

        # Identity is written in both modes: it is what makes the row findable, and it
        # asserts nothing about who judged the pair.
        document: dict[str, Any] = {
            "subjectId": self.subject_id,
            "claimIdLow": low,
            "claimIdHigh": high,
            "withContext": self.with_context,
            "intakeId": self.intake_id,
            "stage3RunId": self.stage3_run_id,
            "runRequestId": self.run_request_id,
            "gkRunRequestId": self.run_request_id,
            "method": self.method.value,
            "attempt": self.attempt,
            "stageScores": {self.slot: self.stage_scores},
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }

        if judge_mode is JudgeMode.SHADOW:
            document["shadow"] = {
                "verdict": self.verdict.value if self.verdict else None,
                "method": self.method.value,
                "stageScores": {self.slot: self.stage_scores},
                "gkRunRequestId": self.run_request_id,
                "escalationReason": (
                    self.escalation_reason.value if self.escalation_reason else None
                ),
            }
            return document

        document["judgeModel"] = self.judge_model
        document["truncated"] = self.truncated
        if self.verdict is not None:
            document["relation"] = self.verdict.value
        if self.confidence is not None:
            document["confidence"] = self.confidence
        if self.escalation_reason is not None:
            document["escalationReason"] = self.escalation_reason.value
        return document


class EdgeWriter:
    """Batched writes and the FROM_START purge over ``stage3_edges``."""

    def __init__(self, client: firestore.Client, *, collection: str = DEFAULT_COLLECTION) -> None:
        self._client = client
        self._collection = client.collection(collection)

    def write_all(self, verdicts: list[EdgeVerdict], judge_mode: JudgeMode) -> int:
        """Upsert a batch of verdicts. Returns the number written."""
        written = 0
        batch = self._client.batch()
        pending = 0

        for verdict in verdicts:
            batch.set(
                self._collection.document(verdict.key),
                verdict.to_firestore(judge_mode),
                merge=True,
            )
            written += 1
            pending += 1
            if pending >= _WRITE_BATCH_LIMIT:
                batch.commit()
                batch = self._client.batch()
                pending = 0

        if pending:
            batch.commit()
        return written

    def purge(self, stage3_run_id: str) -> int:
        """Delete this run's gatekeeper rows — G1's first act on FROM_START (LLD §10).

        Two guards, because this is the one method in the codebase that deletes another
        service's data:

        * the query is scoped to ``stage3RunId``, a field only lakshmana writes on these
          rows, so an ensemble row cannot be in the result set to begin with;
        * every candidate's ``method`` is re-checked against ``GATEKEEPER_METHODS`` before
          the delete is queued, so a row that somehow carried our run id but not our method
          survives.

        `shadow.*` history is untouched by design (D-5) — a shadow row's method is a
        ``GK_*`` value, so it is deleted with the rest of the run's output and re-created
        by the re-run; what D-5 protects is the *ensemble's* history, which is never in
        scope here. Idempotent: a second purge finds nothing to delete.
        """
        allowed = {method.value for method in GATEKEEPER_METHODS}
        deleted = 0
        batch = self._client.batch()
        pending = 0

        query = self._collection.where(filter=FieldFilter("stage3RunId", "==", stage3_run_id))
        for snapshot in query.stream():
            document = snapshot.to_dict() or {}
            if str(document.get("method", "")) not in allowed:
                log.warning(
                    "refusing to purge a non-gatekeeper row",
                    fields={"docId": snapshot.id, "method": document.get("method")},
                )
                continue
            batch.delete(snapshot.reference)
            deleted += 1
            pending += 1
            if pending >= _WRITE_BATCH_LIMIT:
                batch.commit()
                batch = self._client.batch()
                pending = 0

        if pending:
            batch.commit()

        log.info(
            "purged gatekeeper verdicts",
            fields={"stage3RunId": stage3_run_id, "deleted": deleted},
        )
        return deleted
