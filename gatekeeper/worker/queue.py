"""The cascade's pair queue — lakshmana-owned routing state (LLD §7.3, §7.4).

**Where this diverges from the LLD, and why.** §7.3 describes the judge queue as a
Firestore *collection* that lakshmana adds `tier` / `gate` / `leaseOwner` fields to. It is
not one: in vishwamitra the queue is a Neo4j relationship —
`(:Claim)-[:JUDGE_QUEUED {rank, withContext, humanAsserted, status}]->(:Claim)`, written
wholesale by MATCH (`Stage3GraphRepository.applyMatchOutcome`). Lakshmana's graph access is
read-only by construction (D-8: RBAC user, `READ_ACCESS` session), so the routing fields
§7.3 asks for cannot be written where §7.3 puts them.

So the routing lives here instead, in a collection lakshmana owns outright, seeded from the
graph at G1. Every property §7.3 actually wanted is preserved — per-pair tier, current gate,
lease-based batch claiming, `decidedBy` — and two constraints are preserved with it that the
literal reading would have broken: the graph keeps its single writer, and vishwamitra's
queue stays exactly as MATCH left it, so a rollback to `JudgeMode = LLM` needs nothing
undone. The ownership boundary of §7.4 is unchanged in substance: lakshmana writes its own
routing, never vishwamitra's graph.

**The doc id is the pair, not the run.** ``{stage3RunId}|{claimIdLow}|{claimIdHigh}`` —
so seeding is an idempotent upsert, a FROM_START purge resets rows rather than accumulating
them, and a redelivered gate cannot fork one pair into two queue entries.

**Two leases, doing different jobs.** The gate lease on the run doc admits one worker per
gate; the pair lease here is a *cursor*. A worker that dies mid-batch leaves pairs leased
but undecided, and the resumed worker re-claims them once the short lease lapses while
skipping every pair that already moved on to the next gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from gatekeeper.enums import Gate, QueueTier
from gatekeeper.logging import get_logger
from gatekeeper.timeutil import utc_now
from gatekeeper.worker.hydrate import GraphPair

__all__ = ["COLLECTION", "PairQueue", "QueuedPair", "pair_key"]

log = get_logger(__name__)

COLLECTION = "gatekeeper_pairs"
"""snake_case, like every other collection this system owns (LLD §3)."""

DEFAULT_LEASE_MINUTES = 15
_WRITE_BATCH_LIMIT = 400
"""Under Firestore's 500-write ceiling, with room for the batch's own bookkeeping."""


def pair_key(stage3_run_id: str, claim_a_id: str, claim_b_id: str) -> str:
    """The deterministic doc id for a pair within a Stage 3 run.

    Claim ids are sorted so that the same two claims produce one row whichever direction
    the graph edge happened to point — the pair is unordered, the NLI pass is what has a
    direction.
    """
    low, high = sorted((claim_a_id, claim_b_id))
    return f"{stage3_run_id}|{low}|{high}"


@dataclass(slots=True)
class QueuedPair:
    """One pair's position in the cascade."""

    pair_id: str
    stage3_run_id: str
    intake_id: str
    claim_a_id: str
    claim_b_id: str
    rank: int = 0
    with_context: bool = False
    human_asserted: bool = False
    tier: QueueTier = QueueTier.CASCADE
    gate: Gate | None = Gate.G1_NEUTRAL
    contradiction_flag: bool = False
    decided_by: str | None = None
    gk_run_request_id: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None

    @property
    def claim_id_low(self) -> str:
        return min(self.claim_a_id, self.claim_b_id)

    @property
    def claim_id_high(self) -> str:
        return max(self.claim_a_id, self.claim_b_id)

    def lease_held_at(self, now: datetime) -> bool:
        return self.lease_expires_at is not None and self.lease_expires_at > now

    def to_firestore(self) -> dict[str, Any]:
        return {
            "stage3RunId": self.stage3_run_id,
            "intakeId": self.intake_id,
            "claimAId": self.claim_a_id,
            "claimBId": self.claim_b_id,
            "claimIdLow": self.claim_id_low,
            "claimIdHigh": self.claim_id_high,
            "rank": self.rank,
            "withContext": self.with_context,
            "humanAsserted": self.human_asserted,
            "tier": self.tier.value,
            "gate": self.gate.value if self.gate else None,
            "contradictionFlag": self.contradiction_flag,
            "decidedBy": self.decided_by,
            "gkRunRequestId": self.gk_run_request_id,
            "leaseOwner": self.lease_owner,
            "leaseExpiresAt": self.lease_expires_at,
        }

    @classmethod
    def from_firestore(cls, pair_id: str, data: dict[str, Any]) -> QueuedPair:
        raw_gate = data.get("gate")
        return cls(
            pair_id=pair_id,
            stage3_run_id=str(data.get("stage3RunId", "")),
            intake_id=str(data.get("intakeId", "")),
            claim_a_id=str(data.get("claimAId", "")),
            claim_b_id=str(data.get("claimBId", "")),
            rank=int(data.get("rank") or 0),
            with_context=bool(data.get("withContext")),
            human_asserted=bool(data.get("humanAsserted")),
            tier=QueueTier(data.get("tier", QueueTier.CASCADE.value)),
            gate=Gate(raw_gate) if raw_gate else None,
            contradiction_flag=bool(data.get("contradictionFlag")),
            decided_by=data.get("decidedBy"),
            gk_run_request_id=data.get("gkRunRequestId"),
            lease_owner=data.get("leaseOwner"),
            lease_expires_at=data.get("leaseExpiresAt"),
        )


class PairQueue:
    """Firestore-backed routing over one Stage 3 run's pairs."""

    def __init__(
        self,
        client: firestore.Client,
        *,
        collection: str = COLLECTION,
        lease_minutes: int = DEFAULT_LEASE_MINUTES,
        clock: Any = utc_now,
    ) -> None:
        self._client = client
        self._collection = client.collection(collection)
        self._lease = timedelta(minutes=lease_minutes)
        self._clock = clock

    # -- seeding --------------------------------------------------------------

    def seed(self, stage3_run_id: str, intake_id: str, pairs: Sequence[GraphPair]) -> int:
        """Create a queue row for every graph pair that does not have one yet.

        Only *missing* rows are written. A re-run of G1 — a redelivery, a sweeper rescue,
        a resumed worker — must not overwrite the routing of pairs already decided, and a
        FROM_START that genuinely wants them reset calls :meth:`reset` first. Returns the
        number of rows created.
        """
        existing = self.pair_ids(stage3_run_id)
        created = 0
        batch = self._client.batch()
        pending = 0

        for graph_pair in pairs:
            key = pair_key(stage3_run_id, graph_pair.claim_a_id, graph_pair.claim_b_id)
            if key in existing:
                continue
            queued = QueuedPair(
                pair_id=key,
                stage3_run_id=stage3_run_id,
                intake_id=intake_id,
                claim_a_id=graph_pair.claim_a_id,
                claim_b_id=graph_pair.claim_b_id,
                rank=graph_pair.rank,
                with_context=graph_pair.with_context,
                human_asserted=graph_pair.human_asserted,
            )
            document = queued.to_firestore()
            document["createdAt"] = firestore.SERVER_TIMESTAMP
            document["updatedAt"] = firestore.SERVER_TIMESTAMP
            batch.set(self._collection.document(key), document)
            created += 1
            pending += 1
            if pending >= _WRITE_BATCH_LIMIT:
                batch.commit()
                batch = self._client.batch()
                pending = 0

        if pending:
            batch.commit()

        log.info(
            "seeded the pair queue",
            fields={"stage3RunId": stage3_run_id, "created": created, "graphPairs": len(pairs)},
        )
        return created

    def reset(self, stage3_run_id: str) -> int:
        """Return every pair to the head of the cascade — the purge's queue half (LLD §10).

        Idempotent: running it twice leaves the same state, which is what makes the purge
        safe under redelivery.
        """
        reset_count = 0
        batch = self._client.batch()
        pending = 0

        for snapshot in self._run_query(stage3_run_id).stream():
            batch.update(
                snapshot.reference,
                {
                    "tier": QueueTier.CASCADE.value,
                    "gate": Gate.G1_NEUTRAL.value,
                    "contradictionFlag": False,
                    "decidedBy": None,
                    "gkRunRequestId": None,
                    "leaseOwner": None,
                    "leaseExpiresAt": None,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )
            reset_count += 1
            pending += 1
            if pending >= _WRITE_BATCH_LIMIT:
                batch.commit()
                batch = self._client.batch()
                pending = 0

        if pending:
            batch.commit()

        log.info("reset queue routing", fields={"stage3RunId": stage3_run_id, "pairs": reset_count})
        return reset_count

    # -- reads ----------------------------------------------------------------

    def _run_query(self, stage3_run_id: str) -> Any:
        return self._collection.where(filter=FieldFilter("stage3RunId", "==", stage3_run_id))

    def pair_ids(self, stage3_run_id: str) -> set[str]:
        """Every queue doc id for a run — one read, so seeding is one pass."""
        return {snapshot.id for snapshot in self._run_query(stage3_run_id).stream()}

    def all_pairs(self, stage3_run_id: str) -> list[QueuedPair]:
        return [
            QueuedPair.from_firestore(snapshot.id, snapshot.to_dict() or {})
            for snapshot in self._run_query(stage3_run_id).stream()
        ]

    def count_at_gate(self, stage3_run_id: str, gate: Gate) -> int:
        return sum(1 for pair in self.all_pairs(stage3_run_id) if pair.gate is gate)

    # -- claiming -------------------------------------------------------------

    def claim_batch(
        self, stage3_run_id: str, gate: Gate, *, lease_owner: str, limit: int
    ) -> list[QueuedPair]:
        """Lease up to ``limit`` unleased pairs sitting at ``gate``.

        The candidate query is equality-only (``stage3RunId`` + ``gate``), which Firestore
        serves by merging single-field indexes — LLD §7.5 lists the two composite indexes
        this design needs and this is deliberately not a third. Leases are then filtered
        and taken in a transaction, so a rescued worker overlapping its predecessor still
        cannot take a pair the other one holds.
        """
        now = self._clock()
        candidates = [
            QueuedPair.from_firestore(snapshot.id, snapshot.to_dict() or {})
            for snapshot in self._run_query(stage3_run_id)
            .where(filter=FieldFilter("gate", "==", gate.value))
            .limit(limit * 4)
            .stream()
        ]
        free = sorted(
            (pair for pair in candidates if not pair.lease_held_at(now)),
            key=lambda pair: (pair.rank, pair.pair_id),
        )[:limit]
        if not free:
            return []

        return self._lease_all(free, gate, lease_owner=lease_owner, now=now)

    def _lease_all(
        self, pairs: Sequence[QueuedPair], gate: Gate, *, lease_owner: str, now: datetime
    ) -> list[QueuedPair]:
        expires = now + self._lease
        refs = [self._collection.document(pair.pair_id) for pair in pairs]

        @firestore.transactional
        def _claim(transaction: firestore.Transaction) -> list[QueuedPair]:
            # Firestore demands every read before any write, so the whole batch is read
            # first and only then leased.
            snapshots = [ref.get(transaction=transaction) for ref in refs]
            taken: list[QueuedPair] = []
            for ref, snapshot in zip(refs, snapshots, strict=True):
                if not snapshot.exists:
                    continue
                current = QueuedPair.from_firestore(snapshot.id, snapshot.to_dict() or {})
                if current.gate is not gate or current.lease_held_at(now):
                    continue
                transaction.update(ref, {"leaseOwner": lease_owner, "leaseExpiresAt": expires})
                current.lease_owner = lease_owner
                current.lease_expires_at = expires
                taken.append(current)
            return taken

        return _claim(self._client.transaction())

    # -- routing --------------------------------------------------------------

    def route_all(self, updates: Iterable[tuple[QueuedPair, dict[str, Any]]]) -> int:
        """Apply one gate's routing decisions to a batch of pairs.

        Written as an unconditional batch rather than a transaction: the pair lease has
        already established that this worker owns these rows, and the gate lease that it
        is the only worker on this gate. Paying for a transaction per batch would buy
        nothing the two leases have not already bought.
        """
        written = 0
        batch = self._client.batch()
        pending = 0

        for pair, changes in updates:
            payload = dict(changes)
            payload["leaseOwner"] = None
            payload["leaseExpiresAt"] = None
            payload["updatedAt"] = firestore.SERVER_TIMESTAMP
            batch.update(self._collection.document(pair.pair_id), payload)
            written += 1
            pending += 1
            if pending >= _WRITE_BATCH_LIMIT:
                batch.commit()
                batch = self._client.batch()
                pending = 0

        if pending:
            batch.commit()
        return written
