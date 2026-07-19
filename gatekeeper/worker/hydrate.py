"""Hydrating pairs and claim text from the graph — read-only, always (LLD §8, D-8).

Two reads, kept separate on purpose.

**The queue** (:func:`queued_pairs`) is vishwamitra's MATCH output: `JUDGE_QUEUED`
relationships between claims, carrying the rank, the `withContext` flag §11.9's dual
evaluation turns on, and the human-asserted marker. It is read once per run, at G1, to
seed lakshmana's own routing state.

**The text** (:func:`claim_texts`, :func:`explanation_texts`) is fetched per batch, by
claim id. Keeping it out of the queue seed is deliberate: claim text is the one thing this
system tries hard not to copy around — it never transits Pub/Sub (LLD §5) and there is no
reason for it to sit in a second Firestore collection either. The graph is the authority
on it and stays that way.

Every query here is a `MATCH`. The RBAC user cannot write, :class:`Neo4jReader` pins
`READ_ACCESS` on the session, and the queue's routing state lives in Firestore precisely
so that no gate ever needs a write session to the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gatekeeper.clients.neo4j import Neo4jReader
from gatekeeper.gates.prompts import ClaimCard
from gatekeeper.logging import get_logger

__all__ = ["GraphPair", "claim_cards", "claim_texts", "explanation_texts", "queued_pairs"]

log = get_logger(__name__)

_QUEUE_CYPHER = """
MATCH (a:Claim {subjectId: $subjectId})-[q:JUDGE_QUEUED]->(b:Claim)
RETURN a.claimId AS aId, b.claimId AS bId, q.rank AS rank,
       coalesce(q.withContext, false) AS withContext,
       coalesce(q.humanAsserted, false) AS humanAsserted
ORDER BY q.rank ASC
"""
"""Every queued pair for a subject, in MATCH's own rank order.

Deliberately **not** filtered on ``status = 'QUEUED'``: that property is vishwamitra's
JUDGE-phase cursor, and in GATEKEEPER mode its JUDGE phase never runs, so the flag would
be whatever the last LLM run left behind. Lakshmana keeps its own cursor — the ``gate``
field on its queue docs — and reads the whole queue.
"""

_TEXT_CYPHER = """
UNWIND $claimIds AS claimId
MATCH (c:Claim {claimId: claimId})
RETURN c.claimId AS claimId, c.text AS text
"""

_EXPLANATION_CYPHER = """
UNWIND $claimIds AS claimId
MATCH (c:Claim {claimId: claimId})
RETURN c.claimId AS claimId,
       [ (x:Explanation)-[:EXPLAINS]->(c) | x.text ][0] AS text
"""

_CARD_CYPHER = """
UNWIND $claimIds AS claimId
MATCH (c:Claim {claimId: claimId})
RETURN c.claimId AS claimId, c.text AS text, c.type AS type,
       c.sourceClass AS sourceClass, c.claimedDate AS claimedDate,
       c.relationship AS relationship, c.speakerRole AS speakerRole,
       [ (x:Explanation)-[:EXPLAINS]->(c) | x.text ][0] AS explanationText
"""
"""Every field the G4 prompt's claim card renders (LLD §8 G4, O-3).

One query rather than two because G4 needs the sidecar and the card together, and unlike
the encoder gates it is called on a handful of pairs — the round trip is not the cost that
matters here. Each property is one vishwamitra's evidence projection writes onto
``:Claim``; a null simply drops its line from the card.
"""


@dataclass(frozen=True, slots=True)
class GraphPair:
    """One `JUDGE_QUEUED` relationship, as lakshmana needs it."""

    claim_a_id: str
    claim_b_id: str
    rank: int = 0
    with_context: bool = False
    human_asserted: bool = False


def queued_pairs(reader: Neo4jReader, subject_id: str) -> list[GraphPair]:
    """Every pair MATCH queued for ``subject_id``."""
    rows = reader.run(_QUEUE_CYPHER, subjectId=subject_id)
    pairs = [
        GraphPair(
            claim_a_id=str(row["aId"]),
            claim_b_id=str(row["bId"]),
            rank=int(row.get("rank") or 0),
            with_context=bool(row.get("withContext")),
            human_asserted=bool(row.get("humanAsserted")),
        )
        for row in rows
        if row.get("aId") and row.get("bId")
    ]
    log.info(
        "read the judge queue from the graph",
        fields={"subjectId": subject_id, "pairs": len(pairs)},
    )
    return pairs


def claim_texts(reader: Neo4jReader, claim_ids: list[str]) -> dict[str, str]:
    """``{claimId: text}`` for the ids given, skipping any the graph does not have."""
    return _texts(reader, _TEXT_CYPHER, claim_ids)


def explanation_texts(reader: Neo4jReader, claim_ids: list[str]) -> dict[str, str]:
    """``{claimId: explanationText}`` — the §11.9 sidecar, for `withContext` pairs only."""
    return _texts(reader, _EXPLANATION_CYPHER, claim_ids)


def claim_cards(reader: Neo4jReader, claim_ids: list[str]) -> dict[str, ClaimCard]:
    """``{claimId: ClaimCard}`` — the full card G4's prompt renders, sidecar included.

    Claims with no ``text`` are skipped: a card with an empty Text line asks the model to
    judge nothing, and the caller treats a missing card as a hydrate failure.
    """
    unique = sorted({claim_id for claim_id in claim_ids if claim_id})
    if not unique:
        return {}

    rows: list[dict[str, Any]] = reader.run(_CARD_CYPHER, claimIds=unique)
    cards: dict[str, ClaimCard] = {}
    for row in rows:
        claim_id, text = row.get("claimId"), row.get("text")
        if not claim_id or not text:
            continue
        cards[str(claim_id)] = ClaimCard(
            claim_id=str(claim_id),
            text=str(text),
            type=_optional(row.get("type")),
            source_class=_optional(row.get("sourceClass")),
            claimed_date=_optional(row.get("claimedDate")),
            relationship=_optional(row.get("relationship")),
            speaker_role=_optional(row.get("speakerRole")),
            explanation_text=_optional(row.get("explanationText")),
        )
    return cards


def _optional(value: Any) -> str | None:
    """A graph property as a card line, or None when it has nothing to say."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _texts(reader: Neo4jReader, cypher: str, claim_ids: list[str]) -> dict[str, str]:
    unique = sorted({claim_id for claim_id in claim_ids if claim_id})
    if not unique:
        return {}
    rows: list[dict[str, Any]] = reader.run(cypher, claimIds=unique)
    return {
        str(row["claimId"]): str(row["text"])
        for row in rows
        if row.get("claimId") and row.get("text")
    }
