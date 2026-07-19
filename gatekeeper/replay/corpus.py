"""The replay corpus: vishwamitra's ensemble-labelled pairs as JSONL (LLD §15 step 1).

The bake-off needs the 12,208 pairs the LLM ensemble already judged, with its verdicts,
its per-vote detail and the golden flags, so that a candidate roster can be scored against
a label set nobody had to pay to produce.

**Field names are configuration, not constants.** ``stage3_edges`` is vishwamitra's
collection; the LLD (§7.2) specifies only the fields *lakshmana adds* to it, and the
existing ones — pair ids, ensemble verdict, votes, golden markers — are that repo's to
name. Rather than hardcode guesses that would fail silently, this module takes a
:class:`FieldMap` of dotted paths with documented defaults, and ships an ``inspect`` pass
that reports what a real collection actually contains before a single row is exported. A
mismatch is then a one-line map override, not an afternoon.

That design also survives the thing it was written under: the corpus is restored from a
Firestore backup into a **separate** emulator instance, never the shared dev one, and the
person running the export is not necessarily the person who knows the schema.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gatekeeper.enums import Verdict
from gatekeeper.logging import get_logger

__all__ = [
    "DEFAULT_COLLECTION",
    "CorpusStats",
    "FieldMap",
    "PairRecord",
    "export_corpus",
    "inspect_collection",
    "read_corpus",
    "stream_edges",
    "write_corpus",
]

log = get_logger(__name__)

CORPUS_SCHEMA_VERSION = 1


DEFAULT_COLLECTION = "stage3_edges"


def stream_edges(
    client: Any,
    collection: str = DEFAULT_COLLECTION,
    stage3_run_id: str = "",
    limit: int = 0,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(doc_id, document)`` from a ``stage3_edges`` collection.

    ``client`` is a ``google.cloud.firestore.Client``, typed loosely so this module stays
    importable — and testable — without the SDK.
    """
    from google.cloud.firestore_v1.base_query import FieldFilter

    query = client.collection(collection)
    if stage3_run_id:
        query = query.where(filter=FieldFilter("stage3RunId", "==", stage3_run_id))
    if limit:
        query = query.limit(limit)
    for snapshot in query.stream():
        yield snapshot.id, (snapshot.to_dict() or {})


def _dig(document: Mapping[str, Any], path: str) -> Any:
    """Follow a dotted path into a nested document, returning None if it dead-ends."""
    current: Any = document
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _first(document: Mapping[str, Any], paths: Sequence[str]) -> Any:
    for path in paths:
        value = _dig(document, path)
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class FieldMap:
    """Where each corpus field lives in a ``stage3_edges`` document.

    Every entry is a list of candidate dotted paths, tried in order — vishwamitra's
    verdict field has been spelled more than one way across its own history, and an
    exporter that accepts the first path that exists costs nothing and saves a re-run.
    Override any of them from JSON via :meth:`from_json`.
    """

    claim_a_id: tuple[str, ...] = ("claimAId", "claimA", "sourceClaimId", "fromClaimId")
    claim_b_id: tuple[str, ...] = ("claimBId", "claimB", "targetClaimId", "toClaimId")
    claim_a_text: tuple[str, ...] = ("claimAText", "claimATextSnapshot", "textA")
    claim_b_text: tuple[str, ...] = ("claimBText", "claimBTextSnapshot", "textB")
    verdict: tuple[str, ...] = ("verdict", "ensembleVerdict", "judgeVerdict")
    votes: tuple[str, ...] = ("votes", "ensembleVotes", "judgeVotes", "ensemble.votes")
    golden: tuple[str, ...] = ("golden", "isGolden", "humanConfirmed", "contributesToFact")
    with_context: tuple[str, ...] = ("withContext", "hasContext")
    context_a: tuple[str, ...] = ("explanationA", "contextA")
    context_b: tuple[str, ...] = ("explanationB", "contextB")
    source_snippet: tuple[str, ...] = ("sourceSnippet", "evidenceSnippet", "snippet")
    method: tuple[str, ...] = ("method",)
    intake_id: tuple[str, ...] = ("intakeId",)
    stage3_run_id: tuple[str, ...] = ("stage3RunId",)

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> FieldMap:
        """Override any subset of the defaults; a string is read as a one-element list."""
        overrides: dict[str, tuple[str, ...]] = {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        for name, value in document.items():
            if name not in known:
                raise ValueError(f"{name!r} is not a corpus field; known: {sorted(known)}")
            overrides[name] = (value,) if isinstance(value, str) else tuple(value)
        return cls(**overrides)

    @classmethod
    def load(cls, path: Path | None) -> FieldMap:
        if path is None:
            return cls()
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One labelled pair — a single JSONL line."""

    pair_id: str
    claim_a_id: str
    claim_b_id: str
    claim_a_text: str
    claim_b_text: str
    ensemble_verdict: str
    votes: tuple[str, ...] = ()
    golden: bool = False
    with_context: bool = False
    context_a: str = ""
    context_b: str = ""
    source_snippet: str = ""
    intake_id: str = ""
    stage3_run_id: str = ""

    @property
    def usable(self) -> bool:
        """Both texts and a verdict — anything less cannot be scored or graded."""
        return bool(self.claim_a_text and self.claim_b_text and self.ensemble_verdict)

    def to_json(self) -> dict[str, Any]:
        return {
            "pairId": self.pair_id,
            "claimAId": self.claim_a_id,
            "claimBId": self.claim_b_id,
            "claimAText": self.claim_a_text,
            "claimBText": self.claim_b_text,
            "ensembleVerdict": self.ensemble_verdict,
            "votes": list(self.votes),
            "golden": self.golden,
            "withContext": self.with_context,
            "contextA": self.context_a,
            "contextB": self.context_b,
            "sourceSnippet": self.source_snippet,
            "intakeId": self.intake_id,
            "stage3RunId": self.stage3_run_id,
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> PairRecord:
        return cls(
            pair_id=str(document["pairId"]),
            claim_a_id=str(document.get("claimAId", "")),
            claim_b_id=str(document.get("claimBId", "")),
            claim_a_text=str(document.get("claimAText", "")),
            claim_b_text=str(document.get("claimBText", "")),
            ensemble_verdict=str(document.get("ensembleVerdict", "")),
            votes=tuple(str(vote) for vote in document.get("votes", ())),
            golden=bool(document.get("golden", False)),
            with_context=bool(document.get("withContext", False)),
            context_a=str(document.get("contextA", "")),
            context_b=str(document.get("contextB", "")),
            source_snippet=str(document.get("sourceSnippet", "")),
            intake_id=str(document.get("intakeId", "")),
            stage3_run_id=str(document.get("stage3RunId", "")),
        )


@dataclass
class CorpusStats:
    """What the export found — printed at the end, and the thing to sanity-check."""

    scanned: int = 0
    exported: int = 0
    skipped_not_ensemble: int = 0
    skipped_unusable: int = 0
    hydrated_from_graph: int = 0
    verdicts: Counter[str] = field(default_factory=Counter)
    golden: int = 0
    with_context: int = 0
    missing_fields: Counter[str] = field(default_factory=Counter)

    def describe(self) -> str:
        verdicts = ", ".join(f"{name} {count}" for name, count in sorted(self.verdicts.items()))
        return (
            f"scanned {self.scanned} · exported {self.exported} · "
            f"skipped {self.skipped_not_ensemble} non-ensemble / "
            f"{self.skipped_unusable} unusable · golden {self.golden} · "
            f"withContext {self.with_context}\n  verdicts: {verdicts or '(none)'}"
        )


# --- inspection ---------------------------------------------------------------------------


def inspect_collection(documents: Iterable[Mapping[str, Any]], limit: int = 200) -> dict[str, Any]:
    """Report which field names a real collection actually uses.

    Run this before an export against an unfamiliar backup. It answers the only question
    that matters — does the default :class:`FieldMap` resolve against these documents —
    without writing anything.
    """
    field_map = FieldMap()
    seen_keys: Counter[str] = Counter()
    resolved: Counter[str] = Counter()
    sampled = 0

    for document in documents:
        if sampled >= limit:
            break
        sampled += 1
        seen_keys.update(_flatten_keys(document))
        for name in FieldMap.__dataclass_fields__:
            if _first(document, getattr(field_map, name)) is not None:
                resolved[name] += 1

    return {
        "sampled": sampled,
        "topLevelKeys": dict(seen_keys.most_common(60)),
        "resolvedByDefaultMap": {
            name: f"{resolved.get(name, 0)}/{sampled}" for name in FieldMap.__dataclass_fields__
        },
        "unresolved": sorted(
            name for name in FieldMap.__dataclass_fields__ if not resolved.get(name)
        ),
    }


def _flatten_keys(document: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    for key, value in document.items():
        path = f"{prefix}{key}"
        yield path
        if isinstance(value, Mapping):
            yield from _flatten_keys(value, f"{path}.")


# --- export -------------------------------------------------------------------------------

_ENSEMBLE_METHODS = {"ENSEMBLE", "RULE", "EMBEDDING"}
"""Vishwamitra's own methods. ``GK_*`` rows are the gatekeeper's own prior output and must
never become training labels for it (D-5 draws the same line for the purge)."""


def _coerce_verdict(raw: Any) -> str:
    if raw is None:
        return ""
    value = str(raw).strip().upper().replace("-", "_")
    try:
        return Verdict(value).value
    except ValueError:
        # An unrecognised verdict is kept verbatim rather than dropped: the report counts
        # it, and an unexpected label is something the owner should see, not something the
        # exporter should quietly decide about.
        return value


def _coerce_votes(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (_coerce_verdict(raw),)
    if isinstance(raw, Mapping):
        raw = raw.values()
    votes: list[str] = []
    for entry in raw:
        # A vote is either the label itself or a record carrying it.
        label = (
            _first(entry, ("verdict", "label", "value")) if isinstance(entry, Mapping) else entry
        )
        votes.append(_coerce_verdict(label))
    return tuple(vote for vote in votes if vote)


def record_from_document(
    pair_id: str,
    document: Mapping[str, Any],
    field_map: FieldMap,
    texts: Mapping[str, str] | None = None,
) -> PairRecord:
    """Build a :class:`PairRecord`, preferring graph-hydrated text over any snapshot."""
    claim_a_id = str(_first(document, field_map.claim_a_id) or "")
    claim_b_id = str(_first(document, field_map.claim_b_id) or "")
    lookup = texts or {}

    return PairRecord(
        pair_id=pair_id,
        claim_a_id=claim_a_id,
        claim_b_id=claim_b_id,
        claim_a_text=str(lookup.get(claim_a_id) or _first(document, field_map.claim_a_text) or ""),
        claim_b_text=str(lookup.get(claim_b_id) or _first(document, field_map.claim_b_text) or ""),
        ensemble_verdict=_coerce_verdict(_first(document, field_map.verdict)),
        votes=_coerce_votes(_first(document, field_map.votes)),
        golden=bool(_first(document, field_map.golden) or False),
        with_context=bool(_first(document, field_map.with_context) or False),
        context_a=str(_first(document, field_map.context_a) or ""),
        context_b=str(_first(document, field_map.context_b) or ""),
        source_snippet=str(_first(document, field_map.source_snippet) or ""),
        intake_id=str(_first(document, field_map.intake_id) or ""),
        stage3_run_id=str(_first(document, field_map.stage3_run_id) or ""),
    )


def export_corpus(
    documents: Iterable[tuple[str, Mapping[str, Any]]],
    field_map: FieldMap | None = None,
    *,
    hydrate: Any = None,
    stats: CorpusStats | None = None,
) -> Iterator[PairRecord]:
    """Turn ``(doc_id, document)`` pairs into usable :class:`PairRecord` rows.

    Args:
        documents: the source edges, already filtered to whatever scope is wanted.
        field_map: where the fields live; defaults apply when omitted.
        hydrate: optional callable taking claim ids and returning ``{id: text}``, used
            when the edge documents carry no text snapshot. The graph is the authority on
            claim text (LLD §8 hydrates from it at run time), so it wins when both exist.
        stats: accumulator; created internally if not supplied.
    """
    field_map = field_map or FieldMap()
    stats = stats if stats is not None else CorpusStats()

    for pair_id, document in documents:
        stats.scanned += 1

        method = str(_first(document, field_map.method) or "ENSEMBLE").upper()
        if method not in _ENSEMBLE_METHODS:
            stats.skipped_not_ensemble += 1
            continue

        record = record_from_document(pair_id, document, field_map)
        if hydrate is not None and not (record.claim_a_text and record.claim_b_text):
            texts = hydrate([record.claim_a_id, record.claim_b_id])
            if texts:
                record = record_from_document(pair_id, document, field_map, texts)
                stats.hydrated_from_graph += 1

        if not record.usable:
            stats.skipped_unusable += 1
            for name, value in (
                ("claimAText", record.claim_a_text),
                ("claimBText", record.claim_b_text),
                ("ensembleVerdict", record.ensemble_verdict),
            ):
                if not value:
                    stats.missing_fields[name] += 1
            continue

        stats.exported += 1
        stats.verdicts[record.ensemble_verdict] += 1
        stats.golden += bool(record.golden)
        stats.with_context += bool(record.with_context)
        yield record


# --- JSONL --------------------------------------------------------------------------------


def write_corpus(records: Iterable[PairRecord], path: Path) -> int:
    """Write JSONL, one record per line. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        header = {"schemaVersion": CORPUS_SCHEMA_VERSION, "kind": "gatekeeperReplayCorpus"}
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
            written += 1
    return written


def read_corpus(path: Path) -> list[PairRecord]:
    """Read JSONL back, refusing a schema this build does not understand."""
    records: list[PairRecord] = []
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
        if not first.strip():
            return records
        header = json.loads(first)
        if header.get("kind") != "gatekeeperReplayCorpus":
            raise ValueError(f"{path} is not a replay corpus (header {header!r})")
        if int(header.get("schemaVersion", 0)) != CORPUS_SCHEMA_VERSION:
            raise ValueError(
                f"{path} is corpus schemaVersion {header.get('schemaVersion')}; "
                f"this build reads {CORPUS_SCHEMA_VERSION}"
            )
        for line in handle:
            if line.strip():
                records.append(PairRecord.from_json(json.loads(line)))
    return records
