"""Reads of vishwamitra-owned data (LLD §7.4, §9).

Lakshmana never writes here. The one field it must read is ``judgeMode``: the mode is
pinned into ``Stage3Run.paramsSnapshot`` at run start and read from there by *both*
services, which is what makes the split-brain guard work — a mode change mid-flight
cannot make both judges act on the same run.

Shape note, confirmed against vishwamitra-core's ``Stage3RunRepository``: ``stage3_runs``
stores ``paramsSnapshot`` as a **JSON string field**, not a nested map, so it must be
parsed rather than indexed into. ``judgeMode`` itself does not exist there until VA-106
lands; until then the configured default applies.
"""

from __future__ import annotations

import json

from google.cloud import firestore

from gatekeeper.config import Config
from gatekeeper.enums import JudgeMode
from gatekeeper.logging import get_logger

__all__ = ["resolve_judge_mode", "resolve_subject_id", "source_excerpts"]

log = get_logger(__name__)

_GET_ALL_CHUNK = 300
"""``getAll`` is one round trip per chunk; Firestore's own limit is higher than we need."""


def source_excerpts(
    client: firestore.Client, config: Config, claim_ids: list[str]
) -> dict[str, str]:
    """``{claimId: sourceExcerpt}`` — the evidence text G2's grounding mode scores against.

    ``sourceExcerpt`` is the verbatim printed text a claim was extracted from (vishwamitra
    Stage 2, ``ClaimExtractor``). It is what makes MiniCheck's native ``(document, claim)``
    input shape reachable at all: it is *evidence*, where the claim text on both sides of a
    pair is two assertions.

    **It is read from Firestore rather than the graph, and that is not an accident.**
    Vishwamitra's evidence projection (``Stage3GraphRepository.mergeEvidence``) writes
    ``text``/``basis``/``sourceClass`` onto ``:Claim`` but not ``sourceExcerpt``, and the
    ``:Source`` node carries only ``assetId``/``contentType``/``checksum`` — no text at all.
    So the excerpt simply is not in Neo4j, and asking the graph for it would return nothing
    for every pair rather than failing visibly. The `claims` collection is the authority
    (``ClaimRepository.COLLECTION``), keyed by claim id, which is the same id the graph uses
    (vishwamitra stage3 LLD §9.1: ``claimId`` UNIQUE = Firestore id).

    Missing excerpts are **normal**, not an error: a claim extracted from audio may have no
    printed text behind it. G2 falls back to the two claim-vs-claim directions for those,
    which is the ``groundingMode = AUTO`` contract.
    """
    unique = sorted({claim_id for claim_id in claim_ids if claim_id})
    if not unique:
        return {}

    collection = client.collection(config.get_str("gatekeeper.integration.claims-collection"))
    excerpts: dict[str, str] = {}
    for start in range(0, len(unique), _GET_ALL_CHUNK):
        chunk = unique[start : start + _GET_ALL_CHUNK]
        refs = [collection.document(claim_id) for claim_id in chunk]
        for snapshot in client.get_all(refs):
            if not snapshot.exists:
                continue
            excerpt = (snapshot.to_dict() or {}).get("sourceExcerpt")
            if isinstance(excerpt, str) and excerpt.strip():
                excerpts[snapshot.id] = excerpt

    log.info(
        "read source excerpts for grounding",
        fields={"requested": len(unique), "found": len(excerpts)},
    )
    return excerpts


def resolve_subject_id(client: firestore.Client, config: Config, stage3_run_id: str) -> str:
    """Read the ``subjectId`` a Stage 3 run is about.

    The graph is keyed by subject, not by Stage 3 run — `JUDGE_QUEUED` relationships hang
    off ``(:Claim {subjectId})`` — so this is the join between the run the message names and
    the pairs the gate has to judge. It is a plain top-level field on ``stage3_runs``, unlike
    ``judgeMode``, which lives inside the JSON-string ``paramsSnapshot``.

    Returns:
        The subject id, or ``""`` when the run doc or the field is missing. The caller
        decides what that means; for a gate it is a hard failure, because there is nothing
        to judge without it.
    """
    collection = config.get_str("gatekeeper.integration.stage3-runs-collection")
    snapshot = client.collection(collection).document(stage3_run_id).get()
    if not snapshot.exists:
        log.warning("stage3 run doc not found", fields={"stage3RunId": stage3_run_id})
        return ""

    subject_id = str((snapshot.to_dict() or {}).get("subjectId") or "")
    if not subject_id:
        log.warning("stage3 run carries no subjectId", fields={"stage3RunId": stage3_run_id})
    return subject_id


def resolve_judge_mode(client: firestore.Client, config: Config, stage3_run_id: str) -> JudgeMode:
    """Read ``judgeMode`` from the Stage 3 run's frozen params snapshot.

    Falls back to ``gatekeeper.judge-mode-default`` — with a warning naming the reason —
    when the run doc, the snapshot, or the field is absent. A gatekeeper run only exists
    because vishwamitra published for it, so the fallback is a degraded read, not a
    licence to judge in the wrong mode; ``LLM`` is treated as a misroute and logged.
    """
    default = JudgeMode(config.get_str("gatekeeper.judge-mode-default"))
    collection = config.get_str("gatekeeper.integration.stage3-runs-collection")

    snapshot = client.collection(collection).document(stage3_run_id).get()
    if not snapshot.exists:
        log.warning(
            "stage3 run doc not found; falling back to default judge mode",
            fields={"stage3RunId": stage3_run_id, "judgeMode": default.value},
        )
        return default

    raw = (snapshot.to_dict() or {}).get("paramsSnapshot")
    if not isinstance(raw, str) or not raw:
        log.warning(
            "stage3 run has no paramsSnapshot; falling back to default judge mode",
            fields={"stage3RunId": stage3_run_id, "judgeMode": default.value},
        )
        return default

    try:
        params = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning(
            "stage3 paramsSnapshot is not valid JSON; falling back to default judge mode",
            fields={"stage3RunId": stage3_run_id, "judgeMode": default.value, "error": str(exc)},
        )
        return default

    value = params.get("judgeMode") if isinstance(params, dict) else None
    if value is None:
        log.info(
            "stage3 paramsSnapshot carries no judgeMode (pre-VA-106); using default",
            fields={"stage3RunId": stage3_run_id, "judgeMode": default.value},
        )
        return default

    try:
        resolved = JudgeMode(value)
    except ValueError:
        log.warning(
            "stage3 paramsSnapshot has an unknown judgeMode; falling back to default",
            fields={"stage3RunId": stage3_run_id, "found": value, "judgeMode": default.value},
        )
        return default

    if resolved is JudgeMode.LLM:
        # The ensemble owns this run; a gatekeeper message for it is a misroute worth
        # seeing in the logs. The claim still proceeds — dropping it silently would
        # strand the run — but the mode is recorded as published.
        log.warning(
            "stage3 run is in LLM judge mode but a gatekeeper message arrived",
            fields={"stage3RunId": stage3_run_id},
        )

    return resolved
