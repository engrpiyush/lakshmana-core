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

__all__ = ["resolve_judge_mode", "resolve_subject_id"]

log = get_logger(__name__)


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
