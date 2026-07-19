"""Structured JSON logging (LLD §3, §12).

Every line is one JSON object carrying ``runRequestId`` and ``gate`` — always present,
null when unbound — so Cloud Logging can filter a whole run with a single predicate and
the runbook's "worker logs filtered by runRequestId" step actually works.

Field names are camelCase here for the same reason they are in Firestore: this is a
serialization boundary, and the two surfaces are read side by side during an incident.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from gatekeeper.timeutil import utc_now

__all__ = ["StructuredLogger", "configure_logging", "get_logger", "log_context"]

_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("gatekeeper_log_context", default=None)

_ALWAYS_PRESENT = ("runRequestId", "gate")

# Attributes LogRecord always defines; anything else on the record is caller-supplied.
_RESERVED_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind envelope fields onto every log line emitted inside the block.

    Nested blocks merge, so a dispatcher can bind ``runRequestId`` once and each gate
    can add its own ``gate`` without restating the run.
    """
    merged = {
        **(_LOG_CONTEXT.get() or {}),
        **{k: v for k, v in fields.items() if v is not None},
    }
    token = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


class JsonFormatter(logging.Formatter):
    """Render a record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = _LOG_CONTEXT.get() or {}
        for key in _ALWAYS_PRESENT:
            payload[key] = context.get(key)
        payload.update({k: v for k, v in context.items() if k not in _ALWAYS_PRESENT})

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS and key != "gk_fields"
        }
        payload.update(extras)
        payload.update(getattr(record, "gk_fields", None) or {})

        if record.exc_info:
            payload["stackTrace"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class StructuredLogger(logging.LoggerAdapter):
    """Logger whose call sites pass structured fields instead of interpolating strings."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        fields = kwargs.pop("fields", None)
        if fields:
            extra = kwargs.setdefault("extra", {})
            extra["gk_fields"] = fields
        return msg, kwargs


def configure_logging(level: int | str = logging.INFO, *, stream: Any = None) -> None:
    """Install the JSON formatter as the root handler. Idempotent."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> StructuredLogger:
    """A structured logger for ``name``; call ``log.info("msg", fields={...})``."""
    return StructuredLogger(logging.getLogger(name), {})
