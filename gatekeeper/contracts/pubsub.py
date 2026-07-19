"""Pub/Sub push envelope parsing (LLD §5).

The push subscription wraps our payload in Google's envelope; a malformed envelope is
as much a ``GK_E_SCHEMA`` case as a malformed payload, so both fail the same way and
land in the same DLQ.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

from gatekeeper.contracts.payload import GatekeeperRunRequest
from gatekeeper.errors import SchemaError

__all__ = ["PushMessage", "parse_push_envelope"]


@dataclass(frozen=True, slots=True)
class PushMessage:
    """A decoded push delivery: our payload plus the delivery metadata worth logging."""

    request: GatekeeperRunRequest
    message_id: str
    publish_time: str
    delivery_attempt: int | None
    subscription: str


def parse_push_envelope(envelope: Any) -> PushMessage:
    """Decode a Pub/Sub push envelope into a :class:`PushMessage`.

    Raises:
        SchemaError: malformed envelope, undecodable body, or an invalid payload.
    """
    if not isinstance(envelope, dict):
        raise SchemaError(f"push envelope must be a JSON object, got {type(envelope).__name__}")

    message = envelope.get("message")
    if not isinstance(message, dict):
        raise SchemaError("push envelope is missing the 'message' object")

    data = message.get("data")
    if not isinstance(data, str) or not data:
        raise SchemaError("push message is missing base64 'data'")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SchemaError(f"push message data is not valid base64: {exc}") from exc

    delivery_attempt = envelope.get("deliveryAttempt")
    if delivery_attempt is not None and not isinstance(delivery_attempt, int):
        delivery_attempt = None

    return PushMessage(
        request=GatekeeperRunRequest.from_bytes(raw),
        message_id=str(message.get("messageId", "")),
        publish_time=str(message.get("publishTime", "")),
        delivery_attempt=delivery_attempt,
        subscription=str(envelope.get("subscription", "")),
    )
