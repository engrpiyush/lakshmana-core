"""Publishing gate messages (LLD §4, §5).

Two publishers exist in this process: the **worker** chains G(n) → G(n+1) after committing
its gate, and the **sweeper** republishes work whose lease expired. Both send the same
:class:`~gatekeeper.contracts.payload.GatekeeperRunRequest` to the same topic, because the
chain is self-driving — external services only ever publish the first G1 message (owner
requirement #4).

The publish is deliberately **synchronous**. ``publish()`` returns a future and the client
batches in a background thread; a worker that returned from ``main()`` without resolving it
would exit with the next gate still sitting in a buffer, and the run would stall until a
sweeper noticed. Waiting costs milliseconds and removes the failure mode entirely.
"""

from __future__ import annotations

import time
from typing import Protocol

from gatekeeper.config import Config
from gatekeeper.contracts.payload import GatekeeperRunRequest
from gatekeeper.logging import get_logger

__all__ = ["NullPublisher", "PubSubPublisher", "Publisher", "RecordingPublisher", "publisher_for"]

log = get_logger(__name__)

_PUBLISH_TIMEOUT_SECONDS = 30.0


class Publisher(Protocol):
    """Sends one run request to the gatekeeper topic."""

    def publish(self, request: GatekeeperRunRequest) -> str:
        """Publish and return the message id."""
        ...


class PubSubPublisher:
    """The real client, bound to one topic."""

    def __init__(self, project_id: str, topic: str, client: object | None = None) -> None:
        from google.cloud import pubsub_v1

        self._client = client or pubsub_v1.PublisherClient()
        self._topic_path = f"projects/{project_id}/topics/{topic}"

    def publish(self, request: GatekeeperRunRequest) -> str:
        future = self._client.publish(self._topic_path, request.to_bytes())  # type: ignore[attr-defined]
        message_id = str(future.result(timeout=_PUBLISH_TIMEOUT_SECONDS))
        log.info(
            "published gate message",
            fields={
                "topic": self._topic_path,
                "nextGate": request.gate.value,
                "messageId": message_id,
            },
        )
        return message_id


class RecordingPublisher:
    """Keeps published requests in memory — the local e2e and test double."""

    def __init__(self) -> None:
        self.published: list[GatekeeperRunRequest] = []

    def publish(self, request: GatekeeperRunRequest) -> str:
        self.published.append(request)
        return f"recorded-{len(self.published)}"


class NullPublisher:
    """Logs the intent and sends nothing.

    The default when no project is configured — a local worker run should still walk the
    whole gate, and a loud log line is more honest than a client that would fail on ADC.
    """

    def publish(self, request: GatekeeperRunRequest) -> str:
        log.warning(
            "no Pub/Sub project configured; not publishing the next gate",
            fields={"nextGate": request.gate.value},
        )
        return ""


def publisher_for(config: Config) -> Publisher:
    """Build the publisher this environment can actually use."""
    project_id = config.get_str("gatekeeper.firestore.project-id")
    if not project_id:
        return NullPublisher()
    return PubSubPublisher(project_id, config.get_str("gatekeeper.pubsub.topic"))


def dlq_depth(config: Config, client: object | None = None) -> int | None:
    """Undelivered messages on the DLQ subscription, or None if it cannot be read.

    Surfaced by ``/sweep`` (LLD §11c). It is a *monitoring* read: a failure to obtain it
    must not fail the sweep, because the sweep's rescue work is the part that matters.
    """
    project_id = config.get_str("gatekeeper.firestore.project-id")
    subscription = config.get_str("gatekeeper.pubsub.dlq-subscription")
    if not project_id or not subscription:
        return None

    try:
        from google.cloud import monitoring_v3

        resolved = client or monitoring_v3.MetricServiceClient()
        return _read_backlog(resolved, project_id, subscription)
    except Exception as exc:  # noqa: BLE001 — monitoring is best-effort; the rescue is not
        log.warning("could not read DLQ depth", fields={"error": str(exc)})
        return None


_BACKLOG_LOOKBACK_SECONDS = 600
"""The metric is sampled every 60s; ten minutes always contains a point."""


def _read_backlog(client: object, project_id: str, subscription: str) -> int | None:
    from google.cloud import monitoring_v3

    interval = monitoring_v3.TimeInterval()
    now = time.time()
    interval.end_time = {"seconds": int(now)}
    interval.start_time = {"seconds": int(now) - _BACKLOG_LOOKBACK_SECONDS}

    results = client.list_time_series(  # type: ignore[attr-defined]
        request={
            "name": f"projects/{project_id}",
            "filter": (
                'metric.type="pubsub.googleapis.com/subscription/num_undelivered_messages" '
                f'AND resource.labels.subscription_id="{subscription}"'
            ),
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    )
    for series in results:
        for point in series.points:
            return int(point.value.int64_value)
    return None
