"""RFC3339 UTC helpers (LLD §3).

One spelling of "now" for the whole service, and one parser strict enough that a
malformed ``requestTimestamp`` is caught at the contract boundary rather than three
gates later. ``*Timestamp`` fields carry event time in messages; ``*At`` fields are row
lifecycle written with Firestore server time.
"""

from datetime import UTC, datetime

__all__ = ["format_rfc3339", "parse_rfc3339", "utc_now"]


def utc_now() -> datetime:
    """Timezone-aware current UTC instant."""
    return datetime.now(UTC)


def format_rfc3339(value: datetime) -> str:
    """Render as RFC3339 UTC with a ``Z`` suffix and millisecond precision.

    Millisecond precision keeps fixtures byte-stable across Python and the Kotlin
    contract test; naive datetimes are rejected rather than silently assumed UTC.
    """
    if value.tzinfo is None:
        raise ValueError("refusing to format a naive datetime; attach a timezone")
    as_utc = value.astimezone(UTC)
    return as_utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{as_utc.microsecond // 1000:03d}Z"


def parse_rfc3339(value: str) -> datetime:
    """Parse an RFC3339 timestamp, requiring an explicit offset.

    Raises:
        ValueError: if the string is malformed or carries no timezone.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp {value!r} carries no timezone offset")
    return parsed.astimezone(UTC)
