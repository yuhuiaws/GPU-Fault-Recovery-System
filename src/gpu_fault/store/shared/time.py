from __future__ import annotations

from datetime import datetime, timezone


def utc_text(value: datetime) -> str:
    """Render a timestamp in the UTC form used by serialized models."""

    normalized = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return normalized.isoformat().replace("+00:00", "Z")
