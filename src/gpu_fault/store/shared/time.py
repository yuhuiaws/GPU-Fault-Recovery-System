from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.models import datetime_json_text


def utc_text(value: datetime) -> str:
    """Render a timestamp in the UTC form used by serialized models.

    Byte-for-byte what ``StrictModel`` writes into a payload for an aware
    datetime, so a bound parameter compares as text against stored JSON
    (fixed six-digit microseconds; store review 2026-09-07, item E). A naive
    value is taken as UTC, as before.
    """

    normalized = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return datetime_json_text(normalized)
