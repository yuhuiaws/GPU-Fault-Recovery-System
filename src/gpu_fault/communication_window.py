"""Clock-aware overlap; a shared request ID is not a shared observation window."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any


def common_window(
    rows: list[dict[str, Any]], *, max_seconds: float = 600
) -> dict[str, Any]:
    first, last = [], []
    nodes = {row.get("node_id") for row in rows}
    for row in rows:
        try:
            start = datetime.fromisoformat(row["first_observed_at"])
            end = datetime.fromisoformat(row["second_observed_at"])
            error = row.get("clock_uncertainty_seconds")
            if len(nodes) <= 1:
                error = 0
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not math.isfinite(error)
                or not 0 <= error <= 30
                or start.tzinfo is None
                or end.tzinfo is None
                or not 0 < (end - start).total_seconds() <= max_seconds
            ):
                raise ValueError("unknown clock or invalid observation interval")
            first.append(start + timedelta(seconds=error))
            last.append(end - timedelta(seconds=error))
        except (KeyError, TypeError, ValueError):
            return {
                "time_comparable": False,
                "common_window_seconds": None,
                "time_reason": "MISSING_CLOCK_BOUND_OR_INVALID_WINDOW",
            }
    overlap = (min(last) - max(first)).total_seconds() if first else 0
    return {
        "time_comparable": overlap > 0,
        "common_window_seconds": max(0, overlap),
        "time_reason": "OVERLAPPING_OBSERVATIONS"
        if overlap > 0
        else "DISJOINT_WINDOWS",
    }
