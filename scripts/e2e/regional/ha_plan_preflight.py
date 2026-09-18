"""Read-only residual checks used when approving HA fixture creation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any


def require_window(deadline: datetime, *, required_seconds: int = 0) -> None:
    if (
        deadline.tzinfo is None
        or (deadline - datetime.now(timezone.utc)).total_seconds() <= required_seconds
    ):
        raise RuntimeError(
            "approved maintenance window cannot cover the remaining HA operation"
        )


def residual_preflight(
    database: Callable[[], dict[str, Any]],
    registry: Callable[[], dict[str, Any]],
    kubernetes: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    errors = []
    for name, key, probe in (
        ("database", "total", database),
        ("registry", "count", registry),
        ("kubernetes", "count", kubernetes),
    ):
        try:
            value = probe()
            observations[name] = value
            if type(value.get(key)) is not int or value[key] != 0:
                errors.append(f"{name}: residual count is missing or nonzero")
        except Exception as exc:
            observations[name] = {"error_type": type(exc).__name__}
            errors.append(f"{name}: preflight read failed")
    return {**observations, "errors": errors}
