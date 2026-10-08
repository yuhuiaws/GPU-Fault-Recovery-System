"""Read-only residual checks used when approving HA fixture creation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any


def require_window(deadline: datetime, *, required_seconds: int = 0) -> None:
    """Refuse an HA step the approved maintenance window cannot cover.

    The message names what was measured -- the deadline, the seconds left and
    the seconds the step needs -- because the case receipt records it as
    ``error``: an operator reading only the receipt must be able to tell a
    window that was never set from one that is merely short.
    """

    if deadline.tzinfo is None:
        raise RuntimeError(
            "approved maintenance window cannot cover the remaining HA operation: "
            f"the deadline {deadline.isoformat()} carries no timezone"
        )
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining <= required_seconds:
        raise RuntimeError(
            "approved maintenance window cannot cover the remaining HA operation: "
            f"{remaining:.0f}s left before {deadline.isoformat()}, "
            f"{required_seconds}s required"
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
