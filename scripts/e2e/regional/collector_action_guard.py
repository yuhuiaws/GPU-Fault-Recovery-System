"""Admission for new collector actions; cleanup keeps its own bounded timeout."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from scripts.e2e.regional.regional_commands import RegionalFixtureError


@dataclass(frozen=True)
class ActionWindow:
    end: datetime
    monotonic_end: float

    def remaining(self) -> float:
        return min(
            (self.end - datetime.now(timezone.utc)).total_seconds(),
            self.monotonic_end - time.monotonic(),
        )


WINDOW: ContextVar[ActionWindow | None] = ContextVar(
    "collector_action_window", default=None
)
S = TypeVar("S")
F = TypeVar("F")


def finite_seconds(value: float, *, label: str = "timeout") -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise RegionalFixtureError(f"{label} must be finite and positive")
    return value


def require_action_time(seconds: float = 1) -> None:
    finite_seconds(seconds)
    window = WINDOW.get()
    if window is not None and window.remaining() <= seconds:
        raise RegionalFixtureError(
            "approved maintenance window cannot fit the next action"
        )


@contextmanager
def action_window(end: datetime) -> Iterator[None]:
    if end.tzinfo is None or end.utcoffset() is None:
        raise RegionalFixtureError("maintenance deadline must include a timezone")
    remaining = (end - datetime.now(timezone.utc)).total_seconds()
    finite_seconds(remaining, label="remaining maintenance window")
    previous = WINDOW.get()
    window = ActionWindow(end, time.monotonic() + remaining)
    if previous is not None and previous.remaining() < remaining:
        window = previous
    token = WINDOW.set(window)
    try:
        yield
    finally:
        WINDOW.reset(token)


def bounded_collector_case(
    execute: Callable[[S, Path, int, datetime], int],
) -> Callable[[S, Path, int, datetime], int]:
    @wraps(execute)
    def bounded(
        settings: S, run_dir: Path, attempt: int, maintenance_window_end: datetime
    ) -> int:
        with action_window(maintenance_window_end):
            return execute(settings, run_dir, attempt, maintenance_window_end)

    return bounded


def bounded_window_case(
    execute: Callable[[S, F, Path, int, datetime], dict[str, Any]],
) -> Callable[[S, F, Path, int, datetime], dict[str, Any]]:
    @wraps(execute)
    def bounded(
        settings: S, fixture: F, case_dir: Path, attempt: int, deadline: datetime
    ) -> dict[str, Any]:
        with action_window(deadline):
            return execute(settings, fixture, case_dir, attempt, deadline)

    return bounded
