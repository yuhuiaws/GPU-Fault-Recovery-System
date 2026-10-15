"""The claim's statistics refresh, against a scripted database (no PostgreSQL).

``queue_statistics.refresh_if_stale`` runs after a claim whose window came
back full: it reads the planner's row count of the queue and analyses the
queue when that count is below two windows, paced per store and serialised
per site. The PostgreSQL behaviour it targets is pinned in
``tests/store/test_postgres_claim_planner_discipline.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from gpu_fault.store.postgres import queue_statistics
from gpu_fault.store.postgres.queue_statistics import (
    CHECK_INTERVAL_SECONDS,
    REFRESH_INTERVAL_SECONDS,
    QueueStatisticsGuard,
    refresh_if_stale,
)

WINDOW = 68


class _LockTimeout(Exception):
    sqlstate = "55P03"


class _Database:
    """Answers the refresh's statements and records them."""

    def __init__(
        self,
        reltuples: float | None,
        *,
        locked: bool = True,
        fail: Exception | None = None,
    ) -> None:
        self.reltuples = reltuples
        self.locked = locked
        self.fail = fail
        self.statements: list[str] = []
        self._row: Any = None

    @contextmanager
    def transaction(self):
        yield

    @contextmanager
    def cursor(self):
        yield self

    def execute(self, statement: str, parameters: tuple) -> None:
        self.statements.append(statement.split()[0] + " " + statement.split()[1])
        if self.fail is not None:
            raise self.fail
        if "pg_class" in statement:
            self._row = None if self.reltuples is None else (self.reltuples,)
        elif "pg_try_advisory_xact_lock" in statement:
            assert parameters == (queue_statistics.STATISTICS_LOCK_CLASS,), parameters
            self._row = (self.locked,)

    def fetchone(self) -> Any:
        return self._row


def _refresh(database: _Database, guard: QueueStatisticsGuard, now: float) -> bool:
    return refresh_if_stale(database, guard, window_rows=WINDOW, now=now)


@pytest.mark.parametrize("reltuples", [0.0, -1.0, 2 * WINDOW - 1.0])
def test_a_count_below_two_windows_is_analysed(reltuples: float) -> None:
    database, guard = _Database(reltuples), QueueStatisticsGuard()

    assert _refresh(database, guard, now=100.0), "a stale count is refreshed"
    assert database.statements == [
        "SELECT reltuples",
        "SELECT pg_try_advisory_xact_lock(%s,",
        "SET LOCAL",
        "ANALYZE gpu_fault_processor_queue",
    ], database.statements
    assert guard.refreshes_total == 1, guard


@pytest.mark.parametrize("reltuples", [2.0 * WINDOW, 5000.0, None])
def test_a_current_or_missing_count_is_left_alone(reltuples: float | None) -> None:
    database, guard = _Database(reltuples), QueueStatisticsGuard()

    assert not _refresh(database, guard, now=100.0), "nothing to refresh"
    assert database.statements == ["SELECT reltuples"], database.statements
    assert guard.refreshes_total == 0, guard


def test_another_process_holding_the_refresh_lock_skips_it() -> None:
    database, guard = _Database(0.0, locked=False), QueueStatisticsGuard()

    assert not _refresh(database, guard, now=100.0), "one refresh per site"
    assert not any(item.startswith("ANALYZE") for item in database.statements), (
        database.statements
    )


def test_the_reads_and_the_refreshes_are_paced() -> None:
    guard = QueueStatisticsGuard()
    current = _Database(5000.0)
    assert not _refresh(current, guard, now=100.0), "a current count"
    assert not _refresh(current, guard, now=100.0 + CHECK_INTERVAL_SECONDS / 2), (
        "inside the check interval"
    )
    assert current.statements == ["SELECT reltuples"], "one read per interval"

    stale = _Database(0.0)
    assert _refresh(stale, guard, now=100.0 + CHECK_INTERVAL_SECONDS), "now stale"
    later = 100.0 + CHECK_INTERVAL_SECONDS + REFRESH_INTERVAL_SECONDS / 2
    assert not _refresh(stale, guard, now=later), "inside the refresh interval"
    assert stale.statements.count("SELECT reltuples") == 1, stale.statements
    assert _refresh(
        stale, guard, now=100.0 + CHECK_INTERVAL_SECONDS + REFRESH_INTERVAL_SECONDS
    ), "after the refresh interval"
    assert guard.refreshes_total == 2, guard


@pytest.mark.parametrize(
    "failure", [_LockTimeout("lock timeout"), RuntimeError("connection reset")]
)
def test_a_failed_refresh_never_reaches_the_claim(failure: Exception) -> None:
    database, guard = _Database(0.0, fail=failure), QueueStatisticsGuard()

    assert not _refresh(database, guard, now=100.0), "the failure is swallowed"
    assert guard.refreshes_total == 0, guard
