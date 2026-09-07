"""A pooled connection is re-checked against a demoted writer on checkout.

Architecture review 2026-09-07, item D3. ``configure_writer_connection`` runs
once, when the pool opens a connection, and rejects one that Aurora resolved to
a reader. After a failover the writer endpoint moves; a connection opened
before it keeps pointing at the old instance, now a reader. Writes on it fail
with SQLSTATE 25006 and the pool discards it, but autocommit *reads* keep
succeeding against the reader, so the control plane serves stale rows until
``max_lifetime`` recycles the connection. psycopg_pool's ``check`` hook runs
on every checkout; the writer check below asks ``pg_is_in_recovery()`` there,
throttled per connection so the probe is not a round trip on every statement.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.store.postgres.pool import (
    WriterCheck,
    check_writer_connection,
    configure_writer_connection,
    open_writer_pool,
)


class ProbeResult:
    def __init__(self, row: tuple[Any, ...]) -> None:
        self.row = row

    def fetchone(self) -> tuple[Any, ...]:
        return self.row


class ProbeConnection:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = list(rows)
        self.autocommit = True
        self.closed = False
        self.queries: list[str] = []

    def execute(self, query: str) -> ProbeResult:
        self.queries.append(query)
        return ProbeResult(self.rows.pop(0))

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_writer_connection_passes_the_check() -> None:
    connection = ProbeConnection([(False, "off")])

    check_writer_connection(connection)

    assert connection.closed is False
    assert len(connection.queries) == 1


def test_a_connection_now_in_recovery_is_closed_and_refused() -> None:
    from psycopg import OperationalError

    connection = ProbeConnection([(True, "off")])

    with pytest.raises(OperationalError, match="demoted"):
        check_writer_connection(connection)

    assert connection.closed is True, "the pool must not hand this connection out"


def test_a_read_only_session_is_refused_like_recovery() -> None:
    from psycopg import OperationalError

    connection = ProbeConnection([(False, "on")])

    with pytest.raises(OperationalError):
        check_writer_connection(connection)

    assert connection.closed is True


def test_the_probe_is_throttled_per_connection() -> None:
    clock = FakeClock()
    check = WriterCheck(interval_seconds=5.0, clock=clock)
    connection = ProbeConnection([(False, "off"), (False, "off"), (True, "off")])

    check(connection)
    clock.now += 1.0
    check(connection)
    assert len(connection.queries) == 1, "a checkout inside the interval is free"

    clock.now += 5.0
    check(connection)
    assert len(connection.queries) == 2, "the interval elapsed: probe again"

    clock.now += 5.0
    from psycopg import OperationalError

    with pytest.raises(OperationalError):
        check(connection)
    assert connection.closed is True


def test_throttling_is_per_connection_not_global() -> None:
    clock = FakeClock()
    check = WriterCheck(interval_seconds=5.0, clock=clock)
    first = ProbeConnection([(False, "off")])
    second = ProbeConnection([(False, "off")])

    check(first)
    check(second)

    assert len(second.queries) == 1, "a fresh connection is probed on first checkout"


def test_a_failed_probe_does_not_count_as_checked() -> None:
    """A demoted connection that somehow comes back must be probed again."""

    from psycopg import OperationalError

    clock = FakeClock()
    check = WriterCheck(interval_seconds=5.0, clock=clock)
    connection = ProbeConnection([(True, "off"), (True, "off")])

    with pytest.raises(OperationalError):
        check(connection)
    connection.closed = False
    with pytest.raises(OperationalError):
        check(connection)

    assert len(connection.queries) == 2


class RecordingPool:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_the_store_pool_is_opened_with_both_writer_hooks() -> None:
    pool = open_writer_pool(
        RecordingPool,
        "postgresql://example",
        min_size=1,
        max_size=2,
        timeout=2.0,
        kwargs={"autocommit": True},
        max_idle=300.0,
    )

    assert isinstance(pool, RecordingPool), "the pool class is the one handed in"
    assert pool.kwargs["configure"] is configure_writer_connection
    assert pool.kwargs["check"] is check_writer_connection
    assert pool.kwargs["conninfo"] == "postgresql://example"
    assert pool.kwargs["open"] is True
    assert pool.kwargs["max_idle"] == 300.0
