from __future__ import annotations

import time
from contextlib import contextmanager
from threading import RLock, local
from typing import Any, Callable
from weakref import WeakKeyDictionary

from gpu_fault.store.shared.errors import (
    connection_is_lost,
)


def _reject_reader(connection: Any, message: str) -> None:
    """Close ``connection`` and raise if it is a replica or a read-only session."""

    original_autocommit = connection.autocommit
    try:
        if not original_autocommit:
            connection.autocommit = True
        in_recovery, read_only = connection.execute(
            """
            SELECT
                pg_is_in_recovery(),
                current_setting('transaction_read_only')
            """
        ).fetchone()
        if in_recovery or read_only == "on":
            connection.close()
            from psycopg import OperationalError

            raise OperationalError(message)
    finally:
        if not connection.closed and connection.autocommit != original_autocommit:
            connection.autocommit = original_autocommit


def configure_writer_connection(connection: Any) -> None:
    """Reject a new connection that Aurora resolved to a reader."""

    _reject_reader(connection, "PostgreSQL connection resolved to a read-only replica")


# How long a pooled connection's last writer probe stays good. The pool's
# ``check`` hook runs on every checkout and ``PooledPostgresDatabase.cursor``
# checks out per statement outside a transaction, so an unthrottled probe would
# double the round trips of every autocommit read. Five seconds bounds how
# long stale reads can be served from a demoted writer after a failover.
WRITER_CHECK_INTERVAL_SECONDS = 5.0


class WriterCheck:
    """psycopg_pool ``check`` hook: discard a connection whose server was demoted.

    ``configure_writer_connection`` runs once, when the connection is opened.
    After an Aurora failover the writer endpoint moves; a connection opened
    before it now points at a reader. Writes on it fail with SQLSTATE 25006 and
    ``PooledPostgresDatabase`` discards it, but autocommit reads keep succeeding
    against the reader until ``max_lifetime`` recycles the connection
    (architecture review 2026-09-07, item D3). This re-asks
    ``pg_is_in_recovery()`` on checkout, at most once per
    ``interval_seconds`` per connection; a failed probe is not remembered, so
    the next checkout probes again.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = WRITER_CHECK_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._checked_at: WeakKeyDictionary[Any, float] = WeakKeyDictionary()

    def __call__(self, connection: Any) -> None:
        now = self._clock()
        checked_at = self._checked_at.get(connection)
        if checked_at is not None and now - checked_at < self.interval_seconds:
            return
        _reject_reader(
            connection,
            "PostgreSQL connection was demoted to a read-only replica since it "
            "was opened",
        )
        self._checked_at[connection] = now


check_writer_connection = WriterCheck()


def open_writer_pool(
    pool_class: Callable[..., Any], conninfo: str, **kwargs: Any
) -> Any:
    """Open ``pool_class`` with both writer hooks: ``configure`` rejects a
    reader at connect time, ``check`` rejects one demoted since (item D3)."""

    return pool_class(
        conninfo=conninfo,
        configure=configure_writer_connection,
        check=check_writer_connection,
        open=True,
        **kwargs,
    )


class PooledPostgresDatabase:
    """Connection-pool facade that pins transactions to one connection."""

    def __init__(self, pool) -> None:
        self.pool = pool
        self._local = local()
        self._metrics_lock = RLock()
        self._checkout_count = 0
        self._checkout_sum_seconds = 0.0
        self._checkout_max_seconds = 0.0

    @contextmanager
    def _connection(self):
        started = time.monotonic()
        with self.pool.connection() as connection:
            waited = time.monotonic() - started
            with self._metrics_lock:
                self._checkout_count += 1
                self._checkout_sum_seconds += waited
                self._checkout_max_seconds = max(self._checkout_max_seconds, waited)
            try:
                yield connection
            except Exception as exc:
                if connection_is_lost(exc):
                    connection.close()
                raise

    def metrics_snapshot(self) -> dict:
        with self._metrics_lock:
            return {
                "checkout_count": self._checkout_count,
                "checkout_sum_seconds": self._checkout_sum_seconds,
                "checkout_max_seconds": self._checkout_max_seconds,
            }

    @property
    def in_transaction(self) -> bool:
        """Whether this thread is inside ``transaction()`` (F-J4)."""

        return getattr(self._local, "connection", None) is not None

    @contextmanager
    def cursor(self):
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            with connection.cursor() as cursor:
                yield cursor
            return
        with self._connection() as connection:
            with connection.cursor() as cursor:
                yield cursor

    @contextmanager
    def transaction(self):
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            with existing.transaction():
                yield
            return
        with self._connection() as connection:
            self._local.connection = connection
            try:
                with connection.transaction():
                    yield
            finally:
                del self._local.connection

    def close(self) -> None:
        self.pool.close()
