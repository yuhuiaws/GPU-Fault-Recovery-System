from __future__ import annotations

import time
from contextlib import contextmanager
from threading import RLock, local
from typing import Any

from gpu_fault.store.shared.errors import (
    connection_is_lost,
)


def configure_writer_connection(connection: Any) -> None:
    """Reject a new connection that Aurora resolved to a reader."""

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

            raise OperationalError(
                "PostgreSQL connection resolved to a read-only replica"
            )
    finally:
        if not connection.closed and connection.autocommit != original_autocommit:
            connection.autocommit = original_autocommit


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
