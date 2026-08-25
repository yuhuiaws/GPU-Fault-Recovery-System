from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Lock

import pytest

from gpu_fault.store import _PooledPostgresDatabase
from gpu_fault.store.postgres.pool import configure_writer_connection


class FakeCursor:
    pass


class FakeConnection:
    def __init__(self, connection_id: int) -> None:
        self.connection_id = connection_id
        self.cursor_count = 0
        self.closed = False
        self.autocommit = False

    @contextmanager
    def cursor(self):
        self.cursor_count += 1
        yield FakeCursor()

    @contextmanager
    def transaction(self):
        yield

    def close(self) -> None:
        self.closed = True


class FakePool:
    def __init__(self) -> None:
        self._lock = Lock()
        self.checkouts = 0
        self.returns = 0
        self.closed = False
        self.connections: list[FakeConnection] = []

    @contextmanager
    def connection(self):
        with self._lock:
            self.checkouts += 1
            connection = FakeConnection(self.checkouts)
            self.connections.append(connection)
        try:
            yield connection
        finally:
            with self._lock:
                self.returns += 1

    def close(self) -> None:
        self.closed = True


def test_pooled_database_returns_connection_after_cursor() -> None:
    pool = FakePool()
    database = _PooledPostgresDatabase(pool)

    with database.cursor():
        assert pool.checkouts == 1
        assert pool.returns == 0

    assert pool.returns == 1


def test_pooled_database_pins_transaction_to_one_connection() -> None:
    pool = FakePool()
    database = _PooledPostgresDatabase(pool)

    with database.transaction():
        with database.cursor():
            pass
        with database.cursor():
            pass

    assert pool.checkouts == 1
    assert pool.returns == 1
    assert pool.connections[0].cursor_count == 2


def test_pooled_database_allows_parallel_checkouts() -> None:
    pool = FakePool()
    database = _PooledPostgresDatabase(pool)
    barrier = Barrier(4)

    def use_cursor(_: int) -> None:
        with database.cursor():
            barrier.wait()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(use_cursor, range(4)))

    assert pool.checkouts == 4
    assert pool.returns == 4


def test_pooled_database_closes_pool() -> None:
    pool = FakePool()
    database = _PooledPostgresDatabase(pool)

    database.close()

    assert pool.closed is True


def test_pooled_database_discards_read_only_connection() -> None:
    pool = FakePool()
    database = _PooledPostgresDatabase(pool)
    error_type = type(
        "ReadOnlySqlTransaction",
        (Exception,),
        {"__module__": "psycopg.errors", "sqlstate": "25006"},
    )

    with pytest.raises(error_type):
        with database.cursor():
            raise error_type("cannot write in a read-only transaction")

    assert pool.connections[0].closed is True


class WriterProbeResult:
    def __init__(self, row) -> None:
        self.row = row

    def fetchone(self):
        return self.row


class WriterProbeConnection:
    def __init__(self, row) -> None:
        self.row = row
        self.autocommit = False
        self.closed = False
        self.queries = []

    def execute(self, query):
        self.queries.append(query)
        return WriterProbeResult(self.row)

    def close(self) -> None:
        self.closed = True


def test_writer_configuration_accepts_writable_connection() -> None:
    connection = WriterProbeConnection((False, "off"))

    configure_writer_connection(connection)

    assert connection.closed is False
    assert connection.autocommit is False
    assert len(connection.queries) == 1


def test_writer_configuration_rejects_reader_connection() -> None:
    from psycopg import OperationalError

    connection = WriterProbeConnection((False, "on"))

    with pytest.raises(OperationalError, match="read-only replica"):
        configure_writer_connection(connection)

    assert connection.closed is True
