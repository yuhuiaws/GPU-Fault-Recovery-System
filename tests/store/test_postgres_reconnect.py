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


# --- CP-3 against a real server: the pool survives a password rotation ------
#
# A dedicated LOGIN role is created per test so rotating its password cannot
# disturb the shared test user other files connect with.

import os  # noqa: E402
import secrets  # noqa: E402
import time  # noqa: E402
from urllib.parse import urlsplit, urlunsplit  # noqa: E402

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL", "").strip()


def _dsn_for(role: str, password: str) -> str:
    parts = urlsplit(POSTGRES_URL)
    host = parts.hostname or "127.0.0.1"
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit(
        (parts.scheme, f"{role}:{password}@{host}", parts.path, parts.query, "")
    )


@pytest.fixture
def rotating_role():
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    import psycopg

    role = f"gf_rotate_{secrets.token_hex(4)}"
    admin = psycopg.connect(POSTGRES_URL, autocommit=True)
    try:
        try:
            admin.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'first'")
        except psycopg.errors.InsufficientPrivilege:
            pytest.skip("test user cannot CREATE ROLE")
        # These tests observe the server *refusing* a stale password. A server
        # that never checks passwords (``POSTGRES_HOST_AUTH_METHOD=trust``, the
        # release gate's container until 2026-09-09) cannot show that, so the
        # rotation tests are not evidence there.
        try:
            probe = psycopg.connect(
                _dsn_for(role, "not-the-password"), connect_timeout=5
            )
        except psycopg.OperationalError:
            pass
        else:
            probe.close()
            pytest.skip(
                "server accepts any password (trust auth); a rotation is unobservable"
            )

        def rotate(password: str) -> None:
            admin.execute(f"ALTER ROLE {role} PASSWORD '{password}'")

        yield role, rotate
    finally:
        try:
            admin.execute(f"DROP ROLE IF EXISTS {role}")
        finally:
            admin.close()


def _open_pool(credentials):
    from psycopg_pool import ConnectionPool

    from gpu_fault.store.postgres.pool import open_writer_pool

    pool = open_writer_pool(
        ConnectionPool,
        credentials,
        min_size=1,
        max_size=2,
        timeout=15,
        kwargs={"autocommit": True},
        max_idle=60.0,
        max_lifetime=600.0,
        reconnect_timeout=30.0,
    )
    pool.wait(timeout=15)
    return pool


def _one(pool) -> int:
    with pool.connection() as connection:
        return connection.execute("SELECT 1").fetchone()[0]


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


def test_a_rotated_password_already_on_disk_is_used_by_the_next_reconnect(
    rotating_role, tmp_path
) -> None:
    """Happy path: the Secret file was refreshed before the pool had to
    reconnect. No connection error, no restart."""

    from gpu_fault.store.postgres.pool import StoreCredentials

    role, rotate = rotating_role
    path = tmp_path / "postgres-url"
    path.write_text(_dsn_for(role, "first"))
    credentials = StoreCredentials(_dsn_for(role, "first"), path=str(path))
    pool = _open_pool(credentials)
    try:
        assert _one(pool) == 1

        rotate("second")
        path.write_text(_dsn_for(role, "second"))
        # Force the pool to replace its connection: a connection closed while
        # checked out is discarded on return and a new one is opened.
        with pool.connection() as connection:
            connection.close()

        assert _wait_until(lambda: _one(pool) == 1, timeout=20), (
            "the pool must reconnect with the rotated password"
        )
        stats = pool.get_stats()
        assert stats.get("connections_errors", 0) == 0
        assert credentials.metrics()["credential_rotations_total"] == 1
        assert credentials.source == "file"
    finally:
        pool.close()


def test_a_rotation_the_file_has_not_caught_up_with_recovers_when_it_does(
    rotating_role, tmp_path
) -> None:
    """The 2026-09-07 shape: the database already refuses the old password,
    the projected Secret still carries it. The reconnect fails (counted), the
    pool keeps retrying with backoff and reads the file on each attempt; once
    kubelet delivers the new DSN the next attempt succeeds. Before CP-3 this
    was 300 s of retries, a lost slot and PoolTimeout until a Pod restart."""

    from gpu_fault.store.postgres.pool import StoreCredentials

    role, rotate = rotating_role
    path = tmp_path / "postgres-url"
    path.write_text(_dsn_for(role, "first"))
    credentials = StoreCredentials(_dsn_for(role, "first"), path=str(path))
    pool = _open_pool(credentials)
    try:
        assert _one(pool) == 1

        rotate("second")
        with pool.connection() as connection:
            connection.close()
        # The pool is now retrying with the stale password.
        assert _wait_until(
            lambda: pool.get_stats().get("connections_errors", 0) >= 1, timeout=20
        ), pool.get_stats()

        path.write_text(_dsn_for(role, "second"))

        assert _wait_until(lambda: _one(pool) == 1, timeout=30), (
            "the pool must recover once the file catches up"
        )
        assert credentials.metrics()["credential_rotations_total"] == 1
        assert credentials.metrics()["credential_authentication_failures_total"] >= 1
        # The slot was never given up: reconnect_failed did not fire.
        assert credentials.metrics()["credential_reconnect_failures_total"] == 0
    finally:
        pool.close()
