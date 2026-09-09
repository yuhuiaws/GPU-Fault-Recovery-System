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


# --- CP-3 / G-4: the pool re-reads its credentials instead of dying ---------
#
# Aurora rotates the master password every 7 days. The DSN used to be frozen
# in the process at start-up, so once ``max_idle`` recycled a connection the
# pool reconnected with the old password, psycopg_pool retried for
# ``reconnect_timeout`` (300 s) and then gave the slot up; the process
# degraded to PoolTimeout/503 until a CronJob restarted it. Now the Secret is a
# file mount, ``StoreCredentials`` reads it on every connect and the pool's
# ``reconnect_failed`` hook forces a re-read and a ``check()``.


from gpu_fault.store.postgres.pool import (  # noqa: E402
    PooledPostgresDatabase,
    StoreCredentials,
    is_authentication_failure,
)


def test_credentials_prefer_the_mounted_file_over_the_start_up_url(tmp_path) -> None:
    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:file@db/gf\n")

    credentials = StoreCredentials("postgresql://u:env@db/gf", path=str(path))

    assert credentials.conninfo() == "postgresql://u:file@db/gf"
    assert credentials.source == "file"


def test_credentials_fall_back_to_the_start_up_url_without_a_file(tmp_path) -> None:
    credentials = StoreCredentials(
        "postgresql://u:env@db/gf", path=str(tmp_path / "absent")
    )

    assert credentials.conninfo() == "postgresql://u:env@db/gf"
    assert credentials.source == "env"
    # One failed read at construction, one on the connect.
    assert credentials.metrics()["credential_read_failures_total"] == 2


def test_credentials_without_a_path_are_static() -> None:
    credentials = StoreCredentials("postgresql://u:env@db/gf", path=None)

    assert credentials.conninfo() == "postgresql://u:env@db/gf"
    assert credentials.redacted() == "postgresql://u:***@db/gf"


def test_credentials_pick_up_a_rotated_password_on_the_next_connect(tmp_path) -> None:
    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:old@db/gf", path=str(path))
    assert credentials.conninfo().endswith(":old@db/gf"), (
        "the seeded file DSN is read on the first connect"
    )

    path.write_text("postgresql://u:new@db/gf")

    assert credentials.conninfo().endswith(":new@db/gf"), (
        "a rewritten file must be picked up on the next connect"
    )
    assert credentials.metrics()["credential_rotations_total"] == 1
    assert credentials.metrics()["credential_read_failures_total"] == 0


def test_an_empty_or_unreadable_file_keeps_the_last_good_dsn(tmp_path) -> None:
    """kubelet swaps the projected file atomically, but a half-written or
    missing file must never turn into an empty conninfo."""

    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:env@db/gf", path=str(path))
    assert credentials.conninfo().endswith(":old@db/gf"), (
        "the file DSN wins over the env fallback"
    )

    path.write_text("")
    assert credentials.conninfo().endswith(":old@db/gf"), (
        "an empty file must keep the last good DSN"
    )
    path.unlink()
    assert credentials.conninfo().endswith(":old@db/gf"), (
        "a missing file must keep the last good DSN"
    )
    assert credentials.metrics()["credential_read_failures_total"] == 2


def test_authentication_failures_are_recognised_by_sqlstate_or_message() -> None:
    from psycopg import OperationalError

    libpq = OperationalError(
        'connection failed: FATAL:  password authentication failed for user "gf"'
    )
    assert is_authentication_failure(libpq) is True
    typed = type(
        "InvalidPassword",
        (Exception,),
        {"__module__": "psycopg.errors", "sqlstate": "28P01"},
    )("boom")
    assert is_authentication_failure(typed) is True
    assert is_authentication_failure(OperationalError("connection refused")) is False
    assert is_authentication_failure(RuntimeError("password")) is False


class ReloadingPool:
    """psycopg_pool's contract for the parts open_writer_pool relies on."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.checks = 0

    def check(self) -> None:
        self.checks += 1


def test_the_pool_reads_the_dsn_through_the_credentials_on_every_connect(
    tmp_path,
) -> None:
    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:old@db/gf", path=str(path))

    pool = open_writer_pool(ReloadingPool, credentials, min_size=1, max_size=2)

    conninfo = pool.kwargs["conninfo"]
    assert callable(conninfo), "psycopg_pool resolves a callable conninfo per connect"
    assert conninfo().endswith(":old@db/gf"), (
        "the callable resolves the current file DSN"
    )
    path.write_text("postgresql://u:new@db/gf")
    assert conninfo().endswith(":new@db/gf"), (
        "the callable re-reads the file on each connect"
    )
    assert pool.kwargs["configure"] is configure_writer_connection
    assert pool.kwargs["check"] is check_writer_connection
    assert callable(pool.kwargs["reconnect_failed"]), (
        "open_writer_pool must install a reconnect_failed hook"
    )


def test_reconnect_failed_re_reads_the_file_and_rechecks_the_pool(tmp_path) -> None:
    """After ``reconnect_timeout`` psycopg_pool drops a slot and calls this
    hook; it must not be the default no-op logger."""

    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:old@db/gf", path=str(path))
    pool = open_writer_pool(ReloadingPool, credentials, min_size=1, max_size=2)
    path.write_text("postgresql://u:new@db/gf")

    pool.kwargs["reconnect_failed"](pool)

    assert pool.checks == 1
    assert credentials.conninfo().endswith(":new@db/gf"), (
        "reconnect_failed must re-read the rotated password"
    )
    assert credentials.metrics()["credential_reconnect_failures_total"] == 1


class AuthenticatingConnection:
    """Stands in for psycopg.Connection: ``connect`` accepts one password."""

    accepted = "postgresql://u:new@db/gf"
    attempts: list[str] = []

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs: Any):
        from psycopg import OperationalError

        cls.attempts.append(conninfo)
        if conninfo != cls.accepted:
            raise OperationalError(
                'connection failed: FATAL:  password authentication failed for user "u"'
            )
        return cls()


def test_an_authentication_failure_retries_once_with_the_freshly_read_dsn(
    tmp_path,
) -> None:
    """The file changed between the read and the failed handshake (kubelet
    sync lands mid-attempt): retry immediately, do not wait for the backoff."""

    from gpu_fault.store.postgres.pool import reloading_connection_class

    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:old@db/gf", path=str(path))
    AuthenticatingConnection.attempts = []
    connection_class = reloading_connection_class(AuthenticatingConnection, credentials)

    stale = credentials.conninfo()
    path.write_text("postgresql://u:new@db/gf")
    connection = connection_class.connect(stale, autocommit=True)

    assert isinstance(connection, AuthenticatingConnection), (
        "the retry with the fresh DSN must succeed"
    )
    assert AuthenticatingConnection.attempts == [
        "postgresql://u:old@db/gf",
        "postgresql://u:new@db/gf",
    ]
    assert credentials.metrics()["credential_authentication_failures_total"] == 1


def test_an_authentication_failure_with_an_unchanged_file_is_raised(tmp_path) -> None:
    """kubelet has not delivered the new Secret yet: surface the failure so
    psycopg_pool's backoff (which re-reads the file each attempt) takes over."""

    from psycopg import OperationalError

    from gpu_fault.store.postgres.pool import reloading_connection_class

    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:old@db/gf", path=str(path))
    AuthenticatingConnection.attempts = []
    connection_class = reloading_connection_class(AuthenticatingConnection, credentials)

    with pytest.raises(OperationalError, match="password authentication failed"):
        connection_class.connect(credentials.conninfo())

    assert AuthenticatingConnection.attempts == ["postgresql://u:old@db/gf"]
    assert credentials.metrics()["credential_authentication_failures_total"] == 1


def test_a_non_authentication_connect_error_is_not_retried(tmp_path) -> None:
    from psycopg import OperationalError

    from gpu_fault.store.postgres.pool import reloading_connection_class

    class RefusingConnection:
        attempts: list[str] = []

        @classmethod
        def connect(cls, conninfo: str = "", **kwargs: Any):
            cls.attempts.append(conninfo)
            raise OperationalError("connection refused")

    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:old@db/gf", path=str(path))
    connection_class = reloading_connection_class(RefusingConnection, credentials)

    with pytest.raises(OperationalError, match="refused"):
        connection_class.connect(credentials.conninfo())
    assert RefusingConnection.attempts == ["postgresql://u:old@db/gf"]
    assert credentials.metrics()["credential_authentication_failures_total"] == 0


class StatsPool:
    def __init__(self, stats: dict[str, int]) -> None:
        self.stats = stats

    def get_stats(self) -> dict[str, int]:
        return dict(self.stats)


def test_metrics_snapshot_exports_the_psycopg_pool_counters() -> None:
    """G-7: pool_size/available/waiting and the three error counters were the
    only view of a pool losing slots, and none of them was exported."""

    database = PooledPostgresDatabase(
        StatsPool(
            {
                "pool_min": 2,
                "pool_max": 24,
                "pool_size": 5,
                "pool_available": 3,
                "requests_waiting": 1,
                "requests_errors": 7,
                "connections_errors": 4,
                "connections_lost": 2,
            }
        )
    )

    snapshot = database.metrics_snapshot()

    assert snapshot["pool_size"] == 5
    assert snapshot["pool_available"] == 3
    assert snapshot["requests_waiting"] == 1
    assert snapshot["requests_errors_total"] == 7
    assert snapshot["connections_errors_total"] == 4
    assert snapshot["connections_lost_total"] == 2
    assert snapshot["pool_min_size"] == 2
    assert snapshot["pool_max_size"] == 24
    # The original keys survive for the summary Agent 5 already renders.
    assert snapshot["checkout_count"] == 0


def test_metrics_snapshot_zeroes_counters_psycopg_pool_has_not_touched() -> None:
    """psycopg_pool keeps a Counter: a key is absent until first incremented."""

    snapshot = PooledPostgresDatabase(StatsPool({"pool_size": 1})).metrics_snapshot()

    assert snapshot["connections_errors_total"] == 0
    assert snapshot["requests_waiting"] == 0


def test_metrics_snapshot_tolerates_a_pool_without_get_stats() -> None:
    class BarePool:
        pass

    snapshot = PooledPostgresDatabase(BarePool()).metrics_snapshot()

    assert snapshot["pool_size"] == 0
    assert snapshot["connections_errors_total"] == 0


def test_metrics_snapshot_includes_the_credential_counters(tmp_path) -> None:
    path = tmp_path / "postgres-url"
    path.write_text("postgresql://u:old@db/gf")
    credentials = StoreCredentials("postgresql://u:old@db/gf", path=str(path))
    database = PooledPostgresDatabase(StatsPool({}), credentials=credentials)

    snapshot = database.metrics_snapshot()

    assert snapshot["credential_rotations_total"] == 0
    assert snapshot["credential_source_file"] == 1
