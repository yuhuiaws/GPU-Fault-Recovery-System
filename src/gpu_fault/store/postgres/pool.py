from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock, local
from typing import Any, Callable
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from gpu_fault.store.shared.errors import (
    connection_is_lost,
)

LOGGER = logging.getLogger(__name__)

# The Deployment mounts the ``gpu-fault-aurora`` Secret as files and points
# this variable at the ``postgres-url`` key. kubelet rewrites the projected
# file after the credential refresher patches the Secret (no subPath, so the
# update propagates), and the pool reads it on every connect.
STORE_URL_FILE_ENV = "GPU_FAULT_STORE_URL_FILE"

# SQLSTATE class 28 is "invalid authorization specification"; 28P01 is the
# password itself. libpq reports a failed handshake as an OperationalError
# with no SQLSTATE at all, so the FATAL message is matched as well.
_AUTHENTICATION_SQLSTATES = frozenset({"28000", "28P01"})
_AUTHENTICATION_MESSAGES = (
    "password authentication failed",
    "authentication failed",
    "no password supplied",
)


def is_authentication_failure(exc: BaseException) -> bool:
    """Whether ``exc`` (or a cause) is the server refusing our credentials."""

    seen: set[int] = set()
    error: BaseException | None = exc
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        module = type(error).__module__ or ""
        if module.startswith("psycopg"):
            sqlstate = getattr(error, "sqlstate", None)
            if sqlstate in _AUTHENTICATION_SQLSTATES:
                return True
            message = str(error).lower()
            if any(marker in message for marker in _AUTHENTICATION_MESSAGES):
                return True
        error = error.__cause__ or error.__context__
    return False


class StoreCredentials:
    """The DSN the pool connects with, re-read from a mounted Secret file.

    ``conninfo`` is the start-up value (``GPU_FAULT_STORE_URL``); ``path`` is
    the projected Secret file (``GPU_FAULT_STORE_URL_FILE``). When the file
    exists and is non-empty it wins, on every call: psycopg_pool resolves a
    callable ``conninfo`` per connection attempt, so a rotated password is
    picked up by the next reconnect without restarting the process. An
    unreadable or empty file (kubelet mid-swap, mount missing) keeps the last
    good DSN rather than turning into an empty conninfo. All counters are
    exported through ``PooledPostgresDatabase.metrics_snapshot``.
    """

    def __init__(self, conninfo: str, *, path: str | None = None) -> None:
        self._fallback = conninfo
        self.path = Path(path) if path else None
        self._lock = Lock()
        self._current = conninfo
        self.source = "env"
        self._rotations = 0
        self._read_failures = 0
        self._authentication_failures = 0
        self._reconnect_failures = 0
        self._warned_unreadable = False
        # Start-up prefers the file too: the env was captured when the Pod was
        # created, the file is whatever the Secret holds now.
        self.reload()

    def conninfo(self) -> str:
        """The DSN to connect with right now."""

        self.reload()
        with self._lock:
            return self._current

    def reload(self) -> bool:
        """Re-read the file; return whether the DSN changed."""

        if self.path is None:
            return False
        try:
            text = self.path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            with self._lock:
                self._read_failures += 1
            if not self._warned_unreadable:
                self._warned_unreadable = True
                LOGGER.warning(
                    "store credential file %s is unreadable (%s); using %s",
                    self.path,
                    exc,
                    self.redacted(),
                )
            return False
        if not text:
            with self._lock:
                self._read_failures += 1
            return False
        with self._lock:
            changed = text != self._current
            self._current = text
            self.source = "file"
            self._warned_unreadable = False
            if changed:
                self._rotations += 1
        if changed:
            LOGGER.info(
                "store credentials reloaded from %s (%s)", self.path, self.redacted()
            )
        return changed

    def on_authentication_failure(self, exc: BaseException) -> bool:
        """Record a refused handshake; return whether a re-read changed the DSN."""

        with self._lock:
            self._authentication_failures += 1
        changed = self.reload()
        LOGGER.warning(
            "postgres refused our credentials (%s); credential file %s %s",
            exc,
            self.path,
            "carried a new password" if changed else "is unchanged so far",
        )
        return changed

    def reconnect_failed(self, pool: Any) -> None:
        """psycopg_pool ``reconnect_failed`` hook: a slot was given up after
        ``reconnect_timeout``. Re-read, then let the pool re-grow."""

        with self._lock:
            self._reconnect_failures += 1
        changed = self.reload()
        LOGGER.error(
            "postgres pool %s gave up a connection slot; credential file %s %s",
            getattr(pool, "name", "?"),
            self.path,
            "now carries a new password" if changed else "is unchanged",
        )
        check = getattr(pool, "check", None)
        if callable(check):
            try:
                check()
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.exception("postgres pool check after reconnect failure failed")

    def redacted(self) -> str:
        with self._lock:
            current = self._current
        try:
            parts = urlsplit(current)
        except ValueError:
            return "<unparsable dsn>"
        if not parts.hostname:
            return "<unparsable dsn>"
        host = parts.hostname
        if parts.port:
            host = f"{host}:{parts.port}"
        return f"{parts.scheme}://{parts.username or ''}:***@{host}{parts.path}"

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "credential_source_file": 1 if self.source == "file" else 0,
                "credential_rotations_total": self._rotations,
                "credential_read_failures_total": self._read_failures,
                "credential_authentication_failures_total": (
                    self._authentication_failures
                ),
                "credential_reconnect_failures_total": self._reconnect_failures,
            }


def _psycopg_connection_class() -> Any:
    """``psycopg.Connection``, or ``None`` when the driver is not installed."""

    try:
        from psycopg import Connection
    except ImportError:  # pragma: no cover - packaging guard
        return None
    return Connection


def reloading_connection_class(base: Any, credentials: StoreCredentials) -> Any:
    """``base`` (psycopg.Connection) whose ``connect`` retries a refused
    handshake once with a freshly read DSN.

    psycopg_pool reads ``conninfo`` and then opens the socket; when kubelet
    swaps the Secret file in between, the attempt fails on the old password
    although the new one is already on disk. Retrying here costs one connect
    instead of the pool's backoff step. When the file has not changed the
    failure propagates and the pool's own backoff (which re-reads the file
    on every attempt) takes over.
    """

    class ReloadingConnection(base):  # type: ignore[misc]
        @classmethod
        def connect(cls, conninfo: str = "", **kwargs: Any) -> Any:
            try:
                return super().connect(conninfo, **kwargs)
            except Exception as exc:
                if not is_authentication_failure(exc):
                    raise
                if not credentials.on_authentication_failure(exc):
                    raise
                return super().connect(credentials.conninfo(), **kwargs)

    ReloadingConnection.__name__ = f"Reloading{getattr(base, '__name__', 'Connection')}"
    return ReloadingConnection


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
    pool_class: Callable[..., Any],
    conninfo: str | StoreCredentials,
    **kwargs: Any,
) -> Any:
    """Open ``pool_class`` with both writer hooks: ``configure`` rejects a
    reader at connect time, ``check`` rejects one demoted since (item D3).

    With :class:`StoreCredentials` the pool also reads its DSN through them on
    every connect, retries a refused handshake with a fresh read, and re-reads
    in ``reconnect_failed`` instead of silently losing the slot (CP-3).
    """

    if isinstance(conninfo, StoreCredentials):
        credentials = conninfo
        kwargs.setdefault("reconnect_failed", credentials.reconnect_failed)
        if "connection_class" not in kwargs:
            connection_base = _psycopg_connection_class()
            if connection_base is not None:
                kwargs["connection_class"] = reloading_connection_class(
                    connection_base, credentials
                )
        return pool_class(
            conninfo=credentials.conninfo,
            configure=configure_writer_connection,
            check=check_writer_connection,
            open=True,
            **kwargs,
        )
    return pool_class(
        conninfo=conninfo,
        configure=configure_writer_connection,
        check=check_writer_connection,
        open=True,
        **kwargs,
    )


class PooledPostgresDatabase:
    """Connection-pool facade that pins transactions to one connection."""

    def __init__(
        self, pool: Any, *, credentials: StoreCredentials | None = None
    ) -> None:
        self.pool = pool
        self.credentials = credentials
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
            snapshot: dict[str, Any] = {
                "checkout_count": self._checkout_count,
                "checkout_sum_seconds": self._checkout_sum_seconds,
                "checkout_max_seconds": self._checkout_max_seconds,
            }
        # psycopg_pool.get_stats(): the measures are live, the counters are a
        # Counter (a key is absent until first incremented). These are the
        # only view of a pool losing slots to a rotated password or a
        # failover; the checkout summary above cannot show either (G-7).
        stats: dict[str, Any] = {}
        get_stats = getattr(self.pool, "get_stats", None)
        if callable(get_stats):
            try:
                stats = dict(get_stats())
            except Exception:  # pragma: no cover - diagnostics only
                stats = {}
        snapshot.update(
            {
                "pool_min_size": int(stats.get("pool_min", 0)),
                "pool_max_size": int(stats.get("pool_max", 0)),
                "pool_size": int(stats.get("pool_size", 0)),
                "pool_available": int(stats.get("pool_available", 0)),
                "requests_waiting": int(stats.get("requests_waiting", 0)),
                "requests_errors_total": int(stats.get("requests_errors", 0)),
                "connections_errors_total": int(stats.get("connections_errors", 0)),
                "connections_lost_total": int(stats.get("connections_lost", 0)),
            }
        )
        if self.credentials is not None:
            snapshot.update(self.credentials.metrics())
        return snapshot

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
