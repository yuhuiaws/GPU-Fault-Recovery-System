"""The pooled keep-alive transport: reuse, stale retry, idle and age expiry.

Every node-side poster goes through :mod:`gpu_fault.transport.http_client`.
A connection kept across posts must respect the NLB's fixed 350 s idle
timeout (closed here before it is dead there), TCP keepalive must hold the
flow open in between, an aged connection must be replaced so the fleet
re-spreads over Pods and workers, and a connection the peer closed must fail
closed - retried once only for an idempotent request. The outcomes are the
counters the node Agent publishes on ``/healthz``.
"""

from __future__ import annotations

import socket
from typing import Any
from urllib.request import Request

import pytest

from gpu_fault.transport import http_client
from gpu_fault.transport.http_client import (
    NLB_IDLE_TIMEOUT_SECONDS,
    ConnectionPolicy,
    _KeepAliveConnectionPool,
    _PooledConnection,
    enable_tcp_keepalive,
)
from tests.regional._cov95_runtime_http import (  # noqa: F401 - the http_wire fixture
    Wire,
    http_wire_fixture,
)

KEY = ("http", "example.test", 80)
URL = "http://example.test/events"


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.sock: Any = None

    def close(self) -> None:
        self.closed = True


def _pool(
    monkeypatch, *, now: float = 1000.0, **policy: float
) -> _KeepAliveConnectionPool:
    """A pool at a frozen clock with no proxies; ``now`` is ``time.monotonic``."""

    monkeypatch.setattr(http_client.time, "monotonic", lambda: now)
    monkeypatch.setattr(http_client, "getproxies", lambda: {})
    return _KeepAliveConnectionPool(ConnectionPolicy(**policy))


def _cached(
    pool: _KeepAliveConnectionPool,
    connection: FakeConnection,
    *,
    now: float = 1000.0,
    idle_for: float = 0.0,
    expires_at: float | None = None,
) -> _PooledConnection:
    record = _PooledConnection(
        connection,
        opened_at=now - idle_for,
        last_used_at=now - idle_for,
        expires_at=expires_at,
    )
    pool._cache()[KEY] = record
    return record


def _exchange_recorder(pool, seen: list[Any], *, fail_on: FakeConnection | None = None):
    def exchange(record, *_args, **_kwargs):
        seen.append(record.connection)
        if record.connection is fail_on:
            raise OSError("stale keep-alive")
        return "ok"

    return exchange


def test_reused_post_without_idempotency_key_is_not_retried(monkeypatch) -> None:
    pool = _pool(monkeypatch)
    reused = FakeConnection()
    _cached(pool, reused)
    seen: list[Any] = []
    monkeypatch.setattr(
        pool, "_exchange", _exchange_recorder(pool, seen, fail_on=reused)
    )
    with pytest.raises(OSError):
        pool.send(Request(URL, data=b"{}", method="POST"))
    assert seen == [reused]
    assert reused.closed, "a connection that failed mid-request is closed"
    stats = pool.stats()
    assert stats["stale_connection_failures_total"] == 1
    assert stats["stale_connection_retries_total"] == 0


@pytest.mark.parametrize(
    "header",
    [{"Idempotency-Key": "event-1"}, {"X-GPU-Fault-Request-ID": "heartbeat-1"}],
    ids=["idempotency-key", "request-id"],
)
def test_reused_idempotent_post_may_retry_once(
    monkeypatch, header: dict[str, str]
) -> None:
    """``X-GPU-Fault-Request-ID`` is what the node Agent's heartbeat carries."""

    pool = _pool(monkeypatch)
    reused = FakeConnection()
    replacement = FakeConnection()
    _cached(pool, reused)
    monkeypatch.setattr(pool, "_connect", lambda *_args: replacement)
    seen: list[Any] = []
    monkeypatch.setattr(
        pool, "_exchange", _exchange_recorder(pool, seen, fail_on=reused)
    )
    result = pool.send(Request(URL, data=b"{}", headers=header, method="POST"))
    assert result == "ok"
    assert seen == [reused, replacement]
    stats = pool.stats()
    assert stats["stale_connection_retries_total"] == 1
    assert stats["connections_opened_total"] == 1
    assert stats["requests_reused_total"] == 0, "the stale attempt was not a reuse"


def test_an_idle_connection_is_closed_before_the_nlb_would_drop_it(monkeypatch) -> None:
    pool = _pool(monkeypatch, max_idle_seconds=90.0)
    idle = FakeConnection()
    fresh = FakeConnection()
    _cached(pool, idle, idle_for=91.0)
    monkeypatch.setattr(pool, "_connect", lambda *_args: fresh)
    seen: list[Any] = []
    monkeypatch.setattr(pool, "_exchange", _exchange_recorder(pool, seen))
    assert pool.send(Request(URL, data=b"{}", method="POST")) == "ok"
    assert idle.closed, "closed here, cleanly, not discovered dead by the request"
    assert seen == [fresh]
    stats = pool.stats()
    assert stats["idle_expired_total"] == 1 and stats["connections_opened_total"] == 1
    assert stats["stale_connection_retries_total"] == 0, "no attempt was wasted on it"


def test_a_connection_within_its_idle_budget_is_reused(monkeypatch) -> None:
    pool = _pool(monkeypatch, max_idle_seconds=90.0)
    warm = FakeConnection()
    _cached(pool, warm, idle_for=60.0)
    seen: list[Any] = []
    monkeypatch.setattr(pool, "_exchange", _exchange_recorder(pool, seen))
    assert pool.send(URL) == "ok"
    assert seen == [warm] and not warm.closed
    assert pool.stats()["requests_reused_total"] == 1


def test_an_aged_connection_is_replaced_so_the_fleet_respreads(monkeypatch) -> None:
    pool = _pool(monkeypatch, max_age_seconds=1800.0)
    aged = FakeConnection()
    fresh = FakeConnection()
    _cached(pool, aged, idle_for=5.0, expires_at=999.0)
    monkeypatch.setattr(pool, "_connect", lambda *_args: fresh)
    seen: list[Any] = []
    monkeypatch.setattr(pool, "_exchange", _exchange_recorder(pool, seen))
    pool.send(URL)
    assert aged.closed and seen == [fresh]
    assert pool.stats()["age_expired_total"] == 1


def test_connection_lifetimes_are_jittered_and_zero_disables_them(
    monkeypatch, http_wire: Wire
) -> None:
    """A fresh connection's lifetime is the policy's age +-10 %, drawn per
    connection and honoured on the next reuse; 0 means unbounded."""

    clock = {"now": 1000.0}
    monkeypatch.setattr(http_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(http_client, "getproxies", lambda: {})
    # An idle ceiling far above the lifetime, so only the age can end it here.
    pool = _KeepAliveConnectionPool(
        ConnectionPolicy(max_age_seconds=1800.0, max_idle_seconds=10**6)
    )
    pool.jitter = lambda low, high: low  # the shortest lifetime: 1800 x 0.9
    pool.send(URL)
    clock["now"] += 1800.0 * 0.9 - 1.0
    pool.send(URL)
    assert len(http_wire.connections) == 1, "reused before its lifetime ends"
    clock["now"] += 2.0
    pool.send(URL)
    assert len(http_wire.connections) == 2, "replaced at 0.9 x age"
    assert http_wire.connections[0].closed, "the aged connection is closed, not leaked"
    assert pool.stats()["age_expired_total"] == 1
    assert pool.stats()["connections_opened_total"] == 2

    unbounded = _KeepAliveConnectionPool(
        ConnectionPolicy(max_age_seconds=0.0, max_idle_seconds=10**9)
    )
    unbounded.send(URL)
    clock["now"] += 10 * 24 * 3600.0
    unbounded.send(URL)
    assert len(http_wire.connections) == 3, "0 keeps the connection for ever"
    assert unbounded.stats()["requests_reused_total"] == 1


def test_the_policy_fails_closed_on_values_the_nlb_cannot_honour() -> None:
    policy = ConnectionPolicy.from_environment({})
    assert (
        policy.max_idle_seconds,
        policy.max_age_seconds,
        policy.tcp_keepalive_seconds,
    ) == (90.0, 1800.0, 60.0)
    assert policy.max_idle_seconds < NLB_IDLE_TIMEOUT_SECONDS
    assert policy.tcp_keepalive_seconds < NLB_IDLE_TIMEOUT_SECONDS
    custom = ConnectionPolicy.from_environment(
        {
            "GPU_FAULT_HTTP_CONNECTION_IDLE_SECONDS": "300",
            "GPU_FAULT_HTTP_CONNECTION_MAX_AGE_SECONDS": "0",
            "GPU_FAULT_HTTP_TCP_KEEPALIVE_SECONDS": "0",
        }
    )
    assert (
        custom.max_idle_seconds,
        custom.max_age_seconds,
        custom.tcp_keepalive_seconds,
    ) == (300.0, 0.0, 0.0)
    for name, value in (
        ("GPU_FAULT_HTTP_CONNECTION_IDLE_SECONDS", "0"),
        ("GPU_FAULT_HTTP_CONNECTION_IDLE_SECONDS", "350"),
        ("GPU_FAULT_HTTP_CONNECTION_MAX_AGE_SECONDS", "-1"),
        ("GPU_FAULT_HTTP_TCP_KEEPALIVE_SECONDS", "-5"),
        ("GPU_FAULT_HTTP_TCP_KEEPALIVE_SECONDS", "400"),
    ):
        with pytest.raises(ValueError):
            ConnectionPolicy.from_environment({name: value})


def test_tcp_keepalive_is_armed_on_a_real_socket_and_skipped_where_impossible() -> None:
    with socket.socket() as sock:
        assert enable_tcp_keepalive(sock, 60.0), "a real socket takes TCP keepalive"
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 60
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 15
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 4
        assert not enable_tcp_keepalive(sock, 0.0), "0 leaves the socket alone"
    assert not enable_tcp_keepalive(object(), 60.0), "a fake socket is tolerated"
    assert not enable_tcp_keepalive(None, 60.0), "no socket, nothing to arm"


def test_stats_name_every_counter_and_the_policy_and_reset(monkeypatch) -> None:
    pool = _pool(monkeypatch)
    stats = pool.stats()
    assert set(http_client.STAT_NAMES) <= set(stats)
    assert all(stats[name] == 0 for name in http_client.STAT_NAMES), (
        "fresh pool counts nothing"
    )
    assert stats["max_idle_seconds"] == 90.0
    monkeypatch.setattr(pool, "_connect", lambda *_args: FakeConnection())
    monkeypatch.setattr(pool, "_exchange", _exchange_recorder(pool, []))
    pool.send(URL)
    assert pool.stats()["connections_opened_total"] == 1
    pool.reset_stats()
    assert pool.stats()["connections_opened_total"] == 0
    assert set(http_client.transport_stats()) == set(stats)


def test_the_periodic_summary_is_logged_once_per_interval(monkeypatch, caplog) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(http_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(http_client, "getproxies", lambda: {})
    pool = _KeepAliveConnectionPool(ConnectionPolicy())
    monkeypatch.setattr(pool, "_connect", lambda *_args: FakeConnection())
    monkeypatch.setattr(pool, "_exchange", _exchange_recorder(pool, []))
    with caplog.at_level("INFO", logger="gpu_fault.transport.http_client"):
        pool.send(URL)
        clock["now"] += http_client.STATS_LOG_INTERVAL_SECONDS + 1.0
        pool.send(URL)
        clock["now"] += 1.0
        pool.send(URL)
    lines = [
        record.message
        for record in caplog.records
        if "transport pool" in record.message
    ]
    assert len(lines) == 1, "one summary per interval, not per request"
    assert "idle max 90 s" in lines[0] and "tcp keepalive 60 s" in lines[0]
