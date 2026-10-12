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


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.sock: Any = None

    def close(self) -> None:
        self.closed = True


def _pool(**policy: float) -> _KeepAliveConnectionPool:
    pool = _KeepAliveConnectionPool(ConnectionPolicy(**policy))
    pool._proxies = {}
    return pool


def _cached(
    pool: _KeepAliveConnectionPool,
    key: tuple,
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
    pool._cache()[key] = record
    return record


KEY = ("http", "example.test", 80)


def test_reused_post_without_idempotency_key_is_not_retried(monkeypatch) -> None:
    pool = _pool()
    reused = FakeConnection()
    _cached(pool, KEY, reused)
    monkeypatch.setattr(http_client.time, "monotonic", lambda: 1000.0)
    calls = []

    def exchange(*_args, **_kwargs):
        calls.append(1)
        raise OSError("response connection closed")

    monkeypatch.setattr(pool, "_exchange", exchange)
    with pytest.raises(OSError):
        pool.send(Request("http://example.test/events", data=b"{}", method="POST"))
    assert len(calls) == 1
    assert reused.closed
    assert pool.stats()["stale_connection_failures_total"] == 1
    assert pool.stats()["stale_connection_retries_total"] == 0


def test_reused_idempotent_post_may_retry_once(monkeypatch) -> None:
    pool = _pool()
    reused = FakeConnection()
    replacement = FakeConnection()
    _cached(pool, KEY, reused)
    monkeypatch.setattr(http_client.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(pool, "_connect", lambda *_args: replacement)
    calls = []

    def exchange(record, *_args, **_kwargs):
        calls.append(record.connection)
        if record.connection is reused:
            raise OSError("stale keep-alive")
        return "ok"

    monkeypatch.setattr(pool, "_exchange", exchange)
    result = pool.send(
        Request(
            "http://example.test/events",
            data=b"{}",
            headers={"Idempotency-Key": "event-1"},
            method="POST",
        )
    )
    assert result == "ok"
    assert calls == [reused, replacement]
    stats = pool.stats()
    assert stats["stale_connection_retries_total"] == 1
    assert stats["connections_opened_total"] == 1
    assert stats["requests_reused_total"] == 0, "the stale attempt was not a reuse"


def test_a_request_id_header_makes_a_post_retryable() -> None:
    """The node Agent's heartbeat carries ``X-GPU-Fault-Request-ID`` for this."""

    request = Request("http://example.test/heartbeat", data=b"{}", method="POST")
    assert not _KeepAliveConnectionPool._retryable_request(request, {})
    assert _KeepAliveConnectionPool._retryable_request(
        request, {"X-GPU-Fault-Request-ID": "abc"}
    )


def test_an_idle_connection_is_closed_before_the_nlb_would_drop_it(monkeypatch) -> None:
    pool = _pool(max_idle_seconds=90.0)
    idle = FakeConnection()
    fresh = FakeConnection()
    _cached(pool, KEY, idle, now=1000.0, idle_for=91.0)
    monkeypatch.setattr(http_client.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(pool, "_connect", lambda *_args: fresh)
    seen = []
    monkeypatch.setattr(
        pool,
        "_exchange",
        lambda record, *_a, **_k: seen.append(record.connection) or "ok",
    )
    assert (
        pool.send(Request("http://example.test/x", data=b"{}", method="POST")) == "ok"
    )
    assert idle.closed, "closed here, cleanly, not discovered dead by the request"
    assert seen == [fresh]
    stats = pool.stats()
    assert stats["idle_expired_total"] == 1 and stats["connections_opened_total"] == 1
    assert stats["stale_connection_retries_total"] == 0, "no attempt was wasted on it"


def test_a_connection_within_its_idle_budget_is_reused(monkeypatch) -> None:
    pool = _pool(max_idle_seconds=90.0)
    warm = FakeConnection()
    _cached(pool, KEY, warm, now=1000.0, idle_for=60.0)
    monkeypatch.setattr(http_client.time, "monotonic", lambda: 1000.0)
    seen = []
    monkeypatch.setattr(
        pool,
        "_exchange",
        lambda record, *_a, **_k: seen.append(record.connection) or "ok",
    )
    assert pool.send("http://example.test/x") == "ok"
    assert seen == [warm] and not warm.closed
    assert pool.stats()["requests_reused_total"] == 1


def test_an_aged_connection_is_replaced_so_the_fleet_respreads(monkeypatch) -> None:
    pool = _pool(max_age_seconds=1800.0)
    aged = FakeConnection()
    fresh = FakeConnection()
    _cached(pool, KEY, aged, now=1000.0, idle_for=5.0, expires_at=999.0)
    monkeypatch.setattr(http_client.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(pool, "_connect", lambda *_args: fresh)
    seen = []
    monkeypatch.setattr(
        pool,
        "_exchange",
        lambda record, *_a, **_k: seen.append(record.connection) or "ok",
    )
    pool.send("http://example.test/x")
    assert aged.closed and seen == [fresh]
    assert pool.stats()["age_expired_total"] == 1


def test_connection_lifetimes_are_jittered_and_zero_disables_them() -> None:
    pool = _pool(max_age_seconds=1800.0)
    pool._jitter = lambda low, high: high
    assert pool._expiry(0.0) == pytest.approx(1800.0 * 1.1)
    pool._jitter = lambda low, high: low
    assert pool._expiry(0.0) == pytest.approx(1800.0 * 0.9)
    assert _pool(max_age_seconds=0.0)._expiry(0.0) is None, (
        "0 keeps connections for ever"
    )


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
        assert enable_tcp_keepalive(sock, 60.0)
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 60
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 15
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 4
        assert not enable_tcp_keepalive(sock, 0.0), "0 leaves the socket alone"
    assert not enable_tcp_keepalive(object(), 60.0), "a fake socket is tolerated"
    assert not enable_tcp_keepalive(None, 60.0)


def test_stats_name_every_counter_and_the_policy_and_reset() -> None:
    pool = _pool()
    stats = pool.stats()
    assert set(http_client.STAT_NAMES) <= set(stats)
    assert all(stats[name] == 0 for name in http_client.STAT_NAMES)
    assert stats["max_idle_seconds"] == 90.0
    pool._count("connections_opened_total")
    assert pool.stats()["connections_opened_total"] == 1
    pool.reset_stats()
    assert pool.stats()["connections_opened_total"] == 0
    assert set(http_client.transport_stats()) == set(stats)


def test_the_periodic_summary_is_logged_once_per_interval(monkeypatch, caplog) -> None:
    pool = _pool()
    pool._last_log = 0.0
    with caplog.at_level("INFO", logger="gpu_fault.transport.http_client"):
        pool._maybe_log(http_client.STATS_LOG_INTERVAL_SECONDS + 1.0)
        pool._maybe_log(http_client.STATS_LOG_INTERVAL_SECONDS + 2.0)
    lines = [
        record.message
        for record in caplog.records
        if "transport pool" in record.message
    ]
    assert len(lines) == 1
    assert "idle max 90 s" in lines[0] and "tcp keepalive 60 s" in lines[0]
