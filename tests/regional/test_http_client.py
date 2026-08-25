from __future__ import annotations

from urllib.request import Request

import pytest

from gpu_fault.transport.http_client import _KeepAliveConnectionPool


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_reused_post_without_idempotency_key_is_not_retried(monkeypatch) -> None:
    pool = _KeepAliveConnectionPool()
    pool._proxies = {}
    reused = FakeConnection()
    pool._cache()[("http", "example.test", 80)] = reused
    calls = []

    def exchange(*_args, **_kwargs):
        calls.append(1)
        raise OSError("response connection closed")

    monkeypatch.setattr(pool, "_exchange", exchange)
    with pytest.raises(OSError):
        pool.send(Request("http://example.test/events", data=b"{}", method="POST"))
    assert len(calls) == 1
    assert reused.closed


def test_reused_idempotent_post_may_retry_once(monkeypatch) -> None:
    pool = _KeepAliveConnectionPool()
    pool._proxies = {}
    reused = FakeConnection()
    replacement = FakeConnection()
    pool._cache()[("http", "example.test", 80)] = reused
    monkeypatch.setattr(pool, "_connect", lambda *_args: replacement)
    calls = []

    def exchange(connection, *_args, **_kwargs):
        calls.append(connection)
        if connection is reused:
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
