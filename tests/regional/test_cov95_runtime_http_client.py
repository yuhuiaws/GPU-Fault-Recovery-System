from __future__ import annotations

import io
import ssl
from concurrent.futures import ThreadPoolExecutor
from http.client import RemoteDisconnected
from threading import Barrier
from urllib.error import HTTPError, URLError
from urllib.request import OpenerDirector, Request

import pytest

from gpu_fault.transport import http_client
from tests.regional._cov95_runtime_http import Reply, Wire
from tests.regional._cov95_runtime_http import http_wire_fixture as http_wire_fixture
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("status", [403, 429, 503])
def test_received_http_error_is_not_retried_as_a_broken_keepalive_connection(
    http_wire: Wire, method: str, status: int
) -> None:
    http_wire.replies = [Reply(), Reply(status=status, body=b"unit refusal"), Reply()]
    with http_client.urlopen("http://unit.invalid/warm", timeout=2) as warm:
        assert warm.status == 200
    request = Request(
        "http://unit.invalid/action",
        method=method,
        data=b"unit-body" if method == "POST" else None,
        headers={"Idempotency-Key": "unit-idempotency"},
    )
    with pytest.raises(HTTPError) as raised:
        http_client.urlopen(request, timeout=2)
    assert raised.value.code == status
    assert raised.value.read() == b"unit refusal"
    assert [(verb, path) for verb, path, _body, _headers in http_wire.requests] == [
        ("GET", "/warm"),
        (method, "/action"),
    ]
    assert len(http_wire.connections) == 1
    assert http_wire.connections[0].closed is False


@pytest.mark.parametrize("will_close", [False, True])
def test_fresh_http_error_preserves_response_and_respects_server_connection_close(
    http_wire: Wire, will_close: bool
) -> None:
    http_wire.replies = [Reply(status=503, body=b"unit busy", will_close=will_close)]
    with pytest.raises(HTTPError) as raised:
        http_client.urlopen("http://unit.invalid/action", timeout=2)
    assert raised.value.code == 503
    assert raised.value.read() == b"unit busy"
    assert len(http_wire.requests) == len(http_wire.connections) == 1
    assert http_wire.connections[0].closed is will_close
    with http_client.urlopen("http://unit.invalid/next", timeout=2) as result:
        assert result.status == 200
    assert len(http_wire.connections) == (2 if will_close else 1)


@pytest.mark.parametrize("error_kind", ["protocol", "socket"])
@pytest.mark.parametrize(
    "identity_header",
    [None, "Idempotency-Key", "X-GPU-Fault-Request-ID", "X-GPU-Fault-Command-ID"],
)
def test_broken_keepalive_retries_post_only_with_an_idempotency_identity(
    http_wire: Wire, error_kind: str, identity_header: str | None
) -> None:
    error = (
        RemoteDisconnected("unit disconnected")
        if error_kind == "protocol"
        else OSError("unit reset")
    )
    http_wire.replies = [Reply(), error, Reply()]
    with http_client.urlopen("http://unit.invalid/warm", timeout=1) as result:
        assert result.status == 200
    request = Request(
        "http://unit.invalid/action",
        method="POST",
        data=b"unit-body",
        headers={} if identity_header is None else {identity_header: "unit-id"},
    )
    if identity_header is None:
        with pytest.raises(URLError if error_kind == "protocol" else OSError):
            http_client.urlopen(request, timeout=7)
        assert len(http_wire.connections) == 1
        assert len(http_wire.requests) == 2
    else:
        with http_client.urlopen(request, timeout=7) as result:
            assert result.status == 200
        assert len(http_wire.connections) == 2
        assert len(http_wire.requests) == 3
        assert http_wire.requests[1][2] == http_wire.requests[2][2] == b"unit-body"
    assert http_wire.connections[0].closed is True


@pytest.mark.parametrize("error_kind", ["protocol", "socket"])
def test_fresh_connection_failure_is_not_retried_transparently(
    http_wire: Wire, error_kind: str
) -> None:
    http_wire.replies = [
        RemoteDisconnected("unit disconnected")
        if error_kind == "protocol"
        else OSError("unit socket")
    ]
    with pytest.raises(URLError if error_kind == "protocol" else OSError):
        http_client.urlopen("http://unit.invalid", timeout=2)
    assert len(http_wire.connections) == len(http_wire.requests) == 1
    assert http_wire.connections[0].closed is True


def test_reuse_updates_socket_budget_and_keeps_query_and_headers(
    http_wire: Wire,
) -> None:
    with http_client.urlopen("http://unit.invalid", timeout=1):
        pass
    socket = http_wire.connections[0].sock
    with http_client.urlopen(
        Request("http://unit.invalid?scope=owned", headers={"X-Unit": "value"}),
        timeout=7,
    ) as result:
        assert result.read() == b"unit-response"
    assert len(http_wire.connections) == 1
    assert socket.timeouts == [7]
    assert http_wire.connections[0].timeout == 7
    assert http_wire.requests[-1][1] == "/?scope=owned"
    assert http_wire.requests[-1][3]["X-unit"] == "value"
    assert http_wire.requests[-1][3]["Connection"] == "keep-alive"


def test_https_connections_are_separate_for_distinct_trust_contexts(
    http_wire: Wire,
) -> None:
    first = ssl.create_default_context()
    second = ssl.create_default_context()
    for context in (first, first, second):
        with http_client.urlopen(
            "https://unit.invalid", timeout=2, ssl_context=context
        ):
            pass
    assert len(http_wire.connections) == 2
    assert [connection.options["context"] for connection in http_wire.connections] == [
        first,
        second,
    ]
    assert all(connection.port == 443 for connection in http_wire.connections), (
        "HTTPS must retain its secure default port"
    )
    assert first.verify_mode == second.verify_mode == ssl.CERT_REQUIRED


def test_http10_response_is_not_kept_for_reuse(http_wire: Wire) -> None:
    http_wire.replies = [Reply(version=10), Reply()]
    for _ in range(2):
        with http_client.urlopen("http://unit.invalid"):
            pass
    assert len(http_wire.connections) == 2
    assert http_wire.connections[0].closed is True


def test_connection_cache_evicts_only_its_oldest_idle_entry(http_wire: Wire) -> None:
    for index in range(9):
        with http_client.urlopen(f"http://unit-{index}.invalid", timeout=1):
            pass
    assert len(http_wire.connections) == 9
    assert [connection.closed for connection in http_wire.connections] == [True] + [
        False
    ] * 8
    with http_client.urlopen("http://unit-8.invalid", timeout=2):
        pass
    assert len(http_wire.connections) == 9


def test_each_thread_reuses_and_closes_only_its_own_connections(
    http_wire: Wire,
) -> None:
    rendezvous = Barrier(2)

    def work() -> None:
        try:
            rendezvous.wait(timeout=5)
            for _ in range(2):
                with http_client.urlopen("http://unit.invalid", timeout=1):
                    pass
        finally:
            http_client.CONNECTION_POOL.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(work) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert len(http_wire.connections) == 2
    assert len({connection.thread_id for connection in http_wire.connections}) == 2
    assert all(connection.closed for connection in http_wire.connections), (
        "each worker must close its own cached connections"
    )


def test_non_http_scheme_delegates_through_the_mocked_standard_opener(
    monkeypatch: pytest.MonkeyPatch, http_wire: Wire
) -> None:
    calls = []

    def open_local(opener, target, data=None, timeout=None):
        calls.append((target.full_url, timeout))
        return io.BytesIO(b"unit-delegated")

    monkeypatch.setattr(OpenerDirector, "open", open_local)
    with http_client.urlopen("data:text/plain,unit", timeout=3) as response:
        assert response.read() == b"unit-delegated"
    assert calls == [("data:text/plain,unit", 3)]
    assert http_wire.connections == []
