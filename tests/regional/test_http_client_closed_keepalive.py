"""A kept-alive connection the control plane closed, against a real local peer.

The pool reuses one connection per thread, and the peer may close it while it
idles (its keep-alive timeout, a Pod leaving a roll). What the transport may do
about it depends on how much of the next request reached the peer:

* closed before anything was written: the pool sees the FIN or RST on the idle
  socket and writes the request to a fresh connection instead -- for every
  request, a non-idempotent claim included, because no byte was delivered;
* the write itself failed: same, nothing complete reached the application;
* closed after the write: only a request carrying a retry identity
  (``X-GPU-Fault-Command-ID`` and friends) is repeated, once; a claim is
  never repeated here, delivered or not, because the transport cannot tell.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.client import HTTPException, RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest

from gpu_fault.transport import http_client
from gpu_fault.transport.http_client import (
    COMMAND_ID_HEADER,
    IDEMPOTENCY_KEY_HEADER,
    REQUEST_ID_HEADER,
    ConnectionPolicy,
    _KeepAliveConnectionPool,
)
from tests.regional._cov95_runtime_http import Reply

CLAIM = "/v1/regional/executors/claim"
RESULT = "/v1/regional/executors/remote-0123456789abcdef01234567/result"
CLAIM_BODY = b'{"executor_id":"cluster-a/executor-0","max_commands":2}'
RESULT_BODY = b'{"lease_token":"lease-1","status":"SUCCEEDED"}'


@dataclass(frozen=True)
class Delivery:
    connection: int
    method: str
    path: str
    body: bytes
    headers: dict[str, str]


class LocalPeer:
    """An HTTP/1.1 peer that closes a kept-alive connection on cue.

    ``/close-when-idle`` answers and then closes the connection the client
    keeps (FIN, or RST with ``close_mode="rst"``), as a keep-alive expiry does;
    a path in ``drop_next`` is read in full and then closed without a status
    line, as a Pod that dies after reading the request looks to the client.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.connections = 0
        self.deliveries: list[Delivery] = []
        self.drop_next: set[str] = set()
        self.close_mode = "fin"
        self.closed = Event()
        self.url = ""

    def delivered(self, path: str) -> list[Delivery]:
        with self.lock:
            return [item for item in self.deliveries if item.path == path]

    def close(self, connection: socket.socket) -> None:
        if self.close_mode == "rst":
            connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            os.close(connection.detach())
        else:
            connection.shutdown(socket.SHUT_RDWR)
        self.closed.set()

    def wait_closed(self) -> None:
        assert self.closed.wait(2), "the peer must close the idle connection"
        # Loopback delivers the FIN/RST with the syscall; the margin keeps a
        # loaded test host from racing the client's check against it.
        time.sleep(0.05)


@pytest.fixture
def peer() -> Iterator[LocalPeer]:
    state = LocalPeer()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            with state.lock:
                state.connections += 1
                self.number = state.connections

        def log_message(self, *args: Any) -> None:
            pass

        def answer(self, body: bytes) -> None:
            with state.lock:
                state.deliveries.append(
                    Delivery(
                        self.number,
                        self.command,
                        self.path,
                        body,
                        {key.lower(): value for key, value in self.headers.items()},
                    )
                )
                dropped = self.path in state.drop_next
                state.drop_next.discard(self.path)
            if dropped:
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            encoded = json.dumps({"connection": self.number}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
            if self.path == "/close-when-idle":
                self.close_connection = True
                state.close(self.connection)

        def do_GET(self) -> None:
            self.answer(b"")

        def do_POST(self) -> None:
            self.answer(self.rfile.read(int(self.headers.get("Content-Length", "0"))))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    serving.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        serving.join(2)
        server.server_close()
        assert not serving.is_alive(), "the local peer must stop"


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[_KeepAliveConnectionPool]:
    monkeypatch.setattr(http_client, "getproxies", lambda: {})
    subject = _KeepAliveConnectionPool(ConnectionPolicy())
    try:
        yield subject
    finally:
        subject.close()


def answered_by(pool: _KeepAliveConnectionPool, request: Request | str) -> int:
    with pool.send(request, timeout=5) as response:
        return int(json.loads(response.read())["connection"])


def post(url: str, path: str, body: bytes, headers: dict[str, str]) -> Request:
    return Request(
        url + path,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )


REQUESTS = {
    "claim": (CLAIM, CLAIM_BODY, {}),
    "result": (RESULT, RESULT_BODY, {COMMAND_ID_HEADER: "remote-0123"}),
    "get": ("/v1/regional/executors/fleet-rollout-fence", None, {}),
}


@pytest.mark.parametrize("close_mode", ["fin", "rst"])
@pytest.mark.parametrize("kind", sorted(REQUESTS))
def test_a_connection_closed_while_idle_is_replaced_before_anything_is_written(
    peer: LocalPeer, pool: _KeepAliveConnectionPool, close_mode: str, kind: str
) -> None:
    peer.close_mode = close_mode
    assert answered_by(pool, peer.url + "/close-when-idle") == 1
    peer.wait_closed()
    path, body, headers = REQUESTS[kind]
    request = (
        post(peer.url, path, body, headers)
        if body is not None
        else Request(peer.url + path, headers=headers)
    )

    assert answered_by(pool, request) == 2, "written to a fresh connection"

    delivered = peer.delivered(path)
    assert [item.connection for item in delivered] == [2], (
        f"the {kind} reached the peer exactly once, on the new connection"
    )
    stats = pool.stats()
    assert stats["peer_closed_total"] == 1
    assert stats["connections_opened_total"] == 2
    assert stats["stale_connection_retries_total"] == 0, "nothing had to be repeated"
    assert stats["stale_connection_failures_total"] == 0


@pytest.mark.parametrize(
    "identity", [COMMAND_ID_HEADER, REQUEST_ID_HEADER, IDEMPOTENCY_KEY_HEADER]
)
def test_a_post_with_a_retry_identity_is_repeated_once_when_the_close_wins_the_race(
    peer: LocalPeer,
    pool: _KeepAliveConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    """The FIN can land between the idle check and the write; the write then
    goes into a dead connection and only a retry identity licenses the repeat."""

    monkeypatch.setattr(http_client, "peer_closed", lambda connection: False)
    assert answered_by(pool, peer.url + "/close-when-idle") == 1
    peer.wait_closed()

    assert (
        answered_by(pool, post(peer.url, RESULT, RESULT_BODY, {identity: "id-1"})) == 2
    )

    delivered = peer.delivered(RESULT)
    assert [(item.connection, item.body) for item in delivered] == [(2, RESULT_BODY)]
    assert delivered[0].headers[identity.lower()] == "id-1", "the identity travels"
    stats = pool.stats()
    assert stats["stale_connection_retries_total"] == 1
    assert stats["stale_connection_failures_total"] == 0


def test_a_claim_that_meets_the_race_reaches_the_peer_at_most_once(
    peer: LocalPeer, pool: _KeepAliveConnectionPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``http.client`` writes the head and the body separately. When the reset
    the head provokes lands before the body write, the write fails and the
    claim goes to a fresh connection (loopback, usually); when it lands later,
    the claim was written and fails to its caller, which decides
    (``closed_without_response``). Never both, and never twice."""

    monkeypatch.setattr(http_client, "peer_closed", lambda connection: False)
    assert answered_by(pool, peer.url + "/close-when-idle") == 1
    peer.wait_closed()

    try:
        answered = answered_by(pool, post(peer.url, CLAIM, CLAIM_BODY, {}))
    except (URLError, OSError):
        answered = None

    stats = pool.stats()
    outcome = (
        stats["stale_connection_retries_total"],
        stats["stale_connection_failures_total"],
    )
    delivered = [item.connection for item in peer.delivered(CLAIM)]
    if answered is None:
        assert outcome == (0, 1) and delivered == [], "written, failed, not repeated"
    else:
        assert outcome == (1, 0) and delivered == [answered] == [2], (
            "the write failed, so the claim was sent once more and arrived once"
        )


def test_a_claim_whose_body_was_delivered_is_never_repeated(
    peer: LocalPeer, pool: _KeepAliveConnectionPool
) -> None:
    assert answered_by(pool, peer.url + "/warm") == 1
    peer.drop_next = {CLAIM, RESULT}

    with pytest.raises(URLError) as raised:
        answered_by(pool, post(peer.url, CLAIM, CLAIM_BODY, {}))
    assert isinstance(raised.value.reason, RemoteDisconnected), "closed, no status"
    assert [item.body for item in peer.delivered(CLAIM)] == [CLAIM_BODY], (
        "the claim was delivered once and must not be delivered again"
    )
    assert pool.stats()["stale_connection_failures_total"] == 1

    # The same close after a lease-fenced post: its identity licenses one repeat,
    # and the control plane answers the repeat idempotently.
    assert answered_by(pool, peer.url + "/warm") == 2
    command = {COMMAND_ID_HEADER: "remote-0123"}
    assert answered_by(pool, post(peer.url, RESULT, RESULT_BODY, command)) == 3
    delivered = peer.delivered(RESULT)
    assert [(item.connection, item.body) for item in delivered] == [
        (2, RESULT_BODY),
        (3, RESULT_BODY),
    ], "the identical result, repeated once on a fresh connection"
    assert pool.stats()["stale_connection_retries_total"] == 1


class FakeSocket:
    """No descriptor, so the idle check takes it as open."""

    def settimeout(self, timeout: float) -> None:
        pass


class WriteFailingWire:
    """Connections whose ``request`` raises on cue: a failed write."""

    def __init__(self) -> None:
        self.written: list[tuple[int, str, str]] = []
        self.write_failures: list[BaseException | None] = []
        self.opened = 0

    def connect(self, host: str, port: int, **options: Any) -> Any:
        wire = self
        wire.opened += 1
        number = wire.opened

        class Connection:
            sock: Any = None
            timeout = options.get("timeout")

            def request(self, method: str, path: str, **_kwargs: Any) -> None:
                failure = wire.write_failures.pop(0) if wire.write_failures else None
                if failure is not None:
                    raise failure
                wire.written.append((number, method, path))
                self.sock = FakeSocket()

            def getresponse(self) -> Reply:
                return Reply()

            def close(self) -> None:
                self.sock = None

        return Connection()


@pytest.mark.parametrize(
    "failure",
    [BrokenPipeError("unit pipe"), ConnectionResetError("unit reset")],
    ids=["broken-pipe", "reset"],
)
def test_a_request_whose_write_failed_on_a_reused_connection_is_sent_again(
    pool: _KeepAliveConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    wire = WriteFailingWire()
    monkeypatch.setattr(http_client, "HTTPConnection", wire.connect)
    with pool.send("http://unit.invalid/warm", timeout=2):
        pass
    wire.write_failures = [failure]

    with pool.send(post("http://unit.invalid", CLAIM, CLAIM_BODY, {}), timeout=2):
        pass

    assert wire.written == [(1, "GET", "/warm"), (2, "POST", CLAIM)], (
        "the claim reached a peer once: on the fresh connection"
    )
    assert pool.stats()["stale_connection_retries_total"] == 1


@pytest.mark.parametrize(
    ("failure", "raised"),
    [
        (ConnectionRefusedError("unit refused"), ConnectionRefusedError),
        (HTTPException("unit protocol"), URLError),
    ],
    ids=["refused", "protocol"],
)
def test_a_failed_write_on_a_fresh_connection_surfaces_unchanged(
    pool: _KeepAliveConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    raised: type[BaseException],
) -> None:
    wire = WriteFailingWire()
    monkeypatch.setattr(http_client, "HTTPConnection", wire.connect)
    wire.write_failures = [failure]

    with pytest.raises(raised) as caught:
        pool.send(post("http://unit.invalid", CLAIM, CLAIM_BODY, {}), timeout=2)

    if raised is ConnectionRefusedError:
        assert caught.value is failure, (
            "the caller sees the very error the write raised"
        )
    else:
        assert caught.value.reason is failure
    assert wire.opened == 1 and wire.written == [], "a fresh failure is not retried"
    assert pool.stats()["stale_connection_retries_total"] == 0


def test_an_open_idle_connection_is_not_mistaken_for_a_closed_one(
    peer: LocalPeer, pool: _KeepAliveConnectionPool
) -> None:
    assert [answered_by(pool, peer.url + "/warm") for _ in range(3)] == [1, 1, 1]
    stats = pool.stats()
    assert stats["peer_closed_total"] == 0 and stats["requests_reused_total"] == 2
