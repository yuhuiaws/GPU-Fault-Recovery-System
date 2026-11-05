"""``gpu_fault.ras_observation``: NCCL RAS answers it refuses to read.

``tests/node_agent/test_communication_monitor.py`` covers a clean RAS session
and a few defects of the JSON document. These tests pin the rest of the
fail-closed reader: the loopback-only endpoint, every handshake and response
deadline, the byte budget, and each inventory field whose absence would
otherwise read as a healthy communicator. The RAS peer is a real loopback
socket served by a thread; the deadlines run on a test clock the peer moves.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from gpu_fault import ras_observation
from gpu_fault.ras_observation import MAX_RAS_BYTES, parse_ras, query_ras
from tests.node_agent.test_communication_monitor import ras_document

HANDSHAKE = [
    (b"CLIENT PROTOCOL 2\n", b"SERVER PROTOCOL 2\n"),
    (b"TIMEOUT 2\n", b"OK\n"),
    (b"SET FORMAT json\n", b"OK\n"),
]


class RasClock:
    """``time.monotonic`` for the reader; the peer moves it past the deadline."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class RasPeer:
    """A loopback RAS endpoint that answers from a script, on its own thread.

    ``script(stream, client, clock)`` reads requests and writes replies; any
    exception it raises is kept for the test to inspect.
    """

    def __init__(
        self, script: Callable[[Any, socket.socket, RasClock], None], clock: RasClock
    ) -> None:
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.port = int(self.listener.getsockname()[1])
        self.errors: list[BaseException] = []
        self.thread = threading.Thread(
            target=self._serve, args=(script, clock), daemon=True
        )
        self.thread.start()

    def _serve(
        self, script: Callable[[Any, socket.socket, RasClock], None], clock: RasClock
    ) -> None:
        try:
            client = self.listener.accept()[0]
            with client, client.makefile("rb") as stream:
                script(stream, client, clock)
        except Exception as exc:  # noqa: BLE001 - reported to the test
            self.errors.append(exc)

    def close(self) -> None:
        self.listener.close()
        self.thread.join(timeout=10)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> RasClock:
    test_clock = RasClock()
    monkeypatch.setattr(ras_observation, "time", test_clock)
    return test_clock


@pytest.fixture
def peer(clock: RasClock) -> Iterator[Callable[..., RasPeer]]:
    peers: list[RasPeer] = []

    def start(script: Callable[[Any, socket.socket, RasClock], None]) -> RasPeer:
        started = RasPeer(script, clock)
        peers.append(started)
        return started

    yield start
    for started in peers:
        started.close()
        assert not started.thread.is_alive(), "the RAS peer thread must finish"


def _handshake(stream: Any, client: socket.socket) -> None:
    for request, reply in HANDSHAKE:
        assert stream.readline() == request
        client.sendall(reply)


@pytest.mark.parametrize(
    ("host", "port"),
    [("10.0.0.7", 28028), ("0.0.0.0", 28028), ("127.0.0.1", 0), ("::1", 65536)],
)
def test_only_a_namespace_local_loopback_endpoint_is_queried(
    host: str, port: int
) -> None:
    with pytest.raises(ValueError, match="namespace-local loopback endpoint"):
        query_ras(host, port)


def test_a_peer_that_does_not_speak_protocol_two_is_refused(peer) -> None:
    def script(stream: Any, client: socket.socket, clock: RasClock) -> None:
        assert stream.readline() == HANDSHAKE[0][0]
        client.sendall(b"SERVER PROTOCOL 1\n")

    started = peer(script)
    with pytest.raises(ValueError, match="unsupported RAS protocol"):
        query_ras("127.0.0.1", started.port)
    started.close()
    assert started.errors == []


def test_a_peer_that_hangs_up_mid_handshake_is_refused(peer) -> None:
    def script(stream: Any, client: socket.socket, clock: RasClock) -> None:
        assert stream.readline() == HANDSHAKE[0][0]
        client.sendall(b"SERVER")

    started = peer(script)
    with pytest.raises(ValueError, match="unsupported RAS protocol"):
        query_ras("127.0.0.1", started.port)
    started.close()
    assert started.errors == []


def test_a_handshake_reply_that_outlives_the_deadline_times_out(peer) -> None:
    def script(stream: Any, client: socket.socket, clock: RasClock) -> None:
        assert stream.readline() == HANDSHAKE[0][0]
        clock.now = 100.0
        client.sendall(HANDSHAKE[0][1])

    started = peer(script)
    with pytest.raises(TimeoutError, match="RAS handshake deadline"):
        query_ras("127.0.0.1", started.port, timeout=4)
    started.close()
    assert started.errors == []


def test_a_status_response_that_outlives_the_deadline_times_out(peer) -> None:
    def script(stream: Any, client: socket.socket, clock: RasClock) -> None:
        _handshake(stream, client)
        assert stream.readline() == b"VERBOSE STATUS\n"
        clock.now = 100.0
        client.sendall(b'{"nccl_version": ')

    started = peer(script)
    with pytest.raises(TimeoutError, match="RAS response deadline"):
        query_ras("127.0.0.1", started.port, timeout=4)
    started.close()
    assert started.errors == []


def test_a_status_response_over_the_byte_budget_is_refused(peer) -> None:
    def script(stream: Any, client: socket.socket, clock: RasClock) -> None:
        _handshake(stream, client)
        assert stream.readline() == b"VERBOSE STATUS\n"
        try:
            client.sendall(b" " * (MAX_RAS_BYTES + 1))
        except (BrokenPipeError, ConnectionResetError):
            pass  # the reader may hang up as soon as it has seen one byte too many

    started = peer(script)
    with pytest.raises(ValueError, match="exceeds byte budget"):
        query_ras("127.0.0.1", started.port)
    started.close()
    assert started.errors == []


def test_a_document_over_the_byte_budget_is_not_parsed() -> None:
    with pytest.raises(ValueError, match="exceeds byte budget"):
        parse_ras(b" " * (MAX_RAS_BYTES + 1))


def _member(document: dict[str, Any], index: int = 0) -> dict[str, Any]:
    member: dict[str, Any] = document["communicators"][0]["ranks"][index]
    return member


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("communicator-not-a-mapping", "invalid RAS communicator"),
        ("empty-hash", "ambiguous RAS communicator identity"),
        ("duplicate-communicator", "ambiguous RAS communicator identity"),
        ("size-not-int", "incomplete RAS inventory accounting"),
        ("ranks-count-wrong", "incomplete RAS inventory accounting"),
        ("missing-ranks-not-a-list", "incomplete RAS inventory accounting"),
        ("members-do-not-add-up", "incomplete RAS inventory accounting"),
        ("member-not-a-mapping", "invalid RAS member"),
        ("rank-outside-size", "ambiguous RAS local rank"),
        ("host-missing", "RAS member identity unavailable"),
        ("pid-not-int", "RAS member identity unavailable"),
        ("no-collective-counts", "RAS counters/status unavailable"),
        ("status-missing", "RAS counters/status unavailable"),
        ("ras-metadata-missing", "RAS timeout coverage unavailable"),
        ("timeouts-count-negative", "RAS timeout coverage unavailable"),
        ("timeouts-count-not-int", "RAS timeout coverage unavailable"),
    ],
)
def test_an_inventory_field_that_cannot_be_trusted_rejects_the_document(
    defect: str, message: str
) -> None:
    document = ras_document()
    group = document["communicators"][0]
    if defect == "communicator-not-a-mapping":
        document["communicators"] = ["0xab"]
    elif defect == "empty-hash":
        group["hash"] = ""
    elif defect == "duplicate-communicator":
        document["communicators"].append(json.loads(json.dumps(group)))
        document["communicators_count"] = 2
    elif defect == "size-not-int":
        group["size"] = "3"
    elif defect == "ranks-count-wrong":
        group["ranks_count"] = 2
    elif defect == "missing-ranks-not-a-list":
        group["missing_ranks"] = None
    elif defect == "members-do-not-add-up":
        group["size"] = 4
    elif defect == "member-not-a-mapping":
        group["ranks"][2] = 2
    elif defect == "rank-outside-size":
        _member(document, 2)["rank"] = 3
    elif defect == "host-missing":
        del _member(document)["host"]
    elif defect == "pid-not-int":
        _member(document)["pid"] = "100"
    elif defect == "no-collective-counts":
        _member(document)["collective_counts"] = {}
    elif defect == "status-missing":
        del _member(document)["status"]
    elif defect == "ras-metadata-missing":
        del document["ras"]
    elif defect == "timeouts-count-negative":
        document["ras"]["timeouts_count"] = -1
    else:
        document["ras"]["timeouts_count"] = 0.0
    with pytest.raises(ValueError, match=message):
        parse_ras(json.dumps(document).encode())


def test_a_communicator_with_missing_members_reads_incomplete() -> None:
    document = ras_document()
    group = document["communicators"][0]
    lost = group["ranks"].pop()
    group["missing_ranks"] = [{"rank": lost["rank"], "host": "unknown", "pid": 0}]
    group.update(ranks_count=2, missing_ranks_count=1)
    summary = parse_ras(json.dumps(document).encode())
    (parsed,) = summary["groups"]
    assert parsed["complete"] is False, "a missing member is never a complete group"
    assert [row["rank"] for row in parsed["missing_members"]] == [2]
    assert summary["completion_progress_available"] is False
