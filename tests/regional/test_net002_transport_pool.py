"""Exercise the real per-thread pool against a local HTTP/1.1 peer."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.transport.http_client import CONNECTION_POOL, urlopen
from scripts.e2e.regional.probes import net002_executor as probe


@pytest.fixture
def local_peer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    closed_idle = Event()
    lock = Lock()
    connections: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            with lock:
                self.connection_id = len(connections) + 1
                connections.append(self.connection_id)

        def log_message(self, *args: Any) -> None:
            pass

        def respond(self, status: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()

        def do_GET(self) -> None:
            self.respond(200, {"connection_id": self.connection_id})
            if self.path == "/stale":
                # Close without a Connection: close response, like an idle expiry.
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                closed_idle.set()

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.respond(409, {"detail": probe.STALE_LEASE_DETAILS[0]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    serving.start()
    for name, filename in (
        ("BLOCK", "block"),
        ("RESULT_SUBMIT_WAITING", "waiting.json"),
        ("RESULT_SUBMIT_RELEASED", "released.json"),
    ):
        monkeypatch.setattr(probe, name, tmp_path / filename)
    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(time=time.time, sleep=lambda _: probe.BLOCK.unlink()),
    )
    close_pool = CONNECTION_POOL.close
    close_pool()
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{server.server_port}",
            closed_idle=closed_idle,
            receipt=tmp_path / "first-result-submission.json",
        )
    finally:
        close_pool()
        server.shutdown()
        serving.join(2)
        server.server_close()
        assert not serving.is_alive(), "the local HTTP peer must stop"


def read_connection(url: str) -> int:
    with urlopen(url, timeout=2) as response:
        return int(json.loads(response.read())["connection_id"])


def result() -> RemoteCommandResult:
    return RemoteCommandResult(
        lease_token="example-only", status=RemoteCommandStatus.SUCCEEDED
    )


@pytest.mark.parametrize("gated", [False, True], ids=["retained-stale", "gated-fresh"])
def test_stale_keepalive_requires_caller_reset_to_observe_result_409(
    local_peer: Any, gated: bool
) -> None:
    read_connection(local_peer.url + "/stale")
    assert local_peer.closed_idle.wait(1), (
        "test_stale_keepalive_requires_caller_reset_to_observe_result_409: expected local_peer.closed_idle.wait(1)"
    )
    client = probe.GatedRegionalExecutorClient(
        local_peer.url, "net002-synthetic", "example-only", timeout_seconds=2
    )
    if gated:
        probe.BLOCK.touch()
    with pytest.raises(probe.ClusterExecutorError) as raised:
        client.complete(SimpleNamespace(command_id="remote-net002"), result())
    expected_status = 409 if gated else None
    assert raised.value.status_code == expected_status
    receipt = json.loads(local_peer.receipt.read_text())
    assert receipt["status_code"] == expected_status
    assert receipt["caller_transport_pool_closed"] is gated
    assert receipt["stale_lease_reason"] == (
        probe.STALE_LEASE_DETAILS[0] if gated else None
    )


def test_result_gate_closes_only_the_callers_pool_not_another_threads(
    local_peer: Any,
) -> None:
    other_thread_connection = read_connection(local_peer.url + "/session")
    worker_connections: list[int] = []
    failures: list[Exception] = []

    def submit() -> None:
        try:
            worker_connections.append(read_connection(local_peer.url + "/session"))
            probe.BLOCK.touch()
            client = probe.GatedRegionalExecutorClient(
                local_peer.url, "net002-synthetic", "example-only", timeout_seconds=2
            )
            with pytest.raises(probe.ClusterExecutorError) as raised:
                client.complete(SimpleNamespace(command_id="remote-net002"), result())
            assert raised.value.status_code == 409
            worker_connections.append(read_connection(local_peer.url + "/session"))
        except Exception as exc:
            failures.append(exc)
        finally:
            CONNECTION_POOL.close()

    worker = Thread(target=submit)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "the probe caller thread must finish"
    assert not failures, (
        "test_result_gate_closes_only_the_callers_pool_not_another_threads: expected no failures"
    )
    assert len(worker_connections) == 2
    assert worker_connections[0] != worker_connections[1]
    assert read_connection(local_peer.url + "/session") == other_thread_connection
