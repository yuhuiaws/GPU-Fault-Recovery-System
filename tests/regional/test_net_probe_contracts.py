"""Probe-side contracts of the NET-002/003 executor Pods.

The NET-003 proxy is exercised over real loopback TCP so the RST it sends is
the one the client sees; TLS is stood in for by hand-built record headers,
which is all the relay looks at.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import net002_executor, net003_executor

HANDSHAKE = 22
APPLICATION_DATA = 23


def _record(kind: int, body: bytes) -> bytes:
    return bytes([kind, 3, 3]) + len(body).to_bytes(2, "big") + body


# --------------------------------------------------------------------------- #
# TLS record scanner
# --------------------------------------------------------------------------- #
def test_scanner_counts_application_data_records_across_segments() -> None:
    scanner = net003_executor.TlsRecordScanner()
    scanner.feed(_record(HANDSHAKE, b"client hello"))
    assert scanner.application_records == 0

    request = _record(APPLICATION_DATA, b"x" * 40)
    scanner.feed(request[:3])
    scanner.feed(request[3:20])
    scanner.feed(request[20:])
    assert scanner.application_records == 1

    scanner.feed(_record(APPLICATION_DATA, b"y") + _record(APPLICATION_DATA, b"z"))
    assert scanner.application_records == 3


# --------------------------------------------------------------------------- #
# Response-losing relay
# --------------------------------------------------------------------------- #
def _tcp_pair() -> tuple[socket.socket, socket.socket]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    left = socket.create_connection(listener.getsockname())
    right, _ = listener.accept()
    listener.close()
    return left, right


def test_relay_forwards_the_request_withholds_the_response_and_resets() -> None:
    client, proxy_client_side = _tcp_pair()
    proxy_upstream_side, server = _tcp_pair()
    outcome: dict[str, Any] = {}

    def run() -> None:
        outcome.update(
            net003_executor.relay_losing_response(
                proxy_client_side,
                proxy_upstream_side,
                quiet_seconds=0.3,
                max_wait_seconds=3,
            )
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        client.settimeout(3)
        server.settimeout(3)
        client.sendall(_record(HANDSHAKE, b"client hello"))
        assert server.recv(4096) == _record(HANDSHAKE, b"client hello")
        server.sendall(_record(HANDSHAKE, b"server hello"))
        assert client.recv(4096) == _record(HANDSHAKE, b"server hello"), (
            "handshake bytes must still be forwarded to the client"
        )
        request = _record(APPLICATION_DATA, b"POST /result")
        client.sendall(request)
        assert server.recv(4096) == request, "the request must reach the upstream"
        response = _record(APPLICATION_DATA, b"HTTP/1.1 200 OK committed" * 8)
        server.sendall(response)
        thread.join(timeout=5)
        assert not thread.is_alive(), "the relay must exit once the reply is cut"

        with pytest.raises((ConnectionResetError, ConnectionAbortedError)):
            data = client.recv(4096)
            assert data == b"", f"response leaked to the client: {data!r}"
            raise ConnectionResetError("EOF instead of RST is still a lost response")
    finally:
        for item in (client, server, proxy_upstream_side):
            try:
                item.close()
            except OSError:
                pass

    assert outcome["connection_reset"] is True
    assert outcome["request_forwarded"] is True
    assert outcome["upstream_response_bytes"] == len(response)
    assert outcome["mode"] == "forward-then-reset"


def test_relay_without_a_request_forwards_everything_until_the_client_closes() -> None:
    client, proxy_client_side = _tcp_pair()
    proxy_upstream_side, server = _tcp_pair()
    outcome: dict[str, Any] = {}

    def run() -> None:
        outcome.update(
            net003_executor.relay_losing_response(
                proxy_client_side,
                proxy_upstream_side,
                quiet_seconds=0.2,
                max_wait_seconds=1,
            )
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        server.settimeout(3)
        client.sendall(_record(HANDSHAKE, b"hello"))
        assert server.recv(4096) == _record(HANDSHAKE, b"hello")
        client.close()
        thread.join(timeout=5)
    finally:
        server.close()
        proxy_upstream_side.close()

    assert outcome["request_forwarded"] is False
    assert outcome["upstream_response_bytes"] == 0
    assert outcome["client_closed_first"] is True


# --------------------------------------------------------------------------- #
# Interrupting client: one lost response, one replay
# --------------------------------------------------------------------------- #
def _client(
    tmp_path: Path, monkeypatch
) -> net003_executor.InterruptingRegionalExecutorClient:
    # The transport-level constructor needs a live control plane; the
    # interrupting subclass adds nothing the tests below need beyond its
    # own state, so only the base initialiser is stubbed.
    monkeypatch.setattr(
        net003_executor.RegionalExecutorClient,
        "__init__",
        lambda self, *_args, **_kwargs: None,
    )
    for name in (
        "DROP_NEXT",
        "RESULT_SUBMIT_STARTED",
        "RESULT_INTERRUPTED",
        "RESULT_REPLAYS",
    ):
        monkeypatch.setattr(net003_executor, name, tmp_path / name.lower())
    monkeypatch.setattr(net003_executor.time, "sleep", lambda _s: None)
    return net003_executor.InterruptingRegionalExecutorClient()


def _terminal_command() -> SimpleNamespace:
    return SimpleNamespace(
        command_id="remote-x",
        status=SimpleNamespace(value="SUCCEEDED"),
        status_source="net-test",
        updated_at=datetime(2026, 9, 7, 10, 0, 7, tzinfo=timezone.utc),
    )


def test_client_replays_once_after_a_lost_response(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    calls: list[str] = []

    def complete(self: Any, command: Any, result: Any) -> SimpleNamespace:
        calls.append(command.command_id)
        if len(calls) == 1:
            assert net003_executor.DROP_NEXT.exists(), "drop must be armed first"
            raise ConnectionResetError("[Errno 104] Connection reset by peer")
        return _terminal_command()

    monkeypatch.setattr(net003_executor.RegionalExecutorClient, "complete", complete)

    replay = client.complete(SimpleNamespace(command_id="remote-x"), result=None)

    assert calls == ["remote-x", "remote-x"]
    assert replay.status.value == "SUCCEEDED"
    interrupted = json.loads(net003_executor.RESULT_INTERRUPTED.read_text())
    assert interrupted["exception"] == "ConnectionResetError"
    assert interrupted["first_post_succeeded"] is False
    replays = json.loads(net003_executor.RESULT_REPLAYS.read_text())
    assert replays["count"] == 1
    assert replays["responses"][0]["updated_at"] == "2026-09-07T10:00:07+00:00"
    assert replays["replay_sent_at_epoch"] > interrupted["observed_at_epoch"] - 1


def test_a_control_plane_rejection_is_not_a_lost_response(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    def complete(self: Any, command: Any, result: Any) -> None:
        raise net003_executor.ClusterExecutorError("rejected request (409)")

    monkeypatch.setattr(net003_executor.RegionalExecutorClient, "complete", complete)

    with pytest.raises(net003_executor.ClusterExecutorError):
        client.complete(SimpleNamespace(command_id="remote-x"), result=None)

    assert not net003_executor.RESULT_REPLAYS.exists(), (
        "a 4xx/5xx answer was received; nothing may be replayed"
    )


def test_a_first_post_that_got_its_answer_is_recorded_not_replayed(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    calls: list[str] = []

    def complete(self: Any, command: Any, result: Any) -> SimpleNamespace:
        calls.append(command.command_id)
        return _terminal_command()

    monkeypatch.setattr(net003_executor.RegionalExecutorClient, "complete", complete)

    client.complete(SimpleNamespace(command_id="remote-x"), result=None)

    assert calls == ["remote-x"]
    interrupted = json.loads(net003_executor.RESULT_INTERRUPTED.read_text())
    assert interrupted["first_post_succeeded"] is True
    assert not net003_executor.RESULT_REPLAYS.exists(), (
        "a first post that got its answer must not be replayed"
    )


def test_later_completes_pass_straight_through(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        net003_executor.RegionalExecutorClient,
        "complete",
        lambda self, command, result: _terminal_command(),
    )
    # The first complete spends the one injected drop; wipe its state so a
    # second arming would be visible.
    client.complete(SimpleNamespace(command_id="remote-x"), result=None)
    for marker in (
        net003_executor.DROP_NEXT,
        net003_executor.RESULT_SUBMIT_STARTED,
        net003_executor.RESULT_INTERRUPTED,
    ):
        marker.unlink()

    client.complete(SimpleNamespace(command_id="remote-y"), result=None)

    assert not net003_executor.DROP_NEXT.exists(), (
        "only the first result post arms the drop"
    )
    assert not net003_executor.RESULT_SUBMIT_STARTED.exists(), (
        "a later complete must not be recorded as the interrupted submit"
    )
    assert not net003_executor.RESULT_INTERRUPTED.exists(), (
        "a later complete must not be recorded as interrupted"
    )


def test_net003_ready_contract_constants() -> None:
    assert net003_executor.TERMINAL_RESULT_REPLAYS == 1
    assert net003_executor.RESPONSE_LOSS_MODE == "forward-then-reset"
    assert net003_executor.LEASE_SECONDS == 60


# --------------------------------------------------------------------------- #
# NET-002 gated client
# --------------------------------------------------------------------------- #
def test_net002_gated_client_holds_the_result_until_the_block_lifts(
    tmp_path: Path, monkeypatch
) -> None:
    block = tmp_path / "block"
    monkeypatch.setattr(net002_executor, "BLOCK", block)
    monkeypatch.setattr(net002_executor, "RESULT_SUBMIT_WAITING", tmp_path / "waiting")
    monkeypatch.setattr(
        net002_executor, "RESULT_SUBMIT_RELEASED", tmp_path / "released"
    )
    posted: list[float] = []
    monkeypatch.setattr(
        net002_executor.RegionalExecutorClient,
        "complete",
        lambda self, command, result: posted.append(time.monotonic()) or "done",
    )
    client = net002_executor.GatedRegionalExecutorClient.__new__(
        net002_executor.GatedRegionalExecutorClient
    )
    block.touch()
    lifted_at: list[float] = []

    def lift() -> None:
        time.sleep(0.3)
        lifted_at.append(time.monotonic())
        block.unlink()

    threading.Thread(target=lift, daemon=True).start()

    assert client.complete(SimpleNamespace(command_id="remote-x"), None) == "done"
    assert posted and posted[0] >= lifted_at[0]
    assert (tmp_path / "waiting").exists() and (tmp_path / "released").exists()


def test_net002_lease_is_configurable_and_defaults_to_sixty() -> None:
    assert net002_executor.LEASE_SECONDS == 60
    assert net002_executor.BLOCK_ROLLBACK_SECONDS == 100
