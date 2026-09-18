"""Local socket-pair regressions for the NET002 opaque TCP gate."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import net002_executor as probe
from tests.regional._cov95_collect_net import FakeSocket, StopLoop, local_socket_module


@pytest.fixture
def tunnel_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    block = tmp_path / "block"
    monkeypatch.setattr(probe, "BLOCK", block)

    @contextmanager
    def tunnel(*, blocked: bool = False) -> Iterator[Any]:
        caller, client = socket.socketpair()
        upstream, server = socket.socketpair()
        caller.settimeout(1)
        server.settimeout(1)
        threads: list[Thread] = []
        state = SimpleNamespace(
            caller=caller,
            server=server,
            block=block,
            arm_on_recv=None,
            buffered=Event(),
            connections=[],
        )

        class Stream:
            def __init__(self, stream: socket.socket, direction: str) -> None:
                self.stream = stream
                self.direction = direction

            def __getattr__(self, name: str) -> Any:
                return getattr(self.stream, name)

            def __enter__(self) -> Stream:
                return self

            def __exit__(self, *args: Any) -> None:
                self.stream.close()

            def recv(self, size: int) -> bytes:
                data = self.stream.recv(size)
                if data and state.arm_on_recv == self.direction:
                    state.arm_on_recv = None
                    block.touch()
                    state.buffered.set()
                return data

        listener = FakeSocket()
        listener.accepted.append(Stream(client, "request"))

        def connect(address: Any, *, timeout: float) -> Stream:
            state.connections.append((address, timeout))
            return Stream(upstream, "response")

        def thread(**kwargs: Any) -> Thread:
            worker = Thread(**kwargs)
            threads.append(worker)
            return worker

        monkeypatch.setattr(
            probe,
            "socket",
            local_socket_module(socket=lambda *_: listener, create_connection=connect),
        )
        monkeypatch.setattr(probe, "Thread", thread)
        if blocked:
            block.touch()
        try:
            with pytest.raises(StopLoop):
                probe.GateProxy("192.0.2.1", 18443).run()
            yield state
        finally:
            block.unlink(missing_ok=True)
            caller.shutdown(socket.SHUT_WR)
            server.shutdown(socket.SHUT_WR)
            for worker in threads:
                worker.join(2)
            for stream in (caller, client, upstream, server):
                stream.close()
            assert all(not worker.is_alive() for worker in threads), (
                "all local proxy handlers must stop"
            )

    return tunnel


def assert_held(stream: socket.socket) -> None:
    stream.settimeout(0.15)
    try:
        with pytest.raises(TimeoutError):
            stream.recv(65536)
    finally:
        stream.settimeout(1)


def test_established_connection_pauses_both_directions_and_resumes(
    tunnel_factory: Any,
) -> None:
    with tunnel_factory() as tunnel:
        tunnel.caller.sendall(b"established")
        assert tunnel.server.recv(65536) == b"established"
        tunnel.server.sendall(b"ready")
        assert tunnel.caller.recv(65536) == b"ready"

        for _ in range(2):
            tunnel.block.touch()
            tunnel.caller.sendall(b"held-request")
            tunnel.server.sendall(b"held-response")
            assert_held(tunnel.server)
            assert_held(tunnel.caller)
            tunnel.block.unlink()
            assert tunnel.server.recv(65536) == b"held-request"
            assert tunnel.caller.recv(65536) == b"held-response"
        assert len(tunnel.connections) == 1, "unblock must reuse this same tunnel"


@pytest.mark.parametrize("direction", ["request", "response"])
def test_bytes_already_read_are_not_forwarded_after_block_is_armed(
    tunnel_factory: Any, direction: str
) -> None:
    with tunnel_factory() as tunnel:
        tunnel.caller.sendall(b"established")
        assert tunnel.server.recv(65536) == b"established"
        tunnel.arm_on_recv = direction
        sender, receiver = (
            (tunnel.caller, tunnel.server)
            if direction == "request"
            else (tunnel.server, tunnel.caller)
        )
        sender.sendall(b"already-buffered")
        assert tunnel.buffered.wait(1), "the relay must have read the test bytes"
        assert tunnel.block.exists(), (
            "test_bytes_already_read_are_not_forwarded_after_block_is_armed: expected tunnel.block.exists()"
        )
        assert_held(receiver)
        tunnel.block.unlink()
        assert receiver.recv(65536) == b"already-buffered"


def test_new_connection_waits_for_unblock_without_forwarding(
    tunnel_factory: Any,
) -> None:
    with tunnel_factory(blocked=True) as tunnel:
        tunnel.caller.sendall(b"held-before-connect")
        assert_held(tunnel.server)
        assert tunnel.connections == []
        tunnel.block.unlink()
        assert tunnel.server.recv(65536) == b"held-before-connect"
        assert len(tunnel.connections) == 1
