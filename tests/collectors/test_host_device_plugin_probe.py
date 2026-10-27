"""The bounded ``GetDevicePluginOptions`` probe against a device plugin socket.

The fake on the other end is an HTTP/2 server on a unix socket that behaves the
way grpc-go's transport does on the wire (its SETTINGS frame goes out before it
reads the client preface) and then follows one script: answer the call, answer
only at the transport level, reset the stream, go away, close, or -- the case
the probe exists for -- hold the accepted connection and say nothing, which is
what a SIGSTOPped or deadlocked plugin looks like to kubelet and to us.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpu_fault.collectors.host.device_plugin_probe import (
    CLIENT_PREFACE,
    EMPTY_GRPC_MESSAGE,
    FLAG_END_HEADERS,
    FLAG_END_STREAM,
    FRAME_DATA,
    FRAME_GOAWAY,
    FRAME_HEADERS,
    FRAME_RST_STREAM,
    FRAME_SETTINGS,
    GRPC_METHOD_PATH,
    REQUEST_STREAM_ID,
    ProbeState,
    frame,
    probe_device_plugin,
    read_frame,
    request_header_block,
)

TIMEOUT = 0.4


class FakePluginServer:
    """A device plugin's gRPC endpoint, as far as the wire can tell."""

    def __init__(self, path: Path, behaviour: str) -> None:
        self.path = path
        self.behaviour = behaviour
        self.requests: list[tuple[bytes, list[tuple[int, int, int, bytes]]]] = []
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(4)
        self._listener.settimeout(0.1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            with conn:
                self._handle(conn)

    def _read_request(self, conn: socket.socket) -> None:
        preface = b""
        while len(preface) < len(CLIENT_PREFACE):
            chunk = conn.recv(len(CLIENT_PREFACE) - len(preface))
            if not chunk:
                break
            preface += chunk
        frames = []
        while True:
            frames.append(read_frame(conn))
            frame_type, flags, stream_id, _ = frames[-1]
            if stream_id == REQUEST_STREAM_ID and flags & FLAG_END_STREAM:
                break
        self.requests.append((preface, frames))

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(5)
        if self.behaviour == "silent":
            self._stop.wait(5)
            return
        # grpc-go writes its SETTINGS before reading the client preface.
        conn.sendall(frame(FRAME_SETTINGS, 0, 0, b""))
        if self.behaviour == "transport-only":
            self._stop.wait(5)
            return
        self._read_request(conn)
        conn.sendall(frame(FRAME_SETTINGS, 0x1, 0, b""))
        if self.behaviour == "answers":
            conn.sendall(
                frame(
                    FRAME_HEADERS,
                    FLAG_END_HEADERS,
                    REQUEST_STREAM_ID,
                    b"\x88\x0f\x10\x10application/grpc",
                )
            )
            # DevicePluginOptions{GetPreferredAllocationAvailable: true}.
            conn.sendall(
                frame(FRAME_DATA, 0, REQUEST_STREAM_ID, b"\x00\x00\x00\x00\x02\x10\x01")
            )
            conn.sendall(
                frame(
                    FRAME_HEADERS,
                    FLAG_END_HEADERS | FLAG_END_STREAM,
                    REQUEST_STREAM_ID,
                    b"\x00\x0bgrpc-status\x010",
                )
            )
        elif self.behaviour == "trailers-only":
            conn.sendall(
                frame(
                    FRAME_HEADERS,
                    FLAG_END_HEADERS | FLAG_END_STREAM,
                    REQUEST_STREAM_ID,
                    b"\x88\x00\x0bgrpc-status\x0212",
                )
            )
        elif self.behaviour == "resets":
            conn.sendall(
                frame(FRAME_RST_STREAM, 0, REQUEST_STREAM_ID, (7).to_bytes(4, "big"))
            )
        elif self.behaviour == "goaway":
            conn.sendall(
                frame(
                    FRAME_GOAWAY, 0, 0, (0).to_bytes(4, "big") + (11).to_bytes(4, "big")
                )
            )
        # "closes": the connection ends without a word on the stream.

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._listener.close()


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    # A unix socket path is limited to 108 bytes (``sun_path``); pytest's
    # tmp_path carries the test name and overflows it, so a short private
    # directory is used instead.
    path = Path(tempfile.mkdtemp(prefix="gfrs-dp-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _probe(path: Path, timeout: float = TIMEOUT):
    started = time.monotonic()
    result = probe_device_plugin(path, timeout)
    return result, time.monotonic() - started


def test_a_running_plugin_answers_and_the_request_is_a_grpc_call(
    socket_dir: Path,
) -> None:
    server = FakePluginServer(socket_dir / "nvidia-gpu.sock", "answers")
    try:
        result, _ = _probe(server.path)
    finally:
        server.close()

    assert result.state is ProbeState.RESPONDED, result
    assert result.transport_answered is True
    assert result.detail == "response headers"
    preface, frames = server.requests[0]
    assert preface == CLIENT_PREFACE
    assert frames[0][:3] == (FRAME_SETTINGS, 0, 0), frames[0]
    headers = [item for item in frames if item[0] == FRAME_HEADERS]
    assert len(headers) == 1 and headers[0][1] == FLAG_END_HEADERS, headers
    assert headers[0][2] == REQUEST_STREAM_ID
    assert GRPC_METHOD_PATH in headers[0][3]
    assert frames[-1] == (
        FRAME_DATA,
        FLAG_END_STREAM,
        REQUEST_STREAM_ID,
        EMPTY_GRPC_MESSAGE,
    ), "the call carries one empty length-prefixed message and ends the stream"


def test_a_stopped_plugin_is_a_timeout_within_the_bound(socket_dir: Path) -> None:
    """The kernel accepts the connection into the listen backlog of a stopped
    process, so connect succeeds and nothing ever arrives: the hang kubelet
    never notices."""

    server = FakePluginServer(socket_dir / "nvidia-gpu.sock", "silent")
    try:
        result, elapsed = _probe(server.path)
    finally:
        server.close()

    assert result.state is ProbeState.TIMEOUT, result
    assert result.hung is True
    assert result.transport_answered is False
    assert result.detail.startswith("no bytes from the plugin"), result.detail
    assert TIMEOUT <= elapsed < TIMEOUT + 1.0, elapsed


def test_a_live_transport_whose_handler_never_answers_is_still_a_timeout(
    socket_dir: Path,
) -> None:
    server = FakePluginServer(socket_dir / "nvidia-gpu.sock", "transport-only")
    try:
        result, elapsed = _probe(server.path)
    finally:
        server.close()

    assert result.state is ProbeState.TIMEOUT, result
    assert result.transport_answered is True
    assert result.detail.startswith(
        "the gRPC transport answered but the call did not"
    ), result.detail
    assert elapsed < TIMEOUT + 1.0, elapsed


@pytest.mark.parametrize(
    ("behaviour", "state", "detail"),
    [
        ("trailers-only", ProbeState.RESPONDED, "trailers"),
        ("resets", ProbeState.REFUSED, "RST_STREAM error code 7"),
        ("goaway", ProbeState.REFUSED, "GOAWAY error code 11"),
    ],
)
def test_any_answer_on_the_stream_proves_the_plugin_runs(
    socket_dir: Path, behaviour: str, state: ProbeState, detail: str
) -> None:
    server = FakePluginServer(socket_dir / "nvidia-gpu.sock", behaviour)
    try:
        result, elapsed = _probe(server.path)
    finally:
        server.close()

    assert (result.state, result.detail) == (state, detail), result
    assert result.hung is False
    assert elapsed < TIMEOUT, "an answer must not wait out the deadline"


def test_a_connection_closed_without_an_answer_is_refused_not_hung(
    socket_dir: Path,
) -> None:
    server = FakePluginServer(socket_dir / "nvidia-gpu.sock", "closes")
    try:
        result, _ = _probe(server.path)
    finally:
        server.close()

    assert result.state is ProbeState.REFUSED, result
    assert "connection closed" in result.detail


def test_a_missing_socket_is_not_a_hang(socket_dir: Path) -> None:
    result, elapsed = _probe(socket_dir / "nvidia-gpu.sock")

    assert result.state is ProbeState.SOCKET_MISSING, result
    assert result.hung is False
    assert elapsed < TIMEOUT


def test_a_socket_nobody_listens_on_is_refused(socket_dir: Path) -> None:
    """A plugin that exited without unlinking its socket: kubelet saw the
    stream end and marked the resource unhealthy, so the allocatable rule owns
    this, not the liveness probe."""

    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_dir / "nvidia-gpu.sock"))
    stale.close()
    regular = socket_dir / "aws-efa-device-plugin.sock"
    regular.write_text("not a socket")

    for path in (socket_dir / "nvidia-gpu.sock", regular):
        result, _ = _probe(path)
        assert result.state is ProbeState.CONNECT_REFUSED, (path, result)
        assert result.hung is False


def test_the_header_block_is_the_grpc_request_kubelet_would_send() -> None:
    """``:method POST``, ``:scheme http``, the method path, kubelet's
    ``localhost`` authority, ``content-type application/grpc`` and
    ``te trailers``, HPACK-coded as literals without indexing."""

    expected = (
        b"\x83\x86"
        + b"\x04"
        + bytes([len(GRPC_METHOD_PATH)])
        + GRPC_METHOD_PATH
        + b"\x01\x09localhost"
        + b"\x0f\x10\x10application/grpc"
        + b"\x00\x02te\x08trailers"
    )
    assert request_header_block() == expected


def test_the_probe_refuses_a_non_positive_timeout(socket_dir: Path) -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        probe_device_plugin(socket_dir / "nvidia-gpu.sock", 0)
