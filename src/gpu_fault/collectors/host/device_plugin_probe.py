"""A bounded liveness probe of a kubelet device plugin's gRPC socket.

A device plugin is a gRPC server on a unix socket under kubelet's
``device-plugins`` directory (``k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1``,
``service DevicePlugin``). The one call with no side effect and no state is
``GetDevicePluginOptions(Empty) returns (DevicePluginOptions)``: kubelet itself
makes it once at registration (``pkg/kubelet/cm/devicemanager/manager.go``
``PluginConnected``), and the NVIDIA plugin answers it from a constant
(``internal/plugin/server.go`` ``GetDevicePluginOptions``). ``ListAndWatch`` is
deliberately *not* used: the NVIDIA and EFA plugins serve it from one shared
health channel (``case d := <-plugin.health``), so a second ``ListAndWatch``
stream would race kubelet's for every device-health event and could swallow
the one that marks a GPU unhealthy.

The product's node runtime has no gRPC library, so the call is spoken directly:
one HTTP/2 connection (RFC 9113 preface, SETTINGS, a HEADERS frame whose HPACK
block names the method, a DATA frame carrying the empty length-prefixed gRPC
message) and whatever the server says back. The verdict is about the *process*,
not the answer: any frame on the request's stream (response headers, a message,
trailers, even a reset) proves the plugin's gRPC server and handler goroutines
are scheduled and running; a connection the kernel accepted into the listen
backlog on which nothing arrives by the deadline is a plugin that is stopped or
deadlocked -- exactly the state kubelet never notices, because its own
``ListAndWatch`` stream to that plugin stays open without a deadline. A missing
socket or a refused connection is a plugin that is gone, which kubelet does
notice (the stream ends, the resource is marked unhealthy), so neither is a hang.

Standard library only: this module is part of the node runtime.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

#: ``/<package>.<Service>/<Method>`` of the device plugin API.
GRPC_METHOD_PATH = b"/v1beta1.DevicePlugin/GetDevicePluginOptions"
#: What kubelet dials the plugin with (``plugin/v1beta1/client.go`` ``dial``:
#: ``grpc.WithAuthority("localhost")``).
GRPC_AUTHORITY = b"localhost"
CLIENT_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
#: One empty protobuf message, gRPC length-prefixed: compressed flag 0, length 0.
EMPTY_GRPC_MESSAGE = b"\x00\x00\x00\x00\x00"
REQUEST_STREAM_ID = 1

FRAME_DATA = 0x0
FRAME_HEADERS = 0x1
FRAME_RST_STREAM = 0x3
FRAME_SETTINGS = 0x4
FRAME_PING = 0x6
FRAME_GOAWAY = 0x7
FLAG_ACK = 0x1
FLAG_END_STREAM = 0x1
FLAG_END_HEADERS = 0x4
FRAME_HEADER_SIZE = 9
#: A plugin's answer is a few dozen bytes; anything claiming more is not HTTP/2
#: as a gRPC server speaks it, and is not read.
MAX_FRAME_PAYLOAD = 1 << 16

# HPACK static table indices (RFC 7541 appendix A).
_INDEXED_METHOD_POST = 0x83
_INDEXED_SCHEME_HTTP = 0x86
_STATIC_AUTHORITY = 1
_STATIC_PATH = 4
_STATIC_CONTENT_TYPE = 31


class ProbeState(StrEnum):
    #: The stream was answered (headers, a message, trailers): the plugin runs.
    RESPONDED = "RESPONDED"
    #: The connection was accepted and the request sent, nothing came back on
    #: the stream by the deadline: the plugin is stopped or deadlocked.
    TIMEOUT = "TIMEOUT"
    #: The server reset the stream or the connection: alive, not serving.
    REFUSED = "REFUSED"
    #: No socket at the path: no plugin registered there.
    SOCKET_MISSING = "SOCKET_MISSING"
    #: A socket file nobody listens on (a plugin that exited without unlinking
    #: it, or a path that is not a socket).
    CONNECT_REFUSED = "CONNECT_REFUSED"
    #: Any other failure to connect or send.
    CONNECT_ERROR = "CONNECT_ERROR"


@dataclass(frozen=True)
class ProbeResult:
    state: ProbeState
    #: Wall time from connect to verdict, milliseconds.
    latency_ms: float
    detail: str
    #: The server sent at least one connection-level frame (SETTINGS, PING...).
    #: On a TIMEOUT this separates a stopped process (nothing at all) from a
    #: gRPC transport that runs while the handler never answers.
    transport_answered: bool = False

    @property
    def hung(self) -> bool:
        return self.state is ProbeState.TIMEOUT


def frame(frame_type: int, flags: int, stream_id: int, payload: bytes) -> bytes:
    """One HTTP/2 frame (RFC 9113 section 4.1)."""

    header = struct.pack(">I", len(payload))[1:] + bytes([frame_type, flags])
    return header + struct.pack(">I", stream_id & 0x7FFFFFFF) + payload


def _hpack_string(value: bytes) -> bytes:
    """A literal string without Huffman coding; the names here are all short."""

    if len(value) >= 127:
        raise ValueError("HPACK literal too long for a one-byte length")
    return bytes([len(value)]) + value


def _literal_indexed_name(index: int, value: bytes) -> bytes:
    """Literal header field without indexing, name from the static table."""

    head = bytes([index]) if index < 15 else bytes([0x0F, index - 15])
    return head + _hpack_string(value)


def _literal_new_name(name: bytes, value: bytes) -> bytes:
    return b"\x00" + _hpack_string(name) + _hpack_string(value)


def request_header_block(
    method_path: bytes = GRPC_METHOD_PATH, authority: bytes = GRPC_AUTHORITY
) -> bytes:
    """The HPACK block of a unary gRPC request (gRPC over HTTP/2 spec)."""

    return b"".join(
        (
            bytes([_INDEXED_METHOD_POST, _INDEXED_SCHEME_HTTP]),
            _literal_indexed_name(_STATIC_PATH, method_path),
            _literal_indexed_name(_STATIC_AUTHORITY, authority),
            _literal_indexed_name(_STATIC_CONTENT_TYPE, b"application/grpc"),
            _literal_new_name(b"te", b"trailers"),
        )
    )


def request_bytes(method_path: bytes = GRPC_METHOD_PATH) -> bytes:
    """Everything the probe sends: preface, SETTINGS, HEADERS, DATA(END_STREAM)."""

    return b"".join(
        (
            CLIENT_PREFACE,
            frame(FRAME_SETTINGS, 0, 0, b""),
            frame(
                FRAME_HEADERS,
                FLAG_END_HEADERS,
                REQUEST_STREAM_ID,
                request_header_block(method_path),
            ),
            frame(FRAME_DATA, FLAG_END_STREAM, REQUEST_STREAM_ID, EMPTY_GRPC_MESSAGE),
        )
    )


def _recv_exactly(
    sock: socket.socket,
    size: int,
    deadline: float | None,
    monotonic: Callable[[], float],
) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        if deadline is not None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("deadline reached")
            sock.settimeout(remaining)
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(
    sock: socket.socket,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[int, int, int, bytes]:
    """``(type, flags, stream id, payload)`` of the next frame.

    With a ``deadline`` (``monotonic`` seconds) every receive -- the header and
    the payload -- waits only for what is left of it, so a peer that sends a
    header and stalls cannot stretch the probe past its bound; without one the
    socket's own timeout applies. ``TimeoutError`` propagates either way.
    """

    header = _recv_exactly(sock, FRAME_HEADER_SIZE, deadline, monotonic)
    length = int.from_bytes(header[:3], "big")
    if length > MAX_FRAME_PAYLOAD:
        raise ValueError(f"frame of {length} bytes is not a device plugin answer")
    stream_id = struct.unpack(">I", header[5:9])[0] & 0x7FFFFFFF
    payload = _recv_exactly(sock, length, deadline, monotonic)
    return header[3], header[4], stream_id, payload


def probe_device_plugin(
    path: Path,
    timeout_seconds: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> ProbeResult:
    """One bounded ``GetDevicePluginOptions`` against the socket at ``path``.

    Never longer than ``timeout_seconds`` plus the cost of a few syscalls: the
    deadline is set once and every receive waits for what is left of it.
    """

    if timeout_seconds <= 0:
        raise ValueError("device plugin probe timeout must be positive")
    started = monotonic()

    def result(state: ProbeState, detail: str, answered: bool = False) -> ProbeResult:
        elapsed_ms = max(0.0, (monotonic() - started) * 1000.0)
        return ProbeResult(state, round(elapsed_ms, 1), detail, answered)

    if not path.exists():
        return result(ProbeState.SOCKET_MISSING, f"{path} does not exist")
    deadline = started + timeout_seconds
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    transport_answered = False
    try:
        sock.settimeout(timeout_seconds)
        try:
            sock.connect(str(path))
            sock.sendall(request_bytes())
        except FileNotFoundError:
            return result(ProbeState.SOCKET_MISSING, f"{path} vanished")
        except ConnectionRefusedError as exc:
            return result(ProbeState.CONNECT_REFUSED, f"connect: {exc.strerror}")
        except TimeoutError:
            # A full listen backlog on a stopped process: nobody accepts.
            return result(ProbeState.TIMEOUT, "connect did not complete")
        except (BrokenPipeError, ConnectionResetError) as exc:
            return result(ProbeState.REFUSED, f"send: {exc.strerror}")
        except OSError as exc:
            return result(ProbeState.CONNECT_ERROR, f"connect: {exc.strerror or exc}")
        while True:
            if deadline - monotonic() <= 0:
                break
            try:
                frame_type, flags, stream_id, payload = read_frame(
                    sock, deadline, monotonic
                )
            except TimeoutError:
                break
            except (ConnectionError, OSError) as exc:
                return result(
                    ProbeState.REFUSED,
                    f"connection closed before the stream was answered: {exc}",
                    transport_answered,
                )
            except ValueError as exc:
                return result(ProbeState.REFUSED, str(exc), transport_answered)
            if frame_type in (FRAME_SETTINGS, FRAME_PING):
                transport_answered = True
                if not flags & FLAG_ACK:
                    # Acknowledge as a well-behaved peer; a server that has
                    # already answered and closed (EPIPE here) is judged by
                    # what the next read returns, not by this write.
                    try:
                        sock.sendall(
                            frame(
                                frame_type,
                                FLAG_ACK,
                                0,
                                payload if frame_type == FRAME_PING else b"",
                            )
                        )
                    except OSError:
                        pass
                continue
            if frame_type == FRAME_GOAWAY:
                code = int.from_bytes(payload[4:8], "big") if len(payload) >= 8 else -1
                return result(ProbeState.REFUSED, f"GOAWAY error code {code}", True)
            if stream_id != REQUEST_STREAM_ID:
                transport_answered = True
                continue
            if frame_type == FRAME_RST_STREAM:
                code = int.from_bytes(payload[:4], "big") if len(payload) >= 4 else -1
                return result(ProbeState.REFUSED, f"RST_STREAM error code {code}", True)
            if frame_type == FRAME_HEADERS:
                what = "trailers" if flags & FLAG_END_STREAM else "response headers"
                return result(ProbeState.RESPONDED, what, True)
            if frame_type == FRAME_DATA:
                return result(ProbeState.RESPONDED, "response message", True)
            transport_answered = True
        return result(
            ProbeState.TIMEOUT,
            (
                "the gRPC transport answered but the call did not"
                if transport_answered
                else "no bytes from the plugin"
            )
            + f" within {timeout_seconds:g}s",
            transport_answered,
        )
    finally:
        sock.close()
