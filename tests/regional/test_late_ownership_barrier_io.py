from __future__ import annotations

import os
import socket
import struct
import time

import pytest

from scripts.e2e.regional import late_ownership_barrier as barrier
from scripts.e2e.regional.late_ownership_contract import Document, StopReceipt
from tests.regional._late_ownership_support import evidence, mutation_for


class SocketFault:
    def __init__(self, endpoint, defect):
        self.endpoint = endpoint
        self.defect = defect
        self.family = endpoint.family
        self.type = endpoint.type

    def fileno(self):
        return self.endpoint.fileno()

    def setsockopt(self, *arguments):
        return self.endpoint.setsockopt(*arguments)

    def setblocking(self, value):
        self.endpoint.setblocking(value)

    def send(self, payload):
        if self.defect == "short":
            return len(payload) - 1
        raise OSError("owned socket failed")

    def recvmsg(self, *arguments):
        data, ancillary, flags, address = self.endpoint.recvmsg(*arguments)
        level, kind, credentials = ancillary[0]
        pid, uid, gid = struct.unpack("3i", credentials)
        if self.defect == "none":
            ancillary = []
        elif self.defect == "duplicate":
            ancillary *= 2
        elif self.defect == "level":
            ancillary = [(level + 1, kind, credentials)]
        elif self.defect == "kind":
            ancillary = [(level, kind + 1, credentials)]
        elif self.defect == "uid":
            ancillary = [(level, kind, struct.pack("3i", pid, uid + 1, gid))]
        elif self.defect == "credentials":
            ancillary = [(level, kind, b"x")]
        elif self.defect == "os-error":
            raise OSError("owned receive failed")
        return data, ancillary, flags, address

    def close(self):
        self.endpoint.close()


@pytest.mark.parametrize(
    "defect,message",
    [
        ("none", "unique kernel identity"),
        ("duplicate", "unique kernel identity"),
        ("level", "kernel credentials"),
        ("kind", "kernel credentials"),
        ("uid", "another process"),
        ("credentials", "invalid"),
        ("os-error", "invalid"),
    ],
)
def test_corrupt_or_missing_kernel_credential_frame_is_not_a_callback(defect, message):
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel = barrier.PeerChannel(
        SocketFault(first, defect),
        barrier.process_identity(os.getpid()),
        expires=time.monotonic() + 10,
    )
    try:
        second.send(evidence().stop.model_dump_json().encode())
        with pytest.raises(barrier.BoundaryDenied, match=message):
            channel.receive(StopReceipt)
    finally:
        channel.close()
        second.close()


@pytest.mark.parametrize(
    "defect,message", [("short", "incomplete"), ("error", "disconnected")]
)
def test_partial_and_failed_send_never_authorize_recheck(defect, message):
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel = barrier.PeerChannel(
        SocketFault(first, defect),
        barrier.process_identity(os.getpid()),
        expires=time.monotonic() + 10,
    )
    try:
        with pytest.raises(barrier.BoundaryDenied, match=message):
            channel.send(evidence().stop)
    finally:
        channel.close()
        second.close()


class OversizedDocument(Document):
    value: str


def test_send_has_a_hard_packet_bound():
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with barrier.PeerChannel(
        first, barrier.process_identity(os.getpid()), expires=time.monotonic() + 10
    ) as channel:
        try:
            with pytest.raises(barrier.BoundaryDenied, match="too large"):
                channel.send(OversizedDocument(value="x" * barrier.MAX_PACKET_BYTES))
        finally:
            second.close()


def test_empty_wait_result_is_not_an_acknowledgement(monkeypatch):
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel = barrier.PeerChannel(
        first, barrier.process_identity(os.getpid()), expires=time.monotonic() + 10
    )
    try:
        monkeypatch.setattr(barrier.select, "select", lambda *arguments: ([], [], []))
        with pytest.raises(barrier.BoundaryDenied, match="deadline"):
            channel.receive(StopReceipt)
    finally:
        channel.close()
        second.close()


def test_authenticated_sender_cannot_claim_a_different_stop_producer():
    proof = evidence()
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel = barrier.PeerChannel(
        first, barrier.process_identity(os.getpid()), expires=time.monotonic() + 10
    )
    controller = barrier.BoundaryController(proof.scope, channel)
    try:
        with pytest.raises(barrier.BoundaryDenied, match="unreleased"):
            controller.release(proof.mutation)
        second.send(proof.stop.model_dump_json().encode())
        with pytest.raises(barrier.BoundaryDenied, match="producer"):
            controller.wait_for_stop()
    finally:
        channel.close()
        second.close()


def test_unacknowledged_release_is_permanently_consumed():
    proof = evidence()
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    identity = barrier.process_identity(os.getpid())
    channel = barrier.PeerChannel(
        SocketFault(first, "error"), identity, expires=time.monotonic() + 10
    )
    controller = barrier.BoundaryController(proof.scope, channel)
    try:
        stop = proof.stop.model_copy(update={"producer": identity})
        second.send(stop.model_dump_json().encode())
        assert controller.wait_for_stop() == stop
        mutation = mutation_for(proof.scope, stop)
        with pytest.raises(barrier.BoundaryDenied, match="disconnected"):
            controller.release(mutation)
        assert controller.released and channel.closed
        with pytest.raises(barrier.BoundaryDenied, match="unreleased"):
            controller.release(mutation)
    finally:
        channel.close()
        second.close()
