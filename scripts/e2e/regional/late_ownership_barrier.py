"""Private, one-shot causal rendezvous for an owned Executor probe.

The sockets are inherited descriptors, not a network service. Kernel message
credentials and pidfds bind each message to the original peer incarnation.
No process is signalled, and EOF/timeout never means permission to continue.
"""

from __future__ import annotations

import math
import os
import select
import socket
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Self, TypeVar

from pydantic import ValidationError

from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    Document,
    MutationReceipt,
    ProcessIdentity,
    RecheckPermit,
    StopReceipt,
)

MAX_PACKET_BYTES = 65536
T = TypeVar("T", bound=Document)


class BoundaryDenied(RuntimeError):
    """The rendezvous cannot authorize even a fresh product check."""


def process_identity(pid: int, *, proc: Path = Path("/proc")) -> ProcessIdentity:
    """Read identity only; never read argv, environ, or a credential file."""
    try:
        directory = proc / str(pid)
        fields = (
            (directory / "stat").read_text(encoding="ascii").rpartition(") ")[2].split()
        )
        if fields[0] in {"Z", "X"}:
            raise ValueError("peer exited")
        return ProcessIdentity(
            pid=pid,
            start_ticks=int(fields[19]),
            uid=directory.stat().st_uid,
            boot_id=(proc / "sys/kernel/random/boot_id")
            .read_text(encoding="ascii")
            .strip(),
        )
    except (OSError, ValueError, IndexError):
        raise BoundaryDenied("process identity is unavailable") from None


class PeerChannel:
    """One authenticated SOCK_SEQPACKET endpoint, with a finite absolute deadline."""

    def __init__(
        self, endpoint: socket.socket, peer: ProcessIdentity, *, expires: float
    ) -> None:
        self.endpoint = endpoint
        self.peer = peer
        self.expires = expires
        self.pidfd: int | None = None
        self.closed = False
        try:
            if not math.isfinite(expires) or expires <= time.monotonic():
                raise BoundaryDenied("boundary deadline is invalid")
            if (
                endpoint.family != socket.AF_UNIX
                or endpoint.type != socket.SOCK_SEQPACKET
            ):
                raise BoundaryDenied("boundary requires a private sequenced socket")
            endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            endpoint.setblocking(False)
            self.pidfd = os.pidfd_open(peer.pid)
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self) -> None:
        if self.closed or self.pidfd is None:
            raise BoundaryDenied("boundary channel is closed")
        if time.monotonic() >= self.expires:
            raise BoundaryDenied("boundary deadline expired")
        if (
            select.select([self.pidfd], [], [], 0)[0]
            or process_identity(self.peer.pid) != self.peer
        ):
            raise BoundaryDenied("boundary peer was lost or replaced")

    def _wait(self, *, writing: bool) -> None:
        self.check()
        assert self.pidfd is not None
        readers: list[int | socket.socket] = (
            [self.pidfd] if writing else [self.pidfd, self.endpoint]
        )
        ready, writable, _ = select.select(
            readers,
            [self.endpoint] if writing else [],
            [],
            max(0, self.expires - time.monotonic()),
        )
        self.check()
        if not (writable if writing else self.endpoint in ready):
            raise BoundaryDenied("boundary deadline expired")

    def send(self, document: Document) -> None:
        payload = document.model_dump_json().encode("ascii")
        if len(payload) > MAX_PACKET_BYTES:
            raise BoundaryDenied("boundary packet is too large")
        try:
            self._wait(writing=True)
            if self.endpoint.send(payload) != len(payload):
                raise BoundaryDenied("boundary packet was incomplete")
        except OSError:
            raise BoundaryDenied("boundary peer disconnected") from None

    def receive(self, model: type[T]) -> T:
        try:
            self._wait(writing=False)
            data, ancillary, flags, _ = self.endpoint.recvmsg(
                MAX_PACKET_BYTES, socket.CMSG_SPACE(struct.calcsize("3i"))
            )
            if not data:
                raise BoundaryDenied("boundary peer disconnected")
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise BoundaryDenied("boundary packet was truncated")
            if len(ancillary) != 1:
                raise BoundaryDenied("boundary packet has no unique kernel identity")
            level, kind, credentials = ancillary[0]
            if level != socket.SOL_SOCKET or kind != socket.SCM_CREDENTIALS:
                raise BoundaryDenied("boundary packet has no kernel credentials")
            pid, uid, _gid = struct.unpack("3i", credentials)
            if pid != self.peer.pid or uid != self.peer.uid:
                raise BoundaryDenied("boundary message belongs to another process")
            self.check()
            return model.model_validate_json(data)
        except (OSError, ValueError, ValidationError, struct.error):
            raise BoundaryDenied("boundary packet or peer is invalid") from None

    def close(self) -> None:
        self.closed = True
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None
        self.endpoint.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def check_stop(scope: AcceptanceScope, stop: StopReceipt) -> None:
    if (
        stop.scope_sha256 != scope.digest()
        or stop.executor_uid != scope.executor_uid
        or stop.workload != scope.workload
        or set(stop.participants) != set(scope.participants)
        or len(stop.participants) != len(scope.participants)
        or set(stop.absent_pod_uids) != {item.pod_uid for item in scope.participants}
        or len(stop.absent_pod_uids) != len(scope.participants)
        or set(stop.empty_client_node_uids) != {item.uid for item in scope.nodes}
        or len(stop.empty_client_node_uids) != len(scope.nodes)
    ):
        raise BoundaryDenied(
            "STOP does not bind the original owner and all participants"
        )


class BeforeActionRendezvous:
    """Probe-side callback immediately before the product guard's live read."""

    def __init__(self, scope: AcceptanceScope, channel: PeerChannel) -> None:
        self.scope = scope
        self.channel = channel
        self.state = "NEW"
        self._lock = threading.Lock()

    def arrive(self, stop: StopReceipt) -> RecheckPermit:
        with self._lock:
            if self.state != "NEW":
                raise BoundaryDenied("late or duplicate boundary callback")
            self.state = "ARRIVED"
            try:
                self.scope.check_window(datetime.now(timezone.utc))
                check_stop(self.scope, stop)
                self.channel.send(stop)
                permit = self.channel.receive(RecheckPermit)
                if (
                    permit.scope_sha256 != self.scope.digest()
                    or permit.boundary_id != stop.boundary_id
                    or permit.stop_sha256 != stop.digest()
                ):
                    raise BoundaryDenied("recheck permit does not match this STOP")
                self.check()
                self.state = "RECHECK"
                return permit
            except BaseException:
                self.revoke()
                raise

    def check(self) -> None:
        if self.state not in {"ARRIVED", "RECHECK"}:
            raise BoundaryDenied("boundary is not active")
        self.scope.check_window(datetime.now(timezone.utc))
        self.channel.check()

    def revoke(self) -> None:
        self.state = "REVOKED"
        self.channel.close()


class BoundaryController:
    """Controller side; release requires the acknowledged STOP-linked mutation."""

    def __init__(self, scope: AcceptanceScope, channel: PeerChannel) -> None:
        self.scope = scope
        self.channel = channel
        self.stop: StopReceipt | None = None
        self.released = False

    def wait_for_stop(self) -> StopReceipt:
        if self.stop is not None or self.released:
            raise BoundaryDenied("STOP callback has already been consumed")
        stop = self.channel.receive(StopReceipt)
        check_stop(self.scope, stop)
        if stop.producer != self.channel.peer:
            raise BoundaryDenied("STOP producer is not the owned Executor probe")
        self.stop = stop
        return stop

    def release(self, mutation: MutationReceipt) -> RecheckPermit:
        stop = self.stop
        if stop is None or self.released:
            raise BoundaryDenied("no unreleased STOP boundary")
        self.scope.check_window(datetime.now(timezone.utc))
        if (
            mutation.scope_sha256 != self.scope.digest()
            or mutation.stop_sha256 != stop.digest()
            or mutation.executor_uid != self.scope.executor_uid
            or mutation.producer != stop.producer
            or mutation.sequence <= stop.sequence
        ):
            raise BoundaryDenied("mutation did not acknowledge this STOP boundary")
        permit = RecheckPermit(
            scope_sha256=self.scope.digest(),
            boundary_id=stop.boundary_id,
            stop_sha256=stop.digest(),
            mutation_sha256=mutation.digest(),
        )
        self.released = True
        try:
            self.channel.send(permit)
        except BaseException:
            self.channel.close()
            raise
        return permit
