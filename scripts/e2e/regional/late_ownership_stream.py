"""Bounded, UID-pinned stdio for the single live Executor acceptance probe."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_probe_bundle import stdin_loader

MAX_MESSAGE_BYTES = 65536


class ProbeStream:
    def __init__(
        self,
        stream: Any,
        *,
        scope_sha256: str,
        check_identity: Callable[[], None],
        deadline: float,
        holder_check: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise BoundaryDenied("Executor probe requires a finite future deadline")
        self.stream = stream
        self.scope_sha256 = scope_sha256
        self.check_identity = check_identity
        self.deadline = deadline
        self.buffer = ""
        self.diagnostic_bytes = 0
        self.sequence = 0
        self.closed = False
        self.holder_check = holder_check

    def _check(self) -> None:
        if self.closed or time.monotonic() >= self.deadline:
            raise BoundaryDenied("Executor probe deadline or supervision ended")
        self.check_identity()

    def send(self, kind: str, payload: dict[str, Any]) -> None:
        self._check()
        value = (
            json.dumps(
                {"kind": kind, "scope_sha256": self.scope_sha256, "payload": payload},
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        if len(value.encode()) > MAX_MESSAGE_BYTES:
            raise BoundaryDenied("Executor probe request exceeds its bound")
        self.stream.write_stdin(value)
        self._check()

    def _receive(self, kind: str) -> tuple[str, dict[str, Any]]:
        while "\n" not in self.buffer:
            self._check()
            if not self.stream.is_open():
                raise BoundaryDenied("Executor probe exited without an acknowledgement")
            self.stream.update(timeout=min(1, max(0, self.deadline - time.monotonic())))
            self.buffer += self.stream.read_stdout()
            self.diagnostic_bytes += len(self.stream.read_stderr().encode())
            if (
                len(self.buffer.encode()) > MAX_MESSAGE_BYTES
                or self.diagnostic_bytes > MAX_MESSAGE_BYTES
            ):
                raise BoundaryDenied("Executor probe output exceeds its bound")
        line, self.buffer = self.buffer.split("\n", 1)
        self._check()
        try:
            message = json.loads(line)
        except ValueError:
            raise BoundaryDenied(
                "Executor probe output is not a protocol message"
            ) from None
        if (
            not isinstance(message, dict)
            or set(message) != {"kind", "scope_sha256", "sequence", "payload"}
            or (
                message["kind"] != kind
                and not (
                    message["kind"] == "holder-check" and self.holder_check is not None
                )
            )
            or message["scope_sha256"] != self.scope_sha256
            or type(message["sequence"]) is not int
            or message["sequence"] != self.sequence + 1
            or not isinstance(message["payload"], dict)
        ):
            raise BoundaryDenied(
                "Executor probe acknowledgement is stale or out of order"
            )
        self.sequence += 1
        return message["kind"], message["payload"]

    def receive(self, kind: str) -> dict[str, Any]:
        while True:
            observed, payload = self._receive(kind)
            if observed == kind:
                return payload
            assert self.holder_check is not None
            self.send("holder-check-result", self.holder_check(payload))

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.stream.close()

    def finish(self) -> None:
        """Require the remote process exit status, not just its final JSON line."""
        while self.stream.is_open():
            self._check()
            self.stream.update(timeout=min(1, max(0, self.deadline - time.monotonic())))
            self.buffer += self.stream.read_stdout()
            self.diagnostic_bytes += len(self.stream.read_stderr().encode())
            if self.buffer.strip() or self.diagnostic_bytes > MAX_MESSAGE_BYTES:
                raise BoundaryDenied(
                    "Executor probe produced output after terminal closure"
                )
        self._check()
        if self.stream.returncode != 0:
            raise BoundaryDenied("Executor probe did not exit successfully")
        self.close()


def open_executor_stream(
    *,
    kubeconfig: str,
    context: str,
    namespace: str,
    pod: str,
    python: str,
    program: str,
    container: str = "executor",
    chroot: str | None = None,
) -> Any:
    from kubernetes import client, config
    from kubernetes.stream import stream

    api_client = config.new_client_from_config(config_file=kubeconfig, context=context)
    configuration = api_client.configuration
    if not configuration.verify_ssl or not configuration.host.startswith("https://"):
        api_client.close()
        raise BoundaryDenied("Executor probe requires verified Kubernetes TLS")
    core = client.CoreV1Api(api_client)
    try:
        socket = stream(
            core.connect_get_namespaced_pod_exec,
            pod,
            namespace,
            command=(
                (["chroot", chroot] if chroot is not None else [])
                + [python, "-I", "-u", "-c", stdin_loader(program)]
            ),
            container=container,
            stdin=True,
            stdout=True,
            stderr=True,
            tty=False,
            _preload_content=False,
            _request_timeout=(5, 15),
        )
    except BaseException:
        api_client.close()
        raise
    managed = ManagedExecStream(socket, api_client)
    try:
        managed.write_stdin(program)
    except BaseException:
        managed.close()
        raise
    return managed


class ManagedExecStream:
    def __init__(self, socket: Any, api_client: Any) -> None:
        self.socket = socket
        self.api_client = api_client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.socket, name)

    def close(self) -> None:
        try:
            self.socket.close()
        finally:
            self.api_client.close()
