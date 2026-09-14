#!/usr/bin/env python3
"""NET-003 probe executor: the control plane commits a result the client never sees.

The loopback proxy in front of the control plane forwards the first result
post upstream, waits for the control plane's response, discards it and resets
the client connection instead -- the "server committed, client lost the
response" window. The client then does what a client with a lost response
does: it retries the same terminal result once, and the control plane must
answer idempotently. An earlier version of this proxy reset the connection
*before* forwarding, so the control plane never saw the first post and the
case silently degenerated into a lease-expiry reclaim.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import select
import socket
import struct
from threading import Lock, Thread
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    RegionalExecutorClient,
)
from gpu_fault.execution.models import WorkflowStepOutcome


STATE = Path("/state")
DROP_NEXT = STATE / "drop-next"
DROP_OBSERVED = STATE / "drop-observed.json"
ACTION_STARTED = STATE / "action-started"
BLOCK = STATE / "block"
ACTION_GATE_OBSERVED = STATE / "action-gate-observed.json"
LEDGER = STATE / "ledger.json"
READY = STATE / "ready.json"
EXECUTOR_STATE = STATE / "executor-state.json"
ROLLBACK_STATE = STATE / "rollback.json"
RESULT_SUBMIT_STARTED = STATE / "result-submit-started.json"
RESULT_INTERRUPTED = STATE / "result-interrupted.json"
RESULT_REPLAYS = STATE / "result-replays.json"
OWNER = os.getenv("EXECUTOR_OWNER", "gpu-fault-net-test")
DROP_ROLLBACK_SECONDS = int(os.getenv("DROP_ROLLBACK_SECONDS", "10"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "60"))
# How long the upstream must stay silent after its response before the
# client is reset. TLS is end-to-end through this proxy, so the response
# cannot be parsed; a quiet upstream is what marks it complete.
RESPONSE_QUIET_SECONDS = float(os.getenv("RESPONSE_QUIET_SECONDS", "2"))
RESPONSE_MAX_WAIT_SECONDS = float(os.getenv("RESPONSE_MAX_WAIT_SECONDS", "25"))
# The gap between the lost response and the client's single retry. Long
# enough for the runner to snapshot the committed command in between, short
# enough that no lease renewal fires against the now-terminal command.
REPLAY_DELAY_SECONDS = float(os.getenv("REPLAY_DELAY_SECONDS", "5"))
RESPONSE_LOSS_MODE = "forward-then-reset"
TERMINAL_RESULT_REPLAYS = 1
TLS_APPLICATION_DATA = 23
LOST_RESPONSE_LOG = "net003 result submission lost its response"


class LedgerAdapter:
    owner = OWNER

    def __init__(self, notification_id: str) -> None:
        self._lock = Lock()
        self.notification_id = notification_id

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner

    def execute(self, context) -> WorkflowStepOutcome:
        ACTION_STARTED.write_text(context.idempotency_key, encoding="utf-8")
        with self._lock:
            document = (
                json.loads(LEDGER.read_text(encoding="utf-8"))
                if LEDGER.is_file()
                else {"physical_count": 0, "keys": []}
            )
            cached = context.idempotency_key in document["keys"]
        if not cached:
            # Hold the command LEASED at the action gate until the runner has
            # armed the block, so its leased snapshot cannot race the action
            # that commits the result: an ungated 5s action outran the runner's
            # exec+DB round-trip and the command could commit SUCCEEDED before
            # the runner observed LEASED. Unlike NET-002 the block only gates
            # the action here (never a forced lease expiry), so the runner arms
            # it once and never removes it.
            deadline = time.monotonic() + 30
            while not BLOCK.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "network block was not armed before the simulated action"
                    )
                time.sleep(0.05)
            ACTION_GATE_OBSERVED.write_text(
                json.dumps(
                    {
                        "idempotency_key": context.idempotency_key,
                        "observed_at_epoch": time.time(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            time.sleep(5)
        with self._lock:
            document = (
                json.loads(LEDGER.read_text(encoding="utf-8"))
                if LEDGER.is_file()
                else {"physical_count": 0, "keys": []}
            )
            cached = context.idempotency_key in document["keys"]
            if not cached:
                document["keys"].append(context.idempotency_key)
                document["physical_count"] += 1
                temporary = LEDGER.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(document, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(temporary, LEDGER)
        return WorkflowStepOutcome.succeeded(
            operation_id=f"net-test/{context.idempotency_key}",
            details={
                "simulated": True,
                "cached": cached,
                "physical_count": document["physical_count"],
                "notification_id": self.notification_id,
            },
        )


def _write_state(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class InterruptingRegionalExecutorClient(RegionalExecutorClient):
    """Loses the response to the first result post, then retries it once.

    The retry is the client-side half of the window under test: a client that
    never saw the control plane's answer cannot know whether the result was
    committed, so it posts the same terminal result again and the control
    plane has to answer idempotently. One replay is what a real client does;
    the wider terminal-replay protocol matrix belongs to CMD-007.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drop_injected = False

    def complete(self, command, result):
        if self._drop_injected:
            return super().complete(command, result)
        self._drop_injected = True
        _write_state(
            RESULT_SUBMIT_STARTED,
            {"command_id": command.command_id, "observed_at_epoch": time.time()},
        )
        DROP_NEXT.touch()
        try:
            first = super().complete(command, result)
        except Exception as exc:  # noqa: BLE001 - the transport error under test
            # The regional client wraps BOTH a real HTTP rejection and a
            # transport failure into ClusterExecutorError; status_code tells
            # them apart (regional_client._send: a None status is "no answer,
            # safe to retry"). A verdict the client DID receive (non-None
            # status) is not the lost-response window and surfaces as the
            # rejection it is; a None status -- or any other transport
            # exception -- is the connection reset this case injects, which the
            # client must treat as a lost response and replay once.
            if isinstance(exc, ClusterExecutorError) and exc.status_code is not None:
                raise
            interrupted = {
                "command_id": command.command_id,
                "exception": type(exc).__name__,
                "detail": str(exc)[:500],
                "observed_at_epoch": time.time(),
                "first_post_succeeded": False,
            }
            logging.warning("%s: %s: %s", LOST_RESPONSE_LOG, type(exc).__name__, exc)
        else:
            _write_state(
                RESULT_INTERRUPTED,
                {
                    "command_id": command.command_id,
                    "first_post_succeeded": True,
                    "observed_at_epoch": time.time(),
                },
            )
            return first
        _write_state(RESULT_INTERRUPTED, interrupted)
        time.sleep(REPLAY_DELAY_SECONDS)
        replay_sent_at = time.time()
        replay = super().complete(command, result)
        _write_state(
            RESULT_REPLAYS,
            {
                "command_id": command.command_id,
                "count": TERMINAL_RESULT_REPLAYS,
                "replay_sent_at_epoch": replay_sent_at,
                "responses": [
                    {
                        "command_id": replay.command_id,
                        "status": replay.status.value,
                        "status_source": replay.status_source,
                        "updated_at": replay.updated_at.isoformat(),
                    }
                ],
            },
        )
        return replay


class TlsRecordScanner:
    """Count application_data records in one direction of a TLS byte stream.

    Records may straddle TCP segments, so the five-byte header is assembled
    across ``feed`` calls and the body length is skipped statefully. The
    client's first application_data record marks that the HTTP request is on
    its way: in TLS 1.3 that record is the encrypted Finished sent in the same
    flight as the request, in TLS 1.2 it is the request itself. Either way,
    upstream bytes that arrive after it belong to the response.
    """

    def __init__(self) -> None:
        self._header = b""
        self._remaining = 0
        self.application_records = 0

    def feed(self, data: bytes) -> None:
        while data:
            if self._remaining:
                take = min(self._remaining, len(data))
                self._remaining -= take
                data = data[take:]
                continue
            need = 5 - len(self._header)
            self._header += data[:need]
            data = data[need:]
            if len(self._header) < 5:
                return
            if self._header[0] == TLS_APPLICATION_DATA:
                self.application_records += 1
            self._remaining = int.from_bytes(self._header[3:5], "big")
            self._header = b""


def relay_losing_response(
    client: socket.socket,
    upstream: socket.socket,
    *,
    quiet_seconds: float = RESPONSE_QUIET_SECONDS,
    max_wait_seconds: float = RESPONSE_MAX_WAIT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Relay one connection but keep the upstream's response from the client.

    Everything the client sends is forwarded. Upstream bytes are forwarded
    until the client has sent an application_data record; from then on they
    are counted and dropped. Once the upstream has been quiet for
    ``quiet_seconds`` after its first withheld byte (or ``max_wait_seconds``
    have passed, or the upstream closed), the client is reset with an RST
    rather than a FIN, so it sees a connection error and not an EOF that
    could be mistaken for an empty response.
    """

    scanner = TlsRecordScanner()
    response_bytes = 0
    response_started_at: float | None = None
    last_upstream_at: float | None = None
    client_closed = False
    upstream_closed = False
    started = clock()
    while True:
        readable, _, _ = select.select([client, upstream], [], [], 0.1)
        now = clock()
        if response_started_at is not None and last_upstream_at is not None:
            if now - last_upstream_at >= quiet_seconds:
                break
            if now - response_started_at >= max_wait_seconds:
                break
        elif now - started >= max_wait_seconds * 2:
            break
        if not readable:
            continue
        for source in readable:
            data = source.recv(65536)
            if source is client:
                if not data:
                    client_closed = True
                    break
                scanner.feed(data)
                upstream.sendall(data)
                continue
            if not data:
                upstream_closed = True
                break
            if scanner.application_records:
                response_bytes += len(data)
                last_upstream_at = now
                if response_started_at is None:
                    response_started_at = now
            else:
                client.sendall(data)
        if client_closed or upstream_closed:
            break
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()
    return {
        "observed_at_epoch": time.time(),
        "connection_reset": True,
        "mode": RESPONSE_LOSS_MODE,
        "request_forwarded": scanner.application_records > 0,
        "client_application_records": scanner.application_records,
        "upstream_response_bytes": response_bytes,
        "client_closed_first": client_closed,
        "upstream_closed_first": upstream_closed,
        "held_seconds": round(clock() - started, 3),
    }


class GateProxy:
    def __init__(self, target_ip: str, listen_port: int) -> None:
        self.target_ip = target_ip
        self.listen_port = listen_port

    @staticmethod
    def _relay(client: socket.socket, upstream: socket.socket) -> None:
        sockets = [client, upstream]
        while True:
            readable, _, _ = select.select(sockets, [], [], 30)
            if not readable:
                continue
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return
                target = upstream if source is client else client
                target.sendall(data)

    def _handle(self, client: socket.socket) -> None:
        try:
            lose_response = DROP_NEXT.exists()
            if lose_response:
                DROP_NEXT.unlink(missing_ok=True)
            with socket.create_connection(
                (self.target_ip, 443), timeout=30
            ) as upstream:
                if lose_response:
                    observed = relay_losing_response(client, upstream)
                    _write_state(DROP_OBSERVED, observed)
                    return
                self._relay(client, upstream)
        except OSError:
            logging.exception("proxy connection failed")
        finally:
            client.close()

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", self.listen_port))
            listener.listen(32)
            while True:
                client, _address = listener.accept()
                Thread(target=self._handle, args=(client,), daemon=True).start()


def rollback_stale_drop() -> None:
    while True:
        try:
            age_seconds = time.time() - DROP_NEXT.stat().st_mtime
        except FileNotFoundError:
            time.sleep(0.25)
            continue
        if age_seconds >= DROP_ROLLBACK_SECONDS:
            DROP_NEXT.unlink(missing_ok=True)
            _write_state(
                ROLLBACK_STATE, {"automatic": True, "pending_seconds": age_seconds}
            )
            logging.error(
                "automatically removed stale connection-drop marker after %.1f seconds",
                age_seconds,
            )
        time.sleep(0.25)


def record_executor_state(executor: ClusterActionExecutor) -> None:
    while True:
        temporary = EXECUTOR_STATE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "claimed_total": executor.claimed_total,
                    "reported_failures": executor.reported_failures,
                    "unexpected_failures": executor.unexpected_failures,
                    "lease_renewal_failures": executor.lease_renewal_failures,
                    "last_successful_claim_at": (
                        executor.last_successful_claim_at.isoformat()
                        if executor.last_successful_claim_at is not None
                        else None
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, EXECUTOR_STATE)
        time.sleep(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    STATE.mkdir(parents=True, exist_ok=True)
    control_url = os.environ["CONTROL_PLANE_URL"].rstrip("/")
    host = urlsplit(control_url).hostname
    if not host:
        raise RuntimeError("control-plane URL has no hostname")
    original_getaddrinfo = socket.getaddrinfo
    target_ip = original_getaddrinfo(
        host,
        443,
        family=socket.AF_INET,
        type=socket.SOCK_STREAM,
    )[0][4][0]
    listen_port = int(os.getenv("PROXY_PORT", "18443"))

    def proxy_getaddrinfo(name, port, *args, **kwargs):
        if name == host and int(port) == listen_port:
            return original_getaddrinfo("127.0.0.1", port, *args, **kwargs)
        return original_getaddrinfo(name, port, *args, **kwargs)

    proxy = GateProxy(target_ip, listen_port)
    Thread(target=proxy.run, daemon=True, name="net-test-proxy").start()
    Thread(
        target=rollback_stale_drop,
        daemon=True,
        name="connection-drop-rollback",
    ).start()
    socket.getaddrinfo = proxy_getaddrinfo
    registrations = json.loads(Path("/tokens/clusters.json").read_text())
    registration = registrations[0]
    notification_id = os.environ["NOTIFICATION_ID"]
    client = InterruptingRegionalExecutorClient(
        f"https://{host}:{listen_port}",
        registration["cluster_id"],
        registration["token"],
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
        ca_file="/tls/ca.crt",
        executor_artifact_sha256=os.environ["EXECUTOR_ARTIFACT_SHA256"],
        executor_compatibility_digest=os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
    )
    _write_state(
        READY,
        {
            "cluster_id": registration["cluster_id"],
            "owner": OWNER,
            "target_ip": target_ip,
            "proxy_port": listen_port,
            "drop_rollback_seconds": DROP_ROLLBACK_SECONDS,
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
            "lease_seconds": LEASE_SECONDS,
            "result_connection_reset": True,
            "response_loss_mode": RESPONSE_LOSS_MODE,
            "response_quiet_seconds": RESPONSE_QUIET_SECONDS,
            "replay_delay_seconds": REPLAY_DELAY_SECONDS,
            "terminal_result_replays": TERMINAL_RESULT_REPLAYS,
        },
    )
    executor = ClusterActionExecutor(
        client,
        [LedgerAdapter(notification_id)],
        executor_id="net-test-executor",
        allowed_namespaces={"default"},
        poll_seconds=1,
        lease_seconds=LEASE_SECONDS,
        batch_size=1,
        max_concurrent_commands=1,
        claim_backoff_max_seconds=4,
        claim_state_path="/state/claim-state.json",
    )
    Thread(
        target=record_executor_state,
        args=(executor,),
        daemon=True,
        name="executor-state",
    ).start()
    executor.run()


if __name__ == "__main__":
    main()
