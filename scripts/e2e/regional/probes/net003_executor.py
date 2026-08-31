#!/usr/bin/env python3
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
from urllib.parse import urlsplit

from gpu_fault.cluster_executor import ClusterActionExecutor, RegionalExecutorClient
from gpu_fault.execution.models import WorkflowStepOutcome


STATE = Path("/state")
DROP_NEXT = STATE / "drop-next"
DROP_OBSERVED = STATE / "drop-observed.json"
ACTION_STARTED = STATE / "action-started"
LEDGER = STATE / "ledger.json"
READY = STATE / "ready.json"
EXECUTOR_STATE = STATE / "executor-state.json"
ROLLBACK_STATE = STATE / "rollback.json"
RESULT_SUBMIT_STARTED = STATE / "result-submit-started.json"
RESULT_REPLAYS = STATE / "result-replays.json"
OWNER = "gpu-fault-net-test"
DROP_ROLLBACK_SECONDS = int(os.getenv("DROP_ROLLBACK_SECONDS", "10"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))


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


class InterruptingRegionalExecutorClient(RegionalExecutorClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drop_injected = False
        self._terminal_replayed = False

    def complete(self, command, result):
        if not self._drop_injected:
            self._drop_injected = True
            RESULT_SUBMIT_STARTED.write_text(
                json.dumps(
                    {
                        "command_id": command.command_id,
                        "observed_at_epoch": time.time(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            DROP_NEXT.touch()
            return super().complete(command, result)

        completed = super().complete(command, result)
        if not self._terminal_replayed:
            replayed = []
            for _index in range(2):
                replay = super().complete(command, result)
                replayed.append(
                    {
                        "command_id": replay.command_id,
                        "status": replay.status.value,
                    }
                )
            RESULT_REPLAYS.write_text(
                json.dumps(
                    {
                        "command_id": command.command_id,
                        "count": len(replayed),
                        "responses": replayed,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._terminal_replayed = True
        return completed


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
            if DROP_NEXT.exists():
                DROP_NEXT.unlink(missing_ok=True)
                client.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_LINGER,
                    struct.pack("ii", 1, 0),
                )
                DROP_OBSERVED.write_text(
                    json.dumps(
                        {
                            "observed_at_epoch": time.time(),
                            "connection_reset": True,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                return
            with socket.create_connection(
                (self.target_ip, 443), timeout=30
            ) as upstream:
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
            ROLLBACK_STATE.write_text(
                json.dumps(
                    {
                        "automatic": True,
                        "pending_seconds": age_seconds,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
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
    READY.write_text(
        json.dumps(
            {
                "cluster_id": registration["cluster_id"],
                "target_ip": target_ip,
                "proxy_port": listen_port,
                "drop_rollback_seconds": DROP_ROLLBACK_SECONDS,
                "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
                "result_connection_reset": True,
                "terminal_result_replays": 2,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    executor = ClusterActionExecutor(
        client,
        [LedgerAdapter(notification_id)],
        executor_id="net-test-executor",
        allowed_namespaces={"default"},
        poll_seconds=1,
        lease_seconds=60,
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
