#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import select
import socket
from threading import Lock, Thread
import time
from urllib.parse import urlsplit

from gpu_fault.cluster_executor import ClusterActionExecutor, RegionalExecutorClient
from gpu_fault.execution.models import WorkflowStepOutcome


STATE = Path("/state")
BLOCK = STATE / "block"
ACTION_STARTED = STATE / "action-started"
ACTION_GATE_OBSERVED = STATE / "action-gate-observed.json"
LEDGER = STATE / "ledger.json"
READY = STATE / "ready.json"
EXECUTOR_STATE = STATE / "executor-state.json"
ROLLBACK_STATE = STATE / "rollback.json"
RESULT_SUBMIT_WAITING = STATE / "result-submit-waiting.json"
RESULT_SUBMIT_RELEASED = STATE / "result-submit-released.json"
OWNER = os.getenv("EXECUTOR_OWNER", "gpu-fault-net-test")
BLOCK_ROLLBACK_SECONDS = int(os.getenv("BLOCK_ROLLBACK_SECONDS", "100"))
# Deliberately far above the production client's 15s: the gated proxy *holds*
# a connection while the block is on rather than refusing it, so the result
# post and the hanging lease renewal both reach the control plane the moment
# the block lifts -- after the lease has expired server-side -- and are refused
# with 409. A 15s timeout would turn them into local timeouts and the case
# would never observe the stale-lease rejection it exists to prove. The
# runner records this as a limitation of the case, not as a pass condition.
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "180"))
LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "60"))


class LedgerAdapter:
    owner = OWNER

    def __init__(self) -> None:
        self._lock = Lock()

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
            deadline = time.monotonic() + 30
            while not BLOCK.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "network block was not armed before simulated execution"
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
            },
        )


class GatedRegionalExecutorClient(RegionalExecutorClient):
    def complete(self, command, result):
        if BLOCK.exists():
            RESULT_SUBMIT_WAITING.write_text(
                json.dumps(
                    {
                        "command_id": command.command_id,
                        "observed_at_epoch": time.time(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            while BLOCK.exists():
                time.sleep(0.1)
            RESULT_SUBMIT_RELEASED.write_text(
                json.dumps(
                    {
                        "command_id": command.command_id,
                        "observed_at_epoch": time.time(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return super().complete(command, result)


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
            while BLOCK.exists():
                time.sleep(0.1)
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


def rollback_stale_block() -> None:
    while True:
        try:
            age_seconds = time.time() - BLOCK.stat().st_mtime
        except FileNotFoundError:
            time.sleep(0.25)
            continue
        if age_seconds >= BLOCK_ROLLBACK_SECONDS:
            BLOCK.unlink(missing_ok=True)
            ROLLBACK_STATE.write_text(
                json.dumps(
                    {
                        "automatic": True,
                        "blocked_seconds": age_seconds,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            logging.error(
                "automatically removed stale network block after %.1f seconds",
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
        target=rollback_stale_block,
        daemon=True,
        name="network-block-rollback",
    ).start()
    socket.getaddrinfo = proxy_getaddrinfo
    registrations = json.loads(Path("/tokens/clusters.json").read_text())
    registration = registrations[0]
    client = GatedRegionalExecutorClient(
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
                "owner": OWNER,
                "target_ip": target_ip,
                "proxy_port": listen_port,
                "proxy_mode": "hold-while-blocked",
                "block_rollback_seconds": BLOCK_ROLLBACK_SECONDS,
                "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
                "lease_seconds": LEASE_SECONDS,
                "result_submission_gate": True,
                "action_requires_network_block": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    executor = ClusterActionExecutor(
        client,
        [LedgerAdapter()],
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
