#!/usr/bin/env python3
"""Probe executor for GF-REGIONAL-NET-006: a lease lost during a long action.

Runs the deployed ``ClusterActionExecutor`` (imported from the executor image's
own wheel) with one local adapter whose action deliberately outlives the
command lease while the control plane is unreachable. The point is the moment
the action returns: the executor's ``CommandLeaseWatch`` has by then counted
``lease_renewal_failure_limit`` consecutive renewal failures (or watched its
local lease window pass), so the result must be *withheld* -- never posted
under a lease this process can no longer vouch for -- and counted in
``results_withheld_total``. The next lease holder (this same Pod, after the
block lifts) reclaims the command and finishes it from the idempotency ledger.

Differences from ``net002_executor.py``, which proves the *other* half (a
result posted after the lease expired is rejected with 409):

* the proxy **refuses** connections while ``/state/block`` exists instead of
  holding them, so lease renewals fail fast and the failure counter moves;
* the adapter holds the action for ``ACTION_HOLD_SECONDS`` after the block is
  observed, long enough for the watch to declare the lease lost, and records
  the reason the node-action lease guard would have seen on its thread;
* the executor's ``metrics_snapshot()`` is written next to the claim-state
  breadcrumb so the runner reads the same counters an operator would.

The proxy still has the automatic block rollback: a runner that dies mid-case
cannot leave the executor cut off past ``BLOCK_ROLLBACK_SECONDS``.
"""

from __future__ import annotations

import json
import logging
import os
import select
import socket
import time
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit

from gpu_fault.adapters.node_action.lease_guard import lease_hold_reason
from gpu_fault.cluster_executor import ClusterActionExecutor, RegionalExecutorClient
from gpu_fault.execution.models import WorkflowStepOutcome

STATE = Path("/state")
BLOCK = STATE / "block"
ACTION_STARTED = STATE / "action-started"
ACTION_GATE_OBSERVED = STATE / "action-gate-observed.json"
ACTION_RETURNED = STATE / "action-returned.json"
LEASE_GUARD_OBSERVED = STATE / "lease-guard-observed.json"
LEDGER = STATE / "ledger.json"
READY = STATE / "ready.json"
EXECUTOR_STATE = STATE / "executor-state.json"
CLAIM_STATE = STATE / "claim-state.json"
ROLLBACK_STATE = STATE / "rollback.json"
OWNER = os.getenv("EXECUTOR_OWNER", "gpu-fault-net006-test")
EXECUTOR_ID = "net006-test-executor"
BLOCK_ROLLBACK_SECONDS = int(os.getenv("BLOCK_ROLLBACK_SECONDS", "150"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))
ACTION_HOLD_SECONDS = int(os.getenv("ACTION_HOLD_SECONDS", "100"))
LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "60"))
LEASE_FAILURE_LIMIT = int(os.getenv("LEASE_FAILURE_LIMIT", "3"))


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_ledger() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"physical_count": 0, "keys": []}


class HoldingLedgerAdapter:
    """One simulated physical action, idempotent by key, held past the lease."""

    owner = OWNER

    def __init__(self) -> None:
        self._lock = Lock()

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner

    def execute(self, context) -> WorkflowStepOutcome:
        ACTION_STARTED.write_text(context.idempotency_key, encoding="utf-8")
        with self._lock:
            cached = context.idempotency_key in _read_ledger()["keys"]
        if not cached:
            deadline = time.monotonic() + 30
            while not BLOCK.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "network block was not armed before simulated execution"
                    )
                time.sleep(0.05)
            _write(
                ACTION_GATE_OBSERVED,
                {
                    "idempotency_key": context.idempotency_key,
                    "observed_at_epoch": time.time(),
                    "action_hold_seconds": ACTION_HOLD_SECONDS,
                },
            )
            # The action outlives the lease on purpose. Nothing is torn down
            # when the lease is lost -- that is the agent ledger's job on a
            # real node -- so the hold simply runs its course.
            hold_until = time.monotonic() + ACTION_HOLD_SECONDS
            while time.monotonic() < hold_until:
                time.sleep(0.5)
            _write(
                LEASE_GUARD_OBSERVED,
                {
                    "idempotency_key": context.idempotency_key,
                    "observed_at_epoch": time.time(),
                    # What a node-action send on this thread would have been
                    # refused with; None means the guard saw a live lease.
                    "reason": lease_hold_reason(),
                },
            )
        with self._lock:
            document = _read_ledger()
            cached = context.idempotency_key in document["keys"]
            if not cached:
                document["keys"].append(context.idempotency_key)
                document["physical_count"] += 1
                _write(LEDGER, document)
        _write(
            ACTION_RETURNED,
            {
                "idempotency_key": context.idempotency_key,
                "observed_at_epoch": time.time(),
                "cached": cached,
                "physical_count": document["physical_count"],
            },
        )
        return WorkflowStepOutcome.succeeded(
            operation_id=f"net006-test/{context.idempotency_key}",
            details={
                "simulated": True,
                "cached": cached,
                "physical_count": document["physical_count"],
            },
        )


class RefusingProxy:
    """Loopback relay to the control plane that refuses while blocked."""

    def __init__(self, target_ip: str, listen_port: int) -> None:
        self.target_ip = target_ip
        self.listen_port = listen_port
        self.refused_total = 0
        self._lock = Lock()

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
            if BLOCK.exists():
                # Refuse, do not hold: a renewal that hangs for the whole
                # block never fails, and the failure counter this case
                # measures never moves. Closing without a byte is what a
                # dead control plane looks like from the executor.
                with self._lock:
                    self.refused_total += 1
                return
            with socket.create_connection((self.target_ip, 443), timeout=30) as up:
                self._relay(client, up)
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
            _write(ROLLBACK_STATE, {"automatic": True, "blocked_seconds": age_seconds})
            logging.error(
                "automatically removed stale network block after %.1f seconds",
                age_seconds,
            )
        time.sleep(0.25)


def record_executor_state(
    executor: ClusterActionExecutor, proxy: RefusingProxy
) -> None:
    while True:
        snapshot = executor.metrics_snapshot()
        snapshot["proxy_refused_total"] = proxy.refused_total
        snapshot["observed_at_epoch"] = time.time()
        _write(EXECUTOR_STATE, snapshot)
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
        host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM
    )[0][4][0]
    listen_port = int(os.getenv("PROXY_PORT", "18443"))

    def proxy_getaddrinfo(name, port, *args, **kwargs):
        if name == host and int(port) == listen_port:
            return original_getaddrinfo("127.0.0.1", port, *args, **kwargs)
        return original_getaddrinfo(name, port, *args, **kwargs)

    proxy = RefusingProxy(target_ip, listen_port)
    Thread(target=proxy.run, daemon=True, name="net-test-proxy").start()
    Thread(target=rollback_stale_block, daemon=True, name="block-rollback").start()
    socket.getaddrinfo = proxy_getaddrinfo
    registration = json.loads(Path("/tokens/clusters.json").read_text())[0]
    client = RegionalExecutorClient(
        f"https://{host}:{listen_port}",
        registration["cluster_id"],
        registration["token"],
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
        ca_file="/tls/ca.crt",
        executor_artifact_sha256=os.environ["EXECUTOR_ARTIFACT_SHA256"],
        executor_compatibility_digest=os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
    )
    executor = ClusterActionExecutor(
        client,
        [HoldingLedgerAdapter()],
        executor_id=EXECUTOR_ID,
        allowed_namespaces={"default"},
        poll_seconds=1,
        lease_seconds=LEASE_SECONDS,
        batch_size=1,
        max_concurrent_commands=1,
        claim_backoff_max_seconds=4,
        claim_state_path=str(CLAIM_STATE),
        lease_renewal_failure_limit=LEASE_FAILURE_LIMIT,
    )
    _write(
        READY,
        {
            "cluster_id": registration["cluster_id"],
            "executor_id": EXECUTOR_ID,
            "owner": OWNER,
            "target_ip": target_ip,
            "proxy_port": listen_port,
            "proxy_mode": "refuse-while-blocked",
            "block_rollback_seconds": BLOCK_ROLLBACK_SECONDS,
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
            "action_hold_seconds": ACTION_HOLD_SECONDS,
            "lease_seconds": LEASE_SECONDS,
            "lease_failure_limit": LEASE_FAILURE_LIMIT,
            "action_requires_network_block": True,
            "claim_state_path": str(CLAIM_STATE),
        },
    )
    Thread(
        target=record_executor_state,
        args=(executor, proxy),
        daemon=True,
        name="executor-state",
    ).start()
    executor.run()


if __name__ == "__main__":
    main()
