#!/usr/bin/env python3
"""Probe executor for GF-REGIONAL-CMD-018: one idempotent local ledger.

Runs the deployed ``ClusterActionExecutor`` (the executor image's own wheel)
with one adapter that owns the seeded step and whose "action" is a line in a
local ledger, idempotent by key. The case reads that ledger to prove the open
sibling was executed exactly once while the rewritten dispatch was held. The
seeded cluster has no Node Agents and the node id is synthetic, so nothing the
adapter does can reach a machine.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from gpu_fault.cluster_executor import ClusterActionExecutor, RegionalExecutorClient
from gpu_fault.execution.models import WorkflowStepOutcome

STATE = Path("/state")
READY = STATE / "ready.json"
LEDGER = STATE / "ledger.json"
EXECUTOR_STATE = STATE / "executor-state.json"
CLAIM_STATE = STATE / "claim-state.json"
OWNER = os.getenv("EXECUTOR_OWNER", "gpu-fault-cmd018-sibling")
EXECUTOR_ID = "cmd018-ledger-executor"
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))
LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "30"))


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_ledger() -> dict[str, Any]:
    if LEDGER.is_file():
        value = json.loads(LEDGER.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    return {"physical_count": 0, "keys": []}


class LedgerAdapter:
    owner = OWNER
    barriers = None
    registry = None

    def __init__(self) -> None:
        self._lock = Lock()

    def supports(self, step: Any) -> bool:
        return bool(step.execution_owner == self.owner)

    def execute(self, context: Any) -> WorkflowStepOutcome:
        with self._lock:
            document = _read_ledger()
            cached = context.idempotency_key in document["keys"]
            if not cached:
                document["keys"].append(context.idempotency_key)
                document["physical_count"] += 1
                document["last_command_id"] = getattr(context, "command_id", None)
                _write(LEDGER, document)
        return WorkflowStepOutcome.succeeded(
            operation_id=f"cmd018-ledger/{context.idempotency_key}",
            details={
                "simulated": True,
                "cached": cached,
                "physical_count": document["physical_count"],
            },
        )


def record_executor_state(executor: ClusterActionExecutor) -> None:
    while True:
        snapshot = executor.metrics_snapshot()
        snapshot["observed_at_epoch"] = time.time()
        snapshot["ledger"] = _read_ledger()
        _write(EXECUTOR_STATE, snapshot)
        time.sleep(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    STATE.mkdir(parents=True, exist_ok=True)
    registration = json.loads(Path("/tokens/clusters.json").read_text())[0]
    client = RegionalExecutorClient(
        os.environ["CONTROL_PLANE_URL"].rstrip("/"),
        registration["cluster_id"],
        registration["token"],
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
        ca_file="/tls/ca.crt",
        executor_artifact_sha256=os.environ["EXECUTOR_ARTIFACT_SHA256"],
        executor_compatibility_digest=os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
    )
    executor = ClusterActionExecutor(
        client,
        [LedgerAdapter()],
        executor_id=EXECUTOR_ID,
        allowed_namespaces={"default"},
        poll_seconds=2,
        lease_seconds=LEASE_SECONDS,
        batch_size=1,
        max_concurrent_commands=1,
        claim_backoff_max_seconds=8,
        claim_state_path=str(CLAIM_STATE),
    )
    _write(
        READY,
        {
            "cluster_id": registration["cluster_id"],
            "executor_id": EXECUTOR_ID,
            "owner": OWNER,
            "lease_seconds": LEASE_SECONDS,
            "claim_state_path": str(CLAIM_STATE),
        },
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
