#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import socket
from threading import Lock
import time

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    RegionalExecutorClient,
)
from gpu_fault.execution.models import WorkflowStepOutcome


STATE = Path("/state")
READY = STATE / "ready.json"
STATS = STATE / "stats.json"
STOP = STATE / "stop"
LEDGER = STATE / "ledger.json"
OWNER = "gpu-fault-ha-test"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class SimulatedAdapter:
    owner = OWNER

    def __init__(self) -> None:
        self._lock = Lock()

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner

    def execute(self, context) -> WorkflowStepOutcome:
        with self._lock:
            document = (
                json.loads(LEDGER.read_text(encoding="utf-8"))
                if LEDGER.is_file()
                else {"physical_count": 0, "keys": [], "operations": []}
            )
            cached = context.idempotency_key in document["keys"]
            if not cached:
                document["keys"].append(context.idempotency_key)
                document["operations"].append(context.step.operation.value)
                document["physical_count"] += 1
                atomic_json(LEDGER, document)
        return WorkflowStepOutcome.succeeded(
            operation_id=f"ha001/{context.idempotency_key}",
            details={
                "simulated": True,
                "cached": cached,
                "physical_count": document["physical_count"],
            },
        )


def failure_key(exc: Exception) -> str:
    if isinstance(exc, ClusterExecutorError) and exc.status_code is not None:
        return f"http-{exc.status_code}"
    return type(exc).__name__


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    client = RegionalExecutorClient(
        os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
        os.environ["GPU_FAULT_CLUSTER_ID"],
        os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        timeout_seconds=15,
        ca_file=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"],
        executor_artifact_sha256=os.environ["GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"],
        executor_compatibility_digest=os.environ[
            "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"
        ],
    )
    executor = ClusterActionExecutor(
        client,
        [SimulatedAdapter()],
        executor_id=f"ha001-probe/{socket.gethostname()}",
        allowed_namespaces={"default", "gpu-fault-system"},
        poll_seconds=2,
        lease_seconds=60,
        batch_size=1,
        max_concurrent_commands=1,
        claim_backoff_max_seconds=4,
        claim_state_path="/state/claim-state.json",
    )
    atomic_json(
        READY,
        {
            "cluster_id": client.cluster_id,
            "executor_id": executor.executor_id,
            "owner": OWNER,
        },
    )
    started_at = time.time()
    counters: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    recent_errors: list[dict] = []
    failure_started: float | None = None
    max_failure_window_seconds = 0.0
    while not STOP.exists():
        iteration_started = time.monotonic()
        failed = False
        try:
            claimed = executor.run_once()
            counters["claim_success"] += 1
            counters["commands_claimed"] += claimed
        except Exception as exc:
            failed = True
            counters["claim_failure"] += 1
            key = failure_key(exc)
            errors[key] += 1
            recent_errors.append(
                {
                    "phase": "claim",
                    "type": key,
                    "observed_at_epoch": time.time(),
                }
            )
        try:
            health = client._get("/healthz")
            if not isinstance(health, dict) or health.get("status") != "ok":
                raise RuntimeError("healthz was not ok")
            counters["health_success"] += 1
        except Exception as exc:
            failed = True
            counters["health_failure"] += 1
            key = failure_key(exc)
            errors[key] += 1
            recent_errors.append(
                {
                    "phase": "health",
                    "type": key,
                    "observed_at_epoch": time.time(),
                }
            )
        now = time.monotonic()
        if failed:
            if failure_started is None:
                failure_started = now
            max_failure_window_seconds = max(
                max_failure_window_seconds,
                now - failure_started,
            )
        elif failure_started is not None:
            max_failure_window_seconds = max(
                max_failure_window_seconds,
                now - failure_started,
            )
            failure_started = None
        ledger = (
            json.loads(LEDGER.read_text(encoding="utf-8"))
            if LEDGER.is_file()
            else {"physical_count": 0, "keys": [], "operations": []}
        )
        atomic_json(
            STATS,
            {
                "started_at_epoch": started_at,
                "observed_at_epoch": time.time(),
                "counters": dict(counters),
                "error_counts": dict(errors),
                "recent_errors": recent_errors[-20:],
                "current_failure_window_seconds": (
                    round(now - failure_started, 3)
                    if failure_started is not None
                    else 0.0
                ),
                "max_failure_window_seconds": round(
                    max_failure_window_seconds,
                    3,
                ),
                "executor": {
                    "claimed_total": executor.claimed_total,
                    "reported_failures": executor.reported_failures,
                    "unexpected_failures": executor.unexpected_failures,
                    "lease_renewal_failures": executor.lease_renewal_failures,
                },
                "ledger": ledger,
            },
        )
        delay = 2.0 - (time.monotonic() - iteration_started)
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    main()
