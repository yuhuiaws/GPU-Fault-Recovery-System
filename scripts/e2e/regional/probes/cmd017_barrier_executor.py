#!/usr/bin/env python3
"""Probe executor for GF-REGIONAL-CMD-017: a multi-node barrier step at claim.

Runs the deployed ``ClusterActionExecutor`` (the executor image's own wheel)
with one stand-in adapter that owns the seeded step and has no barrier
coordinator -- the same shape the real ``NodeActionWorkflowAdapter`` has on a
regional executor, where ``barriers is None`` because barrier state lives in
the control-plane store. ``ClusterActionExecutor._execute`` must refuse the
two-node ``RESET_ALL_GPUS_NVSWITCHES`` at the claim boundary
(``_barrier_hold``): WAITING with ``status_source=executor-barrier-unavailable``,
``multi_node_barrier_unavailable=True`` in the details, one count on
``barrier_unavailable_holds_total`` per claim, and the adapter never reached.

The adapter is a stand-in on purpose. If the claim boundary regressed, the
adapter's ``execute`` records that it was reached and returns FAILED; nothing
here can address a node, and the seeded cluster has no Node Agents anyway. The
regression evidence is a file, not a fabric reset.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Thread

from gpu_fault.cluster_executor import ClusterActionExecutor, RegionalExecutorClient
from gpu_fault.execution.models import WorkflowStepOutcome

STATE = Path("/state")
READY = STATE / "ready.json"
EXECUTOR_STATE = STATE / "executor-state.json"
CLAIM_STATE = STATE / "claim-state.json"
ADAPTER_EXECUTED = STATE / "adapter-executed.json"
OWNER = os.getenv("EXECUTOR_OWNER", "gpu-fault-cmd017-barrier")
EXECUTOR_ID = "cmd017-barrier-executor"
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))
LEASE_SECONDS = int(os.getenv("LEASE_SECONDS", "30"))


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class BarrierStandInAdapter:
    """Owns the seeded step, coordinates nothing, must never be executed."""

    owner = OWNER
    # What the regional NodeActionWorkflowAdapter has: no coordinator.
    barriers = None
    registry = None

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner

    def execute(self, context) -> WorkflowStepOutcome:
        _write(
            ADAPTER_EXECUTED,
            {
                "idempotency_key": context.idempotency_key,
                "operation": context.step.operation.value,
                "node_ids": list(context.step.node_ids),
                "observed_at_epoch": time.time(),
            },
        )
        return WorkflowStepOutcome.failed(
            "CMD-017 stand-in adapter was reached; the claim boundary did not "
            "hold the multi-node barrier step"
        )


def record_executor_state(executor: ClusterActionExecutor) -> None:
    while True:
        snapshot = executor.metrics_snapshot()
        snapshot["observed_at_epoch"] = time.time()
        snapshot["adapter_executed"] = ADAPTER_EXECUTED.exists()
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
        [BarrierStandInAdapter()],
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
            "adapter_has_barrier_coordinator": False,
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
