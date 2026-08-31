#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from threading import Thread
import time

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    RegionalExecutorClient,
    RegionalFleetRegistry,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import AdvisoryNotification


STATE = Path("/state")
READY = STATE / "ready.json"
EXECUTOR_STATE = STATE / "executor-state.json"
WINNER = STATE / "winner.json"
OWNER = "gpu-fault-ha006-test"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class SharedLedgerAdapter:
    owner = OWNER

    def __init__(
        self,
        registry: RegionalFleetRegistry,
        *,
        run_id: str,
        sleep_seconds: float,
    ) -> None:
        self.registry = registry
        self.run_id = run_id
        self.sleep_seconds = sleep_seconds

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner

    def execute(self, context) -> WorkflowStepOutcome:
        candidate = AdvisoryNotification(
            notification_id=f"notification-{self.run_id}-{socket.gethostname()}",
            deduplication_key=f"{self.run_id}/shared-action-ledger",
            cluster_name=context.incident.cluster_id,
            incident_id=context.incident.incident_id,
            subject="HA-006 shared action ledger drill",
            body_text="Synthetic non-destructive executor takeover marker.",
            support_case_draft="Acceptance drill only; no provider action.",
            drill_id=self.run_id,
            category="HA_ACCEPTANCE",
        )
        saved = self.registry.save_notification_if_absent(candidate)
        won = saved.notification_id == candidate.notification_id
        if won:
            atomic_json(
                WINNER,
                {
                    "notification_id": saved.notification_id,
                    "pod": socket.gethostname(),
                    "observed_at": time.time(),
                },
            )
            time.sleep(self.sleep_seconds)
        return WorkflowStepOutcome.succeeded(
            operation_id=f"ha006/{context.idempotency_key}",
            details={
                "simulated": True,
                "cached": not won,
                "physical_count": 1,
                "shared_notification_id": saved.notification_id,
            },
        )


def record_state(executor: ClusterActionExecutor) -> None:
    while True:
        atomic_json(
            EXECUTOR_STATE,
            {
                "pod": socket.gethostname(),
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
        )
        time.sleep(0.5)


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    registrations = json.loads(Path("/tokens/clusters.json").read_text())
    registration = registrations[0]
    client = RegionalExecutorClient(
        os.environ["CONTROL_PLANE_URL"],
        registration["cluster_id"],
        registration["token"],
        timeout_seconds=30,
        ca_file="/tls/ca.crt",
        executor_artifact_sha256=os.environ["EXECUTOR_ARTIFACT_SHA256"],
        executor_compatibility_digest=os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
    )
    registry = RegionalFleetRegistry(client)
    run_id = os.environ["RUN_ID"]
    lease_seconds = int(os.environ.get("LEASE_SECONDS", "30"))
    executor = ClusterActionExecutor(
        client,
        [
            SharedLedgerAdapter(
                registry,
                run_id=run_id,
                sleep_seconds=float(os.environ.get("WINNER_SLEEP_SECONDS", "60")),
            )
        ],
        executor_id=socket.gethostname(),
        allowed_namespaces={"default"},
        poll_seconds=1,
        lease_seconds=lease_seconds,
        batch_size=1,
        max_concurrent_commands=1,
        claim_backoff_max_seconds=4,
        claim_state_path="/state/claim-state.json",
    )
    atomic_json(
        READY,
        {
            "cluster_id": registration["cluster_id"],
            "executor_id": executor.executor_id,
            "lease_seconds": lease_seconds,
            "run_id": run_id,
        },
    )
    Thread(target=record_state, args=(executor,), daemon=True).start()
    executor.run()


if __name__ == "__main__":
    main()
