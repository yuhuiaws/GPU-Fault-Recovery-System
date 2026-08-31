"""Hold four isolated remote-command advisory locks for CAP-002."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.store import PostgresStore


WORK = Path("/work")


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tag = sys.argv[2] if len(sys.argv) > 2 else "default"
    if not 1 <= count <= 4:
        raise ValueError("lock count must be between 1 and 4")
    ready = WORK / f"store-lock-ready-{tag}"
    release = WORK / f"store-lock-release-{tag}"
    ready.unlink(missing_ok=True)
    release.unlink(missing_ok=True)
    url = (WORK / "store-url").read_text(encoding="utf-8")
    command_ids = []
    now = datetime.now(timezone.utc)
    store = PostgresStore(
        url,
        initialize_schema=False,
        pool_min_size=0,
        pool_max_size=2,
    )
    try:
        for index in range(count):
            cluster_id = f"cap-cluster-{index:03d}"
            command_id = f"command-cap002-{tag}-{index:03d}"
            incident_id = f"incident-cap002-{tag}-{index:03d}"
            workflow_id = f"workflow-cap002-{tag}-{index:03d}"
            step = WorkflowStepSpec(
                operation=WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                execution_owner="gpu-fault-kubernetes-adapter",
                node_ids=[f"node-cap002-{index:03d}"],
                parameters={"capacity_test": True},
            )
            workflow = WorkflowRequest(
                request_id=workflow_id,
                incident_id=incident_id,
                runtime_profile_version="hyperpod-v1",
                status=WorkflowStatus.PENDING,
                official_action="CAPACITY_DIAGNOSTIC",
                fencing_token=1,
                official_steps=[step],
                created_at=now,
                updated_at=now,
            )
            incident = FaultIncident(
                incident_id=incident_id,
                event_id=f"event-cap002-{tag}-{index:03d}",
                event_type="CAPACITY_TEST",
                cluster_id=cluster_id,
                node_ids=step.node_ids,
                policy_version="capacity-v1",
                policy_source="CONTROLLED_DRILL",
                official_action="CAPACITY_DIAGNOSTIC",
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=workflow_id,
                drill_id="cap002",
                created_at=now,
                updated_at=now,
            )
            store.ensure_remote_command(
                RemoteActionCommand(
                    command_id=command_id,
                    cluster_id=cluster_id,
                    workflow_request_id=workflow_id,
                    incident_id=incident_id,
                    step_index=0,
                    fencing_token=1,
                    idempotency_key=(f"{workflow_id}/0/COLLECT_DIAGNOSTIC_BUNDLE"),
                    step=step,
                    workflow=workflow,
                    incident=incident,
                    created_at=now,
                    updated_at=now,
                )
            )
            command_ids.append(command_id)
    finally:
        store.close()

    with psycopg.connect(url) as connection:
        with connection.cursor() as cursor:
            for command_id in command_ids:
                cursor.execute(
                    "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                    (f"remote_command/{command_id}",),
                )
            ready.write_text("ready\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "advisory_locks": "held",
                        "command_count": len(command_ids),
                    }
                ),
                flush=True,
            )
            deadline = time.monotonic() + 600
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(0.25)
            if not release.exists():
                raise RuntimeError("store lock release sentinel was not created")
            for command_id in command_ids:
                cursor.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (f"remote_command/{command_id}",),
                )
    print(json.dumps({"advisory_locks": "released"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
