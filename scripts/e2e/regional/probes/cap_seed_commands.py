"""Seed isolated non-destructive remote commands for CAP-004."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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


def main() -> int:
    cluster_id = "cap-cluster-000"
    count = 25
    store = PostgresStore(
        Path("/work/store-url").read_text(encoding="utf-8"),
        initialize_schema=False,
        pool_min_size=0,
        pool_max_size=2,
    )
    now = datetime.now(timezone.utc)
    try:
        for index in range(count):
            incident_id = f"incident-cap004-{index:03d}"
            workflow_id = f"workflow-cap004-{index:03d}"
            command_id = f"command-cap004-{index:03d}"
            step = WorkflowStepSpec(
                operation=WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                execution_owner="gpu-fault-node-agent",
                node_ids=[f"node-cap004-{index:03d}"],
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
                event_id=f"event-cap004-{index:03d}",
                event_type="CAPACITY_TEST",
                cluster_id=cluster_id,
                node_ids=step.node_ids,
                policy_version="capacity-v1",
                policy_source="CONTROLLED_DRILL",
                official_action="CAPACITY_DIAGNOSTIC",
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=workflow_id,
                drill_id="cap004",
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
    finally:
        store.close()
    print(json.dumps({"cluster_id": cluster_id, "commands_created": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
