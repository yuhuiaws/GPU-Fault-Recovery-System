from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from uuid import uuid4

from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.store import PostgresStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a validation-first workflow that restores scheduling "
            "for one quarantined node."
        )
    )
    parser.add_argument("incident_id")
    parser.add_argument("node_id")
    parser.add_argument("reason")
    arguments = parser.parse_args()
    incident_id = arguments.incident_id
    node_id = arguments.node_id
    reason = arguments.reason
    store = PostgresStore(
        os.environ["GPU_FAULT_STORE_URL"],
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    try:
        incident = store.get_incident(incident_id)
        if node_id not in incident.node_ids:
            raise ValueError(f"{node_id} is outside incident {incident_id}")
        if incident.state not in {
            IncidentState.QUARANTINED,
            IncidentState.ESCALATED,
        }:
            raise ValueError(
                "incident is not quarantined/escalated after a failed "
                f"validation: {incident.state.value}"
            )
        if incident.workflow_request_id:
            current = store.get_workflow(incident.workflow_request_id)
            if current.status in {
                WorkflowStatus.PENDING,
                WorkflowStatus.RUNNING,
                WorkflowStatus.SAFETY_PENDING,
            }:
                raise ValueError(
                    f"incident still has active workflow {current.request_id}"
                )
        now = datetime.now(timezone.utc)
        workflow = WorkflowRequest(
            request_id=f"workflow-validated-restore-{uuid4()}",
            incident_id=incident.incident_id,
            runtime_profile_version="hyperpod-v1",
            status=WorkflowStatus.PENDING,
            official_action="RESTORE_SCHEDULING",
            fencing_token=incident.fencing_token,
            official_steps=[
                WorkflowStepSpec(
                    operation=operation,
                    execution_owner=(
                        "gpu-fault-kubernetes-adapter"
                        if operation is WorkflowOperation.RESTORE_SCHEDULING
                        else "gpu-fault-validation-adapter"
                    ),
                    node_ids=[node_id],
                )
                for operation in (
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowOperation.VALIDATE_HOST,
                    WorkflowOperation.VALIDATE_FABRIC,
                    WorkflowOperation.RESTORE_SCHEDULING,
                )
            ],
            created_at=now,
            updated_at=now,
        )
        incident = incident.model_copy(
            update={
                "state": IncidentState.ACTION_PENDING,
                "workflow_request_id": workflow.request_id,
                "reasons": [*incident.reasons, reason],
                "updated_at": now,
            }
        )
        store.save_incident_and_workflow(incident, workflow)
        print(workflow.request_id)
    finally:
        store.close()


if __name__ == "__main__":
    main()
