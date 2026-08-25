from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.store import PostgresStore


def main() -> None:
    incident_id = sys.argv[1]
    store = PostgresStore(
        os.environ["GPU_FAULT_STORE_URL"],
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    try:
        incident = store.get_incident(incident_id)
        workflow = store.get_workflow(incident.workflow_request_id)
        now = datetime.now(timezone.utc)
        workflow = workflow.model_copy(
            update={
                "status": WorkflowStatus.PENDING,
                "completed_operations": [],
                "completed_step_indexes": [],
                "superseded_step_indexes": [],
                "step_executions": [],
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "execution_deadline": None,
                "pending_failure_step_index": None,
                "pending_failure_error": None,
                "failure_handled_at": None,
                "blocked_reasons": [],
                "not_before": None,
                "updated_at": now,
            }
        )
        incident = incident.model_copy(
            update={
                "state": IncidentState.ACTION_PENDING,
                "updated_at": now,
            }
        )
        store.save_incident_and_workflow(incident, workflow)
        print(
            "requeued",
            incident.incident_id,
            workflow.request_id,
            workflow.status.value,
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
