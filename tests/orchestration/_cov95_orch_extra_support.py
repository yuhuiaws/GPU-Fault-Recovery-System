from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.app.context import default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.store import InMemoryStore
from tests._builders import (
    attempt_observation,
    container_observation,
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
)

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


def memory_store():
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    return store


def observation(*, ranks=2, **updates):
    values = {
        "containers": [
            container_observation(
                f"pod-{rank}",
                f"worker-{rank}",
                rank,
                f"node-{rank}",
                gpu_uuids=[f"GPU-{rank}"],
            )
            for rank in range(ranks)
        ],
        "workload_ids": ["training/job/unit-job"],
        "expected_critical_ranks": ranks,
        **updates,
    }
    return attempt_observation(
        values.pop("job_id", "unit-job"),
        values.pop("attempt_id", "unit-attempt"),
        values.pop("observed_at", NOW),
        **values,
    )


def finding(event_id="unit-finding", **updates):
    values = {
        "observed_at": NOW,
        "category": NodeHealthCategory.GPU,
        "severity": "warning",
        "reason": "unit health observation",
        "recommended_action": RecoveryAction.RUN_DIAGNOSTICS,
        "runtime_profile_version": "simulated-v1",
        "workload_state": WorkloadState.IDLE,
        **updates,
    }
    return node_health_finding(f"finding-{event_id}", event_id, **values)


def stored_workflow(
    store,
    operations,
    *,
    identity="unit-repair",
    node_ids=None,
    incident_updates=None,
    workflow_updates=None,
):
    nodes = ["node-0"] if node_ids is None else list(node_ids)
    incident_values = {
        "state": IncidentState.ACTION_PENDING,
        "workflow_request_id": f"wf-{identity}",
        "fencing_token": 1,
        **(incident_updates or {}),
    }
    incident = fault_incident(
        f"inc-{identity}", f"event-{identity}", node_ids=nodes, **incident_values
    )
    workflow_values = {
        "status": WorkflowStatus.RUNNING,
        "fencing_token": 1,
        "runtime_profile_version": "simulated-v1",
        "official_steps": [
            workflow_step(operation, node_ids=nodes) for operation in operations
        ],
        "execution_owner_id": "unit-executor",
        "execution_lease_expires_at": NOW + timedelta(minutes=10),
        **(workflow_updates or {}),
    }
    workflow = workflow_request(
        f"wf-{identity}", incident.incident_id, **workflow_values
    )
    store.save_incident_and_workflow(incident, workflow)
    return store.get_incident(incident.incident_id), store.get_workflow(
        workflow.request_id
    )
