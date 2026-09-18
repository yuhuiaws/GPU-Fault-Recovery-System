from __future__ import annotations

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store._cov95_compat_support import NOW


def workflow_pair(
    name,
    *,
    cluster_id="cluster-local",
    job_id="job-local",
    attempt_id="attempt-local",
    node_ids=None,
    at=NOW,
    status=WorkflowStatus.PENDING,
    incident_state=IncidentState.ACTION_PENDING,
    official_steps=None,
    **workflow_values,
):
    nodes = ["node-local"] if node_ids is None else node_ids
    incident = fault_incident(
        f"incident/{name}",
        f"event/{name}",
        cluster_id=cluster_id,
        job_id=job_id,
        attempt_id=attempt_id,
        node_ids=nodes,
        state=incident_state,
        workflow_request_id=f"workflow/{name}",
        created_at=at,
        updated_at=at,
    )
    workflow = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        status=status,
        fencing_token=incident.fencing_token,
        official_steps=(
            [workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=nodes)]
            if official_steps is None
            else official_steps
        ),
        created_at=at,
        updated_at=at,
        **workflow_values,
    )
    return incident, workflow


def save_pair(store, name, **values):
    incident, workflow = workflow_pair(name, **values)
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow
