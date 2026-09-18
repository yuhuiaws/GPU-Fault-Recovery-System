from __future__ import annotations

from datetime import UTC, datetime

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand, RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import fault_incident, workflow_request, workflow_step


def save_command(store, name, *, cluster_id="cluster-local", at=None, **values):
    observed_at = datetime.now(UTC) if at is None else at
    incident = fault_incident(
        f"incident/{name}",
        f"event/{name}",
        cluster_id=cluster_id,
        node_ids=["node-local"],
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=f"workflow/{name}",
        fencing_token=3,
        created_at=observed_at,
        updated_at=observed_at,
    )
    workflow = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=incident.fencing_token,
        official_steps=[
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-local"])
        ],
        created_at=observed_at,
        updated_at=observed_at,
    )
    store.save_incident_and_workflow(incident, workflow)
    return store.ensure_remote_command(
        RemoteActionCommand(
            command_id=name,
            cluster_id=cluster_id,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            fencing_token=workflow.fencing_token,
            step_index=0,
            idempotency_key=f"{workflow.request_id}/0/FREEZE_EVIDENCE",
            step=workflow.official_steps[0],
            workflow=workflow,
            incident=incident,
            created_at=observed_at,
            updated_at=observed_at,
            **values,
        )
    )


def claim_command(store, command, *, lease_seconds=600):
    (claimed,) = store.claim_remote_commands(
        command.cluster_id, "executor", limit=1, lease_seconds=lease_seconds
    )
    assert claimed.command_id == command.command_id, (
        "the lifecycle fixture must claim the intended command"
    )
    return claimed


def complete_command(store, command, status, *, details=None):
    return store.complete_remote_command(
        command.cluster_id,
        command.command_id,
        RemoteCommandResult(
            lease_token=command.lease_token,
            status=status,
            details={} if details is None else details,
            error="local execution failure"
            if status is RemoteCommandStatus.FAILED
            else None,
        ),
    )


def command_in_state(store, name, status):
    command = save_command(store, name)
    if status is RemoteCommandStatus.PENDING:
        return command
    leased = claim_command(store, command)
    if status is RemoteCommandStatus.LEASED:
        return leased
    return complete_command(store, leased, status)
