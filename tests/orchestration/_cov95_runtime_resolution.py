from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 12, 18, tzinfo=timezone.utc)


def restore_state():
    workflow = workflow_request(
        "blocked-workflow",
        "repaired-incident",
        WorkflowStatus.BLOCKED,
        7,
        source_plan_id="failed-plan",
        execution_epoch=2,
        official_steps=[
            workflow_step(WorkflowOperation.RESET_GPU),
            workflow_step(WorkflowOperation.VALIDATE_GPU),
        ],
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.RESET_GPU],
        blocked_reasons=["validation failed"],
        updated_at=NOW - timedelta(minutes=5),
    )
    operations = [
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    successor = workflow_request(
        "restored-workflow",
        workflow.incident_id,
        WorkflowStatus.SUCCEEDED,
        7,
        official_steps=[workflow_step(operation) for operation in operations],
        completed_step_indexes=list(range(len(operations))),
        completed_operations=operations,
    )
    incident = fault_incident(
        workflow.incident_id,
        "repaired-event",
        state=IncidentState.RECOVERED,
        fencing_token=7,
        workflow_request_id=successor.request_id,
    )
    plan = RecoveryPlan(
        plan_id="failed-plan",
        incident_id=incident.incident_id,
        attempt_id="attempt-a",
        trigger="unit",
        runtime_profile_version="unit-profile",
        steps=[],
        workflow_request_id=workflow.request_id,
        status=PlanStatus.FAILED,
    )
    return SimpleNamespace(
        workflow=workflow,
        incident=incident,
        successor=successor,
        plan=plan,
        commands=[],
    )


def retired_state():
    state = restore_state()
    state.workflow = state.workflow.model_copy(
        update={
            "status": WorkflowStatus.PENDING,
            "completed_operations": [],
            "completed_step_indexes": [],
            "execution_owner_id": "retired-executor",
            "execution_lease_expires_at": NOW + timedelta(minutes=1),
            "remediation_budget_claims": ["node-a"],
        }
    )
    state.incident = state.incident.model_copy(update={"fencing_token": 8})
    state.successor = state.successor.model_copy(update={"fencing_token": 8})
    return state
