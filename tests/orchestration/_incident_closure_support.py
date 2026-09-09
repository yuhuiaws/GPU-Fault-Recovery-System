"""Builders shared by the incident-closure tests (service, executor hook, API
route). Kept apart from the service test module so the route and hook tests,
which run on the in-memory store only, are not pulled into the postgres shard
by importing a module that reads ``GPU_FAULT_TEST_POSTGRES_URL``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime.now(timezone.utc).replace(microsecond=0)
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
COLLECT = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
VALIDATE = WorkflowOperation.VALIDATE_GPU
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
QUARANTINE = WorkflowOperation.QUARANTINE
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT
OPERATOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"


def _marker(incident_id: str, node_id: str) -> NodeMarker:
    return NodeMarker(
        marker_id=f"marker-{incident_id}",
        source="test-agent",
        cluster_id="cluster-a",
        trusted=True,
        incident_id=incident_id,
        observed_at=NOW - timedelta(minutes=10),
        expires_at=NOW + timedelta(hours=1),
        scope=MarkerScope(node_ids=[node_id]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.RESET_GPU,
        action_owner="simulated-runtime",
        mapping_version="test-v1",
    )


def _escalated_reset(
    store,
    *,
    incident_id: str = "inc-reset",
    node_ids: tuple[str, ...] = ("node-a",),
    cluster_id: str = "cluster-a",
    workflow_status: WorkflowStatus = WorkflowStatus.FAILED,
    with_marker: bool = True,
):
    """A reset remediation that ran out of lifetime and is with an operator."""

    nodes = list(node_ids)
    workflow = workflow_request(
        f"wf-{incident_id}",
        incident_id,
        status=workflow_status,
        official_steps=[
            workflow_step(operation, node_ids=nodes)
            for operation in (FREEZE, MARK, QUIESCE, RESET, RESTORE)
        ],
        completed_step_indexes=[0, 1, 2],
        completed_operations=[FREEZE, MARK, QUIESCE],
        lifetime_deadline_at=NOW - timedelta(minutes=5),
        created_at=NOW - timedelta(hours=1),
        updated_at=NOW - timedelta(minutes=5),
    )
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        cluster_id=cluster_id,
        node_ids=nodes,
        state=IncidentState.ESCALATED,
        workflow_request_id=workflow.request_id,
        reasons=["reset remediation failed; workflow lifetime exceeded"],
        created_at=NOW - timedelta(hours=1),
        updated_at=NOW - timedelta(minutes=5),
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    if with_marker:
        store.add_marker(_marker(incident_id, nodes[0]))
    return incident, workflow


def _restore_workflow(
    incident_id: str,
    node_ids: tuple[str, ...] = ("node-a",),
    *,
    status: WorkflowStatus = WorkflowStatus.SUCCEEDED,
    operations: tuple[WorkflowOperation, ...] = (VALIDATE, RESTORE),
):
    nodes = list(node_ids)
    return workflow_request(
        f"wf-restore-{incident_id}",
        incident_id,
        status=status,
        official_steps=[
            workflow_step(operation, node_ids=nodes) for operation in operations
        ],
        completed_step_indexes=list(range(len(operations))),
        completed_operations=list(operations),
        created_at=NOW - timedelta(minutes=2),
        updated_at=NOW,
    )


def _recovered(
    incident_id: str,
    workflow,
    node_ids: tuple[str, ...] = ("node-a",),
    *,
    state: IncidentState = IncidentState.RECOVERED,
    cluster_id: str = "cluster-a",
):
    return fault_incident(
        incident_id,
        f"event-{incident_id}",
        cluster_id=cluster_id,
        node_ids=list(node_ids),
        state=state,
        workflow_request_id=workflow.request_id,
    )
