from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration import DagBrancher, RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


@pytest.fixture(name="closure_store", params=["memory", "sqlite"])
def closure_store_fixture(request, tmp_path):
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "closure-extra.db"))
    )
    try:
        yield store
    finally:
        if request.param == "sqlite":
            store.close()


def pair(
    store,
    identity="unit-incident",
    *,
    nodes=("node-a",),
    state=IncidentState.ESCALATED,
    status=WorkflowStatus.FAILED,
    workflow_values=None,
    incident_values=None,
    marker=True,
):
    values = {
        "fencing_token": 1,
        "official_steps": [
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=list(nodes))
        ],
        "created_at": NOW,
        "updated_at": NOW,
        **(workflow_values or {}),
    }
    workflow = workflow_request(f"wf-{identity}", identity, status=status, **values)
    incident = fault_incident(
        identity,
        f"event-{identity}",
        node_ids=list(nodes),
        state=state,
        workflow_request_id=workflow.request_id,
        fencing_token=1,
        **(incident_values or {}),
    )
    store.save_incident_and_workflow(incident, workflow)
    if marker:
        store.add_marker(
            NodeMarker(
                marker_id=f"marker-{identity}",
                source="unit-test",
                cluster_id=incident.cluster_id,
                trusted=True,
                incident_id=identity,
                observed_at=NOW,
                expires_at=NOW + timedelta(days=1),
                scope=MarkerScope(node_ids=list(nodes)),
                severity=Severity.CRITICAL,
                recommended_action=RecoveryAction.RESET_GPU,
                mapping_version="unit-v1",
            )
        )
    return store.get_incident(identity), store.get_workflow(workflow.request_id)


def merger():
    arbiter = RecoveryArbiter()
    return WorkflowMergeService(
        arbiter,
        DagBrancher(arbiter),
        preemption_enabled=True,
        workload_scoped_operations=set(WORKLOAD_SCOPED_OPERATIONS),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=NodeConflictService.workflow_resource_claims_by_node,
    )
