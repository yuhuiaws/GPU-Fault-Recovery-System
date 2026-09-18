from __future__ import annotations

import pytest

from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.incident_closure import IncidentClosureService
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step, workflow_step_execution
from tests.orchestration._cov95_closure_extra_support import pair

RESTORE = WorkflowOperation.RESTORE_SCHEDULING


@pytest.mark.parametrize("proof", ["indexes", "receipts", "unproven-dag"])
def test_terminal_closure_requires_successful_release_proof_for_each_node(proof):
    store = InMemoryStore()
    repaired, _ = pair(store, "repaired-a", nodes=("node-a",))
    untouched, _ = pair(store, "unrepaired-b", nodes=("node-b",))
    steps = [
        workflow_step(RESTORE, node_ids=["node-a"]),
        workflow_step(RESTORE, node_ids=["node-b"]),
    ]
    restored, workflow = pair(
        store,
        "restorer",
        state=IncidentState.RECOVERED,
        status=WorkflowStatus.SUCCEEDED,
        workflow_values={
            "dag_enabled": True,
            "dag_revision": 1,
            "official_steps": steps,
            "completed_operations": [RESTORE],
            "completed_step_indexes": [0] if proof == "indexes" else [],
            "step_executions": [
                workflow_step_execution(0, RESTORE, WorkflowStepStatus.SUCCEEDED)
            ]
            if proof == "receipts"
            else [],
        },
    )
    before_b = store.get_incident(untouched.incident_id)
    before_b_workflow = store.get_workflow(untouched.workflow_request_id)
    closed = IncidentClosureService(store).on_terminal(workflow, restored, steps)
    assert closed == ([] if proof == "unproven-dag" else [repaired.incident_id]), (
        "a workflow-level operation name cannot prove every planned release ran",
        proof,
        closed,
    )
    assert store.get_incident(untouched.incident_id) == before_b, (
        "unexecuted node-b release must not close its independent incident"
    )
    assert store.get_workflow(untouched.workflow_request_id) == before_b_workflow, (
        "unproven closure must not write a workflow audit for node-b"
    )
    assert store.list_markers_for_incident(untouched.incident_id)[0].active, (
        "the independent node-b marker must remain active"
    )
