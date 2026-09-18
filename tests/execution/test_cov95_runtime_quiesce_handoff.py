from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from tests._builders import workflow_step, workflow_step_execution
from tests.execution._cov95_runtime_handoff import (
    QUIESCE,
    REBOOT,
    RESET,
    RESTORE,
    VALIDATE,
    handoff_harness,
)


def test_reset_successor_inherits_quiesce_proof_without_replaying_the_mutation() -> (
    None
):
    h = handoff_harness([QUIESCE, RESET, RESTORE])
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert [call.step.operation for call in h.adapter.calls] == [RESET, RESTORE], (
        h.adapter.calls
    )
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.inherited_step_indexes == [0], saved
    inherited = saved.step_executions[0]
    assert inherited.adapter_operation_id == "remote/quiesce-proof", inherited
    assert inherited.details["quiesced_services"] == ["nvidia-fabricmanager"], inherited
    assert inherited.details["inherited_from_workflow_id"] == "workflow-predecessor", (
        inherited
    )
    again = h.execute()
    assert again.status is WorkflowStatus.SUCCEEDED and len(h.adapter.calls) == 2, again


@pytest.mark.parametrize("dag", [False, True])
def test_reboot_successor_restores_services_before_dependent_validation(
    dag: bool,
) -> None:
    h = handoff_harness([REBOOT, VALIDATE, WorkflowOperation.FREEZE_EVIDENCE])
    if dag:
        h.amend(
            dag_enabled=True,
            official_steps=[
                h.workflow.official_steps[0],
                h.workflow.official_steps[1].model_copy(
                    update={"depends_on_step_indexes": [0]}
                ),
                h.workflow.official_steps[2],
            ],
        )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    operations = [call.step.operation for call in h.adapter.calls]
    assert (
        operations.index(REBOOT)
        < operations.index(RESTORE)
        < operations.index(VALIDATE)
    ), operations
    saved = h.store.get_workflow(h.workflow.request_id)
    cleanup = saved.official_steps[-1]
    assert cleanup.parameters["handoff_from_workflow_id"] == "workflow-predecessor", (
        cleanup
    )
    assert cleanup.parameters["services"] == ["nvidia-fabricmanager"], cleanup
    assert saved.official_steps[1].depends_on_step_indexes == [3], saved.official_steps
    assert saved.official_steps[2].depends_on_step_indexes == ([] if dag else [1]), (
        saved.official_steps
    )
    assert cleanup.depends_on_step_indexes == [0], cleanup


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "wrong-successor",
        "wrong-incident",
        "no-proof",
        "failed-proof",
        "no-completion",
    ],
)
def test_reset_never_inherits_unbound_or_unconfirmed_quiesce_evidence(
    defect: str,
) -> None:
    updates = {
        "wrong-successor": {"preempted_by_workflow_id": "other-workflow"},
        "wrong-incident": {"incident_id": "other-incident"},
        "no-proof": {"step_executions": []},
        "failed-proof": {
            "step_executions": [
                workflow_step_execution(0, QUIESCE, WorkflowStepStatus.FAILED)
            ]
        },
        "no-completion": {"completed_step_indexes": [], "completed_operations": []},
    }
    h = handoff_harness(
        [QUIESCE, RESET, RESTORE], predecessor_updates=updates.get(defect)
    )
    if defect == "missing":
        h.amend(predecessor_workflow_id="unavailable-predecessor")
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert [call.step.operation for call in h.adapter.calls] == [
        QUIESCE,
        RESET,
        RESTORE,
    ], h.adapter.calls
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.inherited_step_indexes == [], saved
    assert saved.quiesce_handoff_from_workflow_id is None, saved


@pytest.mark.parametrize(
    "successor", ["reset-other-node", "reboot-other-node", "observe", "no-restore"]
)
def test_quiesce_handoff_preserves_node_scope_and_cleanup_requirements(
    successor: str,
) -> None:
    operations = (
        [QUIESCE, RESET, RESTORE] if successor == "reset-other-node" else [REBOOT]
    )
    if successor == "observe":
        operations = [WorkflowOperation.FREEZE_EVIDENCE]
    updates = (
        {"official_steps": [workflow_step(QUIESCE), workflow_step(RESET)]}
        if successor == "no-restore"
        else None
    )
    h = handoff_harness(operations, predecessor_updates=updates)
    if successor.endswith("other-node"):
        h.amend(
            official_steps=[
                step.model_copy(update={"node_ids": ["node-b"]})
                for step in h.workflow.official_steps
            ]
        )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert [call.step.operation for call in h.adapter.calls] == operations, (
        h.adapter.calls
    )
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.quiesce_handoff_from_workflow_id is None, saved
    assert len(saved.official_steps) == len(operations), saved


@pytest.mark.parametrize("settling_operation", [RESTORE, REBOOT])
def test_reset_successor_requiesces_after_predecessor_already_settled_quiesce(
    settling_operation: WorkflowOperation,
) -> None:
    h = handoff_harness(
        [QUIESCE, RESET, RESTORE],
        predecessor_updates={
            "official_steps": [
                workflow_step(QUIESCE),
                workflow_step(settling_operation),
            ],
            "completed_step_indexes": [0, 1],
            "completed_operations": [QUIESCE, settling_operation],
            "step_executions": [
                workflow_step_execution(0, QUIESCE, WorkflowStepStatus.SUCCEEDED),
                workflow_step_execution(
                    1, settling_operation, WorkflowStepStatus.SUCCEEDED
                ),
            ],
        },
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert [call.step.operation for call in h.adapter.calls] == [
        QUIESCE,
        RESET,
        RESTORE,
    ], (
        "a restored or rebooted predecessor no longer proves that GPU services are quiesced",
        h.adapter.calls,
    )
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.inherited_step_indexes == [], saved
    assert saved.quiesce_handoff_from_workflow_id is None, saved


@pytest.mark.parametrize("restored_node", ["node-a", "node-b"])
def test_partial_restore_only_hands_off_nodes_that_remain_quiesced(
    restored_node: str,
) -> None:
    h = handoff_harness(
        [QUIESCE, RESET, RESTORE],
        predecessor_updates={
            "official_steps": [
                workflow_step(QUIESCE, node_ids=["node-a", "node-b"]),
                workflow_step(RESTORE, node_ids=[restored_node]),
            ],
            "completed_step_indexes": [0, 1],
            "completed_operations": [QUIESCE, RESTORE],
            "step_executions": [
                workflow_step_execution(0, QUIESCE, WorkflowStepStatus.SUCCEEDED),
                workflow_step_execution(1, RESTORE, WorkflowStepStatus.SUCCEEDED),
            ],
        },
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    inherited = restored_node != "node-a"
    expected = [RESET, RESTORE] if inherited else [QUIESCE, RESET, RESTORE]
    assert [call.step.operation for call in h.adapter.calls] == expected, (
        "only still-quiesced nodes may skip their own quiesce",
        h.adapter.calls,
    )
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.inherited_step_indexes == ([0] if inherited else []), saved


def test_settled_later_branch_does_not_hide_an_earlier_live_quiesce() -> None:
    h = handoff_harness(
        [QUIESCE, RESET, RESTORE],
        predecessor_updates={
            "official_steps": [
                workflow_step(QUIESCE, node_ids=["node-a"]),
                workflow_step(QUIESCE, node_ids=["node-b"]),
                workflow_step(RESTORE, node_ids=["node-b"]),
            ],
            "completed_step_indexes": [0, 1, 2],
            "completed_operations": [QUIESCE, RESTORE],
            "step_executions": [
                workflow_step_execution(0, QUIESCE, WorkflowStepStatus.SUCCEEDED),
                workflow_step_execution(1, QUIESCE, WorkflowStepStatus.SUCCEEDED),
                workflow_step_execution(2, RESTORE, WorkflowStepStatus.SUCCEEDED),
            ],
        },
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert [call.step.operation for call in h.adapter.calls] == [RESET, RESTORE], (
        h.adapter.calls
    )
    assert h.store.get_workflow(h.workflow.request_id).inherited_step_indexes == [0], (
        result
    )
