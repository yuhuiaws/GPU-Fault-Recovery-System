"""Compensation reasons per node.

FINAL-建议汇总 F-C4 (1) (P1-64A). ``_has_unrestored_quiesce`` compared the
global maximum indexes of quiesce and restore steps, so a restore on
*another* node counted as having restored this one. (The F-C6 "stop the
batch at its first failure" change that used to live here was withdrawn:
the owner wants sibling node branches to keep repairing and the failed
branch to escalate in place, which is the planned F-N1.)
"""

from __future__ import annotations

from gpu_fault.execution.executor import ProductionWorkflowExecutor
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from tests._builders import build_store, copy_model, workflow_step
from tests.execution._support import (
    FakeAdapter,
    active_workflow_executor,
    workflow_state,
)

QUARANTINE = WorkflowOperation.QUARANTINE
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES


def _dag(store, outcomes: dict[WorkflowOperation, WorkflowStepOutcome]):
    _, workflow = workflow_state(store, list(outcomes))
    store.save_workflow(copy_model(workflow, dag_enabled=True))
    adapter = FakeAdapter(outcomes)
    executor = active_workflow_executor(store, [adapter], set(outcomes))
    return workflow, adapter, executor


def test_unrestored_quiesce_is_judged_per_node():
    store = build_store()
    _, workflow = workflow_state(store, [QUIESCE, RESTORE])
    per_node = copy_model(
        workflow,
        official_steps=[
            workflow_step(QUIESCE, node_ids=["node-a"]),
            workflow_step(RESTORE, node_ids=["node-b"]),
        ],
        completed_step_indexes=[0, 1],
        completed_operations=[QUIESCE, RESTORE],
    )
    restored = copy_model(
        per_node,
        official_steps=[
            workflow_step(QUIESCE, node_ids=["node-a"]),
            workflow_step(RESTORE, node_ids=["node-a", "node-b"]),
        ],
    )

    assert ProductionWorkflowExecutor._has_unrestored_quiesce(per_node) is True
    assert ProductionWorkflowExecutor._has_unrestored_quiesce(restored) is False
