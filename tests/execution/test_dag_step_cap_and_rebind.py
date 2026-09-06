"""DAG execution bounds and the rebind refresh (F-C6).

A DAG grows by appended branches and in-place escalation rungs; the validator
and the ready-set scan are both quadratic in its size, so a graph with no
upper bound is a runaway workflow's way of pinning a dispatcher worker. The
cap is checked where the shape is validated. Separately, a node replacement
rewrites the later steps to the spare; the step that runs next in the same
pass must be read from the rewritten list, not the one captured before it.
"""

from __future__ import annotations

import pytest

from gpu_fault.execution import (
    WorkflowExecutionError,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.execution.executor import MAX_DAG_STEPS
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.store import InMemoryStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step,
)
from tests.execution._support import workflow_state


class _NodeRecordingAdapter:
    def __init__(self, outcomes: dict[WorkflowOperation, WorkflowStepOutcome]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[WorkflowOperation, list[str]]] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a" and step.operation in self.outcomes

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append((context.step.operation, list(context.step.node_ids)))
        return self.outcomes[context.step.operation]


def _chain_dag(store: InMemoryStore, length: int) -> WorkflowRequest:
    _, workflow = workflow_state(store, [WorkflowOperation.FREEZE_EVIDENCE])
    steps = [
        workflow_step(
            WorkflowOperation.FREEZE_EVIDENCE,
            depends_on_step_indexes=[index - 1] if index else [],
        )
        for index in range(length)
    ]
    workflow = copy_model(workflow, dag_enabled=True, official_steps=steps)
    store.save_workflow(workflow)
    return workflow


def test_a_dag_over_the_step_cap_is_rejected_before_any_adapter_runs() -> None:
    store = build_store()
    workflow = _chain_dag(store, MAX_DAG_STEPS + 1)
    adapter = _NodeRecordingAdapter(
        {WorkflowOperation.FREEZE_EVIDENCE: WorkflowStepOutcome.succeeded()}
    )
    executor = active_workflow_executor(
        store, [adapter], {WorkflowOperation.FREEZE_EVIDENCE}
    )

    with pytest.raises(WorkflowExecutionError, match=str(MAX_DAG_STEPS)):
        execute_workflow(executor, workflow.request_id)

    assert adapter.calls == []


def test_a_dag_at_the_step_cap_still_runs() -> None:
    store = build_store()
    workflow = _chain_dag(store, MAX_DAG_STEPS)
    adapter = _NodeRecordingAdapter(
        {WorkflowOperation.FREEZE_EVIDENCE: WorkflowStepOutcome.succeeded()}
    )
    executor = active_workflow_executor(
        store, [adapter], {WorkflowOperation.FREEZE_EVIDENCE}
    )

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert len(adapter.calls) == MAX_DAG_STEPS


def _rebind_adapter() -> _NodeRecordingAdapter:
    return _NodeRecordingAdapter(
        {
            WorkflowOperation.REPLACE_NODE: WorkflowStepOutcome.succeeded(
                details={"node_rebindings": {"node-a": "node-spare"}}
            ),
            WorkflowOperation.VALIDATE_GPU: WorkflowStepOutcome.succeeded(),
        }
    )


def test_a_dag_step_after_a_rebind_runs_on_the_spare_node() -> None:
    store = build_store()
    _, workflow = workflow_state(
        store, [WorkflowOperation.REPLACE_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        official_steps=[
            workflow.official_steps[0],
            copy_model(workflow.official_steps[1], depends_on_step_indexes=[0]),
        ],
    )
    store.save_workflow(workflow)
    adapter = _rebind_adapter()
    executor = active_workflow_executor(store, [adapter], adapter.outcomes)

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == [
        (WorkflowOperation.REPLACE_NODE, ["node-a"]),
        (WorkflowOperation.VALIDATE_GPU, ["node-spare"]),
    ]


def test_a_sequential_step_after_a_rebind_runs_on_the_spare_node() -> None:
    store = build_store()
    _, workflow = workflow_state(
        store, [WorkflowOperation.REPLACE_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    adapter = _rebind_adapter()
    executor = active_workflow_executor(store, [adapter], adapter.outcomes)

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == [
        (WorkflowOperation.REPLACE_NODE, ["node-a"]),
        (WorkflowOperation.VALIDATE_GPU, ["node-spare"]),
    ]
