from __future__ import annotations

import pytest

from gpu_fault.execution.config import MAX_DAG_STEPS
from gpu_fault.execution.models import WorkflowStepOutcome, WorkflowStructureError
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from tests._builders import active_workflow_executor, build_store, execute_workflow
from tests.execution._support import workflow_state
from tests.regional.test_distributed_reset_plan_limit import plan


@pytest.mark.parametrize("count", [256, 257])
def test_distributed_reset_shared_executor_limit_is_exact_and_pre_dispatch(count):
    assert MAX_DAG_STEPS == 256
    store = build_store()
    _, workflow = workflow_state(store, [Op.FREEZE_EVIDENCE] * count)
    workflow = workflow.model_copy(update={"dag_enabled": True})
    store.save_workflow(workflow)
    calls = []

    class EvidenceOnly:
        def supports(self, step):
            return step.operation is Op.FREEZE_EVIDENCE

        def execute(self, context):
            calls.append(context.step_index)
            return WorkflowStepOutcome.succeeded()

    executor = active_workflow_executor(store, [EvidenceOnly()], {Op.FREEZE_EVIDENCE})
    if count == 257:
        with pytest.raises(WorkflowStructureError, match="more than the 256"):
            execute_workflow(executor, workflow.request_id)
        assert calls == []
    else:
        result = execute_workflow(executor, workflow.request_id)
        assert result.status is WorkflowStatus.SUCCEEDED
        assert calls == list(range(256))


@pytest.mark.parametrize(
    ("count", "expanded", "status"),
    [(31, 250, WorkflowStatus.PENDING), (32, 258, WorkflowStatus.SAFETY_PENDING)],
)
def test_distributed_reset_compiler_preserves_all_nodes_at_the_expand_boundary(
    count, expanded, status
):
    store, _incident, workflow, nodes = plan(count)
    assert workflow.status is status
    if status is WorkflowStatus.PENDING:
        assert len(workflow.official_steps) == expanded
        assert [
            step.node_ids[0]
            for step in workflow.official_steps
            if step.operation is Op.RESET_GPU
        ] == nodes
    else:
        assert any(
            f"requires {expanded} steps" in text for text in workflow.blocked_reasons
        ), "the refusal must explain the actual expanded plan size"
        assert workflow.safety_only and not workflow.dag_enabled
        assert all(step.node_ids == nodes for step in workflow.safety_steps), (
            "bounded containment must still cover every faulted node"
        )
        assert store.list_remote_commands() == []
