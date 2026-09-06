"""Two merge-path guards that are each one line.

FINAL-建议汇总 F-B6 (P0-57C b) and F-B4 (P0-56A). Widening a workflow with an
empty GPU mapping used to rewrite every node step's ``gpu_uuids`` to ``[]``,
which the node-action adapter then refused; and a BLOCKED workflow was reused
as the merge target because only SUCCEEDED / FAILED / SUPERSEDED counted as
"terminal", so a fault was absorbed into a record that will never execute.
"""

from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from tests._builders import (
    build_context,
    fault_incident,
    workflow_request,
    workflow_step,
)


def test_widening_with_no_gpu_mapping_leaves_the_steps_untouched() -> None:
    context = build_context()
    workflow = workflow_request(
        "wf-widen",
        "inc-widen",
        official_steps=[
            workflow_step(
                WorkflowOperation.RESET_GPU,
                node_ids=["node-a"],
                gpu_uuids=["GPU-a"],
                parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"]}},
            ),
            workflow_step(
                WorkflowOperation.VALIDATE_GPU, node_ids=["node-a"], gpu_uuids=["GPU-a"]
            ),
        ],
    )

    widened = context.orchestrator._widen_node_action_scope(workflow, {})

    assert widened.official_steps == workflow.official_steps


@pytest.mark.parametrize(
    "status", [WorkflowStatus.BLOCKED, WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED]
)
def test_reopen_if_terminal_never_reuses_a_non_executable_workflow(status) -> None:
    incident = fault_incident("inc-x", "event-x", workflow_request_id="wf-x")
    workflow = workflow_request("wf-x", "inc-x", status=status)

    assert NodeConflictService.reopen_if_terminal(incident, workflow) == (None, None)


@pytest.mark.parametrize(
    "status",
    [WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING],
)
def test_reopen_if_terminal_keeps_an_executable_workflow(status) -> None:
    incident = fault_incident("inc-x", "event-x", workflow_request_id="wf-x")
    workflow = workflow_request("wf-x", "inc-x", status=status)

    assert NodeConflictService.reopen_if_terminal(incident, workflow) == (
        incident,
        workflow,
    )
