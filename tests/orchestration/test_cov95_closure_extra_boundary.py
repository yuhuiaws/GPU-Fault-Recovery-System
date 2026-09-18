from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.orchestration.preemption_boundary import (
    WaitingVerdict,
    preemption_boundary,
)
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.orchestration._cov95_closure_extra_safety import (
    closure_extra_isolation as closure_extra_isolation,
)

RESET = WorkflowOperation.RESET_GPU
VALIDATE = WorkflowOperation.VALIDATE_GPU
FREEZE = WorkflowOperation.FREEZE_EVIDENCE


def test_empty_or_fully_completed_scope_has_no_supersedable_step():
    for count in (0, 2):
        workflow = workflow_request(
            "boundary",
            "incident",
            official_steps=[workflow_step(FREEZE) for _ in range(count)],
            completed_step_indexes=list(range(count)),
        )
        boundary = preemption_boundary(workflow)
        assert boundary.open is True, "a settled scope invented a blocking wait"
        assert boundary.first_supersedable is None, "a completed step was replaceable"
        assert boundary.replaceable == frozenset(), (
            "settled indexes remained replaceable"
        )
        assert boundary.reason == "every waiting step can be stopped", (
            "an empty waiting population produced a failure reason"
        )


def test_open_boundary_names_first_pending_step_and_keeps_completed_predecessor():
    workflow = workflow_request(
        "boundary",
        "incident",
        official_steps=[
            workflow_step(FREEZE),
            workflow_step(VALIDATE),
            workflow_step(RESET),
        ],
        completed_step_indexes=[0],
        step_executions=[
            workflow_step_execution(1, VALIDATE, WorkflowStepStatus.WAITING)
        ],
    )
    boundary = preemption_boundary(workflow)
    assert boundary.first_supersedable == 1, (
        "the first replaceable validation was not selected"
    )
    assert boundary.predecessor == 0, "completed predecessor was not preserved"
    assert boundary.protected == frozenset({0}), (
        "a stoppable wait incorrectly protected its index"
    )
    assert boundary.waiting[0].verdict is WaitingVerdict.STOPPABLE, (
        "the local read-only validation wait became noncancelable"
    )


def test_branch_scope_does_not_consume_another_branches_blocking_wait():
    workflow = workflow_request(
        "boundary",
        "incident",
        official_steps=[
            workflow_step(FREEZE),
            workflow_step(RESET),
            workflow_step(VALIDATE),
        ],
        completed_step_indexes=[0],
        step_executions=[
            workflow_step_execution(
                1,
                RESET,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/other-branch",
                details={"remote_status": "RUNNING"},
            )
        ],
    )
    whole = preemption_boundary(workflow)
    branch = preemption_boundary(workflow, indexes=[0, 2])
    assert whole.open is False, (
        "an in-flight destructive command did not hold the workflow"
    )
    assert whole.first_supersedable is None, (
        "a closed whole-workflow boundary offered preemption"
    )
    assert branch.open is True and branch.first_supersedable == 2, (
        "an unrelated branch wait blocked the selected branch"
    )
    assert branch.waiting == (), (
        "out-of-scope waiting records leaked into the branch verdict"
    )


@pytest.mark.parametrize("declared", [None, "", 4])
def test_cancellable_remote_record_uses_operation_id_when_its_explicit_id_is_unusable(
    declared,
):
    workflow = workflow_request(
        "boundary",
        "incident",
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(RESET)],
        step_executions=[
            workflow_step_execution(
                0,
                RESET,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/unit-command",
                details={"remote_status": "PENDING", "remote_command_id": declared},
            )
        ],
    )
    available = preemption_boundary(workflow)
    unavailable = preemption_boundary(workflow, remote_cancellation_available=False)
    assert [item.remote_command_id for item in available.cancellations] == [
        "unit-command"
    ], "a cancellable command lost its usable transport identifier"
    assert available.first_supersedable == 0, (
        "available cancellation did not open the boundary"
    )
    assert unavailable.blocking == frozenset({0}), (
        "planning silently cancelled a remote command"
    )
    assert unavailable.first_supersedable is None, (
        "a caller without cancellation could preempt"
    )


@pytest.mark.parametrize("declared", [None, "explicit-command"])
def test_remote_prefix_without_a_suffix_requires_an_explicit_command_identity(declared):
    workflow = workflow_request(
        "boundary",
        "incident",
        official_steps=[workflow_step(RESET)],
        step_executions=[
            workflow_step_execution(
                0,
                RESET,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/",
                details={"remote_status": "PENDING", "remote_command_id": declared},
            )
        ],
    )
    boundary = preemption_boundary(workflow)
    assert boundary.open is (declared is not None), (
        "command identity did not gate cancellation"
    )
    if declared is None:
        assert boundary.cancellations == (), (
            "an unnamed command was offered for cancellation"
        )
        assert "names no remote command" in boundary.reason, (
            "missing command identity lost its refusal"
        )
    else:
        assert boundary.cancellations[0].remote_command_id == declared, (
            "the explicit remote identity was discarded"
        )
