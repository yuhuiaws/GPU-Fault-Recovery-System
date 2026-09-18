from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.execution.node_action_uncertainty import (
    has_unresolved_node_action,
    unresolved_node_action,
)
from gpu_fault.models import (
    BlockedKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    workflow_is_open,
)
from tests._builders import workflow_request, workflow_step_execution
from tests.execution._cov95_runtime_workflows import FlowHarness

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES


@pytest.mark.parametrize(
    "details",
    [
        {"outcome_unknown": True},
        {"node_action_interrupted": True},
        {"node_action_response_unknown": True},
        {"ownership_permit_delivery_unknown": True},
        {"manual_confirmation_required": True},
        {"node_action_state": "PENDING", "node_action_command_id": "owned/reset"},
        {"outcome_unknown": True, "node_action_not_started": True},
    ],
)
@pytest.mark.parametrize(
    "status", [WorkflowStepStatus.WAITING, WorkflowStepStatus.FAILED]
)
def test_uncertain_physical_record_is_not_a_completion_receipt(
    details: dict[str, Any], status: WorkflowStepStatus
) -> None:
    execution = workflow_step_execution(1, RESET, status, details=details)
    assert unresolved_node_action(execution), "uncertain physical work must stay held"


@pytest.mark.parametrize(
    "details",
    [
        {},
        {"manual_confirmation_required": False},
        {"manual_confirmation_required": True, "node_action_not_started": True},
        {"node_action_state": "PENDING", "node_action_command_id": ""},
        {"node_action_state": "PENDING", "node_action_command_id": None},
        {"node_action_state": "TRANSPORT_RETRY", "node_action_command_id": "owned"},
    ],
)
def test_failure_without_unresolved_physical_work_can_compensate(
    details: dict[str, Any],
) -> None:
    execution = workflow_step_execution(
        1, RESET, WorkflowStepStatus.FAILED, details=details
    )
    assert not unresolved_node_action(execution), (
        "confirmed ordinary failures retain their existing compensation path"
    )


def test_success_read_only_work_and_refused_restore_do_not_self_fence() -> None:
    for operation, status, details in (
        (RESET, WorkflowStepStatus.SUCCEEDED, {}),
        (
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowStepStatus.FAILED,
            {"outcome_unknown": True},
        ),
        (
            RESTORE,
            WorkflowStepStatus.FAILED,
            {"outcome_unknown": True, "restore_gpu_services_withheld": True},
        ),
    ):
        assert not unresolved_node_action(
            workflow_step_execution(1, operation, status, details=details)
        ), "completed, read-only and unsubmitted compensation records do not self-hold"


def test_new_terminal_receipt_supersedes_uncertainty_only_for_the_same_identity() -> (
    None
):
    unknown = workflow_step_execution(
        1,
        RESET,
        WorkflowStepStatus.FAILED,
        phase="official",
        details={"outcome_unknown": True},
    )
    complete = workflow_step_execution(1, RESET, phase="official")
    workflow = workflow_request("owned", "incident")
    workflow = workflow.model_copy(update={"step_executions": [unknown, complete]})
    assert not has_unresolved_node_action(workflow), (
        "a newer completion resolves the same step's older uncertainty"
    )
    workflow = workflow.model_copy(
        update={
            "step_executions": [
                unknown,
                complete.model_copy(update={"phase": "safety"}),
            ]
        }
    )
    assert has_unresolved_node_action(workflow), (
        "a different execution phase cannot resolve the original action"
    )


def test_restore_polling_is_exempt_but_an_unresolved_restore_cannot_release_node() -> (
    None
):
    workflow = workflow_request("owned", "incident").model_copy(
        update={
            "step_executions": [
                workflow_step_execution(
                    2,
                    RESTORE,
                    WorkflowStepStatus.WAITING,
                    details={
                        "node_action_state": "PENDING",
                        "node_action_command_id": "owned/restore",
                    },
                )
            ]
        }
    )
    assert has_unresolved_node_action(workflow), (
        "an unresolved restore cannot justify releasing node occupancy"
    )
    assert not has_unresolved_node_action(workflow, include_restoration=False), (
        "an existing compensation remains eligible for polling"
    )


@pytest.mark.parametrize("dag", [False, True])
def test_unknown_reset_blocks_compensation_and_keeps_node_occupied(dag: bool) -> None:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    h.amend(dag_enabled=dag)
    h.adapter.outcomes[RESET] = WorkflowStepOutcome.failed(
        "node response was unreadable",
        details={"outcome_unknown": True, "manual_confirmation_required": True},
    )
    result = h.execute()
    saved = h.store.get_workflow(h.workflow.request_id)
    assert result.status is WorkflowStatus.BLOCKED
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "operator-blocked physical uncertainty must continue to occupy its node"
    )
    assert [context.step.operation for context in h.adapter.calls] == [QUIESCE, RESET]
    assert RESTORE not in saved.completed_operations
    assert h.execute().status is WorkflowStatus.BLOCKED
    assert [context.step.operation for context in h.adapter.calls] == [QUIESCE, RESET]


@pytest.mark.parametrize("known_refusal", [False, True])
def test_ordinary_failure_and_confirmed_no_start_still_restore_services(
    known_refusal: bool,
) -> None:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    h.adapter.outcomes[RESET] = WorkflowStepOutcome.failed(
        "reset failed",
        details=(
            {
                "manual_confirmation_required": True,
                "node_action_not_started": True,
                "physical_ownership_checks": [],
            }
            if known_refusal
            else {}
        ),
    )
    assert h.execute().status is WorkflowStatus.FAILED
    assert [context.step.operation for context in h.adapter.calls] == [
        QUIESCE,
        RESET,
        RESTORE,
    ]


@pytest.mark.parametrize("flagged_unknown", [False, True])
def test_workflow_deadline_retains_pending_node_identity_and_blocks_restore(
    flagged_unknown: bool,
) -> None:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    details: dict[str, Any] = {
        "node_action_state": "PENDING",
        "node_action_command_id": "owned/reset",
    }
    if flagged_unknown:
        details["outcome_unknown"] = True
    h.amend(
        completed_step_indexes=[0],
        completed_operations=[QUIESCE],
        execution_deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        step_executions=[
            workflow_step_execution(0, QUIESCE, phase="official"),
            workflow_step_execution(
                1, RESET, WorkflowStepStatus.WAITING, phase="official", details=details
            ),
        ],
    )
    result = h.execute()
    saved = h.store.get_workflow(h.workflow.request_id)
    assert result.status is WorkflowStatus.BLOCKED
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert h.adapter.calls == [], "a deadline cannot authorize automatic restoration"
    reset = next(item for item in saved.step_executions if item.step_index == 1)
    assert reset.details["node_action_command_id"] == "owned/reset"
    assert reset.details["outcome_unknown"] is True
    assert "node_action_not_started" not in reset.details
