"""Completion path: a poisoned event row heals, and a terminal event is
decided immediately whatever the state of its containment workflow; recovery
is chained behind an open containment through ``predecessor_workflow_id``
instead of being held with a 409.

FINAL-建议汇总 F-G2 (P0-62A / P0-62B) and F-G3 (P0-62C).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    DecisionStatus,
    RecoveryAction,
    TerminalEvent,
    WorkflowStatus,
)
from gpu_fault.passive import (
    ACTION_OPERATION,
    PassiveCompileError,
    PassiveWorkflowCompiler,
    operation_for_action,
)
from gpu_fault.service import CompletionService
from gpu_fault.watcher import (
    AllocationCompleteness,
    FailureDetectedEvent,
    failure_containment_ids,
)
from tests._builders import copy_model


def _detect_failure(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> str:
    event = FailureDetectedEvent(
        cluster_id=failed_event.cluster_id,
        job_id=failed_event.job_id,
        attempt_id=failed_event.attempt_id,
        detected_at=ended_at,
        runtime_profile_version=failed_event.runtime_profile_version,
        workload_ids=["training/pytorchjob/distributed-training"],
        node_ids=["node-a", "node-b"],
        gpu_uuids=["GPU-a", "GPU-b"],
        first_failed_rank=0,
        node_id="node-a",
        exit_code=1,
        reason="critical container exited non-zero",
        allocation_completeness=AllocationCompleteness.COMPLETE,
    )
    context.completion.handle_failure_detected(event)
    incident_id, _ = failure_containment_ids(event.event_key)
    containment_id = context.store.get_incident(incident_id).workflow_request_id
    assert containment_id
    return containment_id


def test_event_row_without_decision_is_recoverable_on_redelivery(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    """The poisoned shape of P0-62A: event saved, decision never written.

    A crash between the two writes used to make every redelivery raise
    ``RuntimeError("event exists without completion decision")`` forever.
    """

    assert context.store.save_event_if_absent(failed_event) is True
    assert context.store.get_decision_by_event(failed_event.event_key) is None

    decision = context.completion.handle_terminal(failed_event)

    assert decision.event_key == failed_event.event_key
    assert context.store.get_decision_by_event(failed_event.event_key) is not None


@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatus.PENDING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.FAILED,
        WorkflowStatus.SUPERSEDED,
        WorkflowStatus.SUCCEEDED,
    ],
)
def test_terminal_is_decided_whatever_the_containment_status(
    context: ApplicationContext,
    failed_event: TerminalEvent,
    ended_at: datetime,
    status: WorkflowStatus,
) -> None:
    """An open containment no longer answers 409 (P0-62C's forever-queue):
    the recovery workflow is created at once and sequenced behind it."""

    containment_id = _detect_failure(context, failed_event, ended_at)
    containment = context.store.get_workflow(containment_id)
    context.store.save_workflow(copy_model(containment, status=status))

    completion = CompletionService(
        context.store, workflow_compiler=PassiveWorkflowCompiler(context.store)
    )

    decision = completion.handle_terminal(failed_event)

    assert decision.event_key == failed_event.event_key
    assert decision.status is DecisionStatus.PLAN_CREATED
    plan = context.store.get_plan(decision.recovery_plan_id)
    recovery = context.store.get_workflow(plan.workflow_request_id)
    if status in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}:
        assert recovery.predecessor_workflow_id == containment_id


def test_every_actionable_recovery_action_compiles_to_an_operation() -> None:
    assert set(ACTION_OPERATION) == set(RecoveryAction) - {RecoveryAction.NO_ACTION}


def test_an_unmapped_action_is_a_named_error_not_a_key_error() -> None:
    with pytest.raises(PassiveCompileError, match="NO_ACTION"):
        operation_for_action(RecoveryAction.NO_ACTION)
