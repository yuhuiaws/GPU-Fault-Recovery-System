"""Which step set executes is an explicit flag, not "blocked_reasons is
non-empty".

FINAL-建议汇总 F-C8 (2) (P0-45B, P0-61A, P0-53A). ``is_safety`` was derived
from ``bool(workflow.blocked_reasons)``, so a single warning in that list
switched the executor -- and four other readers -- to ``safety_steps``. A
PENDING record with a warning and no safety steps then had "no executable
steps" (or, on a retry path, zero steps and an instant SUCCEEDED).
"""

from __future__ import annotations

from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from tests._builders import build_store, copy_model, workflow_step
from tests.execution._support import (
    FakeAdapter,
    WorkflowStepOutcome,
    active_workflow_executor,
    workflow_state,
)

OP = WorkflowOperation.QUARANTINE


def test_safety_only_defaults_false_and_follows_the_safety_pending_status():
    store = build_store()
    _, workflow = workflow_state(store, [OP])

    assert workflow.safety_only is False
    assert workflow.executes_safety_steps is False
    assert copy_model(
        workflow, status=WorkflowStatus.SAFETY_PENDING
    ).executes_safety_steps, (
        "expected copy_model( workflow, status=WorkflowStatus.SAFETY_PENDING ).executes_safety_steps to be true"
    )
    assert copy_model(workflow, safety_only=True).executes_safety_steps, (
        "expected copy_model(workflow, safety_only=True).executes_safety_steps to be true"
    )
    # A warning alone does not flip the step set any more.
    assert (
        copy_model(workflow, blocked_reasons=["warning"]).executes_safety_steps is False
    )


def test_a_warning_in_blocked_reasons_does_not_switch_to_the_empty_safety_steps():
    store = build_store()
    _, workflow = workflow_state(store, [OP])
    store.save_workflow(
        copy_model(workflow, blocked_reasons=["evidence: dmesg capture degraded"])
    )
    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    executor = active_workflow_executor(store, [adapter], {OP})

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    assert len(adapter.calls) == 1
    assert store.get_workflow(workflow.request_id).completed_operations == [OP]


def test_a_safety_only_workflow_runs_its_safety_steps_while_running():
    """Mid-run the status is RUNNING, so the flag -- not the status -- has to
    carry the answer."""

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    store.save_workflow(
        copy_model(
            workflow,
            status=WorkflowStatus.RUNNING,
            safety_only=True,
            safety_steps=[workflow_step(OP)],
            blocked_reasons=[],
        )
    )
    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    executor = active_workflow_executor(store, [adapter], {OP})

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    assert result.status is WorkflowStatus.BLOCKED
    assert len(adapter.calls) == 1
