"""What an exceeded workflow deadline settles besides the step it fails.

``step_bounds.workflow_deadline_failure`` is the lease holder's enforcement
point, and it carries three obligations the failed step alone does not express:
the node-side remote command the workflow left in flight has to be cancelled
(and the cancellation reported on the outcome, including when the store cannot
perform it), the hard remediation lifetime has to outrank the ordinary
execution deadline so the record goes to an operator instead of another
escalation rung, and the compensation that undoes the workflow's own quiesce
has to stay startable after the deadline. Nothing else may start.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution import (
    ProductionWorkflowExecutor,
    WorkflowStepOutcome,
    step_bounds,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import InMemoryStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
REBOOT = WorkflowOperation.RESTART_NODE
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES


def _overdue_workflow(
    store: InMemoryStore,
    *,
    operation: WorkflowOperation = FREEZE,
    execution_overdue_seconds: int | None = 600,
    lifetime_overdue_seconds: int | None = None,
) -> WorkflowRequest:
    """A leased RUNNING workflow whose deadlines sit where the case needs them.

    ``None`` for either offset leaves that deadline in the future, which is how
    the two verdicts are told apart: the lifetime one has to fire off a deadline
    the execution bound alone would still consider unexpired.
    """

    _, created = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    workflow = copy_model(
        created,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-a",
        execution_epoch=1,
        execution_lease_expires_at=now + timedelta(seconds=180),
        execution_deadline=(
            now + timedelta(seconds=900)
            if execution_overdue_seconds is None
            else now - timedelta(seconds=execution_overdue_seconds)
        ),
        lifetime_deadline_at=(
            None
            if lifetime_overdue_seconds is None
            else now - timedelta(seconds=lifetime_overdue_seconds)
        ),
    )
    # Deliberate out-of-band lease: ``expected`` names the row as read
    # (store review 2026-09-07, item B).
    store.save_workflow(workflow, expected=created)
    return workflow


def _executor(
    store: InMemoryStore, adapter: FakeAdapter | None = None
) -> ProductionWorkflowExecutor:
    outcomes = {FREEZE: WorkflowStepOutcome.succeeded()}
    return active_workflow_executor(
        store,
        [FakeAdapter(outcomes) if adapter is None else adapter],
        {FREEZE, QUIESCE, REBOOT, RESTORE},
    )


def _in_flight_command(
    store: InMemoryStore, workflow: WorkflowRequest, command_id: str
) -> None:
    incident = store.get_incident(workflow.incident_id)
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id=command_id,
            cluster_id=incident.cluster_id,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=workflow.fencing_token,
            idempotency_key=f"{workflow.request_id}/0/{FREEZE.value}",
            step=workflow.official_steps[0],
            workflow=workflow,
            incident=incident,
            status=RemoteCommandStatus.WAITING,
        )
    )


def test_the_deadline_cancels_the_in_flight_remote_command_it_reports() -> None:
    store = build_store()
    workflow = _overdue_workflow(store)
    _in_flight_command(store, workflow, "cmd-in-flight")

    outcome = step_bounds.workflow_deadline_failure(
        _executor(store), workflow, workflow.official_steps[0], 0
    )

    assert outcome is not None, "an overdue workflow must fail its next step"
    assert outcome.details["workflow_deadline_remote_command_cancellation"] == {
        "cancelled": 1,
        "cancellation_requested": 0,
    }
    command = store.get_remote_command("cmd-in-flight")
    assert command.status is RemoteCommandStatus.FAILED
    assert command.status_source == "workflow-timeout"


def test_the_deadline_still_lands_when_the_store_cannot_cancel() -> None:
    def _refuse(workflow_request_id: str, *, reason: str) -> dict[str, int]:
        raise RuntimeError("remote command table is unavailable")

    store = build_store(cancel_remote_commands_for_workflow=_refuse)
    workflow = _overdue_workflow(store)

    outcome = step_bounds.workflow_deadline_failure(
        _executor(store), workflow, workflow.official_steps[0], 0
    )

    assert outcome is not None, "a failed cancellation must not swallow the deadline"
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["workflow_deadline_remote_command_cancellation"] == (
        "RuntimeError: remote command table is unavailable"
    )


def test_an_expired_lifetime_outranks_an_unexpired_execution_deadline() -> None:
    store = build_store()
    workflow = _overdue_workflow(
        store, execution_overdue_seconds=None, lifetime_overdue_seconds=30
    )

    outcome = step_bounds.workflow_deadline_failure(
        _executor(store), workflow, workflow.official_steps[0], 0
    )

    assert outcome is not None, "an expired lifetime must fail the step on its own"
    assert outcome.details["workflow_lifetime_exceeded"] is True
    assert outcome.details["workflow_execution_deadline"] == (
        workflow.lifetime_deadline_at.isoformat()
    )
    assert "workflow lifetime" in (outcome.error or "")


def test_an_expired_lifetime_counts_the_workflow_once() -> None:
    store = build_store()
    workflow = _overdue_workflow(
        store, execution_overdue_seconds=600, lifetime_overdue_seconds=30
    )
    executor = _executor(store)
    before = executor.lifetime_exceeded_total

    step_bounds.workflow_deadline_failure(
        executor, workflow, workflow.official_steps[0], 0
    )

    assert executor.lifetime_exceeded_total == before + 1


def test_only_the_execution_deadline_passing_is_not_a_lifetime_failure() -> None:
    store = build_store()
    workflow = _overdue_workflow(store, execution_overdue_seconds=600)
    executor = _executor(store)
    before = executor.lifetime_exceeded_total

    outcome = step_bounds.workflow_deadline_failure(
        executor, workflow, workflow.official_steps[0], 0
    )

    assert outcome is not None, "an overdue execution deadline must fail the step"
    assert outcome.details["workflow_lifetime_exceeded"] is False
    assert "workflow execution deadline" in (outcome.error or "")
    assert executor.lifetime_exceeded_total == before


def test_the_restore_the_workflow_owes_is_exempt_from_the_deadline() -> None:
    store = build_store()
    workflow = _overdue_workflow(store, operation=RESTORE, lifetime_overdue_seconds=30)

    outcome = step_bounds.workflow_deadline_failure(
        _executor(store), workflow, workflow.official_steps[0], 0
    )

    assert outcome is None, "RESTORE_GPU_SERVICES must stay startable past the deadline"


def test_a_workflow_past_its_deadline_starts_only_the_restore_it_owes() -> None:
    store = build_store()
    _, created = workflow_state(store, [QUIESCE, REBOOT, RESTORE])
    now = datetime.now(timezone.utc)
    workflow = copy_model(
        created,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-a",
        execution_epoch=1,
        execution_lease_expires_at=now + timedelta(seconds=180),
        execution_deadline=now - timedelta(seconds=600),
        completed_step_indexes=[0],
        completed_operations=[QUIESCE],
        step_executions=[workflow_step_execution(0, QUIESCE)],
    )
    store.save_workflow(workflow, expected=created)
    adapter = FakeAdapter(
        {
            QUIESCE: WorkflowStepOutcome.succeeded(),
            REBOOT: WorkflowStepOutcome.succeeded(),
            RESTORE: WorkflowStepOutcome.succeeded(),
        }
    )

    result = execute_workflow(_executor(store, adapter), workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == [f"{workflow.request_id}/2/{RESTORE.value}"]
    final = store.get_workflow(workflow.request_id)
    assert final.status is WorkflowStatus.FAILED
    assert [item.status for item in final.step_executions if item.step_index == 2] == [
        WorkflowStepStatus.SUCCEEDED
    ]
