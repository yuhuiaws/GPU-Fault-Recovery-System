"""Dispatcher terminalization writes audit where it belongs, and picks the
workflow its incident is waiting on.

FINAL-建议汇总 F-C8 (P0-61A / P0-45B / P0-53A), F-A5 (P0-46C) and F-C1 /
F-A2 (P1-79E, the 2026-09-04 starvation). ``blocked_reasons`` is the switch
that makes the executor read ``safety_steps`` instead of ``official_steps``;
the dispatcher used to append human-readable audit text to it when it
terminalized a record, which flipped the executable step set of anything a
retry or an operator later revived, and made the reservation release scan an
empty list. The watchdog's FAILED carried no step execution at all, so the
escalation classifier had nothing to classify.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state

NOW = datetime(2026, 9, 5, 17, 0, tzinfo=timezone.utc)


def _dispatcher(store, operations):
    adapter = FakeAdapter({op: WorkflowStepOutcome.succeeded() for op in operations})
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], operations),
        WorkflowDispatcherConfig(enabled=True, batch_size=100, max_workers=1),
    )


def test_watchdog_failure_records_a_failed_step_execution_not_a_blocked_reason():
    store = build_store()
    operations = [WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.RESET_GPU]
    _, workflow = workflow_state(store, operations)
    now = datetime.now(timezone.utc)
    store.save_workflow(
        copy_model(
            workflow,
            status=WorkflowStatus.RUNNING,
            completed_step_indexes=[0],
            execution_owner_id=None,
            execution_lease_expires_at=None,
            execution_deadline=now - timedelta(seconds=600),
        )
    )

    _dispatcher(store, operations).run_once()

    failed = store.get_workflow(workflow.request_id)
    assert failed.status is WorkflowStatus.FAILED
    assert failed.blocked_reasons == []
    executions = [
        e for e in failed.step_executions if e.status is WorkflowStepStatus.FAILED
    ]
    assert [(e.step_index, e.operation) for e in executions] == [
        (1, WorkflowOperation.RESET_GPU)
    ]
    assert "deadline" in (executions[0].error or "")
    assert "workflow_deadline_remote_command_cancellation" in executions[0].details


def test_abandoned_generation_supersession_leaves_blocked_reasons_alone():
    store = build_store()
    incident = fault_incident(
        "inc-gen",
        "event-gen",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-current",
        fencing_token=4,
        created_at=NOW,
        updated_at=NOW,
    )
    abandoned = workflow_request(
        "wf-abandoned",
        "inc-gen",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        created_at=NOW,
        updated_at=NOW,
    )
    current = workflow_request(
        "wf-current",
        "inc-gen",
        status=WorkflowStatus.PENDING,
        fencing_token=4,
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        created_at=NOW + timedelta(minutes=3),
        updated_at=NOW + timedelta(minutes=3),
    )
    store.save_incident_and_workflow(incident, current)
    store.save_workflow(abandoned)

    _dispatcher(store, [WorkflowOperation.FREEZE_EVIDENCE]).run_once()

    superseded = store.get_workflow("wf-abandoned")
    assert superseded.status is WorkflowStatus.SUPERSEDED
    assert superseded.blocked_reasons == []
    assert "superseded by wf-current" in (superseded.preemption_reason or "")


class _RecordingExecutor:
    def __init__(self) -> None:
        self.requested: list[str] = []
        from types import SimpleNamespace

        # The sweep reads the executor's RESTART_WORKLOAD waiting cap when it
        # releases a superseded record's reservations (F-C9).
        self.config = SimpleNamespace(
            executor_id="executor-recording", step_waiting_limit=lambda operation: 600
        )

    def execute(self, request_id, request):
        self.requested.append(request_id)
        raise WorkflowLeaseError("recorded only")


def test_incident_dedupe_dispatches_the_workflow_the_incident_points_at():
    """The 2026-09-04 shape: an older same-incident record starving the real one.

    Both are PENDING, neither links to the other, the incident names the
    newer one. Choosing by ``updated_at`` picked the stale record every tick
    for five hours; the incident pointer is the tie-breaker.
    """

    store = build_store()
    incident = fault_incident(
        "inc-twin",
        "event-twin",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-current",
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    stale = workflow_request(
        "wf-stale",
        "inc-twin",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        created_at=NOW,
        updated_at=NOW,
    )
    current = workflow_request(
        "wf-current",
        "inc-twin",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        created_at=NOW + timedelta(minutes=1),
        updated_at=NOW + timedelta(minutes=1),
    )
    store.save_incident_and_workflow(incident, current)
    store.save_workflow(stale)
    executor = _RecordingExecutor()
    dispatcher = WorkflowDispatcher(
        store,
        executor,  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, batch_size=100, max_workers=1),
    )

    dispatcher.run_once()

    assert executor.requested == ["wf-current"]
