"""Dispatcher gates and mirrors that only act on a second look.

A predecessor row that retention removed is treated as finished, a workflow
whose incident row is gone has no other remediation to wait for, a job-scoped
incident adds its attempt to the processor-queue scope, a FAILED outcome under
a pending preemption is left to the successor, a handler that ran against a
row no longer FAILED stamps nothing, and a plan already in the mirrored status
is not rewritten.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryAction,
    RecoveryPlan,
    WorkflowExecutionResult,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from tests._builders import build_store, fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


class _OutcomeExecutor:
    """Reports a fixed outcome for every row without touching the store."""

    def __init__(self, status: WorkflowStatus) -> None:
        self.status = status
        self.requested: list[str] = []
        self.config = SimpleNamespace(executor_id="executor-outcome")

    def execute(self, request_id: str, request: Any) -> WorkflowExecutionResult:
        self.requested.append(request_id)
        return WorkflowExecutionResult(
            workflow_request_id=request_id,
            incident_id=f"inc-{request_id}",
            status=self.status,
            completed_operations=[],
            error="fixed outcome" if self.status is WorkflowStatus.FAILED else None,
        )


def _dispatcher(
    store: Any, executor: Any, *, failure_handler: Any = None
) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        executor,
        WorkflowDispatcherConfig(enabled=True, max_workers=1, batch_size=10),
        failure_handler=failure_handler,
    )


def _pending(
    store: Any,
    request_id: str,
    *,
    with_incident: bool = True,
    incident_values: dict[str, Any] | None = None,
    **workflow_values: Any,
) -> WorkflowRequest:
    incident_id = f"inc-{request_id}"
    workflow = workflow_request(
        request_id,
        incident_id,
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
        **{
            "official_steps": [
                workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])
            ],
            **workflow_values,
        },
    )
    if not with_incident:
        store.save_workflow(workflow)
        return workflow
    store.save_incident_and_workflow(
        fault_incident(
            incident_id,
            f"event-{request_id}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=request_id,
            fencing_token=1,
            created_at=NOW,
            updated_at=NOW,
            **(incident_values or {}),
        ),
        workflow,
    )
    return workflow


def test_a_predecessor_row_that_is_gone_releases_its_successor() -> None:
    store = build_store()
    _pending(store, "wf-successor", predecessor_workflow_id="wf-retained-away")
    executor = _OutcomeExecutor(WorkflowStatus.SUCCEEDED)

    report = _dispatcher(store, executor).run_once()

    assert report.filtered.get("predecessor_missing") == 1, report.filtered
    assert executor.requested == ["wf-successor"], (
        "a successor whose predecessor row no longer exists was held back"
    )


def test_a_job_workflow_naming_no_nodes_waits_on_no_other_remediation() -> None:
    """A RESTART_WORKLOAD step carries workload ids, not node ids; with no node
    there is no node another remediation could be holding."""

    store = build_store()
    _pending(
        store,
        "wf-job-only",
        official_steps=[
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                node_ids=[],
                workload_ids=["training/job/train-1"],
            )
        ],
    )
    executor = _OutcomeExecutor(WorkflowStatus.SUCCEEDED)

    report = _dispatcher(store, executor).run_once()

    assert executor.requested == ["wf-job-only"]
    assert "node_busy" not in report.filtered, report.filtered


def test_a_job_scoped_incident_adds_its_attempt_to_the_processor_queue_scope() -> None:
    store = build_store()
    _pending(
        store,
        "wf-job",
        incident_values={"job_id": "job-a", "attempt_id": "attempt-a"},
        # Judged against the wall clock, so the window has to be open now.
        aggregation_max_deadline=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    executor = _OutcomeExecutor(WorkflowStatus.SUCCEEDED)

    report = _dispatcher(store, executor).run_once()

    assert executor.requested == ["wf-job"], (
        "an empty processor queue held a job-scoped workflow back"
    )
    assert "processor_queue" not in report.filtered, report.filtered


def test_a_failed_outcome_under_a_pending_preemption_is_left_to_the_successor() -> None:
    store = build_store()
    _pending(store, "wf-failed")
    store.save_workflow(
        workflow_request(
            "wf-stronger",
            "inc-wf-failed",
            status=WorkflowStatus.PENDING,
            fencing_token=2,
            official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
            predecessor_workflow_id="wf-failed",
            preempt_predecessor=True,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    handled: list[str] = []
    executor = _OutcomeExecutor(WorkflowStatus.FAILED)

    report = _dispatcher(
        store,
        executor,
        failure_handler=lambda workflow: handled.append(workflow.request_id),
    ).run_once()

    assert "wf-failed" in executor.requested
    assert [failure.workflow_request_id for failure in report.failures] == ["wf-failed"]
    assert handled == [], (
        "the failure handler ran although a stronger successor is about to preempt"
    )


def test_a_handler_that_ran_against_a_row_no_longer_failed_stamps_nothing() -> None:
    """The executor reported FAILED, but the row the handler saw afterwards is
    not FAILED any more (here: the stub never wrote the terminal status), so the
    handled mark is not written onto a row that may still execute."""

    store = build_store()
    _pending(store, "wf-failed")
    handled: list[str] = []
    executor = _OutcomeExecutor(WorkflowStatus.FAILED)

    _dispatcher(
        store,
        executor,
        failure_handler=lambda workflow: handled.append(workflow.request_id),
    ).run_once()

    assert handled == ["wf-failed"]
    current = store.get_workflow("wf-failed")
    assert current.status is WorkflowStatus.PENDING
    assert current.failure_handled_at is None, (
        "a non-FAILED row was stamped as a handled failure"
    )


def _plan(status: PlanStatus) -> RecoveryPlan:
    return RecoveryPlan(
        incident_id="inc-wf-planned",
        attempt_id="attempt-plan",
        trigger="quick-triage:PASS",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": RecoveryAction.RESTART_WORKLOAD,
                "node_ids": ["node-a"],
                "execution_owner": "simulated-runtime",
            }
        ],
        status=status,
    )


def test_a_plan_already_in_the_mirrored_status_is_not_rewritten() -> None:
    store = build_store()
    plan = _plan(PlanStatus.SUCCEEDED)
    store.save_plan(plan)
    _pending(store, "wf-planned", source_plan_id=plan.plan_id)
    dispatcher = _dispatcher(store, _OutcomeExecutor(WorkflowStatus.SUCCEEDED))

    dispatcher.run_once()

    assert store.get_plan(plan.plan_id) == plan, (
        "an already-mirrored plan was rewritten"
    )
    assert dispatcher.plan_sync_misses_total == 0


def test_a_plan_behind_the_workflow_outcome_is_mirrored_to_it() -> None:
    store = build_store()
    plan = _plan(PlanStatus.RUNNING)
    store.save_plan(plan)
    _pending(store, "wf-planned", source_plan_id=plan.plan_id)
    dispatcher = _dispatcher(store, _OutcomeExecutor(WorkflowStatus.SUCCEEDED))

    dispatcher.run_once()

    assert store.get_plan(plan.plan_id).status is PlanStatus.SUCCEEDED
    assert dispatcher.plan_sync_misses_total == 0
