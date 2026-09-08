"""A DAG step's waiting clock is not reset by the claim that redispatches it.

Control-plane review 2026-09-08, D-1. ``claim_deadlines`` re-stamps a DAG
workflow's ``execution_deadline`` on every claim (every dispatcher tick), and
``step_bounds.step_elapsed_since`` recovered the execution window's start by
subtracting the budget from that deadline. For a DAG the recovered start was
therefore the current tick, ``step_waiting_seconds`` was 0 on every pass, and
neither the per-step cap nor rule A's 240 s premise window nor the slow-step
warning ever fired: a job workflow's only bound was its one-hour lifetime.

The window now opens at the record's first ``CLAIM`` event, which is written
once and kept at the head of the bounded event list; rows claimed before that
event existed keep the deadline arithmetic, which is stable for a sequential
workflow.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    record_workflow_event,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
RESET = WorkflowOperation.RESET_GPU


def _waiting_for(store, *, dag_enabled: bool, waited: timedelta, budget: int = 1800):
    """A leased RUNNING workflow whose step 0 has waited ``waited`` already."""

    _, created = workflow_state(store, [FREEZE, RESET])
    now = datetime.now(timezone.utc)
    first_claim = now - waited
    workflow = copy_model(
        created,
        dag_enabled=dag_enabled,
        official_steps=[
            workflow_step(FREEZE, node_ids=["node-a"]),
            workflow_step(RESET, node_ids=["node-a"], depends_on_step_indexes=[0]),
        ],
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-a",
        execution_epoch=1,
        execution_lease_expires_at=now + timedelta(seconds=180),
        # What the first claim stamped, ``waited`` ago.
        execution_deadline=first_claim + timedelta(seconds=budget),
        lifetime_deadline_at=first_claim + timedelta(seconds=3600),
        step_executions=[
            workflow_step_execution(
                0,
                FREEZE,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/cmd-0",
                details={"remote_status": "LEASED", "remote_command_id": "cmd-0"},
                started_at=first_claim,
            )
        ],
    )
    workflow = record_workflow_event(
        workflow,
        WorkflowEventKind.CLAIM,
        code=WorkflowEventCode.CLAIMED.value,
        actor="executor-a",
        status=WorkflowStatus.RUNNING.value,
        details={"execution_epoch": 1},
        at=first_claim,
    )
    # Deliberate out-of-band lease: ``expected`` names the row as read
    # (store review 2026-09-07, item B).
    store.save_workflow(workflow, expected=created)
    return workflow


def _executor(store, adapter: FakeAdapter):
    return active_workflow_executor(
        store,
        [adapter],
        {FREEZE, RESET},
        step_waiting_timeout_seconds=600,
        step_waiting_warning_seconds=300,
    )


def test_a_dag_step_waiting_past_the_cap_fails_even_though_the_claim_moved_the_deadline():
    store = build_store()
    workflow = _waiting_for(store, dag_enabled=True, waited=timedelta(minutes=20))
    adapter = FakeAdapter(
        {FREEZE: WorkflowStepOutcome.waiting(operation_id="remote/cmd-0")}
    )

    result = execute_workflow(_executor(store, adapter), workflow.request_id)

    saved = store.get_workflow(workflow.request_id)
    assert saved.execution_deadline != workflow.execution_deadline, (
        "the DAG claim re-stamps the execution deadline; the wait must survive it"
    )
    assert result.status is WorkflowStatus.FAILED, saved.step_executions
    record = next(item for item in saved.step_executions if item.step_index == 0)
    assert record.status is WorkflowStepStatus.FAILED
    assert record.details["step_waiting_timeout_seconds"] == 600
    assert 1200 - 5 <= record.details["step_waiting_seconds"] <= 1200 + 60


def test_a_dag_step_still_inside_the_cap_reports_its_real_age():
    store = build_store()
    workflow = _waiting_for(store, dag_enabled=True, waited=timedelta(minutes=6))
    adapter = FakeAdapter(
        {FREEZE: WorkflowStepOutcome.waiting(operation_id="remote/cmd-0")}
    )

    result = execute_workflow(_executor(store, adapter), workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    record = next(
        item
        for item in store.get_workflow(workflow.request_id).step_executions
        if item.step_index == 0
    )
    assert record.status is WorkflowStepStatus.WAITING
    assert 360 - 5 <= record.details["step_waiting_seconds"] <= 360 + 60
    assert record.details["step_waiting_slow"] is True


def test_the_sequential_cap_is_unchanged():
    store = build_store()
    workflow = _waiting_for(store, dag_enabled=False, waited=timedelta(minutes=20))
    adapter = FakeAdapter(
        {FREEZE: WorkflowStepOutcome.waiting(operation_id="remote/cmd-0")}
    )

    result = execute_workflow(_executor(store, adapter), workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    record = next(
        item
        for item in store.get_workflow(workflow.request_id).step_executions
        if item.step_index == 0
    )
    assert record.details["step_waiting_timeout_seconds"] == 600
