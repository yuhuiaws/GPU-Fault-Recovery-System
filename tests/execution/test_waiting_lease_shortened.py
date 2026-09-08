"""A WAITING workflow does not stay pinned to one process for the full lease.

Control-plane review 2026-09-08, D-7. Returning WAITING kept the executor's
180 s lease on the row, and ``claim_workflow`` refuses another executor while a
lease is live. When the fleet's dispatch lease changed hands -- a tick longer
than the dispatch lease, a Pod restart during a rollout -- every in-flight
WAITING workflow stalled for 90-180 s until the old holder's lease lapsed, and
each tick spent one refused claim transaction per row.

The dispatcher now tells its executor how long a WAITING row's lease should
outlive the tick (three polls, floored at ``WAITING_LEASE_FLOOR_SECONDS``);
the executor stamps that shorter lease on the WAITING save. The floor exists
because the remediation budget only counts RUNNING rows with a live lease: a
lease that lapses before the next dispatch would let another workflow take the
same scope.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import (
    WAITING_LEASE_FLOOR_SECONDS,
    WorkflowDispatcher,
)
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from tests._builders import active_workflow_executor, build_store, execute_workflow
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state

OP = WorkflowOperation.FREEZE_EVIDENCE


def test_a_waiting_save_carries_the_shortened_lease() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [OP])
    adapter = FakeAdapter({OP: WorkflowStepOutcome.waiting(operation_id="op-1")})
    executor = active_workflow_executor(store, [adapter], {OP})
    executor.waiting_lease_duration = timedelta(seconds=15)

    before = datetime.now(timezone.utc)
    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    saved = store.get_workflow(workflow.request_id)
    assert saved.execution_owner_id == executor.config.executor_id
    assert saved.execution_lease_expires_at is not None
    assert before + timedelta(seconds=13) <= saved.execution_lease_expires_at
    assert saved.execution_lease_expires_at <= before + timedelta(seconds=20)


def test_another_executor_claims_the_row_once_the_short_lease_lapses() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [OP])
    adapter = FakeAdapter({OP: WorkflowStepOutcome.waiting(operation_id="op-1")})
    first = active_workflow_executor(store, [adapter], {OP}, executor_id="executor-a")
    first.waiting_lease_duration = timedelta(seconds=15)
    execute_workflow(first, workflow.request_id)

    claimed = store.claim_workflow(
        workflow.request_id,
        "executor-b",
        workflow.fencing_token,
        now=datetime.now(timezone.utc) + timedelta(seconds=20),
    )

    assert claimed.execution_owner_id == "executor-b"


def test_without_a_dispatcher_the_full_lease_is_kept() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [OP])
    adapter = FakeAdapter({OP: WorkflowStepOutcome.waiting(operation_id="op-1")})
    executor = active_workflow_executor(
        store, [adapter], {OP}, lease_duration_seconds=180
    )

    before = datetime.now(timezone.utc)
    execute_workflow(executor, workflow.request_id)

    saved = store.get_workflow(workflow.request_id)
    assert saved.execution_lease_expires_at >= before + timedelta(seconds=170)


def test_the_dispatcher_sets_three_polls_floored() -> None:
    store = build_store()
    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    short = active_workflow_executor(store, [adapter], {OP})
    WorkflowDispatcher(
        store, short, WorkflowDispatcherConfig(enabled=True, poll_interval_seconds=5)
    )
    long = active_workflow_executor(store, [adapter], {OP})
    WorkflowDispatcher(
        store, long, WorkflowDispatcherConfig(enabled=True, poll_interval_seconds=20)
    )

    assert short.waiting_lease_duration == timedelta(
        seconds=WAITING_LEASE_FLOOR_SECONDS
    )
    assert long.waiting_lease_duration == timedelta(seconds=60)
    assert WAITING_LEASE_FLOOR_SECONDS >= 30
