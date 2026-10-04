"""A WAITING row's lease is cut to the dispatcher's waiting lease (D-7).

The executor leases a row for its full lease while a step runs. A step that
answers WAITING is done for this tick, so the row keeps its owner but its
lease is shortened to ``waiting_lease_duration`` when that is sooner than the
lease already held -- a dispatch-lease handover can then pick it up without
waiting out the full lease. Without a waiting lease the full lease stands.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowExecutionRequest, WorkflowOperation, WorkflowStatus
from tests._builders import active_workflow_executor, build_store
from tests.execution._support import FakeAdapter, workflow_state

OP = WorkflowOperation.QUARANTINE
LEASE_SECONDS = 180


def _execute(waiting_lease: timedelta | None) -> datetime:
    store = build_store()
    incident, workflow = workflow_state(store, [OP])
    adapter = FakeAdapter({OP: WorkflowStepOutcome.waiting(operation_id="op-1")})
    executor = active_workflow_executor(
        store, [adapter], {OP}, lease_duration_seconds=LEASE_SECONDS
    )
    executor.waiting_lease_duration = waiting_lease
    started = datetime.now(timezone.utc)

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=incident.fencing_token),
    )

    assert result.status is WorkflowStatus.RUNNING, result
    assert result.waiting_step_index == 0
    current = store.get_workflow(workflow.request_id)
    assert current.execution_owner_id == executor.config.executor_id
    assert current.execution_lease_expires_at is not None, (
        "a waiting row lost its lease instead of keeping a shorter one"
    )
    return current.execution_lease_expires_at - started


def test_a_waiting_row_keeps_its_owner_under_a_shortened_lease() -> None:
    remaining = _execute(timedelta(seconds=10))

    assert timedelta(0) < remaining <= timedelta(seconds=10 + 5), remaining


def test_a_waiting_lease_longer_than_the_lease_held_changes_nothing() -> None:
    remaining = _execute(timedelta(seconds=LEASE_SECONDS * 2))

    assert (
        timedelta(seconds=LEASE_SECONDS - 5)
        < remaining
        <= timedelta(seconds=LEASE_SECONDS + 5)
    ), remaining


def test_without_a_waiting_lease_the_full_lease_stands() -> None:
    remaining = _execute(None)

    assert (
        timedelta(seconds=LEASE_SECONDS - 5)
        < remaining
        <= timedelta(seconds=LEASE_SECONDS + 5)
    ), remaining
