"""The executor and the dispatcher check the workflow invariants before every
write they own (RF-1 item 3, review item 6).

``check_workflow_invariants`` existed; nothing called it. The executor's three
save funnels (leased, leased-with-incident, terminal) and the dispatcher's two
terminal paths -- which now run through the executor's terminal funnel -- call
it with the executor's configured mode. RAISE stops the write; LOG writes and
reports.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.invariants import InvariantMode, WorkflowInvariantError
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
INVARIANT_LOGGER = "gpu_fault.execution.invariants"


def _executor(store, mode: InvariantMode):
    adapter = FakeAdapter({FREEZE: WorkflowStepOutcome.succeeded()})
    executor = active_workflow_executor(store, [adapter], [FREEZE])
    executor.config = replace(executor.config, workflow_invariant_mode=mode)
    return executor


def _malformed(store, **values):
    """A one-step workflow that claims to have completed step 7."""

    _, workflow = workflow_state(store, [FREEZE])
    malformed = copy_model(workflow, completed_step_indexes=[7], **values)
    store.save_workflow(malformed)
    return malformed


def test_raise_mode_stops_the_executors_leased_write() -> None:
    store = build_store()
    workflow = _malformed(store)
    executor = _executor(store, InvariantMode.RAISE)

    with pytest.raises(WorkflowInvariantError, match="outside the step list"):
        execute_workflow(executor, workflow.request_id)

    saved = store.get_workflow(workflow.request_id)
    assert saved.status is not WorkflowStatus.SUCCEEDED, (
        "the write the invariant rejected must not have landed"
    )


def test_log_mode_writes_and_reports_the_violation(caplog) -> None:
    store = build_store()
    workflow = _malformed(store)
    executor = _executor(store, InvariantMode.LOG)

    with caplog.at_level(logging.ERROR, logger=INVARIANT_LOGGER):
        result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.SUCCEEDED
    errors = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and workflow.request_id in record.getMessage()
        and "outside the step list" in record.getMessage()
    ]
    assert errors, "LOG mode names the workflow and the broken invariant"


@pytest.mark.parametrize(
    "mode, reaped",
    [(InvariantMode.RAISE, False), (InvariantMode.LOG, True)],
    ids=["raise-refuses-the-reap", "log-reaps-and-reports"],
)
def test_the_watchdog_reap_checks_the_invariants(
    caplog, mode: InvariantMode, reaped: bool
) -> None:
    store = build_store()
    now = datetime.now(timezone.utc)
    workflow = _malformed(
        store,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-gone",
        execution_epoch=1,
        execution_lease_expires_at=now - timedelta(minutes=5),
        execution_deadline=now - timedelta(minutes=1),
    )
    dispatcher = WorkflowDispatcher(
        store,
        _executor(store, mode),
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )

    with caplog.at_level(logging.ERROR):
        dispatcher.run_once()

    saved = store.get_workflow(workflow.request_id)
    assert (saved.status is WorkflowStatus.FAILED) is reaped, saved.status
    # LOG mode names the violation in the message; RAISE mode surfaces it as
    # the exception the sweep logs when it skips the record.
    reported = [
        record
        for record in caplog.records
        if workflow.request_id in record.getMessage()
        and (
            "outside the step list" in record.getMessage()
            or (
                record.exc_info is not None
                and "outside the step list" in str(record.exc_info[1])
            )
        )
    ]
    assert reported, [record.getMessage() for record in caplog.records]
