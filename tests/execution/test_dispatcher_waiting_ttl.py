"""The dispatcher's terminalization paths apply the RESTART_WORKLOAD waiting
TTL when they release restart reservations (F-C9, log 60 item 2).

``release_unattempted_restart_reservations`` keeps a WAITING record's
reservation unless told how long a restart may wait before its reservation
counts as never used. The executor's own terminal writes pass that TTL; the
watchdog reap, the internal-error BLOCK and the retired-generation revocation
did not, so a restart still waiting on an approval when the watchdog reaped its
workflow kept the job's budget for nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.models import WorkflowRecordInvalidError
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import RESTART_PARAMETERS, workflow_state

CLUSTER = RESTART_PARAMETERS["cluster_id"]
JOB = RESTART_PARAMETERS["job_id"]
RESTART = WorkflowOperation.RESTART_WORKLOAD
QUARANTINE = WorkflowOperation.QUARANTINE
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
# ``active_workflow_executor`` caps a step's wait at 600 seconds.
TTL = timedelta(seconds=600)


def _reservation(workflow: WorkflowRequest, index: int) -> str:
    return f"{workflow.request_id}/{index}/{RESTART.value}"


def _budget_is_free(store) -> bool:
    _, accepted = store.reserve_job_restart(CLUSTER, JOB, 1, "a-later-real-restart")
    return accepted


def _dispatcher(store, adapter=None) -> WorkflowDispatcher:
    adapters = [adapter] if adapter is not None else []
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, adapters, {QUARANTINE, FREEZE, RESTART}),
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )


def _waiting_restart(index: int, started_at: datetime, **details):
    # The shape the adapter's approval hold leaves: the TTL only releases a
    # wait that says nothing was submitted.
    details.setdefault(
        "details", {"reason": "GPU_COUNT_CHANGED", "restart_submitted": False}
    )
    return workflow_step_execution(
        index,
        RESTART,
        WorkflowStepStatus.WAITING,
        started_at=started_at,
        updated_at=started_at,
        **details,
    )


@pytest.mark.parametrize(
    "age, released",
    [(TTL + timedelta(seconds=1), True), (timedelta(seconds=10), False)],
    ids=["older-than-ttl-released", "fresher-than-ttl-kept"],
)
def test_the_watchdog_reap_applies_the_waiting_ttl(
    age: timedelta, released: bool
) -> None:
    store = build_store()
    _, workflow = workflow_state(store, [QUARANTINE, FREEZE, RESTART])
    now = datetime.now(timezone.utc)
    overdue = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[QUARANTINE],
        # The deadline verdict lands on step 1, the first unfinished step; the
        # restart's own WAITING record at step 2 is what the TTL judges.
        step_executions=[_waiting_restart(2, now - age)],
        execution_owner_id="executor-gone",
        execution_epoch=1,
        execution_lease_expires_at=now - timedelta(minutes=5),
        execution_deadline=now - timedelta(minutes=1),
    )
    store.save_workflow(overdue, expected=workflow)
    store.reserve_job_restart(CLUSTER, JOB, 1, _reservation(overdue, 2))

    _dispatcher(store).run_once()

    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.FAILED
    assert _budget_is_free(store) is released


@pytest.mark.parametrize(
    "age, released",
    [(TTL + timedelta(seconds=1), True), (timedelta(seconds=10), False)],
    ids=["older-than-ttl-released", "fresher-than-ttl-kept"],
)
def test_blocking_on_an_internal_error_applies_the_waiting_ttl(
    age: timedelta, released: bool
) -> None:
    store = build_store()
    _, workflow = workflow_state(store, [QUARANTINE, RESTART])
    now = datetime.now(timezone.utc)
    store.save_workflow(
        copy_model(
            workflow,
            status=WorkflowStatus.RUNNING,
            step_executions=[_waiting_restart(1, now - age)],
        )
    )
    store.reserve_job_restart(CLUSTER, JOB, 1, _reservation(workflow, 1))
    # The one internal error that proves the record itself is unusable (D-5).
    invalid = WorkflowRecordInvalidError("workflow row cannot be decoded")

    class _Raising:
        owner = "simulated-runtime"

        def supports(self, _step) -> bool:
            raise invalid

        def execute(self, _context):
            raise AssertionError("unreachable")

    _dispatcher(store, _Raising()).run_once()

    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.BLOCKED
    assert _budget_is_free(store) is released


@pytest.mark.parametrize(
    "age, released",
    [(TTL + timedelta(seconds=1), True), (timedelta(seconds=10), False)],
    ids=["older-than-ttl-released", "fresher-than-ttl-kept"],
)
def test_revoking_a_retired_generation_applies_the_waiting_ttl(
    age: timedelta, released: bool
) -> None:
    """A retired generation whose remote-backed restart was still WAITING when
    its command settled: revocable, and its reservation is judged by the TTL."""

    store = build_store()
    now = datetime.now(timezone.utc)
    retired_id, current_id, incident_id = "workflow-retired", "workflow-current", "inc"
    store.save_incident(
        fault_incident(
            incident_id,
            "event-a",
            cluster_id=CLUSTER,
            state=IncidentState.ACTION_PENDING,
            fencing_token=4,
            workflow_request_id=current_id,
        )
    )
    retired = workflow_request(
        retired_id,
        incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_action="RESTART_APP",
        official_steps=[
            workflow_step(FREEZE),
            workflow_step(RESTART, parameters=dict(RESTART_PARAMETERS)),
        ],
        completed_step_indexes=[0],
        completed_operations=[FREEZE],
        step_executions=[
            _waiting_restart(
                1,
                now - age,
                adapter_operation_id="remote/cmd-restart",
                details={
                    "remote_status": "WAITING",
                    "remote_command_id": "cmd-restart",
                    # The data plane's hold, carried by the regional adapter.
                    "restart_submitted": False,
                },
            )
        ],
    )
    store.save_workflow(retired)
    store.save_workflow(
        workflow_request(
            current_id,
            incident_id,
            fencing_token=4,
            official_action="RUN_DIAGNOSTICS",
            official_steps=[workflow_step(WorkflowOperation.VALIDATE_GPU)],
        )
    )
    store.reserve_job_restart(CLUSTER, JOB, 1, _reservation(retired, 1))

    sweep = _dispatcher(store)
    sweep.run_once()
    sweep.run_once()

    assert store.get_workflow(retired_id).status is WorkflowStatus.SUPERSEDED
    assert _budget_is_free(store) is released
