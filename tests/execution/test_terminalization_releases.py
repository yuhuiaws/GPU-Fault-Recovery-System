"""Every path that terminalizes a workflow releases the restart reservations
no adapter ever attempted.

FINAL-建议汇总 F-C9 (P0-23A, P0-45C, P0-53B, P0-65A). Planning reserves a
job's restart budget for each RESTART_WORKLOAD step before any adapter runs.
The executor's own terminal writes release them through ``_save_terminal``;
the dispatcher's deadline reap, its internal-error BLOCK and the operator's
restore-reconcile did not, so a workflow that never restarted anything
silently spent the budget of the next real restart of the same job.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import (
    RESTART_PARAMETERS,
    active_workflow_executor,
    workflow_state,
)

NOW = datetime(2026, 9, 6, 1, 0, tzinfo=timezone.utc)
CLUSTER = RESTART_PARAMETERS["cluster_id"]
JOB = RESTART_PARAMETERS["job_id"]
RESTART = WorkflowOperation.RESTART_WORKLOAD


def _reservation(workflow: WorkflowRequest, index: int) -> str:
    return f"{workflow.request_id}/{index}/{RESTART.value}"


def _budget_is_free(store) -> bool:
    _, accepted = store.reserve_job_restart(CLUSTER, JOB, 1, "a-later-real-restart")
    return accepted


def _dispatcher(store, adapter=None):
    adapters = [adapter] if adapter is not None else []
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, adapters, {WorkflowOperation.QUARANTINE}),
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )


def test_the_deadline_reap_releases_the_unattempted_restart_reservation():
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.QUARANTINE, RESTART])
    # The watchdog compares against the wall clock, so the lease and the
    # deadline have to be in the real past.
    now = datetime.now(timezone.utc)
    overdue = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-gone",
        execution_epoch=1,
        execution_lease_expires_at=now - timedelta(minutes=5),
        execution_deadline=now - timedelta(minutes=1),
    )
    store.save_workflow(overdue, expected=workflow)
    store.reserve_job_restart(CLUSTER, JOB, 1, _reservation(overdue, 1))
    assert _budget_is_free(store) is False

    _dispatcher(store).run_once()

    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.FAILED
    assert _budget_is_free(store) is True


def test_blocking_on_an_internal_error_releases_the_reservation():
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.QUARANTINE, RESTART])
    store.reserve_job_restart(CLUSTER, JOB, 1, _reservation(workflow, 1))
    try:
        WorkflowRequest.model_validate({"incident_id": "x"})
    except ValidationError as error:
        invalid = error

    class _Raising:
        owner = "simulated-runtime"

        def supports(self, _step) -> bool:
            raise invalid

        def execute(self, _context):
            raise AssertionError("unreachable")

    _dispatcher(store, _Raising()).run_once()

    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.BLOCKED
    assert _budget_is_free(store) is True


def test_operator_restore_reconcile_releases_the_reservation():
    store = build_store()
    incident_id, blocked_id, successor_id, plan_id = (
        "incident-restored",
        "workflow-blocked",
        "workflow-restored",
        "plan-failed",
    )
    store.save_plan(
        RecoveryPlan(
            plan_id=plan_id,
            incident_id=incident_id,
            attempt_id="attempt-a",
            trigger="test",
            runtime_profile_version="profile-v1",
            steps=[],
            workflow_request_id=blocked_id,
            status=PlanStatus.FAILED,
            created_at=NOW - timedelta(hours=2),
        )
    )
    blocked = workflow_request(
        blocked_id,
        incident_id,
        status=WorkflowStatus.BLOCKED,
        fencing_token=7,
        source_plan_id=plan_id,
        official_action=WorkflowOperation.QUARANTINE.value,
        official_steps=[
            workflow_step(WorkflowOperation.QUARANTINE),
            workflow_step(RESTART, parameters=dict(RESTART_PARAMETERS)),
        ],
        updated_at=NOW - timedelta(hours=1),
    )
    store.save_workflow(blocked)
    store.save_workflow(
        workflow_request(
            successor_id,
            incident_id,
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=7,
            predecessor_workflow_id=blocked_id,
            completed_operations=[
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_HOST,
                WorkflowOperation.VALIDATE_FABRIC,
                WorkflowOperation.RESTORE_SCHEDULING,
            ],
            updated_at=NOW - timedelta(minutes=30),
        )
    )
    store.save_incident(
        fault_incident(
            incident_id,
            "event-restored",
            state=IncidentState.RECOVERED,
            workflow_request_id=successor_id,
            fencing_token=7,
            updated_at=NOW - timedelta(minutes=30),
        )
    )
    store.reserve_job_restart(CLUSTER, JOB, 1, _reservation(blocked, 1))
    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)

    apply_workflow_reconcile_plan(
        store,
        workflow_ids=[blocked_id],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-1",
        now=NOW + timedelta(minutes=1),
    )

    assert store.get_workflow(blocked_id).status is WorkflowStatus.SUPERSEDED
    assert _budget_is_free(store) is True
