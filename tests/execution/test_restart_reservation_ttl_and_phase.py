"""A restart reservation follows the restart, not the record (F-C9).

Planning reserves a job's restart budget for every pending RESTART_WORKLOAD
step. The restart adapter only ever answers WAITING *before* it submits
anything (an approval is pending, or the incident is not recovered yet), so a
reservation behind a wait that never ended is budget spent on a restart that
never happened. Two fixes are pinned here:

* a WAITING restart record older than the step's waiting cap, or one the cap
  already turned into a failure, releases its reservation when the workflow
  terminalizes;
* the safety and official phases reserve under different ids, so a
  safety-phase restart at index N cannot be mistaken for the official one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.execution.restart_budget_preflight import (
    release_unattempted_restart_reservations,
    reserve_restart_budgets,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import RESTART_PARAMETERS, workflow_state

CLUSTER = RESTART_PARAMETERS["cluster_id"]
JOB = RESTART_PARAMETERS["job_id"]
RESTART = WorkflowOperation.RESTART_WORKLOAD
CAP = timedelta(seconds=600)


class _KeyRecordingAdapter:
    def __init__(self, outcome: WorkflowStepOutcome) -> None:
        self.outcome = outcome
        self.keys: list[str] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.keys.append(context.idempotency_key)
        return self.outcome


def _restart_only_workflow(store: InMemoryStore) -> WorkflowRequest:
    _, workflow = workflow_state(store, [RESTART])
    return workflow


def _reserve(store: InMemoryStore, workflow: WorkflowRequest) -> None:
    _, accepted = store.reserve_job_restart(
        CLUSTER, JOB, 1, f"{workflow.request_id}/0/{RESTART.value}"
    )
    assert accepted is True, "the test needs the reservation to be taken"


def _with_restart_record(
    workflow: WorkflowRequest,
    status: WorkflowStepStatus,
    *,
    age: timedelta,
    details: dict[str, object] | None = None,
) -> WorkflowRequest:
    now = datetime.now(timezone.utc)
    return copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0,
                RESTART,
                status,
                started_at=now - age,
                updated_at=now - age,
                details=details or {},
            )
        ],
    )


def _restart_count(store: InMemoryStore) -> int:
    return store.get_restart_budget(CLUSTER, JOB).restart_count


def test_a_waiting_restart_older_than_the_ttl_is_released() -> None:
    store = build_store()
    workflow = _with_restart_record(
        _restart_only_workflow(store),
        WorkflowStepStatus.WAITING,
        age=CAP + timedelta(minutes=1),
        details={"approval_required": True},
    )
    _reserve(store, workflow)

    release_unattempted_restart_reservations(store, workflow, waiting_ttl=CAP)

    assert _restart_count(store) == 0


def test_a_waiting_restart_inside_the_ttl_keeps_its_reservation() -> None:
    store = build_store()
    workflow = _with_restart_record(
        _restart_only_workflow(store),
        WorkflowStepStatus.WAITING,
        age=CAP - timedelta(minutes=1),
        details={"approval_required": True},
    )
    _reserve(store, workflow)

    release_unattempted_restart_reservations(store, workflow, waiting_ttl=CAP)

    assert _restart_count(store) == 1


def test_a_waiting_restart_is_kept_when_no_ttl_is_given() -> None:
    store = build_store()
    workflow = _with_restart_record(
        _restart_only_workflow(store), WorkflowStepStatus.WAITING, age=timedelta(days=1)
    )
    _reserve(store, workflow)

    release_unattempted_restart_reservations(store, workflow)

    assert _restart_count(store) == 1


def test_a_restart_the_waiting_cap_failed_is_released_without_a_ttl() -> None:
    store = build_store()
    workflow = _with_restart_record(
        _restart_only_workflow(store),
        WorkflowStepStatus.FAILED,
        age=CAP,
        details={"step_waiting_seconds": 601, "step_waiting_timeout_seconds": 600},
    )
    _reserve(store, workflow)

    release_unattempted_restart_reservations(store, workflow)

    assert _restart_count(store) == 0


def test_a_restart_the_adapter_itself_failed_keeps_its_reservation() -> None:
    store = build_store()
    workflow = _with_restart_record(
        _restart_only_workflow(store),
        WorkflowStepStatus.FAILED,
        age=timedelta(days=1),
        details={"reason": "RESTART_BUDGET_EXHAUSTED"},
    )
    _reserve(store, workflow)

    release_unattempted_restart_reservations(store, workflow, waiting_ttl=CAP)

    assert _restart_count(store) == 1


def test_a_remote_restart_still_running_keeps_its_reservation() -> None:
    store = build_store()
    workflow = _with_restart_record(
        _restart_only_workflow(store),
        WorkflowStepStatus.WAITING,
        age=timedelta(days=1),
        details={"remote_status": "RUNNING", "remote_command_id": "cmd-1"},
    )
    _reserve(store, workflow)

    release_unattempted_restart_reservations(store, workflow, waiting_ttl=CAP)

    assert _restart_count(store) == 1


def test_the_waiting_cap_on_a_restart_frees_the_budget_end_to_end() -> None:
    store = build_store()
    now = datetime.now(timezone.utc)
    workflow = copy_model(
        _with_restart_record(
            _restart_only_workflow(store),
            WorkflowStepStatus.WAITING,
            age=timedelta(minutes=20),
            details={"approval_required": True},
        ),
        status=WorkflowStatus.RUNNING,
        # A live window that started 20 minutes ago and has 10 to go: the
        # step's wait is measured inside it, past the 600 s cap, while the
        # workflow deadline itself has not passed.
        execution_deadline=now + timedelta(minutes=10),
    )
    store.save_workflow(workflow)
    # A WAITING restart record means its adapter already reserved (D-10: the
    # claim no longer re-reserves an attempted restart).
    _reserve(store, workflow)
    adapter = _KeyRecordingAdapter(
        WorkflowStepOutcome.waiting(
            operation_id="approval", details={"approval_required": True}
        )
    )
    executor = active_workflow_executor(store, [adapter], {RESTART})

    result = execute_workflow(executor, workflow.request_id)

    persisted = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert persisted.step_executions[-1].status is WorkflowStepStatus.FAILED
    assert _restart_count(store) == 0


def _two_phase_workflow(store: InMemoryStore) -> WorkflowRequest:
    incident = fault_incident(
        "incident-phases",
        "event-phases",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
    )
    workflow = workflow_request(
        "workflow-phases",
        incident.incident_id,
        status=WorkflowStatus.SAFETY_PENDING,
        safety_only=True,
        official_steps=[workflow_step(RESTART, parameters=dict(RESTART_PARAMETERS))],
        safety_steps=[workflow_step(RESTART, parameters=dict(RESTART_PARAMETERS))],
    )
    store.save_incident(copy_model(incident, workflow_request_id=workflow.request_id))
    store.save_workflow(workflow)
    return workflow


def test_safety_and_official_restart_reservations_are_distinct() -> None:
    store = build_store()
    workflow = _two_phase_workflow(store)
    incident = store.get_incident(workflow.incident_id)
    parameters = {**RESTART_PARAMETERS, "restart_budget": 2}
    official = [workflow_step(RESTART, parameters=parameters)]
    safety = [workflow_step(RESTART, parameters=parameters)]

    official_failure = reserve_restart_budgets(
        store, workflow, incident, official, phase="official"
    )
    safety_failure = reserve_restart_budgets(
        store, workflow, incident, safety, phase="safety"
    )

    assert official_failure is None, "the official restart must fit the budget"
    assert safety_failure is None, "the safety restart must fit the budget"
    # Budget 2, two phases: both reservations stand only if their ids differ.
    assert _restart_count(store) == 2


def test_the_safety_phase_adapter_key_is_the_id_the_preflight_reserved() -> None:
    store = build_store()
    workflow = _two_phase_workflow(store)
    adapter = _KeyRecordingAdapter(WorkflowStepOutcome.succeeded())
    executor = active_workflow_executor(store, [adapter], {RESTART})

    result = execute_workflow(executor, workflow.request_id)

    state = store.get_restart_budget(CLUSTER, JOB)
    assert result.status is WorkflowStatus.BLOCKED
    assert adapter.keys == state.reservation_ids
    assert adapter.keys != [f"{workflow.request_id}/0/{RESTART.value}"], (
        "the safety-phase restart must not share the official phase's id"
    )
