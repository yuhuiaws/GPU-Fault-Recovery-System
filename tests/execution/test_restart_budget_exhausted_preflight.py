"""Restart-budget exhaustion at preflight does not abort the remediation chain.

逻辑 4. ``reserve_restart_budgets`` runs before any step of a claimed
workflow. When the job's restart budget was already spent, the preflight
used to record the RESTART_WORKLOAD step FAILED and terminalize the whole
workflow FAILED / ESCALATED -- so a RESET_GPU or REBOOT_NODE chain never
cordoned, never stopped and never reset anything: the faulty GPU stayed
as-is with a live job on it, ``HardwareEscalationService.classify`` opened
no successor (the only FAILED execution was RESTART_WORKLOAD), and no
BUDGET_EXHAUSTED notification was produced (the adapter path does produce
one).

Now the exhausted restart is withheld: every other step runs in order, the
restart step never executes and is recorded FAILED with reason
RESTART_BUDGET_EXHAUSTED, the BUDGET_EXHAUSTED notification is saved once
and sent, and the workflow ends FAILED (the job was stopped and not
restarted) with the incident ESCALATED -- or QUARANTINED when the cordon was
never lifted.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.execution import (
    ProductionWorkflowExecutor,
    WorkflowExecutionRequest,
    WorkflowStepContext,
    WorkflowStepOutcome,
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
    fault_incident,
    workflow_request,
    workflow_step,
)

NOW = datetime(2026, 9, 8, 6, 30, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
JOB = "training-a"
RESTART = WorkflowOperation.RESTART_WORKLOAD
CHAIN = (
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.STOP_WORKLOADS,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.RESTORE_SCHEDULING,
    RESTART,
)
# The same chain on a node that stays cordoned (no RESTORE_SCHEDULING).
CORDONED_CHAIN = tuple(
    operation
    for operation in CHAIN
    if operation is not WorkflowOperation.RESTORE_SCHEDULING
)
RESTART_PARAMETERS: dict[str, object] = {
    "cluster_id": CLUSTER,
    "job_id": JOB,
    "source_attempt_id": "attempt-a",
    "source_gpu_count": 1,
    "restart_budget": 1,
}
RESERVATION = f"workflow-a/{CHAIN.index(RESTART)}/{RESTART.value}"


class RecordingAdapter:
    def __init__(self) -> None:
        self.calls: list[WorkflowOperation] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append(context.step.operation)
        return WorkflowStepOutcome.succeeded()


def _state(
    store: InMemoryStore,
    operations: tuple[WorkflowOperation, ...] = CHAIN,
    *,
    restart_parameters: dict[str, object] | None = None,
    dag_enabled: bool = False,
    request_id: str = "workflow-a",
    incident_id: str = "incident-a",
) -> WorkflowRequest:
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    parameters = (
        dict(RESTART_PARAMETERS) if restart_parameters is None else restart_parameters
    )
    steps = [
        workflow_step(
            operation,
            workload_ids=[f"training/job/{JOB}"],
            parameters=parameters if operation is RESTART else {},
            depends_on_step_indexes=[index - 1] if dag_enabled and index else [],
        )
        for index, operation in enumerate(operations)
    ]
    workflow = workflow_request(
        request_id,
        incident.incident_id,
        fencing_token=1,
        official_steps=steps,
        dag_enabled=dag_enabled,
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident(copy_model(incident, workflow_request_id=workflow.request_id))
    store.save_workflow(workflow)
    return workflow


def _executor(
    store: InMemoryStore, adapter: RecordingAdapter, sent: list[str] | None = None
) -> ProductionWorkflowExecutor:
    return active_workflow_executor(
        store,
        [adapter],
        CHAIN,
        notification_sender=None if sent is None else sent.append,
    )


def _request() -> WorkflowExecutionRequest:
    return WorkflowExecutionRequest(expected_fencing_token=1)


def _exhaust_budget(store: InMemoryStore) -> None:
    _, reserved = store.reserve_job_restart(CLUSTER, JOB, 1, "previous-restart")
    assert reserved, "the fixture's first reservation must consume the budget of 1"


def _budget_exhausted_notifications(store: InMemoryStore) -> list:
    return [
        item
        for item in store.list_notifications()
        if "restart-budget-exhausted" in item.deduplication_key
    ]


def test_exhausted_budget_runs_the_chain_and_withholds_only_the_restart() -> None:
    store = build_store()
    _exhaust_budget(store)
    workflow = _state(store)
    adapter = RecordingAdapter()
    sent: list[str] = []

    result = _executor(store, adapter, sent).execute(workflow.request_id, _request())

    persisted = store.get_workflow(workflow.request_id)
    incident = store.get_incident(workflow.incident_id)
    non_restart = [operation for operation in CHAIN if operation is not RESTART]
    assert adapter.calls == non_restart
    assert result.status is WorkflowStatus.FAILED
    assert persisted.status is WorkflowStatus.FAILED
    assert persisted.completed_operations == non_restart
    assert persisted.completed_step_indexes == list(range(len(CHAIN) - 1))
    succeeded = {
        item.step_index
        for item in persisted.step_executions
        if item.status is WorkflowStepStatus.SUCCEEDED
    }
    assert succeeded == set(range(len(CHAIN) - 1))
    restart_records = [
        item for item in persisted.step_executions if item.operation is RESTART
    ]
    assert len(restart_records) == 1
    (record,) = restart_records
    assert record.step_index == CHAIN.index(RESTART)
    assert record.status is WorkflowStepStatus.FAILED
    assert record.details["reason"] == "RESTART_BUDGET_EXHAUSTED"
    assert record.details["restart_count"] == 1
    assert record.details["restart_budget"] == 1
    assert "1/1" in (record.error or "")
    assert "restart budget exhausted" in (result.error or "")
    assert incident.state is IncidentState.ESCALATED
    notifications = _budget_exhausted_notifications(store)
    assert len(notifications) == 1
    assert notifications[0].incident_id == incident.incident_id
    assert sent == [notifications[0].notification_id]
    assert record.details["notification_id"] == notifications[0].notification_id


def test_exhausted_budget_never_reserves_or_releases_a_phantom_reservation() -> None:
    store = build_store()
    _exhaust_budget(store)
    workflow = _state(store)

    _executor(store, RecordingAdapter()).execute(workflow.request_id, _request())

    state = store.get_restart_budget(CLUSTER, JOB)
    assert state.restart_count == 1
    assert state.reservation_ids == ["previous-restart"]
    assert store.list_remote_commands() == []


def test_exhausted_budget_keeps_the_incident_quarantined_when_cordon_stays() -> None:
    store = build_store()
    _exhaust_budget(store)
    workflow = _state(store, CORDONED_CHAIN)
    adapter = RecordingAdapter()

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == [op for op in CORDONED_CHAIN if op is not RESTART]
    assert store.get_incident(workflow.incident_id).state is IncidentState.QUARANTINED


def test_exhausted_budget_notification_is_saved_once_across_reclaims() -> None:
    store = build_store()
    _exhaust_budget(store)
    workflow = _state(store)
    adapter = RecordingAdapter()
    sent: list[str] = []
    executor = _executor(store, adapter, sent)
    executor.execute(workflow.request_id, _request())

    # A second workflow for the same job hits the same exhausted budget: the
    # deduplication key keeps one notification for the job.
    other = _state(store, request_id="workflow-b", incident_id="incident-b")
    executor.execute(other.request_id, _request())

    assert len(_budget_exhausted_notifications(store)) == 1


def test_exhausted_budget_in_a_dag_chain_ends_failed_after_every_other_step() -> None:
    store = build_store()
    _exhaust_budget(store)
    workflow = _state(store, dag_enabled=True)
    adapter = RecordingAdapter()

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    persisted = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == [op for op in CHAIN if op is not RESTART]
    assert persisted.completed_operations == [op for op in CHAIN if op is not RESTART]
    assert store.get_incident(workflow.incident_id).state is IncidentState.ESCALATED
    assert len(_budget_exhausted_notifications(store)) == 1


def test_available_budget_still_runs_the_restart() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter()

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget(CLUSTER, JOB)
    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == list(CHAIN)
    assert state.restart_count == 1
    assert state.reservation_ids == [RESERVATION]
    assert store.get_incident(workflow.incident_id).state is IncidentState.RECOVERED
    assert _budget_exhausted_notifications(store) == []


def test_missing_restart_context_still_aborts_before_any_step() -> None:
    store = build_store()
    parameters = dict(RESTART_PARAMETERS)
    del parameters["source_gpu_count"]
    workflow = _state(store, restart_parameters=parameters)
    adapter = RecordingAdapter()

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    persisted = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []
    assert persisted.completed_step_indexes == []
    (execution,) = persisted.step_executions
    assert execution.step_index == CHAIN.index(RESTART)
    assert execution.details["reason"] == "RESTART_SAFETY_CONTEXT_MISSING"
    assert store.get_incident(workflow.incident_id).state is IncidentState.ESCALATED
