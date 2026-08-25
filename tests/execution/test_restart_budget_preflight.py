from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.execution import (
    ProductionWorkflowExecutor,
    WorkflowExecutionRequest,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.execution.restart_budget_preflight import (
    release_unattempted_restart_reservations,
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
    workflow_step_execution,
)

NOW = datetime(2026, 8, 20, 6, 30, tzinfo=timezone.utc)
OPERATIONS = (
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.STOP_WORKLOADS,
    WorkflowOperation.RESTART_WORKLOAD,
)


class RecordingAdapter:
    def __init__(
        self,
        store: InMemoryStore,
        outcomes: dict[WorkflowOperation, WorkflowStepOutcome] | None = None,
    ) -> None:
        self.store = store
        self.outcomes = outcomes or {}
        self.calls: list[WorkflowOperation] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        state = self.store.get_restart_budget("cluster-a", "training-a")
        assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]
        self.calls.append(context.step.operation)
        return self.outcomes.get(
            context.step.operation, WorkflowStepOutcome.succeeded()
        )


def _state(
    store: InMemoryStore, *, restart_parameters: dict[str, object] | None = None
) -> WorkflowRequest:
    incident = fault_incident(
        "incident-a",
        "event-a",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    parameters = {
        "cluster_id": "cluster-a",
        "job_id": "training-a",
        "source_attempt_id": "attempt-a",
        "source_gpu_count": 1,
        "restart_budget": 1,
    }
    if restart_parameters is not None:
        parameters = restart_parameters
    workflow = workflow_request(
        "workflow-a",
        incident.incident_id,
        fencing_token=1,
        official_steps=[
            workflow_step(
                operation,
                workload_ids=["training/job/training-a"],
                parameters=parameters
                if operation is WorkflowOperation.RESTART_WORKLOAD
                else {},
            )
            for operation in OPERATIONS
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident(copy_model(incident, workflow_request_id=workflow.request_id))
    store.save_workflow(workflow)
    return workflow


def _executor(
    store: InMemoryStore, adapter: RecordingAdapter
) -> ProductionWorkflowExecutor:
    return active_workflow_executor(store, [adapter], OPERATIONS)


def _request() -> WorkflowExecutionRequest:
    return WorkflowExecutionRequest(expected_fencing_token=1)


def test_exhausted_budget_fails_before_any_adapter_call() -> None:
    store = build_store()
    store.reserve_job_restart("cluster-a", "training-a", 1, "previous-restart")
    workflow = _state(store)
    adapter = RecordingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    persisted = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []
    assert persisted.completed_step_indexes == []
    assert len(persisted.step_executions) == 1
    execution = persisted.step_executions[0]
    assert execution.step_index == 2
    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.details["reason"] == ("RESTART_BUDGET_EXHAUSTED")
    assert "1/1" in execution.error
    assert store.list_remote_commands() == []


def test_missing_restart_context_fails_before_stop() -> None:
    store = build_store()
    workflow = _state(
        store,
        restart_parameters={
            "cluster_id": "cluster-a",
            "job_id": "training-a",
            "source_attempt_id": "attempt-a",
            "restart_budget": 1,
        },
    )
    adapter = RecordingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []
    assert execution.step_index == 2
    assert execution.details["reason"] == ("RESTART_SAFETY_CONTEXT_MISSING")
    assert "source_gpu_count" in execution.error


def test_budget_is_reserved_before_first_adapter_call() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == list(OPERATIONS)
    assert state.restart_count == 1
    assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]


def test_failure_before_restart_releases_reservation() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(
        store,
        {
            WorkflowOperation.FREEZE_EVIDENCE: (
                WorkflowStepOutcome.failed("evidence collection failed")
            )
        },
    )

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == [WorkflowOperation.FREEZE_EVIDENCE]
    assert state.restart_count == 0
    assert state.reservation_ids == []


def test_waiting_retry_does_not_double_reserve() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(
        store,
        {
            WorkflowOperation.FREEZE_EVIDENCE: (
                WorkflowStepOutcome.waiting(operation_id="evidence/waiting")
            )
        },
    )
    active = _executor(store, adapter)

    first = active.execute(workflow.request_id, _request())
    second = active.execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert first.status is WorkflowStatus.RUNNING
    assert second.status is WorkflowStatus.RUNNING
    assert state.restart_count == 1
    assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]


def test_cancelled_waiting_restart_releases_reservation() -> None:
    store = build_store()
    workflow = _state(store)
    store.reserve_job_restart(
        "cluster-a", "training-a", 1, "workflow-a/2/RESTART_WORKLOAD"
    )
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                2, WorkflowOperation.RESTART_WORKLOAD, WorkflowStepStatus.WAITING
            )
        ],
    )

    release_unattempted_restart_reservations(store, workflow, release_step_indexes={2})

    state = store.get_restart_budget("cluster-a", "training-a")
    assert state.restart_count == 0
    assert state.reservation_ids == []
