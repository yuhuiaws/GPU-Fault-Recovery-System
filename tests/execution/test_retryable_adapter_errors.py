"""A transient adapter failure is a wait, not a hardware escalation (ARCH-B1).

``_dispatch_step`` re-raised only a transient *store* error and turned every
other adapter exception into a terminal FAILED step, which is what feeds
isolation and escalation. A Kubernetes 409/429/5xx, a urllib3 timeout or a
reset connection says nothing about the node; the step is retried a bounded
number of times and only then fails with the last error.
"""

from __future__ import annotations

import pytest

from gpu_fault.execution import ProductionWorkflowExecutor
from gpu_fault.execution.models import (
    WorkflowExecutionError,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.execution.transient_errors import retryable_adapter_error
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step_execution,
)
from tests.execution._support import workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
kubernetes_client = pytest.importorskip("kubernetes.client")
urllib3_exceptions = pytest.importorskip("urllib3.exceptions")


class RaisingAdapter:
    """Raises the queued exceptions one per call, then succeeds."""

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = list(failures)
        self.calls = 0

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a" and step.operation is FREEZE

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return WorkflowStepOutcome.succeeded(operation_id="freeze-done")


def _api_exception(status: int) -> BaseException:
    return kubernetes_client.ApiException(status=status, reason=f"HTTP {status}")


def _step_record(store, request_id: str):
    workflow = store.get_workflow(request_id)
    records = [item for item in workflow.step_executions if item.step_index == 0]
    assert len(records) == 1, f"expected one record for step 0, got {records}"
    return workflow, records[0]


@pytest.mark.parametrize("status", [409, 429, 500, 502, 503, 504])
def test_kubernetes_api_exception_with_transient_status_is_retryable(status: int):
    assert retryable_adapter_error(_api_exception(status)), (
        f"ApiException({status}) must be classified retryable"
    )


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_kubernetes_api_exception_with_definitive_status_is_not_retryable(status: int):
    assert not retryable_adapter_error(_api_exception(status)), (
        f"ApiException({status}) is a definitive answer, not a retry"
    )


def test_urllib3_and_transport_errors_are_retryable():
    pool = urllib3_exceptions.ReadTimeoutError(None, "/api", "read timed out")
    max_retry = urllib3_exceptions.MaxRetryError(None, "/api", reason=pool)
    protocol = urllib3_exceptions.ProtocolError("connection broken")
    assert retryable_adapter_error(pool), "ReadTimeoutError must be retryable"
    assert retryable_adapter_error(max_retry), "MaxRetryError must be retryable"
    assert retryable_adapter_error(protocol), "ProtocolError must be retryable"
    assert retryable_adapter_error(ConnectionResetError(104, "reset")), (
        "a reset connection must be retryable"
    )
    assert retryable_adapter_error(TimeoutError("timed out")), (
        "a timeout must be retryable"
    )


def test_wrapped_transient_cause_is_retryable_and_plain_bugs_are_not():
    wrapped = RuntimeError("patch failed")
    wrapped.__cause__ = _api_exception(429)
    assert retryable_adapter_error(wrapped), "the cause chain must be walked"
    assert not retryable_adapter_error(ValueError("bad manifest")), (
        "a ValueError is a defect, not a transient"
    )
    assert not retryable_adapter_error(KeyError("node")), (
        "a KeyError is a defect, not a transient"
    )


def test_transient_adapter_error_becomes_a_waiting_step_not_a_failure():
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE])
    adapter = RaisingAdapter([_api_exception(503), _api_exception(429)])
    executor = active_workflow_executor(store, [adapter], {FREEZE})

    first = execute_workflow(executor, workflow.request_id)
    current, record = _step_record(store, workflow.request_id)

    assert first.status is WorkflowStatus.RUNNING, first
    assert first.waiting_step_index == 0, first
    assert record.status is WorkflowStepStatus.WAITING, record
    assert record.details.get("retryable_adapter_error") is True, record.details
    assert record.details.get("error_class") == "ApiException", record.details
    assert record.details.get("attempt") == 1, record.details
    assert current.completed_step_indexes == [], current.completed_step_indexes
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.ACTION_PENDING
    ), "a transient adapter error must not move the incident"

    second = execute_workflow(executor, workflow.request_id)
    _, record = _step_record(store, workflow.request_id)
    assert second.status is WorkflowStatus.RUNNING, second
    assert record.details.get("attempt") == 2, record.details

    third = execute_workflow(executor, workflow.request_id)
    assert third.status is WorkflowStatus.SUCCEEDED, third
    assert adapter.calls == 3, adapter.calls
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED, (
        "the workflow that finally succeeded closes its incident"
    )


def test_transient_adapter_error_keeps_the_in_flight_pointers_of_the_step():
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    in_flight = workflow_step_execution(
        0,
        FREEZE,
        WorkflowStepStatus.WAITING,
        adapter_operation_id="remote/cmd-0",
        details={"remote_status": "RUNNING", "remote_command_id": "cmd-0"},
    )
    store.save_workflow(
        copy_model(workflow, status=WorkflowStatus.RUNNING, step_executions=[in_flight])
    )
    adapter = RaisingAdapter([_api_exception(503)])
    executor = active_workflow_executor(store, [adapter], {FREEZE})

    execute_workflow(executor, workflow.request_id)
    _, record = _step_record(store, workflow.request_id)

    assert record.adapter_operation_id == "remote/cmd-0", record
    assert record.details.get("remote_command_id") == "cmd-0", record.details
    assert record.details.get("remote_status") == "RUNNING", record.details
    assert record.details.get("retryable_adapter_error") is True, record.details


def test_transient_adapter_error_retries_on_a_dag_branch():
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    store.save_workflow(copy_model(workflow, dag_enabled=True, dag_revision=1))
    adapter = RaisingAdapter([_api_exception(502)])
    executor = active_workflow_executor(store, [adapter], {FREEZE})

    result = execute_workflow(executor, workflow.request_id)
    current, record = _step_record(store, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING, result
    assert result.waiting_step_index == 0, result
    assert record.status is WorkflowStepStatus.WAITING, record
    assert current.status is WorkflowStatus.RUNNING, current.status


def test_retry_limit_exhaustion_falls_through_to_the_failed_path():
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE])
    adapter = RaisingAdapter([_api_exception(503), _api_exception(503)])
    template = active_workflow_executor(store, [adapter], {FREEZE})
    executor = ProductionWorkflowExecutor(
        store, [adapter], template.config, step_transient_retry_limit=1
    )

    first = execute_workflow(executor, workflow.request_id)
    assert first.status is WorkflowStatus.RUNNING, first

    second = execute_workflow(executor, workflow.request_id)
    current, record = _step_record(store, workflow.request_id)

    assert second.status is WorkflowStatus.FAILED, second
    assert current.status is WorkflowStatus.FAILED, current.status
    assert record.status is WorkflowStepStatus.FAILED, record
    assert record.details.get("retryable_adapter_error") is True, record.details
    assert record.details.get("attempt") == 2, record.details
    assert "ApiException" in (record.error or ""), record.error
    # A FREEZE-only workflow is diagnostic-only: its failure closes the
    # incident RECOVERED ("diagnostic inconclusive") rather than ESCALATED.
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED, (
        "past the bound the last error ends the workflow like any adapter failure"
    )


def test_retry_limit_reads_the_environment_and_zero_disables_retries(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_WORKFLOW_STEP_TRANSIENT_RETRY_LIMIT", "0")
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    adapter = RaisingAdapter([_api_exception(503)])
    executor = active_workflow_executor(store, [adapter], {FREEZE})

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.FAILED, result
    assert executor.step_transient_retry_limit == 0, executor.step_transient_retry_limit


def test_negative_retry_limit_is_refused(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_WORKFLOW_STEP_TRANSIENT_RETRY_LIMIT", "-1")
    store = build_store()
    with pytest.raises(WorkflowExecutionError, match="TRANSIENT_RETRY_LIMIT"):
        active_workflow_executor(store, [], {FREEZE})


def test_a_definitive_adapter_exception_still_fails_the_step_at_once():
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE])
    adapter = RaisingAdapter([ValueError("manifest is not valid")])
    executor = active_workflow_executor(store, [adapter], {FREEZE})

    result = execute_workflow(executor, workflow.request_id)
    _, record = _step_record(store, workflow.request_id)

    assert result.status is WorkflowStatus.FAILED, result
    assert record.status is WorkflowStepStatus.FAILED, record
    assert "retryable_adapter_error" not in record.details, record.details
    # Diagnostic-only workflow (FREEZE alone): the incident ends RECOVERED.
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED, (
        "a definitive adapter failure still ends the workflow at once"
    )


def test_a_transient_store_error_inside_the_adapter_still_propagates():
    class OperationalError(Exception):
        pass

    OperationalError.__module__ = "psycopg"
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    adapter = RaisingAdapter([OperationalError("SSL error: unexpected eof")])
    executor = active_workflow_executor(store, [adapter], {FREEZE})

    with pytest.raises(OperationalError):
        execute_workflow(executor, workflow.request_id)
