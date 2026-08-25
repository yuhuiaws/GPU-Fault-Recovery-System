from __future__ import annotations

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowOperation
from tests._builders import (
    active_workflow_executor,
    build_context,
    build_store,
    copy_model,
    execute_workflow,
    processor_request,
)


def test_shared_builders_preserve_control_plane_defaults() -> None:
    store = build_store()
    adapter = object()
    executor = active_workflow_executor(store, [adapter], [WorkflowOperation.RESET_GPU])

    assert executor.store is store
    assert executor.adapters == [adapter]
    assert executor.config.enabled is True
    assert executor.config.executor_id == "executor-a"
    assert executor.config.allowed_operations == frozenset(
        {WorkflowOperation.RESET_GPU}
    )
    assert executor.config.workflow_preemption_enabled is True

    request = processor_request("/v1/gpu-events/xid")
    assert request.method == "POST"
    assert request.path == "/v1/gpu-events/xid"
    assert request.query == ""
    assert request.content_type == "application/json"
    assert request.cluster_id == "cluster-a"
    assert request.execution_authorized is False

    replacement = copy_model(request, cluster_id="cluster-b")
    assert request.cluster_id == "cluster-a"
    assert replacement.cluster_id == "cluster-b"

    second = build_store()
    context = build_context()
    assert second is not store
    assert isinstance(context, ApplicationContext)


def test_execute_workflow_builds_the_fencing_request() -> None:
    class RecordingExecutor:
        def execute(self, request_id, request):
            return request_id, request

    request_id, request = execute_workflow(
        RecordingExecutor(),
        "workflow-a",
        confirmed_adapter_operation_ids=["operation-a"],
    )

    assert request_id == "workflow-a"
    assert request.expected_fencing_token == 3
    assert request.confirmed_adapter_operation_ids == ["operation-a"]
