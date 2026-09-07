"""Post-action GPU validation must see every targeted GPU, not just one."""

from __future__ import annotations

from tests._builders import build_store, copy_model, workflow_step_execution

from ._support import (
    CollectorKind,
    CollectorStatus,
    GpuValidationAdapter,
    SimpleNamespace,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    datetime,
    timedelta,
    timezone,
    workflow_state,
)


def _validation_context(store, sample_uuids: list[str], reset_at):
    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=reset_at + timedelta(seconds=5),
                    ingested_at=reset_at + timedelta(seconds=5),
                    last_success_at=reset_at + timedelta(seconds=5),
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=reset_at + timedelta(seconds=5),
                    sample=SimpleNamespace(
                        canonical_name="gpu_temperature_c", gpu_uuid=uuid
                    ),
                )
                for uuid in sample_uuids
            ]

        def findings(self, cluster_id, node_id):
            return []

    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESET_GPU, WorkflowOperation.VALIDATE_GPU]
    )
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.RESET_GPU, updated_at=reset_at)
        ],
    )
    step = copy_model(
        workflow.official_steps[1],
        execution_owner="gpu-fault-validation-adapter",
        gpu_uuids=["GPU-a", "GPU-b"],
    )
    adapter = GpuValidationAdapter(ValidationMetrics(), store=ValidationMetrics.store)
    return adapter, WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=1,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/per-gpu",
    )


def test_validate_gpu_waits_until_every_targeted_gpu_reports_after_reset() -> None:
    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    adapter, context = _validation_context(build_store(), ["GPU-a"], reset_at)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_pending"]["node-a"][
        "missing_recent_metrics_by_gpu"
    ] == {"GPU-b": ["gpu_temperature_c"]}


def test_validate_gpu_succeeds_when_every_targeted_gpu_reports() -> None:
    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    adapter, context = _validation_context(build_store(), ["GPU-a", "GPU-b"], reset_at)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED


def test_validate_gpu_does_not_let_a_sibling_gpu_answer_for_a_missing_one() -> None:
    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    adapter, context = _validation_context(
        build_store(), ["GPU-a", "GPU-a", None], reset_at
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["pending_nodes"] == ["node-a"]
