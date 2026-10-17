"""VALIDATE_GPU refuses a Hopper NVSwitch node whose fabric did not train.

Live (round 9, plain EKS, GF-REGIONAL-COLLECT-015): the product's own reboot
came back with Fabric Manager failed ("not all local fabric manager instances
finished their configuration") and every GPU "in progress"; VALIDATE_GPU,
VALIDATE_HOST and VALIDATE_FABRIC all passed and scheduling was restored on a
node where CUDA could not start. VALIDATE_GPU now reads every GPU's fabric
state from the fresh post-action samples (DCGM field 170) and the node's Fabric
Manager unit from the Agent's heartbeat record: training waits, a failed
registration or a state held past the window fails -- and the existing ladder
climbs (reset -> reboot -> replacement).
"""

from __future__ import annotations

import pytest

from gpu_fault.collector_requirements import CollectorServiceState
from gpu_fault.gpu_fabric_manager import (
    FABRIC_MANAGER_HELD_SECONDS,
    FABRIC_MANAGER_STATUS,
    FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS,
)
from gpu_fault.gpu_metrics import GpuMetricSample
from gpu_fault.store.shared.errors import NotFoundError
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

GPUS = ["GPU-a", "GPU-b"]
WINDOW = FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS


def _sample(canonical: str, value: float, uuid: str, model: str) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=canonical,
        canonical_name=canonical,
        value=value,
        gpu_uuid=uuid,
        labels={"modelName": model},
    )


def _context(
    *,
    fabric: int | None,
    held: float | None = None,
    model: str = "NVIDIA H200",
    after: WorkflowOperation = WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    unit: dict[str, str] | None = None,
    heartbeat_after_action: bool = True,
    restored: bool = False,
):
    acted_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    seen_at = acted_at + timedelta(seconds=20)
    restored_at = seen_at + timedelta(seconds=10)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=collector,
                    observed_at=seen_at,
                    ingested_at=seen_at,
                    last_success_at=seen_at,
                )
                for collector in (
                    CollectorKind.GPU_METRICS,
                    CollectorKind.HOST_TELEMETRY,
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

        def get_agent(self, cluster_id, node_id):
            if unit is None:
                raise NotFoundError(f"{cluster_id}/{node_id}")
            return SimpleNamespace(
                collector_services={
                    "nvidia-fabricmanager": CollectorServiceState.model_validate(unit)
                },
                last_seen_at=seen_at if heartbeat_after_action else acted_at,
            )

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            samples = []
            for uuid in GPUS:
                samples.append(_sample("gpu_temperature_c", 35.0, uuid, model))
                if fabric is not None:
                    samples.append(_sample(FABRIC_MANAGER_STATUS, fabric, uuid, model))
                if held is not None:
                    samples.append(
                        _sample(FABRIC_MANAGER_HELD_SECONDS, held, uuid, model)
                    )
            return [
                SimpleNamespace(observed_at=seen_at, sample=item) for item in samples
            ]

        def findings(self, cluster_id, node_id):
            return []

    store = build_store()
    restore = [WorkflowOperation.RESTORE_GPU_SERVICES] if restored else []
    operations = [after, *restore, WorkflowOperation.VALIDATE_GPU]
    incident, workflow = workflow_state(store, operations)
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(0, after, updated_at=acted_at),
            *(
                [workflow_step_execution(1, restore[0], updated_at=restored_at)]
                if restored
                else []
            ),
        ],
    )
    step = copy_model(
        workflow.official_steps[-1],
        execution_owner="gpu-fault-validation-adapter",
        gpu_uuids=GPUS,
    )
    adapter = GpuValidationAdapter(ValidationMetrics(), store=ValidationMetrics.store)
    return adapter, WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=len(operations) - 1,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/nvlink-fabric",
    )


RUNNING = {"active": "active", "enabled": "enabled"}


def test_every_gpu_registered_and_fabric_manager_running_passes() -> None:
    adapter, context = _context(fabric=3, unit=RUNNING)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details


def test_training_that_may_still_finish_waits() -> None:
    adapter, context = _context(fabric=2, held=30.0, unit=RUNNING)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_pending"]["node-a"]["nvlink_fabric_training"] == {
        "GPU-a": "IN_PROGRESS",
        "GPU-b": "IN_PROGRESS",
    }


@pytest.mark.parametrize(
    "after",
    [WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, WorkflowOperation.RESTART_NODE],
)
def test_a_fabric_held_untrained_past_the_window_fails_the_recovery(after) -> None:
    """The round-9 COLLECT-015 shape after a reboot, and the reset's own."""

    adapter, context = _context(fabric=2, held=float(WINDOW), after=after)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome.details
    assert outcome.details["node_failures"] == {
        "node-a": ["nvlink_fabric_not_trained(IN_PROGRESS: 2/2 GPUs)"]
    }
    assert "nvlink_fabric_not_trained" in str(outcome.error)


def test_a_failed_registration_fails_without_waiting() -> None:
    adapter, context = _context(fabric=4, held=0.0)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["node_failures"]["node-a"] == [
        "nvlink_fabric_not_trained(FAILURE: 2/2 GPUs)"
    ]


@pytest.mark.parametrize(
    ("fabric", "model"),
    [
        (None, "NVIDIA H200"),
        (2, "NVIDIA L40S"),
        (2, "NVIDIA GB200"),
        (0, "NVIDIA H200"),
    ],
    ids=["field-not-exported", "no-nvswitch", "multi-node-nvlink", "not-supported"],
)
def test_nothing_graded_is_not_a_failure(fabric, model) -> None:
    adapter, context = _context(
        fabric=fabric,
        held=float(WINDOW * 4) if fabric else None,
        model=model,
        unit={"active": "failed", "enabled": "enabled"},
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details


def test_a_failed_fabric_manager_unit_fails_even_with_a_trained_fabric() -> None:
    adapter, context = _context(
        fabric=3, unit={"active": "failed", "enabled": "enabled"}
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["node_failures"]["node-a"] == ["fabric_manager_unit_failed"]


@pytest.mark.parametrize(
    ("unit", "heartbeat_after_action", "pending"),
    [
        ({"active": "activating", "enabled": "enabled"}, True, "activating"),
        (RUNNING, False, "not_reported_since_action"),
    ],
)
def test_a_starting_or_unreported_unit_waits(
    unit, heartbeat_after_action, pending
) -> None:
    adapter, context = _context(
        fabric=3, unit=unit, heartbeat_after_action=heartbeat_after_action
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_pending"]["node-a"]["fabric_manager_unit"] == pending


@pytest.mark.parametrize(
    "unit",
    [
        None,
        {"active": "inactive", "enabled": "disabled"},
        {"active": "inactive", "enabled": "unknown"},
    ],
    ids=["older-agent", "disabled", "not-a-systemd-unit"],
)
def test_a_unit_systemd_does_not_run_is_not_judged(unit) -> None:
    adapter, context = _context(fabric=3, unit=unit)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details


def test_a_heartbeat_from_inside_the_window_does_not_judge_the_restored_unit() -> None:
    """Reset, then a heartbeat that still sees Fabric Manager down, then the
    RESTORE that starts it: the validation waits for a heartbeat after the
    restore instead of failing on the stale "inactive"."""

    adapter, context = _context(
        fabric=3, unit={"active": "inactive", "enabled": "enabled"}, restored=True
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome.details
    assert outcome.details["node_pending"]["node-a"]["fabric_manager_unit"] == (
        "not_reported_since_action"
    )
