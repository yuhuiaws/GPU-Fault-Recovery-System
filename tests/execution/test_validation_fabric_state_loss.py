"""VALIDATE_GPU refuses a node whose quiesce window lost the fabric state (batch 128).

A window that stopped Fabric Manager and reset GPUs one at a time hands the
restore a new Fabric Manager instance that knows only the GPUs it reset: every
multicast team with another GPU fails ("All GPUs in the partition need to be
reset to recover") while the per-GPU fabric state and unicast P2P still look
healthy. An Agent that predates ``fabric_manager=keep`` still stops Fabric
Manager, and so may any later path; VALIDATE_GPU fails such a node with
``nvlink_fabric_state_lost`` unless a full GPU + NVSwitch reset, a reboot or a
replacement re-registered every GPU afterwards. Judged only where batch 127
grades the fabric (a Hopper NVSwitch node reporting field 170); the existing
ladder then climbs to the reboot, whose own VALIDATE_GPU passes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.gpu_fabric_manager import FABRIC_MANAGER_STATUS
from gpu_fault.gpu_fabric_window import fabric_manager_stopped, fabric_state_loss
from gpu_fault.gpu_metrics import GpuMetricSample
from gpu_fault.orchestration.escalation import next_rung
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
    workflow_state,
)

Op = WorkflowOperation
FM = "nvidia-fabricmanager"
GPUS = ["GPU-a", "GPU-b"]
STOPPED = {"node_results": {"node-a": {"active_services": [FM, "nvidia-dcgm"]}}}
KEPT = {
    "node_results": {
        "node-a": {
            "active_services": ["nvidia-dcgm"],
            "kept_services": [{"service": FM, "reason": "per-GPU reset"}],
        }
    }
}
LOST = (
    "nvlink_fabric_state_lost(step 0 stopped nvidia-fabricmanager, step 1 reset "
    "GPUs individually; no full GPU + NVSwitch reset or reboot since)"
)


def _context(
    executions: list[tuple[WorkflowOperation, dict[str, Any]]],
    *,
    model: str = "NVIDIA H200",
    fabric: int | None = 3,
    node_ids: list[str] | None = None,
):
    """A VALIDATE_GPU after ``executions`` (one step each, finished in order)."""

    started = datetime.now(timezone.utc) - timedelta(minutes=10)
    seen_at = started + timedelta(minutes=9)

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
            raise NotFoundError(f"{cluster_id}/{node_id}")

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            samples = []
            for uuid in GPUS:
                for name, value in (
                    ("gpu_temperature_c", 35.0),
                    (FABRIC_MANAGER_STATUS, fabric),
                ):
                    if value is None:
                        continue
                    samples.append(
                        GpuMetricSample(
                            metric_name=name,
                            canonical_name=name,
                            value=value,
                            gpu_uuid=uuid,
                            labels={"modelName": model},
                        )
                    )
            return [
                SimpleNamespace(observed_at=seen_at, sample=item) for item in samples
            ]

        def findings(self, cluster_id, node_id):
            return []

    operations = [operation for operation, _details in executions]
    incident, workflow = workflow_state(build_store(), [*operations, Op.VALIDATE_GPU])
    steps = [
        copy_model(step, node_ids=node_ids or ["node-a"])
        for step in workflow.official_steps
    ]
    workflow = copy_model(
        workflow,
        official_steps=steps,
        step_executions=[
            workflow_step_execution(
                index,
                operation,
                details=details,
                updated_at=started + timedelta(minutes=index + 1),
            )
            for index, (operation, details) in enumerate(executions)
        ],
    )
    step = copy_model(
        steps[-1], execution_owner="gpu-fault-validation-adapter", gpu_uuids=GPUS
    )
    adapter = GpuValidationAdapter(ValidationMetrics(), store=ValidationMetrics.store)
    return adapter, WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=len(operations),
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/fabric-state",
    )


@pytest.mark.parametrize("model", ["NVIDIA H200", "NVIDIA B200", "NVIDIA B300 SXM6 AC"])
def test_a_stopped_fabric_manager_around_a_per_gpu_reset_fails_the_node(model) -> None:
    """The pre-batch-128 Agent's window: quiesce stopped Fabric Manager, one GPU
    reset, the restore started a new instance. HGX B200/B300 (batch 131) are
    judged the same way: their unit stops and starts NVLSM with Fabric
    Manager."""

    adapter, context = _context(
        [
            (Op.QUIESCE_GPU_SERVICES, STOPPED),
            (Op.RESET_GPU, {}),
            (Op.RESTORE_GPU_SERVICES, {"node_results": {"node-a": {}}}),
        ],
        model=model,
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome.details
    assert outcome.details["node_failures"] == {"node-a": [LOST]}
    assert "nvlink_fabric_state_lost" in str(outcome.error)
    assert next_rung(Op.VALIDATE_GPU, {Op.RESET_GPU}) is Op.RESTART_NODE, (
        "the existing ladder climbs to the reboot"
    )


@pytest.mark.parametrize(
    "executions",
    [
        [(Op.QUIESCE_GPU_SERVICES, KEPT), (Op.RESET_GPU, {})],
        [(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.RESET_ALL_GPUS_NVSWITCHES, {})],
        [(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.RESET_GPU, {}), (Op.RESTART_NODE, {})],
        [(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.RESET_GPU, {}), (Op.REPLACE_NODE, {})],
        [(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.UPDATE_SOFTWARE_FIRMWARE, {})],
        [
            (
                Op.QUIESCE_GPU_SERVICES,
                {"node_results": {"node-a": {"inactive_services": [FM]}}},
            ),
            (Op.RESET_GPU, {}),
        ],
        [(Op.QUIESCE_GPU_SERVICES, {}), (Op.RESET_GPU, {})],
    ],
    ids=[
        "kept-running",
        "full-fabric-reset",
        "reboot-after",
        "replacement-after",
        "no-reset-in-the-window",
        "fabric-manager-was-not-running",
        "receipt-names-no-services",
    ],
)
def test_a_window_that_kept_or_re_registered_the_fabric_passes(executions) -> None:
    adapter, context = _context(executions)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details


@pytest.mark.parametrize(
    ("model", "fabric"),
    [
        ("NVIDIA L40S", 3),
        ("NVIDIA GB200", 3),
        ("NVIDIA GB300", 3),
        ("NVIDIA H100 80GB HBM3", 0),
        ("NVIDIA H200", None),
        ("NVIDIA B200", None),
    ],
    ids=[
        "no-nvswitch",
        "multi-node-nvlink",
        "gb300-nvl72",
        "not-supported",
        "field-not-exported",
        "b200-field-not-exported",
    ],
)
def test_a_node_whose_fabric_is_not_graded_is_not_judged(model, fabric) -> None:
    adapter, context = _context(
        [(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.RESET_GPU, {})],
        model=model,
        fabric=fabric,
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details


def test_a_trained_fabric_with_a_failure_reports_both() -> None:
    adapter, context = _context(
        [(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.RESET_GPU, {})], fabric=4
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["node_failures"]["node-a"] == sorted(
        ["nvlink_fabric_not_trained(FAILURE: 2/2 GPUs)", LOST]
    )


def _workflow(executions, *, node_ids=None):
    return _context(executions, node_ids=node_ids)[1].workflow


def test_an_earlier_loss_is_not_cleared_by_a_later_window_that_kept_it() -> None:
    workflow = _workflow(
        [
            (Op.QUIESCE_GPU_SERVICES, STOPPED),
            (Op.RESET_GPU, {}),
            (Op.QUIESCE_GPU_SERVICES, KEPT),
            (Op.RESET_GPU, {}),
        ]
    )

    assert fabric_state_loss(workflow, "node-a") == LOST


def test_only_the_node_the_window_acted_on_is_judged() -> None:
    workflow = _workflow([(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.RESET_GPU, {})])

    assert fabric_state_loss(workflow, "node-a") == LOST
    assert fabric_state_loss(workflow, "node-b") is None


def test_failed_or_safety_phase_records_say_nothing() -> None:
    workflow = _workflow([(Op.QUIESCE_GPU_SERVICES, STOPPED), (Op.RESET_GPU, {})])
    failed = copy_model(
        workflow,
        step_executions=[
            workflow.step_executions[0],
            copy_model(workflow.step_executions[1], status=WorkflowStepStatus.FAILED),
        ],
    )
    safety = copy_model(
        workflow,
        step_executions=[
            copy_model(item, phase="safety") for item in workflow.step_executions
        ],
    )

    assert fabric_state_loss(failed, "node-a") is None
    assert fabric_state_loss(safety, "node-a") is None


@pytest.mark.parametrize(
    ("details", "stopped"),
    [
        (STOPPED, True),
        (KEPT, False),
        ({"stopped_services": {"node-a": [FM]}}, True),
        (
            {
                "stopped_services": {"node-a": [FM]},
                "kept_services": {"node-a": [{"service": FM, "reason": "x"}]},
            },
            False,
        ),
        ({"stopped_services": {"node-b": [FM]}}, False),
        ({"node_results": {"node-a": {"already_quiesced": True}}}, False),
        ({"node_results": "garbled"}, False),
    ],
    ids=[
        "receipt-stopped",
        "receipt-kept",
        "lifted-stopped",
        "lifted-kept",
        "another-node",
        "receipt-without-services",
        "garbled",
    ],
)
def test_the_quiesce_evidence_is_read_from_the_receipt_or_the_step(
    details, stopped
) -> None:
    assert fabric_manager_stopped(details, "node-a") is stopped
