"""An untrained NVLink fabric is recovered the NVSwitch way (batch 127).

The node-level ``fabric_manager_status`` finding asks for the coarse
RESET_GPU with the official action RESET_ALL_GPUS_AND_NVSWITCHES -- the fatal
trunk SXID's policy: one reset of every GPU and NVSwitch inside the quiesce
window (Fabric Manager stopped before, started after), VALIDATE_GPU and
VALIDATE_FABRIC, then scheduling back; a failed reset or validation climbs to a
reboot and then a replacement. A chain that owns Fabric Manager (quiesce,
reset, restore, reboot, replacement, Fabric Manager restart) absorbs the
finding its own window raises, and its VALIDATE_GPU judges it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.gpu_fabric_manager import (
    FABRIC_MANAGER_HELD_SECONDS,
    FABRIC_MANAGER_STATUS,
    FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS,
)
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.families.conflicts import (
    DCGM_TELEMETRY_UNAVAILABLE_OPERATIONS,
    FABRIC_MANAGER_OWNED_OPERATIONS,
    INVENTORY_FINDING_VALIDATIONS,
    TRANSIENT_INVENTORY_FINDINGS,
)
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    gpu_metric_batch,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

CLUSTER = "cluster-a"
NOW = datetime.now(timezone.utc).replace(microsecond=0)
GPUS = [f"GPU-{index}" for index in range(8)]
FABRIC_RESET_PLAN = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    WorkflowOperation.RESTORE_GPU_SERVICES,
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_FABRIC,
    WorkflowOperation.RESTORE_SCHEDULING,
]


def _fabric_finding(batch_id: str = "dcgm-1"):
    """The node-health finding the production path books for an untrained
    fabric: exporter samples -> GpuMetricsService -> node-health finding."""

    samples = []
    for uuid in GPUS:
        status = GpuMetricSample(
            metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
            canonical_name=FABRIC_MANAGER_STATUS,
            value=2.0,
            gpu_uuid=uuid,
            labels={"modelName": "NVIDIA H200"},
        )
        held = status.model_copy(
            update={
                "canonical_name": FABRIC_MANAGER_HELD_SECONDS,
                "value": float(FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS + 600),
            }
        )
        samples += [status, held]
    batch = gpu_metric_batch(
        batch_id,
        NOW,
        GpuMetricSource.DCGM_EXPORTER,
        samples,
        cluster_id=CLUSTER,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )
    result = GpuMetricsService(store=build_store()).ingest(batch)
    (graded,) = [
        item
        for item in result.new_findings
        if item.canonical_name == FABRIC_MANAGER_STATUS
    ]
    (finding,) = gpu_node_health_findings(None, [graded])
    return finding


def _store():
    store = build_store()
    store.save_profile(default_simulated_profile())
    return store


def test_an_untrained_fabric_plans_the_full_gpu_and_nvswitch_reset() -> None:
    store = _store()
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    incident, workflow = orchestrator.ingest_node_health(_fabric_finding())

    assert workflow is not None
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert [step.operation for step in workflow.official_steps] == FABRIC_RESET_PLAN
    assert workflow.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
    (reset,) = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    ]
    assert reset.gpu_uuids == GPUS, (
        "the Node Agent refuses a fabric reset whose UUIDs are not the inventory"
    )
    assert incident.effective_action is RecoveryAction.RESET_GPU
    assert WorkflowOperation.RESET_GPU not in {
        step.operation for step in workflow.official_steps
    }, "a single-GPU reset cannot re-register the fabric"


def test_a_plain_reset_gpu_finding_keeps_the_single_gpu_chain() -> None:
    store = _store()
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)
    finding = copy_model(
        _fabric_finding(),
        finding_id="finding-remap",
        event_id="gpu-dcgm-1-GPU-0-row_remap_pending",
        metric_name="row_remap_pending",
        official_action="RESET_GPU_IN_SERVICE_WINDOW",
        gpu_uuids=["GPU-0"],
        device="GPU-0",
    )

    _incident, workflow = orchestrator.ingest_node_health(finding)

    assert workflow is not None
    operations = [step.operation for step in workflow.official_steps]
    assert WorkflowOperation.RESET_GPU in operations
    assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES not in operations
    assert WorkflowOperation.VALIDATE_FABRIC not in operations


def test_the_finding_is_the_chain_s_own_while_it_owns_fabric_manager() -> None:
    covering = TRANSIENT_INVENTORY_FINDINGS[FABRIC_MANAGER_STATUS]
    assert covering is FABRIC_MANAGER_OWNED_OPERATIONS
    assert covering >= DCGM_TELEMETRY_UNAVAILABLE_OPERATIONS | {
        WorkflowOperation.RESTART_FABRIC_MANAGER,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
    }
    assert WorkflowOperation.RUN_DCGM_DIAGNOSTIC not in covering, (
        "a diagnostic reads the fabric; it does not take it down"
    )
    assert INVENTORY_FINDING_VALIDATIONS[FABRIC_MANAGER_STATUS] == {
        WorkflowOperation.VALIDATE_GPU
    }


def _running(store, operations, completed: int) -> None:
    incident = fault_incident(
        "inc-owner",
        "event-owner",
        cluster_id=CLUSTER,
        node_ids=["node-a"],
        workflow_request_id="wf-owner",
        fencing_token=1,
    )
    store.save_incident_and_workflow(
        incident,
        workflow_request(
            "wf-owner",
            "inc-owner",
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            official_steps=[workflow_step(operation) for operation in operations],
            completed_step_indexes=list(range(completed)),
            completed_operations=operations[:completed],
            created_at=NOW,
            updated_at=NOW,
        ),
    )


@pytest.mark.parametrize(
    "owner",
    [
        FABRIC_RESET_PLAN,
        [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
    ],
    ids=["fabric-reset", "reboot"],
)
def test_a_running_chain_absorbs_the_finding_its_window_raises(owner) -> None:
    store = _store()
    _running(store, owner, completed=owner.index(WorkflowOperation.VALIDATE_GPU))
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    incident, workflow = orchestrator.ingest_node_health(_fabric_finding("dcgm-9"))

    assert incident.incident_id == "inc-owner"
    assert workflow is not None and workflow.request_id == "wf-owner"
    assert [item.request_id for item in store.list_workflows()] == ["wf-owner"], (
        "no second fabric reset is queued behind the chain that is training it"
    )


def test_a_chain_past_its_gpu_validation_does_not_absorb_the_finding() -> None:
    store = _store()
    _running(
        store,
        FABRIC_RESET_PLAN,
        completed=FABRIC_RESET_PLAN.index(WorkflowOperation.VALIDATE_GPU) + 1,
    )
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    incident, _workflow = orchestrator.ingest_node_health(_fabric_finding("dcgm-9"))

    assert incident.incident_id != "inc-owner", (
        "a fabric that broke after the chain validated it is a new fault"
    )


def _failed_validation(after: WorkflowOperation):
    operations = [after, WorkflowOperation.VALIDATE_GPU]
    workflow = workflow_request(
        "wf-recovery",
        "inc-recovery",
        status=WorkflowStatus.FAILED,
        official_steps=[workflow_step(operation) for operation in operations],
    )
    return copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(0, after),
            copy_model(
                workflow_step_execution(1, WorkflowOperation.VALIDATE_GPU),
                status=WorkflowStepStatus.FAILED,
                details={
                    "failed_nodes": ["node-a"],
                    "node_failures": {
                        "node-a": ["nvlink_fabric_not_trained(IN_PROGRESS: 8/8 GPUs)"]
                    },
                    "validation": "VALIDATE_GPU",
                },
            ),
        ],
    )


@pytest.mark.parametrize(
    ("after", "stage", "action"),
    [
        (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            "reset_validation",
            RecoveryAction.REBOOT_NODE,
        ),
        (
            WorkflowOperation.RESTART_NODE,
            "reboot_validation",
            RecoveryAction.REPLACE_NODE,
        ),
    ],
)
def test_a_fabric_that_does_not_train_climbs_the_existing_ladder(
    after, stage, action
) -> None:
    verdict = HardwareEscalationService.classify(_failed_validation(after))

    assert verdict is not None
    assert (verdict[0], verdict[1]) == (stage, action)
