"""RESTART_FM on a Hopper NVSwitch node is NVIDIA's H100-and-later restart (batch 128).

XID 45 alone finalizes as the catalog's RESTART_FM. On a node whose NVLink
fabric batch 127 grades (a Hopper SXM part on NVSwitches, field 170 reported),
a Fabric Manager restart alone loses the fabric state: the restarted service
needs every GPU of the fabric reset (Fabric Manager user guide), and the
documented procedure is stop the CUDA applications and GPU services, stop
Fabric Manager, ``nvidia-smi -r``, start Fabric Manager. The plan there is the
fatal-trunk SXID's chain on the node's graded GPUs; the window stops Fabric
Manager (it is a full reset), the restore starts it, both validations run.
Every other node keeps FREEZE_EVIDENCE -> RESTART_FABRIC_MANAGER.
"""

from __future__ import annotations

import pytest

from gpu_fault.gpu_fabric_manager import FABRIC_MANAGER_STATUS, graded_fabric_gpus
from gpu_fault.gpu_fabric_window import quiesce_command_parameters
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from tests._builders import gpu_metric_batch

from ._support import (
    NOW,
    ApplicationContext,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
    event,
)

Op = WorkflowOperation
GPUS = [f"GPU-{index}" for index in range(8)]
FABRIC_RESTART_PLAN = [
    Op.FREEZE_EVIDENCE,
    Op.MARK_UNSCHEDULABLE,
    Op.QUIESCE_GPU_SERVICES,
    Op.VERIFY_NO_GPU_CLIENTS,
    Op.RESET_ALL_GPUS_NVSWITCHES,
    Op.RESTORE_GPU_SERVICES,
    Op.VALIDATE_GPU,
    Op.VALIDATE_FABRIC,
    Op.RESTORE_SCHEDULING,
]


def _report(context: ApplicationContext, model: str, status: float) -> None:
    samples = [
        GpuMetricSample(
            metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
            canonical_name=FABRIC_MANAGER_STATUS,
            value=status,
            gpu_uuid=uuid,
            labels={"modelName": model},
        )
        for uuid in GPUS
    ]
    GpuMetricsService(store=context.store).ingest(
        gpu_metric_batch(
            "dcgm-fabric",
            NOW,
            GpuMetricSource.DCGM_EXPORTER,
            samples,
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
        )
    )


def _solo_xid45(context: ApplicationContext, **values):
    xid_event = event(45, event_id=values.pop("event_id", "solo-xid45"), **values)
    decision = context.policy.evaluate_xid(xid_event, xid45_window_closed=True)
    assert decision.official_action == "RESTART_FM"
    incident, workflow = context.orchestrator.ingest(xid_event, decision)
    return incident, workflow


def test_a_hopper_nvswitch_node_plans_the_full_gpu_and_nvswitch_reset(
    context: ApplicationContext,
) -> None:
    _report(context, "NVIDIA H200", 3.0)

    incident, workflow = _solo_xid45(context)

    assert incident.state is IncidentState.ACTION_PENDING
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert workflow.official_action == "RESTART_FM"
    assert [step.operation for step in workflow.official_steps] == FABRIC_RESTART_PLAN
    reset = workflow.official_steps[4]
    assert reset.gpu_uuids == GPUS, "every graded GPU, not the XID's one"
    assert reset.parameters == {
        "fabric_partition": "cluster-a/node-a/local-nvswitch",
        "xid": 45,
    }
    assert {tuple(step.gpu_uuids) for step in workflow.official_steps} == {
        tuple(GPUS)
    }, "every step of the chain names the whole fabric"
    quiesce = workflow.official_steps[2]
    assert quiesce_command_parameters(workflow, 2, quiesce) == {}, (
        "a full reset: the window stops Fabric Manager and the restore starts it"
    )


def test_an_active_workload_is_stopped_and_restarted_around_it(
    context: ApplicationContext,
) -> None:
    _report(context, "NVIDIA H100 80GB HBM3", 3.0)

    _incident, workflow = _solo_xid45(
        context,
        event_id="solo-xid45-active",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )

    operations = [step.operation for step in workflow.official_steps]
    assert operations[:4] == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.STOP_WORKLOADS,
        Op.QUIESCE_GPU_SERVICES,
    ]
    assert operations[-1] is Op.RESTART_WORKLOAD
    assert Op.RESTART_FABRIC_MANAGER not in operations


@pytest.mark.parametrize(
    ("model", "status"),
    [
        (None, None),
        ("NVIDIA L40S", 3.0),
        ("NVIDIA B200", 3.0),
        ("NVIDIA GB200", 3.0),
        ("NVIDIA H100 PCIe", 3.0),
        ("NVIDIA H100 80GB HBM3", 0.0),
    ],
    ids=[
        "no-field-170",
        "no-nvswitch",
        "blackwell-hgx",
        "nvl72",
        "pcie",
        "not-supported-single-gpu",
    ],
)
def test_every_other_node_keeps_the_fabric_manager_restart(
    context: ApplicationContext, model, status
) -> None:
    if model is not None:
        _report(context, model, status)

    _incident, workflow = _solo_xid45(context)

    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert [step.operation for step in workflow.official_steps] == [
        Op.FREEZE_EVIDENCE,
        Op.RESTART_FABRIC_MANAGER,
    ]


def test_an_xid45_that_follows_a_companion_is_not_a_fabric_manager_restart(
    context: ApplicationContext,
) -> None:
    _report(context, "NVIDIA H200", 3.0)
    companion = event(95, event_id="companion-95")
    xid_event = event(45, event_id="not-solo-xid45")
    decision = context.policy.evaluate_xid(xid_event, companion_events=[companion])

    _incident, workflow = context.orchestrator.ingest(xid_event, decision)

    assert decision.official_action == "RESET_GPU"
    assert Op.RESET_ALL_GPUS_NVSWITCHES not in {
        step.operation for step in workflow.official_steps
    }


def test_the_graded_inventory_is_the_gpus_that_report_a_fabric_state() -> None:
    def sample(uuid: str | None, value: float, model: str = "NVIDIA H200"):
        return (
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
                canonical_name=FABRIC_MANAGER_STATUS,
                value=value,
                gpu_uuid=uuid,
                labels={"modelName": model},
            ),
            None,
        )

    assert graded_fabric_gpus([sample("GPU-b", 3.0), sample("GPU-a", 2.0)]) == [
        "GPU-a",
        "GPU-b",
    ]
    assert graded_fabric_gpus([sample("GPU-a", 0.0)]) == [], "not supported"
    assert graded_fabric_gpus([sample("GPU-a", 3.0, "NVIDIA L40S")]) == []
    assert graded_fabric_gpus([sample(None, 3.0)]) == [], "no UUID to reset by"
