"""A fatal access-link SXID resets the port's GPU and every GPU of the affected workload.

NVIDIA Fabric Manager User Guide, "Runtime NVSwitch and GPU Errors" >
"NVSwitch SXid Errors" > "Fatal SXid Errors" (page "Last updated on Oct 07,
2026"),
https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#fatal-sxid-errors:
"When a fatal SXid error is reported on a NVSwitch port that connects a GPU and
an NVSwitch, the corresponding error will be propagated to the GPU. The CUDA
jobs that are running on that GPU will be aborted" -- and the recovery
procedure's first step: "Reset the specified GPU and all the participating GPUs
in the affected workload".

``FaultIngestionService._enrich_sxid_scope`` built that set only when the
trusted NVSwitch port topology named no GPU: with the port's GPU from the
topology it stopped there, so RESET_GPU, VALIDATE_GPU and the marker's GPU
containment covered one GPU of a job that ran on four, and the job's other GPUs
went back to the scheduler with the error state the guide says must be cleared
first. Without the topology (the H100/H200 product invariant names the scope,
not the GPU) the set was every GPU of the node's active jobs and never the port's
GPU. Now the two are merged: the port's GPU, the GPU in the NVSwitch's PCI slot
and every GPU of the node's active managed jobs -- the jobs the same workflow
stops.

Driven through the real collector route, normalizer, topology, policy and
planner.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.app.ingest.faults import (
    NVIDIA_FM_FATAL_ACCESS_SXID_QUOTE,
    NVIDIA_FM_FATAL_ACCESS_SXID_RECOVERY_QUOTE,
    NVIDIA_FM_FATAL_SXID_REFERENCE,
)
from gpu_fault.channel_registry import FABRIC_MANAGER_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.nvidia_logs import FabricManagerLogEvent, NvidiaKernelLogEvent
from gpu_fault.telemetry import TelemetryMetricLatest
from tests._builders import attempt_observation, build_context, container_observation
from tests.app_services._cov95_runtime_ingest import post

NOW = datetime.now(timezone.utc).replace(microsecond=0)
GPUS = [f"GPU-{index}" for index in range(8)]
#: The job's four GPUs on the node; GPU 5 is idle.
JOB_GPUS = GPUS[:4]


def _inventory(context: Any) -> None:
    context.store.save_gpu_inventory_snapshot(
        GpuInventorySnapshot(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=NOW,
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id="boot-a",
            expected_gpu_count=len(GPUS),
            devices=[
                GpuInventoryDevice(
                    gpu_index=index,
                    gpu_uuid=gpu,
                    pci_bdf=f"0000:{0x18 + 0x10 * index:02x}:00.0",
                )
                for index, gpu in enumerate(GPUS)
            ],
        )
    )


def _job(context: Any, gpus: list[str]) -> None:
    """``job-a`` running one container per GPU on the node: rank ``i`` holds
    ``gpus[i]`` (one Pod per rank, as torchrun elastic launches lay it out)."""

    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            NOW,
            expected_critical_ranks=len(gpus),
            workload_ids=["training/job/job-a"],
            containers=[
                container_observation(
                    f"pod-{index}", f"worker-{index}", index, "node-a", gpu_uuids=[gpu]
                )
                for index, gpu in enumerate(gpus)
            ],
        )
    )


def _port_topology(context: Any, gpu: str) -> None:
    """Trusted port topology: switch 0 port 3 is the access link of ``gpu``."""

    context.store.observe_telemetry_metric(
        TelemetryMetricLatest(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=NOW - timedelta(seconds=1),
            name="nvswitch_port_topology",
            value=1,
            device="nvidia-nvswitch0/3",
            labels={
                "trusted": "true",
                "switch_id": "nvidia-nvswitch0",
                "port": "3",
                "link_scope": "ACCESS",
                "fabric_partition": "fabric-a",
                "gpu_uuid": gpu,
            },
        )
    )


def _fatal_access_sxid(context: Any, path: str) -> tuple[dict, dict, Any]:
    values: dict[str, Any] = {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "record_id": f"access-{path.rsplit('/', 1)[-1]}",
        "observed_at": NOW,
        "message": "nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11001, Fatal, Link 3",
        "product": "H200",
        "runtime_profile_version": "simulated-v1",
    }
    event = (
        NvidiaKernelLogEvent(**values, source_boot_id="boot-a")
        if path == NVIDIA_KERNEL_PATH
        else FabricManagerLogEvent(**values, source="journal", collected_at=NOW)
    )
    response = post(context, path, event)
    assert response.status_code == 200, response.text
    body = response.json()
    (normalized,) = body["normalized"]["sxid_events"]
    (decision,) = body["decisions"]
    return (
        normalized,
        decision,
        context.store.get_workflow(decision["workflow_request_id"]),
    )


@pytest.mark.parametrize("path", [NVIDIA_KERNEL_PATH, FABRIC_MANAGER_PATH])
@pytest.mark.parametrize(
    ("port_gpu", "job_gpus", "expected"),
    [
        # The port's GPU runs the job: the job's other three GPUs join it.
        ("GPU-0", JOB_GPUS, JOB_GPUS),
        # The port's GPU is idle: it is reset beside the job's GPUs, which the
        # workflow stops all the same.
        ("GPU-5", JOB_GPUS, [*JOB_GPUS, "GPU-5"]),
        # The H100/H200 product invariant names the scope, not the GPU.
        (None, JOB_GPUS, JOB_GPUS),
        # An idle node: the port's GPU alone.
        ("GPU-5", [], ["GPU-5"]),
    ],
    ids=["port-gpu-in-job", "idle-port-gpu", "no-port-topology", "idle-node"],
)
def test_the_reset_and_its_containment_cover_the_port_gpu_and_the_jobs_gpus(
    path: str, port_gpu: str | None, job_gpus: list[str], expected: list[str]
) -> None:
    context = build_context()
    _inventory(context)
    if job_gpus:
        _job(context, job_gpus)
    if port_gpu is not None:
        _port_topology(context, port_gpu)

    normalized, decision, workflow = _fatal_access_sxid(context, path)

    assert normalized["link_scope"] == "ACCESS", normalized
    assert normalized["link_scope_source"] == (
        "TRUSTED_NVSWITCH_TOPOLOGY" if port_gpu else "NVIDIA_PRODUCT_INVARIANT"
    ), normalized
    assert normalized["participating_gpu_uuids"] == sorted(expected), (
        "the access-link reset must name the port's GPU and every GPU of the "
        f"affected workload: {normalized['participating_gpu_uuids']}"
    )
    assert decision["official_action"] == "RESET_PARTICIPATING_GPUS", decision
    assert decision["disposition"] == "EXECUTABLE", decision
    assert sorted(decision["marker"]["scope"]["gpu_uuids"]) == sorted(expected), (
        "the marker contains exactly the GPUs the reset covers"
    )
    scoped = {
        step.operation: sorted(step.gpu_uuids)
        for step in workflow.official_steps
        if step.operation
        in {
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.VALIDATE_GPU,
        }
    }
    assert scoped == {
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS: sorted(expected),
        WorkflowOperation.RESET_GPU: sorted(expected),
        WorkflowOperation.VALIDATE_GPU: sorted(expected),
    }, scoped
    if job_gpus:
        stops = [
            step
            for step in workflow.official_steps
            if step.operation is WorkflowOperation.STOP_WORKLOADS
        ]
        assert [step.workload_ids for step in stops] == [["training/job/job-a"]], (
            "every job whose GPUs are reset is stopped first"
        )


def test_the_fabric_manager_evidence_is_pinned() -> None:
    assert NVIDIA_FM_FATAL_SXID_REFERENCE == (
        "https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/"
        "index.html#fatal-sxid-errors"
    )
    assert NVIDIA_FM_FATAL_ACCESS_SXID_QUOTE == (
        "When a fatal SXid error is reported on a NVSwitch port that connects a GPU "
        "and an NVSwitch, the corresponding error will be propagated to the GPU. "
        "The CUDA jobs that are running on that GPU will be aborted"
    )
    assert NVIDIA_FM_FATAL_ACCESS_SXID_RECOVERY_QUOTE == (
        "Reset the specified GPU and all the participating GPUs in the affected "
        "workload"
    )
