from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSample,
    GpuMetricSource,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.telemetry import CollectorKind, EvidenceKind
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    container_observation,
    copy_model,
    gpu_metric_batch,
)

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def official_cloudwatch_message(xid: int) -> str:
    return json.dumps(
        {
            "level": "info",
            "ts": "2026-07-20T10:00:01Z",
            "msg": "NPD caught event: %v",
            "details: ": {
                "severity": "warn",
                "timestamp": "2026-07-20T10:00:00Z",
                "reason": "XidHardwareFailure",
                "message": (
                    "Node condition NvidiaErrorReboot is now: True, "
                    "reason: XidHardwareFailure, message: "
                    f'"NVRM: Xid (PCI:0000:b9:00): {xid}, '
                    'pid=<unknown>, name=<unknown>"'
                ),
            },
            "HealthMonitoringAgentDetectionEvent": "HealthEvent",
        }
    )


def test_kernel_event_uses_observed_workload_topology() -> None:
    context = build_context()
    context.store.save_attempt_observation(
        attempt_observation(
            "training-job-11",
            "training-job-11-a001",
            NOW,
            cluster_id="hp-cluster",
            workload_ids=["gpu-fault-system/pytorchjob/training-job-11"],
            containers=[
                container_observation(
                    "pod-11",
                    "training-job-11-master-0",
                    0,
                    "worker-1",
                    container_name="pytorch",
                    role="master",
                    gpu_uuids=["GPU-11"],
                )
            ],
        )
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "kmsg-boot-11-41011",
                    "observed_at": NOW.isoformat(),
                    "source_boot_id": "boot-11",
                    "source_monotonic_us": 123456789,
                    "message": (
                        "NVRM: Xid (PCI:0000:59:00): 11, pid=1234, name=python"
                    ),
                    "product": "H200",
                },
            )

        assert response.status_code == 200, response.text
        event = response.json()["normalized"]["xid_events"][0]
        assert event["workload_state"] == "ACTIVE"
        assert event["affected_workload_ids"] == [
            "gpu-fault-system/pytorchjob/training-job-11"
        ]
        assert event["runtime_profile_version"] == "simulated-v1"

    asyncio.run(scenario())

    statuses = context.store.list_collector_statuses("hp-cluster")
    assert len(statuses) == 1
    assert statuses[0].collector is CollectorKind.NVIDIA_KERNEL
    assert statuses[0].batch_id == "kmsg-boot-11-41011"

    evidence = context.store.list_raw_evidence("hp-cluster", node_id="worker-1")
    assert len(evidence) == 1
    assert evidence[0].kind is EvidenceKind.NVIDIA_KERNEL
    assert evidence[0].record_id == ("nvidia-kernel/kmsg-boot-11-41011")
    assert evidence[0].attempt_ids == ["training-job-11-a001"]
    assert evidence[0].payload["affected_workload_ids"] == [
        "gpu-fault-system/pytorchjob/training-job-11"
    ]


@pytest.mark.parametrize("identity_age_seconds", [30, 330])
def test_kernel_xid_resolves_recent_unique_pci_to_gpu_uuid(
    identity_age_seconds: int,
) -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            metrics = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=gpu_metric_batch(
                    f"gpu-identity-worker-1-{identity_age_seconds}",
                    NOW - timedelta(seconds=identity_age_seconds),
                    GpuMetricSource.NVIDIA_SMI,
                    [
                        GpuMetricSample(
                            metric_name="gpu_identity",
                            canonical_name="gpu_identity",
                            value=1,
                            gpu_index="0",
                            gpu_uuid="GPU-H200-0",
                            pci_bdf="00000000:59:00.0",
                        )
                    ],
                    cluster_id="hp-cluster",
                    node_id="worker-1",
                    product="H200",
                    workload_state="IDLE",
                ).model_dump(mode="json"),
            )
            kernel = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": (f"kmsg-xid95-worker-1-{identity_age_seconds}"),
                    "observed_at": NOW.isoformat(),
                    "message": "NVRM: Xid (PCI:0000:59:00): 95",
                    "product": "H200",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )

        assert metrics.status_code == 200, metrics.text
        assert kernel.status_code == 200, kernel.text
        event = kernel.json()["normalized"]["xid_events"][0]
        decision = kernel.json()["decisions"][0]
        assert event["gpu_uuid"] == "GPU-H200-0"
        assert decision["action"] == "RESET_GPU"
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        reset = next(
            step
            for step in workflow.official_steps
            if step.operation.value == "RESET_GPU"
        )
        assert reset.gpu_uuids == ["GPU-H200-0"]
        assert workflow.status.value == "PENDING"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("event_boot_id", "expected_gpu_uuid"), [("boot-a", "GPU-H200-0"), ("boot-b", None)]
)
def test_kernel_xid_uses_boot_fenced_inventory_channel(
    event_boot_id: str, expected_gpu_uuid: str | None
) -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            inventory = await client.post(
                "/v1/collector-events/gpu-inventory",
                json=GpuInventorySnapshot(
                    snapshot_id="inventory-worker-1",
                    cluster_id="hp-cluster",
                    node_id="worker-1",
                    observed_at=NOW - timedelta(seconds=120),
                    source=GpuMetricSource.NVIDIA_SMI,
                    source_boot_id="boot-a",
                    devices=[
                        GpuInventoryDevice(
                            gpu_index=0,
                            gpu_uuid="GPU-H200-0",
                            pci_bdf="00000000:59:00.0",
                            product="H200",
                        )
                    ],
                    expected_gpu_count=1,
                    runtime_profile_version="simulated-v1",
                ).model_dump(mode="json"),
            )
            kernel = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": ("inventory-xid95-" + event_boot_id),
                    "observed_at": NOW.isoformat(),
                    "source_boot_id": event_boot_id,
                    "message": "NVRM: Xid (PCI:0000:59:00): 95",
                    "product": "H200",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )

        assert inventory.status_code == 200, inventory.text
        assert kernel.status_code == 200, kernel.text
        assert (
            kernel.json()["normalized"]["xid_events"][0]["gpu_uuid"]
            == expected_gpu_uuid
        )

    asyncio.run(scenario())


def test_kernel_xid74_first_mechanical_event_resets_gpu() -> None:
    context = build_context()
    context.gpu_metrics.ingest(
        gpu_metric_batch(
            "gpu-xid74-identity",
            NOW - timedelta(seconds=30),
            GpuMetricSource.NVIDIA_SMI,
            [
                GpuMetricSample(
                    metric_name="gpu_identity",
                    canonical_name="gpu_identity",
                    value=1,
                    gpu_index="0",
                    gpu_uuid="GPU-H200-0",
                    pci_bdf="00000000:59:00.0",
                )
            ],
            cluster_id="hp-cluster",
            node_id="worker-1",
            product="H200",
            workload_state="IDLE",
        )
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "kmsg-xid74-worker-1",
                    "observed_at": NOW.isoformat(),
                    "message": (
                        "NVRM: Xid (PCI:0000:59:00): 74, "
                        "pid=1234, name=python, Link 3, "
                        "0x100 0x0 0x0 0x0 0x0 0x0 0x0"
                    ),
                    "product": "H200",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["normalized"]["xid_events"][0]["registers"] == [
            0x100,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        decision = body["decisions"][0]
        assert decision["action"] == "RESET_GPU"
        assert decision["nvlink_link_id"] == 3
        assert decision["nvlink_occurrence_counts"] == {"register1.bit8": 1}
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        operations = [step.operation for step in workflow.official_steps]
        assert WorkflowOperation.CHECK_MECHANICALS not in operations
        assert WorkflowOperation.RESET_GPU in operations
        assert WorkflowOperation.RUN_NVLINK74_WORKFLOW not in operations

    asyncio.run(scenario())


def test_kernel_xid_does_not_guess_from_stale_or_ambiguous_pci() -> None:
    async def unresolved(samples: list[GpuMetricSample], observed_at: datetime) -> dict:
        context = build_context()
        context.gpu_metrics.ingest(
            gpu_metric_batch(
                "gpu-identities",
                observed_at,
                GpuMetricSource.NVIDIA_SMI,
                samples,
                cluster_id="hp-cluster",
                node_id="worker-1",
                product="H200",
                workload_state="IDLE",
            )
        )
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "kmsg-unresolved-xid95",
                    "observed_at": NOW.isoformat(),
                    "message": "NVRM: Xid (PCI:0000:59:00): 95",
                    "product": "H200",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        workflow_id = body["decisions"][0]["workflow_request_id"]
        workflow = context.store.get_workflow(workflow_id)
        body["_workflow_status"] = workflow.status.value
        body["_blocked_reasons"] = workflow.blocked_reasons
        return body

    sample = GpuMetricSample(
        metric_name="gpu_identity",
        canonical_name="gpu_identity",
        value=1,
        gpu_index="0",
        gpu_uuid="GPU-H200-0",
        pci_bdf="00000000:59:00.0",
    )
    stale = asyncio.run(unresolved([sample], NOW - timedelta(minutes=11)))
    ambiguous = asyncio.run(
        unresolved(
            [
                sample,
                copy_model(
                    sample, gpu_index="1", gpu_uuid="GPU-H200-1", pci_bdf="0000:59:00.1"
                ),
            ],
            NOW - timedelta(seconds=30),
        )
    )

    assert stale["normalized"]["xid_events"][0]["gpu_uuid"] is None
    stale_workflow = stale["decisions"][0]["workflow_request_id"]
    assert stale_workflow is not None
    assert stale["_workflow_status"] == "SAFETY_PENDING"
    assert any(
        reason.startswith("RESET_GPU requires an explicit GPU UUID")
        for reason in stale["_blocked_reasons"]
    )
    assert ambiguous["normalized"]["xid_events"][0]["gpu_uuid"] is None
    ambiguous_workflow = ambiguous["decisions"][0]["workflow_request_id"]
    assert ambiguous_workflow is not None
    assert ambiguous["_workflow_status"] == "SAFETY_PENDING"
    assert any(
        reason.startswith("RESET_GPU requires an explicit GPU UUID")
        for reason in ambiguous["_blocked_reasons"]
    )


def test_delayed_cross_source_event_merges_within_five_minutes() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            kernel = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "kmsg-delayed-94",
                    "observed_at": (NOW + timedelta(minutes=4)).isoformat(),
                    "collected_at": (NOW + timedelta(minutes=4)).isoformat(),
                    "source_monotonic_us": 42_000_000,
                    "message": ("NVRM: Xid (PCI:0000:b9:00): 94"),
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "ACTIVE",
                    "affected_workload_ids": ["training-job-94"],
                },
            )
            hma_message = json.loads(official_cloudwatch_message(94))
            hma_message["details: "]["timestamp"] = "2026-07-20T18:00:00+08:00"
            hma = await client.post(
                "/v1/provider-events/hyperpod-hma/cloudwatch",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "log_event_id": "hma-delayed-94",
                    "observed_at": NOW.isoformat(),
                    "collected_at": (NOW + timedelta(minutes=4)).isoformat(),
                    "message": json.dumps(hma_message),
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "ACTIVE",
                    "affected_workload_ids": ["training-job-94"],
                },
            )

        kernel_decision = kernel.json()["decisions"][0]
        hma_decision = hma.json()["decisions"][0]
        assert hma_decision["duplicate"]
        assert hma_decision["incident_id"] == (kernel_decision["incident_id"])
        marker = context.store.list_markers()[-1]
        assert marker.source_event_time.isoformat() == ("2026-07-20T18:00:00+08:00")
        assert marker.collected_at == NOW + timedelta(minutes=4)
        assert marker.ingested_at is not None

    asyncio.run(scenario())


def test_cross_source_pci_mismatch_does_not_merge() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "pci-a",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "observed_at": NOW.isoformat(),
                    "source_event_time": NOW.isoformat(),
                    "collected_at": NOW.isoformat(),
                    "event_source": "CLOUDWATCH_LOG",
                    "xid": 94,
                    "pci_bdf": "0000:b9:00",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )
            second = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "pci-b",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "observed_at": NOW.isoformat(),
                    "collected_at": NOW.isoformat(),
                    "event_source": "KERNEL_LOG",
                    "xid": 94,
                    "pci_bdf": "0000:ba:00",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )
        assert first.json()["incident_id"] != second.json()["incident_id"]
        assert not second.json()["duplicate"]

    asyncio.run(scenario())


def test_same_kernel_source_prefers_monotonic_time() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "monotonic-a",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "observed_at": NOW.isoformat(),
                    "source_monotonic_us": 10_000_000,
                    "source_boot_id": "boot-a",
                    "event_source": "KERNEL_LOG",
                    "xid": 94,
                    "pci_bdf": "0000:b9:00",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )
            second = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "monotonic-b",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "observed_at": (NOW + timedelta(hours=1)).isoformat(),
                    "source_monotonic_us": 15_000_000,
                    "source_boot_id": "boot-a",
                    "event_source": "KERNEL_LOG",
                    "xid": 94,
                    "pci_bdf": "0000:b9:00",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )
            different_boot = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "monotonic-new-boot",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "observed_at": (NOW + timedelta(hours=2)).isoformat(),
                    "source_monotonic_us": 16_000_000,
                    "source_boot_id": "boot-b",
                    "event_source": "KERNEL_LOG",
                    "xid": 94,
                    "pci_bdf": "0000:b9:00",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )

        assert second.json()["duplicate"]
        assert second.json()["incident_id"] == first.json()["incident_id"]
        assert not different_boot.json()["duplicate"]
        assert different_boot.json()["incident_id"] != first.json()["incident_id"]

    asyncio.run(scenario())
