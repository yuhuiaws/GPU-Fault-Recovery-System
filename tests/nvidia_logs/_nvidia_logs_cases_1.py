from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSample,
    GpuMetricSource,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
from gpu_fault.policy import SxidClassification, SxidLinkScope
from gpu_fault.telemetry import CollectorKind
from tests._builders import asgi_client, build_context, gpu_metric_batch

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def test_kernel_ingestion_wakes_dispatcher() -> None:
    context = build_context()
    wake_calls = []
    context.dispatcher.wake = lambda: wake_calls.append(True)

    async def scenario():
        async with asgi_client(context) as client:
            return await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "kmsg-wake-14",
                    "observed_at": NOW.isoformat(),
                    "message": "NVRM: Xid (PCI:0000:b9:00): 14",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                },
            )

    response = asyncio.run(scenario())

    assert response.status_code == 200
    assert wake_calls == [True]


def test_kernel_ingestion_rejects_blank_node_identity() -> None:
    context = build_context()

    async def scenario(field: str):
        async with asgi_client(context) as client:
            payload = {
                "cluster_id": "hp-cluster",
                "node_id": "worker-1",
                "record_id": "kmsg-blank-14",
                "observed_at": NOW.isoformat(),
                "message": "NVRM: Xid (PCI:0000:b9:00): 14",
            }
            payload[field] = ""
            return await client.post("/v1/collector-events/nvidia-kernel", json=payload)

    for field in ("cluster_id", "node_id"):
        assert asyncio.run(scenario(field)).status_code == 422
    assert not context.store.list_raw_evidence("hp-cluster")


def test_kernel_event_identity_is_scoped_by_cluster() -> None:
    normalizer = NvidiaLogNormalizer()

    def normalized(cluster_id: str):
        return normalizer.normalize_kernel(
            NvidiaKernelLogEvent(
                cluster_id=cluster_id,
                node_id="shared-node",
                record_id="shared-record",
                observed_at=NOW,
                message="NVRM: Xid (PCI:0000:b9:00): 11",
                product="H100",
                driver_branch=575,
                cuda_version="12.9",
                runtime_profile_version="simulated-v1",
            )
        )

    first = normalized("cluster-a")
    repeated = normalized("cluster-a")
    other = normalized("cluster-b")

    assert first.xid_events[0].event_id == (repeated.xid_events[0].event_id)
    assert first.xid_events[0].event_id != other.xid_events[0].event_id
    assert first.xid_events[0].event_id.startswith(
        "kernel-log-shared-record-xid-11-cluster-"
    )


def test_same_kernel_record_in_two_clusters_creates_two_incidents() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            responses = []
            for cluster_id in ("cluster-a", "cluster-b"):
                responses.append(
                    await client.post(
                        "/v1/collector-events/nvidia-kernel",
                        json={
                            "cluster_id": cluster_id,
                            "node_id": "shared-node",
                            "record_id": "shared-record",
                            "observed_at": NOW.isoformat(),
                            "message": ("NVRM: Xid (PCI:0000:b9:00): 11"),
                            "product": "H100",
                            "driver_branch": 575,
                            "cuda_version": "12.9",
                            "runtime_profile_version": "simulated-v1",
                        },
                    )
                )

        assert all(response.status_code == 200 for response in responses)
        decisions = [response.json()["decisions"][0] for response in responses]
        assert decisions[0]["event_id"] != decisions[1]["event_id"]
        assert not decisions[0]["duplicate"]
        assert not decisions[1]["duplicate"]
        incidents = [
            context.store.get_incident_by_event(item["event_id"]) for item in decisions
        ]
        assert [item.cluster_id for item in incidents] == ["cluster-a", "cluster-b"]

    asyncio.run(scenario())


def test_sxid_accepts_official_summary_and_rejects_continuation() -> None:
    normalizer = NvidiaLogNormalizer()
    fatal = normalizer.normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="sxid-fatal",
            observed_at=NOW,
            message=(
                "nvidia-nvswitch3: "
                "SXid (PCI:0000:c1:00.0): 11001, Fatal, "
                "Link 46 ingress invalid command"
            ),
            product="H200",
        )
    )
    unknown = normalizer.normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="sxid-unknown",
            observed_at=NOW,
            message=(
                "nvidia-nvswitch3: "
                "SXid (PCI:0000:c1:00.0): 11001, Severity 0 "
                "Engine instance 46 Sub-engine instance 00"
            ),
            product="H200",
        )
    )

    assert fatal.sxid_events[0].classification is (SxidClassification.FATAL)
    assert fatal.sxid_events[0].classification_source == "NVIDIA_FABRIC_MANAGER_CATALOG"
    assert fatal.sxid_events[0].link_scope is SxidLinkScope.UNKNOWN
    assert fatal.sxid_events[0].switch_id == "3"
    assert fatal.sxid_events[0].pci_bdf == "0000:c1:00.0"
    assert fatal.sxid_events[0].port == "46"
    assert not unknown.sxid_events
    assert any(
        "classification is absent" in reason
        for reason in (unknown.provider_signals[0].unresolved_reasons)
    )


def test_kernel_always_fatal_sxid_uses_local_gpu_inventory() -> None:
    async def scenario() -> None:
        context = build_context()
        context.gpu_metrics.ingest(
            gpu_metric_batch(
                "h200-inventory",
                NOW,
                GpuMetricSource.NVIDIA_SMI,
                [
                    GpuMetricSample(
                        metric_name="gpu_identity",
                        canonical_name="gpu_identity",
                        value=1,
                        gpu_index=str(index),
                        gpu_uuid=f"GPU-H200-{index}",
                    )
                    for index in range(8)
                ],
                cluster_id="hp-cluster",
                node_id="worker-1",
                product="H200",
            )
        )
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "kmsg-fatal-trunk-sxid",
                    "observed_at": NOW.isoformat(),
                    "message": (
                        "nvidia-nvswitch3: "
                        "SXid (PCI:0000:c1:00.0): 12020, "
                        "Fatal, Link 46 egress sequence ID error"
                    ),
                    "product": "H200",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )

        assert response.status_code == 200, response.text
        normalized = response.json()["normalized"]["sxid_events"][0]
        decision = response.json()["decisions"][0]
        assert normalized["classification_source"] == ("NVIDIA_FABRIC_MANAGER_CATALOG")
        assert normalized["classification"] == "ALWAYS_FATAL"
        assert normalized["switch_id"] == "3"
        assert normalized["port"] == "46"
        assert normalized["fabric_partition"] == ("hp-cluster/worker-1/local-nvswitch")
        assert normalized["participating_gpu_uuids"] == [
            f"GPU-H200-{index}" for index in range(8)
        ]
        assert decision["disposition"] == "EXECUTABLE"
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        operations = [step.operation for step in workflow.official_steps]
        assert WorkflowOperation.RESTART_NODE in operations
        assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES not in operations

    asyncio.run(scenario())


def test_fabric_manager_log_endpoint_creates_sxid_workflow() -> None:
    async def scenario() -> None:
        context = build_context()
        context.gpu_metrics.ingest(
            gpu_metric_batch(
                "fm-h200-inventory",
                NOW,
                GpuMetricSource.NVIDIA_SMI,
                [
                    GpuMetricSample(
                        metric_name="gpu_identity",
                        canonical_name="gpu_identity",
                        value=1,
                        gpu_index=str(index),
                        gpu_uuid=f"GPU-H200-{index}",
                    )
                    for index in range(8)
                ],
                cluster_id="hp-cluster",
                node_id="worker-1",
                product="H200",
            )
        )
        payload = {
            "cluster_id": "hp-cluster",
            "node_id": "worker-1",
            "record_id": "fm-journal-cursor-1",
            "observed_at": NOW.isoformat(),
            "collected_at": NOW.isoformat(),
            "message": (
                "nvidia-nvswitch3: "
                "SXid (PCI:0000:c1:00.0): 12020, "
                "Fatal, Link 46 egress sequence ID error"
            ),
            "source": "journal",
            "unit": "nvidia-fabricmanager.service",
            "runtime_profile_version": "simulated-v1",
            "product": "H200",
            "workload_state": "IDLE",
            "evidence_ref": ("journal://worker-1/cursor-1"),
        }
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/fabric-manager", json=payload
            )
            replay = await client.post(
                "/v1/collector-events/fabric-manager", json=payload
            )

        assert response.status_code == 200, response.text
        assert replay.status_code == 200, replay.text
        normalized = response.json()["normalized"]["sxid_events"][0]
        decision = response.json()["decisions"][0]
        replay_decision = replay.json()["decisions"][0]
        assert normalized["event_source"] == "FABRIC_MANAGER_LOG"
        assert normalized["classification_source"] == ("NVIDIA_FABRIC_MANAGER_CATALOG")
        assert decision["disposition"] == "EXECUTABLE"
        assert replay_decision["incident_id"] == decision["incident_id"]
        assert replay_decision["workflow_request_id"] == decision["workflow_request_id"]
        persisted = context.store.get_xid_policy_decision(normalized["event_id"])
        assert persisted is not None
        assert persisted.model_dump(mode="json") == replay_decision
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        operations = [step.operation for step in workflow.official_steps]
        assert WorkflowOperation.RESTART_NODE in operations
        assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES not in operations
        assert len(context.store.list_workflows(limit=100)) == 1
        status = next(
            item
            for item in context.store.list_collector_statuses("hp-cluster", "worker-1")
            if item.collector is CollectorKind.FABRIC_MANAGER_LOG
        )
        assert status.sample_count == 1

    asyncio.run(scenario())


def test_fabric_manager_sxid10003_resolves_local_fabric_reset() -> None:
    async def scenario() -> None:
        context = build_context()
        context.gpu_metrics.ingest(
            gpu_metric_batch(
                "fm-h200-sxid10003-inventory",
                NOW,
                GpuMetricSource.NVIDIA_SMI,
                [
                    GpuMetricSample(
                        metric_name="gpu_identity",
                        canonical_name="gpu_identity",
                        value=1,
                        gpu_index=str(index),
                        gpu_uuid=f"GPU-H200-{index}",
                    )
                    for index in range(8)
                ],
                cluster_id="hp-cluster",
                node_id="worker-1",
                product="H200",
            )
        )
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/fabric-manager",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "fm-sxid10003",
                    "observed_at": NOW.isoformat(),
                    "collected_at": NOW.isoformat(),
                    "message": (
                        "nvidia-nvswitch3: "
                        "SXid (PCI:0000:c1:00.0): 10003, "
                        "Fatal, Link 46 Host_unhandled_interrupt"
                    ),
                    "source": "file",
                    "runtime_profile_version": "simulated-v1",
                    "product": "H200",
                    "workload_state": "IDLE",
                    "evidence_ref": ("file:///var/log/fabricmanager.log#10003"),
                },
            )

        assert response.status_code == 200, response.text
        normalized = response.json()["normalized"]["sxid_events"][0]
        decision = response.json()["decisions"][0]
        assert normalized["link_scope"] == "ACCESS"
        assert normalized["fabric_partition"] == ("hp-cluster/worker-1/local-nvswitch")
        assert normalized["participating_gpu_uuids"] == [
            f"GPU-H200-{index}" for index in range(8)
        ]
        assert decision["disposition"] == "EXECUTABLE"
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in {
            step.operation for step in workflow.official_steps
        }

    asyncio.run(scenario())


def test_fabric_manager_rejects_sxid_severity_conflict() -> None:
    async def scenario() -> None:
        context = build_context()
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/fabric-manager",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "fm-invalid-20001-fatal",
                    "observed_at": NOW.isoformat(),
                    "message": (
                        "nvidia-nvswitch3: "
                        "SXid (PCI:0000:c1:00.0): 20001, Fatal, "
                        "Trunk Link 46 egress sequence ID error"
                    ),
                    "source": "file",
                    "product": "H200",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                    "evidence_ref": ("file:///var/log/fabricmanager.log#1:2:3"),
                },
            )

        assert response.status_code == 200, response.text
        normalized = response.json()["normalized"]["sxid_events"][0]
        decision = response.json()["decisions"][0]
        assert normalized["classification"] == "FATAL"
        assert normalized["link_scope"] == "ACCESS"
        assert normalized["link_scope_source"] == ("NVIDIA_PRODUCT_INVARIANT")
        assert decision["disposition"] == "BLOCKED_MISSING_EVIDENCE"
        assert decision["official_action"] == "IGNORE"
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        operations = {step.operation for step in workflow.official_steps}
        assert WorkflowOperation.QUARANTINE in operations
        assert not operations.intersection(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                WorkflowOperation.RESTART_NODE,
            }
        )

    asyncio.run(scenario())


def test_fabric_manager_sxid_19084_requires_full_reset() -> None:
    async def scenario() -> None:
        context = build_context()
        context.gpu_metrics.ingest(
            gpu_metric_batch(
                "fm-19084-inventory",
                NOW,
                GpuMetricSource.NVIDIA_SMI,
                [
                    GpuMetricSample(
                        metric_name="gpu_identity",
                        canonical_name="gpu_identity",
                        value=1,
                        gpu_index=str(index),
                        gpu_uuid=f"GPU-H200-{index}",
                    )
                    for index in range(8)
                ],
                cluster_id="hp-cluster",
                node_id="worker-1",
                product="H200",
            )
        )
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/fabric-manager",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "fm-19084",
                    "observed_at": NOW.isoformat(),
                    "message": (
                        "nvidia-nvswitch3: "
                        "SXid (PCI:0000:c1:00.0): 19084, "
                        "Non-fatal, Link 46 AN1 Heartbeat Timeout"
                    ),
                    "source": "file",
                    "product": "H200",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )

        assert response.status_code == 200, response.text
        decision = response.json()["decisions"][0]
        assert decision["disposition"] == "EXECUTABLE"
        assert decision["official_action"] == ("RESET_ALL_GPUS_AND_NVSWITCHES")
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        operations = {step.operation for step in workflow.official_steps}
        assert WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE in operations
        assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in operations

    asyncio.run(scenario())


def test_fabric_manager_full_reset_uses_mandatory_inventory_channel() -> None:
    """SXID scope does not depend on the metrics health-summary."""

    async def scenario() -> None:
        context = build_context()
        context.store.save_gpu_inventory_snapshot(
            GpuInventorySnapshot(
                snapshot_id="fm-19084-inventory",
                cluster_id="hp-cluster",
                node_id="worker-1",
                observed_at=NOW - timedelta(seconds=120),
                source=GpuMetricSource.NVIDIA_SMI,
                source_boot_id="boot-a",
                devices=[
                    GpuInventoryDevice(
                        gpu_index=index,
                        gpu_uuid=f"GPU-H200-{index}",
                        pci_bdf=f"0000:{0x59 + index:02x}:00.0",
                        product="H200",
                    )
                    for index in range(8)
                ],
                expected_gpu_count=8,
            )
        )
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/fabric-manager",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "fm-19084-cadence-gap",
                    "observed_at": NOW.isoformat(),
                    "message": (
                        "nvidia-nvswitch3: "
                        "SXid (PCI:0000:c1:00.0): 19084, "
                        "Non-fatal, Link 46 AN1 Heartbeat Timeout"
                    ),
                    "source": "file",
                    "product": "H200",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )

        assert response.status_code == 200, response.text
        normalized = response.json()["normalized"]["sxid_events"][0]
        decision = response.json()["decisions"][0]
        assert normalized["fabric_partition"] == ("hp-cluster/worker-1/local-nvswitch")
        assert normalized["participating_gpu_uuids"] == [
            f"GPU-H200-{index}" for index in range(8)
        ]
        assert decision["disposition"] == "EXECUTABLE"
        assert decision["reasons"] == [
            "NVIDIA SXID 19084 rule requires a coordinated reset of "
            "all local GPUs and NVSwitches"
        ]
        workflow = context.store.get_workflow(decision["workflow_request_id"])
        assert workflow.blocked_reasons == []
        assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in {
            step.operation for step in workflow.official_steps
        }

    asyncio.run(scenario())


def test_kernel_captures_xid94() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-sequence-42",
            observed_at=NOW,
            message=("NVRM: Xid (PCI:0000:b9:00): 94, pid=1234, name=python"),
            product="H100",
            driver_branch=575,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
            workload_state="ACTIVE",
            affected_workload_ids=["training-job-94"],
        )
    )

    assert result.provider_signals[0].source.value == "KERNEL_LOG"
    assert result.xid_events[0].xid == 94
    assert result.xid_events[0].pci_bdf == "0000:b9:00"
    assert result.xid_events[0].affected_workload_ids == ["training-job-94"]


def test_kernel_drill_id_is_parsed_out_of_raw_message() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-drill",
            observed_at=NOW,
            message=(
                "NVRM: Xid (PCI:0000:b9:00): 54, "
                "pid=1234, name=python, "
                "drill_id=maintenance-20260811"
            ),
            product="H100",
            driver_branch=575,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
            workload_state="IDLE",
        )
    )

    assert result.xid_events[0].drill_id == ("maintenance-20260811")


def test_kernel_xid74_extracts_exact_seven_registers() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-xid74",
            observed_at=NOW,
            message=(
                "NVRM: Xid (PCI:0000:b9:00): 74, "
                "pid=1234, name=python, Link 3, "
                "0x100 0x0 0x40 0x0 0x100000 0x0 0x0"
            ),
            product="H100",
            driver_branch=575,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
        )
    )

    assert result.xid_events[0].registers == [0x100, 0, 0x40, 0, 0x100000, 0, 0]
    assert result.xid_events[0].nvlink_link_id == 3
    assert result.xid_events[0].nvlink_link_identity_source == "explicit-kernel-message"


def test_kernel_xid74_does_not_invent_missing_link_identity() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-xid74-no-link",
            observed_at=NOW,
            message=(
                "NVRM: Xid (PCI:0000:b9:00): 74, "
                "pid=1234, name=python, "
                "0x10 0x0 0x0 0x0 0x0 0x0 0x0"
            ),
            product="H100",
            driver_branch=575,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
        )
    )

    assert result.xid_events[0].nvlink_link_id is None
    assert result.xid_events[0].nvlink_link_identity_source is None


def test_kernel_nvlink5_extracts_official_register_payload() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-xid145",
            observed_at=NOW,
            message=(
                "NVRM: Xid (PCI:0000:b9:00): 145, RLW, "
                "nonfatal, cross-contain=0, injected=0, Link 3 "
                "(0x00000004 0x00000001 0x00000000 "
                "0x00000000 0x00000000 0x00000000 0x00000000)"
            ),
            product="B200",
            driver_branch=575,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
        )
    )

    event = result.xid_events[0]
    assert event.intr_info == 0x00000004
    assert event.error_status == 0x00000001


def test_kernel_nvlink5_accepts_fixed_width_unprefixed_hex() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-xid144-unprefixed",
            observed_at=NOW,
            message=(
                "NVRM: Xid (PCI:0000:b9:00): 144, SAW fatal 0 0 "
                "Link 1 (00000021, 00000002, 00000000, "
                "00000000, 00000000, 00000000, 00000000)"
            ),
            product="B200",
            driver_branch=570,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
        )
    )

    event = result.xid_events[0]
    assert event.intr_info == 0x21
    assert event.error_status == 0x2


def test_kernel_nvlink5_incomplete_payload_remains_missing() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-xid150-incomplete",
            observed_at=NOW,
            message=(
                "NVRM: Xid (PCI:0000:b9:00): 150, MSE fatal 0 0 "
                "Link 2 (0x00000001 0x00000002)"
            ),
            product="B200",
            driver_branch=575,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
        )
    )

    event = result.xid_events[0]
    assert event.intr_info is None
    assert event.error_status is None


def test_kernel_nvlink5_does_not_use_registers_from_another_xid() -> None:
    result = NvidiaLogNormalizer().normalize_kernel(
        NvidiaKernelLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="kmsg-multiple-xids",
            observed_at=NOW,
            message=(
                "NVRM: Xid (PCI:0000:b9:00): 145, RLW; "
                "NVRM: Xid (PCI:0000:b9:00): 146, TLW fatal 0 0 "
                "Link 3 (0x00000004 0x00000001 0x00000000 "
                "0x00000000 0x00000000 0x00000000 0x00000000)"
            ),
            product="B200",
            driver_branch=575,
            cuda_version="12.9",
            runtime_profile_version="simulated-v1",
        )
    )

    events = {event.xid: event for event in result.xid_events}
    assert events[145].intr_info is None
    assert events[145].error_status is None
    assert events[146].intr_info == 0x4
    assert events[146].error_status == 0x1


def test_fabric_manager_and_kernel_observations_merge_into_one_incident() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            fabric_manager = await client.post(
                "/v1/collector-events/fabric-manager",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "fm-duplicate-94",
                    "source": "journal",
                    "observed_at": NOW.isoformat(),
                    "message": "NVRM: Xid (PCI:0000:b9:00): 94",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "ACTIVE",
                    "affected_workload_ids": ["training-job-94"],
                },
            )
            kernel = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "record_id": "kmsg-duplicate-94",
                    "observed_at": NOW.isoformat(),
                    "message": (
                        "NVRM: Xid (PCI:0000:b9:00): 94, pid=1234, name=python"
                    ),
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "ACTIVE",
                    "affected_workload_ids": ["training-job-94"],
                },
            )

            assert fabric_manager.status_code == 200
            assert kernel.status_code == 200
            fm_decision = fabric_manager.json()["decisions"][0]
            kernel_decision = kernel.json()["decisions"][0]
            assert kernel_decision["duplicate"]
            assert kernel_decision["incident_id"] == (fm_decision["incident_id"])
            assert (
                kernel_decision["workflow_request_id"]
                == (fm_decision["workflow_request_id"])
            )

    asyncio.run(scenario())
