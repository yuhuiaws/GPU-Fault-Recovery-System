from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import IncidentState, RecoveryAction, Severity, WorkloadState
from gpu_fault.policy import SxidClassification, XidEvent
from tests._builders import (
    build_context,
    build_sxid_event,
    fault_incident,
    node_health_finding,
)

NOW = datetime(2026, 8, 13, 16, 30, tzinfo=timezone.utc)


def _save_finding_marker(
    context: ApplicationContext, finding: NodeHealthFinding
) -> None:
    marker = finding.marker()
    context.completion.add_marker(marker)
    context.store.save_incident(
        fault_incident(
            marker.incident_id,
            finding.event_id,
            "NODE_HEALTH",
            finding.cluster_id,
            node_ids=[finding.node_id],
            gpu_uuids=finding.gpu_uuids,
            policy_version=finding.policy_version,
            policy_source=finding.policy_source,
            effective_action=finding.recommended_action,
            state=IncidentState.DETECTED,
            reasons=[finding.reason],
            created_at=finding.observed_at,
            updated_at=finding.observed_at,
        )
    )


def _xid48() -> XidEvent:
    return XidEvent(
        event_id="kernel-xid48",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=10),
        event_source="NVIDIA_KERNEL_LOG",
        xid=48,
        gpu_uuid="GPU-a",
        pci_bdf="0000:59:00.0",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def test_dcgm_memory_finding_correlates_with_xid48_by_structure() -> None:
    context = build_context()
    finding = node_health_finding(
        "dcgm-memory",
        "dcgm-memory",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason="DCGM field 319 increased by one",
        metric_name="dcgm_ecc_dbe_delta",
        gpu_uuids=["GPU-a"],
        pci_bdf="0000:59:00.0",
        recommended_action=RecoveryAction.RESET_GPU,
        runtime_profile_version="simulated-v1",
    )
    _save_finding_marker(context, finding)

    decision = context.policy.evaluate_xid(_xid48())
    correlated = context.orchestrator.correlate_provider_event(
        decision, cluster_id="cluster-a"
    )

    assert finding.reason != decision.marker.raw_reason
    assert finding.marker().fault_class == "GPU_MEMORY"
    assert decision.marker.fault_class == "GPU_MEMORY"
    assert correlated.duplicate
    assert correlated.marker.incident_id == "inc-dcgm-memory"


def test_gpu_thermal_finding_does_not_correlate_with_xid48() -> None:
    context = build_context()
    finding = node_health_finding(
        "dcgm-temperature",
        "dcgm-temperature",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity=Severity.WARNING,
        reason="GPU temperature exceeded warning threshold",
        metric_name="gpu_temperature_celsius",
        gpu_uuids=["GPU-a"],
        pci_bdf="0000:59:00.0",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
    )
    _save_finding_marker(context, finding)

    correlated = context.orchestrator.correlate_provider_event(
        context.policy.evaluate_xid(_xid48()), cluster_id="cluster-a"
    )

    assert finding.marker().fault_class == "GPU_THERMAL"
    assert not correlated.duplicate
    assert correlated.marker.incident_id == "inc-kernel-xid48"


def test_efa_and_sxid_do_not_correlate_across_fault_domains() -> None:
    context = build_context()
    finding = node_health_finding(
        "efa-fatal",
        "efa-fatal",
        observed_at=NOW,
        category=NodeHealthCategory.RDMA,
        severity=Severity.CRITICAL,
        reason="EFA RDMA link fatal timeout",
        metric_name="rdma_link_down",
        pci_bdf="0000:ab:00.0",
        recommended_action=RecoveryAction.QUARANTINE,
    )
    _save_finding_marker(context, finding)
    event = build_sxid_event(
        "sxid-fatal",
        NOW + timedelta(seconds=10),
        23001,
        SxidClassification.ALWAYS_FATAL,
        "NVIDIA_FABRIC_MANAGER",
        event_source="FABRIC_MANAGER_LOG",
        product="H200",
        pci_bdf="0000:ab:00.0",
        participating_gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
    )

    decision = context.policy.evaluate_sxid(event)
    correlated = context.orchestrator.correlate_provider_event(
        decision, cluster_id="cluster-a"
    )

    assert finding.marker().fault_class == "NETWORK_FABRIC"
    assert decision.marker.fault_class == "GPU_FABRIC"
    assert not correlated.duplicate
    assert correlated.marker.incident_id == "inc-sxid-fatal"
