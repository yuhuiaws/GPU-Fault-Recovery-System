"""Explicit hardware evidence, separate from readiness and observation failures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gpu_fault.collector_kinds import collector_silent_thresholds
from gpu_fault.models import RecoveryAction
from gpu_fault.markers import describe_blocking_marker
from gpu_fault.observation_health import observation_reason
from gpu_fault.telemetry import CollectorKind

HARDWARE_PREFIX = "confirmed hardware fault: "


def confirmed_marker(marker: Any) -> bool:
    return (
        marker.trusted
        and 0
        <= (datetime.now(timezone.utc) - marker.observed_at).total_seconds()
        <= 420
        and marker.event_source in {"KERNEL_LOG", "FABRIC_MANAGER_LOG"}
        and marker.action_disposition == "EXECUTABLE"
        and marker.recommended_action
        in {RecoveryAction.RESET_GPU, RecoveryAction.REBOOT_NODE, RecoveryAction.DRAIN}
    )


def confirmed_finding(finding: Any) -> bool:
    observed = getattr(finding, "observed_at", None)
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        return False
    return (
        getattr(finding, "canonical_name", None)
        in {"row_remap_failure", "ecc_dbe_volatile_total", "fabric_manager_status"}
        and getattr(getattr(finding, "severity", None), "value", None) == "CRITICAL"
        and 0 <= (datetime.now(timezone.utc) - observed).total_seconds() <= 420
    )


def fault_reasons(markers: list[Any], findings: list[Any]) -> list[str]:
    reasons = []
    if markers:
        reasons.append(
            "active trusted node fault marker exists: "
            + ", ".join(describe_blocking_marker(marker) for marker in markers)
        )
    if findings:
        reasons.append("active GPU health finding exists")
    proven = [marker.marker_id for marker in markers if confirmed_marker(marker)] + [
        finding.canonical_name for finding in findings if confirmed_finding(finding)
    ]
    if proven:
        reasons.append(HARDWARE_PREFIX + ", ".join(sorted(set(proven))))
    return reasons


def observation_reasons(
    store: Any, cluster_id: str, node_id: str, now: datetime
) -> list[str]:
    statuses = {
        status.collector: status
        for status in store.list_collector_statuses(cluster_id, node_id)
    }
    return [
        f"collector observation UNKNOWN: {collector.value}/{reason}"
        for collector in (CollectorKind.GPU_INVENTORY, CollectorKind.GPU_METRICS)
        if (
            reason := observation_reason(
                statuses.get(collector), now, collector_silent_thresholds()[collector]
            )
        )
        is not None
    ]
