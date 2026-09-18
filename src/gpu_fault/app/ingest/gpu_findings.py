from __future__ import annotations

from typing import Any

from gpu_fault.gpu_metric_models import GpuHealthFinding
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import RecoveryAction, Severity


def gpu_node_health_findings(
    context: Any, findings: list[GpuHealthFinding]
) -> list[NodeHealthFinding]:
    """Use the same node-health ingestion for scrape and kernel composites."""
    result = []
    for finding in findings:
        severity = (
            Severity.CRITICAL
            if finding.severity.value == "CRITICAL"
            else Severity.WARNING
        )
        action = (
            RecoveryAction(finding.automatic_action)
            if finding.automatic_action
            else context.orchestrator.gpu_metric_action(
                cluster_id=finding.cluster_id,
                node_id=finding.node_id,
                metric_name=finding.canonical_name,
                has_explicit_gpu=bool(finding.gpu_uuid),
                default=(
                    RecoveryAction.QUARANTINE
                    if severity is Severity.CRITICAL
                    else RecoveryAction.RUN_DIAGNOSTICS
                ),
            )
        )
        result.append(
            NodeHealthFinding(
                finding_id=finding.finding_id,
                event_id=f"gpu-{finding.finding_id}",
                cluster_id=finding.cluster_id,
                node_id=finding.node_id,
                observed_at=finding.observed_at,
                category=NodeHealthCategory.GPU,
                severity=severity,
                reason=(
                    finding.reason
                    if finding.finding_kind == "METRIC"
                    else (
                        f"{finding.reason}; correlation_rule="
                        f"{finding.correlation_rule_id}; "
                        "component_metrics="
                        f"{','.join(finding.component_metrics)}; "
                        f"confidence={finding.confidence}"
                    )
                ),
                recommended_action=action,
                metric_name=finding.canonical_name,
                value=finding.value,
                device=(
                    finding.gpu_uuid
                    or finding.pci_bdf
                    or (
                        ",".join(finding.affected_gpu_uuids)
                        if finding.affected_gpu_uuids
                        else None
                    )
                ),
                pci_bdf=finding.pci_bdf,
                gpu_uuids=(
                    finding.affected_gpu_uuids
                    or ([finding.gpu_uuid] if finding.gpu_uuid else [])
                ),
                evidence_ref=finding.evidence_ref,
                runtime_profile_version=finding.runtime_profile_version,
                workload_state=finding.workload_state,
                affected_workload_ids=finding.affected_workload_ids,
                policy_version=finding.policy_version,
                policy_source=finding.policy_source,
                policy_reference=finding.policy_reference,
                official_action=finding.official_action,
                diagnostic_parameters=(
                    {
                        "correlation_rule_id": finding.correlation_rule_id,
                        "component_finding_ids": finding.component_finding_ids,
                        "component_metrics": finding.component_metrics,
                        "component_evidence_refs": finding.component_evidence_refs,
                        "confidence": finding.confidence,
                    }
                    if finding.finding_kind == "COMPOSITE"
                    else {}
                ),
            )
        )
    return result
