from gpu_fault.notifications.common import (
    HARDWARE_INVENTORY_EMAIL_TEMPLATE,
    HARDWARE_INVENTORY_TEMPLATE_VERSION,
    AdvisoryNotification,
)


class HardwareInventoryEmailBuilder:
    """Renders the fixed GPU/EFA inventory mismatch notification."""

    def build(
        self,
        *,
        cluster_id: str,
        node_id: str,
        incident_id: str,
        workflow_id: str | None,
        event_id: str,
        observed_at: str,
        severity: str,
        metric_name: str,
        workload_state: str,
        workload_ids: list[str],
        recommended_action: str,
        policy_source: str,
        policy_version: str,
        workflow_steps: list[str],
        inventory: dict,
        evidence_refs: list[str],
    ) -> AdvisoryNotification:
        resource_type = str(
            inventory.get(
                "resource",
                "GPU" if metric_name.startswith("gpu_") else "EFA",
            )
        )
        body = HARDWARE_INVENTORY_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            node_id=node_id,
            incident_id=incident_id,
            workflow_id=workflow_id or "NONE",
            event_id=event_id,
            observed_at=observed_at,
            node_instance_type=inventory.get("node_instance_type", "UNKNOWN"),
            resource_type=resource_type,
            severity=severity,
            metric_name=metric_name,
            expected_count=inventory.get("expected_count", "UNKNOWN"),
            observed_count=inventory.get("observed_count", "UNKNOWN"),
            discovered_count=inventory.get("discovered_count", "UNKNOWN"),
            missing_count=inventory.get("missing_count", "UNKNOWN"),
            excess_count=inventory.get("excess_count", "UNKNOWN"),
            consecutive_samples=inventory.get(
                "consecutive_mismatch_samples", "UNKNOWN"
            ),
            required_samples=inventory.get("required_consecutive_samples", "UNKNOWN"),
            workload_state=workload_state,
            workloads=", ".join(workload_ids) or "NONE",
            recommended_action=recommended_action,
            workflow_steps=" -> ".join(workflow_steps) or "NONE",
            policy_source=policy_source,
            policy_version=policy_version,
            evidence_refs=(
                "\n".join(f"- {item}" for item in evidence_refs) or "- NONE"
            ),
            template_version=HARDWARE_INVENTORY_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{event_id}/hardware-inventory/{HARDWARE_INVENTORY_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(f"[GPU故障][{resource_type}掉卡] {cluster_id} {node_id}"),
            body_text=body,
            support_case_draft="",
            evidence_refs=evidence_refs,
        )
