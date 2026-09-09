from gpu_fault.notifications.common import (
    WARM_SPARE_REPLACEMENT_EMAIL_TEMPLATE,
    WARM_SPARE_REPLACEMENT_TEMPLATE_VERSION,
    AdvisoryNotification,
)


class WarmSpareReplacementEmailBuilder:
    """Renders a deduplicated successful warm-spare failover notice."""

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        policy_source: str,
        official_action: str | None,
        effective_action: str | None,
        reasons: list[str],
        operation_id: str,
        fault_node_ids: list[str],
        spare_node_ids: list[str],
        node_rebindings: dict[str, str],
        confirmation_source: str,
        provider_mutation_submitted: bool,
    ) -> AdvisoryNotification:
        mappings = [
            f"  - {node_id} -> {node_rebindings.get(node_id, 'UNKNOWN')}"
            for node_id in fault_node_ids
        ]
        body = WARM_SPARE_REPLACEMENT_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            policy_source=policy_source,
            official_action=official_action or "NONE",
            effective_action=effective_action or "REPLACE_NODE",
            fault_nodes=", ".join(fault_node_ids) or "UNKNOWN",
            spare_nodes=", ".join(spare_node_ids) or "UNKNOWN",
            node_rebindings="\n".join(mappings) or "  - UNKNOWN",
            confirmation_source=confirmation_source,
            provider_mutation_submitted=str(provider_mutation_submitted).lower(),
            operation_id=operation_id,
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            template_version=(WARM_SPARE_REPLACEMENT_TEMPLATE_VERSION),
        )
        return AdvisoryNotification(
            deduplication_key=(f"{incident_id}/warm-spare-replacement/{operation_id}"),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[通知][GPU warm-spare替换成功] {cluster_id}: "
                f"{', '.join(fault_node_ids)} -> "
                f"{', '.join(spare_node_ids)}"
            ),
            body_text=body,
            support_case_draft="",
            category="ACTION_COMPLETED",
            priority=10,
        )
