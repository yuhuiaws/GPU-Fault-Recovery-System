from gpu_fault.notifications.common import (
    AdvisoryNotification,
    HARDWARE_ESCALATION_EMAIL_TEMPLATE,
    HARDWARE_ESCALATION_TEMPLATE_VERSION,
)


class HardwareEscalationEmailBuilder:
    """Renders the fixed terminal hardware-escalation template."""

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        event_type: str,
        node_ids: list[str],
        workload_ids: list[str],
        reasons: list[str],
        failed_operations: list[str],
        policy_source: str,
        official_action: str | None,
        ticket_id: str,
    ) -> AdvisoryNotification:
        body = HARDWARE_ESCALATION_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            node_ids=", ".join(node_ids) or "UNKNOWN",
            workloads=", ".join(workload_ids) or "NONE",
            event_type=event_type,
            event_id=event_id,
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            failed_operations=(", ".join(failed_operations) or "NONE_RECORDED"),
            ticket_id=ticket_id,
            policy_source=policy_source,
            official_action=official_action or "NONE",
            template_version=HARDWARE_ESCALATION_TEMPLATE_VERSION,
        )
        case_draft = "\n".join(
            [
                f"Ticket ID: {ticket_id}",
                f"Cluster: {cluster_id}",
                f"Incident: {incident_id}",
                f"Nodes: {', '.join(node_ids) or 'UNKNOWN'}",
                "Disposition: HARDWARE_OFFLINE",
                (
                    "Failed operations: "
                    + (", ".join(failed_operations) or "NONE_RECORDED")
                ),
                "Reasons:",
                *[f"- {item}" for item in reasons],
            ]
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{incident_id}/hardware-escalation/"
                f"{HARDWARE_ESCALATION_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[GPU故障][硬件下线] {cluster_id} {', '.join(node_ids) or 'UNKNOWN'}"
            ),
            body_text=body,
            support_case_draft=case_draft,
        )
