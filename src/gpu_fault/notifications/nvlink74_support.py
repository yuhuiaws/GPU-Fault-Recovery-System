from gpu_fault.notifications.common import (
    NVLINK74_SUPPORT_EMAIL_TEMPLATE,
    NVLINK74_SUPPORT_TEMPLATE_VERSION,
    AdvisoryNotification,
)


class Nvlink74SupportEmailBuilder:
    """Renders the fixed direct-support template for XID 74."""

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        node_ids: list[str],
        workload_ids: list[str],
        reasons: list[str],
        policy_source: str,
        official_action: str | None,
        ticket_id: str,
    ) -> AdvisoryNotification:
        body = NVLINK74_SUPPORT_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            node_ids=", ".join(node_ids) or "UNKNOWN",
            workloads=", ".join(workload_ids) or "NONE",
            event_id=event_id,
            policy_source=policy_source,
            official_action=official_action or "NONE",
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            ticket_id=ticket_id,
            template_version=NVLINK74_SUPPORT_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{incident_id}/xid74-support/{NVLINK74_SUPPORT_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[GPU故障][XID 74 NVLink] {cluster_id} "
                f"{', '.join(node_ids) or 'UNKNOWN'}"
            ),
            body_text=body,
            support_case_draft="\n".join(
                [
                    f"Ticket ID: {ticket_id}",
                    f"Cluster: {cluster_id}",
                    f"Incident: {incident_id}",
                    f"Nodes: {', '.join(node_ids) or 'UNKNOWN'}",
                    "Event: XID 74 NVLink register workflow",
                    "Reasons:",
                    *[f"- {item}" for item in reasons],
                ]
            ),
        )
