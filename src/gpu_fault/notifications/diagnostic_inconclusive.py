from gpu_fault.notifications.common import (
    DIAGNOSTIC_INCONCLUSIVE_EMAIL_TEMPLATE,
    DIAGNOSTIC_INCONCLUSIVE_TEMPLATE_VERSION,
    AdvisoryNotification,
)


class DiagnosticInconclusiveEmailBuilder:
    """Renders the advisory for a diagnostic-only workflow that failed.

    The workflow observed the node and did not conclude; nothing was changed
    or isolated, the incident ended RECOVERED and its markers were retired.
    One notice per workflow: the deduplication key carries the workflow id
    and nothing that varies between two attempts to send it.
    """

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        node_ids: list[str],
        operations: list[str],
        failed_operation: str | None,
        error: str | None,
        policy_source: str,
        official_action: str | None,
        reasons: list[str],
    ) -> AdvisoryNotification:
        nodes = ", ".join(node_ids) or "UNKNOWN"
        body = DIAGNOSTIC_INCONCLUSIVE_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            nodes=nodes,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            policy_source=policy_source,
            official_action=official_action or "NONE",
            operations=" -> ".join(operations) or "NONE",
            failed_operation=failed_operation or "UNKNOWN",
            error=error or "UNKNOWN",
            reasons="\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN",
            template_version=DIAGNOSTIC_INCONCLUSIVE_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{incident_id}/diagnostic-inconclusive/{workflow_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=f"[通知][GPU 诊断未定论] {cluster_id}: {nodes}",
            body_text=body,
            support_case_draft="",
            evidence_refs=[],
            category="ADVISORY",
        )
