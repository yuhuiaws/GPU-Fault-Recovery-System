from gpu_fault.notifications.common import (
    XID_INVESTIGATORY_EMAIL_TEMPLATE,
    XID_INVESTIGATORY_TEMPLATE_VERSION,
    AdvisoryNotification,
    render_evidence_refs,
)


class XidInvestigatoryEmailBuilder:
    """Renders the fixed per-XID administrator decision template."""

    def build(
        self,
        *,
        cluster_id: str,
        node_id: str,
        incident_id: str,
        workflow_id: str | None,
        event_id: str,
        xid: int,
        product: str | None,
        observed_at: str,
        workload_state: str,
        workload_ids: list[str],
        disposition: str,
        official_action: str | None,
        effective_action: str | None,
        investigatory_action: str | None,
        policy_source: str,
        policy_version: str,
        reasons: list[str],
        evidence_refs: list[str],
        occurrence_event_id: str | None = None,
    ) -> AdvisoryNotification:
        """``occurrence_event_id`` names the first report of this XID occurrence
        when ``event_id`` is its other source's report (the kernel line and the
        DCGM field-230 change, ``policy/xid_cross_source``): the two share one
        deduplication key, so the second report finds the first one's notice
        instead of mailing the same fault again (batch 238)."""

        body = XID_INVESTIGATORY_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            node_id=node_id,
            xid=xid,
            product=product or "UNKNOWN",
            observed_at=observed_at,
            event_id=event_id,
            incident_id=incident_id,
            workflow_id=workflow_id or "NONE",
            workload_state=workload_state,
            workloads=", ".join(workload_ids) or "NONE",
            disposition=disposition,
            official_action=official_action or "NONE",
            effective_action=effective_action or "NONE",
            investigatory_action=investigatory_action or "NONE",
            policy_source=policy_source,
            policy_version=policy_version,
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - NONE"),
            evidence_refs=render_evidence_refs(evidence_refs, node_id=node_id),
            template_version=XID_INVESTIGATORY_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{occurrence_event_id or event_id}/xid-investigatory/"
                f"{XID_INVESTIGATORY_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[GPU FAULT][XID {xid} investigatory action] {cluster_id} {node_id}"
            ),
            body_text=body,
            support_case_draft="",
            evidence_refs=evidence_refs,
        )
