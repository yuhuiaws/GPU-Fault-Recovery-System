from gpu_fault.notifications.common import (
    AdvisoryNotification,
    NOT_APPLICABLE_EMAIL_TEMPLATE,
    NOT_APPLICABLE_TEMPLATE_VERSION,
)


class NotApplicableEmailBuilder:
    """Renders a field-only notification for NOT_APPLICABLE XIDs."""

    def build(
        self,
        *,
        cluster_id: str,
        node_id: str,
        incident_id: str,
        event_id: str,
        xid: int,
        product: str | None,
        driver_branch: int | None,
        cuda_version: str | None,
        observed_at: str,
        workload_state: str,
        workload_ids: list[str],
        official_action: str | None,
        investigatory_action: str | None = None,
        policy_source: str,
        policy_version: str,
        reasons: list[str],
        safety_action: str | None,
        workflow_id: str,
        workflow_status: str,
        safety_steps: list[str],
        official_steps: list[str],
        evidence_refs: list[str],
    ) -> AdvisoryNotification:
        body = NOT_APPLICABLE_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            node_id=node_id,
            xid=xid,
            product=product or "UNKNOWN",
            driver_branch=(
                str(driver_branch) if driver_branch is not None else "UNKNOWN"
            ),
            cuda_version=cuda_version or "UNKNOWN",
            observed_at=observed_at,
            event_id=event_id,
            workload_state=workload_state,
            workloads=", ".join(workload_ids) or "NONE",
            official_action=official_action or "NONE",
            investigatory_action=investigatory_action or "NONE",
            policy_source=policy_source,
            policy_version=policy_version,
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - NONE"),
            safety_action=safety_action or "NONE",
            incident_id=incident_id,
            workflow_id=workflow_id,
            workflow_status=workflow_status,
            safety_steps=", ".join(safety_steps) or "NONE",
            official_steps=", ".join(official_steps) or "NONE",
            evidence_refs=(
                "\n".join(f"- {item}" for item in evidence_refs) or "- NONE"
            ),
            template_version=NOT_APPLICABLE_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{incident_id}/xid-{xid}/not-applicable/{policy_version}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(f"[GPU故障][NOT_APPLICABLE] {cluster_id} {node_id} XID {xid}"),
            body_text=body,
            support_case_draft=(
                f"Cluster: {cluster_id}\n"
                f"Node: {node_id}\n"
                f"Incident: {incident_id}\n"
                f"XID: {xid}\n"
                "Disposition: NOT_APPLICABLE\n"
                f"Policy version: {policy_version}"
            ),
            evidence_refs=evidence_refs,
        )
