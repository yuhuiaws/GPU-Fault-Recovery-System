from gpu_fault.notifications.common import (
    AdvisoryNotification,
    HyperPodRecoveryAdvisory,
)


class HyperPodAdvisoryEmailBuilder:
    """Renders an advisory and an optional AWS Support Case draft."""

    def build(
        self,
        advisory: HyperPodRecoveryAdvisory,
        *,
        cluster_name: str,
        incident_id: str,
        node_ids: list[str],
        issue_summary: str,
        region_name: str | None = None,
    ) -> AdvisoryNotification:
        nodes = ", ".join(node_ids) if node_ids else "unknown"
        rationale = "\n".join(f"- {item}" for item in advisory.rationale)
        evidence = "\n".join(f"- {item}" for item in advisory.evidence_refs) or "- none"
        location = f"{cluster_name} ({region_name})" if region_name else cluster_name
        case_draft = "\n".join(
            [
                "Service: Amazon SageMaker HyperPod",
                f"Cluster: {location}",
                f"Incident: {incident_id}",
                f"Affected nodes: {nodes}",
                f"Issue: {issue_summary}",
                (f"Recommended action: {advisory.recommended_action}"),
                f"Current execution owner: {advisory.execution_owner}",
                "Diagnostic rationale:",
                rationale,
                "Evidence references:",
                evidence,
                (
                    "Request: Please review the managed recovery "
                    "behavior and advise whether provider-side "
                    "remediation or hardware replacement is required."
                ),
            ]
        )
        body = "\n".join(
            [
                "GPU failure automation advisory",
                "",
                f"Cluster: {location}",
                f"Incident: {incident_id}",
                f"Affected nodes: {nodes}",
                f"Summary: {issue_summary}",
                (f"Recommendation: {advisory.recommended_action}"),
                f"Disposition: {advisory.disposition.value}",
                f"Execution owner: {advisory.execution_owner}",
                "",
                "Rationale:",
                rationale,
                "",
                "Evidence references:",
                evidence,
                "",
                (
                    "No recovery action or AWS Support case was "
                    "created by this notification."
                ),
                (
                    "Decision owner: customer administrator. Review "
                    "the following optional AWS Support Case draft:"
                ),
                "",
                case_draft,
            ]
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{incident_id}/{advisory.capability.value}/"
                f"{advisory.recommended_action}"
            ),
            cluster_name=cluster_name,
            incident_id=incident_id,
            subject=(
                f"[GPU advisory] {cluster_name} {incident_id}: "
                f"{advisory.recommended_action}"
            ),
            body_text=body,
            support_case_draft=case_draft,
            evidence_refs=advisory.evidence_refs,
        )
