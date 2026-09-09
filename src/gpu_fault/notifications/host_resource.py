from gpu_fault.notifications.common import (
    HOST_RESOURCE_EVENT_EMAIL_TEMPLATE,
    HOST_RESOURCE_EVENT_TEMPLATE_VERSION,
    AdvisoryNotification,
)


class HostResourceEventEmailBuilder:
    """Renders a fixed notification for a sustained host resource risk."""

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
        value: float | None,
        device: str | None,
        reason: str,
        workload_state: str,
        workload_ids: list[str],
        recommended_action: str,
        policy_source: str,
        policy_version: str,
        workflow_steps: list[str],
        diagnostic_parameters: dict,
        evidence_refs: list[str],
    ) -> AdvisoryNotification:
        body = HOST_RESOURCE_EVENT_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            node_id=node_id,
            incident_id=incident_id,
            workflow_id=workflow_id or "NONE",
            event_id=event_id,
            observed_at=observed_at,
            severity=severity,
            signal=diagnostic_parameters.get("signal", "UNKNOWN"),
            metric_name=metric_name,
            device=device or "NODE",
            value=value if value is not None else "UNKNOWN",
            comparison=diagnostic_parameters.get("comparison", "UNKNOWN"),
            threshold_percent=diagnostic_parameters.get("threshold_percent", "UNKNOWN"),
            minimum_active_seconds=diagnostic_parameters.get(
                "minimum_active_seconds", "UNKNOWN"
            ),
            reason=reason,
            related_metrics=diagnostic_parameters.get("related_metrics", {}),
            workload_state=workload_state,
            workloads=", ".join(workload_ids) or "NONE",
            recommended_action=recommended_action,
            workflow_steps=" -> ".join(workflow_steps) or "NONE",
            policy_source=policy_source,
            policy_version=policy_version,
            evidence_refs=(
                "\n".join(f"- {item}" for item in evidence_refs) or "- NONE"
            ),
            template_version=HOST_RESOURCE_EVENT_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{event_id}/host-resource-event/{HOST_RESOURCE_EVENT_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(f"[GPU故障][节点资源隐患] {cluster_id} {node_id} {metric_name}"),
            body_text=body,
            support_case_draft="",
            evidence_refs=evidence_refs,
        )
