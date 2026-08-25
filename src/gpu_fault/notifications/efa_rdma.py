from gpu_fault.notifications.common import (
    AdvisoryNotification,
    EFA_RDMA_EVENT_EMAIL_TEMPLATE,
    EFA_RDMA_EVENT_TEMPLATE_VERSION,
)


class EfaRdmaEventEmailBuilder:
    """Renders the fixed notification for an EFA/RDMA finding."""

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
        job_id: str | None,
        attempt_id: str | None,
        recommended_action: str,
        policy_source: str,
        policy_version: str,
        workflow_steps: list[str],
        diagnostic_parameters: dict,
        evidence_refs: list[str],
    ) -> AdvisoryNotification:
        is_traffic = metric_name == "efa_traffic_bytes_per_second"
        traffic_signal = str(
            diagnostic_parameters.get("diagnostic_reason", "NONE")
        ).removeprefix("EFA_TRAFFIC_")
        baseline = diagnostic_parameters.get("baseline_bytes_per_second")
        body = EFA_RDMA_EVENT_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            node_id=node_id,
            incident_id=incident_id,
            workflow_id=workflow_id or "NONE",
            event_id=event_id,
            observed_at=observed_at,
            event_type=("EFA_TRAFFIC_ANOMALY" if is_traffic else "RDMA_LINK_OR_ERROR"),
            severity=severity,
            metric_name=metric_name,
            device=device or "NODE",
            value=value if value is not None else "UNKNOWN",
            traffic_signal=traffic_signal if is_traffic else "NONE",
            baseline=baseline if baseline is not None else "NONE",
            reason=reason,
            workload_state=workload_state,
            job_id=job_id or "NONE",
            attempt_id=attempt_id or "NONE",
            workloads=", ".join(workload_ids) or "NONE",
            recommended_action=recommended_action,
            policy_source=policy_source,
            policy_version=policy_version,
            workflow_steps=(" -> ".join(workflow_steps) or "NONE"),
            evidence_refs=(
                "\n".join(f"- {item}" for item in evidence_refs) or "- NONE"
            ),
            template_version=EFA_RDMA_EVENT_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{event_id}/efa-rdma-event/{EFA_RDMA_EVENT_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(f"[GPU故障][EFA/RDMA] {cluster_id} {node_id} {metric_name}"),
            body_text=body,
            support_case_draft="",
            evidence_refs=evidence_refs,
        )
