from __future__ import annotations

import hashlib

from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
    NodeHealthIngestionResult,
)
from gpu_fault.models import AdvisoryNotification

#: Findings that exist to put a human in the loop rather than to drive a
#: recovery: an unparsable fault line, a GPU whose identity changed under the
#: workload, a node whose GPU invariant nobody configured. Their incident
#: carries only FREEZE_EVIDENCE / diagnostics, so without a notification the
#: operator would have to find them by browsing incidents.
OPERATOR_REVIEW_METRICS = frozenset(
    {
        "unparsed_xid_line",
        "unparsed_sxid_line",
        "unclassified_sxid",
        "gpu_inventory_identity_changed",
        "gpu_expected_count_unknown",
    }
)


def operator_review_notification(
    incident_id: str, finding: NodeHealthFinding
) -> AdvisoryNotification:
    scope = hashlib.sha256(
        "/".join(
            [
                finding.cluster_id,
                finding.node_id,
                str(finding.metric_name),
                finding.event_id,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return AdvisoryNotification(
        deduplication_key=f"operator-review/{scope}",
        cluster_name=finding.cluster_id,
        incident_id=incident_id,
        subject=(
            f"[GPU fault review] {finding.node_id} "
            f"{finding.metric_name} ({finding.severity.value})"
        ),
        body_text=(
            f"Cluster: {finding.cluster_id}\n"
            f"Node: {finding.node_id}\n"
            f"Finding: {finding.metric_name}\n"
            f"Severity: {finding.severity.value}\n"
            f"Reason: {finding.reason}\n"
            f"Recommended action: {finding.recommended_action.value}\n"
            f"Evidence: {finding.evidence_ref or 'n/a'}"
        ),
        support_case_draft=(
            "Review the frozen evidence for this incident and decide whether "
            "the signal is a real fault or a collector/catalog drift."
        ),
        evidence_refs=[finding.evidence_ref] if finding.evidence_ref else [],
        category="OPERATOR_REVIEW",
        priority=100,
    )


class NodeHealthIngestionService:
    def __init__(self, context) -> None:
        self.context = context

    def ingest(self, batch_id: str, findings) -> NodeHealthIngestionResult:
        markers = []
        incidents = []
        workflows = []
        notifications = []
        host_resource_notification_groups = {}
        for finding in findings:
            # The marker is built before ingestion picks an incident, so it
            # cannot know the pointer: grouping and merging route a finding
            # into an incident that already owns the attempt or the node, and
            # that incident keeps its own id. It used to guess `inc-<event_id>`
            # and be re-pointed afterwards, which left a marker pointing at a
            # record nobody ever persisted whenever ingestion raised between
            # the two writes (F-G6 / P0-50B). It is now written first as an
            # *observational* marker with no incident pointer -- nothing may
            # observe the node without a marker while ingestion is deciding,
            # and `marker_blocks_spare` fails closed on a missing pointer --
            # and re-pointed once the incident is known.
            marker = finding.marker().model_copy(update={"incident_id": ""})
            self.context.completion.add_marker(marker)
            incident, workflow = self.context.orchestrator.ingest_node_health(finding)
            marker = marker.model_copy(update={"incident_id": incident.incident_id})
            self.context.completion.add_marker(marker)
            markers.append(marker.marker_id)
            incidents.append(incident.incident_id)
            if workflow is not None:
                workflows.append(workflow.request_id)
            if finding.metric_name in {
                "gpu_inventory_mismatch",
                "efa_inventory_mismatch",
            }:
                notification = self.context.advisory_notifications.preview_hardware_inventory_event(
                    incident.incident_id, finding
                )
                self.context.advisory_notifications.send(notification.notification_id)
                notifications.append(notification.notification_id)
            elif finding.metric_name in OPERATOR_REVIEW_METRICS:
                notification = self.context.store.save_notification_if_absent(
                    operator_review_notification(incident.incident_id, finding)
                )
                self.context.advisory_notifications.send(notification.notification_id)
                notifications.append(notification.notification_id)
            elif (
                finding.metric_name
                in {
                    "cpu_usage_percent",
                    "host_gpu_utilization_percent",
                    "memory_available_percent",
                    "page_cache_percent",
                    "local_filesystem_used_percent",
                }
                and finding.policy_source == "SITE_HOST_RESOURCE_HEALTH"
            ):
                group_key = (
                    finding.cluster_id,
                    finding.node_id,
                    finding.metric_name,
                    tuple(sorted(finding.affected_workload_ids)),
                )
                host_resource_notification_groups.setdefault(group_key, []).append(
                    (incident, finding)
                )
            elif (
                finding.category is NodeHealthCategory.RDMA
                and finding.metric_name is not None
            ):
                notification = (
                    self.context.advisory_notifications.preview_efa_rdma_event(
                        incident.incident_id, finding
                    )
                )
                self.context.advisory_notifications.send(notification.notification_id)
                notifications.append(notification.notification_id)
        for grouped in host_resource_notification_groups.values():
            incident, representative = grouped[0]
            devices = sorted(
                {finding.device for _, finding in grouped if finding.device}
            )
            aggregate = representative.model_copy(
                update={
                    "device": (
                        ", ".join(devices) if devices else representative.device
                    ),
                    "diagnostic_parameters": {
                        **representative.diagnostic_parameters,
                        "affected_device_count": len(devices),
                        "affected_devices": devices,
                    },
                }
            )
            notification = (
                self.context.advisory_notifications.preview_host_resource_event(
                    incident.incident_id, aggregate
                )
            )
            self.context.advisory_notifications.send(notification.notification_id)
            notifications.append(notification.notification_id)
        return NodeHealthIngestionResult(
            batch_id=batch_id,
            findings=findings,
            marker_ids=markers,
            incident_ids=incidents,
            workflow_request_ids=workflows,
            notification_ids=notifications,
        )
