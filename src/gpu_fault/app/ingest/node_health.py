from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from typing import Any

from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
    NodeHealthIngestionResult,
)
from gpu_fault.models import AdvisoryNotification, FaultIncident, RecoveryAction
from gpu_fault.orchestration.families.health import (
    HOST_RESOURCE_POLICY_SOURCE,
    UNSETTLED_INCIDENT_STATES,
    host_resource_signal,
    host_resource_signal_event,
    incident_opened_by_host_resource_signal,
)
from gpu_fault.orchestration.families.incident_labels import (
    finding_marker_id,
    routed_finding_markers,
)
from gpu_fault.shared_storage_failure_domain import SHARED_FILESYSTEM_METRIC

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
        # gpu_fault.nvidia_logs.UNSCHEDULABLE_WITHOUT_CODE_REASON: HMA cordoned the
        # node for a fault that names no XID/SXID, so only a human can say
        # what it was.
        "hma_unschedulable_without_code",
        "gpu_inventory_identity_changed",
        "gpu_expected_count_unknown",
        # host_health.gpu_inventory_verdicts: the configured expected GPU count
        # contradicts the instance type while the node lists that type's count
        # (GF-GENERAL-LIVE-012 expectation_wrong) -- a configuration error
        # for a human, graded WARNING / evidence only, never a reboot.
        "gpu_expected_count_contradicted",
        # b233, DCGM field 389: banks with no spare row left. RUN_DIAGNOSTICS
        # and an operator notice; no automatic hardware action.
        "row_remap_banks_none",
        # b235, DCGM fields 174/175: degraded fabric capacity (RUN_DIAGNOSTICS),
        # and a confirmed unhealthy fabric before its full reset, are a human's
        # to know of (gpu_fault.gpu_fabric_health).
        "fabric_health",
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


#: The sustained host-resource signals whose mail is the fixed
#: ``host-resource-event-zh-v1`` (``preview_host_resource_event``).
HOST_RESOURCE_MAIL_METRICS = frozenset(
    {
        "training_rank_cpu_percent",
        "host_gpu_utilization_percent",
        "memory_available_percent",
        "page_cache_percent",
        "local_filesystem_used_percent",
    }
)


def host_resource_signal_reported(
    store: Any,
    incident: FaultIncident,
    finding: NodeHealthFinding,
    batch_markers: Collection[str],
) -> bool:
    """Whether the operator has heard of ``finding``'s signal on its node.

    For a finding that did not mint ``incident`` (the caller mails one that
    did). The host-resource mail is per incident, node and signal, so the
    operator has heard of it when an earlier finding of the same node and
    signal -- not one of this batch, ``batch_markers`` -- reached the same
    incident: the incident that signal minted on the node (the health family
    absorbs a re-arm into it while it is unsettled, ``families/health.py``)
    or one an earlier finding of it was folded into and mailed for; or when
    the incident that signal minted on the node is still unsettled
    (ACTION_PENDING / ESCALATED, an operator owns it) -- the reboot window
    takes a finding before that absorb does. A finding the product routed
    into an incident another fault opened has not been heard of otherwise:
    the attempt grouping folds it into a managed attempt's in-flight
    recovery (whose plan takes on the finding's own FREEZE_EVIDENCE +
    VALIDATE_HOST), the reboot window absorbs it into the reboot of its node
    (``orchestration/reboot_window.py``) -- and that incident's own mail, if
    any, went out before and names its own fault (b215).

    The incident a signal minted on this node alone answers without a read;
    any other is judged on the markers ingestion re-pointed at it, which
    keep each finding's node, source and signal whatever the incident's
    label, identity or policy source became (``incident_labels``), and on
    the node's unsettled incidents (two indexed reads, folded findings only).
    """

    signal = host_resource_signal(finding)
    if signal is None:
        return False
    metric_name, rule_id = signal

    def minted_here(candidate: FaultIncident) -> bool:
        return candidate.node_ids == [finding.node_id] and (
            incident_opened_by_host_resource_signal(candidate, metric_name, rule_id)
        )

    if minted_here(incident):
        return True
    if any(
        marker.marker_id not in batch_markers
        and marker.event_source == HOST_RESOURCE_POLICY_SOURCE
        and finding.node_id in marker.scope.node_ids
        and host_resource_signal_event(marker.marker_id, metric_name, rule_id)
        for marker in routed_finding_markers(store, incident)
    ):
        return True
    return any(
        minted_here(candidate)
        for candidate in store.list_incidents_by_state(
            finding.cluster_id,
            UNSETTLED_INCIDENT_STATES,
            node_ids={finding.node_id},
        )
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
            # Log evidence is already durably captured by the ingest transaction.
            # Workload errors are not hardware markers or a second recovery owner.
            if (
                finding.policy_source == "SITE_APPLICATION_LOG"
                and finding.recommended_action is RecoveryAction.NO_ACTION
            ):
                continue
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
            if finding.metric_name == SHARED_FILESYSTEM_METRIC:
                # Additive: the per-node chain above is untouched; this only
                # records the export's common failure domain when peers
                # report the same export unavailable (one notification).
                correlator = getattr(self.context, "shared_storage_domains", None)
                if correlator is not None:
                    _domain, notification_id = correlator.observe_finding(
                        finding, incident
                    )
                    if notification_id is not None:
                        notifications.append(notification_id)
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
                finding.metric_name in HOST_RESOURCE_MAIL_METRICS
                and finding.policy_source == HOST_RESOURCE_POLICY_SOURCE
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
        batch_markers = {finding_marker_id(finding.event_id) for finding in findings}
        for grouped in host_resource_notification_groups.values():
            # A finding that minted (or re-posted) its incident is what the
            # mail is for; one routed into an incident the operator has not
            # heard of this signal on -- another fault's recovery, the reboot
            # of its node -- is mailed the same way, naming that incident. A
            # re-arm on an incident the signal already reached (still
            # ESCALATED / ACTION_PENDING for it) mails nothing: nothing new
            # has been decided since. Notifications are per incident, node and
            # signal, not per finding (GF-REGIONAL-NOTIFY-005 counts per node,
            # not per GPU).
            heard: dict[str, bool] = {}
            mailed = []
            for incident, finding in grouped:
                if incident.event_id != finding.event_id:
                    if incident.incident_id not in heard:
                        heard[incident.incident_id] = host_resource_signal_reported(
                            self.context.store, incident, finding, batch_markers
                        )
                    if heard[incident.incident_id]:
                        continue
                mailed.append((incident, finding))
            if not mailed:
                continue
            incident, representative = mailed[0]
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

    def record_regrades(
        self, batch_id: str, findings: Sequence[NodeHealthFinding]
    ) -> NodeHealthIngestionResult:
        """Record re-graded findings on the incidents their events already have.

        A GPU composite keeps one finding id -- one event id, one incident --
        for as long as its episode lasts (``gpu_composites.composite_episode_identity``).
        When its severity changes within the episode the finding arrives here
        instead of through :meth:`ingest`: the open incident gains the new grade
        as a bounded reason, the node marker is rewritten at the new severity
        and nothing new is planned -- the automatic action did not change (a
        changed action is a new finding id and comes through :meth:`ingest`).
        A re-grade whose event has no incident -- the activation's incident
        write never committed on a nontransactional backend -- is ingested as
        new, the same repair F-M1 asks for: a CRITICAL finding gets its incident.
        """

        markers = []
        incidents = []
        pending = []
        for finding in findings:
            incident = self.context.orchestrator.record_health_regrade(finding)
            if incident is None:
                pending.append(finding)
                continue
            marker = finding.marker().model_copy(
                update={"incident_id": incident.incident_id}
            )
            self.context.completion.add_marker(marker)
            markers.append(marker.marker_id)
            incidents.append(incident.incident_id)
        result = NodeHealthIngestionResult(
            batch_id=batch_id,
            findings=list(findings),
            marker_ids=markers,
            incident_ids=incidents,
        )
        if not pending:
            return result
        minted = self.ingest(batch_id, pending)
        return result.model_copy(
            update={
                "marker_ids": [*markers, *minted.marker_ids],
                "incident_ids": [*incidents, *minted.incident_ids],
                "workflow_request_ids": minted.workflow_request_ids,
                "notification_ids": minted.notification_ids,
            }
        )
