from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.gpu_metrics import GpuInventorySnapshot
from gpu_fault.hma import (
    UNCLASSIFIED_SXID_REASON,
    UNPARSED_SXID_REASON,
    UNPARSED_XID_REASON,
    UNSCHEDULABLE_WITHOUT_CODE_REASON,
    HmaNormalizedBatch,
    HmaProviderSignal,
    unresolved_reason_kind,
)
from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
    NodeHealthIngestionResult,
)
from gpu_fault.models import RecoveryAction, Severity, WorkloadState
from gpu_fault.policy import (
    ActionDisposition,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)
from gpu_fault.processor_diagnostics import report_processor_replay_phase
from gpu_fault.store.shared.health_signals import finding_health_signal_key

LOGGER = logging.getLogger(__name__)

#: Every kind ``unresolved_reason_kind`` can return. A provider signal that
#: carries no unresolved reason clears the episode of each one, so a kind that
#: is missing here would open a WARNING finding that never closes.
UNRESOLVED_SIGNAL_KINDS = (
    UNPARSED_XID_REASON,
    UNPARSED_SXID_REASON,
    UNCLASSIFIED_SXID_REASON,
    UNSCHEDULABLE_WITHOUT_CODE_REASON,
)


class FaultIngestionService:
    def __init__(self, context) -> None:
        self.context = context
        context.xid_correlation.finalizer = self.finalize_xid
        # Per-kind count of provider signals whose fault line could not be
        # resolved into an XID/SXID event. Exposed for the metrics snapshot;
        # the finding below is the per-episode signal, this is the per-line
        # one, so a drift that hits every line still shows its full rate.
        self.unresolved_signal_totals: dict[str, int] = {}

    def ingest_unresolved_signals(
        self,
        normalized: HmaNormalizedBatch,
        *,
        batch_id: str,
    ) -> NodeHealthIngestionResult | None:
        """Turn an unparsable fault line into something an operator sees.

        The collectors forward every ``NVRM ... Xid`` and ``SXid`` line; when
        the normalizer cannot extract a code (or an SXID's classification
        word) the line used to land as evidence with ``status=success`` and
        the reason in a response body nobody reads. A catalog or format drift
        could therefore swallow real XIDs silently. Now every such line is a
        WARNING and a counter, and the first one per node per episode opens a
        WARNING node-health finding with ``COLLECT_EVIDENCE`` -- it freezes
        evidence and routes to operator review without inventing a code and
        without cordoning the node. A later line from the same node that does
        parse ends the episode, so the next drift is reported again.
        """

        clears: list[tuple[str, bool, datetime, float]] = []
        candidates: list[tuple[tuple[str, bool, datetime, float], NodeHealthFinding]]
        candidates = []
        for signal in normalized.provider_signals:
            if not signal.unresolved_reasons:
                clears.extend(
                    (
                        f"{signal.cluster_id}/{signal.node_id}/{kind}/node",
                        False,
                        signal.observed_at,
                        0.0,
                    )
                    for kind in UNRESOLVED_SIGNAL_KINDS
                )
                continue
            by_kind: dict[str, list[str]] = {}
            for reason in signal.unresolved_reasons:
                by_kind.setdefault(unresolved_reason_kind(reason), []).append(reason)
            for kind, reasons in sorted(by_kind.items()):
                self.unresolved_signal_totals[kind] = (
                    self.unresolved_signal_totals.get(kind, 0) + 1
                )
                LOGGER.warning(
                    "fault line could not be resolved; no code was invented "
                    "kind=%s cluster=%s node=%s source=%s signal=%s reasons=%s",
                    kind,
                    signal.cluster_id,
                    signal.node_id,
                    signal.source.value,
                    signal.signal_id,
                    reasons,
                )
                finding = self._unresolved_signal_finding(
                    signal, kind=kind, reasons=reasons, batch_id=batch_id
                )
                candidates.append(
                    (
                        (
                            finding_health_signal_key(finding),
                            True,
                            signal.observed_at,
                            0.0,
                        ),
                        finding,
                    )
                )
        # The receive clock orders the episode: two kernel records can carry
        # the same second, and a sample that is not newer on the state
        # machine's clock is dropped as a replay.
        received_at = datetime.now(timezone.utc)
        if clears:
            self.context.store.claim_health_signal_transitions(
                clears, received_at=received_at
            )
        if not candidates:
            return None
        emitted = self.context.store.claim_health_signal_transitions(
            [transition for transition, _ in candidates],
            received_at=received_at,
        )
        findings = [
            finding
            for (_, finding), emit in zip(candidates, emitted, strict=True)
            if emit
        ]
        if not findings:
            return None
        result = NodeHealthIngestionService(self.context).ingest(batch_id, findings)
        notified_at = datetime.now(timezone.utc)
        for finding in findings:
            self.context.store.mark_health_signal_notified(
                finding_health_signal_key(finding), notified_at=notified_at
            )
        return result

    @staticmethod
    def _unresolved_signal_finding(
        signal: HmaProviderSignal,
        *,
        kind: str,
        reasons: list[str],
        batch_id: str,
    ) -> NodeHealthFinding:
        event_id = f"{signal.signal_id}-{kind}"
        raw_message = (signal.raw_message or "")[:500] or None
        return NodeHealthFinding(
            finding_id=f"finding-{event_id}",
            event_id=event_id,
            cluster_id=signal.cluster_id,
            node_id=signal.node_id,
            observed_at=signal.observed_at,
            category=NodeHealthCategory.GPU,
            severity=Severity.WARNING,
            reason=(
                "a fault line from the node could not be resolved into an "
                "XID/SXID event; operator review is required before the "
                "signal is trusted or dismissed"
            ),
            recommended_action=RecoveryAction.COLLECT_EVIDENCE,
            metric_name=kind,
            value=1.0,
            raw_message=raw_message,
            evidence_ref=f"hma://{signal.source.value.lower()}/{signal.signal_id}",
            diagnostic_parameters={
                "unresolved_reasons": list(reasons),
                "signal_source": signal.source.value,
                "record_batch_id": batch_id,
            },
            policy_reference=(
                "unresolved fault lines fail closed into operator review"
            ),
        )

    def _enrich_fault_identity(
        self,
        event: XidEvent | SxidEvent,
    ) -> XidEvent | SxidEvent:
        if event.job_id and event.attempt_id:
            return event
        context = self.context.topology.resolve(
            event.cluster_id,
            event.node_id,
            event.observed_at,
            target_gpu_uuids=(
                {event.gpu_uuid}
                if isinstance(event, XidEvent) and event.gpu_uuid
                else set(event.participating_gpu_uuids)
                if isinstance(event, SxidEvent)
                else None
            ),
            pod_uid=event.pod_uid,
            container_id=event.container_id,
            host_pid=event.host_pid,
            cgroup_path=event.cgroup_path,
        )
        identities = set(context.job_attempt_ids)
        if len(identities) != 1:
            source = (
                "NO_ACTIVE_MANAGED_ATTEMPT"
                if not identities
                else "AMBIGUOUS_ACTIVE_ATTEMPTS"
            )
            return event.model_copy(update={"workload_identity_source": source})
        job_id, attempt_id = next(iter(identities))
        updates = {
            "job_id": event.job_id or job_id,
            "attempt_id": event.attempt_id or attempt_id,
            "workload_identity_source": (
                event.workload_identity_source or "SOLE_ACTIVE_ATTEMPT_ON_NODE"
            ),
        }
        if not event.affected_workload_ids:
            updates["affected_workload_ids"] = context.workload_ids
        if event.workload_state is WorkloadState.UNKNOWN:
            updates["workload_state"] = WorkloadState(context.workload_state)
        if (
            event.runtime_profile_version is None
            and context.runtime_profile_version is not None
        ):
            updates["runtime_profile_version"] = context.runtime_profile_version
        return event.model_copy(update=updates)

    def _fresh_gpu_inventory(
        self,
        *,
        cluster_id: str,
        node_id: str,
        event_time: datetime,
        source_boot_id: str | None,
    ) -> GpuInventorySnapshot | None:
        snapshot = self.context.store.get_gpu_inventory_snapshot(cluster_id, node_id)
        if snapshot is None:
            return None
        age = event_time - snapshot.observed_at
        if age < -timedelta(seconds=30) or age > self.context.gpu_inventory_max_age:
            return None
        if (
            source_boot_id
            and snapshot.source_boot_id
            and source_boot_id != snapshot.source_boot_id
        ):
            return None
        return snapshot

    def _pci_slot(
        self,
        value: str,
    ) -> tuple[str, str, str | None] | None:
        match = re.fullmatch(
            r"(?:(?:[0-9a-f]{4}|[0-9a-f]{8}):)?"
            r"([0-9a-f]{2}):([0-9a-f]{2})(?:\.([0-7]))?",
            value.strip().lower(),
        )
        return match.groups() if match is not None else None

    def _enrich_sxid_scope(self, event: SxidEvent) -> SxidEvent:
        event = self.context.nvswitch_topology.resolve(event)
        catalog_rule = self.context.policy.sxid_catalog_rule(event.sxid)
        context = self.context.topology.resolve(
            event.cluster_id, event.node_id, event.observed_at
        )
        updates = {}
        inventory_snapshot = self._fresh_gpu_inventory(
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            event_time=event.observed_at,
            source_boot_id=event.source_boot_id,
        )
        if (
            event.classification.value == "FATAL"
            and event.link_scope.value == "ACCESS"
            and not event.participating_gpu_uuids
            and not (
                catalog_rule is not None
                and catalog_rule.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
            )
        ):
            participants = set(context.gpu_uuids)
            if event.pci_bdf and inventory_snapshot is not None:
                event_slot = self._pci_slot(event.pci_bdf)
                for device in inventory_snapshot.devices:
                    sample_slot = self._pci_slot(device.pci_bdf)
                    if (
                        event_slot is not None
                        and sample_slot is not None
                        and sample_slot[:2] == event_slot[:2]
                        and (event_slot[2] is None or sample_slot[2] == event_slot[2])
                    ):
                        participants.add(device.gpu_uuid)
            elif event.pci_bdf:
                event_slot = self._pci_slot(event.pci_bdf)
                for latest in self.context.store.list_gpu_metrics_latest(
                    event.cluster_id, event.node_id
                ):
                    sample_slot = (
                        self._pci_slot(latest.sample.pci_bdf)
                        if latest.sample.pci_bdf
                        else None
                    )
                    age = event.observed_at - latest.observed_at
                    if (
                        event_slot is not None
                        and sample_slot is not None
                        and latest.sample.gpu_uuid
                        and -timedelta(seconds=30)
                        <= age
                        <= self.context.legacy_gpu_metrics_inventory_max_age
                        and sample_slot[:2] == event_slot[:2]
                        and (event_slot[2] is None or sample_slot[2] == event_slot[2])
                    ):
                        participants.add(latest.sample.gpu_uuid)
            if participants:
                updates["participating_gpu_uuids"] = sorted(participants)
        elif (
            catalog_rule is not None
            and catalog_rule.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
        ) or (
            event.classification.value == "ALWAYS_FATAL"
            or (
                event.classification.value == "FATAL"
                and event.link_scope.value != "ACCESS"
            )
        ):
            inventory = (
                {device.gpu_uuid for device in inventory_snapshot.devices}
                if inventory_snapshot is not None
                else {
                    latest.sample.gpu_uuid
                    for latest in self.context.store.list_gpu_metrics_latest(
                        event.cluster_id, event.node_id
                    )
                    if (
                        latest.sample.gpu_uuid
                        and -timedelta(seconds=30)
                        <= event.observed_at - latest.observed_at
                        <= self.context.legacy_gpu_metrics_inventory_max_age
                    )
                }
            )
            if inventory:
                updates["participating_gpu_uuids"] = sorted(inventory)
                updates["fabric_partition"] = (
                    event.fabric_partition
                    or f"{event.cluster_id}/{event.node_id}/local-nvswitch"
                )
        return event.model_copy(update=updates) if updates else event

    def _resolve_kernel_xid_gpu_uuid(self, event: XidEvent) -> XidEvent:
        if event.gpu_uuid or not event.pci_bdf:
            return event

        event_slot = self._pci_slot(event.pci_bdf)
        if event_slot is None:
            return event
        snapshot = self._fresh_gpu_inventory(
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            event_time=event.observed_at,
            source_boot_id=event.source_boot_id,
        )
        if snapshot is not None:
            candidates = {
                device.gpu_uuid
                for device in snapshot.devices
                if (
                    (sample_slot := self._pci_slot(device.pci_bdf)) is not None
                    and sample_slot[:2] == event_slot[:2]
                    and (event_slot[2] is None or sample_slot[2] == event_slot[2])
                )
            }
            if len(candidates) == 1:
                return event.model_copy(update={"gpu_uuid": candidates.pop()})
            return event
        candidates = set()
        for latest in self.context.store.list_gpu_metrics_latest(
            event.cluster_id, event.node_id
        ):
            sample = latest.sample
            sample_slot = self._pci_slot(sample.pci_bdf) if sample.pci_bdf else None
            age = event.observed_at - latest.observed_at
            if (
                sample_slot is None
                or not sample.gpu_uuid
                or age < -timedelta(seconds=30)
                or age > self.context.legacy_gpu_metrics_inventory_max_age
                or sample_slot[:2] != event_slot[:2]
                or (event_slot[2] is not None and sample_slot[2] != event_slot[2])
            ):
                continue
            candidates.add(sample.gpu_uuid)
        if len(candidates) != 1:
            return event
        return event.model_copy(update={"gpu_uuid": candidates.pop()})

    def finalize_xid(
        self, event: XidEvent, decision: FaultPolicyDecision
    ) -> FaultPolicyDecision:
        report_processor_replay_phase("xid_action_fence")
        decision = self.context.orchestrator.apply_fault_action_generation_fence(
            event, decision
        )
        if decision.correlated_event_id:
            report_processor_replay_phase("xid_companion_lookup")
            companion_incident = self.context.store.get_incident_by_event(
                decision.correlated_event_id
            )
            if companion_incident is None:
                companion_event = self.context.store.get_xid_event(
                    decision.correlated_event_id
                )
                companion_decision = self.context.store.get_xid_policy_decision(
                    decision.correlated_event_id
                )
                if (
                    companion_decision is not None
                    and companion_decision.disposition
                    is not ActionDisposition.PENDING_CORRELATION
                ):
                    self.finalize_xid(companion_event, companion_decision)
                companion_incident = self.context.store.get_incident_by_event(
                    decision.correlated_event_id
                )
            if companion_incident is not None:
                decision = decision.model_copy(
                    update={
                        "duplicate": True,
                        "marker": decision.marker.model_copy(
                            update={"incident_id": (companion_incident.incident_id)}
                        ),
                    }
                )
        report_processor_replay_phase("xid_provider_correlation")
        decision = self.context.orchestrator.correlate_provider_event(
            decision,
            cluster_id=event.cluster_id,
        )
        report_processor_replay_phase("xid_marker")
        self.context.completion.add_marker(decision.marker)
        report_processor_replay_phase("xid_incident_workflow")
        incident, workflow = self.context.orchestrator.ingest(event, decision)
        report_processor_replay_phase("xid_notification")
        if decision.disposition is ActionDisposition.NOT_APPLICABLE:
            notification = self.context.advisory_notifications.preview_not_applicable(
                incident.incident_id, event, decision
            )
            self.context.advisory_notifications.send(notification.notification_id)
            investigatory_notification = notification
        else:
            notification = self.context.advisory_notifications.try_preview(
                incident.incident_id
            )
            investigatory_notification = (
                self.context.advisory_notifications.preview_xid_investigatory(
                    incident.incident_id, event, decision
                )
            )
            self.context.advisory_notifications.send(
                investigatory_notification.notification_id
            )
        finalized = decision.model_copy(
            update={
                "incident_id": incident.incident_id,
                "workflow_request_id": (workflow.request_id if workflow else None),
                "advisory_notification_id": (
                    notification.notification_id if notification else None
                ),
                "investigatory_notification_id": (
                    investigatory_notification.notification_id
                ),
            }
        )
        report_processor_replay_phase("xid_decision_persist")
        self.context.store.save_xid_policy_decision(finalized)
        return finalized

    def ingest_xid(self, event: XidEvent) -> FaultPolicyDecision:
        report_processor_replay_phase("xid_identity")
        # xid_154_action is derived server-side from the XID 154 log line
        # (see XidCorrelationCoordinator.prepare_xid154). Drop any inbound
        # value at the ingest boundary so a client can never seed or steer
        # the dynamic recovery action (security review H-10).
        if event.xid_154_action is not None:
            event = event.model_copy(update={"xid_154_action": None})
        event = self._enrich_fault_identity(event)
        if event.ingested_at is None:
            event = event.model_copy(update={"ingested_at": datetime.now(timezone.utc)})
        report_processor_replay_phase("xid_correlation")
        decision = self.context.xid_correlation.ingest(event)
        if decision.disposition is ActionDisposition.PENDING_CORRELATION:
            return decision
        return self.finalize_xid(event, decision)

    def ingest_sxid(self, event: SxidEvent) -> FaultPolicyDecision:
        report_processor_replay_phase("sxid_identity")
        event = self._enrich_fault_identity(event)
        if event.ingested_at is None:
            event = event.model_copy(update={"ingested_at": datetime.now(timezone.utc)})
        report_processor_replay_phase("sxid_policy")
        decision = self.context.policy.evaluate_sxid(event)
        report_processor_replay_phase("sxid_action_fence")
        decision = self.context.orchestrator.apply_fault_action_generation_fence(
            event, decision
        )
        report_processor_replay_phase("sxid_provider_correlation")
        decision = self.context.orchestrator.correlate_provider_event(
            decision,
            cluster_id=event.cluster_id,
        )
        report_processor_replay_phase("sxid_incident_workflow")
        incident, workflow = self.context.orchestrator.ingest(event, decision)
        decision = decision.model_copy(
            update={
                "marker": decision.marker.model_copy(
                    update={"incident_id": incident.incident_id}
                )
            }
        )
        report_processor_replay_phase("sxid_marker")
        self.context.completion.add_marker(decision.marker)
        report_processor_replay_phase("sxid_notification")
        notification = self.context.advisory_notifications.try_preview(
            incident.incident_id
        )
        sxid_notification = self.context.advisory_notifications.preview_sxid_event(
            incident.incident_id, event, decision
        )
        self.context.advisory_notifications.send(sxid_notification.notification_id)
        finalized = decision.model_copy(
            update={
                "incident_id": incident.incident_id,
                "workflow_request_id": (workflow.request_id if workflow else None),
                "advisory_notification_id": (
                    notification.notification_id if notification else None
                ),
                "investigatory_notification_id": (sxid_notification.notification_id),
            }
        )
        report_processor_replay_phase("sxid_decision_persist")
        self.context.store.save_xid_policy_decision(finalized)
        return finalized
