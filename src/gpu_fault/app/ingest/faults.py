from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from gpu_fault.gpu_metrics import GpuInventorySnapshot
from gpu_fault.models import WorkloadState
from gpu_fault.policy import (
    ActionDisposition,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)
from gpu_fault.processor_diagnostics import report_processor_replay_phase


class FaultIngestionService:
    def __init__(self, context) -> None:
        self.context = context
        context.xid_correlation.finalizer = self.finalize_xid

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
