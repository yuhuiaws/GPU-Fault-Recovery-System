from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Callable

from gpu_fault.hyperpod import HyperPodAdvisoryDisposition, HyperPodRecoveryAdvisory
from gpu_fault.models import AdvisoryNotification, CapabilityName, WorkflowOperation
from gpu_fault.notifications.registry import (
    NotificationBuilderRegistry,
    NotificationKind,
)
from gpu_fault.orchestrator import OPERATION_CAPABILITY
from gpu_fault.policy import ActionDisposition, FaultPolicyDecision, SxidEvent, XidEvent


MANAGED_HYPERPOD_OWNER_PREFIX = "hyperpod-managed-"
ADVISORY_OPERATION_PRIORITY = {
    WorkflowOperation.REPLACE_NODE: 30,
    WorkflowOperation.RESTART_NODE: 20,
    WorkflowOperation.RESTART_WORKLOAD: 10,
}


class AdvisoryNotApplicableError(ValueError):
    pass


@dataclass(frozen=True)
class NotificationPreviewCallbacks:
    incident_lock: Callable
    evidence_refs_for_incident: Callable


class NotificationPreviewService:
    def __init__(
        self,
        store,
        builders: NotificationBuilderRegistry,
        callbacks: NotificationPreviewCallbacks,
    ) -> None:
        self.store = store
        self.builders = builders
        self.callbacks = callbacks

    def preview_not_applicable(
        self,
        incident_id: str,
        event: XidEvent,
        decision: FaultPolicyDecision,
    ) -> AdvisoryNotification:
        if decision.disposition is not ActionDisposition.NOT_APPLICABLE:
            raise AdvisoryNotApplicableError(
                "decision disposition is not NOT_APPLICABLE"
            )
        with self.callbacks.incident_lock(incident_id):
            incident = self.store.get_incident(incident_id)
            if not incident.workflow_request_id:
                raise AdvisoryNotApplicableError(
                    "NOT_APPLICABLE incident has no safety workflow"
                )
            workflow = self.store.get_workflow(incident.workflow_request_id)
            evidence_refs = self.callbacks.evidence_refs_for_incident(
                incident_id,
                event.evidence_ref,
            )
            notification = self.builders.build(
                NotificationKind.NOT_APPLICABLE,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                incident_id=incident.incident_id,
                event_id=event.event_id,
                xid=event.xid,
                product=event.product,
                driver_branch=event.driver_branch,
                cuda_version=event.cuda_version,
                observed_at=event.observed_at.isoformat(),
                workload_state=event.workload_state.value,
                workload_ids=event.affected_workload_ids,
                official_action=decision.official_action,
                investigatory_action=decision.investigatory_action,
                policy_source=decision.source.value,
                policy_version=decision.policy_version,
                reasons=decision.reasons,
                safety_action=(
                    decision.safety_action.value if decision.safety_action else None
                ),
                workflow_id=workflow.request_id,
                workflow_status=workflow.status.value,
                safety_steps=[item.operation.value for item in workflow.safety_steps],
                official_steps=[
                    item.operation.value for item in workflow.official_steps
                ],
                evidence_refs=evidence_refs,
            )
            return self.store.save_notification_if_absent(notification)

    def preview_xid_investigatory(
        self,
        incident_id: str,
        event: XidEvent,
        decision: FaultPolicyDecision,
    ) -> AdvisoryNotification:
        with self.callbacks.incident_lock(incident_id):
            incident = self.store.get_incident(incident_id)
            evidence_refs = self.callbacks.evidence_refs_for_incident(
                incident_id,
                event.evidence_ref,
            )
            notification = self.builders.build(
                NotificationKind.XID_INVESTIGATORY,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                incident_id=incident.incident_id,
                workflow_id=incident.workflow_request_id,
                event_id=event.event_id,
                xid=event.xid,
                product=event.product,
                observed_at=event.observed_at.isoformat(),
                workload_state=event.workload_state.value,
                workload_ids=event.affected_workload_ids,
                disposition=decision.disposition.value,
                official_action=decision.official_action,
                effective_action=(
                    incident.effective_action.value
                    if incident.effective_action
                    else (decision.action.value if decision.action else None)
                ),
                investigatory_action=decision.investigatory_action,
                policy_source=decision.source.value,
                policy_version=decision.policy_version,
                reasons=decision.reasons,
                evidence_refs=evidence_refs,
            )
            notification = notification.model_copy(
                update={
                    "category": "FAULT_DETECTED",
                    "priority": 0,
                }
            )
            return self.store.save_notification_if_absent(notification)

    def preview_sxid_event(
        self,
        incident_id: str,
        event: SxidEvent,
        decision: FaultPolicyDecision,
    ) -> AdvisoryNotification:
        with self.callbacks.incident_lock(incident_id):
            incident = self.store.get_incident(incident_id)
            notification = self.builders.build(
                NotificationKind.SXID_EVENT,
                cluster_id=event.cluster_id,
                node_id=event.node_id,
                incident_id=incident.incident_id,
                workflow_id=incident.workflow_request_id,
                event_id=event.event_id,
                sxid=event.sxid,
                product=event.product,
                observed_at=event.observed_at.isoformat(),
                event_source=event.event_source,
                evidence_ref=event.evidence_ref,
                workload_state=event.workload_state.value,
                workload_ids=event.affected_workload_ids,
                classification=event.classification.value,
                classification_source=event.classification_source,
                link_scope=event.link_scope.value,
                link_scope_source=event.link_scope_source,
                switch_id=event.switch_id,
                port=event.port,
                pci_bdf=event.pci_bdf,
                fabric_partition=event.fabric_partition,
                disposition=decision.disposition.value,
                official_action=decision.official_action,
                effective_action=(
                    incident.effective_action.value
                    if incident.effective_action
                    else (decision.action.value if decision.action else None)
                ),
                safety_action=(
                    decision.safety_action.value if decision.safety_action else None
                ),
                investigatory_action=decision.investigatory_action,
                policy_source=decision.source.value,
                policy_version=decision.policy_version,
                reasons=decision.reasons,
            )
            return self.store.save_notification_if_absent(notification)

    def preview_efa_rdma_event(self, incident_id: str, finding) -> AdvisoryNotification:
        with self.callbacks.incident_lock(incident_id):
            incident = self.store.get_incident(incident_id)
            workflow = (
                self.store.get_workflow(incident.workflow_request_id)
                if incident.workflow_request_id
                else None
            )
            evidence_refs = self.callbacks.evidence_refs_for_incident(
                incident_id,
                finding.evidence_ref,
            )
            notification = self.builders.build(
                NotificationKind.EFA_RDMA_EVENT,
                cluster_id=finding.cluster_id,
                node_id=finding.node_id,
                incident_id=incident.incident_id,
                workflow_id=incident.workflow_request_id,
                event_id=finding.event_id,
                observed_at=finding.observed_at.isoformat(),
                severity=finding.severity.value,
                metric_name=finding.metric_name or "UNKNOWN",
                value=finding.value,
                device=finding.device,
                reason=finding.reason,
                workload_state=finding.workload_state.value,
                workload_ids=finding.affected_workload_ids,
                job_id=finding.job_id,
                attempt_id=finding.attempt_id,
                recommended_action=finding.recommended_action.value,
                policy_source=finding.policy_source,
                policy_version=finding.policy_version,
                workflow_steps=(
                    [step.operation.value for step in workflow.official_steps]
                    if workflow
                    else []
                ),
                diagnostic_parameters=finding.diagnostic_parameters,
                evidence_refs=evidence_refs,
            )
            notification = notification.model_copy(
                update={
                    "category": "FAULT_DETECTED",
                    "priority": 0,
                }
            )
            return self.store.save_notification_if_absent(notification)

    def preview_hardware_inventory_event(
        self, incident_id: str, finding
    ) -> AdvisoryNotification:
        with self.callbacks.incident_lock(incident_id):
            incident = self.store.get_incident(incident_id)
            workflow = (
                self.store.get_workflow(incident.workflow_request_id)
                if incident.workflow_request_id
                else None
            )
            evidence_refs = self.callbacks.evidence_refs_for_incident(
                incident_id,
                finding.evidence_ref,
            )
            notification = self.builders.build(
                NotificationKind.HARDWARE_INVENTORY,
                cluster_id=finding.cluster_id,
                node_id=finding.node_id,
                incident_id=incident.incident_id,
                workflow_id=incident.workflow_request_id,
                event_id=finding.event_id,
                observed_at=finding.observed_at.isoformat(),
                severity=finding.severity.value,
                metric_name=finding.metric_name or "UNKNOWN",
                workload_state=finding.workload_state.value,
                workload_ids=finding.affected_workload_ids,
                recommended_action=finding.recommended_action.value,
                policy_source=finding.policy_source,
                policy_version=finding.policy_version,
                workflow_steps=(
                    [step.operation.value for step in workflow.official_steps]
                    if workflow
                    else []
                ),
                inventory=finding.diagnostic_parameters,
                evidence_refs=evidence_refs,
            )
            return self.store.save_notification_if_absent(notification)

    def preview_host_resource_event(
        self, incident_id: str, finding
    ) -> AdvisoryNotification:
        with self.callbacks.incident_lock(incident_id):
            incident = self.store.get_incident(incident_id)
            workflow = (
                self.store.get_workflow(incident.workflow_request_id)
                if incident.workflow_request_id
                else None
            )
            evidence_refs = self.callbacks.evidence_refs_for_incident(
                incident_id,
                finding.evidence_ref,
            )
            notification = self.builders.build(
                NotificationKind.HOST_RESOURCE,
                cluster_id=finding.cluster_id,
                node_id=finding.node_id,
                incident_id=incident.incident_id,
                workflow_id=incident.workflow_request_id,
                event_id=finding.event_id,
                observed_at=finding.observed_at.isoformat(),
                severity=finding.severity.value,
                metric_name=finding.metric_name or "UNKNOWN",
                value=finding.value,
                device=finding.device,
                reason=finding.reason,
                workload_state=finding.workload_state.value,
                workload_ids=finding.affected_workload_ids,
                recommended_action=finding.recommended_action.value,
                policy_source=finding.policy_source,
                policy_version=finding.policy_version,
                workflow_steps=(
                    [step.operation.value for step in workflow.official_steps]
                    if workflow
                    else []
                ),
                diagnostic_parameters=finding.diagnostic_parameters,
                evidence_refs=evidence_refs,
            )
            cooldown_seconds = int(
                os.getenv(
                    "GPU_FAULT_HOST_NOTIFICATION_COOLDOWN_SECONDS",
                    "3600",
                )
            )
            bucket = int(finding.observed_at.timestamp()) // max(60, cooldown_seconds)
            scope = "|".join(
                [
                    finding.cluster_id,
                    finding.node_id,
                    finding.metric_name or "UNKNOWN",
                    ",".join(sorted(finding.affected_workload_ids)),
                    str(finding.diagnostic_parameters.get("signal", "UNKNOWN")),
                    str(bucket),
                ]
            )
            notification = notification.model_copy(
                update={
                    "deduplication_key": (
                        "host-resource/" + hashlib.sha256(scope.encode()).hexdigest()
                    ),
                    "category": "HEALTH_TREND",
                    "priority": 200,
                }
            )
            return self.store.save_notification_if_absent(notification)

    def preview(self, incident_id: str) -> AdvisoryNotification:
        with self.callbacks.incident_lock(incident_id):
            incident = self.store.get_incident(incident_id)
            if not incident.workflow_request_id:
                raise AdvisoryNotApplicableError("incident has no recovery workflow")
            workflow = self.store.get_workflow(incident.workflow_request_id)
            managed_steps = [
                step
                for step in workflow.official_steps
                if step.execution_owner.startswith(MANAGED_HYPERPOD_OWNER_PREFIX)
                and step.operation in ADVISORY_OPERATION_PRIORITY
            ]
            if not managed_steps:
                raise AdvisoryNotApplicableError(
                    "incident has no HyperPod-managed node or workload recovery action"
                )
            selected = max(
                managed_steps,
                key=lambda step: ADVISORY_OPERATION_PRIORITY[step.operation],
            )
            capability = OPERATION_CAPABILITY[selected.operation]
            if capability not in {
                CapabilityName.NODE_REBOOT,
                CapabilityName.NODE_REPLACE,
                CapabilityName.WORKLOAD_RESTART,
            }:
                raise AdvisoryNotApplicableError(
                    "managed operation is not advisory-enabled"
                )
            evidence_refs = self.callbacks.evidence_refs_for_incident(incident_id)
            recommended_action = incident.official_action or (
                incident.effective_action.value
                if incident.effective_action
                else selected.operation.value
            )
            advisory = HyperPodRecoveryAdvisory(
                capability=capability,
                recommended_action=recommended_action,
                execution_owner=selected.execution_owner,
                disposition=(HyperPodAdvisoryDisposition.ADVISE_ONLY),
                rationale=incident.reasons or ["managed recovery recommendation"],
                evidence_refs=list(dict.fromkeys(evidence_refs)),
            )
            notification = self.builders.build(
                NotificationKind.HYPERPOD_ADVISORY,
                advisory,
                cluster_name=incident.cluster_id,
                incident_id=incident.incident_id,
                node_ids=incident.node_ids,
                issue_summary=(
                    f"{incident.event_type} incident; policy recommends "
                    f"{recommended_action}"
                ),
            )
            return self.store.save_notification_if_absent(notification)

    def try_preview(self, incident_id: str) -> AdvisoryNotification | None:
        try:
            return self.preview(incident_id)
        except AdvisoryNotApplicableError:
            return None
