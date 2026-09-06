from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import (
    bounded_reasons,
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowRequest,
    WorkflowStatus,
)


@dataclass(frozen=True)
class DrainOperationCallbacks:
    active_node_exclusive_workflow: Callable
    attempt_group_key: Callable
    merge_disposition: Callable
    node_group_key: Callable
    ingest_node_health: Callable
    incident_state_for_workflow: Callable
    prepare_preempting_successor: Callable


class DrainOperationService:
    def __init__(self, store, brancher, callbacks: DrainOperationCallbacks) -> None:
        self.store = store
        self.brancher = brancher
        self.callbacks = callbacks

    def ingest_terminal_node_quarantine(
        self,
        finding: NodeHealthFinding,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        """Attach a critical site quarantine to the node's active workflow.

        A standalone quarantine races the workflow that already owns the
        node: its MARK is rejected by Kubernetes ownership fencing, while
        the incumbent can later readmit the node and restart the workload.
        Reusing the incumbent's merge key lets the existing terminal-branch
        logic suppress readmission and restart atomically.
        """
        if (
            finding.recommended_action is not RecoveryAction.QUARANTINE
            or finding.severity.value != "critical"
        ):
            return None
        incumbent = self.callbacks.active_node_exclusive_workflow(
            finding.cluster_id, {finding.node_id}
        )
        if incumbent is None:
            return None
        incumbent_incident = self.store.get_incident(incumbent.incident_id)
        group_key = (
            self.callbacks.attempt_group_key(
                incumbent_incident.cluster_id,
                incumbent_incident.job_id,
                incumbent_incident.attempt_id,
            )
            if (incumbent_incident.job_id and incumbent_incident.attempt_id)
            else self.callbacks.node_group_key(finding.cluster_id, finding.node_id)
        )
        candidate_incident, candidate_workflow = self.callbacks.ingest_node_health(
            finding,
            _skip_attempt_grouping=True,
            _skip_terminal_quarantine_merge=True,
            _persist=False,
        )
        if candidate_workflow is None:
            return None

        def build(
            existing_incident: FaultIncident | None,
            existing_workflow: WorkflowRequest | None,
        ) -> tuple[FaultIncident, WorkflowRequest]:
            now = datetime.now(timezone.utc)
            target_incident = existing_incident or incumbent_incident
            target_workflow = existing_workflow or self.store.get_workflow(
                incumbent.request_id
            )
            if (
                target_workflow.status
                not in {
                    WorkflowStatus.PENDING,
                    WorkflowStatus.RUNNING,
                    WorkflowStatus.SAFETY_PENDING,
                }
                or finding.node_id not in target_incident.node_ids
            ):
                return (
                    candidate_incident,
                    candidate_workflow.model_copy(
                        update={
                            "predecessor_workflow_id": (
                                target_workflow.request_id
                                if target_workflow.status
                                in {
                                    WorkflowStatus.PENDING,
                                    WorkflowStatus.RUNNING,
                                    WorkflowStatus.SAFETY_PENDING,
                                }
                                else None
                            ),
                            "updated_at": now,
                        }
                    ),
                )
            disposition = self.callbacks.merge_disposition(
                target_workflow,
                candidate_workflow,
                finding.node_id,
                set(finding.gpu_uuids),
            )
            if disposition in {"ABSORB", "ABSORB_RECORD_ONLY"}:
                merged = target_workflow.model_copy(update={"updated_at": now})
            else:
                merged = self.brancher.append_parallel_job_branch_successor(
                    target_workflow,
                    candidate_workflow,
                    finding.node_id,
                )
            finding_name = finding.metric_name or finding.diagnostic_parameters.get(
                "diagnostic_reason", finding.category.value
            )
            incident = target_incident.model_copy(
                update={
                    "event_type": "NODE_HEALTH_GROUP",
                    "node_ids": sorted(
                        set(target_incident.node_ids) | {finding.node_id}
                    ),
                    "gpu_uuids": sorted(
                        set(target_incident.gpu_uuids) | set(finding.gpu_uuids)
                    ),
                    "policy_version": finding.policy_version,
                    "policy_source": finding.policy_source,
                    "policy_reference": finding.policy_reference,
                    "official_action": (
                        finding.official_action or RecoveryAction.QUARANTINE.value
                    ),
                    "effective_action": RecoveryAction.QUARANTINE,
                    "state": IncidentState.ACTION_PENDING,
                    "workflow_request_id": merged.request_id,
                    "fencing_token": merged.fencing_token,
                    "reasons": bounded_reasons(
                        [
                            *target_incident.reasons,
                            (f"{finding.node_id}: {finding_name}: {finding.reason}"),
                        ]
                    ),
                    "updated_at": now,
                }
            )
            return (
                incident,
                merged.model_copy(
                    update={
                        "incident_id": incident.incident_id,
                        "updated_at": now,
                    }
                ),
            )

        return self.store.merge_attempt_fault_workflow(
            group_key, finding.event_id, build
        )

    def ingest_grouped_node_resource_finding(
        self,
        finding: NodeHealthFinding,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        """Arbitrate related device-resource findings beyond the time window."""
        if finding.metric_name not in {
            "efa_inventory_mismatch",
            "efa_kubernetes_allocatable_mismatch",
            "gpu_kubernetes_allocatable_mismatch",
        }:
            return None
        incumbent = self.callbacks.active_node_exclusive_workflow(
            finding.cluster_id, {finding.node_id}
        )
        if incumbent is None:
            return None
        incumbent_incident = self.store.get_incident(incumbent.incident_id)
        candidate_incident, candidate_workflow = self.callbacks.ingest_node_health(
            finding,
            _skip_attempt_grouping=True,
            _skip_node_resource_merge=True,
            _skip_terminal_quarantine_merge=True,
            _persist=False,
        )
        if candidate_workflow is None:
            return None
        group_key = self.callbacks.node_group_key(finding.cluster_id, finding.node_id)

        def build(
            existing_incident: FaultIncident | None,
            existing_workflow: WorkflowRequest | None,
        ) -> tuple[FaultIncident, WorkflowRequest]:
            now = datetime.now(timezone.utc)
            target_incident = existing_incident or incumbent_incident
            target_workflow = existing_workflow or self.store.get_workflow(
                incumbent.request_id
            )
            disposition = self.callbacks.merge_disposition(
                target_workflow,
                candidate_workflow,
                finding.node_id,
                set(finding.gpu_uuids),
                allow_job_branch_merge=False,
            )
            winner_is_candidate = False
            if disposition in {"ABSORB", "ABSORB_RECORD_ONLY", "WIDEN_IN_PLACE"}:
                merged = target_workflow.model_copy(update={"updated_at": now})
            elif disposition == "REPLACE_IN_PLACE":
                winner_is_candidate = True
                merged = candidate_workflow.model_copy(
                    update={
                        "request_id": target_workflow.request_id,
                        "incident_id": target_incident.incident_id,
                        "fencing_token": (target_workflow.fencing_token + 1),
                        "predecessor_workflow_id": (
                            target_workflow.predecessor_workflow_id
                        ),
                        "created_at": target_workflow.created_at,
                        "updated_at": now,
                    }
                )
            else:
                winner_is_candidate = True
                successor = candidate_workflow.model_copy(
                    update={
                        "incident_id": target_incident.incident_id,
                        "fencing_token": target_workflow.fencing_token,
                        "predecessor_workflow_id": (target_workflow.request_id),
                        "not_before": None,
                        "updated_at": now,
                    }
                )
                merged = self.callbacks.prepare_preempting_successor(
                    target_workflow, successor
                )
            finding_name = finding.metric_name or finding.category.value
            incident = target_incident.model_copy(
                update={
                    "event_type": "NODE_RESOURCE_GROUP",
                    "workflow_request_id": merged.request_id,
                    "fencing_token": merged.fencing_token,
                    "state": self.callbacks.incident_state_for_workflow(merged),
                    "policy_version": (
                        finding.policy_version
                        if winner_is_candidate
                        else target_incident.policy_version
                    ),
                    "policy_source": (
                        finding.policy_source
                        if winner_is_candidate
                        else target_incident.policy_source
                    ),
                    "policy_reference": (
                        finding.policy_reference
                        if winner_is_candidate
                        else target_incident.policy_reference
                    ),
                    "official_action": (
                        finding.official_action
                        if winner_is_candidate
                        else target_incident.official_action
                    ),
                    "effective_action": (
                        finding.recommended_action
                        if winner_is_candidate
                        else target_incident.effective_action
                    ),
                    "reasons": bounded_reasons(
                        [
                            *target_incident.reasons,
                            (f"{finding.node_id}: {finding_name}: {finding.reason}"),
                        ]
                    ),
                    "updated_at": now,
                }
            )
            return (
                incident,
                merged.model_copy(
                    update={
                        "incident_id": incident.incident_id,
                        "updated_at": now,
                    }
                ),
            )

        return self.store.merge_attempt_fault_workflow(
            group_key, finding.event_id, build
        )
