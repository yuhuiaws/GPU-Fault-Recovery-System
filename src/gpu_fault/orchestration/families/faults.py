from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from gpu_fault.models import (
    bounded_reasons,
    FaultIncident,
    IncidentState,
    WorkflowRequest,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.policy import (
    ActionDisposition,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)
from gpu_fault.orchestration.disposition import DispositionApplier


@dataclass(frozen=True)
class NodeScopedFaultCallbacks:
    active_node_exclusive_workflow: Callable
    aggregation_deadlines: Callable
    build_workflow: Callable
    claims_node_exclusively: Callable
    incident_state_for_workflow: Callable
    merge_disposition: Callable
    node_group_key: Callable
    prepare_preempting_successor: Callable
    preempt_parallel_job_branch: Callable[
        [WorkflowRequest, WorkflowRequest, str], WorkflowRequest
    ]
    reopen_if_terminal: Callable
    runtime_effective_action: Callable
    widen_node_action_scope: Callable


class NodeScopedFaultService:
    def __init__(
        self,
        store,
        arbiter,
        brancher,
        callbacks: NodeScopedFaultCallbacks,
        *,
        aggregation_window_seconds: int,
        merge_actions: set,
        workflow_preemption_enabled: bool,
    ) -> None:
        self.store = store
        self.arbiter = arbiter
        self.brancher = brancher
        self.callbacks = callbacks
        self.dispositions = DispositionApplier(
            arbiter=arbiter,
            brancher=brancher,
            aggregation_deadlines=callbacks.aggregation_deadlines,
            prepare_preempting_successor=callbacks.prepare_preempting_successor,
            preempt_parallel_job_branch=callbacks.preempt_parallel_job_branch,
            workflow_preemption_enabled=workflow_preemption_enabled,
        )
        self.aggregation_window_seconds = aggregation_window_seconds
        self.merge_actions = merge_actions

    def ingest(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        if not self._eligible(event, decision):
            return None
        group_key = self.callbacks.node_group_key(event.cluster_id, event.node_id)

        def build(
            incident: FaultIncident | None,
            workflow: WorkflowRequest | None,
        ) -> tuple[FaultIncident, WorkflowRequest]:
            return self._build(event, decision, incident, workflow)

        return self.store.merge_attempt_fault_workflow(
            group_key,
            event.event_id,
            build,
        )

    def _eligible(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> bool:
        if (
            self.aggregation_window_seconds == 0
            or decision.disposition is not ActionDisposition.EXECUTABLE
            or decision.action not in self.merge_actions
            or decision.safety_action is not None
        ):
            return False
        return not (
            event.workload_state is WorkloadState.ACTIVE and event.affected_workload_ids
        )

    def _build(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        now = datetime.now(timezone.utc)
        existing_incident, existing_workflow = self.callbacks.reopen_if_terminal(
            existing_incident,
            existing_workflow,
        )
        event_gpus = self._event_gpus(event)
        candidate, candidate_workflow = self._candidate(
            event,
            decision,
            event_gpus,
            now,
        )
        if existing_incident is None or existing_workflow is None:
            return self._new_workflow(
                event,
                candidate,
                candidate_workflow,
                now,
            )
        return self._merge(
            event,
            decision,
            event_gpus,
            candidate,
            candidate_workflow,
            existing_incident,
            existing_workflow,
            now,
        )

    @staticmethod
    def _event_gpus(
        event: XidEvent | SxidEvent,
    ) -> set[str]:
        if isinstance(event, XidEvent) and event.gpu_uuid:
            return {event.gpu_uuid}
        if isinstance(event, SxidEvent):
            return set(event.participating_gpu_uuids)
        return set()

    def _candidate(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        event_gpus: set[str],
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        effective_action, runtime_reason = self.callbacks.runtime_effective_action(
            event, decision
        )
        incident = FaultIncident(
            incident_id=decision.marker.incident_id,
            event_id=event.event_id,
            event_type="GPU_FAULT_GROUP",
            event_source=event.event_source,
            source_boot_id=event.source_boot_id,
            cluster_id=event.cluster_id,
            node_ids=[event.node_id],
            gpu_uuids=sorted(event_gpus),
            job_id=event.job_id,
            attempt_id=event.attempt_id,
            workload_identity_source=event.workload_identity_source,
            policy_version=decision.policy_version,
            policy_source=decision.source.value,
            official_action=decision.official_action,
            effective_action=effective_action,
            safety_action=decision.safety_action,
            drill_id=event.drill_id,
            reasons=[
                *decision.reasons,
                *([runtime_reason] if runtime_reason else []),
            ],
            created_at=now,
            updated_at=now,
        )
        workflow = self.callbacks.build_workflow(
            event,
            decision,
            incident,
        )
        incident = incident.model_copy(
            update={
                "workflow_request_id": workflow.request_id,
                "state": (
                    IncidentState.ACTION_PENDING
                    if workflow.status is WorkflowStatus.PENDING
                    else IncidentState.SAFETY_PENDING
                ),
            }
        )
        return incident, workflow

    def _new_workflow(
        self,
        event: XidEvent | SxidEvent,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        not_before, maximum = self.callbacks.aggregation_deadlines(
            now,
            workflow,
        )
        incumbent = None
        if (
            self.callbacks.claims_node_exclusively(workflow.official_steps)
            and workflow.status is WorkflowStatus.PENDING
        ):
            incumbent = self.callbacks.active_node_exclusive_workflow(
                event.cluster_id,
                {event.node_id},
                exclude_request_ids=frozenset({workflow.request_id}),
                candidate_steps=workflow.official_steps,
            )
        return incident, workflow.model_copy(
            update={
                "not_before": not_before,
                "aggregation_max_deadline": maximum,
                "predecessor_workflow_id": (
                    incumbent.request_id if incumbent is not None else None
                ),
            }
        )

    def _merge(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        event_gpus: set[str],
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        mutable = (
            existing_workflow.status is WorkflowStatus.PENDING
            and existing_workflow.execution_owner_id is None
            and not existing_workflow.completed_step_indexes
        )
        disposition = self.callbacks.merge_disposition(
            existing_workflow,
            candidate_workflow,
            event.node_id,
            event_gpus,
        )
        workflow, winner = self._apply_disposition(
            disposition,
            event.node_id,
            candidate,
            candidate_workflow,
            existing_incident,
            existing_workflow,
            event_gpus,
            mutable,
            now,
        )
        incident = self._merged_incident(
            event,
            decision,
            event_gpus,
            winner,
            existing_incident,
            workflow,
            now,
        )
        scope_source = (
            existing_workflow
            if workflow.request_id == existing_workflow.request_id
            else None
        )
        gpu_scope = self.arbiter.merged_gpu_scope(
            scope_source,
            event.node_id,
            event_gpus,
        )
        workflow = self.callbacks.widen_node_action_scope(
            workflow.model_copy(update={"incident_id": incident.incident_id}),
            gpu_scope,
        )
        return incident, workflow

    def _apply_disposition(
        self,
        disposition: str,
        node_id: str,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        event_gpus: set[str],
        mutable: bool,
        now: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        return self.dispositions.apply(
            disposition,
            node_id=node_id,
            candidate=candidate,
            candidate_workflow=candidate_workflow,
            existing_incident=existing_incident,
            existing_workflow=existing_workflow,
            gpu_uuids=event_gpus,
            mutable=mutable,
            now=now,
        )

    def _merged_incident(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        event_gpus: set[str],
        winner: FaultIncident,
        existing: FaultIncident,
        workflow: WorkflowRequest,
        now: datetime,
    ) -> FaultIncident:
        event_label = (
            f"XID {event.xid}" if isinstance(event, XidEvent) else f"SXID {event.sxid}"
        )
        return existing.model_copy(
            update={
                "event_type": "GPU_FAULT_GROUP",
                "event_source": winner.event_source or existing.event_source,
                "source_boot_id": (winner.source_boot_id or existing.source_boot_id),
                "node_ids": sorted(set(existing.node_ids) | {event.node_id}),
                "gpu_uuids": sorted(set(existing.gpu_uuids) | event_gpus),
                "official_action": winner.official_action,
                "effective_action": winner.effective_action,
                "safety_action": winner.safety_action,
                "policy_source": winner.policy_source,
                "policy_version": winner.policy_version,
                "fencing_token": workflow.fencing_token,
                "reasons": bounded_reasons(
                    [
                        *existing.reasons,
                        *(
                            f"{event.node_id}: {event_label}: {reason}"
                            for reason in decision.reasons
                        ),
                    ]
                ),
                "workflow_request_id": workflow.request_id,
                "state": self.callbacks.incident_state_for_workflow(workflow),
                "updated_at": now,
            }
        )
