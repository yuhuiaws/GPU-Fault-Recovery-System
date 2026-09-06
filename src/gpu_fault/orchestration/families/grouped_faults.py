from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.models import (
    bounded_reasons,
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.policy import (
    ActionDisposition,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)
from gpu_fault.orchestration.disposition import DispositionApplier


@dataclass(frozen=True)
class GroupedFaultCallbacks:
    active_job_recovery_workflow: Callable
    active_node_exclusive_workflow: Callable
    aggregation_deadlines: Callable
    attempt_group_key: Callable
    attempt_observation: Callable
    build_workflow: Callable
    claims_node_exclusively: Callable
    generation_fence: Callable
    incident_state_for_workflow: Callable
    merge_disposition: Callable
    prepare_preempting_successor: Callable
    preempt_parallel_job_branch: Callable
    quiesce_parameters: Callable
    reopen_if_terminal: Callable
    runtime_effective_action: Callable
    widen_node_action_scope: Callable


@dataclass(frozen=True)
class GroupedFaultContext:
    event: XidEvent | SxidEvent
    decision: FaultPolicyDecision
    observation: Any
    group_key: str
    allocation_nodes: list[str]
    workload_ids: list[str]


class GroupedFaultService:
    def __init__(
        self,
        store,
        arbiter,
        brancher,
        callbacks: GroupedFaultCallbacks,
        *,
        aggregation_window_seconds: int,
        workflow_preemption_enabled: bool,
        attempt_group_actions: set,
        groupable_operations: set,
        official_operation: dict,
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
        self.workflow_preemption_enabled = workflow_preemption_enabled
        self.attempt_group_actions = attempt_group_actions
        self.groupable_operations = groupable_operations
        self.official_operation = official_operation

    def ingest(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        context = self._context(event, decision)
        if context is None:
            return None

        def build(
            incident: FaultIncident | None,
            workflow: WorkflowRequest | None,
        ) -> tuple[FaultIncident, WorkflowRequest]:
            return self._build(context, incident, workflow)

        return self.store.merge_attempt_fault_workflow(
            context.group_key,
            event.event_id,
            build,
        )

    def _context(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> GroupedFaultContext | None:
        official = self.official_operation.get(decision.official_action or "")
        if (
            self.aggregation_window_seconds == 0
            or decision.disposition is not ActionDisposition.EXECUTABLE
            or not (
                decision.action in self.attempt_group_actions
                or official in self.groupable_operations
            )
        ):
            return None
        observation = self.callbacks.attempt_observation(event)
        if observation is None:
            return None
        return GroupedFaultContext(
            event=event,
            decision=decision,
            observation=observation,
            group_key=self.callbacks.attempt_group_key(
                event.cluster_id,
                observation.job_id,
                observation.attempt_id,
            ),
            allocation_nodes=sorted(
                {
                    container.node_id
                    for container in observation.containers
                    if container.node_id and not container.terminated
                }
            ),
            workload_ids=sorted(
                set(observation.workload_ids).union(event.affected_workload_ids)
            ),
        )

    def _build(
        self,
        context: GroupedFaultContext,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        now = datetime.now(timezone.utc)
        existing_incident, existing_workflow = self.callbacks.reopen_if_terminal(
            existing_incident,
            existing_workflow,
        )
        (
            existing_incident,
            existing_workflow,
            fenced,
        ) = self._generation_fence(
            context,
            existing_incident,
            existing_workflow,
            now,
        )
        if fenced is not None:
            return fenced
        existing_incident, existing_workflow = self._active_attempt_recovery(
            context,
            existing_incident,
            existing_workflow,
        )
        candidate, candidate_workflow = self._candidate(
            context,
            now,
        )
        node_incumbent = self._node_incumbent(
            context,
            existing_incident,
            existing_workflow,
        )
        if existing_incident is None or existing_workflow is None:
            incident, workflow = self._new_workflow(
                candidate,
                candidate_workflow,
                node_incumbent,
                now,
            )
        else:
            incident, workflow = self._merge_existing(
                context,
                candidate,
                candidate_workflow,
                existing_incident,
                existing_workflow,
                now,
            )
        return self._scope_workflow(
            context,
            incident,
            workflow,
            existing_workflow,
        )

    def _generation_fence(
        self,
        context: GroupedFaultContext,
        incident: FaultIncident | None,
        workflow: WorkflowRequest | None,
        now: datetime,
    ) -> tuple[
        FaultIncident | None,
        WorkflowRequest | None,
        tuple[FaultIncident, WorkflowRequest] | None,
    ]:
        incident, workflow, reason = self.callbacks.generation_fence(
            context.event,
            context.decision,
            context.observation,
            incident,
            workflow,
        )
        if reason is None or incident is None or workflow is None:
            return incident, workflow, None
        fenced = (
            incident.model_copy(
                update={
                    "reasons": bounded_reasons([*incident.reasons, reason]),
                    "updated_at": now,
                }
            ),
            workflow,
        )
        return incident, workflow, fenced

    def _active_attempt_recovery(
        self,
        context: GroupedFaultContext,
        incident: FaultIncident | None,
        workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident | None, WorkflowRequest | None]:
        if incident is not None or workflow is not None:
            return incident, workflow
        active = self.callbacks.active_job_recovery_workflow(context.observation)
        if (
            active is not None
            and active[0].attempt_id == context.observation.attempt_id
        ):
            return active
        return incident, workflow

    def _candidate(
        self,
        context: GroupedFaultContext,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        event = context.event
        decision = context.decision
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
            gpu_uuids=sorted(self._event_gpus(event)),
            job_id=context.observation.job_id,
            attempt_id=context.observation.attempt_id,
            workload_identity_source=(
                event.workload_identity_source or "SOLE_ACTIVE_ATTEMPT_ON_NODE"
            ),
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

    def _node_incumbent(
        self,
        context: GroupedFaultContext,
        incident: FaultIncident | None,
        workflow: WorkflowRequest | None,
    ):
        if incident is not None or workflow is not None:
            return None
        return self.callbacks.active_node_exclusive_workflow(
            context.event.cluster_id,
            {context.event.node_id},
        )

    def _new_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        node_incumbent,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        not_before, maximum = self.callbacks.aggregation_deadlines(
            now,
            workflow,
        )
        predecessor = None
        if (
            node_incumbent is not None
            and node_incumbent.request_id != workflow.request_id
            and self.callbacks.claims_node_exclusively(workflow.official_steps)
        ):
            predecessor = node_incumbent.request_id
        return incident, workflow.model_copy(
            update={
                "not_before": not_before,
                "aggregation_max_deadline": maximum,
                "predecessor_workflow_id": predecessor,
            }
        )

    def _merge_existing(
        self,
        context: GroupedFaultContext,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        event_gpus = self._event_gpus(context.event)
        disposition = self.callbacks.merge_disposition(
            existing_workflow,
            candidate_workflow,
            context.event.node_id,
            event_gpus,
            allow_job_branch_merge=(
                existing_incident.attempt_id == context.observation.attempt_id
            ),
        )
        mutable = (
            existing_workflow.status is WorkflowStatus.PENDING
            and existing_workflow.execution_owner_id is None
            and not existing_workflow.completed_step_indexes
        )
        workflow, winner = self._apply_disposition(
            disposition,
            context,
            candidate,
            candidate_workflow,
            existing_incident,
            existing_workflow,
            event_gpus,
            mutable,
            now,
        )
        return (
            self._merged_incident(
                context,
                winner,
                existing_incident,
                workflow,
                now,
            ),
            workflow,
        )

    def _apply_disposition(
        self,
        disposition: str,
        context: GroupedFaultContext,
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
            node_id=context.event.node_id,
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
        context: GroupedFaultContext,
        winner: FaultIncident,
        existing: FaultIncident,
        workflow: WorkflowRequest,
        now: datetime,
    ) -> FaultIncident:
        event = context.event
        event_label = (
            f"XID {event.xid}" if isinstance(event, XidEvent) else f"SXID {event.sxid}"
        )
        return existing.model_copy(
            update={
                "event_type": "GPU_FAULT_GROUP",
                "event_source": winner.event_source or existing.event_source,
                "source_boot_id": (winner.source_boot_id or existing.source_boot_id),
                "node_ids": sorted(set(existing.node_ids) | {event.node_id}),
                "gpu_uuids": sorted(set(existing.gpu_uuids) | self._event_gpus(event)),
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
                            for reason in context.decision.reasons
                        ),
                    ]
                ),
                "updated_at": now,
            }
        )

    def _scope_workflow(
        self,
        context: GroupedFaultContext,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        restart_parameters = self._restart_parameters(context)
        workload_operations = {
            WorkflowOperation.CHECKPOINT_WORKLOADS,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_WORKLOAD,
        }
        steps = []
        for step in workflow.official_steps:
            parameters = step.parameters
            if step.operation is WorkflowOperation.STOP_WORKLOADS:
                parameters = {
                    **parameters,
                    "termination_initiator_incident_id": (incident.incident_id),
                }
            elif step.operation is WorkflowOperation.RESTART_WORKLOAD:
                parameters = {**parameters, **restart_parameters}
            elif step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
                parameters = self.callbacks.quiesce_parameters(
                    parameters,
                    context.observation,
                )
            steps.append(
                step.model_copy(
                    update={
                        "node_ids": (
                            context.allocation_nodes
                            if step.operation in workload_operations
                            else step.node_ids
                        ),
                        "workload_ids": context.workload_ids,
                        "parameters": parameters,
                    }
                )
            )
        scope_source = (
            existing_workflow
            if existing_workflow is not None
            and workflow.request_id == existing_workflow.request_id
            else None
        )
        gpu_scope = self.arbiter.merged_gpu_scope(
            scope_source,
            context.event.node_id,
            self._event_gpus(context.event),
        )
        workflow = workflow.model_copy(
            update={
                "incident_id": incident.incident_id,
                "official_steps": steps,
            }
        )
        if not workflow.dag_enabled:
            workflow = self.callbacks.widen_node_action_scope(
                workflow,
                gpu_scope,
            )
        return (
            incident.model_copy(
                update={
                    "workflow_request_id": workflow.request_id,
                    "state": (self.callbacks.incident_state_for_workflow(workflow)),
                }
            ),
            workflow,
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

    @staticmethod
    def _restart_parameters(
        context: GroupedFaultContext,
    ) -> dict:
        observation = context.observation
        return {
            "cluster_id": context.event.cluster_id,
            "job_id": observation.job_id,
            "source_attempt_id": observation.attempt_id,
            "source_gpu_count": observation.gpu_count,
            "restart_budget": observation.restart_budget,
        }
