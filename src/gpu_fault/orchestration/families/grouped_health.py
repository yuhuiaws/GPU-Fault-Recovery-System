from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import (
    bounded_reasons,
    FaultIncident,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration.disposition import DispositionApplier


@dataclass(frozen=True)
class GroupedHealthCallbacks:
    active_job_recovery_workflow: Callable
    aggregation_deadlines: Callable
    attempt_group_key: Callable
    attempt_observation: Callable
    incident_state_for_workflow: Callable
    ingest_node_health: Callable
    is_attempt_grouped_health_finding: Callable
    merge_disposition: Callable
    prepare_preempting_successor: Callable
    preempt_parallel_job_branch: Callable[
        [WorkflowRequest, WorkflowRequest, str], WorkflowRequest
    ]
    quiesce_parameters: Callable
    reopen_if_terminal: Callable
    utc: Callable
    widen_node_action_scope: Callable


@dataclass(frozen=True)
class GroupedHealthContext:
    finding: NodeHealthFinding
    observation: Any
    group_key: str
    allocation_nodes: list[str]
    workload_ids: list[str]
    profile_version: str
    source_gpu_count: int


class GroupedHealthService:
    def __init__(
        self,
        store,
        arbiter,
        brancher,
        callbacks: GroupedHealthCallbacks,
        *,
        aggregation_window_seconds: int,
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

    def ingest(
        self,
        finding: NodeHealthFinding,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        context = self._context(finding)
        if context is None:
            return None

        def build(
            incident: FaultIncident | None,
            workflow: WorkflowRequest | None,
        ) -> tuple[FaultIncident, WorkflowRequest]:
            return self._build(context, incident, workflow)

        return self.store.merge_attempt_fault_workflow(
            context.group_key,
            finding.event_id,
            build,
        )

    def _context(
        self,
        finding: NodeHealthFinding,
    ) -> GroupedHealthContext | None:
        if (
            self.aggregation_window_seconds == 0
            or finding.workload_state is not WorkloadState.ACTIVE
            or not finding.affected_workload_ids
        ):
            return None
        observation = self.callbacks.attempt_observation(finding)
        if observation is None or not self._eligible(
            finding,
            observation,
        ):
            return None
        allocation_nodes = sorted(
            {
                container.node_id
                for container in observation.containers
                if container.node_id and not container.terminated
            }
        )
        if finding.node_id not in allocation_nodes:
            return None
        return GroupedHealthContext(
            finding=finding,
            observation=observation,
            group_key=self.callbacks.attempt_group_key(
                finding.cluster_id,
                observation.job_id,
                observation.attempt_id,
            ),
            allocation_nodes=allocation_nodes,
            workload_ids=sorted(
                set(observation.workload_ids).union(finding.affected_workload_ids)
            ),
            profile_version=(
                finding.runtime_profile_version or observation.runtime_profile_version
            ),
            source_gpu_count=observation.gpu_count,
        )

    def _eligible(
        self,
        finding: NodeHealthFinding,
        observation,
    ) -> bool:
        if self.callbacks.is_attempt_grouped_health_finding(finding):
            return True
        if finding.recommended_action is not RecoveryAction.RUN_DIAGNOSTICS:
            return False
        active = self.callbacks.active_job_recovery_workflow(observation)
        return bool(
            active is not None
            and WorkflowOperation.RESTART_WORKLOAD
            in {step.operation for step in active[1].official_steps}
        )

    def _build(
        self,
        context: GroupedHealthContext,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        now = datetime.now(timezone.utc)
        existing_incident, existing_workflow = self.callbacks.reopen_if_terminal(
            existing_incident,
            existing_workflow,
        )
        candidate, candidate_workflow = self._candidate(context)
        existing_incident, existing_workflow = self._active_attempt_recovery(
            context,
            existing_incident,
            existing_workflow,
        )
        stale = self._stale_generation(
            context,
            candidate_workflow,
            existing_incident,
            existing_workflow,
            now,
        )
        if stale is not None:
            return stale
        if existing_incident is None or existing_workflow is None:
            incident, workflow = self._new_workflow(
                candidate,
                candidate_workflow,
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
        return self._finalize(
            context,
            incident,
            workflow,
            existing_workflow,
            now,
        )

    def _candidate(
        self,
        context: GroupedHealthContext,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        finding = context.finding.model_copy(
            update={
                "runtime_profile_version": context.profile_version,
                "affected_workload_ids": context.workload_ids,
            }
        )
        incident, workflow = self.callbacks.ingest_node_health(
            finding,
            _skip_attempt_grouping=True,
            _skip_terminal_quarantine_merge=True,
            _persist=False,
        )
        if workflow is None:
            raise RuntimeError(
                "attempt-scoped health finding did not produce a workflow"
            )
        workflow = workflow.model_copy(
            update={
                "official_steps": self._scope_candidate_steps(
                    context,
                    incident,
                    workflow,
                )
            }
        )
        incident = incident.model_copy(
            update={
                "event_type": "GPU_FAULT_GROUP",
                "job_id": context.observation.job_id,
                "attempt_id": context.observation.attempt_id,
                "workload_identity_source": ("SOLE_ACTIVE_ATTEMPT_ON_NODE"),
            }
        )
        return incident, workflow

    def _scope_candidate_steps(
        self,
        context: GroupedHealthContext,
        incident: FaultIncident,
        workflow: WorkflowRequest,
    ) -> list:
        restart = {
            "cluster_id": context.finding.cluster_id,
            "job_id": context.observation.job_id,
            "source_attempt_id": context.observation.attempt_id,
            "source_gpu_count": context.source_gpu_count,
            "restart_budget": context.observation.restart_budget,
        }
        workload_operations = {
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_WORKLOAD,
        }
        scoped = []
        for step in workflow.official_steps:
            parameters = step.parameters
            if step.operation is WorkflowOperation.STOP_WORKLOADS:
                parameters = {
                    **parameters,
                    "termination_initiator_incident_id": (incident.incident_id),
                }
            elif step.operation is WorkflowOperation.RESTART_WORKLOAD:
                parameters = {**parameters, **restart}
            elif step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
                parameters = self.callbacks.quiesce_parameters(
                    parameters,
                    context.observation,
                )
            scoped.append(
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
        return scoped

    def _active_attempt_recovery(
        self,
        context: GroupedHealthContext,
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

    def _stale_generation(
        self,
        context: GroupedHealthContext,
        candidate: WorkflowRequest,
        incident: FaultIncident | None,
        workflow: WorkflowRequest | None,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        if incident is None or workflow is None:
            return None
        observation = context.observation
        if observation.started_at is None or self.callbacks.utc(
            observation.started_at
        ) <= self.callbacks.utc(context.finding.observed_at):
            return None
        candidate_rank = self.arbiter.workflow_recovery_rank(candidate)
        current_rank = self.arbiter.workflow_recovery_rank(workflow)
        candidate_intent = self.arbiter.merge_intent(candidate)
        current_intent = self.arbiter.merge_intent(workflow)
        if candidate_rank > current_rank or (
            candidate_rank == current_rank and not candidate_intent <= current_intent
        ):
            return None
        reason = (
            "Ignored stale health action for previous attempt "
            "generation: event_time="
            f"{self.callbacks.utc(context.finding.observed_at).isoformat()}, "
            f"current_attempt={observation.attempt_id}, "
            "current_attempt_started_at="
            f"{self.callbacks.utc(observation.started_at).isoformat()}, "
            f"candidate_rank={candidate_rank}, "
            f"current_recovery_rank={current_rank}; "
            "action is not an escalation"
        )
        return (
            incident.model_copy(
                update={
                    "reasons": bounded_reasons([*incident.reasons, reason]),
                    "updated_at": now,
                }
            ),
            workflow,
        )

    def _new_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        not_before, maximum = self.callbacks.aggregation_deadlines(
            now,
            workflow,
        )
        return incident, workflow.model_copy(
            update={
                "not_before": not_before,
                "aggregation_max_deadline": maximum,
            }
        )

    def _merge_existing(
        self,
        context: GroupedHealthContext,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        gpus = set(context.finding.gpu_uuids)
        disposition = self.callbacks.merge_disposition(
            existing_workflow,
            candidate_workflow,
            context.finding.node_id,
            gpus,
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
            gpus,
            mutable,
            now,
        )
        finding = context.finding
        finding_name = finding.metric_name or finding.diagnostic_parameters.get(
            "diagnostic_reason",
            "node health",
        )
        incident = existing_incident.model_copy(
            update={
                "event_type": "GPU_FAULT_GROUP",
                "node_ids": sorted(set(existing_incident.node_ids) | {finding.node_id}),
                "gpu_uuids": sorted(set(existing_incident.gpu_uuids) | gpus),
                "job_id": context.observation.job_id,
                "attempt_id": context.observation.attempt_id,
                "workload_identity_source": ("SOLE_ACTIVE_ATTEMPT_ON_NODE"),
                "fencing_token": workflow.fencing_token,
                "policy_version": winner.policy_version,
                "policy_source": winner.policy_source,
                "policy_reference": winner.policy_reference,
                "official_action": winner.official_action,
                "effective_action": winner.effective_action,
                "safety_action": winner.safety_action,
                "reasons": bounded_reasons(
                    [
                        *existing_incident.reasons,
                        (f"{finding.node_id}: {finding_name}: {finding.reason}"),
                    ]
                ),
                "updated_at": now,
            }
        )
        return incident, workflow

    def _apply_disposition(
        self,
        disposition: str,
        context: GroupedHealthContext,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        gpus: set[str],
        mutable: bool,
        now: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        return self.dispositions.apply(
            disposition,
            node_id=context.finding.node_id,
            candidate=candidate,
            candidate_workflow=candidate_workflow,
            existing_incident=existing_incident,
            existing_workflow=existing_workflow,
            gpu_uuids=gpus,
            mutable=mutable,
            now=now,
        )

    def _finalize(
        self,
        context: GroupedHealthContext,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        existing_workflow: WorkflowRequest | None,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        finding = context.finding
        if (
            existing_workflow is not None
            and workflow.request_id == existing_workflow.request_id
            and not workflow.dag_enabled
        ):
            workflow = self.callbacks.widen_node_action_scope(
                workflow,
                self.arbiter.merged_gpu_scope(
                    existing_workflow,
                    finding.node_id,
                    set(finding.gpu_uuids),
                ),
            )
        steps = [
            (
                step.model_copy(
                    update={
                        "parameters": {
                            "termination_initiator_incident_id": (incident.incident_id)
                        }
                    }
                )
                if step.operation is WorkflowOperation.STOP_WORKLOADS
                else step
            )
            for step in workflow.official_steps
        ]
        workflow = workflow.model_copy(
            update={
                "incident_id": incident.incident_id,
                "official_steps": steps,
            }
        )
        incident = incident.model_copy(
            update={
                "workflow_request_id": workflow.request_id,
                "state": self.callbacks.incident_state_for_workflow(workflow),
                "updated_at": now,
            }
        )
        return incident, workflow
