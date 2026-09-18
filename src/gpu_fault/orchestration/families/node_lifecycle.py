from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.regional_compatibility import ACTIVATION_INHIBITION_VERSION
from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkloadState,
    bounded_reasons,
    resolved_step_indexes,
)
from gpu_fault.orchestration.disposition import Disposition, DispositionApplier
from gpu_fault.orchestration.families.identity import derived_record_id
from gpu_fault.orchestration.workflow_merge import (
    WorkflowMergeService,
    workflow_is_mutable,
)
from gpu_fault.workflow_quarantine import inherit_terminal_quarantine
from gpu_fault.store import NotFoundError


@dataclass(frozen=True)
class NodeLifecycleCallbacks:
    active_job_recovery_workflow: Callable
    active_node_exclusive_workflow: Callable
    aggregation_deadlines: Callable
    can_append_parallel_job_branch: Callable
    claims_node_exclusively: Callable
    inventory_validation_parameters: Callable
    preemption_scope_matches: Callable
    prepare_preempting_successor: Callable


@dataclass(frozen=True)
class ReplacementContext:
    finding: NodeHealthFinding
    observation: Any
    allocation_nodes: list[str]
    profile_version: str
    profile: Any
    profile_errors: list[str]
    group_key: str
    workload_ids: list[str]
    source_gpu_count: int


@dataclass
class ReplacementState:
    merge_existing: bool
    incident_id: str
    workflow_id: str
    opened_at: datetime
    not_before: datetime | None
    aggregation_max_deadline: datetime | None
    fault_nodes: list[str]
    gpu_uuids: list[str]
    reasons: list[str]
    primary_event_id: str


class NodeLifecycleOperationService:
    activation_inhibition_version = ACTIVATION_INHIBITION_VERSION

    def __init__(
        self,
        store,
        builder,
        merger: WorkflowMergeService,
        callbacks: NodeLifecycleCallbacks,
        *,
        aggregation_window_seconds: int,
    ) -> None:
        self.store = store
        self.builder = builder
        self.merger = merger
        self.arbiter = merger.arbiter
        self.brancher = merger.brancher
        self.callbacks = callbacks
        self.aggregation_window_seconds = aggregation_window_seconds
        self.dispositions = DispositionApplier(
            arbiter=self.arbiter,
            brancher=self.brancher,
            aggregation_deadlines=callbacks.aggregation_deadlines,
            prepare_preempting_successor=callbacks.prepare_preempting_successor,
            preempt_parallel_job_branch=merger.preempt_parallel_branch,
            workflow_preemption_enabled=merger.preemption_enabled,
        )

    def ingest_grouped_node_replacement(
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

        return self.store.merge_replacement_workflow(
            context.group_key,
            finding.event_id,
            build,
        )

    def _context(
        self,
        finding: NodeHealthFinding,
    ) -> ReplacementContext | None:
        if (
            self.aggregation_window_seconds == 0
            or finding.recommended_action
            not in {
                RecoveryAction.REBOOT_NODE,
                RecoveryAction.REPLACE_NODE,
            }
            or finding.workload_state is not WorkloadState.ACTIVE
            or not finding.affected_workload_ids
        ):
            return None
        workload_ids = set(finding.affected_workload_ids)
        candidates = [
            observation
            for observation in self.store.list_attempt_observations(finding.cluster_id)
            if (
                observation.job_id in workload_ids
                or workload_ids.intersection(observation.workload_ids)
            )
            and observation.workload_phase.value in {"PENDING", "RUNNING"}
        ]
        if not candidates:
            return None
        observation = max(
            candidates,
            key=lambda item: item.observed_at,
        )
        allocation_nodes = sorted(
            {
                container.node_id
                for container in observation.containers
                if container.node_id
            }
        )
        if not allocation_nodes or finding.node_id not in allocation_nodes:
            return None
        profile_version = (
            finding.runtime_profile_version or observation.runtime_profile_version
        )
        profile = None
        errors = []
        try:
            profile = self.store.get_profile(profile_version)
        except NotFoundError:
            errors.append("runtime profile does not exist: " + profile_version)
        group_key = json.dumps(
            [
                finding.cluster_id,
                observation.job_id,
                observation.attempt_id,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return ReplacementContext(
            finding=finding,
            observation=observation,
            allocation_nodes=allocation_nodes,
            profile_version=profile_version,
            profile=profile,
            profile_errors=errors,
            group_key=group_key,
            workload_ids=sorted(workload_ids.union(observation.workload_ids)),
            source_gpu_count=observation.gpu_count,
        )

    def _state(
        self,
        context: ReplacementContext,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
        now: datetime,
    ) -> ReplacementState:
        # PENDING only. Merging into a BLOCKED record recompiled it under the
        # same id and, once the profile had arrived, silently un-blocked it
        # with its ``blocked_kind`` cleared -- exactly what F-B4 forbids for
        # every other family through ``disposition()`` (C-06). A finding that
        # lands while the group's record is BLOCKED gets its own record.
        merge = (
            existing_incident is not None
            and existing_workflow is not None
            and existing_workflow.not_before is not None
            and existing_workflow.not_before > now
            and workflow_is_mutable(existing_workflow)
            and self._same_activation_authority(context.finding, existing_workflow)
        )
        if merge:
            not_before, maximum = self.callbacks.aggregation_deadlines(
                now,
                existing_workflow,
            )
            return ReplacementState(
                merge_existing=True,
                incident_id=existing_incident.incident_id,
                workflow_id=existing_workflow.request_id,
                opened_at=existing_incident.created_at,
                not_before=not_before,
                aggregation_max_deadline=maximum,
                fault_nodes=sorted(
                    set(existing_incident.node_ids) | {context.finding.node_id}
                ),
                gpu_uuids=sorted(
                    set(existing_incident.gpu_uuids) | set(context.finding.gpu_uuids)
                ),
                reasons=list(
                    dict.fromkeys(
                        [
                            *existing_incident.reasons,
                            context.finding.reason,
                        ]
                    )
                ),
                primary_event_id=existing_incident.event_id,
            )
        not_before, maximum = self.callbacks.aggregation_deadlines(now)
        # Derived from the finding's event: a re-post whose link went stale is
        # rebuilt into the same records rather than a second pair (F-B7).
        return ReplacementState(
            merge_existing=False,
            incident_id=derived_record_id(
                "incident", "node-replacement", context.finding.event_id
            ),
            workflow_id=derived_record_id(
                "workflow", "node-replacement", context.finding.event_id
            ),
            opened_at=now,
            not_before=not_before,
            aggregation_max_deadline=maximum,
            fault_nodes=[context.finding.node_id],
            gpu_uuids=sorted(set(context.finding.gpu_uuids)),
            reasons=[context.finding.reason],
            primary_event_id=context.finding.event_id,
        )

    @staticmethod
    def _same_activation_authority(
        finding: NodeHealthFinding, workflow: WorkflowRequest
    ) -> bool:
        inhibited = any(
            "activation_forbidden" in step.parameters
            for step in workflow.official_steps
        )
        incoming = "activation_forbidden" in finding.diagnostic_parameters
        return inhibited == incoming and (
            not inhibited or finding.recommended_action is RecoveryAction.REPLACE_NODE
        )

    def _compile_steps(
        self,
        context: ReplacementContext,
        state: ReplacementState,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[list, list[str]]:
        finding = context.finding
        operations = [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
            WorkflowOperation.STOP_WORKLOADS,
            (
                WorkflowOperation.RESTART_NODE
                if finding.recommended_action is RecoveryAction.REBOOT_NODE
                else WorkflowOperation.REPLACE_NODE
            ),
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
            WorkflowOperation.RESTART_WORKLOAD,
        ]
        steps, errors = self.builder.compile_steps(
            operations,
            context.profile,
            state.fault_nodes,
            state.gpu_uuids,
            context.workload_ids,
        )
        requirements = {
            WorkflowOperation.VALIDATE_GPU: {},
            WorkflowOperation.VALIDATE_FABRIC: {},
        }
        if existing_workflow is not None:
            for step in existing_workflow.official_steps:
                if step.operation not in requirements:
                    continue
                value = step.parameters.get("inventory_requirements_by_node", {})
                if isinstance(value, dict):
                    requirements[step.operation].update(value)
        current = self.callbacks.inventory_validation_parameters(finding).get(
            "inventory_requirements_by_node", {}
        )
        for value in requirements.values():
            value.update(current)
        restart = {
            "cluster_id": finding.cluster_id,
            "job_id": context.observation.job_id,
            "source_attempt_id": context.observation.attempt_id,
            "source_gpu_count": context.source_gpu_count,
            "restart_budget": context.observation.restart_budget,
        }
        result = []
        for step in steps:
            parameters = step.parameters
            if step.operation is WorkflowOperation.RESTART_WORKLOAD:
                parameters = restart
            elif step.operation is WorkflowOperation.STOP_WORKLOADS:
                parameters = {"termination_initiator_incident_id": state.incident_id}
            elif step.operation is WorkflowOperation.REPLACE_NODE:
                if (
                    finding.diagnostic_parameters.get("replacement_strategy")
                    == "HEALTHY_WARM_SPARE_ONLY"
                ):
                    parameters = {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
                if "activation_forbidden" in finding.diagnostic_parameters:
                    parameters = {
                        **parameters,
                        "activation_forbidden": finding.diagnostic_parameters[
                            "activation_forbidden"
                        ],
                    }
            elif step.operation in requirements and requirements[step.operation]:
                parameters = {
                    **parameters,
                    "inventory_requirements_by_node": requirements[step.operation],
                }
            result.append(
                step.model_copy(
                    update={
                        "node_ids": (
                            context.allocation_nodes
                            if step.operation
                            in {
                                WorkflowOperation.STOP_WORKLOADS,
                                WorkflowOperation.RESTART_WORKLOAD,
                            }
                            else state.fault_nodes
                        ),
                        "parameters": parameters,
                    }
                )
            )
        return result, [*context.profile_errors, *errors]

    def _build(
        self,
        context: ReplacementContext,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        now = datetime.now(timezone.utc)
        state = self._state(
            context,
            existing_incident,
            existing_workflow,
            now,
        )
        if (
            state.merge_existing
            and existing_incident is not None
            and existing_workflow is not None
            and self._mixed_recovery_intent(context.finding, existing_workflow)
        ):
            return self._merge_mixed(
                context, state, existing_incident, existing_workflow, now
            )
        steps, errors = self._compile_steps(
            context,
            state,
            existing_workflow,
        )
        incumbent = None
        predecessor = None
        if (
            not state.merge_existing
            and not errors
            and self.callbacks.claims_node_exclusively(steps)
        ):
            incumbent = self.callbacks.active_node_exclusive_workflow(
                context.finding.cluster_id,
                set(state.fault_nodes),
                exclude_request_ids=frozenset({state.workflow_id}),
                candidate_steps=steps,
            )
            if incumbent is not None:
                predecessor = incumbent.request_id
                state.reasons.append(
                    "serialized behind in-flight node-exclusive "
                    f"workflow {incumbent.request_id}"
                )
        # A workflow this family emits must never sit below its incident's
        # generation: ``fencing_token=1`` on an incident already at generation 4
        # is exactly the record the abandoned-generation sweep terminalizes
        # as left behind (P0-59B).
        generation = max(
            1,
            existing_incident.fencing_token if existing_incident is not None else 1,
            existing_workflow.fencing_token if existing_workflow is not None else 1,
        )
        incident = self._incident(context, state, errors, now, generation=generation)
        workflow = self._workflow(
            context,
            state,
            steps,
            errors,
            predecessor,
            now,
            generation=generation,
        )
        quarantine_source = existing_workflow if state.merge_existing else incumbent
        if quarantine_source is not None:
            workflow = inherit_terminal_quarantine(quarantine_source, workflow)
        parallel = self._parallel_branch(
            context,
            state,
            incident,
            workflow,
            now,
            incumbent=incumbent,
        )
        if parallel is not None:
            return parallel
        if incumbent is not None and self.callbacks.preemption_scope_matches(
            self.store.get_incident(incumbent.incident_id),
            incident,
        ):
            workflow = self.callbacks.prepare_preempting_successor(
                incumbent,
                workflow,
            )
        return incident, workflow

    @staticmethod
    def _mixed_recovery_intent(
        finding: NodeHealthFinding, workflow: WorkflowRequest
    ) -> bool:
        operation = (
            WorkflowOperation.RESTART_NODE
            if finding.recommended_action is RecoveryAction.REBOOT_NODE
            else WorkflowOperation.REPLACE_NODE
        )
        return any(
            step.operation
            in {WorkflowOperation.RESTART_NODE, WorkflowOperation.REPLACE_NODE}
            and step.operation is not operation
            and index not in workflow.superseded_step_indexes
            for index, step in enumerate(workflow.official_steps)
        )

    def _merge_mixed(
        self,
        context: ReplacementContext,
        merged_state: ReplacementState,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        candidate_state = self._state(context, None, None, now)
        steps, errors = self._compile_steps(context, candidate_state, existing_workflow)
        generation = max(
            existing_incident.fencing_token, existing_workflow.fencing_token
        )
        candidate = self._incident(
            context, candidate_state, errors, now, generation=generation
        )
        candidate_workflow = self._workflow(
            context, candidate_state, steps, errors, None, now, generation=generation
        )
        disposition = self.merger.disposition(
            existing_workflow,
            candidate_workflow,
            context.finding.node_id,
            set(context.finding.gpu_uuids),
        )
        if disposition is Disposition.REPLACE_IN_PLACE:
            # The arbiter authorized replacing this still-unissued whole plan.
            # Recompile its complete fault scope, not only the latest reporter.
            steps, errors = self._compile_steps(
                context, merged_state, existing_workflow
            )
            candidate_workflow = self._workflow(
                context,
                candidate_state,
                steps,
                errors,
                None,
                now,
                generation=generation,
            )
        workflow, winner = self.dispositions.apply(
            disposition,
            node_id=context.finding.node_id,
            candidate=candidate,
            candidate_workflow=candidate_workflow,
            existing_incident=existing_incident,
            existing_workflow=existing_workflow,
            gpu_uuids=set(context.finding.gpu_uuids),
            mutable=True,
            now=now,
        )
        immutable = resolved_step_indexes(workflow) | {
            execution.step_index for execution in workflow.step_executions
        }
        workflow = workflow.model_copy(
            update={
                "incident_id": existing_incident.incident_id,
                "official_steps": [
                    step.model_copy(
                        update={
                            "parameters": {
                                **step.parameters,
                                "termination_initiator_incident_id": existing_incident.incident_id,
                            }
                        }
                    )
                    if step.operation is WorkflowOperation.STOP_WORKLOADS
                    and index not in immutable
                    else step
                    for index, step in enumerate(workflow.official_steps)
                ],
            }
        )
        incident = existing_incident.model_copy(
            update={
                "node_ids": merged_state.fault_nodes,
                "gpu_uuids": merged_state.gpu_uuids,
                "workflow_request_id": workflow.request_id,
                "fencing_token": workflow.fencing_token,
                "official_action": winner.official_action,
                "effective_action": winner.effective_action,
                "policy_source": winner.policy_source,
                "policy_version": winner.policy_version,
                "policy_reference": winner.policy_reference,
                "state": (
                    IncidentState.ACTION_PENDING
                    if workflow.status
                    in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}
                    else IncidentState.ESCALATED
                ),
                "reasons": bounded_reasons(merged_state.reasons),
                "updated_at": now,
            }
        )
        return incident, workflow

    @staticmethod
    def _incident(
        context: ReplacementContext,
        state: ReplacementState,
        errors: list[str],
        now: datetime,
        *,
        generation: int = 1,
    ) -> FaultIncident:
        finding = context.finding
        return FaultIncident(
            incident_id=state.incident_id,
            event_id=state.primary_event_id,
            event_type="NODE_HEALTH_GROUP",
            cluster_id=finding.cluster_id,
            node_ids=state.fault_nodes,
            gpu_uuids=state.gpu_uuids,
            job_id=context.observation.job_id,
            attempt_id=context.observation.attempt_id,
            workload_identity_source="SOLE_ACTIVE_ATTEMPT_ON_NODE",
            policy_version="site-node-health-policy/v1",
            policy_source=(
                finding.policy_source
                if finding.policy_source == "SITE_SYNTHETIC_REPLACEMENT_TEST"
                and "activation_forbidden" in finding.diagnostic_parameters
                else "SITE_NODE_HEALTH"
            ),
            official_action=None,
            effective_action=finding.recommended_action,
            drill_id=finding.drill_id,
            state=(IncidentState.ESCALATED if errors else IncidentState.ACTION_PENDING),
            workflow_request_id=state.workflow_id,
            fencing_token=generation,
            reasons=state.reasons,
            created_at=state.opened_at,
            updated_at=now,
        )

    @staticmethod
    def _workflow(
        context: ReplacementContext,
        state: ReplacementState,
        steps: list,
        errors: list[str],
        predecessor: str | None,
        now: datetime,
        *,
        generation: int = 1,
    ) -> WorkflowRequest:
        status = WorkflowStatus.BLOCKED if errors else WorkflowStatus.PENDING
        return WorkflowRequest(
            request_id=state.workflow_id,
            incident_id=state.incident_id,
            runtime_profile_version=context.profile_version,
            status=status,
            official_action=context.finding.recommended_action.value,
            fencing_token=generation,
            official_steps=steps,
            predecessor_workflow_id=predecessor,
            blocked_reasons=errors,
            blocked_kind=(BlockedKind.NEEDS_OPERATOR if errors else None),
            not_before=state.not_before,
            aggregation_max_deadline=state.aggregation_max_deadline,
            created_at=state.opened_at,
            updated_at=now,
        )

    def _parallel_branch(
        self,
        context: ReplacementContext,
        state: ReplacementState,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        now: datetime,
        *,
        incumbent: WorkflowRequest | None = None,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        if state.merge_existing:
            return None
        if incumbent is not None:
            # The candidate is serialized behind an in-flight node-exclusive
            # workflow; branching it into the active job workflow as well
            # would run it twice (P0-55B).
            return None
        active = self.callbacks.active_job_recovery_workflow(context.observation)
        if (
            active is None
            or active[1].request_id == workflow.request_id
            or not self._same_activation_authority(context.finding, active[1])
            or not (
                active[1].status is WorkflowStatus.RUNNING
                or active[1].not_before is None
                or active[1].not_before > now
            )
            or not self.callbacks.can_append_parallel_job_branch(active[1], workflow)
        ):
            return None
        active_incident, active_workflow = active
        parallel = self.brancher.append_parallel_job_branch(
            active_workflow,
            workflow,
        )
        winner = (
            incident
            if self.arbiter.workflow_recovery_rank(workflow)
            > self.arbiter.workflow_recovery_rank(active_workflow)
            else active_incident
        )
        return (
            active_incident.model_copy(
                update={
                    "node_ids": sorted(
                        set(active_incident.node_ids) | set(incident.node_ids)
                    ),
                    "gpu_uuids": sorted(
                        set(active_incident.gpu_uuids) | set(incident.gpu_uuids)
                    ),
                    "official_action": winner.official_action,
                    "effective_action": winner.effective_action,
                    "workflow_request_id": parallel.request_id,
                    "reasons": bounded_reasons(
                        [
                            *active_incident.reasons,
                            *incident.reasons,
                        ]
                    ),
                    "updated_at": now,
                }
            ),
            parallel,
        )
