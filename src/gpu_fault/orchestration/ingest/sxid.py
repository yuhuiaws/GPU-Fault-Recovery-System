from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkloadState,
    bounded_reasons,
    resolved_step_indexes,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.disposition import MERGING_DISPOSITIONS, Disposition
from gpu_fault.orchestration.workflow_builder import WorkflowBuilder
from gpu_fault.orchestration.workflow_merge import workflow_is_mutable
from gpu_fault.policy import (
    ActionDisposition,
    FaultPolicyDecision,
    SxidEvent,
)
from gpu_fault.store.shared.errors import NotFoundError


@dataclass(frozen=True)
class SxidIngestionCallbacks:
    attempt_observation: Callable[[SxidEvent], Any]
    attempt_group_key: Callable[[str, str, str], str]
    reopen_if_terminal: Callable
    generation_fence: Callable
    active_job_recovery_workflow: Callable
    candidate_recovery_workflow: Callable
    claims_node_exclusively: Callable
    active_node_exclusive_workflow: Callable
    merge_disposition: Callable
    widen_node_action_scope: Callable
    aggregation_deadlines: Callable
    prepare_preempting_successor: Callable
    quiesce_parameters: Callable


@dataclass(frozen=True)
class _SxidContext:
    event: SxidEvent
    decision: FaultPolicyDecision
    observation: Any
    allocation_nodes: list[str]
    profile_version: str | None
    group_key: str
    resolved_workload_ids: list[str]
    source_gpu_count: int


@dataclass(frozen=True)
class _SxidBuildState:
    now: datetime
    existing_incident: FaultIncident | None
    existing_workflow: WorkflowRequest | None
    serialization_predecessor: WorkflowRequest | None
    reset_operation: WorkflowOperation
    has_existing: bool
    mutable: bool
    candidate_preview: WorkflowRequest
    disposition: str | None


@dataclass
class _SxidScope:
    incident_id: str
    workflow_id: str
    primary_event_id: str
    opened_at: datetime
    predecessor_workflow_id: str | None
    not_before: datetime | None
    aggregation_max_deadline: datetime | None
    incident_nodes: list[str]
    fault_nodes: list[str]
    reasons: list[str]
    gpu_mapping: dict[str, list[str]]
    partition_mapping: dict[str, str]
    sxid_mapping: dict[str, list[int]]
    all_gpu_uuids: list[str]


@dataclass(frozen=True)
class _CompiledSxidWorkflow:
    status: WorkflowStatus
    official_steps: list
    safety_steps: list
    errors: list[str]


def merge_sxid_step_scope(
    workflow: WorkflowRequest,
    event: SxidEvent,
) -> WorkflowRequest:
    scoped_operations = {
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    }
    # Finished, superseded and in-flight steps keep the scope they ran with;
    # only pending steps take the new node's GPUs and SXIDs (F-B6).
    untouchable = set(resolved_step_indexes(workflow)) | {
        execution.step_index for execution in workflow.step_executions
    }
    steps = []
    for index, step in enumerate(workflow.official_steps):
        if step.operation not in scoped_operations or index in untouchable:
            steps.append(step)
            continue
        parameters = dict(step.parameters)
        gpu_mapping = {
            str(node_id): list(values)
            for node_id, values in (parameters.get("gpu_uuids_by_node", {})).items()
            if isinstance(values, list)
        }
        gpu_mapping[event.node_id] = sorted(
            set(gpu_mapping.get(event.node_id, [])) | set(event.participating_gpu_uuids)
        )
        sxid_mapping = {
            str(node_id): list(values)
            for node_id, values in (parameters.get("sxids_by_node", {})).items()
            if isinstance(values, list)
        }
        sxid_mapping[event.node_id] = sorted(
            set(sxid_mapping.get(event.node_id, [])) | {event.sxid}
        )
        partition_mapping = dict(parameters.get("fabric_partitions_by_node", {}))
        if event.fabric_partition:
            partition_mapping[event.node_id] = event.fabric_partition
        parameters.update(
            {
                "gpu_uuids_by_node": gpu_mapping,
                "sxids_by_node": sxid_mapping,
                "fabric_partitions_by_node": (partition_mapping),
            }
        )
        steps.append(step.model_copy(update={"parameters": parameters}))
    return workflow.model_copy(update={"official_steps": steps})


class SxidIngestionService:
    def __init__(
        self,
        store,
        builder: WorkflowBuilder,
        arbiter: RecoveryArbiter,
        brancher: DagBrancher,
        callbacks: SxidIngestionCallbacks,
        *,
        multi_node_aggregation_window_seconds: int,
        target_driver_branch: int | None,
        target_firmware_version: str | None,
    ) -> None:
        self.store = store
        self.builder = builder
        self.arbiter = arbiter
        self.brancher = brancher
        self.callbacks = callbacks
        self.multi_node_aggregation_window_seconds = (
            multi_node_aggregation_window_seconds
        )
        self.target_driver_branch = target_driver_branch
        self.target_firmware_version = target_firmware_version

    def ingest_grouped(
        self,
        event: SxidEvent,
        decision: FaultPolicyDecision,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        context = self._context(event, decision)
        if context is None:
            return None

        def build(existing_incident, existing_workflow):
            return self._build(context, existing_incident, existing_workflow)

        return self.store.merge_attempt_fault_workflow(
            context.group_key, event.event_id, build
        )

    def _context(
        self,
        event: SxidEvent,
        decision: FaultPolicyDecision,
    ) -> _SxidContext | None:
        if (
            self.multi_node_aggregation_window_seconds == 0
            or decision.disposition is not ActionDisposition.EXECUTABLE
            or decision.official_action
            not in {
                "RESET_PARTICIPATING_GPUS",
                "RESET_ALL_GPUS_AND_NVSWITCHES",
            }
            or event.workload_state is not WorkloadState.ACTIVE
            or not event.affected_workload_ids
            or not event.participating_gpu_uuids
            or (
                decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
                and not event.fabric_partition
            )
        ):
            return None
        observation = self.callbacks.attempt_observation(event)
        if observation is None:
            return None
        allocation_nodes = sorted(
            {
                container.node_id
                for container in observation.containers
                if container.node_id and not container.terminated
            }
        )
        if event.node_id not in allocation_nodes:
            return None
        profile_version = (
            event.runtime_profile_version or observation.runtime_profile_version
        )
        return _SxidContext(
            event=event,
            decision=decision,
            observation=observation,
            allocation_nodes=allocation_nodes,
            profile_version=profile_version,
            group_key=self.callbacks.attempt_group_key(
                event.cluster_id,
                observation.job_id,
                observation.attempt_id,
            ),
            resolved_workload_ids=sorted(
                set(observation.workload_ids).union(event.affected_workload_ids)
            ),
            source_gpu_count=observation.gpu_count,
        )

    def _build(
        self,
        context: _SxidContext,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        now = datetime.now(timezone.utc)
        existing_incident, existing_workflow = self.callbacks.reopen_if_terminal(
            existing_incident, existing_workflow
        )
        (
            existing_incident,
            existing_workflow,
            generation_ignore_reason,
        ) = self.callbacks.generation_fence(
            context.event,
            context.decision,
            context.observation,
            existing_incident,
            existing_workflow,
        )
        if (
            generation_ignore_reason is not None
            and existing_incident is not None
            and existing_workflow is not None
        ):
            return (
                existing_incident.model_copy(
                    update={
                        "reasons": bounded_reasons(
                            [
                                *existing_incident.reasons,
                                generation_ignore_reason,
                            ]
                        ),
                        "updated_at": now,
                    }
                ),
                existing_workflow,
            )
        state = self._build_state(
            context,
            now,
            existing_incident,
            existing_workflow,
        )
        if state.disposition in MERGING_DISPOSITIONS:
            return self._merge_existing(context, state)
        scope = self._scope(context, state)
        compiled = self._compile(context, state, scope)
        return self._emit(context, state, scope, compiled)

    def _build_state(
        self,
        context: _SxidContext,
        now: datetime,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> _SxidBuildState:
        if existing_incident is None and existing_workflow is None:
            active_recovery = self.callbacks.active_job_recovery_workflow(
                context.observation
            )
            if (
                active_recovery is not None
                and active_recovery[0].attempt_id == context.observation.attempt_id
            ):
                existing_incident, existing_workflow = active_recovery
        reset_operation = (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            if context.decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
            else WorkflowOperation.RESET_GPU
        )
        has_existing = existing_incident is not None and existing_workflow is not None
        mutable = bool(
            has_existing
            and existing_workflow is not None
            and workflow_is_mutable(existing_workflow)
        )
        candidate = self.callbacks.candidate_recovery_workflow(
            context.event, context.decision
        )
        predecessor = None
        if not has_existing and self.callbacks.claims_node_exclusively(
            candidate.official_steps
        ):
            predecessor = self.callbacks.active_node_exclusive_workflow(
                context.event.cluster_id,
                {context.event.node_id},
                exclude_request_ids=frozenset({candidate.request_id}),
                candidate_steps=candidate.official_steps,
            )
        disposition = (
            self.callbacks.merge_disposition(
                existing_workflow,
                candidate,
                context.event.node_id,
                set(context.event.participating_gpu_uuids),
                allow_job_branch_merge=(
                    existing_incident.attempt_id == context.observation.attempt_id
                ),
            )
            if has_existing
            else None
        )
        if disposition == Disposition.REPLACE_IN_PLACE:
            # ``disposition`` says so for a mutable row, and (C-03) for a
            # BLOCKED(NEEDS_OPERATOR) row that never ran a step. In this
            # family "mutable" is what selects the in-place rewrite -- same
            # ``request_id``, ``fencing_token + 1`` -- in ``_scope``/``_emit``.
            mutable = True
        return _SxidBuildState(
            now=now,
            existing_incident=existing_incident,
            existing_workflow=existing_workflow,
            serialization_predecessor=predecessor,
            reset_operation=reset_operation,
            has_existing=has_existing,
            mutable=mutable,
            candidate_preview=candidate,
            disposition=disposition,
        )

    def _merge_existing(
        self,
        context: _SxidContext,
        state: _SxidBuildState,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        event = context.event
        existing_incident = state.existing_incident
        existing_workflow = state.existing_workflow
        assert existing_incident is not None
        assert existing_workflow is not None
        event_gpus = set(event.participating_gpu_uuids)
        if state.disposition == "WIDEN_BRANCH":
            merged = self.brancher.widen_parallel_job_branch(
                existing_workflow,
                state.candidate_preview,
                event.node_id,
                event_gpus,
            )
        elif state.disposition == "WIDEN_IN_PLACE" or (
            state.disposition == "ABSORB" and state.mutable
        ):
            merged = self.callbacks.widen_node_action_scope(
                existing_workflow,
                self.arbiter.merged_gpu_scope(
                    existing_workflow,
                    event.node_id,
                    event_gpus,
                ),
            )
        else:
            merged = existing_workflow
        merged = merge_sxid_step_scope(merged, event)
        workflow_updates = {"updated_at": state.now}
        if state.mutable and state.disposition == "ABSORB":
            not_before, maximum = self.callbacks.aggregation_deadlines(
                state.now, existing_workflow
            )
            workflow_updates.update(
                {
                    "not_before": not_before,
                    "aggregation_max_deadline": maximum,
                }
            )
        merged = merged.model_copy(update=workflow_updates)
        candidate_wins = self.arbiter.workflow_recovery_rank(
            state.candidate_preview
        ) > self.arbiter.workflow_recovery_rank(existing_workflow)
        decision = context.decision
        incident = existing_incident.model_copy(
            update={
                "event_type": "GPU_FAULT_GROUP",
                "node_ids": sorted(set(existing_incident.node_ids) | {event.node_id}),
                "gpu_uuids": sorted(set(existing_incident.gpu_uuids) | event_gpus),
                "official_action": (
                    decision.official_action
                    if candidate_wins
                    else existing_incident.official_action
                ),
                "effective_action": (
                    decision.action
                    if candidate_wins
                    else existing_incident.effective_action
                ),
                "policy_source": (
                    decision.source.value
                    if candidate_wins
                    else existing_incident.policy_source
                ),
                "policy_version": (
                    decision.policy_version
                    if candidate_wins
                    else existing_incident.policy_version
                ),
                "reasons": bounded_reasons(
                    [
                        *existing_incident.reasons,
                        *(
                            f"{event.node_id}: SXID {event.sxid}: {reason}"
                            for reason in decision.reasons
                        ),
                    ]
                ),
                "updated_at": state.now,
            }
        )
        return incident, merged

    def _scope(
        self,
        context: _SxidContext,
        state: _SxidBuildState,
    ) -> _SxidScope:
        event = context.event
        existing_incident = state.existing_incident
        existing_workflow = state.existing_workflow
        previous_reset = None
        reset_parameters = {}
        predecessor_id = (
            state.serialization_predecessor.request_id
            if state.serialization_predecessor is not None
            else None
        )
        if state.has_existing:
            assert existing_incident is not None
            assert existing_workflow is not None
            previous_reset = next(
                (
                    step
                    for step in existing_workflow.official_steps
                    if step.operation is state.reset_operation
                ),
                None,
            )
            if previous_reset is not None:
                reset_parameters = previous_reset.parameters
            incident_id = existing_incident.incident_id
            workflow_id = (
                existing_workflow.request_id if state.mutable else f"workflow-{uuid4()}"
            )
            if state.mutable:
                not_before, maximum = self.callbacks.aggregation_deadlines(
                    state.now, existing_workflow
                )
            else:
                not_before = maximum = None
                predecessor_id = existing_workflow.request_id
            incident_nodes = sorted(set(existing_incident.node_ids) | {event.node_id})
            fault_nodes = sorted(
                set(previous_reset.node_ids if previous_reset is not None else [])
                | {event.node_id}
            )
            primary_event_id = existing_incident.event_id
            opened_at = existing_incident.created_at
            reasons = list(existing_incident.reasons)
        else:
            incident_id = context.decision.marker.incident_id
            workflow_id = f"workflow-{uuid4()}"
            primary_event_id = event.event_id
            opened_at = state.now
            not_before, maximum = self.callbacks.aggregation_deadlines(state.now)
            incident_nodes = fault_nodes = [event.node_id]
            reasons = []
        gpu_mapping = {
            node_id: list(values)
            for node_id, values in (
                reset_parameters.get("gpu_uuids_by_node", {})
            ).items()
        }
        partitions = dict(reset_parameters.get("fabric_partitions_by_node", {}))
        sxids = {
            node_id: list(values)
            for node_id, values in (reset_parameters.get("sxids_by_node", {})).items()
        }
        gpu_mapping[event.node_id] = sorted(
            set(gpu_mapping.get(event.node_id, [])) | set(event.participating_gpu_uuids)
        )
        if event.fabric_partition:
            partitions[event.node_id] = event.fabric_partition
        sxids[event.node_id] = sorted(set(sxids.get(event.node_id, [])) | {event.sxid})
        all_gpu_uuids = sorted(
            (set(existing_incident.gpu_uuids) if state.has_existing else set())
            | {gpu_uuid for values in gpu_mapping.values() for gpu_uuid in values}
        )
        reasons = list(
            dict.fromkeys(
                [
                    *reasons,
                    *(
                        f"{event.node_id}: SXID {event.sxid}: {reason}"
                        for reason in context.decision.reasons
                    ),
                ]
            )
        )
        return _SxidScope(
            incident_id=incident_id,
            workflow_id=workflow_id,
            primary_event_id=primary_event_id,
            opened_at=opened_at,
            predecessor_workflow_id=predecessor_id,
            not_before=not_before,
            aggregation_max_deadline=maximum,
            incident_nodes=incident_nodes,
            fault_nodes=fault_nodes,
            reasons=reasons,
            gpu_mapping=gpu_mapping,
            partition_mapping=partitions,
            sxid_mapping=sxids,
            all_gpu_uuids=all_gpu_uuids,
        )

    def _compile(
        self,
        context: _SxidContext,
        state: _SxidBuildState,
        scope: _SxidScope,
    ) -> _CompiledSxidWorkflow:
        profile = None
        errors = []
        try:
            profile = self.store.get_profile(context.profile_version)
        except NotFoundError:
            errors.append(f"runtime profile does not exist: {context.profile_version}")
        operations = [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            *self.builder.sxid_remediation_operations(context.event),
            state.reset_operation,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
            WorkflowOperation.RESTART_WORKLOAD,
        ]
        steps, compile_errors = self.builder.compile_steps(
            operations,
            profile,
            scope.fault_nodes,
            scope.all_gpu_uuids,
            context.resolved_workload_ids,
        )
        errors.extend(compile_errors)
        scoped_steps = self._scope_steps(context, state, scope, steps)
        safety_steps, safety_errors = self.builder.compile_steps(
            [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.QUARANTINE,
            ],
            profile,
            scope.fault_nodes,
            scope.all_gpu_uuids,
            context.resolved_workload_ids,
        )
        errors.extend(safety_errors)
        status = (
            WorkflowStatus.PENDING
            if not errors
            else WorkflowStatus.SAFETY_PENDING
            if safety_steps and not safety_errors
            else WorkflowStatus.BLOCKED
        )
        return _CompiledSxidWorkflow(
            status=status,
            official_steps=scoped_steps,
            safety_steps=safety_steps,
            errors=list(dict.fromkeys(errors)),
        )

    def _scope_steps(
        self,
        context: _SxidContext,
        state: _SxidBuildState,
        scope: _SxidScope,
        steps,
    ) -> list:
        workload_operations = {
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_WORKLOAD,
        }
        node_action_operations = {
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.REMEDIATE_DRIVER,
            WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
        }
        restart_parameters = {
            "cluster_id": context.event.cluster_id,
            "job_id": context.observation.job_id,
            "source_attempt_id": context.observation.attempt_id,
            "source_gpu_count": context.source_gpu_count,
            "restart_budget": context.observation.restart_budget,
        }
        reset_parameters = {
            "gpu_uuids_by_node": scope.gpu_mapping,
            "fabric_partitions_by_node": scope.partition_mapping,
            "sxids_by_node": scope.sxid_mapping,
        }
        if len(scope.fault_nodes) == 1:
            reset_parameters.update(
                {
                    "fabric_partition": (context.event.fabric_partition),
                    "sxid": context.event.sxid,
                }
            )
        scoped_steps = []
        for step in steps:
            parameters = step.parameters
            if step.operation is WorkflowOperation.STOP_WORKLOADS:
                parameters = {
                    **parameters,
                    "termination_initiator_incident_id": (scope.incident_id),
                }
            elif step.operation is WorkflowOperation.RESTART_WORKLOAD:
                parameters = {
                    **parameters,
                    **restart_parameters,
                }
            elif step.operation in node_action_operations:
                parameters = {
                    **parameters,
                    "gpu_uuids_by_node": scope.gpu_mapping,
                }
                parameters = self._node_action_parameters(
                    context,
                    step.operation,
                    parameters,
                    reset_parameters,
                )
            scoped_steps.append(
                step.model_copy(
                    update={
                        "node_ids": (
                            context.allocation_nodes
                            if step.operation in workload_operations
                            else scope.fault_nodes
                        ),
                        "gpu_uuids": scope.all_gpu_uuids,
                        "parameters": parameters,
                    }
                )
            )
        return scoped_steps

    def _node_action_parameters(
        self,
        context: _SxidContext,
        operation: WorkflowOperation,
        parameters: dict,
        reset_parameters: dict,
    ) -> dict:
        if operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE:
            parameters.update(
                {
                    "sxids_by_node": reset_parameters["sxids_by_node"],
                    "fabric_partitions_by_node": reset_parameters[
                        "fabric_partitions_by_node"
                    ],
                    "classification": (context.event.classification.value),
                    "classification_source": (context.event.classification_source),
                }
            )
        if operation is WorkflowOperation.REMEDIATE_DRIVER:
            parameters["target_driver_branch"] = self.target_driver_branch
        elif operation is WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE:
            parameters["target_firmware_version"] = self.target_firmware_version
        if operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES:
            parameters = {
                **parameters,
                **reset_parameters,
            }
        if operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
            parameters = self.callbacks.quiesce_parameters(
                parameters, context.observation
            )
        return parameters

    def _emit(
        self,
        context: _SxidContext,
        state: _SxidBuildState,
        scope: _SxidScope,
        compiled: _CompiledSxidWorkflow,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        existing_workflow = state.existing_workflow
        generation_token = (
            existing_workflow.fencing_token + 1
            if state.has_existing and state.mutable
            else existing_workflow.fencing_token
            if state.has_existing
            else 1
        )
        incident = FaultIncident(
            incident_id=scope.incident_id,
            event_id=scope.primary_event_id,
            event_type="GPU_FAULT_GROUP",
            event_source=context.event.event_source,
            source_boot_id=context.event.source_boot_id,
            cluster_id=context.event.cluster_id,
            node_ids=scope.incident_nodes,
            gpu_uuids=scope.all_gpu_uuids,
            job_id=context.observation.job_id,
            attempt_id=context.observation.attempt_id,
            workload_identity_source=(
                context.event.workload_identity_source or "SOLE_ACTIVE_ATTEMPT_ON_NODE"
            ),
            policy_version=context.decision.policy_version,
            policy_source=context.decision.source.value,
            official_action=context.decision.official_action,
            effective_action=None,
            drill_id=context.event.drill_id,
            state=(
                IncidentState.ACTION_PENDING
                if compiled.status is WorkflowStatus.PENDING
                else IncidentState.SAFETY_PENDING
                if compiled.status is WorkflowStatus.SAFETY_PENDING
                else IncidentState.ESCALATED
            ),
            workflow_request_id=scope.workflow_id,
            fencing_token=generation_token,
            reasons=scope.reasons,
            created_at=scope.opened_at,
            updated_at=state.now,
        )
        workflow = WorkflowRequest(
            request_id=scope.workflow_id,
            incident_id=scope.incident_id,
            predecessor_workflow_id=(scope.predecessor_workflow_id),
            runtime_profile_version=context.profile_version,
            status=compiled.status,
            official_action=context.decision.official_action,
            fencing_token=generation_token,
            safety_steps=compiled.safety_steps,
            official_steps=compiled.official_steps,
            blocked_reasons=compiled.errors,
            safety_only=compiled.status is WorkflowStatus.SAFETY_PENDING,
            blocked_kind=(
                BlockedKind.NEEDS_OPERATOR
                if compiled.status is WorkflowStatus.BLOCKED
                else None
            ),
            not_before=scope.not_before,
            aggregation_max_deadline=(scope.aggregation_max_deadline),
            lifetime_deadline_at=(
                existing_workflow.lifetime_deadline_at
                if existing_workflow is not None
                else None
            ),
            created_at=scope.opened_at,
            updated_at=state.now,
        )
        if (
            state.has_existing
            and not state.mutable
            and scope.predecessor_workflow_id == existing_workflow.request_id
        ):
            workflow = self.callbacks.prepare_preempting_successor(
                existing_workflow, workflow
            )
        return incident, workflow
