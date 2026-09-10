"""Attempt-grouped SXID ingestion.

Fatal access/trunk SXIDs on a training attempt share one workflow keyed by
``cluster + job + attempt``. The first event compiles the SXID plan for its
node; every later event compiles the same plan for *its* node and hands the
merge verdict to :class:`DispositionApplier`, exactly as the XID family does,
so a second node with a different reset class becomes a ``branch:<node>``
beside ``branch:initial`` under one ``shared`` STOP and one ``join`` RESTART.
This module used to apply the verdict itself and rewrote the plan in place
with only the newest node's reset, or minted a successor workflow with a
second STOP/RESTART once the first was running.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkloadState,
    bounded_reasons,
    resolved_step_indexes,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.disposition import Disposition, DispositionApplier
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
    claims_node_exclusively: Callable
    active_node_exclusive_workflow: Callable
    merge_disposition: Callable
    widen_node_action_scope: Callable
    aggregation_deadlines: Callable
    prepare_preempting_successor: Callable
    preempt_parallel_job_branch: Callable
    incident_state_for_workflow: Callable
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
class _CompiledSxidWorkflow:
    status: WorkflowStatus
    official_steps: list[WorkflowStepSpec]
    safety_steps: list[WorkflowStepSpec]
    errors: list[str]


_WORKLOAD_OPERATIONS = frozenset(
    {
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_WORKLOAD,
    }
)
_NODE_ACTION_OPERATIONS = frozenset(
    {
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.REMEDIATE_DRIVER,
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    }
)


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
        if (
            step.operation not in scoped_operations
            or index in untouchable
            # In a DAG each node's reset lives in its own branch; another
            # node's branch does not take this event's GPUs (F-B6 (2)).
            or (workflow.dag_enabled and event.node_id not in step.node_ids)
        ):
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
        workflow_preemption_enabled: bool = True,
    ) -> None:
        self.store = store
        self.builder = builder
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
        if existing_incident is None and existing_workflow is None:
            # An XID on this attempt may already have opened the job workflow;
            # the SXID joins it instead of racing it.
            active_recovery = self.callbacks.active_job_recovery_workflow(
                context.observation
            )
            if (
                active_recovery is not None
                and active_recovery[0].attempt_id == context.observation.attempt_id
            ):
                existing_incident, existing_workflow = active_recovery
        if existing_incident is None or existing_workflow is None:
            return self._new_workflow(context, now)
        return self._merge(context, now, existing_incident, existing_workflow)

    def _new_workflow(
        self,
        context: _SxidContext,
        now: datetime,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        incident, workflow = self._candidate(
            context, now, context.decision.marker.incident_id
        )
        predecessor = None
        if self.callbacks.claims_node_exclusively(workflow.official_steps):
            incumbent = self.callbacks.active_node_exclusive_workflow(
                context.event.cluster_id,
                {context.event.node_id},
                exclude_request_ids=frozenset({workflow.request_id}),
                candidate_steps=workflow.official_steps,
            )
            if incumbent is not None:
                predecessor = incumbent.request_id
        not_before, maximum = self.callbacks.aggregation_deadlines(now)
        return incident, workflow.model_copy(
            update={
                "predecessor_workflow_id": predecessor,
                "not_before": not_before,
                "aggregation_max_deadline": maximum,
            }
        )

    def _merge(
        self,
        context: _SxidContext,
        now: datetime,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        event = context.event
        event_gpus = set(event.participating_gpu_uuids)
        candidate, candidate_workflow = self._candidate(
            context, now, existing_incident.incident_id
        )
        mutable = workflow_is_mutable(existing_workflow)
        disposition = Disposition(
            self.callbacks.merge_disposition(
                existing_workflow,
                candidate_workflow,
                event.node_id,
                event_gpus,
                allow_job_branch_merge=(
                    existing_incident.attempt_id == context.observation.attempt_id
                ),
            )
        )
        workflow, winner = self.dispositions.apply(
            disposition,
            node_id=event.node_id,
            candidate=candidate,
            candidate_workflow=candidate_workflow,
            existing_incident=existing_incident,
            existing_workflow=existing_workflow,
            gpu_uuids=event_gpus,
            mutable=mutable,
            now=now,
        )
        workflow = self._scope_merged(context, workflow, existing_workflow)
        decision = context.decision
        incident = existing_incident.model_copy(
            update={
                "event_type": "GPU_FAULT_GROUP",
                "event_source": winner.event_source or existing_incident.event_source,
                "source_boot_id": (
                    winner.source_boot_id or existing_incident.source_boot_id
                ),
                "node_ids": sorted(set(existing_incident.node_ids) | {event.node_id}),
                "gpu_uuids": sorted(set(existing_incident.gpu_uuids) | event_gpus),
                "official_action": winner.official_action,
                "effective_action": winner.effective_action,
                "policy_source": winner.policy_source,
                "policy_version": winner.policy_version,
                "fencing_token": workflow.fencing_token,
                "workflow_request_id": workflow.request_id,
                "state": self.callbacks.incident_state_for_workflow(workflow),
                "reasons": bounded_reasons(
                    [
                        *existing_incident.reasons,
                        *(
                            f"{event.node_id}: SXID {event.sxid}: {reason}"
                            for reason in decision.reasons
                        ),
                    ]
                ),
                "updated_at": now,
            }
        )
        return incident, workflow

    def _scope_merged(
        self,
        context: _SxidContext,
        workflow: WorkflowRequest,
        existing_workflow: WorkflowRequest,
    ) -> WorkflowRequest:
        """Point the merged plan's pending steps at the attempt as it is now.

        Finished, superseded and in-flight steps are history or a command an
        agent already holds (F-B6, C-08); only steps still to run take this
        event's observation. A flat plan then widens its node steps to the
        merged GPU scope; a DAG keeps each node's scope in its own branch.
        """
        event = context.event
        restart_parameters = self._restart_parameters(context)
        untouchable = set(resolved_step_indexes(workflow)) | {
            execution.step_index for execution in workflow.step_executions
        }
        steps = []
        for index, step in enumerate(workflow.official_steps):
            if index in untouchable:
                steps.append(step)
                continue
            parameters = step.parameters
            if step.operation is WorkflowOperation.STOP_WORKLOADS:
                parameters = {
                    **parameters,
                    "termination_initiator_incident_id": workflow.incident_id,
                }
            elif step.operation is WorkflowOperation.RESTART_WORKLOAD:
                parameters = {**parameters, **restart_parameters}
            elif step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
                parameters = self.callbacks.quiesce_parameters(
                    parameters, context.observation
                )
            steps.append(
                step.model_copy(
                    update={
                        "node_ids": (
                            context.allocation_nodes
                            if step.operation in _WORKLOAD_OPERATIONS
                            else step.node_ids
                        ),
                        "workload_ids": context.resolved_workload_ids,
                        "parameters": parameters,
                    }
                )
            )
        workflow = workflow.model_copy(update={"official_steps": steps})
        if not workflow.dag_enabled:
            scope_source = (
                existing_workflow
                if workflow.request_id == existing_workflow.request_id
                else None
            )
            workflow = self.callbacks.widen_node_action_scope(
                workflow,
                self.arbiter.merged_gpu_scope(
                    scope_source,
                    event.node_id,
                    set(event.participating_gpu_uuids),
                ),
            )
        return merge_sxid_step_scope(workflow, event)

    def _candidate(
        self,
        context: _SxidContext,
        now: datetime,
        incident_id: str,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        """The SXID plan for this event's node alone, under ``incident_id``.

        The first event of an attempt persists this as is; a later event
        hands it to the disposition applier, whose branching copies these
        steps -- so they already carry the per-node GPU/SXID/partition
        mappings, the diagnostics parameters and the site remediation.
        """
        event = context.event
        decision = context.decision
        compiled = self._compile(context, incident_id)
        workflow_id = f"workflow-{uuid4()}"
        workflow = WorkflowRequest(
            request_id=workflow_id,
            incident_id=incident_id,
            runtime_profile_version=context.profile_version,
            status=compiled.status,
            official_action=decision.official_action,
            fencing_token=1,
            safety_steps=compiled.safety_steps,
            official_steps=compiled.official_steps,
            blocked_reasons=compiled.errors,
            safety_only=compiled.status is WorkflowStatus.SAFETY_PENDING,
            blocked_kind=(
                BlockedKind.NEEDS_OPERATOR
                if compiled.status is WorkflowStatus.BLOCKED
                else None
            ),
            created_at=now,
            updated_at=now,
        )
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=event.event_id,
            event_type="GPU_FAULT_GROUP",
            event_source=event.event_source,
            source_boot_id=event.source_boot_id,
            cluster_id=event.cluster_id,
            node_ids=[event.node_id],
            gpu_uuids=sorted(set(event.participating_gpu_uuids)),
            job_id=context.observation.job_id,
            attempt_id=context.observation.attempt_id,
            workload_identity_source=(
                event.workload_identity_source or "SOLE_ACTIVE_ATTEMPT_ON_NODE"
            ),
            policy_version=decision.policy_version,
            policy_source=decision.source.value,
            official_action=decision.official_action,
            effective_action=None,
            drill_id=event.drill_id,
            state=self.callbacks.incident_state_for_workflow(workflow),
            workflow_request_id=workflow_id,
            fencing_token=1,
            reasons=list(
                dict.fromkeys(
                    f"{event.node_id}: SXID {event.sxid}: {reason}"
                    for reason in decision.reasons
                )
            ),
            created_at=now,
            updated_at=now,
        )
        return incident, workflow

    def _compile(
        self,
        context: _SxidContext,
        incident_id: str,
    ) -> _CompiledSxidWorkflow:
        event = context.event
        profile = None
        errors = []
        try:
            profile = self.store.get_profile(context.profile_version)
        except NotFoundError:
            errors.append(f"runtime profile does not exist: {context.profile_version}")
        reset_operation = (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            if context.decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
            else WorkflowOperation.RESET_GPU
        )
        operations = [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            *self.builder.sxid_remediation_operations(event),
            reset_operation,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
            WorkflowOperation.RESTART_WORKLOAD,
        ]
        node_ids = [event.node_id]
        gpu_uuids = sorted(set(event.participating_gpu_uuids))
        steps, compile_errors = self.builder.compile_steps(
            operations,
            profile,
            node_ids,
            gpu_uuids,
            context.resolved_workload_ids,
        )
        errors.extend(compile_errors)
        scoped_steps = self._scope_steps(context, incident_id, steps)
        safety_steps, safety_errors = self.builder.compile_steps(
            [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.QUARANTINE,
            ],
            profile,
            node_ids,
            gpu_uuids,
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
        incident_id: str,
        steps: list[WorkflowStepSpec],
    ) -> list[WorkflowStepSpec]:
        event = context.event
        gpu_mapping = {event.node_id: sorted(set(event.participating_gpu_uuids))}
        reset_parameters = {
            "gpu_uuids_by_node": gpu_mapping,
            "fabric_partitions_by_node": (
                {event.node_id: event.fabric_partition}
                if event.fabric_partition
                else {}
            ),
            "sxids_by_node": {event.node_id: [event.sxid]},
            "fabric_partition": event.fabric_partition,
            "sxid": event.sxid,
        }
        scoped_steps = []
        for step in steps:
            parameters = step.parameters
            if step.operation is WorkflowOperation.STOP_WORKLOADS:
                parameters = {
                    **parameters,
                    "termination_initiator_incident_id": incident_id,
                }
            elif step.operation is WorkflowOperation.RESTART_WORKLOAD:
                parameters = {
                    **parameters,
                    **self._restart_parameters(context),
                }
            elif step.operation in _NODE_ACTION_OPERATIONS:
                parameters = self._node_action_parameters(
                    context,
                    step.operation,
                    {**parameters, "gpu_uuids_by_node": gpu_mapping},
                    reset_parameters,
                )
            scoped_steps.append(
                step.model_copy(
                    update={
                        "node_ids": (
                            context.allocation_nodes
                            if step.operation in _WORKLOAD_OPERATIONS
                            else [event.node_id]
                        ),
                        "parameters": parameters,
                    }
                )
            )
        return scoped_steps

    def _node_action_parameters(
        self,
        context: _SxidContext,
        operation: WorkflowOperation,
        parameters: dict[str, object],
        reset_parameters: dict[str, object],
    ) -> dict[str, object]:
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

    @staticmethod
    def _restart_parameters(context: _SxidContext) -> dict[str, object]:
        return {
            "cluster_id": context.event.cluster_id,
            "job_id": context.observation.job_id,
            "source_attempt_id": context.observation.attempt_id,
            "source_gpu_count": context.source_gpu_count,
            "restart_budget": context.observation.restart_budget,
        }
