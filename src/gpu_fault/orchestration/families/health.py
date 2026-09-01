from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Callable
from uuid import uuid4

from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.store import NotFoundError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeHealthCallbacks:
    active_node_exclusive_workflow: Callable
    active_workflow_covers_inventory_finding: Callable
    attempt_observation: Callable
    claims_node_exclusively: Callable
    ingest_grouped_health_finding: Callable
    ingest_grouped_node_replacement: Callable
    ingest_grouped_node_resource_finding: Callable
    ingest_terminal_node_quarantine: Callable
    inventory_validation_parameters: Callable
    node_group_key: Callable
    preemption_scope_matches: Callable
    prepare_preempting_successor: Callable
    sample_hung_triage_nodes: Callable


@dataclass
class HealthBuildContext:
    target_node_ids: list[str]
    diagnostic_parameters: dict
    hung_triage_requested: bool
    lowest_rank_by_node: dict[str, int]


class NodeHealthPlanBuilder:
    def __init__(
        self,
        store,
        workflow_builder,
        callbacks: NodeHealthCallbacks,
    ) -> None:
        self.store = store
        self.workflow_builder = workflow_builder
        self.callbacks = callbacks

    def build(
        self,
        finding: NodeHealthFinding,
        workflow_request_id: str | None,
    ) -> tuple[
        FaultIncident,
        WorkflowRequest,
        list[str],
        list[str],
    ]:
        now = datetime.now(timezone.utc)
        context = self._context(finding)
        incident = self._incident(
            finding,
            context.target_node_ids,
            now,
        )
        operations = self._operations(
            finding,
            context.hung_triage_requested,
        )
        steps, errors = self._compile_steps(
            finding,
            context,
            incident,
            operations,
        )
        workflow_status = WorkflowStatus.BLOCKED if errors else WorkflowStatus.PENDING
        workflow = WorkflowRequest(
            request_id=workflow_request_id or f"workflow-{uuid4()}",
            incident_id=incident.incident_id,
            runtime_profile_version=finding.runtime_profile_version,
            status=workflow_status,
            official_action=(
                finding.official_action or finding.recommended_action.value
            ),
            fencing_token=incident.fencing_token,
            official_steps=steps,
            dag_enabled=context.hung_triage_requested,
            dag_revision=1 if context.hung_triage_requested else 0,
            predecessor_workflow_id=None,
            blocked_reasons=errors,
            created_at=now,
            updated_at=now,
        )
        return incident, workflow, context.target_node_ids, errors

    def _context(
        self,
        finding: NodeHealthFinding,
    ) -> HealthBuildContext:
        target_node_ids = [finding.node_id]
        parameters = dict(finding.diagnostic_parameters)
        hung = parameters.get(
            "diagnostic_reason"
        ) == "EFA_TRAFFIC_HUNG_SUSPECTED" and bool(
            parameters.get("capture_process_state")
        )
        lowest_rank_by_node: dict[str, int] = {}
        if not hung:
            return HealthBuildContext(
                target_node_ids,
                parameters,
                False,
                lowest_rank_by_node,
            )
        observation = self.callbacks.attempt_observation(finding)
        if observation is None:
            snapshot = parameters.get("attempt_node_ids")
            if (
                isinstance(snapshot, list)
                and snapshot
                and all(isinstance(node_id, str) and node_id for node_id in snapshot)
                and finding.node_id in snapshot
            ):
                target_node_ids = sorted(set(snapshot))
            return HealthBuildContext(
                target_node_ids,
                parameters,
                True,
                lowest_rank_by_node,
            )
        active = [
            container
            for container in observation.containers
            if container.node_id and not container.terminated
        ]
        target_node_ids = sorted({container.node_id for container in active})
        parameters["attempt_node_ids"] = target_node_ids
        self._add_gpu_and_rank_context(
            active,
            target_node_ids,
            parameters,
            lowest_rank_by_node,
        )
        return HealthBuildContext(
            target_node_ids,
            parameters,
            True,
            lowest_rank_by_node,
        )

    @staticmethod
    def _add_gpu_and_rank_context(
        containers,
        node_ids: list[str],
        parameters: dict,
        lowest_rank_by_node: dict[str, int],
    ) -> None:
        gpu_uuids_by_node = {
            node_id: sorted(
                {
                    gpu_uuid
                    for container in containers
                    if container.node_id == node_id
                    for gpu_uuid in container.gpu_uuids
                }
            )
            for node_id in node_ids
        }
        for container in containers:
            if container.rank is None:
                continue
            previous = lowest_rank_by_node.get(container.node_id)
            if previous is None or container.rank < previous:
                lowest_rank_by_node[container.node_id] = container.rank
        rank_by_pid_by_node = {
            node_id: {
                str(container.host_pid): container.rank
                for container in containers
                if container.node_id == node_id and container.host_pid is not None
            }
            for node_id in node_ids
        }
        if any(rank_by_pid_by_node.values()):
            parameters["rank_by_pid_by_node"] = rank_by_pid_by_node
        if node_ids and all(gpu_uuids_by_node[node_id] for node_id in node_ids):
            parameters["gpu_uuids_by_node"] = gpu_uuids_by_node

    @staticmethod
    def _incident(
        finding: NodeHealthFinding,
        target_node_ids: list[str],
        now: datetime,
    ) -> FaultIncident:
        return FaultIncident(
            incident_id=f"inc-{finding.event_id}",
            event_id=finding.event_id,
            event_type="NODE_HEALTH",
            cluster_id=finding.cluster_id,
            node_ids=target_node_ids,
            gpu_uuids=finding.gpu_uuids,
            job_id=finding.job_id,
            attempt_id=finding.attempt_id,
            policy_version=finding.policy_version,
            policy_source=finding.policy_source,
            policy_reference=finding.policy_reference,
            official_action=finding.official_action,
            effective_action=finding.recommended_action,
            drill_id=finding.drill_id,
            state=IncidentState.DETECTED,
            reasons=[finding.reason],
            created_at=now,
            updated_at=now,
        )

    def _operations(
        self,
        finding: NodeHealthFinding,
        hung_triage_requested: bool,
    ) -> list[WorkflowOperation]:
        operations = [WorkflowOperation.FREEZE_EVIDENCE]
        action = finding.recommended_action
        active = finding.workload_state is WorkloadState.ACTIVE
        restart = bool(finding.affected_workload_ids)
        if action is RecoveryAction.DRAIN:
            operations += [WorkflowOperation.MARK_UNSCHEDULABLE]
            if active:
                operations += [WorkflowOperation.STOP_WORKLOADS]
            operations += [
                WorkflowOperation.QUARANTINE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            ]
            if finding.official_action == "RUN_FIELD_DIAGNOSTIC_FOR_RMA":
                operations += [WorkflowOperation.RUN_FIELD_DIAGNOSTIC]
            operations += [WorkflowOperation.VALIDATE_GPU]
        elif action is RecoveryAction.QUARANTINE:
            operations += [
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.QUARANTINE,
            ]
        elif action is RecoveryAction.REMEDIATE_EFA_DRIVER:
            operations += [WorkflowOperation.MARK_UNSCHEDULABLE]
            if active:
                operations += [WorkflowOperation.STOP_WORKLOADS]
            operations += [
                WorkflowOperation.REMEDIATE_EFA_DRIVER,
                WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
                WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
                WorkflowOperation.VALIDATE_FABRIC,
                WorkflowOperation.RESTORE_SCHEDULING,
            ]
            if restart:
                operations += [WorkflowOperation.RESTART_WORKLOAD]
        elif action in {
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
        }:
            operations += self._plugin_operations(action)
        elif action is RecoveryAction.RUN_DIAGNOSTICS:
            operations += self._diagnostic_operations(
                finding,
                hung_triage_requested,
            )
        elif action is RecoveryAction.REPLACE_NODE:
            operations += self._replacement_operations(active, restart)
        elif action is RecoveryAction.RESET_GPU:
            operations += self._reset_operations(active, restart)
        elif action is RecoveryAction.REBOOT_NODE:
            operations += self._reboot_operations(active, restart)
        return operations

    @staticmethod
    def _plugin_operations(
        action: RecoveryAction,
    ) -> list[WorkflowOperation]:
        plugin = (
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN
            if action is RecoveryAction.RESTART_EFA_DEVICE_PLUGIN
            else WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN
        )
        validate = (
            WorkflowOperation.VALIDATE_FABRIC
            if action is RecoveryAction.RESTART_EFA_DEVICE_PLUGIN
            else WorkflowOperation.VALIDATE_GPU
        )
        return [
            plugin,
            WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
            validate,
        ]

    @staticmethod
    def _diagnostic_operations(
        finding: NodeHealthFinding,
        hung: bool,
    ) -> list[WorkflowOperation]:
        if hung:
            return [
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.VALIDATE_FABRIC,
            ]
        operations = []
        if finding.diagnostic_parameters.get("capture_process_state"):
            operations.append(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE)
        if finding.category.value in {
            "NETWORK",
            "RDMA",
            "NCCL",
            "TRAINING",
        }:
            operations.append(WorkflowOperation.VALIDATE_FABRIC)
        elif finding.category.value == "GPU":
            operations.extend(
                [
                    WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
                    WorkflowOperation.VALIDATE_GPU,
                ]
            )
        else:
            operations.append(WorkflowOperation.VALIDATE_HOST)
        return operations

    @staticmethod
    def _replacement_operations(
        active: bool,
        restart: bool,
    ) -> list[WorkflowOperation]:
        operations = [
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
        ]
        if active:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations += [
            WorkflowOperation.REPLACE_NODE,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
        ]
        if restart:
            operations.append(WorkflowOperation.RESTART_WORKLOAD)
        return operations

    @staticmethod
    def _reset_operations(
        active: bool,
        restart: bool,
    ) -> list[WorkflowOperation]:
        operations = [WorkflowOperation.MARK_UNSCHEDULABLE]
        if active:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations += [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.RESTORE_SCHEDULING,
        ]
        if restart:
            operations.append(WorkflowOperation.RESTART_WORKLOAD)
        return operations

    @staticmethod
    def _reboot_operations(
        active: bool,
        restart: bool,
    ) -> list[WorkflowOperation]:
        operations = [
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
        ]
        if active:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations += [
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
        ]
        if restart:
            operations.append(WorkflowOperation.RESTART_WORKLOAD)
        return operations

    def _compile_steps(
        self,
        finding: NodeHealthFinding,
        context: HealthBuildContext,
        incident: FaultIncident,
        operations: list[WorkflowOperation],
    ) -> tuple[list, list[str]]:
        profile, errors = self._profile(finding)
        if (
            finding.recommended_action is RecoveryAction.RESET_GPU
            and not finding.gpu_uuids
        ):
            errors.append("RESET_GPU requires an explicit GPU UUID")
        steps, compile_errors = self.workflow_builder.compile_steps(
            operations,
            profile,
            context.target_node_ids,
            finding.gpu_uuids,
            finding.affected_workload_ids,
        )
        restart = self.workflow_builder.restart_step_parameters(
            finding.cluster_id,
            finding.affected_workload_ids,
            job_id=finding.job_id,
            fallback_attempt_id=finding.attempt_id or finding.event_id,
            observed_at=finding.observed_at,
            node_id=finding.node_id,
            fallback_gpu_uuids=finding.gpu_uuids,
        )
        inventory = self.callbacks.inventory_validation_parameters(finding)
        steps = [
            self._enrich_step(
                step,
                finding,
                context.diagnostic_parameters,
                restart,
                inventory,
            )
            for step in steps
        ]
        if context.hung_triage_requested:
            steps, triage_errors = self._hung_triage_dag(
                steps,
                finding,
                context,
            )
            errors.extend(triage_errors)
        steps = self._resource_escalation(steps, finding)
        errors.extend(compile_errors)
        return steps, errors

    def _profile(
        self,
        finding: NodeHealthFinding,
    ):
        if not finding.runtime_profile_version:
            return None, ["runtime_profile_version is required for execution"]
        try:
            return (
                self.store.get_profile(finding.runtime_profile_version),
                [],
            )
        except NotFoundError:
            return None, [
                "runtime profile does not exist: " + finding.runtime_profile_version
            ]

    @staticmethod
    def _enrich_step(
        step,
        finding: NodeHealthFinding,
        diagnostic: dict,
        restart: dict,
        inventory: dict,
    ):
        parameters = step.parameters
        if step.operation is WorkflowOperation.RESTART_WORKLOAD:
            parameters = restart
        elif step.operation is WorkflowOperation.RUN_FIELD_DIAGNOSTIC:
            parameters = {
                "procedure": "NVIDIA_FIELD_DIAGNOSTIC",
                "pci_bdf": finding.pci_bdf,
            }
        elif (
            step.operation
            in {
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            }
            and diagnostic
        ):
            parameters = {**parameters, **diagnostic}
        elif (
            step.operation
            in {
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_FABRIC,
            }
            and inventory
        ):
            parameters = {**parameters, **inventory}
        elif (
            step.operation is WorkflowOperation.REPLACE_NODE
            and diagnostic.get("replacement_strategy") == "HEALTHY_WARM_SPARE_ONLY"
        ):
            parameters = {"replacement_strategy": diagnostic["replacement_strategy"]}
        return step.model_copy(update={"parameters": parameters})

    def _hung_triage_dag(
        self,
        steps: list,
        finding: NodeHealthFinding,
        context: HealthBuildContext,
    ) -> tuple[list, list[str]]:
        by_operation = {step.operation: index for index, step in enumerate(steps)}
        required = {
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.COLLECT_HUNG_TRIAGE,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowOperation.VALIDATE_FABRIC,
        }
        if not required <= by_operation.keys():
            return steps, ["NCCL hung triage DAG is missing required steps"]
        freeze = by_operation[WorkflowOperation.FREEZE_EVIDENCE]
        triage = by_operation[WorkflowOperation.COLLECT_HUNG_TRIAGE]
        bundle = by_operation[WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE]
        validate = by_operation[WorkflowOperation.VALIDATE_FABRIC]
        updated = list(steps)
        updated[freeze] = updated[freeze].model_copy(
            update={"depends_on_step_indexes": []}
        )
        targets, skipped = self.callbacks.sample_hung_triage_nodes(
            context.target_node_ids,
            reporting_node_id=finding.node_id,
            lowest_rank_by_node=context.lowest_rank_by_node,
        )
        parameters = {
            **updated[triage].parameters,
            **context.diagnostic_parameters,
            "triage_timeout_seconds": 10,
            "expand_python_cgroup_processes": False,
        }
        if skipped:
            parameters["not_sampled_nodes"] = skipped
            LOGGER.info(
                "hung triage for %s samples %d of %d attempt nodes; skipped %s",
                finding.event_id,
                len(targets),
                len(context.target_node_ids),
                ",".join(skipped),
            )
        updated[triage] = updated[triage].model_copy(
            update={
                "depends_on_step_indexes": [freeze],
                "node_ids": targets,
                "parameters": parameters,
            }
        )
        updated[bundle] = updated[bundle].model_copy(
            update={
                "depends_on_step_indexes": [triage],
                "parameters": {
                    **updated[bundle].parameters,
                    **context.diagnostic_parameters,
                    "capture_process_state": False,
                    "hung_triage_target_pending": True,
                },
            }
        )
        updated[validate] = updated[validate].model_copy(
            update={"depends_on_step_indexes": [triage]}
        )
        return updated, []

    @staticmethod
    def _resource_escalation(
        steps: list,
        finding: NodeHealthFinding,
    ) -> list:
        if finding.metric_name not in {
            "efa_inventory_mismatch",
            "efa_kubernetes_allocatable_mismatch",
            "gpu_kubernetes_allocatable_mismatch",
        }:
            return steps
        raw_expected = finding.diagnostic_parameters.get(
            "expected_count",
            finding.diagnostic_parameters.get(
                "expected_efa_device_count",
                1,
            ),
        )
        try:
            expected = int(raw_expected)
        except (TypeError, ValueError):
            expected = 1
        updated = []
        for step in steps:
            parameters = dict(step.parameters)
            if step.operation in {
                WorkflowOperation.REMEDIATE_EFA_DRIVER,
                WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
                WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
            }:
                parameters.update(
                    {
                        "expected_count": expected,
                        "failure_escalation_action": (RecoveryAction.REBOOT_NODE.value),
                    }
                )
            if step.operation in {
                WorkflowOperation.VALIDATE_FABRIC,
                WorkflowOperation.VALIDATE_GPU,
            }:
                parameters["failure_escalation_action"] = (
                    RecoveryAction.REBOOT_NODE.value
                )
            updated.append(step.model_copy(update={"parameters": parameters}))
        return updated


class NodeHealthIngestionService:
    def __init__(
        self,
        store,
        lock: RLock,
        plan_builder: NodeHealthPlanBuilder,
        callbacks: NodeHealthCallbacks,
    ) -> None:
        self.store = store
        self.lock = lock
        self.plan_builder = plan_builder
        self.callbacks = callbacks

    def ingest(
        self,
        finding: NodeHealthFinding,
        *,
        workflow_request_id: str | None = None,
        skip_attempt_grouping: bool = False,
        skip_node_resource_merge: bool = False,
        skip_terminal_quarantine_merge: bool = False,
        persist: bool = True,
    ) -> tuple[FaultIncident, WorkflowRequest | None]:
        with self.lock:
            routed = self._existing_or_grouped(
                finding,
                skip_attempt_grouping,
                skip_node_resource_merge,
                skip_terminal_quarantine_merge,
            )
            if routed is not None:
                return routed
            incident, workflow, target_nodes, errors = self.plan_builder.build(
                finding,
                workflow_request_id,
            )

            def finalize() -> tuple[
                FaultIncident,
                WorkflowRequest,
            ]:
                return self._finalize(
                    finding,
                    incident,
                    workflow,
                    target_nodes,
                    errors,
                    persist,
                )

            if not persist:
                return finalize()
            persisted_incident, persisted_workflow, _ = (
                self.store.create_incident_workflow_if_absent(
                    finding.event_id,
                    finalize,
                    serialization_key=self.callbacks.node_group_key(
                        finding.cluster_id,
                        finding.node_id,
                    ),
                )
            )
            return persisted_incident, persisted_workflow

    def _existing_or_grouped(
        self,
        finding: NodeHealthFinding,
        skip_attempt: bool,
        skip_resource: bool,
        skip_terminal: bool,
    ) -> tuple[FaultIncident, WorkflowRequest | None] | None:
        existing = self.store.get_incident_by_event(finding.event_id)
        if existing is not None:
            workflow = (
                self.store.get_workflow(existing.workflow_request_id)
                if existing.workflow_request_id
                else None
            )
            return existing, workflow
        covered = self.callbacks.active_workflow_covers_inventory_finding(finding)
        if covered is not None:
            incident, workflow = covered
            self.store.link_event_to_incident(
                finding.event_id,
                incident.incident_id,
            )
            LOGGER.info(
                "absorbed transient GPU inventory finding %s into RUNNING workflow %s",
                finding.event_id,
                workflow.request_id,
            )
            return incident, workflow
        routes = [
            (
                not skip_attempt,
                self.callbacks.ingest_grouped_health_finding,
            ),
            (
                not skip_resource,
                self.callbacks.ingest_grouped_node_resource_finding,
            ),
            (
                not skip_terminal,
                self.callbacks.ingest_terminal_node_quarantine,
            ),
            (
                not skip_attempt,
                self.callbacks.ingest_grouped_node_replacement,
            ),
        ]
        for enabled, route in routes:
            if not enabled:
                continue
            grouped = route(finding)
            if grouped is not None:
                return grouped
        return None

    def _finalize(
        self,
        finding: NodeHealthFinding,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        target_nodes: list[str],
        errors: list[str],
        persist: bool,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        finalized_incident = incident
        finalized_workflow = workflow
        if (
            persist
            and not errors
            and self.callbacks.claims_node_exclusively(workflow.official_steps)
        ):
            incumbent = self.callbacks.active_node_exclusive_workflow(
                finding.cluster_id,
                set(target_nodes),
                candidate_steps=workflow.official_steps,
            )
            if incumbent is not None:
                finalized_incident, finalized_workflow = (
                    self._serialize_behind_incumbent(
                        incident,
                        workflow,
                        target_nodes,
                        incumbent,
                    )
                )
        state = IncidentState.ESCALATED if errors else IncidentState.ACTION_PENDING
        return (
            finalized_incident.model_copy(
                update={
                    "state": state,
                    "workflow_request_id": (finalized_workflow.request_id),
                }
            ),
            finalized_workflow,
        )

    def _serialize_behind_incumbent(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        target_nodes: list[str],
        incumbent: WorkflowRequest,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        incumbent_incident = self.store.get_incident(incumbent.incident_id)
        overlap = sorted(set(target_nodes) & set(incumbent_incident.node_ids))
        incident = incident.model_copy(
            update={
                "reasons": [
                    *incident.reasons,
                    "serialized behind in-flight node-exclusive "
                    f"workflow {incumbent.request_id} on " + ",".join(overlap),
                ]
            }
        )
        workflow = workflow.model_copy(
            update={"predecessor_workflow_id": incumbent.request_id}
        )
        if self.callbacks.preemption_scope_matches(
            incumbent_incident,
            incident,
        ):
            workflow = self.callbacks.prepare_preempting_successor(
                incumbent,
                workflow,
            )
        return incident, workflow
