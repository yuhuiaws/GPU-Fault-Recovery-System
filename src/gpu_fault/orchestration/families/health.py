from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Callable

from gpu_fault.host_health import NodeHealthFinding
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
)
from gpu_fault.orchestration.families.identity import derived_record_id
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)

#: ``NodeHealthPolicy._sustained_rule`` stamps every sustained host-resource
#: finding with this source; its event id is
#: ``<batch>-<metric_name>-<device|node>-<rule_id lower>`` and the rule id
#: travels in ``diagnostic_parameters["signal"]``.
HOST_RESOURCE_POLICY_SOURCE = "SITE_HOST_RESOURCE_HEALTH"

#: An incident in one of these states has not been settled: ACTION_PENDING
#: means its diagnostic is queued or running, ESCALATED means the diagnostic
#: ended and an operator owns the close. RECOVERED is an operator's close
#: and QUARANTINED a hand-off; a finding after either rightly opens anew.
UNSETTLED_INCIDENT_STATES = frozenset(
    {IncidentState.ACTION_PENDING, IncidentState.ESCALATED}
)


def host_resource_signal(finding: NodeHealthFinding) -> tuple[str, str] | None:
    """``(metric_name, rule_id)`` of a sustained host-resource finding, or
    ``None`` for any other finding."""

    if finding.policy_source != HOST_RESOURCE_POLICY_SOURCE:
        return None
    rule_id = finding.diagnostic_parameters.get("signal")
    if not finding.metric_name or not isinstance(rule_id, str) or not rule_id:
        return None
    return finding.metric_name, rule_id


def incident_opened_by_host_resource_signal(
    incident: FaultIncident, metric_name: str, rule_id: str
) -> bool:
    """Whether ``incident`` was minted by a sustained host-resource finding of
    ``metric_name`` + ``rule_id``, on any device.

    The incident does not carry the signal; its ``event_id`` is the finding's,
    ``<batch>-<metric_name>-<device|node>-<rule_id lower>``. Batch ids and
    devices may themselves contain hyphens, so the match is "ends with the
    rule" and "names the metric before it" rather than a positional split.
    """

    if incident.policy_source != HOST_RESOURCE_POLICY_SOURCE:
        return False
    suffix = f"-{rule_id.lower()}"
    if not incident.event_id.endswith(suffix):
        return False
    return f"-{metric_name}-" in incident.event_id[: -len(suffix)]


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
    # Counts a sustained host-resource finding recorded on an unsettled
    # same-signal incident instead of minting its own (record-only
    # accounting, ``gpu_fault_workflow_merge_record_only_total``).
    record_host_resource_absorb: Callable[[], None]


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
            # Derived from the event, like the incident's ``inc-<event_id>``:
            # a rebuild after a dangling link re-creates the same record
            # instead of a second one (F-B7).
            request_id=workflow_request_id
            or derived_record_id("workflow", "node-health", finding.event_id),
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
            blocked_kind=(BlockedKind.NEEDS_OPERATOR if errors else None),
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
            # DRAIN findings are RMA-class (row-remap failure, DBE totals,
            # NVLink error counters): NVIDIA's guidance is drain and replace,
            # so there is deliberately no RESTORE_SCHEDULING and the node
            # stays cordoned and tainted. The hand-off has to be explicit:
            # ESCALATE_SUPPORT drafts the vendor ticket and ends the incident
            # ESCALATED, the one state an operator may close by hand once
            # the RMA is done. Without it the workflow ended SUCCEEDED with
            # the node held forever and nobody told (logic item 8).
            operations += [
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.ESCALATE_SUPPORT,
            ]
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
        # The finding's workload state reaches the shared gate: a plan that
        # mutates the node is refused while that state is UNKNOWN, the same
        # way the fault family refuses it (ARCH-B2). ``_operations`` only ever
        # asked "is it ACTIVE?" to insert STOP_WORKLOADS.
        steps, compile_errors = self.workflow_builder.compile_steps(
            operations,
            profile,
            context.target_node_ids,
            finding.gpu_uuids,
            finding.affected_workload_ids,
            workload_state=finding.workload_state,
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
                incident.incident_id,
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
        incident_id: str,
    ):
        parameters = step.parameters
        if step.operation is WorkflowOperation.RESTART_WORKLOAD:
            parameters = restart
        elif step.operation is WorkflowOperation.STOP_WORKLOADS:
            # The completion watcher tells a controller-initiated stop from a
            # user stop by this marker; without it the job the system stopped
            # to repair the node came back as a failed job (F-C9).
            parameters = {
                **parameters,
                "termination_initiator_incident_id": incident_id,
            }
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
                persist=persist,
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
        *,
        persist: bool = True,
    ) -> tuple[FaultIncident, WorkflowRequest | None] | None:
        existing = self.store.get_incident_by_event(finding.event_id)
        if existing is not None:
            if not existing.workflow_request_id:
                return existing, None
            workflow = self._workflow_if_present(existing.workflow_request_id)
            if workflow is not None:
                return existing, workflow
            # The link says "already handled" but the workflow it was handled
            # by is gone. Raising here made every re-post of the event fail
            # the same way (P2-51H); building again is the repair -- the
            # incident id is derived from the event, so the same record is
            # rewritten with a workflow pointer that resolves.
            LOGGER.warning(
                "incident %s for re-posted event %s points at workflow %s which "
                "is missing; rebuilding the workflow",
                existing.incident_id,
                finding.event_id,
                existing.workflow_request_id,
            )
        absorbed = self._absorb_into_unsettled_host_resource_incident(
            finding, persist=persist
        )
        if absorbed is not None:
            return absorbed
        covered = self.callbacks.active_workflow_covers_inventory_finding(finding)
        if covered is not None:
            incident, workflow = covered
            if not persist:
                # A trial run (the attempt-group candidate, C-12) reports the
                # covering incumbent and writes nothing.
                return incident, workflow
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

    def _workflow_if_present(self, request_id: str) -> WorkflowRequest | None:
        try:
            workflow: WorkflowRequest = self.store.get_workflow(request_id)
        except NotFoundError:
            return None
        return workflow

    def _absorb_into_unsettled_host_resource_incident(
        self,
        finding: NodeHealthFinding,
        *,
        persist: bool,
    ) -> tuple[FaultIncident, WorkflowRequest | None] | None:
        """Record a sustained host-resource finding on the node's unsettled
        same-signal incident instead of minting another.

        The signal is tracked per GPU device on purpose (several jobs may
        share a node), so one activation of an 8-GPU node yields eight
        findings and every re-arm yields eight more. Each used to mint its
        own ``inc-<event_id>`` plus a RUN_DIAGNOSTICS workflow; once the
        diagnostic failed the incident parked ESCALATED and the next finding
        minted the next one -- 42 ESCALATED incidents on one node in a day,
        each running ``dcgmi diag`` on it. While an incident for the same
        cluster + node + metric + rule is ACTION_PENDING (diagnostic queued or
        running) or ESCALATED (operator owns the close), a further finding
        is linked to it with one bounded reason and nothing is scheduled.
        """

        signal = host_resource_signal(finding)
        if signal is None:
            return None
        metric_name, rule_id = signal
        incident = next(
            (
                candidate
                for candidate in self.store.list_incidents_by_state(
                    finding.cluster_id,
                    UNSETTLED_INCIDENT_STATES,
                    node_ids={finding.node_id},
                )
                if incident_opened_by_host_resource_signal(
                    candidate, metric_name, rule_id
                )
            ),
            None,
        )
        if incident is None:
            return None
        workflow = (
            self._workflow_if_present(incident.workflow_request_id)
            if incident.workflow_request_id
            else None
        )
        if not persist:
            return incident, workflow
        disposition = (
            "incident awaits operator"
            if incident.state is IncidentState.ESCALATED
            else "diagnostic already in flight"
        )
        reason = (
            f"absorbed {rule_id} finding {finding.event_id} on device "
            f"{finding.device or 'node'}: recorded only, {disposition}"
        )
        updated = incident.model_copy(
            update={
                "reasons": bounded_reasons([*incident.reasons, reason]),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        # Compare-and-set on the copy read above (ARCH-D1): a concurrent
        # state change raises ``StaleWriteError`` out of the ingest and the
        # data plane re-posts the batch against a fresh read.
        self.store.save_incident(
            updated, expected=incident, extra_event_ids=[finding.event_id]
        )
        self.callbacks.record_host_resource_absorb()
        LOGGER.info(
            "absorbed %s finding %s on %s/%s into unsettled incident %s (%s)",
            rule_id,
            finding.event_id,
            finding.node_id,
            finding.device or "node",
            incident.incident_id,
            incident.state.value,
        )
        return updated, workflow

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
