from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable

from gpu_fault.health_recovery_contract import (
    COOLDOWN_TIMEOUT_SECONDS,
    HEALTH_CONTRACT,
    is_hung_finding,
    recovery_contract,
)
from gpu_fault.device_plugin_liveness import (
    DEVICE_PLUGIN_UNRESPONSIVE_METRICS,
    NODE_DEVICE_RESOURCE_METRICS,
    PLUGIN_UNRESPONSIVE_PARAMETER,
)
from gpu_fault.host_diagnostics import HOST_DIAGNOSTIC_PROBE, diagnostic_probe
from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.hung_task_summary import hung_task_summary
from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkloadState,
    bounded_reasons,
)
from gpu_fault.orchestration.families.identity import derived_record_id
from gpu_fault.orchestration.reboot_window import RebootOwner, absorb_finding
from gpu_fault.orchestration.workflow_builder import containment_handoff_reason
from gpu_fault.regional_compatibility import ACTIVATION_INHIBITION_VERSION
from gpu_fault.store import NotFoundError
from gpu_fault.workflow_quarantine import (
    TERMINAL_QUARANTINE_NODES,
    inherit_terminal_quarantine,
)
from gpu_fault.workload_container_scope import (
    WORKLOAD_CONTAINER_IDS,
    container_ids_by_node,
)

LOGGER = logging.getLogger(__name__)

#: ``NodeHealthPolicy._sustained_rule`` stamps every sustained host-resource
#: finding with this source; its event id is
#: ``<batch>-<metric_name>-<device|node>-<rule_id lower>`` and the rule id
#: travels in ``diagnostic_parameters["signal"]``.
HOST_RESOURCE_POLICY_SOURCE = "SITE_HOST_RESOURCE_HEALTH"
#: The official action that turns a RESET_GPU node-health finding into the full
#: GPU + NVSwitch reset (``gpu_fabric_manager.FABRIC_MANAGER_OFFICIAL_ACTION``).
FULL_FABRIC_RESET_ACTION = "RESET_ALL_GPUS_AND_NVSWITCHES"

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


def regrade_reason(finding: NodeHealthFinding) -> str:
    """The incident reason that records ``finding``'s new grade.

    One line per grade, so a replay of the batch that re-graded finds its own
    line and writes nothing (``NodeHealthIngestionService.record_regrade``).
    """

    verb = (
        "escalated to"
        if finding.severity in {Severity.CRITICAL, Severity.FATAL}
        else "re-graded"
    )
    return (
        f"{finding.metric_name or 'node health'} {verb} "
        f"{finding.severity.value.upper()}: {finding.reason}"
    )


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
    # The incident/workflow currently rebooting a node (``reboot_window``);
    # a finding on that node is its evidence, not a new plan.
    owning_reboot_workflow: Callable[[str, str], RebootOwner] | None = None


@dataclass
class HealthBuildContext:
    target_node_ids: list[str]
    diagnostic_parameters: dict
    hung_triage_requested: bool
    lowest_rank_by_node: dict[str, int]
    task_summary: dict[str, Any] = field(default_factory=dict)


class NodeHealthPlanBuilder:
    activation_inhibition_version = ACTIVATION_INHIBITION_VERSION

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
        steps, errors, containment = self._compile_steps(
            finding,
            context,
            incident,
            operations,
        )
        # A plan refused only for a missing executable owner runs the
        # containment it compiled before it blocks: SAFETY_PENDING with the
        # prefix as its safety steps, the fault family's safety-plan shape
        # (``WorkflowBuilder.containment_before_block``). The executor settles
        # it BLOCKED/SAFETY_SETTLED with the incident QUARANTINED -- the
        # documented operator hand-off (validated restore, or a close on node
        # evidence) -- instead of the compile-time block that left a CRITICAL
        # row-remap failure's node schedulable (live 2026-10-26).
        if errors and containment:
            workflow_status = WorkflowStatus.SAFETY_PENDING
            incident = incident.model_copy(
                update={
                    "reasons": bounded_reasons(
                        [
                            *incident.reasons,
                            containment_handoff_reason(
                                containment, steps, errors, operations
                            ),
                        ]
                    )
                }
            )
        elif errors:
            workflow_status = WorkflowStatus.BLOCKED
        else:
            workflow_status = WorkflowStatus.PENDING
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
            safety_steps=(
                containment if workflow_status is WorkflowStatus.SAFETY_PENDING else []
            ),
            safety_only=workflow_status is WorkflowStatus.SAFETY_PENDING,
            official_steps=steps,
            dag_enabled=context.hung_triage_requested,
            dag_revision=1 if context.hung_triage_requested else 0,
            predecessor_workflow_id=None,
            blocked_reasons=errors,
            blocked_kind=(
                BlockedKind.NEEDS_OPERATOR
                if workflow_status is WorkflowStatus.BLOCKED
                else None
            ),
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
        hung = bool(parameters.get("capture_process_state")) and is_hung_finding(
            finding
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
                hung_task_summary(self.store, finding, None, target_node_ids),
            )
        active = [
            container
            for container in observation.containers
            if container.node_id and not container.terminated
        ]
        target_node_ids = sorted({container.node_id for container in active})
        parameters["attempt_node_ids"] = target_node_ids
        parameters.update(job_id=observation.job_id, attempt_id=observation.attempt_id)
        parameters["fabric_partition_by_node"] = {
            node: next(iter(values))
            for node in target_node_ids
            if len(
                values := {
                    item.fabric_partition
                    for item in active
                    if item.node_id == node and item.fabric_partition
                }
            )
            == 1
        }
        parameters["workload_cgroup_paths_by_node"] = {
            node: sorted(
                {
                    item.cgroup_path
                    for item in active
                    if item.node_id == node and item.cgroup_path
                }
            )
            for node in target_node_ids
        }
        # A managed Pod names no cgroup (nothing sets the annotation); the node
        # finds the cgroup its runtime named after the kubelet's container id.
        container_ids = container_ids_by_node(active, target_node_ids)
        if any(container_ids.values()):
            parameters[WORKLOAD_CONTAINER_IDS] = container_ids
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
            hung_task_summary(self.store, finding, observation, target_node_ids),
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
        contract = recovery_contract(finding) or {}
        if contract.get("mode") in {"THERMAL_UNRESOLVED", "ENVIRONMENT_REVIEW"}:
            operations += [WorkflowOperation.MARK_UNSCHEDULABLE]
            if active:
                operations += [WorkflowOperation.STOP_WORKLOADS]
            operations += [
                WorkflowOperation.QUARANTINE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
                WorkflowOperation.ESCALATE_SUPPORT,
            ]
        elif contract.get("mode") == "THERMAL_COOLDOWN":
            operations += [WorkflowOperation.MARK_UNSCHEDULABLE]
            if active:
                operations += [WorkflowOperation.STOP_WORKLOADS]
            operations += [
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.RESTORE_SCHEDULING,
            ]
            if restart:
                operations += [WorkflowOperation.RESTART_WORKLOAD]
        elif action is RecoveryAction.STOP_WORKLOAD:
            if active:
                operations += [WorkflowOperation.STOP_WORKLOADS]
            operations += [
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.ESCALATE_SUPPORT,
            ]
        elif action is RecoveryAction.DRAIN:
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
            operations += [WorkflowOperation.MARK_UNSCHEDULABLE]
            if active and (
                finding.diagnostic_parameters.get("stop_on_isolation")
                or finding.metric_name
                in {
                    "ecc_dbe_volatile_total",
                    "row_remap_failure",
                    "composite:GPU_MEMORY_DEGRADATION",
                }
            ):
                operations += [WorkflowOperation.STOP_WORKLOADS]
            operations += [WorkflowOperation.QUARANTINE]
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
            operations += self._reset_operations(
                active,
                restart,
                fabric=finding.official_action == FULL_FABRIC_RESET_ACTION,
            )
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
        contract = recovery_contract(finding) or {}
        if contract.get("mode") == "OBSERVABILITY":
            return [
                WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
                WorkflowOperation.VALIDATE_HOST,
            ]
        if hung:
            return [
                WorkflowOperation.COLLECT_HUNG_SUMMARY,
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.VALIDATE_FABRIC,
                *(
                    [WorkflowOperation.ESCALATE_SUPPORT]
                    if contract.get("mode") == "HUNG_DIAGNOSTIC"
                    else []
                ),
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
            if contract.get("mode") != "DIAGNOSTIC_ONLY":
                operations.append(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)
            operations.append(WorkflowOperation.VALIDATE_GPU)
        else:
            operations.append(WorkflowOperation.VALIDATE_HOST)
        if finding.policy_source == "SITE_APPLICATION_LOG":
            operations.append(WorkflowOperation.ESCALATE_SUPPORT)
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
        *,
        fabric: bool = False,
    ) -> list[WorkflowOperation]:
        """The GPU reset chain; ``fabric`` is the full GPU + NVSwitch reset.

        A finding whose official action is the full fabric reset (the node's
        NVLink fabric never trained, ``gpu_fabric_manager``) gets the chain a
        fatal trunk SXID gets: one reset of every GPU and NVSwitch inside the
        quiesce window, which stops Fabric Manager first and starts it after,
        then both validations.
        """

        operations = [WorkflowOperation.MARK_UNSCHEDULABLE]
        if active:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations += [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            if fabric
            else WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.VALIDATE_GPU,
            *([WorkflowOperation.VALIDATE_FABRIC] if fabric else []),
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
    ) -> tuple[list, list[str], list[WorkflowStepSpec]]:
        """``(steps, errors, containment)``: the compiled official steps, why
        the plan cannot run, and the containment prefix it runs first when the
        only reason is a missing executable owner
        (``WorkflowBuilder.containment_before_block``; ``[]`` otherwise, and
        always for a hung-triage DAG, whose steps the DAG rewrite owns)."""

        profile, errors = self._profile(finding)
        if (
            finding.recommended_action is RecoveryAction.RESET_GPU
            and not finding.gpu_uuids
        ):
            errors.append("RESET_GPU requires an explicit GPU UUID")
        if (recovery_contract(finding) or {}).get("mode") == "THERMAL_COOLDOWN":
            if not finding.gpu_uuids:
                errors.append("thermal readmission requires explicit GPU UUIDs")
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
        containment = (
            []
            if context.hung_triage_requested
            else self.workflow_builder.containment_before_block(
                operations, profile, steps, errors
            )
        )
        return steps, errors, containment

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
        elif (
            step.operation is WorkflowOperation.QUARANTINE
            and finding.recommended_action
            in {RecoveryAction.QUARANTINE, RecoveryAction.DRAIN}
        ):
            parameters = {
                **parameters,
                TERMINAL_QUARANTINE_NODES: list(step.node_ids),
            }
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
        elif step.operation is WorkflowOperation.REPLACE_NODE:
            if diagnostic.get("replacement_strategy") == "HEALTHY_WARM_SPARE_ONLY":
                parameters = {
                    "replacement_strategy": diagnostic["replacement_strategy"]
                }
            if "activation_forbidden" in diagnostic:
                parameters = {
                    **parameters,
                    "activation_forbidden": diagnostic["activation_forbidden"],
                }
        if (
            step.operation
            in {
                WorkflowOperation.VALIDATE_HOST,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_FABRIC,
            }
            and (probe := diagnostic_probe(finding)) is not None
        ):
            parameters = {**parameters, HOST_DIAGNOSTIC_PROBE: probe}
        updates = {}
        if (contract := recovery_contract(finding)) is not None:
            parameters = {**parameters, HEALTH_CONTRACT: contract}
            if (
                contract["mode"] == "THERMAL_COOLDOWN"
                and step.operation is WorkflowOperation.VALIDATE_GPU
            ):
                updates["timeout_seconds"] = COOLDOWN_TIMEOUT_SECONDS
        return step.model_copy(update={"parameters": parameters, **updates})

    def _hung_triage_dag(
        self,
        steps: list,
        finding: NodeHealthFinding,
        context: HealthBuildContext,
    ) -> tuple[list, list[str]]:
        from gpu_fault.orchestration.hung_dag import hung_dag

        return hung_dag(
            steps, finding, context, self.callbacks.sample_hung_triage_nodes
        )

    @staticmethod
    def _resource_escalation(
        steps: list,
        finding: NodeHealthFinding,
    ) -> list:
        if finding.metric_name not in NODE_DEVICE_RESOURCE_METRICS:
            return steps
        # A plugin the node's socket probe found hung still advertises every
        # device, so the restart step must be told not to read the full
        # allocatable as "already healthy" (node_operations).
        unresponsive = finding.metric_name in DEVICE_PLUGIN_UNRESPONSIVE_METRICS
        reboot_allowed = finding.metric_name != "efa_inventory_mismatch" or (
            finding.diagnostic_parameters.get("failure_mode")
            in {"PCI_DEVICE_MISSING", "DRIVER_UNBOUND", "LINK_INACTIVE"}
        )
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
                parameters["expected_count"] = expected
                if unresponsive:
                    parameters[PLUGIN_UNRESPONSIVE_PARAMETER] = True
                if reboot_allowed:
                    parameters["failure_escalation_action"] = (
                        RecoveryAction.REBOOT_NODE.value
                    )
            if reboot_allowed and step.operation in {
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

    def record_regrade(self, finding: NodeHealthFinding) -> FaultIncident | None:
        """The incident of ``finding``'s event with its new grade recorded, or
        ``None`` when the event has no incident yet (the caller ingests it).

        A GPU composite re-graded within its episode keeps its finding id and
        with it the event id this incident was minted (or absorbed) under
        (``gpu_composites.composite_episode_identity``): the grade lands on
        that incident as one bounded reason -- the automatic action is
        unchanged, so its plan stands and nothing is scheduled -- and the
        incident keeps its state, whichever it is: an operator who closed it
        while the signal held reads the escalation there, and the next episode
        after a clear opens anew. A replay of the same grade writes nothing.
        """

        with self.lock:
            incident: FaultIncident | None = self.store.get_incident_by_event(
                finding.event_id
            )
            if incident is None:
                return None
            reason = regrade_reason(finding)
            if reason in incident.reasons:
                return incident
            updated = incident.model_copy(
                update={
                    "reasons": bounded_reasons([*incident.reasons, reason]),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            # Compare-and-set on the copy read above (ARCH-D1), like the
            # host-resource absorption: a concurrent state change raises
            # ``StaleWriteError`` out of the ingest and the batch is re-posted.
            self.store.save_incident(
                updated, expected=incident, extra_event_ids=[finding.event_id]
            )
            LOGGER.info(
                "recorded %s %s on incident %s (%s) for %s/%s",
                finding.metric_name,
                finding.severity.value,
                incident.incident_id,
                incident.state.value,
                finding.node_id,
                finding.device or "node",
            )
            return updated

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
        if self.callbacks.owning_reboot_workflow is not None:
            owner = self.callbacks.owning_reboot_workflow(
                finding.cluster_id, finding.node_id
            )
            if owner is not None:
                return absorb_finding(self.store, finding, owner, persist=persist)
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
                "absorbed transient %s finding %s into RUNNING workflow %s",
                finding.metric_name,
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
        if finalized_workflow.status is WorkflowStatus.SAFETY_PENDING:
            # The containment prefix runs first; the incident follows the
            # fault family's shape (``_incident_state_for_workflow``) and
            # settles QUARANTINED with the record (``terminal_incident_state``).
            state = IncidentState.SAFETY_PENDING
        elif errors:
            state = IncidentState.ESCALATED
        else:
            state = IncidentState.ACTION_PENDING
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
        workflow = inherit_terminal_quarantine(incumbent, workflow).model_copy(
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
