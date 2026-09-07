from __future__ import annotations

import time
from datetime import datetime, timezone

from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
    bounded_reasons,
)
from gpu_fault.notifications import HardwareEscalationEmailBuilder
from gpu_fault.operation_registry import (
    HARDWARE_ESCALATION_RELEVANT_OPERATIONS,
    NODE_ACTION_SCOPE_OPERATIONS,
)
from gpu_fault.store.shared.errors import NotFoundError


RESTART_SAFETY_PARAMETERS = (
    "cluster_id",
    "job_id",
    "source_attempt_id",
    "source_gpu_count",
    "restart_budget",
)


_RESET_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    }
)
_VALIDATION_OPERATIONS = frozenset(
    {
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
    }
)
# Containment and release steps. Their failure is not a hardware rung but it
# is not nothing either: a node whose RESTORE_SCHEDULING failed stays
# cordoned until someone acts, so the failure goes to an operator (F-H1).
_CONTAINMENT_RELEASE_OPERATIONS = frozenset(
    {
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.QUARANTINE,
    }
)
_CLASSIFIABLE_OPERATIONS = (
    HARDWARE_ESCALATION_RELEVANT_OPERATIONS
    | _CONTAINMENT_RELEASE_OPERATIONS
    | _VALIDATION_OPERATIONS
)

# Stage names the classifier hands to ``emit``. A containment or release step
# that failed for an ordinary reason is re-cordoned by the support workflow
# (F-H1); one the adapter *refused* -- the node is absent, or owned by another
# generation -- cannot be isolated by planning the same step again, so the
# support workflow carries no isolation (ARCH-E2E-2A finding 1, DESTR-020).
CONTAINMENT_STAGE = "containment_or_release"
CONTAINMENT_REFUSED_STAGE = "containment_refused"
# Written by the Kubernetes node adapter on a step it refused rather than
# failed: ``safety_rejection`` on every refusal, ``absent`` when the node
# could not be read at all.
SAFETY_REJECTION_DETAIL = "safety_rejection"
NODE_ABSENT_DETAIL = "absent"

ESCALATION_NAMES: dict[RecoveryAction, str] = {
    RecoveryAction.DRAIN: "drain",
    RecoveryAction.REBOOT_NODE: "reboot",
    RecoveryAction.REPLACE_NODE: "replace",
    RecoveryAction.ESCALATE_OPERATOR: "support",
}
_ESCALATION_EVENT_SEPARATOR = "-after-"
# Escalations that are the last rung. Their product is an operator hand-off;
# when that product fails, opening another one on the same node answers
# nothing and, driven by the failed-workflow reconcile, never stops.
CHAIN_TERMINAL_ESCALATIONS = frozenset({"support"})
ESCALATION_CHAIN_TERMINATED_REASON = "escalation chain terminated"


def escalation_origin(event_id: str) -> tuple[str, str] | None:
    """``(escalation_name, source_workflow_request_id)`` when ``event_id`` was
    minted by :meth:`HardwareEscalationService.escalate`, else ``None``.

    The event id is the field the escalation is deduplicated on
    (``create_incident_workflow_if_absent``), so it is the authoritative record
    of an incident being an escalation product; the workflow request id
    repeats it with a ``workflow-`` prefix.
    """

    name, separator, source_request_id = event_id.partition(_ESCALATION_EVENT_SEPARATOR)
    if not separator or not source_request_id:
        return None
    if name not in ESCALATION_NAMES.values():
        return None
    return name, source_request_id


def _refused_containment(
    executions: list[WorkflowStepExecution],
) -> bool:
    """Every failed containment/release execution was a safety refusal."""

    refusals = [
        execution
        for execution in executions
        if execution.status is WorkflowStepStatus.FAILED
        and execution.operation in _CONTAINMENT_RELEASE_OPERATIONS
    ]
    return bool(refusals) and all(
        execution.details.get(SAFETY_REJECTION_DETAIL) is True for execution in refusals
    )


def next_rung(
    failed_operation: WorkflowOperation,
    recovery_context: set[WorkflowOperation] | frozenset[WorkflowOperation],
) -> WorkflowOperation | None:
    """The next hardware-recovery rung after ``failed_operation`` failed.

    Mirrors the ladder ``HardwareEscalationService._classify`` applies to a
    whole workflow, for one node branch (F-N1): reset -> reboot -> warm-spare
    replacement. ``None`` means the ladder is exhausted for this branch and
    the outcome belongs to an operator (support escalation / drain), which
    the whole-workflow path produces once the workflow fails.
    ``recovery_context`` is the set of recovery operations that already ran
    on the branch; it decides what a failed *validation* escalates from.
    """

    if failed_operation in _RESET_OPERATIONS:
        return WorkflowOperation.RESTART_NODE
    if failed_operation is WorkflowOperation.RESTART_NODE:
        return WorkflowOperation.REPLACE_NODE
    if failed_operation in _VALIDATION_OPERATIONS:
        if WorkflowOperation.REPLACE_NODE in recovery_context:
            return None
        if WorkflowOperation.RESTART_NODE in recovery_context:
            return WorkflowOperation.REPLACE_NODE
        if recovery_context & _RESET_OPERATIONS:
            return WorkflowOperation.RESTART_NODE
        return None
    return None


def _escalation_operations(
    failed_stage: str,
    next_action: RecoveryAction,
    next_operation: WorkflowOperation | None,
    workload_ids: list[str],
) -> list[WorkflowOperation]:
    """The operation sequence the escalation workflow plans (see ``emit``)."""

    if failed_stage == CONTAINMENT_REFUSED_STAGE:
        # The adapter refused the isolation (absent node, foreign
        # generation): there is nothing this workflow can isolate, and
        # planning MARK_UNSCHEDULABLE / QUARANTINE again on the same node
        # only reproduces the refusal. Evidence and the hand-off remain.
        operations = [WorkflowOperation.FREEZE_EVIDENCE]
    elif next_action is RecoveryAction.DRAIN:
        operations = [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
        ]
        if workload_ids:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations.extend(
            [
                WorkflowOperation.QUARANTINE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.VALIDATE_GPU,
            ]
        )
    else:
        operations = [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
        ]
    if next_operation is WorkflowOperation.ESCALATE_SUPPORT:
        operations.append(WorkflowOperation.ESCALATE_SUPPORT)
    elif next_action is not RecoveryAction.DRAIN:
        if workload_ids:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations.extend(
            [
                next_operation,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_HOST,
                WorkflowOperation.VALIDATE_FABRIC,
                WorkflowOperation.RESTORE_SCHEDULING,
            ]
        )
        if workload_ids:
            operations.append(WorkflowOperation.RESTART_WORKLOAD)
    return operations


def _successor_lifetime(
    workflow: WorkflowRequest,
    next_action: RecoveryAction,
    next_operation: WorkflowOperation | None,
    now: datetime,
) -> datetime | None:
    """The lifetime the escalation successor starts with.

    The hard lifetime (F-N1) bounds *automatic* remediation: a reboot or a
    replacement emitted after a failed reset shares the chain's single
    clock, so an escalation ladder cannot outlive the window by re-issuing
    itself. An operator hand-off -- the support ticket, a drain -- is the
    thing the window ends *in*: inheriting an already-expired deadline
    failed the support workflow at its first claim, before
    FREEZE_EVIDENCE ran, and the node never reached an operator
    (DESTR-018). Hand-offs therefore start their own clock; ``None`` lets
    the first claim stamp the node lifetime (and, for CHECK_MECHANICALS,
    the operator-acknowledgement floor). An automatic rung emitted while
    the chain's lifetime is still ahead keeps it; one emitted after the
    lifetime passed keeps the expired value on purpose, so it fails closed
    instead of running hardware actions past the window.
    """

    operator_hand_off = (
        next_operation is WorkflowOperation.ESCALATE_SUPPORT
        or next_operation is WorkflowOperation.CHECK_MECHANICALS
        or next_action in {RecoveryAction.ESCALATE_OPERATOR, RecoveryAction.DRAIN}
    )
    if operator_hand_off:
        return None
    return workflow.lifetime_deadline_at


def _chain_termination_reason(
    workflow: WorkflowRequest,
    source: FaultIncident,
    origin: tuple[str, str],
    failed_operations: list[str],
) -> str:
    """The incident reason ``_terminate_chain`` records; also its idempotency
    key, so the text must stay stable for a given failure."""

    escalation_name, source_request_id = origin
    return (
        f"{ESCALATION_CHAIN_TERMINATED_REASON}: {escalation_name} escalation "
        f"{source.event_id} (after {source_request_id}) failed again in "
        f"{workflow.request_id} at "
        f"{', '.join(failed_operations) or 'no recorded step'}; no further "
        f"{escalation_name} workflow is opened, operator attention required"
    )


class HardwareEscalationService:
    def __init__(self, store, builder) -> None:
        self.store = store
        self.builder = builder
        # Failed support escalations that were handed to an operator instead
        # of a further support escalation (the chain bound), and the Unix
        # time of the newest one (ARCH-E E4 convention).
        self.escalation_chain_terminated_total = 0
        self.escalation_chain_terminated_last_seen_timestamp_seconds = 0.0
        # Support escalations opened for a refused containment, without
        # re-planning the isolation the adapter refused.
        self.containment_refused_escalations_total = 0

    @staticmethod
    def _active_workload_dcgm_execution_review(
        workflow: WorkflowRequest,
    ) -> bool:
        if not any(step.workload_ids for step in workflow.official_steps):
            return False
        failed = [
            execution
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.FAILED
            and execution.operation is WorkflowOperation.RUN_DCGM_DIAGNOSTIC
        ]
        if not failed:
            return False
        for execution in failed:
            results = execution.details.get("node_results")
            if not isinstance(results, dict) or not results:
                return False
            for result in results.values():
                if not isinstance(result, dict):
                    return False
                if any(
                    result.get(name)
                    for name in (
                        "diagnostic_findings",
                        "failed_checks",
                        "warning_checks",
                    )
                ):
                    return False
                recommendations = result.get("recommended_actions", [])
                if {
                    item.get("action_code")
                    for item in recommendations
                    if isinstance(item, dict)
                } != {"DCGM_EXECUTION_REVIEW"}:
                    return False
        return True

    @staticmethod
    def classify(workflow: WorkflowRequest):
        if HardwareEscalationService._active_workload_dcgm_execution_review(workflow):
            return None
        return HardwareEscalationService._classify(workflow)

    @staticmethod
    def _classify(workflow: WorkflowRequest):
        lifetime_failures = [
            execution
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.FAILED
            and execution.details.get("workflow_lifetime_exceeded") is True
        ]
        if lifetime_failures:
            # The remediation ran out of lifetime (F-N1): no further hardware
            # rung, the node belongs to an operator now.
            return (
                "lifetime_exceeded",
                RecoveryAction.ESCALATE_OPERATOR,
                WorkflowOperation.ESCALATE_SUPPORT,
                lifetime_failures,
            )
        failed_operation_set = {
            execution.operation
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.FAILED
        }
        successful_operation_set = set(workflow.completed_operations) | {
            execution.operation
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.SUCCEEDED
        }
        # What actually ran before the failure, by execution record -- not the
        # steps that merely sit earlier in the list. In a DAG a sibling branch's
        # RESTART_NODE may precede the failed step positionally without ever
        # having run on this node (F-C6).
        executed_operation_set = {
            execution.operation for execution in workflow.step_executions
        }
        recovery_context_operations = successful_operation_set | executed_operation_set
        if (
            not (recovery_context_operations - failed_operation_set)
            & HARDWARE_ESCALATION_RELEVANT_OPERATIONS
        ):
            # A record with no execution history for its recovery steps (rows
            # written before completions were recorded): fall back to the
            # steps planned ahead of the failure.
            recovery_context_operations |= {
                step.operation
                for execution in workflow.step_executions
                if (
                    execution.status is WorkflowStepStatus.FAILED
                    and 0 <= execution.step_index < len(workflow.official_steps)
                )
                for step in workflow.official_steps[: execution.step_index]
            }
        requested_escalations = {
            str(
                workflow.official_steps[execution.step_index].parameters.get(
                    "failure_escalation_action"
                )
            )
            for execution in workflow.step_executions
            if (
                execution.status is WorkflowStepStatus.FAILED
                and execution.step_index < len(workflow.official_steps)
                and workflow.official_steps[execution.step_index].parameters.get(
                    "failure_escalation_action"
                )
            )
        }
        if (
            RecoveryAction.REBOOT_NODE.value in requested_escalations
            and WorkflowOperation.RESTART_NODE not in recovery_context_operations
        ):
            failed_stage = "efa_recovery"
            next_action = RecoveryAction.REBOOT_NODE
            next_operation = WorkflowOperation.RESTART_NODE
        elif WorkflowOperation.RUN_DCGM_DIAGNOSTIC in (failed_operation_set):
            failed_stage = "temperature_diagnostic"
            next_action = RecoveryAction.DRAIN
            next_operation = None
        elif failed_operation_set.intersection(
            {
                WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
            }
        ):
            failed_stage = "nvidia_field_diagnostic"
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        elif failed_operation_set.intersection(
            {
                WorkflowOperation.REMEDIATE_DRIVER,
                WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
            }
        ):
            failed_stage = "software_or_firmware_remediation"
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        elif WorkflowOperation.REPLACE_NODE in failed_operation_set:
            failed_stage = "replacement"
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        elif WorkflowOperation.RESTART_NODE in failed_operation_set:
            failed_stage = "reboot"
            next_action = RecoveryAction.REPLACE_NODE
            next_operation = WorkflowOperation.REPLACE_NODE
        elif failed_operation_set.intersection(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            }
        ):
            failed_stage = "reset"
            next_action = RecoveryAction.REBOOT_NODE
            next_operation = WorkflowOperation.RESTART_NODE
        elif failed_operation_set.intersection(_VALIDATION_OPERATIONS):
            if WorkflowOperation.REPLACE_NODE in recovery_context_operations:
                failed_stage = "replacement_validation"
                next_action = RecoveryAction.ESCALATE_OPERATOR
                next_operation = WorkflowOperation.ESCALATE_SUPPORT
            elif WorkflowOperation.RESTART_NODE in recovery_context_operations:
                failed_stage = "reboot_validation"
                next_action = RecoveryAction.REPLACE_NODE
                next_operation = WorkflowOperation.REPLACE_NODE
            elif recovery_context_operations.intersection(
                {
                    WorkflowOperation.RESET_GPU,
                    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                }
            ):
                failed_stage = "reset_validation"
                next_action = RecoveryAction.REBOOT_NODE
                next_operation = WorkflowOperation.RESTART_NODE
            elif recovery_context_operations.intersection(
                {
                    WorkflowOperation.REMEDIATE_DRIVER,
                    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
                    WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
                    WorkflowOperation.RUN_NVLINK74_WORKFLOW,
                }
            ):
                failed_stage = "remediation_validation"
                next_action = RecoveryAction.ESCALATE_OPERATOR
                next_operation = WorkflowOperation.ESCALATE_SUPPORT
            elif WorkflowOperation.RUN_DCGM_DIAGNOSTIC in recovery_context_operations:
                failed_stage = "temperature_validation"
                next_action = RecoveryAction.DRAIN
                next_operation = None
            else:
                return None
        elif failed_operation_set.intersection(_CONTAINMENT_RELEASE_OPERATIONS):
            failed_stage = (
                CONTAINMENT_REFUSED_STAGE
                if _refused_containment(workflow.step_executions)
                else CONTAINMENT_STAGE
            )
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        else:
            return None

        failed_executions = [
            execution
            for execution in workflow.step_executions
            if (
                execution.status is WorkflowStepStatus.FAILED
                and execution.operation in _CLASSIFIABLE_OPERATIONS
            )
        ]
        if not failed_executions:
            return None

        return (
            failed_stage,
            next_action,
            next_operation,
            failed_executions,
        )

    @staticmethod
    def collect_scope(
        workflow: WorkflowRequest,
        source: FaultIncident,
        failed_executions,
    ) -> dict:
        failed_nodes: set[str] = set()
        node_failures: dict[str, list[str]] = {}
        failed_operations = set()
        for execution in failed_executions:
            failed_operations.add(execution.operation)
            step_nodes = (
                workflow.official_steps[execution.step_index].node_ids
                if execution.step_index < len(workflow.official_steps)
                else source.node_ids
            )
            reported = execution.details.get("failed_nodes")
            selected = (
                [node_id for node_id in reported if node_id in step_nodes]
                if isinstance(reported, list)
                else list(step_nodes)
            )
            if not selected:
                selected = list(step_nodes)
            failed_nodes.update(selected)
            reported_failures = execution.details.get("node_failures", {})
            for node_id in selected:
                reasons = reported_failures.get(node_id, [])
                if not isinstance(reasons, list):
                    reasons = [str(reasons)]
                node_failures.setdefault(node_id, []).extend(
                    str(reason) for reason in reasons
                )
        if not failed_nodes:
            failed_nodes.update(source.node_ids)
        ordered_failed_nodes = sorted(failed_nodes)

        workload_ids = list(
            dict.fromkeys(
                workload_id
                for step in workflow.official_steps
                for workload_id in step.workload_ids
            )
        )
        gpu_uuids_by_node: dict[str, list[str]] = {}
        for step in workflow.official_steps:
            mapping = step.parameters.get("gpu_uuids_by_node", {})
            if isinstance(mapping, dict):
                for node_id, values in mapping.items():
                    if isinstance(values, list):
                        gpu_uuids_by_node.setdefault(node_id, []).extend(
                            str(value) for value in values
                        )
        gpu_uuids = list(
            dict.fromkeys(
                gpu_uuid
                for node_id in ordered_failed_nodes
                for gpu_uuid in gpu_uuids_by_node.get(node_id, [])
            )
        )
        if not gpu_uuids:
            gpu_uuids = list(
                dict.fromkeys(
                    [
                        *source.gpu_uuids,
                        *(
                            gpu_uuid
                            for step in workflow.official_steps
                            for gpu_uuid in step.gpu_uuids
                        ),
                    ]
                )
            )
        failed = ", ".join(sorted(operation.value for operation in failed_operations))
        node_reason = "; ".join(
            (
                f"{node_id}: "
                + (
                    ",".join(sorted(set(node_failures[node_id])))
                    if node_failures.get(node_id)
                    else "validation failed"
                )
            )
            for node_id in ordered_failed_nodes
        )
        diagnostic_guidance: list[str] = []
        diagnostic_evidence: list[str] = []
        for execution in workflow.step_executions:
            if execution.operation is not WorkflowOperation.RUN_DCGM_DIAGNOSTIC:
                continue
            raw_node_results = execution.details.get("node_results", {})
            if not isinstance(raw_node_results, dict):
                continue
            for (
                node_id,
                node_result,
            ) in raw_node_results.items():
                if node_id not in ordered_failed_nodes or not isinstance(
                    node_result, dict
                ):
                    continue
                evidence_ref = node_result.get("evidence_ref")
                if isinstance(evidence_ref, str) and evidence_ref:
                    diagnostic_evidence.append(
                        f"DCGM evidence for {node_id}: {evidence_ref}"
                    )
                recommendations = node_result.get("recommended_actions", [])
                if not isinstance(recommendations, list):
                    continue
                for recommendation in recommendations:
                    if not isinstance(recommendation, dict):
                        continue
                    action_code = recommendation.get("action_code")
                    instruction = recommendation.get("instruction")
                    if not (
                        isinstance(action_code, str)
                        and action_code
                        and isinstance(instruction, str)
                        and instruction
                    ):
                        continue
                    trigger_tests = recommendation.get("trigger_tests", [])
                    tests = (
                        ", ".join(str(item) for item in trigger_tests)
                        if isinstance(trigger_tests, list)
                        else "unknown"
                    )
                    diagnostic_guidance.append(
                        f"DCGM guidance for {node_id} "
                        f"[{action_code}] ({tests}): "
                        f"{instruction}"
                    )
        return {
            "ordered_failed_nodes": ordered_failed_nodes,
            "workload_ids": workload_ids,
            "gpu_uuids": gpu_uuids,
            # Per-node GPU scope of the failed nodes: node-action steps of
            # the replacement plan carry it so a multi-node step is not
            # refused by the agent barrier (F-H1).
            "gpu_uuids_by_node": {
                node_id: list(dict.fromkeys(gpu_uuids_by_node[node_id]))
                for node_id in ordered_failed_nodes
                if gpu_uuids_by_node.get(node_id)
            },
            "failed_operations": failed_operations,
            "failed": failed,
            "node_reason": node_reason,
            "diagnostic_guidance": diagnostic_guidance,
            "diagnostic_evidence": diagnostic_evidence,
        }

    def _compile_replacement_steps(
        self,
        workflow: WorkflowRequest,
        operations: list[WorkflowOperation],
        workload_scope: list[str],
        ordered_failed_nodes: list[str],
        gpu_uuids: list[str],
        workload_ids: list[str],
        gpu_uuids_by_node: dict[str, list[str]] | None = None,
    ) -> tuple[list, list[str]]:
        profile = None
        errors = []
        if workflow.runtime_profile_version:
            try:
                profile = self.store.get_profile(workflow.runtime_profile_version)
            except NotFoundError:
                errors.append(
                    "runtime profile does not exist: "
                    + workflow.runtime_profile_version
                )
        else:
            errors.append("runtime_profile_version is required for execution")
        replacement_steps = []
        inventory_parameters_by_operation: dict[WorkflowOperation, dict] = {}
        restart_source_steps = [
            source_step
            for source_step in workflow.official_steps
            if source_step.operation is WorkflowOperation.RESTART_WORKLOAD
        ]
        restart_contexts = [
            {name: source_step.parameters[name] for name in RESTART_SAFETY_PARAMETERS}
            for source_step in restart_source_steps
            if all(name in source_step.parameters for name in RESTART_SAFETY_PARAMETERS)
        ]
        restart_context = restart_contexts[0] if restart_contexts else None
        restart_context_error = None
        if len(restart_contexts) != len(restart_source_steps):
            restart_context_error = (
                "failed workflow has no complete restart safety context"
            )
            restart_context = None
        elif any(value != restart_context for value in restart_contexts[1:]):
            restart_context_error = (
                "failed workflow has inconsistent restart safety context"
            )
            restart_context = None
        if restart_context_error is not None:
            errors.append(restart_context_error)
        for source_step in workflow.official_steps:
            requirements = source_step.parameters.get("inventory_requirements_by_node")
            if (
                source_step.operation
                in {
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowOperation.VALIDATE_FABRIC,
                }
                and isinstance(requirements, dict)
                and requirements
            ):
                inventory_parameters_by_operation[source_step.operation] = {
                    "inventory_requirements_by_node": {
                        node_id: requirement
                        for node_id, requirement in requirements.items()
                        if node_id in ordered_failed_nodes
                    }
                }
        workload_operations = {
            WorkflowOperation.CHECKPOINT_WORKLOADS,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_WORKLOAD,
        }
        for operation in operations:
            compiled, compile_errors = self.builder.compile_steps(
                [operation],
                profile,
                (
                    workload_scope
                    if operation in workload_operations
                    else ordered_failed_nodes
                ),
                gpu_uuids,
                workload_ids,
            )
            replacement_steps.extend(compiled)
            errors.extend(compile_errors)
        normalized_steps = []
        missing_restart_context_reported = restart_context_error is not None
        for step in replacement_steps:
            parameters = dict(step.parameters)
            if step.operation is WorkflowOperation.REPLACE_NODE:
                parameters["replacement_strategy"] = "HEALTHY_WARM_SPARE_ONLY"
            if step.operation is WorkflowOperation.RESTART_WORKLOAD:
                if restart_context is None:
                    if not missing_restart_context_reported:
                        errors.append(
                            "failed workflow has no complete restart safety context"
                        )
                        missing_restart_context_reported = True
                else:
                    parameters.update(restart_context)
            inventory_parameters = inventory_parameters_by_operation.get(step.operation)
            if inventory_parameters:
                parameters.update(inventory_parameters)
            if step.operation in NODE_ACTION_SCOPE_OPERATIONS and gpu_uuids_by_node:
                scoped = {
                    node_id: list(gpu_uuids_by_node[node_id])
                    for node_id in step.node_ids
                    if gpu_uuids_by_node.get(node_id)
                }
                if scoped:
                    parameters["gpu_uuids_by_node"] = scoped
            normalized_steps.append(step.model_copy(update={"parameters": parameters}))
        replacement_steps = normalized_steps
        return replacement_steps, errors

    def emit(
        self,
        workflow: WorkflowRequest,
        source: FaultIncident,
        *,
        failed_stage: str,
        next_action: RecoveryAction,
        next_operation: WorkflowOperation | None,
        escalation_name: str,
        event_id: str,
        scope: dict,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        ordered_failed_nodes = scope["ordered_failed_nodes"]
        workload_ids = scope["workload_ids"]
        gpu_uuids = scope["gpu_uuids"]
        failed = scope["failed"]
        node_reason = scope["node_reason"]
        diagnostic_guidance = scope["diagnostic_guidance"]
        diagnostic_evidence = scope["diagnostic_evidence"]
        now = datetime.now(timezone.utc)
        incident = FaultIncident(
            incident_id=f"inc-{event_id}",
            event_id=event_id,
            # The escalation is a continuation of the source incident: it
            # reports the policy, workload and drill that produced it (F-H1).
            event_type=source.event_type,
            event_source=source.event_source,
            source_boot_id=source.source_boot_id,
            cluster_id=source.cluster_id,
            node_ids=ordered_failed_nodes,
            gpu_uuids=gpu_uuids,
            job_id=source.job_id,
            attempt_id=source.attempt_id,
            workload_identity_source=source.workload_identity_source,
            policy_version=source.policy_version,
            policy_source=source.policy_source,
            policy_reference=source.policy_reference,
            drill_id=source.drill_id,
            effective_action=next_action,
            state=IncidentState.DETECTED,
            reasons=[
                f"{failed_stage} remediation failed; "
                f"failed operation or validation: {failed}; "
                f"nodes: {node_reason}",
                *list(dict.fromkeys(diagnostic_evidence)),
                *list(dict.fromkeys(diagnostic_guidance)),
            ],
            created_at=now,
            updated_at=now,
        )
        operations = _escalation_operations(
            failed_stage, next_action, next_operation, workload_ids
        )

        workload_scope = list(
            dict.fromkeys(
                node_id
                for step in workflow.official_steps
                if step.operation
                in {
                    WorkflowOperation.CHECKPOINT_WORKLOADS,
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowOperation.RESTART_WORKLOAD,
                }
                for node_id in step.node_ids
            )
        ) or list(source.node_ids)
        replacement_steps, errors = self._compile_replacement_steps(
            workflow,
            operations,
            workload_scope,
            ordered_failed_nodes,
            gpu_uuids,
            workload_ids,
            scope.get("gpu_uuids_by_node") or None,
        )
        replacement = WorkflowRequest(
            request_id=(f"workflow-{escalation_name}-after-{workflow.request_id}"),
            incident_id=incident.incident_id,
            runtime_profile_version=(workflow.runtime_profile_version),
            status=(WorkflowStatus.PENDING if not errors else WorkflowStatus.BLOCKED),
            official_action=next_action.value,
            fencing_token=incident.fencing_token,
            official_steps=replacement_steps,
            blocked_reasons=errors,
            blocked_kind=(BlockedKind.NEEDS_OPERATOR if errors else None),
            lifetime_deadline_at=_successor_lifetime(
                workflow, next_action, next_operation, now
            ),
            created_at=now,
            updated_at=now,
        )
        incident = incident.model_copy(
            update={
                "state": (
                    IncidentState.ACTION_PENDING
                    if not errors
                    else IncidentState.ESCALATED
                ),
                "workflow_request_id": replacement.request_id,
            }
        )
        # One transaction, keyed by the deterministic event id: two workers
        # escalating the same failed workflow get the same pair, and a
        # workflow never exists ahead of its incident (F-H1 / F-B1).
        created_incident, created_workflow, created = (
            self.store.create_incident_workflow_if_absent(
                event_id, lambda: (incident, replacement)
            )
        )
        if created and failed_stage == CONTAINMENT_REFUSED_STAGE:
            self.containment_refused_escalations_total += 1
        return created_incident, created_workflow

    def _terminate_chain(
        self,
        workflow: WorkflowRequest,
        source: FaultIncident,
        origin: tuple[str, str],
    ) -> None:
        """Hand a failed operator hand-off to an operator without opening
        another one.

        ``source`` is itself the product of a terminal escalation (a support
        ticket) and the workflow that was to deliver it failed. Escalating that
        failure would mint one more FAILED incident/workflow pair on the same
        node every reconcile tick (ARCH-E2E-2A finding 1). Instead the incident
        stays ESCALATED with the reason recorded, the operator receives the
        hardware-escalation notification the ESCALATE_SUPPORT step would have
        sent (same builder, same deduplication key, so a step that did run
        never doubles it), and the event is counted. Idempotent per incident:
        a replay finds the reason and the notification already there.
        """

        failed_operations = sorted(
            {
                execution.operation.value
                for execution in workflow.step_executions
                if execution.status is WorkflowStepStatus.FAILED
            }
        )
        reason = _chain_termination_reason(workflow, source, origin, failed_operations)
        first_termination = reason not in source.reasons
        if first_termination or source.state is not IncidentState.ESCALATED:
            now = datetime.now(timezone.utc)
            self.store.save_incident(
                source.model_copy(
                    update={
                        "state": IncidentState.ESCALATED,
                        "reasons": bounded_reasons([*source.reasons, reason]),
                        "updated_at": now,
                    }
                ),
                expected=source,
            )
        workload_ids = list(
            dict.fromkeys(
                workload_id
                for step in workflow.official_steps
                for workload_id in step.workload_ids
            )
        )
        notification = HardwareEscalationEmailBuilder().build(
            cluster_id=source.cluster_id,
            incident_id=source.incident_id,
            workflow_id=workflow.request_id,
            event_id=source.event_id,
            event_type=source.event_type,
            node_ids=list(source.node_ids),
            workload_ids=workload_ids,
            reasons=[*source.reasons, reason],
            failed_operations=failed_operations,
            policy_source=source.policy_source,
            official_action=source.official_action,
            ticket_id=f"vendor-ticket-{source.incident_id}",
        )
        self.store.save_notification_if_absent(notification)
        if first_termination:
            self.escalation_chain_terminated_total += 1
            self.escalation_chain_terminated_last_seen_timestamp_seconds = time.time()

    def escalate(
        self, workflow: WorkflowRequest
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        if workflow.status is not WorkflowStatus.FAILED:
            return None
        source = self.store.get_incident(workflow.incident_id)
        origin = escalation_origin(source.event_id)
        if origin is not None and origin[0] in CHAIN_TERMINAL_ESCALATIONS:
            # Bound on the escalation chain: a failed support escalation is
            # handed over, never escalated into another support escalation,
            # whatever its failed step would otherwise classify as.
            self._terminate_chain(workflow, source, origin)
            return None
        classification = self.classify(workflow)
        if classification is None:
            return None
        (
            failed_stage,
            next_action,
            next_operation,
            failed_executions,
        ) = classification
        escalation_name = ESCALATION_NAMES[next_action]
        event_id = (
            f"{escalation_name}{_ESCALATION_EVENT_SEPARATOR}{workflow.request_id}"
        )
        existing = self.store.get_incident_by_event(event_id)
        if existing is not None:
            if not existing.workflow_request_id:
                return None
            return (
                existing,
                self.store.get_workflow(existing.workflow_request_id),
            )
        scope = self.collect_scope(workflow, source, failed_executions)
        return self.emit(
            workflow,
            source,
            failed_stage=failed_stage,
            next_action=next_action,
            next_operation=next_operation,
            escalation_name=escalation_name,
            event_id=event_id,
            scope=scope,
        )
