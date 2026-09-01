from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    HARDWARE_ESCALATION_RELEVANT_OPERATIONS,
)
from gpu_fault.store.shared.errors import NotFoundError


RESTART_SAFETY_PARAMETERS = (
    "cluster_id",
    "job_id",
    "source_attempt_id",
    "source_gpu_count",
    "restart_budget",
)


class HardwareEscalationService:
    def __init__(self, store, builder) -> None:
        self.store = store
        self.builder = builder

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
        prior_operation_set = {
            step.operation
            for execution in workflow.step_executions
            if (
                execution.status is WorkflowStepStatus.FAILED
                and 0 <= execution.step_index < len(workflow.official_steps)
            )
            for step in workflow.official_steps[: execution.step_index]
        }
        recovery_context_operations = successful_operation_set | prior_operation_set
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
        elif failed_operation_set.intersection(
            {
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_FABRIC,
            }
        ):
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
        else:
            return None

        failed_executions = [
            execution
            for execution in workflow.step_executions
            if (
                execution.status is WorkflowStepStatus.FAILED
                and execution.operation in HARDWARE_ESCALATION_RELEVANT_OPERATIONS
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
            event_type="NODE_HEALTH_BATCH",
            cluster_id=source.cluster_id,
            node_ids=ordered_failed_nodes,
            gpu_uuids=gpu_uuids,
            policy_version="site-node-health-policy/v1",
            policy_source="SITE_NODE_HEALTH",
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
        if next_action is RecoveryAction.DRAIN:
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
        self.store.save_workflow(replacement)
        self.store.save_incident(incident)
        return incident, replacement

    def escalate(
        self, workflow: WorkflowRequest
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        if workflow.status is not WorkflowStatus.FAILED:
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
        escalation_name = {
            RecoveryAction.DRAIN: "drain",
            RecoveryAction.REBOOT_NODE: "reboot",
            RecoveryAction.REPLACE_NODE: "replace",
            RecoveryAction.ESCALATE_OPERATOR: "support",
        }[next_action]
        event_id = f"{escalation_name}-after-{workflow.request_id}"
        existing = self.store.get_incident_by_event(event_id)
        if existing is not None:
            if not existing.workflow_request_id:
                return None
            return (
                existing,
                self.store.get_workflow(existing.workflow_request_id),
            )
        source = self.store.get_incident(workflow.incident_id)
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
