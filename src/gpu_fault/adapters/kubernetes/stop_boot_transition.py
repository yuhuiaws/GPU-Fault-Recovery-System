"""Prove a STOP participant's boot change from this workflow's reboot receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from gpu_fault.execution.models import WorkflowStepContext
from gpu_fault.hyperpod import HyperPodAction, hyperpod_submission_idempotency_key
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import unknown_outcome_failure

REBOOT_AUTHORIZATION_KEY = "stop_reboot_authorization_v1"


@dataclass(frozen=True)
class _RebootTransition:
    old_boot: str
    new_boot: str
    old_incarnation: str
    new_incarnation: str


def _observations(values: object, targets: set[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list) or len(values) != len(targets):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            return {}
        node = value.get("node_id")
        if not isinstance(node, str) or node not in targets or node in result:
            return {}
        result[node] = value
    return result


def _confirmed_transition(
    context: WorkflowStepContext,
    execution: WorkflowStepExecution,
    node_id: str,
    node_uid: str,
    receipt_digest: str,
    after: datetime,
) -> _RebootTransition | None:
    workflow = context.workflow
    index = execution.step_index
    step = workflow.official_steps[index]
    details = execution.details
    targets = set(step.node_ids)
    if (
        execution.status is not WorkflowStepStatus.SUCCEEDED
        or not execution.adapter_operation_id
        or execution.error is not None
        or index not in workflow.completed_step_indexes
        or index in workflow.inherited_step_indexes
        or index in workflow.superseded_step_indexes
        or len(targets) != len(step.node_ids)
        or details.get("preemption_reuse")
        or details.get("inherited_from_workflow_id")
        or unknown_outcome_failure(details)
        or details.get("action") != HyperPodAction.REBOOT.value
        or details.get("externally_confirmed") is not True
        or details.get("confirmation_source")
        != "hyperpod-running-and-new-agent-incarnation"
        or details.get("submission_idempotency_key")
        != hyperpod_submission_idempotency_key(
            workflow.request_id, index, WorkflowOperation.RESTART_NODE
        )
    ):
        return None
    authorization = details.get(REBOOT_AUTHORIZATION_KEY)
    if not isinstance(authorization, dict):
        return None
    if any(
        authorization.get(key) != expected
        or type(authorization.get(key)) is not type(expected)
        for key, expected in {
            "version": 1,
            "cluster_id": context.incident.cluster_id,
            "workflow_id": workflow.request_id,
            "incident_id": context.incident.incident_id,
            "fencing_token": workflow.fencing_token,
            "phase": "official",
            "step_index": index,
            "execution_owner": step.execution_owner,
            "submission_idempotency_key": details["submission_idempotency_key"],
            "stop_receipt_sha256": receipt_digest,
        }.items()
    ):
        return None
    epoch = authorization.get("execution_epoch")
    authorized_nodes = authorization.get("nodes")
    checked_at = authorization.get("checked_at")
    if (
        type(epoch) is not int
        or not 0 < epoch <= workflow.execution_epoch
        or not isinstance(authorized_nodes, dict)
        or set(authorized_nodes) != targets
        or not isinstance(checked_at, str)
    ):
        return None
    try:
        checked = datetime.fromisoformat(checked_at)
    except ValueError:
        return None
    authorized_node = authorized_nodes.get(node_id)
    if (
        checked.tzinfo is None
        # The CPU can record the first attempt after the GPU Executor submitted it.
        or not after <= checked <= execution.updated_at
        or not isinstance(authorized_node, dict)
        or authorized_node.get("uid") != node_uid
    ):
        return None
    baselines = details.get("agent_baselines")
    isolation = details.get("observed_isolation")
    if (
        not isinstance(baselines, dict)
        or set(baselines) != targets
        or not isinstance(isolation, dict)
        or set(isolation) != targets
    ):
        return None
    baseline = baselines.get(node_id)
    isolated = isolation.get(node_id)
    agents = _observations(details.get("agent_observations"), targets)
    providers = _observations(details.get("provider_observations"), targets)
    if (
        not isinstance(baseline, dict)
        or not isinstance(isolated, dict)
        or node_id not in agents
        or node_id not in providers
        or isolated.get("kubernetes_node") != node_id
        or isolated.get("unschedulable") is not True
        or isolated.get("incident") != context.incident.incident_id
        or isolated.get("fencing_token") != str(workflow.fencing_token)
    ):
        return None
    submitted = details.get("submitted_nodes")
    logical_ids = [provider.get("node_logical_id") for provider in providers.values()]
    if (
        not isinstance(submitted, list)
        or not all(isinstance(value, str) and value for value in submitted)
        or not all(isinstance(value, str) and value for value in logical_ids)
        or len(set(logical_ids)) != len(targets)
        or len(submitted) != len(targets)
        or set(submitted) != set(logical_ids)
        or any(
            provider.get("status") != "Running"
            or not isinstance(provider.get("instance_id"), str)
            or not provider["instance_id"]
            for provider in providers.values()
        )
    ):
        return None
    observed = agents[node_id]
    old_boot, new_boot = baseline.get("boot_id"), observed.get("boot_id")
    old_incarnation = baseline.get("agent_incarnation_id")
    new_incarnation = observed.get("agent_incarnation_id")
    if (
        not isinstance(old_boot, str)
        or not old_boot
        or authorized_node.get("boot_id") != old_boot
        or not isinstance(new_boot, str)
        or not new_boot
        or old_boot == new_boot
        or not isinstance(old_incarnation, str)
        or not old_incarnation
        or not isinstance(new_incarnation, str)
        or not new_incarnation
        or old_incarnation == new_incarnation
    ):
        return None
    return _RebootTransition(old_boot, new_boot, old_incarnation, new_incarnation)


def authorized_boot_transition(
    context: WorkflowStepContext,
    *,
    node_id: str,
    node_uid: str,
    receipt_digest: str,
    previous_boot_id: str,
    current_boot_id: str,
    stopped_at: datetime,
) -> bool:
    """Accept only a complete, ordered chain of locally authorized reboots.

    The result must carry the UID/boot and immutable STOP digest checked before
    submission. The caller still checks live Node UID and all workload/Pod ownership.
    """

    workflow = context.workflow
    if (
        workflow.executes_safety_steps
        or workflow.incident_id != context.incident.incident_id
        or workflow.fencing_token != context.incident.fencing_token
        or context.request.expected_fencing_token != workflow.fencing_token
        or stopped_at.tzinfo is None
    ):
        return False
    latest: dict[int, WorkflowStepExecution] = {}
    for execution in workflow.step_executions:
        if (
            execution.phase == "official"
            and execution.operation is WorkflowOperation.RESTART_NODE
            and 0 <= execution.step_index < len(workflow.official_steps)
            and workflow.official_steps[execution.step_index].operation
            is WorkflowOperation.RESTART_NODE
            and node_id in workflow.official_steps[execution.step_index].node_ids
        ):
            latest[execution.step_index] = execution
    boot = previous_boot_id
    incarnation: str | None = None
    after = stopped_at
    now = datetime.now(timezone.utc)
    if any(
        execution.started_at.tzinfo is None or execution.updated_at.tzinfo is None
        for execution in latest.values()
    ):
        return False
    for execution in sorted(latest.values(), key=lambda item: item.updated_at):
        if (
            execution.started_at.tzinfo is None
            or execution.updated_at.tzinfo is None
            or not after <= execution.started_at <= execution.updated_at <= now
        ):
            return False
        transition = _confirmed_transition(
            context, execution, node_id, node_uid, receipt_digest, after
        )
        if (
            transition is None
            or transition.old_boot != boot
            or (incarnation is not None and transition.old_incarnation != incarnation)
        ):
            return False
        boot = transition.new_boot
        incarnation = transition.new_incarnation
        after = execution.updated_at
    return bool(latest) and boot == current_boot_id and boot != previous_boot_id
