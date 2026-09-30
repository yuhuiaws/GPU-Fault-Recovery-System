"""Prove a STOP participant's boot change from this workflow's reboot receipts.

A STOP receipt binds every participant node by UID and boot id. When a later
node action re-verifies the receipt and a participant's boot id has changed,
the change is ownership drift unless this workflow's own reboot records
account for it. Two shapes are accepted:

1. **Confirmed chain** (``_confirmed_transition``): the node's ``RESTART_NODE``
   executions SUCCEEDED, externally confirmed by HyperPod ``Running`` plus a
   new Node Agent incarnation, each carrying the ``stop_reboot_authorization_v1``
   record the pre-submit check signed for this receipt, and the recorded
   old/new boot ids chain from the receipt's boot id to the current one.

2. **In-flight sibling reboot** (``_in_flight_transition``): the node is not a
   target of the action being checked, and its latest ``RESTART_NODE``
   execution is still WAITING with a real provider submission (``action`` is
   the HyperPod reboot, ``requires_external_confirmation`` is set, the
   submission key is this workflow's, ``submitted_nodes`` names one logical id
   per target and the record is not a cached duplicate) whose
   ``stop_reboot_authorization_v1`` was signed for this receipt's digest and
   binds this node's UID and the boot id the chain expects. The product
   itself rebooted that node from the boot the receipt recorded and has not
   yet confirmed it, so the node coming back under any other boot id is that
   reboot, not a foreign actor. GF-REGIONAL-DESTR-014: two nodes share one
   receipt; the sibling's reboot is in flight (its agent stays down, so it can
   never be confirmed) when the fault node escalates to its own reboot, and the
   fault node's pre-submit check must not read the sibling's new boot as drift.

A boot change on a node with no such record, a record bound to another
receipt digest, a terminal (FAILED or unconfirmed SUCCEEDED) reboot, a node
that is itself the action's target, or a changed node UID remains
``STOP_OWNERSHIP_DRIFT``. Nothing here mutates the receipt.
"""

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


def _submitted_nodes(submitted: object, count: int) -> bool:
    """Whether ``submitted_nodes`` names one provider logical id per target."""

    return (
        isinstance(submitted, list)
        and len(submitted) == count
        and count > 0
        and all(isinstance(value, str) and value for value in submitted)
    )


def _bound_authorization(
    context: WorkflowStepContext,
    execution: WorkflowStepExecution,
    node_id: str,
    node_uid: str,
    receipt_digest: str,
    after: datetime,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The pre-submit authorization and agent baseline a reboot record binds.

    Shared by the confirmed-chain and the in-flight acceptance: the record must
    be this workflow's own reboot of a step naming ``node_id``, signed for this
    receipt digest, this node UID, and the isolation the incident holds. Returns
    ``(authorized node, agent baseline)`` for ``node_id`` or ``None``.
    """

    workflow = context.workflow
    index = execution.step_index
    step = workflow.official_steps[index]
    details = execution.details
    targets = set(step.node_ids)
    if (
        len(targets) != len(step.node_ids)
        or index in workflow.inherited_step_indexes
        or index in workflow.superseded_step_indexes
        or details.get("preemption_reuse")
        or details.get("inherited_from_workflow_id")
        or details.get("action") != HyperPodAction.REBOOT.value
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
    if (
        not isinstance(baseline, dict)
        or not isinstance(isolated, dict)
        or isolated.get("kubernetes_node") != node_id
        or isolated.get("unschedulable") is not True
        or isolated.get("incident") != context.incident.incident_id
        or isolated.get("fencing_token") != str(workflow.fencing_token)
    ):
        return None
    return authorized_node, baseline


def _in_flight_transition(
    context: WorkflowStepContext,
    execution: WorkflowStepExecution,
    node_id: str,
    node_uid: str,
    receipt_digest: str,
    after: datetime,
) -> str | None:
    """The boot id a still-unconfirmed, product-submitted reboot left from.

    Accepts only a WAITING ``RESTART_NODE`` record that carries a real provider
    submission for this receipt (shape 2 of the module docstring); returns the
    boot id its authorization bound, or ``None``. The record's unknown-outcome
    flags are expected here: the CPU marks every unconfirmed remote node action
    ``outcome_unknown`` until it is confirmed or an operator resolves it.
    """

    workflow = context.workflow
    index = execution.step_index
    details = execution.details
    if (
        execution.status is not WorkflowStepStatus.WAITING
        or not execution.adapter_operation_id
        or index in workflow.completed_step_indexes
        or details.get("requires_external_confirmation") is not True
        or details.get("provider_submission_duplicate") is True
        or details.get("node_action_not_started") is True
        or details.get("externally_confirmed") is True
    ):
        return None
    bound = _bound_authorization(
        context, execution, node_id, node_uid, receipt_digest, after
    )
    if bound is None:
        return None
    authorized_node, baseline = bound
    old_boot = authorized_node.get("boot_id")
    if (
        not isinstance(old_boot, str)
        or not old_boot
        or baseline.get("boot_id") != old_boot
        or not _submitted_nodes(
            details.get("submitted_nodes"),
            len(workflow.official_steps[index].node_ids),
        )
    ):
        return None
    return old_boot


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
    details = execution.details
    targets = set(workflow.official_steps[index].node_ids)
    if (
        execution.status is not WorkflowStepStatus.SUCCEEDED
        or not execution.adapter_operation_id
        or execution.error is not None
        or index not in workflow.completed_step_indexes
        or unknown_outcome_failure(details)
        or details.get("externally_confirmed") is not True
        or details.get("confirmation_source")
        != "hyperpod-running-and-new-agent-incarnation"
    ):
        return None
    bound = _bound_authorization(
        context, execution, node_id, node_uid, receipt_digest, after
    )
    if bound is None:
        return None
    authorized_node, baseline = bound
    agents = _observations(details.get("agent_observations"), targets)
    providers = _observations(details.get("provider_observations"), targets)
    if node_id not in agents or node_id not in providers:
        return None
    submitted = details.get("submitted_nodes")
    if not isinstance(submitted, list) or not _submitted_nodes(submitted, len(targets)):
        return None
    logical_ids = [provider.get("node_logical_id") for provider in providers.values()]
    if (
        not all(isinstance(value, str) and value for value in logical_ids)
        or len(set(logical_ids)) != len(targets)
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

    Every link must carry the UID/boot and immutable STOP digest checked before
    submission; the last link may still be in flight when ``node_id`` is not a
    target of ``context.step`` (module docstring, shape 2). The caller still
    checks live Node UID and all workload/Pod ownership.
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
    ordered = sorted(latest.values(), key=lambda item: item.updated_at)
    for position, execution in enumerate(ordered):
        if (
            execution.started_at.tzinfo is None
            or execution.updated_at.tzinfo is None
            or not after <= execution.started_at <= execution.updated_at <= now
        ):
            return False
        transition = _confirmed_transition(
            context, execution, node_id, node_uid, receipt_digest, after
        )
        if transition is None:
            # Shape 2: the chain may end in a reboot the product submitted and
            # has not confirmed, for a node this action does not itself touch.
            # The node left ``boot`` under our submission; whatever boot it
            # shows now is that reboot's, so the chain closes on it.
            in_flight = (
                None
                if node_id in context.step.node_ids or position != len(ordered) - 1
                else _in_flight_transition(
                    context, execution, node_id, node_uid, receipt_digest, after
                )
            )
            return in_flight is not None and in_flight == boot != current_boot_id
        if transition.old_boot != boot or (
            incarnation is not None and transition.old_incarnation != incarnation
        ):
            return False
        boot = transition.new_boot
        incarnation = transition.new_incarnation
        after = execution.updated_at
    return bool(latest) and boot == current_boot_id and boot != previous_boot_id
