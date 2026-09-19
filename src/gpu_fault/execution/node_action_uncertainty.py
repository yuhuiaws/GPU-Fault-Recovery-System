"""Keep uncertain physical actions fenced when automatic execution ends."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    StepPhase,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    execution_matches_step,
    execution_phase,
)
from gpu_fault.operation_registry import NODE_MUTATING_OPERATIONS
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY, RemoteCommandStatus

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand
    from gpu_fault.store.contracts import ControlPlaneStore

UNRESOLVED_NODE_ACTION = "NODE_ACTION_OUTCOME_UNRESOLVED"
# ``details`` key under which ``gpu-fault-admin submit-remediation --disposition
# confirm-node-action`` records the operator's confirmation of a node action
# whose outcome the executor never observed (DESTR-014: the node rebooted while
# its agent was down, so the RESTART_NODE failed at its cap with
# ``outcome_unknown``). Written by ``execution.node_action_confirmation``; read
# here so a confirmed record resolves everywhere the uncertainty is judged.
OPERATOR_CONFIRMED_KEY = "operator_confirmed"
# A confirmation is only one when it says who, under which reference, when,
# on which node and for which operation; anything less is not honoured.
OPERATOR_CONFIRMATION_REQUIRED_KEYS = (
    "actor",
    "reference",
    "confirmed_at",
    "node_id",
    "operation",
)


def operator_confirmation(details: Mapping[str, Any]) -> dict[str, Any] | None:
    """The operator confirmation ``details`` carries, or ``None``.

    The reader side of ``OPERATOR_CONFIRMED_KEY``: the writer records the
    evidence it judged verbatim, and this only checks the attribution is
    complete. A confirmation beats the uncertainty flags beside it -- the
    operator judged later evidence than the receipt that set them, and a
    refreshed receipt for the *same* command must not reopen the question
    (``refresh_remote_action_state`` keeps a confirmed record as written).
    """

    value = details.get(OPERATOR_CONFIRMED_KEY)
    if not isinstance(value, Mapping):
        return None
    for key in OPERATOR_CONFIRMATION_REQUIRED_KEYS:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            return None
    return dict(value)


def unresolved_node_action(execution: WorkflowStepExecution) -> bool:
    if (
        execution.status is WorkflowStepStatus.SUCCEEDED
        or execution.operation not in NODE_MUTATING_OPERATIONS
        or (
            execution.operation is WorkflowOperation.RESTORE_GPU_SERVICES
            and execution.details.get("restore_gpu_services_withheld") is True
        )
    ):
        return False
    return _unresolved_details(execution.details)


def _unresolved_details(details: Mapping[str, Any]) -> bool:
    if operator_confirmation(details) is not None:
        return False
    if any(
        details.get(name) is True
        for name in (
            "outcome_unknown",
            "node_action_interrupted",
            "node_action_response_unknown",
            "ownership_permit_delivery_unknown",
        )
    ):
        return True
    if (
        details.get("node_action_state") == "PENDING"
        and isinstance(details.get("node_action_command_id"), str)
        and details["node_action_command_id"]
    ):
        return True
    return (
        details.get("manual_confirmation_required") is True
        and details.get("node_action_not_started") is not True
    )


def _known_remote_no_start(details: Mapping[str, Any]) -> bool:
    if _unresolved_details(details) or any(
        details.get(key)
        for key in (
            "node_action_accepted_nodes",
            "completed_nodes",
            "node_results",
            "activated_spare_nodes",
            "provider_operation_id",
            "provider_mutation_submitted",
            "requires_external_confirmation",
        )
    ):
        return False
    state = details.get("node_action_state")
    return any(
        details.get(key) is True
        for key in (
            "node_action_not_started",
            "fleet_preflight_blocked",
            "multi_node_barrier_unavailable",
        )
    ) or (
        isinstance(state, str) and state in {"TRANSPORT_RETRY", "NEW_COMMAND_REQUIRED"}
    )


def pending_remote_action_details(
    operation: WorkflowOperation,
    status: RemoteCommandStatus,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """A leased mutation may have started since its last progress receipt."""
    result = dict(details)
    if operation in NODE_MUTATING_OPERATIONS and (
        status is RemoteCommandStatus.LEASED
        or (
            status is RemoteCommandStatus.WAITING
            and not _known_remote_no_start(details)
        )
    ):
        result.update(outcome_unknown=True, manual_confirmation_required=True)
        result.pop("node_action_not_started", None)
    return result


def _remote_step_execution(
    command: RemoteActionCommand, index: int, step: WorkflowStepSpec
) -> WorkflowStepExecution | None:
    status = command.status
    details = dict(command.result_details)
    entries = details.get(BATCHED_RESULTS_KEY)
    entry = entries.get(str(index)) if isinstance(entries, dict) else None
    completed_entry = False
    if isinstance(entry, dict):
        raw = entry.get("details")
        details = dict(raw) if isinstance(raw, dict) else {"outcome_unknown": True}
        try:
            raw_status = entry.get("status")
            if not isinstance(raw_status, str):
                raise ValueError("remote step status is missing or malformed")
            recorded_status = RemoteCommandStatus(raw_status)
        except (TypeError, ValueError):
            recorded_status = RemoteCommandStatus.WAITING
            details["outcome_unknown"] = True
        completed_entry = recorded_status in {
            RemoteCommandStatus.SUCCEEDED,
            RemoteCommandStatus.FAILED,
        }
        status = (
            RemoteCommandStatus.LEASED
            if command.status is RemoteCommandStatus.LEASED and not completed_entry
            else recorded_status
        )
    elif index != command.step_index and isinstance(entries, dict):
        if command.status is not RemoteCommandStatus.LEASED:
            return None
        details = {}
    if (
        command.status is RemoteCommandStatus.FAILED
        and command.status_source
        in {"workflow-timeout", "workflow-preempted", "completed-after-cancellation"}
        and not completed_entry
    ):
        after = command.result_details.get("post_cancellation_status")
        if isinstance(after, str) and after in {"SUCCEEDED", "FAILED"}:
            status = RemoteCommandStatus(after)
        elif (
            command.last_lease_owner is None
            and command.lease_owner is None
            and not details
        ):
            status = RemoteCommandStatus.PENDING
        else:
            status = RemoteCommandStatus.WAITING
    details = pending_remote_action_details(step.operation, status, details)
    return WorkflowStepExecution(
        step_index=index,
        operation=step.operation,
        phase=execution_phase(command.workflow),
        status=(
            WorkflowStepStatus.SUCCEEDED
            if status is RemoteCommandStatus.SUCCEEDED
            else WorkflowStepStatus.FAILED
            if status is RemoteCommandStatus.FAILED
            else WorkflowStepStatus.WAITING
        ),
        adapter_operation_id=f"remote/{command.command_id}",
        details={
            **details,
            "remote_command_id": command.command_id,
            "remote_cluster_id": command.cluster_id,
            "remote_status": command.status.value,
            "remote_fencing_token": command.fencing_token,
        },
        error=command.error,
        started_at=command.created_at,
        updated_at=command.updated_at,
    )


_BOUND_DETAILS = frozenset(
    {
        "workflow_execution_deadline",
        "workflow_deadline_overdue_seconds",
        "workflow_deadline_remote_command_cancellation",
        "workflow_lifetime_exceeded",
        "step_waiting_timeout_seconds",
        "step_waiting_seconds",
        "step_waiting_slow",
    }
)


def refresh_remote_action_state(
    store: ControlPlaneStore, workflow: WorkflowRequest
) -> WorkflowRequest:
    """Refresh receipts, never infer physical completion from remote cancellation."""
    candidates: dict[
        tuple[StepPhase | None, int, WorkflowOperation], list[WorkflowStepExecution]
    ] = {}
    for command in store.list_remote_commands(
        workflow_request_ids=[workflow.request_id]
    ):
        for index, step in [
            (command.step_index, command.step),
            *((item.step_index, item.step) for item in command.batched_steps),
        ]:
            if step.operation not in NODE_MUTATING_OPERATIONS:
                continue
            snapshot = _remote_step_execution(command, index, step)
            if snapshot is None or (
                command.fencing_token != workflow.fencing_token
                and not unresolved_node_action(snapshot)
            ):
                continue
            key = (snapshot.phase, index, step.operation)
            candidates.setdefault(key, []).append(snapshot)
    executions = list(workflow.step_executions)
    for (phase, index, operation), snapshots in candidates.items():
        unresolved = [item for item in snapshots if unresolved_node_action(item)]
        snapshot = max(unresolved or snapshots, key=lambda item: item.updated_at)
        positions = [
            position
            for position, item in enumerate(executions)
            if execution_matches_step(item, index, operation, phase)
        ]
        previous = executions[positions[-1]] if positions else None
        if previous is None:
            if unresolved:
                executions.append(snapshot)
            continue
        if (
            previous.status is WorkflowStepStatus.SUCCEEDED
            or operator_confirmation(previous.details) is not None
        ):
            # A finished step, or one whose outcome an operator confirmed on
            # evidence, is not rewritten by a receipt for the same command.
            # A *different* command still unresolved on this step is a new
            # question and is appended as its own record.
            if (
                unresolved
                and previous.adapter_operation_id != snapshot.adapter_operation_id
            ):
                executions.append(snapshot)
            continue
        details = {
            key: value
            for key, value in previous.details.items()
            if key in _BOUND_DETAILS
        }
        details.update(snapshot.details)
        if unresolved:
            details["unresolved_remote_command_ids"] = sorted(
                {str(item.details["remote_command_id"]) for item in unresolved}
            )
        executions[positions[-1]] = previous.model_copy(
            update={
                "adapter_operation_id": snapshot.adapter_operation_id,
                "details": details,
            }
        )
    if executions == workflow.step_executions:
        return workflow
    return workflow.model_copy(update={"step_executions": executions})


def has_unresolved_node_action(
    workflow: WorkflowRequest, *, include_restoration: bool = True
) -> bool:
    seen: set[tuple[StepPhase | None, int, WorkflowOperation]] = set()
    for execution in reversed(workflow.step_executions):
        identity = (execution.phase, execution.step_index, execution.operation)
        if identity in seen:
            continue
        seen.add(identity)
        if (
            not include_restoration
            and execution.operation is WorkflowOperation.RESTORE_GPU_SERVICES
        ):
            continue
        if unresolved_node_action(execution):
            return True
    return False


def latest_node_action_executions(
    workflow: WorkflowRequest,
) -> list[WorkflowStepExecution]:
    """The newest record of every node-mutating step, in execution order.

    The same identity rule ``has_unresolved_node_action`` applies: one record
    per (phase, step index, operation), the last written one answering.
    """

    seen: set[tuple[StepPhase | None, int, WorkflowOperation]] = set()
    latest: list[WorkflowStepExecution] = []
    for execution in reversed(workflow.step_executions):
        identity = (execution.phase, execution.step_index, execution.operation)
        if identity in seen or execution.operation not in NODE_MUTATING_OPERATIONS:
            continue
        seen.add(identity)
        latest.append(execution)
    latest.reverse()
    return latest


def node_actions_operator_confirmed(workflow: WorkflowRequest) -> bool:
    """Whether a BLOCKED record's physical uncertainty was answered by hand.

    True when the record is BLOCKED, at least one of its node actions carries
    an operator confirmation and none is still unresolved. Such a record holds
    its node only as paperwork: the incident close (on node evidence) and the
    dispatcher's settled-incident sweep may end it, where an unconfirmed
    NEEDS_OPERATOR record must keep occupying the node (F-A4). Not BLOCKED
    means not parked, and the predicate does not apply.
    """

    if workflow.status is not WorkflowStatus.BLOCKED:
        return False
    latest = latest_node_action_executions(workflow)
    confirmed = any(operator_confirmation(item.details) is not None for item in latest)
    return confirmed and not any(unresolved_node_action(item) for item in latest)


def restoration_refusal(
    workflow: WorkflowRequest, step: WorkflowStepSpec
) -> WorkflowStepOutcome | None:
    if (
        step.operation is not WorkflowOperation.RESTORE_GPU_SERVICES
        or not has_unresolved_node_action(workflow, include_restoration=False)
    ):
        return None
    return WorkflowStepOutcome.failed(
        "automatic GPU service restoration requires resolved node actions",
        details={
            "reason": UNRESOLVED_NODE_ACTION,
            "outcome_unknown": True,
            "manual_confirmation_required": True,
            "restore_gpu_services_withheld": True,
            "safety_rejection": True,
        },
    )


def quiesce_needs_restoration(
    workflow: WorkflowRequest,
    steps: list[WorkflowStepSpec] | None,
    *,
    settling_operations: Collection[WorkflowOperation],
) -> bool:
    """Match completed quiesce and restoration by node and execution order."""

    if steps is None:
        steps = (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
    completed = set(workflow.completed_step_indexes)
    for quiesce_index, quiesce in enumerate(steps):
        if (
            quiesce_index not in completed
            or quiesce.operation is not WorkflowOperation.QUIESCE_GPU_SERVICES
        ):
            continue
        restored: set[str] = set()
        for restore_index, restore in enumerate(steps):
            if (
                restore_index > quiesce_index
                and restore_index in completed
                and (
                    restore.operation is WorkflowOperation.RESTORE_GPU_SERVICES
                    or restore.operation in settling_operations
                )
            ):
                restored.update(restore.node_ids)
        if set(quiesce.node_ids) - restored:
            return True
    return False
