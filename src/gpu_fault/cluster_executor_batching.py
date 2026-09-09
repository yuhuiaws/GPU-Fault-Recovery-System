"""Run a compound remote command's steps in order on the data plane (性能 C).

The control plane hands the executor one ``RemoteActionCommand`` whose
``batched_steps`` are the node-side steps that follow its head step
(``gpu_fault.remote_step_batching``). Each step here gets the
``WorkflowStepContext`` it would have had as a command of its own: its own
index and idempotency key, and a workflow copy whose ``step_executions``
record the steps already finished under this command, so the node-action
adapter's maintenance-window and agent-generation checks find the QUIESCE
evidence exactly where they find it today.

One lease covers the run. Between steps the lease guard is consulted: a
cancellation the control plane asked for (a stronger workflow, a timeout)
stops before the next step starts and reports what ran, and a lease this
executor stopped trusting is withheld by ``_execute_and_report`` as before.
"""

from __future__ import annotations

import logging
from typing import Any

from gpu_fault.adapters.node_action.lease_guard import lease_hold_reason
from gpu_fault.cluster_executor_results import (
    failure_result,
    fleet_preflight_hold,
    outcome_result,
)
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepStatus,
    execution_phase,
)
from gpu_fault.regional import (
    BatchedStepResult,
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY

LOGGER = logging.getLogger(__name__)

STOPPED_BETWEEN_STEPS_STATUS_SOURCE = "executor-stopped-between-batched-steps"


def _plan(command: RemoteActionCommand) -> list[tuple[int, Any, str]]:
    return [
        (command.step_index, command.step, command.idempotency_key),
        *(
            (item.step_index, item.step, item.idempotency_key)
            for item in command.batched_steps
        ),
    ]


def _prior_results(command: RemoteActionCommand) -> dict[str, dict[str, Any]]:
    """Per-step verdicts a previous claim left on the row (resume point)."""

    results = command.result_details.get(BATCHED_RESULTS_KEY)
    if not isinstance(results, dict):
        return {}
    return {
        str(index): dict(entry)
        for index, entry in results.items()
        if isinstance(entry, dict)
    }


def _workflow_view(
    command: RemoteActionCommand,
    results: dict[str, dict[str, Any]],
    current_index: int,
) -> WorkflowRequest:
    """The workflow as the control plane would have recorded it before
    ``current_index``: SUCCEEDED records (and completed indexes/operations)
    for the covered steps already done, plus this step's own last WAITING
    record when a previous claim handed the command back at it -- the same
    synthetic prior record a single re-claimed command gets."""

    workflow = command.workflow
    phase = execution_phase(workflow)
    operation_id = f"remote/{command.command_id}"
    executions = list(workflow.step_executions)
    completed = set(workflow.completed_step_indexes)
    operations = list(workflow.completed_operations)
    for index, step, _ in _plan(command):
        entry = results.get(str(index))
        if entry is None:
            continue
        status = entry.get("status")
        details = entry.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        if status == RemoteCommandStatus.SUCCEEDED.value:
            executions.append(
                WorkflowStepExecution(
                    step_index=index,
                    operation=step.operation,
                    status=WorkflowStepStatus.SUCCEEDED,
                    phase=phase,
                    adapter_operation_id=operation_id,
                    details=details,
                )
            )
            completed.add(index)
            if step.operation not in operations:
                operations.append(step.operation)
        elif index == current_index and status == RemoteCommandStatus.WAITING.value:
            executions.append(
                WorkflowStepExecution(
                    step_index=index,
                    operation=step.operation,
                    status=WorkflowStepStatus.WAITING,
                    phase=phase,
                    adapter_operation_id=operation_id,
                    details=details,
                )
            )
    return workflow.model_copy(
        update={
            "step_executions": executions,
            "completed_step_indexes": sorted(completed),
            "completed_operations": operations,
        }
    )


def _post_progress(
    executor: Any, command: RemoteActionCommand, entry: dict[str, dict[str, Any]]
) -> None:
    try:
        executor.client.progress(command, executor.executor_id, entry)
    except Exception as exc:  # noqa: BLE001 - best effort, see module docstring
        executor.batched_progress_failures_total += 1
        LOGGER.warning(
            "regional cluster executor could not report batched progress; the "
            "terminal result will carry it: command=%s cluster=%s steps=%s: %s: %s",
            command.command_id,
            command.cluster_id,
            ",".join(entry),
            type(exc).__name__,
            exc,
        )


def execute_batched_command(
    executor: Any,
    command: RemoteActionCommand,
    adapter: Any,
    lease_token: str,
) -> RemoteCommandResult:
    """Run the covered steps in order; stop at the first that does not succeed.

    SUCCEEDED continues; WAITING reports the whole command WAITING with the
    progress so far, so the next claim resumes at that step; FAILED reports
    FAILED with the failing step recorded and the rest never run. A hold
    reason from the lease guard (cancellation, lost lease) stops before the
    next step. ``executor`` is the ``ClusterActionExecutor``: its adapter
    match, preflight, outcome mapping and exception taxonomy are reused so a
    step inside a compound command is judged exactly like one on its own.
    """

    executor.batched_commands_total += 1
    results = _prior_results(command)
    plan = _plan(command)
    for index, step, idempotency_key in plan:
        prior = results.get(str(index))
        if prior is not None and prior.get("status") == (
            RemoteCommandStatus.SUCCEEDED.value
        ):
            continue
        hold = lease_hold_reason()
        if hold is not None:
            LOGGER.warning(
                "compound remote command stopped between steps: command=%s "
                "cluster=%s next_step=%s/%s reason=%s",
                command.command_id,
                command.cluster_id,
                index,
                step.operation.value,
                hold,
            )
            return RemoteCommandResult(
                lease_token=lease_token,
                status=RemoteCommandStatus.WAITING,
                status_source=STOPPED_BETWEEN_STEPS_STATUS_SOURCE,
                details={
                    BATCHED_RESULTS_KEY: results,
                    "batched_stopped_before_step_index": index,
                    "node_action_not_started": True,
                    "reason": hold,
                },
            )
        workflow = _workflow_view(command, results, index)
        preflight_hold = fleet_preflight_hold(
            executor, command, workflow, step.operation, lease_token
        )
        if preflight_hold is not None:
            return preflight_hold.model_copy(
                update={
                    "details": {
                        **preflight_hold.details,
                        BATCHED_RESULTS_KEY: results,
                        "batched_step_index": index,
                    }
                }
            )
        context = WorkflowStepContext(
            workflow=workflow,
            incident=command.incident,
            step=step,
            step_index=index,
            request=executor._execution_request(command),
            idempotency_key=idempotency_key,
        )
        try:
            sub_result = outcome_result(
                adapter.execute(context), lease_token, operation=step.operation
            )
        except Exception as exc:  # noqa: BLE001 - the executor's taxonomy classifies it
            sub_result = failure_result(
                executor,
                exc,
                command,
                lease_token,
                operation=step.operation,
                node_ids=step.node_ids,
            )
        entry = BatchedStepResult(
            status=sub_result.status,
            status_source=sub_result.status_source,
            details=sub_result.details,
            error=sub_result.error,
        ).model_dump(mode="json")
        results[str(index)] = entry
        executor.batched_steps_total += 1
        _post_progress(executor, command, {str(index): entry})
        if sub_result.status is not RemoteCommandStatus.SUCCEEDED:
            return RemoteCommandResult(
                lease_token=lease_token,
                status=sub_result.status,
                status_source=sub_result.status_source,
                error=sub_result.error,
                details={
                    **sub_result.details,
                    BATCHED_RESULTS_KEY: results,
                    "batched_step_index": index,
                    "batched_operation": step.operation.value,
                },
            )
    return RemoteCommandResult(
        lease_token=lease_token,
        status=RemoteCommandStatus.SUCCEEDED,
        details={
            BATCHED_RESULTS_KEY: results,
            "batched_step_indexes": [index for index, _, _ in plan],
        },
    )
