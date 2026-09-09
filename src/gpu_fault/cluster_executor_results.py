"""How one step's outcome or exception becomes a ``RemoteCommandResult``.

Pulled out of ``ClusterActionExecutor._execute`` so a step inside a compound
command (``cluster_executor_batching``) is judged by exactly the same rules as
a command carrying one step: the same fleet preflight hold, the same status
mapping, the same taxonomy of exceptions (a control-plane rejection, a
retryable transport or adapter error, an AWS configuration gap, an executor
defect). ``executor`` is the ``ClusterActionExecutor`` whose counters and
registry these read; ``operation``/``node_ids`` are the step's own so a later
step of a compound command is logged and tagged as itself, not as the head.
"""

from __future__ import annotations

import logging
from typing import Any

from gpu_fault.aws_errors import aws_configuration_error
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.execution.fleet_preflight import (
    command_requires_fleet_preflight,
    fleet_preflight_reason,
)
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)

LOGGER = logging.getLogger(__name__)


class ClusterExecutorError(RuntimeError):
    """Raised by the executor for a command it refuses on purpose (cluster
    mismatch, stale fence, no or ambiguous adapter) and by the control-plane
    client for a non-2xx answer; ``status_code`` carries the HTTP status."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def fleet_preflight_hold(
    executor: Any,
    command: RemoteActionCommand,
    workflow: Any,
    operation: WorkflowOperation,
    lease_token: str,
) -> RemoteCommandResult | None:
    """The WAITING hold the fleet preflight puts on a destructive step.

    ``workflow`` is passed separately from ``command`` because a compound
    command's later steps are judged against the workflow as it stands
    after the earlier steps ran (their completion is what lets the
    preflight stop fencing compensation), not as it was when minted.
    """

    if executor.fleet_registry is None or not command_requires_fleet_preflight(
        operation
    ):
        return None
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    preflight_error = fleet_preflight_reason(
        executor.fleet_registry,
        workflow,
        command.incident,
        steps,
    )
    if preflight_error is None:
        return None
    LOGGER.warning(
        "remote command held before destructive action: "
        "command=%s workflow=%s operation=%s reason=%s",
        command.command_id,
        command.workflow_request_id,
        operation.value,
        preflight_error,
    )
    return RemoteCommandResult(
        lease_token=lease_token,
        status=RemoteCommandStatus.WAITING,
        details={
            "fleet_preflight_blocked": True,
            "reason": preflight_error,
        },
    )


def outcome_result(
    outcome: WorkflowStepOutcome,
    lease_token: str,
    *,
    operation: WorkflowOperation,
) -> RemoteCommandResult:
    status = {
        WorkflowStepStatus.WAITING: (RemoteCommandStatus.WAITING),
        WorkflowStepStatus.SUCCEEDED: (RemoteCommandStatus.SUCCEEDED),
        WorkflowStepStatus.FAILED: (RemoteCommandStatus.FAILED),
    }[outcome.status]
    details = dict(outcome.details or {})
    error = outcome.error
    if status is RemoteCommandStatus.FAILED and not error:
        # A FAILED outcome without a message is still the adapter's
        # verdict, not an executor defect. Left as None it failed the
        # result model's validation inside this try block and was
        # caught below as an executor-internal-error -- a stack trace,
        # an unexpected-failure count and an alert for a refusal the
        # adapter merely forgot to describe.
        error = f"{operation.value} adapter reported FAILED without an error message"
        details["error_message_missing"] = True
    return RemoteCommandResult(
        lease_token=lease_token,
        status=status,
        details=details,
        error=error,
    )


def failure_result(
    executor: Any,
    exc: BaseException,
    command: RemoteActionCommand,
    lease_token: str,
    *,
    operation: WorkflowOperation,
    node_ids: list[str],
) -> RemoteCommandResult:
    """Classify an exception raised while running one step of ``command``.

    ``operation``/``node_ids`` are the step's own so a compound command's
    later steps are logged and tagged as themselves, not as the head.
    """

    if isinstance(exc, ClusterExecutorError):
        retryable: RemoteCommandResult | None = (
            executor._retryable_control_plane_result(exc, command, lease_token)
        )
        if retryable is not None:
            return retryable
        # Rejections the executor raises on purpose: cluster
        # mismatch, stale fencing token, no or ambiguous adapter.
        # These are legitimate FAILED results, not executor bugs.
        LOGGER.warning(
            "regional cluster executor rejected command: "
            "command=%s cluster=%s operation=%s owner=%s nodes=%s: %s",
            command.command_id,
            command.cluster_id,
            operation.value,
            command.step.execution_owner,
            ",".join(node_ids),
            exc,
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.FAILED,
            status_source="executor-rejected",
            error=f"{type(exc).__name__}: {exc}",
        )
    retryable = executor._retryable_result(exc, command, lease_token)
    if retryable is not None:
        return retryable
    configuration_reason = aws_configuration_error(exc)
    if configuration_reason is not None:
        # A missing IRSA annotation, an unassumable role or a
        # denied API call is a deployment gap, not a defect: no
        # stack trace, no internal-error count (which alerting
        # watches), and a reason that names the knob. Counting
        # these as executor bugs is what once made "the
        # ServiceAccount has no role-arn" look like an adapter
        # crash for the operator reading the step details.
        LOGGER.error(
            "regional cluster executor is misconfigured for "
            "AWS: command=%s cluster=%s operation=%s nodes=%s: "
            "%s (%s)",
            command.command_id,
            command.cluster_id,
            operation.value,
            ",".join(node_ids),
            configuration_reason,
            type(exc).__name__,
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.FAILED,
            status_source="executor-configuration-error",
            error=configuration_reason,
            details={
                "configuration_error": True,
                "executor_id": executor.executor_id,
                "exception_type": type(exc).__name__,
            },
        )
    # Anything else is an executor-side defect (a missing
    # attribute, a bad adapter wiring, an unhandled provider
    # error). Reporting it as a bare FAILED string hides the
    # difference between "the action was refused" and "the
    # executor is broken", so record a stack trace, tag the
    # result, and count it for alerting.
    executor.unexpected_failures += 1
    LOGGER.exception(
        "regional cluster executor raised while executing: "
        "command=%s cluster=%s operation=%s owner=%s nodes=%s",
        command.command_id,
        command.cluster_id,
        operation.value,
        command.step.execution_owner,
        ",".join(node_ids),
        exc_info=exc,
    )
    return RemoteCommandResult(
        lease_token=lease_token,
        status=RemoteCommandStatus.FAILED,
        status_source="executor-internal-error",
        error=f"{type(exc).__name__}: {exc}",
        details={
            "executor_internal_error": True,
            "executor_id": executor.executor_id,
            "exception_type": type(exc).__name__,
        },
    )
