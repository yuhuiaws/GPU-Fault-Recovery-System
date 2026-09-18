"""Physical outcome evidence required before releasing recovery protection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gpu_fault.models import BlockedKind, WorkflowStatus, workflow_is_open
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY

CANCELLATION_SOURCES = frozenset(
    {"workflow-timeout", "workflow-preempted", "completed-after-cancellation"}
)
TERMINAL_COMMAND_STATUSES = ("SUCCEEDED", "FAILED")


def cancellation_unresolved(
    record: Mapping[str, Any], details: Mapping[str, Any]
) -> bool:
    """A SQL cancellation needs a subsequent completion or proven no submission."""
    source = record.get("status_source")
    if source is not None and not isinstance(source, str):
        return True
    if source not in CANCELLATION_SOURCES:
        return False
    if details.get("post_cancellation_status") in TERMINAL_COMMAND_STATUSES:
        return False
    never_leased = (
        "last_lease_owner" in record
        and "lease_owner" in record
        and record["last_lease_owner"] is None
        and record["lease_owner"] is None
        and not details
    )
    no_start = details.get("node_action_not_started") is True and not any(
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
    )
    return not (never_leased or no_start)


def unresolved_details(value: object, *, depth: int = 0) -> bool:
    """Read current, legacy and nested receipts; malformed evidence stays held."""
    if not isinstance(value, Mapping) or depth > 24:
        return True
    flags = (
        "outcome_unknown",
        "manual_confirmation_required",
        "node_action_interrupted",
        "node_action_response_unknown",
        "ownership_permit_delivery_unknown",
    )
    if any(value.get(key) is not None and value.get(key) is not False for key in flags):
        return True
    if "reset_outcome_unknown" in value and (
        not isinstance(value["reset_outcome_unknown"], list)
        or value["reset_outcome_unknown"]
    ):
        return True
    if value.get("node_action_state") == "PENDING":
        return True
    if "details" in value and unresolved_details(value["details"], depth=depth + 1):
        return True
    for key in ("node_results", "node_failure_details", BATCHED_RESULTS_KEY):
        if key not in value:
            continue
        nested = value[key]
        if not isinstance(nested, Mapping):
            return True
        for item in nested.values():
            if not isinstance(item, Mapping) or unresolved_details(
                item, depth=depth + 1
            ):
                return True
            if key == BATCHED_RESULTS_KEY and (
                item.get("status") not in TERMINAL_COMMAND_STATUSES
                or cancellation_unresolved(item, item.get("details", {}))
            ):
                return True
    return False


def workflow_recovery_error(workflow: object) -> str | None:
    if not isinstance(workflow, Mapping):
        return "collector recovery workflow is malformed"
    try:
        status = WorkflowStatus(workflow["status"])
        raw_kind = workflow.get("blocked_kind")
        kind = BlockedKind(raw_kind) if raw_kind is not None else None
    except (KeyError, TypeError, ValueError):
        return "collector recovery workflow status is unknown"
    if (
        kind is BlockedKind.NEEDS_OPERATOR
        or workflow_is_open(status, kind)
        or (status is WorkflowStatus.BLOCKED and kind is None)
    ):
        return "workflow still owns recovery; operator hold retained"
    executions = workflow.get("step_executions", [])
    if not isinstance(executions, list) or any(
        not isinstance(item, Mapping) or unresolved_details(item.get("details", {}))
        for item in executions
    ):
        return "workflow physical outcome is unresolved; operator hold retained"
    return None


def command_recovery_error(command: object) -> str | None:
    if not isinstance(command, Mapping) or command.get("status") not in (
        TERMINAL_COMMAND_STATUSES
    ):
        return "remote command is not terminal; operator hold retained"
    details = command.get("result_details", {})
    if unresolved_details(details):
        return "remote physical outcome is unresolved; operator hold retained"
    if cancellation_unresolved(command, details):
        return "remote cancellation is not physical completion"
    return None


def recovery_safety_errors(workflows: object, commands: object) -> list[str]:
    """The same complete inventories gate CPU restoration and runner cleanup."""
    if not isinstance(workflows, list) or not isinstance(commands, list):
        return ["collector recovery snapshot is incomplete"]
    return [
        error
        for values, check in (
            (workflows, workflow_recovery_error),
            (commands, command_recovery_error),
        )
        for value in values
        if (error := check(value)) is not None
    ]
