"""Verdicts for HA-001's explicitly simulated remote-command closure."""

from __future__ import annotations

from collections import Counter
from typing import Any

from gpu_fault.models import WorkflowOperation


def closure_seed(run_id: str) -> dict[str, Any]:
    operations = [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_WORKLOAD,
    ]
    return {
        "incident_id": f"incident-{run_id}",
        "event_id": f"event-{run_id}",
        "workflow_id": f"workflow-{run_id}",
        "command_ids": [f"remote-{run_id}-{index}" for index in range(len(operations))],
        "operations": [item.value for item in operations],
    }


def closure_summary(
    seed: dict[str, Any], closure: dict[str, Any], ledger: dict[str, Any]
) -> dict[str, Any]:
    commands = closure.get("commands") or []
    succeeded = [item for item in commands if item.get("status") == "SUCCEEDED"]
    by_step: Counter[str] = Counter(str(item.get("step_index")) for item in succeeded)
    owners = {
        str(item.get("command_id")): item.get("last_lease_owner")
        or item.get("lease_owner")
        for item in commands
    }
    physical = sum(
        1
        for item in succeeded
        if (item.get("result_details") or {}).get("cached") is False
    )
    return {
        "workflow_id": closure.get("workflow_id"),
        "workflow_status_observed": closure.get("workflow_status_observed"),
        "command_statuses": {
            str(item.get("command_id")): item.get("status") for item in commands
        },
        "succeeded_by_step_index": dict(sorted(by_step.items())),
        "lease_owners_by_command": owners,
        "physical_executions": physical,
        "ledger_physical_count": ledger.get("physical_count"),
        "ledger_operations": ledger.get("operations"),
        "expected_operations": seed.get("operations"),
    }


def closure_errors(
    seed: dict[str, Any], summary: dict[str, Any], *, executor_id: str
) -> list[str]:
    errors = []
    command_ids = [str(value) for value in seed.get("command_ids", [])]
    expected_steps = {str(index): 1 for index in range(len(command_ids))}
    if summary["succeeded_by_step_index"] != expected_steps:
        errors.append(
            "synthetic closure does not have exactly one SUCCEEDED command per "
            f"step_index: {summary['succeeded_by_step_index']}"
        )
    for command_id in command_ids:
        if summary["command_statuses"].get(command_id) != "SUCCEEDED":
            errors.append(f"synthetic command {command_id} is not SUCCEEDED")
        owner = summary["lease_owners_by_command"].get(command_id)
        if not owner:
            errors.append(f"synthetic command {command_id} has no lease owner")
        elif owner != executor_id:
            errors.append(
                f"synthetic command {command_id} was completed by {owner}, "
                f"not the probe executor {executor_id}"
            )
    if summary["physical_executions"] != len(command_ids):
        errors.append(
            "synthetic closure commands were not each executed physically once: "
            f"{summary['physical_executions']} != {len(command_ids)}"
        )
    if summary["ledger_physical_count"] != len(command_ids):
        errors.append(
            f"synthetic closure ledger physical count is not {len(command_ids)}"
        )
    if summary["ledger_operations"] != summary["expected_operations"]:
        errors.append("synthetic closure operations are incomplete or reordered")
    return errors
