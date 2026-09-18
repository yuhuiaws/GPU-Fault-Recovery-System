"""Bind reset audit rows to the injected workflow, fence and exact GPU identities."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from gpu_fault.node_agent.ledger import canonical_digest
from gpu_fault.node_agent.operations.reset import MAX_RESET_BUSY_ATTEMPTS
from scripts.e2e.regional.collector_acceptance_fixture import select_workflow

RESET_OPERATIONS = {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"}


def gpu_identity_map(inventory: list[dict[str, Any]]) -> dict[str, str]:
    identities = {
        str(item.get("uuid") or ""): str(item.get("pci_bdf") or "")
        for item in inventory
    }
    if (
        not inventory
        or len(identities) != len(inventory)
        or "" in identities
        or any(not value for value in identities.values())
    ):
        raise ValueError("GPU inventory identity is missing or duplicated")
    return identities


def _stamp(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("reset evidence timestamp has no timezone")
    return result


def _reset_counts_valid(details: dict[str, Any], expected_successes: int) -> bool:
    attempts = details.get("reset_attempts")
    successes = details.get("reset_successes")
    busy = details.get("reset_busy_refusals")
    if (
        type(attempts) is not int
        or not expected_successes
        <= attempts
        <= expected_successes * MAX_RESET_BUSY_ATTEMPTS
        or any(
            details.get(field)
            for field in (
                "outcome_unknown",
                "manual_confirmation_required",
                "reset_outcome_unknown",
                "reset_failed",
                "reset_not_attempted",
            )
        )
    ):
        return False
    if successes is None and busy is None:
        return attempts == expected_successes
    return (
        type(successes) is int
        and successes == expected_successes
        and type(busy) is int
        and busy >= 0
        and attempts == successes + busy
    )


def physical_reset_errors(
    baseline: dict[str, Any],
    after: dict[str, Any],
    state: dict[str, Any],
    *,
    node: str,
    operation: str = "RESET_GPU",
) -> list[str]:
    errors = []
    try:
        before_inventory = gpu_identity_map(baseline.get("gpu_inventory") or [])
        after_inventory = gpu_identity_map(after.get("gpu_inventory") or [])
        if before_inventory != after_inventory:
            errors.append("GPU UUID/PCI identity changed across the reset")
    except ValueError as exc:
        return [str(exc)]
    if not baseline.get("boot_id") or after.get("boot_id") != baseline["boot_id"]:
        errors.append("reset audit boot identity is absent or changed")
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    if (
        not workflow.get("request_id")
        or workflow.get("incident_id") != incident.get("incident_id")
        or incident.get("node_ids") != [node]
    ):
        errors.append("reset workflow/incident/node binding is incomplete")
    expected = (
        set(before_inventory)
        if operation == "RESET_ALL_GPUS_NVSWITCHES"
        else {str((state.get("event") or {}).get("gpu_uuid") or "")}
    )
    if not expected or "" in expected or not expected <= set(before_inventory):
        errors.append("reset target UUID is not in the original inventory")
    known = {
        (row.get("command_id"), row.get("attempt"))
        for row in baseline.get("ledger") or []
    }
    rows = [
        row
        for row in after.get("ledger") or []
        if row.get("operation") in RESET_OPERATIONS
        and (row.get("command_id"), row.get("attempt")) not in known
    ]
    if len(rows) != 1:
        return [*errors, "reset audit must add exactly one physical reset attempt"]
    row = rows[0]
    targets = row.get("gpu_uuids") or []
    if (
        row.get("operation") != operation
        or type(row.get("attempt")) is not int
        or row["attempt"] != 1
        or row.get("state") != "SUCCEEDED"
        or row.get("workflow_request_id") != workflow.get("request_id")
        or row.get("incident_id") != incident.get("incident_id")
        or type(row.get("fencing_token")) is not int
        or row.get("fencing_token") != workflow.get("fencing_token")
        or set(targets) != expected
        or len(targets) != len(expected)
        or row.get("signature_digest_present") is not True
    ):
        errors.append(
            "physical reset audit has the wrong operation, workflow, fence or UUIDs"
        )
    steps = [
        step
        for step in workflow.get("official_steps") or []
        if step.get("operation") == operation and node in (step.get("node_ids") or [])
    ]
    if len(steps) != 1:
        errors.append("reset audit has no single matching workflow step")
    else:
        step = steps[0]
        if set(step.get("gpu_uuids") or []) != expected or row.get(
            "parameters_digest"
        ) != canonical_digest(step.get("parameters") or {}):
            errors.append(
                "reset audit parameters or UUIDs differ from the authorized step"
            )
    result = row.get("result") or {}
    details = result.get("details") or {}
    if (
        result.get("command_id") != row.get("command_id")
        or result.get("operation") != operation
        or result.get("status") != "SUCCEEDED"
        or type(result.get("attempt")) is not int
        or result["attempt"] != 1
        or set(details.get("reset_gpu_uuids") or []) != expected
        or len(details.get("reset_gpu_uuids") or []) != len(expected)
        or details.get("verified_no_gpu_clients") is not True
        or not _reset_counts_valid(
            details, len(expected) if operation == "RESET_GPU" else 1
        )
    ):
        errors.append(
            "Node Agent result does not prove exactly one reset of the selected UUIDs"
        )
    if operation == "RESET_ALL_GPUS_NVSWITCHES" and (
        details.get("reset_scope") != "ALL_LOCAL_GPUS_AND_NVSWITCHES"
        or details.get("inventory_verified_before") is not True
        or details.get("inventory_verified_after") is not True
    ):
        errors.append("full fabric reset lacks both exact inventory checks")
    try:
        if not (
            _stamp(baseline.get("captured_at"))
            <= _stamp(row.get("started_at"))
            <= _stamp(row.get("completed_at"))
            <= _stamp(after.get("captured_at"))
        ):
            errors.append("physical reset occurred outside the captured audit window")
    except (ValueError, TypeError):
        errors.append("physical reset audit timing is unknown")
    return errors


def full_fabric_reset_errors(
    baseline: dict[str, Any],
    after: dict[str, Any],
    state: dict[str, Any],
    *,
    audit_before: dict[str, Any],
    audit_after: dict[str, Any],
    node: str,
) -> list[str]:
    errors = []
    workflow = select_workflow(
        state.get("workflows") or [], operation="RESET_ALL_GPUS_NVSWITCHES"
    )
    if workflow is None:
        errors.append("no workflow planned RESET_ALL_GPUS_NVSWITCHES")
        workflow = {}
    if workflow.get("status") != "SUCCEEDED":
        errors.append("full GPU/NVSwitch reset workflow is not SUCCEEDED")
    execution = next(
        (
            item
            for item in workflow.get("step_executions", [])
            if item.get("operation") == "RESET_ALL_GPUS_NVSWITCHES"
            and item.get("status") == "SUCCEEDED"
        ),
        None,
    )
    if execution is None:
        errors.append("RESET_ALL_GPUS_NVSWITCHES did not succeed")
    # Host snapshots include the whole ledger, so count only this injection's rows.
    known = {item.get("command_id") for item in baseline["ledger"]}
    rows = [
        item
        for item in after["ledger"]
        if item.get("operation") == "RESET_ALL_GPUS_NVSWITCHES"
        and item.get("command_id") not in known
    ]
    if len(rows) != 1:
        errors.append("full fabric reset ledger count is not one")
    if len(after["gpu_inventory"]) != len(baseline["gpu_inventory"]):
        errors.append("GPU inventory changed after full fabric reset")
    if after.get("boot_id") != baseline.get("boot_id"):
        errors.append("node boot ID changed: the reset became a reboot")
    incident: dict[str, Any] = next(
        (
            item
            for item in state.get("incidents") or []
            if item.get("incident_id") == workflow.get("incident_id")
        ),
        {},
    )
    errors.extend(
        physical_reset_errors(
            audit_before,
            audit_after,
            {"workflow": workflow, "incident": incident},
            node=node,
            operation="RESET_ALL_GPUS_NVSWITCHES",
        )
    )
    return errors
