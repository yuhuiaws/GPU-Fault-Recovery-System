"""Bind destructive drill timers to an observed, still-waiting workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError


def barrier_authorization(
    state: dict[str, Any],
    *,
    run_id: str,
    node: str,
    boot_id: str,
    device: str,
    drill_id: str,
    maintenance_window_end: datetime,
) -> dict[str, Any]:
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    if (
        not workflow.get("request_id")
        or not incident.get("incident_id")
        or workflow.get("incident_id") != incident["incident_id"]
        or incident.get("drill_id") != drill_id
        or workflow.get("status") != "RUNNING"
        or type(workflow.get("fencing_token")) is not int
        or workflow["fencing_token"] < 1
        or not boot_id
    ):
        raise RegionalFixtureError(
            "destructive timer has no exact drill/workflow binding"
        )
    executions = workflow.get("step_executions") or []
    quiesce = [
        item for item in executions if item.get("operation") == "QUIESCE_GPU_SERVICES"
    ]
    verify = [
        item for item in executions if item.get("operation") == "VERIFY_NO_GPU_CLIENTS"
    ]
    if (
        not quiesce
        or quiesce[-1].get("status") != "SUCCEEDED"
        or not verify
        or verify[-1].get("status") != "WAITING"
        or any(
            item.get("operation")
            in {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTORE_GPU_SERVICES"}
            for item in executions
        )
    ):
        raise RegionalFixtureError(
            "destructive timer requires the exact WAITING barrier"
        )
    details = quiesce[-1].get("details") or {}
    generation = (details.get("agent_generations") or {}).get(node)
    try:
        expires = datetime.fromisoformat(details["maintenance_window_expires_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RegionalFixtureError(
            "destructive timer has no pinned maintenance window"
        ) from exc
    now = datetime.now(timezone.utc)
    if (
        type(generation) is not int
        or generation < 1
        or expires.tzinfo is None
        or maintenance_window_end.tzinfo is None
        or now >= min(expires, maintenance_window_end)
    ):
        raise RegionalFixtureError(
            "destructive timer generation or deadline is invalid"
        )
    command_ids: dict[str, str] = {}
    for operation in ("QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS"):
        entries = [
            entry
            for command in state.get("commands") or []
            if command.get("workflow_request_id") == workflow["request_id"]
            for entry in [command, *(command.get("batched_steps") or [])]
            if (entry.get("step") or {}).get("operation") == operation
            and (entry.get("step") or {}).get("node_ids") == [node]
        ]
        if len(entries) != 1 or not entries[0].get("idempotency_key"):
            raise RegionalFixtureError(f"destructive timer cannot identify {operation}")
        command_ids[operation] = (
            f"{entries[0]['idempotency_key']}/{node}/agent-{generation}"
        )
    return {
        "run_id": run_id,
        "node_id": node,
        "boot_id": boot_id,
        "device": device,
        "drill_id": drill_id,
        "incident_id": incident["incident_id"],
        "workflow_request_id": workflow["request_id"],
        "fencing_token": workflow["fencing_token"],
        "agent_generation": generation,
        "command_ids": command_ids,
        "observed_at": now.isoformat(),
        "window_expires_at": expires.isoformat(),
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "waiting": True,
    }
