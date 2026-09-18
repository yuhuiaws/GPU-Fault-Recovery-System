"""Bind destructive drill timers to an observed, still-waiting workflow.

Two proofs live here, one per direction the evidence travels:

* ``conditional_pre_authorization`` is what the runner hands the on-node holder
  *before* the fault is injected. ``QUIESCE_GPU_SERVICES`` stops kubelet, and
  with it the ``kubectl exec`` channel every host probe answers over, so no
  exec can reach the node between the quiesce and the restore. The holder is
  therefore authorized in advance, conditionally: it may act only once the
  Node Agent ledger it reads locally shows this run's workflow parked at the
  client-verification barrier, and only inside the bounds pinned here.
* ``barrier_authorization`` is the runner's own store-side observation of the
  barrier, taken through the control plane while the node cannot be reached.
  It is never delivered to the node any more; it is evidence, and
  ``fired_after_barrier_errors`` grades the host's fire record against it
  after the node is back: the host must have fired *after* the runner saw the
  barrier, on the same workflow, incident and boot.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError

CONDITIONAL_KIND = "conditional-barrier-pre-authorization"
# The ledger rows the holder must see before it may act, spelled once here and
# once in each probe; a probe refuses a proof whose shape differs from its own.
LEDGER_SHAPE: dict[str, Any] = {
    "quiesce": "QUIESCE_GPU_SERVICES",
    "verify": "VERIFY_NO_GPU_CLIENTS",
    # ``clients.py`` raises "GPU compute clients are still active: ..." or
    # "GPU device clients are still active: ..."; the barrier folds either into
    # the WAITING reason by this same substring.
    "verify_refusal": "clients are still active",
    "forbidden": ["RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTORE_GPU_SERVICES"],
}
MIN_PRE_AUTHORIZATION_SECONDS = 60
MAX_PRE_AUTHORIZATION_SECONDS = 3600
MIN_MAINTENANCE_WINDOW_SECONDS = 30
MAX_MAINTENANCE_WINDOW_SECONDS = 3600
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_DEVICE = re.compile(r"^/dev/nvidia(?:[0-9]|1[0-5])$")

# The executor's own bounds: the node workflow lifetime, the per-step waiting
# caps and the agent maintenance window the barrier pins at quiesce time. Read
# from the running executor because that is the only reading that cannot drift
# from what will judge the case.
EXECUTOR_BOUNDS_PROBE = r"""
import json
import os

from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.models import WorkflowOperation

config = ProductionExecutorConfig.from_environment()
print(json.dumps({
    "node_workflow_lifetime_seconds": config.node_workflow_lifetime_seconds,
    "step_waiting_timeout_seconds": config.step_waiting_timeout_seconds,
    "verify_waiting_limit_seconds": config.step_waiting_limit(
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS
    ),
    "restore_waiting_limit_seconds": config.step_waiting_limit(
        WorkflowOperation.RESTORE_GPU_SERVICES
    ),
    "agent_maintenance_window_seconds": int(
        os.getenv("GPU_FAULT_AGENT_MAINTENANCE_WINDOW_SECONDS", "420")
    ),
    "gpu_client_verify_max_attempts": int(
        os.getenv("GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS", "60")
    ),
}, sort_keys=True))
"""


def executor_bounds(regional: Any) -> dict[str, int]:
    """The deployed executor bounds, every value an ``int`` or a refusal."""

    bounds = regional.executor_python(EXECUTOR_BOUNDS_PROBE)
    result: dict[str, int] = {}
    for key in (
        "node_workflow_lifetime_seconds",
        "step_waiting_timeout_seconds",
        "verify_waiting_limit_seconds",
        "restore_waiting_limit_seconds",
        "agent_maintenance_window_seconds",
        "gpu_client_verify_max_attempts",
    ):
        try:
            result[key] = int(bounds[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RegionalFixtureError(f"executor bounds lack {key}") from exc
    return result


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def conditional_pre_authorization(
    *,
    run_id: str,
    node: str,
    boot_id: str,
    device: str,
    drill_id: str,
    marker: str,
    maintenance_window_end: datetime,
    maintenance_window_seconds: int,
    valid_for_seconds: int,
    not_before_seconds: dict[str, int],
    now: datetime | None = None,
) -> dict[str, Any]:
    """The proof the runner delivers to the holder before the injection.

    Nothing in it says the barrier exists; it says which barrier the holder may
    act on and until when. The holder evaluates the condition itself, against
    the ledger on the node, every few seconds after its timer fires.
    """

    current = now or datetime.now(timezone.utc)
    for label, value in (
        ("run ID", run_id),
        ("node", node),
        ("drill ID", drill_id),
        ("marker", marker),
    ):
        if SAFE_ID.fullmatch(value or "") is None:
            raise RegionalFixtureError(f"pre-authorization has an unsafe {label}")
    if SAFE_DEVICE.fullmatch(device or "") is None:
        raise RegionalFixtureError("pre-authorization device is not a GPU device node")
    if not boot_id:
        raise RegionalFixtureError("pre-authorization has no baseline boot id")
    if maintenance_window_end.tzinfo is None or current >= maintenance_window_end:
        raise RegionalFixtureError(
            "maintenance window ended or is naive before the pre-authorization"
        )
    if not (
        MIN_MAINTENANCE_WINDOW_SECONDS
        <= int(maintenance_window_seconds)
        <= MAX_MAINTENANCE_WINDOW_SECONDS
    ):
        raise RegionalFixtureError(
            "pre-authorization agent maintenance window is out of bounds"
        )
    if (
        not MIN_PRE_AUTHORIZATION_SECONDS
        <= int(valid_for_seconds)
        <= MAX_PRE_AUTHORIZATION_SECONDS
    ):
        raise RegionalFixtureError("pre-authorization validity is out of bounds")
    if not not_before_seconds:
        raise RegionalFixtureError("pre-authorization names no phase")
    for phase, delay in not_before_seconds.items():
        if SAFE_ID.fullmatch(phase or "") is None:
            raise RegionalFixtureError("pre-authorization phase name is unsafe")
        if type(delay) is not int or delay < 0 or delay >= int(valid_for_seconds):
            raise RegionalFixtureError(
                f"pre-authorization not-before delay for {phase} is outside its validity"
            )
    expires_at = current + timedelta(seconds=int(valid_for_seconds))
    return {
        "kind": CONDITIONAL_KIND,
        "conditional": True,
        "run_id": run_id,
        "node_id": node,
        "boot_id": boot_id,
        "device": device,
        "drill_id": drill_id,
        "marker": marker,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "maintenance_window_seconds": int(maintenance_window_seconds),
        "authorized_at": current.isoformat(),
        "expires_at": min(expires_at, maintenance_window_end).isoformat(),
        "not_before_seconds": dict(not_before_seconds),
        "ledger": {
            "quiesce": LEDGER_SHAPE["quiesce"],
            "verify": LEDGER_SHAPE["verify"],
            "verify_refusal": LEDGER_SHAPE["verify_refusal"],
            "forbidden": list(LEDGER_SHAPE["forbidden"]),
        },
    }


def fired_after_barrier_errors(
    record: dict[str, Any] | None,
    *,
    store_proof: dict[str, Any],
    label: str,
) -> list[str]:
    """The host fired after the runner observed the barrier, on the same barrier.

    ``record`` is what the holder wrote on the node when it acted
    (``fire_requested_at`` and the ledger ``condition`` it matched); the
    ``store_proof`` is the runner's ``barrier_authorization`` taken from the
    control plane. A fire the runner cannot place after its own observation,
    or one matched to another workflow, incident or boot, is not the injection
    this case claims.
    """

    if not record:
        return [f"{label}: the host recorded no fire"]
    errors: list[str] = []
    fired = _parse(record.get("fire_requested_at"))
    observed = _parse(store_proof.get("observed_at"))
    if fired is None or observed is None:
        errors.append(f"{label}: the fire or barrier observation timestamp is missing")
    elif fired <= observed:
        errors.append(
            f"{label}: the host fired at {fired.isoformat()}, not after the runner "
            f"observed the barrier at {observed.isoformat()}"
        )
    condition = record.get("condition") or {}
    for key in ("workflow_request_id", "incident_id", "boot_id"):
        expected = store_proof.get(key)
        if not expected:
            errors.append(f"{label}: the barrier observation has no {key}")
        elif condition.get(key) != expected:
            errors.append(
                f"{label}: the host matched {key} {condition.get(key)!r}, the runner "
                f"observed {expected!r}"
            )
    return errors


def holder_disarm_decision(
    *,
    node_ready: bool,
    baseline_boot_id: str | None,
    current_boot_id: str | None,
    failsafe_deadline: datetime | None,
    now: datetime,
) -> dict[str, Any]:
    """Whether cleanup may exec into the node, and what to assume when it cannot.

    kubelet is down from the quiesce to the restore, so a holder can only be
    disarmed through the probe once the node answers again. When it never
    does, the holder is still known to be gone if the node rebooted (transient
    units die with the boot) or if its own bounded lifetime has passed.
    """

    if node_ready:
        return {"exec_allowed": True, "assume_disarmed": False, "reason": "node Ready"}
    if baseline_boot_id and current_boot_id and current_boot_id != baseline_boot_id:
        return {
            "exec_allowed": False,
            "assume_disarmed": True,
            "reason": "the node boot id changed; transient units died with the boot",
        }
    if failsafe_deadline is not None and now >= failsafe_deadline:
        return {
            "exec_allowed": False,
            "assume_disarmed": True,
            "reason": "the holder's bounded lifetime has passed",
        }
    return {
        "exec_allowed": False,
        "assume_disarmed": False,
        "reason": "the node is not Ready and the holder may still be alive",
    }


def host_reachability(
    regional: Any,
    *,
    node: str,
    baseline_boot_id: str | None,
    holder_armed_at: datetime | None,
    max_hold_seconds: int,
    budget_seconds: int,
) -> dict[str, Any]:
    """Wait for kubelet to answer again before cleanup execs into the node.

    QUIESCE_GPU_SERVICES stops kubelet, so between the quiesce and the restore
    a probe exec can only fail. Cleanup waits for the node to be Ready first
    (the DESTR-002 pattern for a node coming back from a reboot) and, when it
    never is, decides from the boot id and the holder's bounded lifetime
    whether the holder and its timers can be treated as already gone. The arm
    watcher may take up to ``max_hold_seconds`` to see the quiesce and the
    holder then lives ``max_hold_seconds`` from its own start, hence a
    failsafe of twice the hold plus slack.
    """

    ready = True
    wait_error = ""
    try:
        snapshot = regional.wait_node_ready(node, timeout_seconds=budget_seconds)
    except Exception as exc:  # noqa: BLE001 - an unreachable node is a decision input
        ready = False
        wait_error = f"{type(exc).__name__}: {exc}"
        try:
            snapshot = regional.node_snapshot(node)
        except Exception:  # noqa: BLE001 - no snapshot means no boot id
            snapshot = {}
    failsafe = None
    if holder_armed_at is not None:
        failsafe = holder_armed_at + timedelta(seconds=2 * max_hold_seconds + 120)
    decision = holder_disarm_decision(
        node_ready=ready,
        baseline_boot_id=baseline_boot_id,
        current_boot_id=snapshot.get("boot_id"),
        failsafe_deadline=failsafe,
        now=datetime.now(timezone.utc),
    )
    return {
        **decision,
        "node_ready": ready,
        "node_boot_id": snapshot.get("boot_id"),
        "wait_error": wait_error,
    }


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
