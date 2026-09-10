"""Pure verdict functions and constants of GF-REGIONAL-DESTR-024.

The fail-closed half of the coverage heartbeat. With the completion watcher
scaled to zero and every heartbeat and attempt observation older than the
freshness window, ``WorkloadTopologyService.resolve`` returns UNKNOWN and
``workflow_builder.compile_steps`` refuses the RESET_GPU plan with
``node workload state is UNKNOWN``. What the product then does -- and what
this module asserts -- is the shape DESTR-016 attempt 4 produced live on
2026-09-08 (memory: idle-cluster-workload-state-unknown):

* the incident's safety steps still run: ``FREEZE_EVIDENCE ->
  MARK_UNSCHEDULABLE -> QUARANTINE``; the node ends cordoned with the
  ``gpu-fault.io/`` quarantine taint;
* the workflow ends ``BLOCKED`` with ``blocked_kind=SAFETY_SETTLED`` and the
  UNKNOWN reason among ``blocked_reasons``;
* no physical operation is compiled, dispatched or executed: no remote
  command, no step execution and no Node Agent ledger row for
  QUIESCE/VERIFY/RESET/RESTORE, no quiesce state on the host.

The isolation is then lifted only through the validation-first restore
workflow (``WarmSpareLiveFixture.create_restore_workflow``); the taint is
never deleted by hand.
"""

from __future__ import annotations

import shlex
from typing import Any

from scripts.e2e.regional.remote_command_shapes import command_operations

from scripts.e2e.regional.destr023_verdicts import (
    WATCHER_DEPLOYMENT,
    WORKLOAD_STATE_UNKNOWN_REASON,
    fresh_coverage_errors,
    heartbeat_observed_at,
)

CASE_ID = "GF-REGIONAL-DESTR-024"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-023"
CONFIRMATION = "DESTR024_EXECUTE"

CONTAINMENT_OPERATIONS = ("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE")
# Anything that acts on the node's physical state. None of these may appear
# anywhere -- plan, remote command, execution, ledger -- on the BLOCKED run.
PHYSICAL_OPERATIONS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTART_NODE",
    "REPLACE_NODE",
    "RESTART_FABRIC_MANAGER",
)
EXPECTED_BLOCKED_KIND = "SAFETY_SETTLED"
EXPECTED_ACTION = "RESET_GPU"
QUARANTINE_TAINT_PREFIX = "gpu-fault.io/"
# Statuses under which a step execution or remote command counts as "the
# product tried": a FAILED row would be just as damning, so every status is.
ACTIVE_STATUSES = frozenset(
    {"PENDING", "LEASED", "WAITING", "RUNNING", "SUCCEEDED", "FAILED"}
)

# Phase budgets (seconds). The watchdog that scales the watcher back has to
# outlive every phase it guards: the wait for coverage to expire, the settle
# margin, and the blocked workflow's own run.
STALE_SETTLE_BUDGET_SECONDS = 180
WORKFLOW_BUDGET_SECONDS = 900
WATCHDOG_MARGIN_SECONDS = 300
WATCHER_GONE_BUDGET_SECONDS = 120
WATCHER_ROLLOUT_BUDGET_SECONDS = 180
HEARTBEAT_RETURN_BUDGET_SECONDS = 300
RESTORE_BUDGET_SECONDS = 900


def _operations(
    items: list[dict[str, Any]] | None, key: str = "operation"
) -> list[str]:
    return [str(item.get(key)) for item in items or []]


def _command_operation(command: dict[str, Any]) -> str:
    return str((command.get("step") or {}).get("operation"))


def blocked_workflow_errors(state: dict[str, Any], *, xid: int = 46) -> list[str]:
    errors = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    if event.get("xid") != xid:
        errors.append(f"matched event is not XID {xid}")
    if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
        errors.append("XID evidence is not backed by kmsg://")
    if decision.get("official_action") != EXPECTED_ACTION:
        errors.append(
            f"policy did not decide XID {xid} as {EXPECTED_ACTION}; the block "
            "would not be attributable to the compile gate"
        )
    if not workflow:
        errors.append("no workflow was opened for the injected event")
        return errors
    if workflow.get("status") != "BLOCKED":
        errors.append(f"workflow is {workflow.get('status')}, not BLOCKED")
    reasons = [str(item) for item in workflow.get("blocked_reasons") or []]
    if WORKLOAD_STATE_UNKNOWN_REASON not in reasons:
        errors.append(
            f"blocked_reasons lack {WORKLOAD_STATE_UNKNOWN_REASON!r}: {reasons}"
        )
    if workflow.get("blocked_kind") != EXPECTED_BLOCKED_KIND:
        errors.append(
            f"blocked_kind is {workflow.get('blocked_kind')}, not {EXPECTED_BLOCKED_KIND}"
        )
    physical = set(PHYSICAL_OPERATIONS)
    compiled = physical & set(_operations(workflow.get("official_steps")))
    if compiled:
        errors.append(f"physical steps were compiled: {sorted(compiled)}")
    completed = physical & {
        str(item) for item in workflow.get("completed_operations") or []
    }
    if completed:
        errors.append(f"physical operations completed: {sorted(completed)}")
    executed = physical & {
        str(item.get("operation"))
        for item in workflow.get("step_executions") or []
        if str(item.get("status")) in ACTIVE_STATUSES
    }
    if executed:
        errors.append(f"physical step executions exist: {sorted(executed)}")
    dispatched = physical & {
        operation
        for item in state.get("commands") or []
        for operation in command_operations(item)
    }
    if dispatched:
        errors.append(f"physical remote commands were dispatched: {sorted(dispatched)}")
    return errors


def containment_errors(state: dict[str, Any]) -> list[str]:
    """Fail-closed still contains: the safety steps ran to completion."""

    workflow = state.get("workflow") or {}
    completed = [str(item) for item in workflow.get("completed_operations") or []]
    return [
        f"safety step {operation} did not complete on the BLOCKED workflow"
        for operation in ("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE")
        if operation not in completed
    ]


def is_isolated(node: dict[str, Any]) -> bool:
    return bool(
        node.get("unschedulable")
        or node.get("ownership_annotations")
        or any(
            str(taint.get("key") or "").startswith(QUARANTINE_TAINT_PREFIX)
            for taint in node.get("taints") or []
        )
    )


def node_isolated_errors(node: dict[str, Any]) -> list[str]:
    if is_isolated(node):
        return []
    return ["target node is not isolated after the fail-closed plan"]


def host_untouched_errors(baseline: dict[str, Any], after: dict[str, Any]) -> list[str]:
    errors = []
    if len(after.get("gpu_inventory") or []) != len(
        baseline.get("gpu_inventory") or []
    ):
        errors.append("GPU inventory count changed on a run that must not reset")
    baseline_rows = {
        (item.get("command_id"), item.get("operation"))
        for item in baseline.get("ledger") or []
    }
    added = [
        str(item.get("operation"))
        for item in after.get("ledger") or []
        if (item.get("command_id"), item.get("operation")) not in baseline_rows
    ]
    physical = sorted(set(added) & set(PHYSICAL_OPERATIONS))
    if physical:
        errors.append(f"Node Agent ledger gained physical rows: {physical}")
    if after.get("quiesce_states"):
        errors.append("a GPU quiesce state exists on the host")
    for unit, before in (baseline.get("services") or {}).items():
        if before.get("ActiveState") != "active":
            continue
        current = (after.get("services") or {}).get(unit, {})
        if current.get("ActiveState") != "active":
            errors.append(f"service is no longer active: {unit}")
    return errors


def watcher_absent_errors(
    summary: dict[str, Any], pods: list[dict[str, Any]]
) -> list[str]:
    errors = []
    if int(summary.get("replicas") or 0) != 0:
        errors.append(
            f"{WATCHER_DEPLOYMENT} still has replicas={summary.get('replicas')}"
        )
    if pods:
        errors.append(
            f"{WATCHER_DEPLOYMENT} Pod(s) still present: "
            f"{sorted(str(item.get('name')) for item in pods)}"
        )
    return errors


def watchdog_script(
    kubectl_base: list[str],
    *,
    replicas: int,
    delay_seconds: int,
    deployment: str = WATCHER_DEPLOYMENT,
) -> str:
    """The detached restore: sleep, then scale the watcher back.

    Armed *before* the scale-down, in its own session, so a runner that dies
    mid-case still leaves a cluster whose coverage feed comes back.
    """

    scale = [
        *kubectl_base,
        "scale",
        f"deployment/{deployment}",
        f"--replicas={replicas}",
    ]
    return f"sleep {int(delay_seconds)}; " + shlex.join(scale)


def watchdog_delay_seconds(*, expiry_wait_seconds: int) -> int:
    return (
        int(expiry_wait_seconds)
        + STALE_SETTLE_BUDGET_SECONDS
        + WORKFLOW_BUDGET_SECONDS
        + WATCHDOG_MARGIN_SECONDS
    )


def heartbeat_recovered_errors(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """The watcher is back: a heartbeat newer than the stale one, fresh, IDLE."""

    errors = fresh_coverage_errors(after, require_stale_observations=False)
    previous = heartbeat_observed_at(before)
    current = heartbeat_observed_at(after)
    if previous is not None and (current is None or current <= previous):
        errors.append(
            f"coverage heartbeat did not advance after the watcher returned "
            f"(before={previous.isoformat()}, after={current.isoformat() if current else None})"
        )
    return errors


def restore_errors(restored: dict[str, Any], node: dict[str, Any]) -> list[str]:
    errors = []
    if restored.get("status") != "SUCCEEDED":
        errors.append(
            f"validated restore workflow is {restored.get('status')}, not SUCCEEDED"
        )
    if node.get("ready") != "True":
        errors.append("target node is not Ready after the restore")
    if node.get("unschedulable"):
        errors.append("target node is still unschedulable after the restore")
    if any(
        str(taint.get("key") or "").startswith(QUARANTINE_TAINT_PREFIX)
        for taint in node.get("taints") or []
    ):
        errors.append("target node still carries a gpu-fault taint after the restore")
    if node.get("ownership_annotations"):
        errors.append(
            "target node still carries ownership annotations after the restore"
        )
    return errors
