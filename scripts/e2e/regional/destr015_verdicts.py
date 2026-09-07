"""Pure verdict functions and constants of GF-REGIONAL-DESTR-015.

Split out of ``run_destr015_parallel_branch_join.py`` so the runner stays a
driver: everything here is unit-tested against synthetic control-plane, node
and CloudTrail snapshots and touches no cluster.

The case proves the happy path of the multi-node promise: two nodes of one
training job fault inside the aggregation window, land in one job DAG, are
reset in parallel on their own branches, and the job is restarted exactly
once after both branches released their nodes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Every hardware step one node branch must run, in order, exactly once.
BRANCH_OPERATIONS = (
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
)
# Job-level steps that must appear once for the whole DAG, never per branch.
SHARED_ONCE_OPERATIONS = ("STOP_WORKLOADS", "RESTART_WORKLOAD")
# A reset case that reaches any of these has left the path it exists to prove.
FORBIDDEN_OPERATIONS = frozenset(
    {
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESET_ALL_GPUS_NVSWITCHES",
        "QUARANTINE",
        "ESCALATE_SUPPORT",
        "RESTART_VM",
    }
)
# Node Agent operations both nodes must advertise before the case starts.
AGENT_OPERATIONS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
)
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
EXPECTED_XID = 46

# Wall-clock allowances (seconds) for the lifetime arithmetic. The two branches
# run side by side, so one reset-and-validate tail is paid once; the shared
# containment and the restart are paid once each. Deliberately generous so a
# passing estimate is a real safety margin.
CONTAINMENT_ALLOWANCE_SECONDS = 180
RESET_BRANCH_ALLOWANCE_SECONDS = 900
RESTART_ALLOWANCE_SECONDS = 600


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _step_node(step: dict[str, Any]) -> list[str]:
    return list(step.get("branch_node_ids") or step.get("node_ids") or [])


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _branch_executions(
    steps: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    node: str,
) -> list[dict[str, Any]]:
    result = []
    for item in executions:
        index = item.get("step_index")
        if isinstance(index, int) and 0 <= index < len(steps):
            if _step_node(steps[index]) == [node]:
                result.append(item)
    return result


def _executions_of(
    executions: list[dict[str, Any]], operation: str
) -> list[dict[str, Any]]:
    return [item for item in executions if item.get("operation") == operation]


def _interval(executions: list[dict[str, Any]]) -> tuple[datetime, datetime] | None:
    starts = [_parse(item.get("started_at")) for item in executions]
    ends = [_parse(item.get("updated_at")) for item in executions]
    starts = [item for item in starts if item is not None]
    ends = [item for item in ends if item is not None]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


# --------------------------------------------------------------------------- #
# Control-plane verdicts
# --------------------------------------------------------------------------- #
def workflow_errors(
    workflow: dict[str, Any],
    incident: dict[str, Any],
    *,
    nodes: tuple[str, str],
    expected_gpu_count: int,
    restart_budget: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"workflow status is not SUCCEEDED: {workflow.get('status')}")
    if incident.get("state") != "RECOVERED":
        errors.append(f"incident state is not RECOVERED: {incident.get('state')}")
    if not workflow.get("dag_enabled"):
        errors.append("workflow is not dag_enabled; the two faults did not form a DAG")
    if workflow.get("superseded_step_indexes"):
        errors.append(
            "a clean two-branch reset has superseded steps: "
            f"{workflow.get('superseded_step_indexes')}"
        )
    if workflow.get("branch_escalation_counts") or workflow.get("exhausted_branch_ids"):
        errors.append(
            "a branch escalated or was exhausted: "
            f"{workflow.get('branch_escalation_counts')} "
            f"{workflow.get('exhausted_branch_ids')}"
        )
    if workflow.get("workload_withdrawn_at") is not None:
        errors.append("the workflow was withdrawn; the job was stopped externally")
    if workflow.get("terminal_failure_reason"):
        errors.append(
            f"workflow carries a failure reason: {workflow.get('terminal_failure_reason')}"
        )

    steps = workflow.get("official_steps") or []
    executions = workflow.get("step_executions") or []
    operations = [step.get("operation") for step in steps]
    for operation in SHARED_ONCE_OPERATIONS:
        if operations.count(operation) != 1:
            errors.append(
                f"workflow does not contain exactly one {operation}: "
                f"{operations.count(operation)}"
            )
    forbidden = sorted(FORBIDDEN_OPERATIONS.intersection(operations))
    if forbidden:
        errors.append(f"workflow contains a node mutation beyond reset: {forbidden}")

    stop = next((s for s in steps if s.get("operation") == "STOP_WORKLOADS"), None)
    if stop is not None:
        missing = [node for node in nodes if node not in (stop.get("node_ids") or [])]
        if missing:
            errors.append(f"shared STOP_WORKLOADS does not cover {missing}")

    tails: dict[str, int | None] = {}
    for node in nodes:
        branch = _branch_executions(steps, executions, node)
        for operation in BRANCH_OPERATIONS:
            matches = _executions_of(branch, operation)
            if len(matches) != 1:
                errors.append(
                    f"{node} branch does not have exactly one {operation} execution: "
                    f"{len(matches)}"
                )
                continue
            if matches[0].get("status") != "SUCCEEDED":
                errors.append(
                    f"{node} {operation} is not SUCCEEDED: {matches[0].get('status')}"
                )
        tail = [
            index
            for index, step in enumerate(steps)
            if _step_node(step) == [node]
            and step.get("operation") == "RESTORE_SCHEDULING"
        ]
        tails[node] = tail[-1] if tail else None

    restarts = _executions_of(executions, "RESTART_WORKLOAD")
    join = next((s for s in steps if s.get("operation") == "RESTART_WORKLOAD"), None)
    if len(restarts) != 1:
        errors.append(
            f"RESTART_WORKLOAD does not have exactly one execution: {len(restarts)}"
        )
    else:
        restart = restarts[0]
        if restart.get("status") != "SUCCEEDED":
            errors.append(f"RESTART_WORKLOAD is not SUCCEEDED: {restart.get('status')}")
        details = restart.get("details") or {}
        context = details.get("notification_context") or {}
        source = details.get("source_gpu_count", context.get("source_gpu_count"))
        target = details.get("target_gpu_count", context.get("target_gpu_count"))
        if source != expected_gpu_count:
            errors.append(
                f"RESTART_WORKLOAD source GPU count is not {expected_gpu_count}: {source}"
            )
        if target != expected_gpu_count:
            errors.append(
                f"RESTART_WORKLOAD target GPU count is not {expected_gpu_count}: {target}"
            )
        restart_started = _parse(restart.get("started_at"))
        for node in nodes:
            release = _executions_of(
                _branch_executions(steps, executions, node), "RESTORE_SCHEDULING"
            )
            released = _parse(release[0].get("updated_at")) if release else None
            if (
                restart_started is not None
                and released is not None
                and released > restart_started
            ):
                errors.append(
                    f"RESTART_WORKLOAD started before {node} RESTORE_SCHEDULING finished"
                )
    if join is not None:
        depends = set(join.get("depends_on_step_indexes") or [])
        for node, tail in tails.items():
            if tail is None or tail not in depends:
                errors.append(
                    f"the join does not depend on the {node} branch tail "
                    f"(RESTORE_SCHEDULING index {tail}): {sorted(depends)}"
                )

    intervals = {
        node: _interval(_branch_executions(steps, executions, node)) for node in nodes
    }
    first, second = (intervals[node] for node in nodes)
    if first is not None and second is not None:
        overlap = min(first[1], second[1]) > max(first[0], second[0])
        if not overlap:
            errors.append(
                "the two node branches did not run in parallel: "
                f"{nodes[0]} {first[0].isoformat()}..{first[1].isoformat()}, "
                f"{nodes[1]} {second[0].isoformat()}..{second[1].isoformat()}"
            )

    count = restart_budget.get("restart_count")
    budget = restart_budget.get("budget")
    if count != 1 or budget != 1:
        errors.append(
            f"restart budget did not advance exactly once: count={count} budget={budget}"
        )
    return errors


def injection_errors(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    nodes: tuple[str, str],
) -> list[str]:
    errors: list[str] = []
    request_ids = set()
    for node, state in zip(nodes, (first, second), strict=True):
        event = state.get("event") or {}
        decision = state.get("decision") or {}
        incident = state.get("incident") or {}
        workflow = state.get("workflow") or {}
        if event.get("xid") != EXPECTED_XID:
            errors.append(f"{node} event is not XID {EXPECTED_XID}: {event.get('xid')}")
        if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
            errors.append(f"{node} XID evidence is not backed by kmsg://")
        if decision.get("official_action") != "RESET_GPU":
            errors.append(
                f"{node} policy did not resolve to RESET_GPU: "
                f"{decision.get('official_action')}"
            )
        request_id = workflow.get("request_id") or incident.get("workflow_request_id")
        if not request_id:
            errors.append(f"{node} event has no workflow")
        request_ids.add(request_id)
        if incident.get("workflow_request_id") not in {None, request_id}:
            errors.append(f"{node} incident points at another workflow")
    if len(request_ids) != 1:
        errors.append(
            f"the two events did not land in one workflow: {sorted(map(str, request_ids))}"
        )
    return errors


def event_observed_at(
    states: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """``{node: observed_at}`` of each node's stored event.

    The runner stamps ``injected_at`` when the probe exec *returns*, which is
    one kubectl round trip after the kmsg write and lands the two nodes' stamps
    seconds apart when the writes were simultaneous. The event's ``observed_at``
    is the node's own clock at collection time and is what the aggregation
    window is measured against, so the spread verdict reads that instead. A
    node whose event carries no ``observed_at`` is left out, and
    :func:`spread_errors` then reports the stamps as incomplete.
    """

    result: dict[str, str] = {}
    for node, state in states.items():
        event = state.get("event") or {}
        observed = event.get("observed_at")
        if observed:
            result[node] = str(observed)
    return result


def spread_errors(
    injected_at: dict[str, str],
    *,
    aggregation_window_seconds: int,
) -> list[str]:
    """The two faults must land inside one aggregation window, or the second
    was never a candidate for the in-window merge. ``injected_at`` is the
    per-node stamp to compare, normally :func:`event_observed_at`."""

    stamps = [_parse(value) for value in injected_at.values()]
    if len(stamps) != 2 or any(item is None for item in stamps):
        return ["injection timestamps are incomplete"]
    spread = abs((stamps[0] - stamps[1]).total_seconds())  # type: ignore[operator]
    if spread >= aggregation_window_seconds:
        return [
            f"the two injections are {spread:.1f}s apart, outside the "
            f"{aggregation_window_seconds}s aggregation window"
        ]
    return []


# --------------------------------------------------------------------------- #
# Data-plane verdicts
# --------------------------------------------------------------------------- #
def host_errors(
    hosts: dict[str, dict[str, dict[str, Any]]],
    *,
    nodes: tuple[str, str],
    expected_gpu_count: int,
) -> list[str]:
    """``hosts[node] = {"before": snapshot, "after": snapshot}`` per node."""

    errors: list[str] = []
    for node in nodes:
        before = (hosts.get(node) or {}).get("before") or {}
        after = (hosts.get(node) or {}).get("after") or {}
        if before.get("boot_id") != after.get("boot_id"):
            errors.append(f"{node} boot id changed; a reset case must not reboot")
        if len(after.get("gpu_inventory") or []) != expected_gpu_count:
            errors.append(
                f"{node} GPU inventory is not {expected_gpu_count} after reset"
            )
        baseline_rows = {
            (row.get("command_id"), row.get("operation"))
            for row in before.get("ledger") or []
        }
        added = [
            row
            for row in after.get("ledger") or []
            if (row.get("command_id"), row.get("operation")) not in baseline_rows
        ]
        resets = [row for row in added if row.get("operation") == "RESET_GPU"]
        if len(resets) != 1 or resets[0].get("state") != "SUCCEEDED":
            errors.append(
                f"{node} Node Agent ledger did not add exactly one successful RESET_GPU: "
                f"{[(r.get('command_id'), r.get('state')) for r in resets]}"
            )
        for operation in ("QUIESCE_GPU_SERVICES", "RESTORE_GPU_SERVICES"):
            rows = [
                row
                for row in added
                if row.get("operation") == operation and row.get("state") == "SUCCEEDED"
            ]
            if not rows:
                errors.append(f"{node} Node Agent ledger has no successful {operation}")
        if any(row.get("operation") == "RESET_ALL_GPUS_NVSWITCHES" for row in added):
            errors.append(f"{node} Node Agent ledger shows a full fabric reset")
        if len({row.get("command_id") for row in added}) != len(added):
            errors.append(f"{node} Node Agent ledger contains duplicate command IDs")
        if after.get("quiesce_states"):
            errors.append(f"{node} GPU quiesce state remains after restore")
        if after.get("gpu_fault_timers") != before.get("gpu_fault_timers"):
            errors.append(
                f"{node} gpu-fault timer inventory did not return to baseline"
            )
        for unit, state in (before.get("services") or {}).items():
            if state.get("ActiveState") != "active":
                continue
            current = (after.get("services") or {}).get(unit) or {}
            if current.get("ActiveState") != "active":
                errors.append(f"{node} service did not return active: {unit}")
    return errors


def schedulability_errors(
    snapshots: dict[str, dict[str, Any]],
    *,
    nodes: tuple[str, str],
) -> list[str]:
    errors: list[str] = []
    for node in nodes:
        snapshot = snapshots.get(node) or {}
        if snapshot.get("ready") != "True":
            errors.append(f"{node} is not Ready after the case")
        if snapshot.get("unschedulable"):
            errors.append(f"{node} is not schedulable after RESTORE_SCHEDULING")
        taints = [
            taint.get("key")
            for taint in snapshot.get("taints") or []
            if str(taint.get("key") or "").startswith("gpu-fault.io/")
        ]
        if taints:
            errors.append(f"{node} still carries a gpu-fault taint: {taints}")
        if snapshot.get("ownership_annotations"):
            errors.append(
                f"{node} still carries an ownership annotation: "
                f"{sorted(snapshot['ownership_annotations'])}"
            )
    return errors


def workload_errors(
    *,
    pods: list[dict[str, Any]],
    source_uids: set[str],
    nodes: tuple[str, str],
) -> list[str]:
    errors: list[str] = []
    if len(pods) != 2:
        errors.append(f"restarted job does not have two Pods: {len(pods)}")
    uids = {str(item.get("uid")) for item in pods}
    if uids & source_uids:
        errors.append("restarted Pod UIDs overlap the source attempt's Pod UIDs")
    placed = {str(item.get("node")) for item in pods}
    if placed != set(nodes):
        errors.append(
            f"restarted Pods are not on the two repaired nodes: {sorted(placed)}"
        )
    if any(item.get("phase") != "Running" or not item.get("ready") for item in pods):
        errors.append("a restarted Pod is not Running and Ready")
    return errors


def cloudtrail_errors(events: list[dict[str, Any]]) -> list[str]:
    if not events:
        return []
    return [
        "provider mutation appeared during a reset-only case: "
        + ", ".join(sorted({str(item.get("event_name")) for item in events}))
    ]


# --------------------------------------------------------------------------- #
# Timeline and arithmetic
# --------------------------------------------------------------------------- #
def step_transitions(
    previous: dict[str, str],
    executions: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Fold step executions into ``{index/operation#occurrence: status}``;
    return the new state and only the entries whose status changed.

    The occurrence ordinal is part of the key on purpose: a barrier step that
    parks WAITING before it succeeds keeps both rows in ``step_executions``,
    and a key of index/operation alone would see the pair flip status on every
    poll and append the same two "changes" for ever.
    """

    state = dict(previous)
    changes: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    seen: dict[str, int] = {}
    for item in executions:
        step = f"{item.get('step_index')}/{item.get('operation')}"
        occurrence = seen.get(step, 0)
        seen[step] = occurrence + 1
        key = f"{step}#{occurrence}"
        status = str(item.get("status") or "")
        if state.get(key) == status:
            continue
        state[key] = status
        changes.append(
            {
                "observed_at": now,
                "step": step,
                "occurrence": occurrence,
                "status": status,
                "error": item.get("error"),
                "started_at": item.get("started_at"),
                "updated_at": item.get("updated_at"),
            }
        )
    return state, changes


def estimated_duration_seconds() -> int:
    return (
        CONTAINMENT_ALLOWANCE_SECONDS
        + RESET_BRANCH_ALLOWANCE_SECONDS
        + RESTART_ALLOWANCE_SECONDS
    )


def lifetime_errors(
    *,
    estimated_seconds: int,
    lifetime_seconds: int | None,
) -> list[str]:
    if lifetime_seconds is None:
        return ["job workflow lifetime is unknown; cannot bound the case duration"]
    if estimated_seconds >= lifetime_seconds:
        return [
            f"estimated duration {estimated_seconds}s does not fit the "
            f"{lifetime_seconds}s job workflow lifetime"
        ]
    return []
