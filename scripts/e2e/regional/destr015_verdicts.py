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

from scripts.e2e.regional.destr015_physical_evidence import (
    CLOCK_MARGIN_NS,
    MAX_CLOCK_DRIFT_PPM,
    ResetIntervalScope,
    physical_overlap_errors,
)

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
        return None
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
    for item in executions:
        if item.get("operation") not in (*BRANCH_OPERATIONS, *SHARED_ONCE_OPERATIONS):
            continue
        started = _parse(item.get("started_at"))
        ended = _parse(item.get("updated_at"))
        if started is None or ended is None or ended < started:
            errors.append(
                f"{item.get('operation')} execution has incomplete or unordered timestamps"
            )
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

    # These are dispatch timestamps. Physical overlap is separately checked
    # against the identity-bound Agent exec witnesses in the data-plane phase.
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


# --------------------------------------------------------------------------- #
# Detached witness records
# --------------------------------------------------------------------------- #
# The keys of ``final`` that complete the legacy capture ``end`` the interval
# arithmetic in ``destr015_physical_evidence`` reads.
WITNESS_END_KEYS = (
    "start_sha256",
    "trace_complete",
    "closed",
    "lost_events",
    "trace_sha256",
    "trace_bytes",
    "calibration_execs",
    "actions",
    "wall_minus_monotonic_min_ns",
    "wall_minus_monotonic_max_ns",
)
WITNESS_END_REASONS = frozenset({"finish-request", "deadline"})
EXCHANGE_KINDS = frozenset({"arm", "collect"})


def _exchanges(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in exchanges if item.get("kind") in EXCHANGE_KINDS]


def clock_binding(exchanges: list[dict[str, Any]]) -> dict[str, Any]:
    """The documented bound on ``runner_monotonic - host_monotonic``.

    Derived from the arm and collect exchanges alone: each bounds the offset
    between the runner's send and receive stamps around the host's sample, and
    the host may drift at most ``MAX_CLOCK_DRIFT_PPM`` over the span plus the
    fixed margin. This is evidence, never a substitute for the interval
    arithmetic, which recomputes the same bound and refuses when it is empty.
    """

    clocks = _exchanges(exchanges)
    keys = ("sent_ns", "received_ns", "monotonic_ns")
    if len(clocks) < 2 or any(
        type(item.get(key)) is not int for item in clocks for key in keys
    ):
        return {"bounded": False, "reason": "fewer than two integer clock exchanges"}
    span = clocks[-1]["received_ns"] - clocks[0]["sent_ns"]
    if span <= 0:
        return {"bounded": False, "reason": "controller clock is reversed"}
    drift = span * MAX_CLOCK_DRIFT_PPM // 1_000_000 + CLOCK_MARGIN_NS
    lower = max(item["sent_ns"] - item["monotonic_ns"] - drift for item in clocks)
    upper = min(item["received_ns"] - item["monotonic_ns"] + drift for item in clocks)
    if lower > upper:
        return {
            "bounded": False,
            "reason": "the exchanges admit no common clock offset",
            "span_ns": span,
            "drift_allowance_ns": drift,
        }
    return {
        "bounded": True,
        "exchanges": len(clocks),
        "span_ns": span,
        "drift_allowance_ns": drift,
        "offset_min_ns": lower,
        "offset_max_ns": upper,
        "uncertainty_ns": upper - lower,
        "max_clock_drift_ppm": MAX_CLOCK_DRIFT_PPM,
        "margin_ns": CLOCK_MARGIN_NS,
    }


def _monotonic(value: Any) -> int | None:
    stamp = (value or {}).get("monotonic_ns") if isinstance(value, dict) else None
    return stamp if type(stamp) is int else None


def _realtime(value: Any) -> int | None:
    stamp = (value or {}).get("realtime_ns") if isinstance(value, dict) else None
    return stamp if type(stamp) is int else None


def witness_record_errors(
    node: str,
    *,
    scope: ResetIntervalScope,
    receipt: dict[str, Any] | None,
    collection: dict[str, Any] | None,
) -> list[str]:
    """The witness's records were written on the host during the run.

    Same boot id on the scope, the arm receipt, both records and the
    collection; one witness id from ``armed`` to ``final``; and every record
    stamped inside the arm..collect window on the host's own monotonic clock,
    which the arm request, the unit and the collect request all share on one
    boot. A record outside that window, or from another boot, is not this run's
    evidence, however well its interval would fit.
    """

    if not isinstance(receipt, dict) or not isinstance(receipt.get("armed"), dict):
        return [f"{node} witness has no armed record"]
    if not isinstance(collection, dict):
        return [f"{node} witness was not collected"]
    armed = receipt["armed"]
    final = collection.get("final")
    if not isinstance(final, dict):
        unit = (collection.get("unit_state") or {}).get("ActiveState")
        return [
            f"{node} witness left no final record (unit {unit}); "
            "its interval evidence is lost"
        ]
    errors: list[str] = []
    if final.get("refusal"):
        errors.append(f"{node} witness refused: {final['refusal']}")
    if final.get("reason") not in WITNESS_END_REASONS:
        errors.append(
            f"{node} witness ended by {final.get('reason')}, not by collection "
            "or its own deadline"
        )
    digest = scope.digest()
    for label, record in (("armed", armed), ("final", final)):
        if record.get("scope_sha256") != digest or record.get("run_id") != scope.run_id:
            errors.append(f"{node} {label} record is not bound to this scope")
    boots = {
        receipt.get("boot_id"),
        armed.get("boot_id"),
        final.get("boot_id"),
        collection.get("boot_id"),
        (collection.get("state") or {}).get("boot_id"),
    }
    if boots != {scope.boot_id}:
        errors.append(f"{node} witness records do not all carry the scope's boot id")
    if not armed.get("witness_id") or armed.get("witness_id") != final.get(
        "witness_id"
    ):
        errors.append(f"{node} final record belongs to another witness")
    if armed.get("unit") != receipt.get("unit") or final.get("unit") != receipt.get(
        "unit"
    ):
        errors.append(f"{node} witness records name another unit")
    stamps = [
        _monotonic(receipt.get("host_clock_start")),
        _monotonic(armed),
        _monotonic(final),
        _monotonic(collection.get("host_clock")),
    ]
    if any(stamp is None for stamp in stamps) or any(
        earlier > later
        for earlier, later in zip(stamps, stamps[1:], strict=False)  # type: ignore[operator]
    ):
        errors.append(
            f"{node} witness records were not written inside the arm..collect "
            "window on the host clock"
        )
    walls = [
        _realtime(receipt.get("host_clock_start")),
        _realtime(armed),
        _realtime(final),
        _realtime(collection.get("host_clock")),
    ]
    if any(wall is None for wall in walls) or any(
        earlier - CLOCK_MARGIN_NS > later
        for earlier, later in zip(walls, walls[1:], strict=False)  # type: ignore[operator]
    ):
        errors.append(
            f"{node} witness record realtime stamps leave the arm..collect window"
        )
    return errors


def witness_capture(
    receipt: dict[str, Any],
    collection: dict[str, Any],
    exchanges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fold the durable records into the capture shape the interval arithmetic
    reads: the armed record's ``start``, the final record's closure, and the
    arm/collect exchanges as the bounded clock calibration."""

    start = dict(receipt["armed"]["start"])
    final = collection["final"]
    end = {**start, **{key: final.get(key) for key in WITNESS_END_KEYS}}
    clocks = [
        {
            "sent_ns": item["sent_ns"],
            "received_ns": item["received_ns"],
            "monotonic_ns": item["monotonic_ns"],
        }
        for item in _exchanges(exchanges)
    ]
    return {"start": start, "end": end, "clock_exchanges": clocks}


def physical_witness_errors(
    witnesses: dict[str, dict[str, Any]],
    *,
    workflow: dict[str, Any],
    hosts: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    """``witnesses[node] = {"scope", "receipt", "collection", "exchanges"}``.

    The record checks run first and alone: intervals folded from lost or
    unbound records must never reach the overlap arithmetic, because that
    arithmetic would otherwise grade an interval nobody proved was this run's.
    """

    errors: list[str] = []
    for node, item in witnesses.items():
        errors.extend(
            witness_record_errors(
                node,
                scope=item["scope"],
                receipt=item.get("receipt"),
                collection=item.get("collection"),
            )
        )
    if errors:
        errors.append(
            "the actual reset process intervals cannot be proven from lost or "
            "unbound witness records"
        )
        return errors
    captures = {
        node: witness_capture(item["receipt"], item["collection"], item["exchanges"])
        for node, item in witnesses.items()
    }
    scopes = {node: item["scope"] for node, item in witnesses.items()}
    return physical_overlap_errors(
        captures, scopes=scopes, workflow=workflow, hosts=hosts
    )
