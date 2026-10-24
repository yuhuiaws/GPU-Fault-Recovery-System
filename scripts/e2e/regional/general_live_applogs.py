"""GF-GENERAL-LIVE-017 application error logs (reviewed journal-source replay).

The node log collector reads application and kernel lines from the same real
``journalctl`` source and grades them with the shared log rules: an
out-of-memory line is a WARNING ``RUN_DIAGNOSTICS`` from any origin, while NCCL
and distributed-training errors are ``NO_ACTION`` evidence owned by the
completion watcher -- never a hardware marker or an extra diagnostic workflow
(``log_rules`` / ``log_recovery_context``). This module drives the kernel-log
replay helper in journal mode with reviewed application/OOM records, so the text
travels through the real journal source and log rules (spec 017: "NCCL text
replay proves the log path, not a real hang").

Attribution to a live/exited container, old-Pod rejection, CRI rotation across
polls and the independent low-RDMA hung trigger need a managed training attempt
with CRI Pod logs; those stay NOT_RUN with their reason -- the denominator keeps
them.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import activity
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_kernel_replay import kernel_replay, reviewed_record
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import (
    hardware_errors,
    node_isolated,
    operations,
    verdict,
    workflows_of,
)

CASE = "GF-GENERAL-LIVE-017"
ENABLE = "ENABLE_NODE_LOG_COLLECTOR"
DETECTION_BUDGET = 420
DIAGNOSTIC_WORKFLOW_OPS = frozenset(
    {"RUN_DIAGNOSTICS", "COLLECT_DIAGNOSTIC_BUNDLE", "RUN_DCGM_DIAGNOSTIC"}
)


def _log_incidents(activity_document: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            item
            for item in activity_document.get("incidents") or []
            if item.get("event_type") == "NODE_HEALTH"
            and str(item.get("event_id") or "").startswith("log-")
        ),
        key=lambda item: str(item.get("created_at")),
    )


def _drive(ctx: VariantContext, records: list[dict[str, Any]]) -> Any:
    replay = kernel_replay(ctx, mode="journal")
    replay.create()
    replay.install(records)
    opened = replay.open(approved=True, restore_seconds=ctx.spec.max_minutes * 60)
    ctx.evidence.write("replay-open", opened)
    started = utc_now()
    replay.arm()
    ctx.hold(90, name="inject")
    oracle = replay.read_back()
    ctx.evidence.append_oracle([{"injected": oracle.get("injected")}], source="applog")
    if not (oracle.get("injected") or {}).get("emission_counts"):
        raise VariantInconclusive("the journal wrapper reported no application records ingested")
    return started


def _wait_incident(ctx: VariantContext, since: Any, name: str) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, kinds=("NODE_LOGS",))
        return last if _log_incidents(last) else None

    value = ctx.wait_until(
        accept, timeout_seconds=DETECTION_BUDGET, poll_seconds=15, name=name
    )
    return value if value is not None else last


def cgroup_oom(ctx: VariantContext) -> dict[str, Any]:
    """A reviewed OOM-kill line is a diagnostic, never isolation or a STOP."""

    message = "kernel: Out of memory: Killed process 44021 (python) total-vm:..., anon-rss:..."
    started = _drive(ctx, [reviewed_record(message, transport="kernel", priority=3)])
    observed = _wait_incident(ctx, started, "oom-detect")
    incidents = _log_incidents(observed)
    stages = {
        "diagnostic": []
        if any(item.get("effective_action") == "RUN_DIAGNOSTICS" for item in incidents)
        else ["an OOM line did not produce a RUN_DIAGNOSTICS finding"],
        "no_isolation": ["node isolated on an OOM line"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
        "no_stop": _no_stop(observed, incidents),
        "hardware": hardware_errors(observed),
    }
    return _result(ctx, stages, notes=["out-of-memory is a WARNING diagnostic; no STOP without attribution"])


def _evidence_only_variant(ctx: VariantContext, message: str, note: str) -> dict[str, Any]:
    started = _drive(ctx, [reviewed_record(message, transport="stdout", unit="train-app.service")])
    observed = _wait_incident(ctx, started, "evidence-detect")
    incidents = _log_incidents(observed)
    stages = {
        "no_hardware_marker": hardware_errors(observed),
        "no_quarantine": [
            f"an application error quarantined the node: {item['incident_id']}"
            for item in incidents
            if item.get("effective_action") == "QUARANTINE"
        ],
        "no_diagnostic_workflow": _no_diagnostic_workflow(observed, incidents),
        "not_isolated": ["node isolated by an application error"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
    }
    return _result(ctx, stages, notes=[note])


def nccl_error(ctx: VariantContext) -> dict[str, Any]:
    return _evidence_only_variant(
        ctx,
        "test-app: NCCL WARN Bootstrap : unhandled error, aborting",
        "a plain NCCL error is evidence-only (completion watcher), not a hardware fault",
    )


def distributed_error(ctx: VariantContext) -> dict[str, Any]:
    return _evidence_only_variant(
        ctx,
        "test-app: torch.distributed.DistBackendError: collective operation timeout",
        "a distributed-training error is evidence-only, not a hardware fault",
    )


def _no_stop(activity_document: dict[str, Any], incidents: list[dict[str, Any]]) -> list[str]:
    errors = []
    for incident in incidents:
        for workflow in workflows_of(activity_document, str(incident["incident_id"])):
            if "STOP_WORKLOADS" in operations(workflow):
                errors.append(
                    f"workflow {workflow.get('request_id')} stopped a workload with no attribution"
                )
    return errors


def _no_diagnostic_workflow(
    activity_document: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[str]:
    errors = []
    for incident in incidents:
        for workflow in workflows_of(activity_document, str(incident["incident_id"])):
            found = sorted(set(operations(workflow)) & DIAGNOSTIC_WORKFLOW_OPS)
            if found:
                errors.append(
                    f"an evidence-only application error ran a diagnostic workflow {found}"
                )
    return errors


def _result(
    ctx: VariantContext,
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        "proof_scope": "live-replay",
        **(extra or {}),
    }


def _spec(variant: str, injection: str, expected: str, execute: Any) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="close the journalctl wrapper, disable the node log collector, confirm clean",
        max_minutes=16,
        approvals=(ENABLE,),
        execute=execute,
    )


_ATTEMPT_FIXTURE = (
    "FIXTURE_NOT_READY: needs a managed training attempt with CRI Pod logs under "
    "/var/log/pods and attempt observations in the store so the log is attributed to a "
    "live/exited container; not built in this lane"
)

VARIANTS = [
    _spec(
        "cgroup_oom",
        "reviewed kernel-transport OOM-kill line via the node log collector journal source",
        "WARNING RUN_DIAGNOSTICS diagnostic; no isolation, no STOP without attribution, no hardware op",
        cgroup_oom,
    ),
    _spec(
        "nccl_error",
        "reviewed application 'NCCL WARN ... unhandled error' line (journal source)",
        "NO_ACTION evidence (completion watcher); no diagnostic workflow, no hardware marker",
        nccl_error,
    ),
    _spec(
        "distributed_error",
        "reviewed application 'torch.distributed ... timeout' line (journal source)",
        "NO_ACTION evidence; no diagnostic workflow, no hardware marker",
        distributed_error,
    ),
    deferred_variant(
        CASE,
        "live_pid",
        proof_mode=LIVE_REPLAY,
        reason=_ATTEMPT_FIXTURE,
        lanes=("eks", "hp"),
        expected="a log from a live container is attributed LIVE to its Pod UID/PID",
    ),
    deferred_variant(
        CASE,
        "exited_pid",
        proof_mode=LIVE_REPLAY,
        reason=_ATTEMPT_FIXTURE,
        lanes=("eks", "hp"),
        expected="a log from an exited container is EXITED; no stack capture is attempted",
    ),
    deferred_variant(
        CASE,
        "old_pod",
        proof_mode=LIVE_REPLAY,
        reason=_ATTEMPT_FIXTURE,
        lanes=("eks", "hp"),
        expected="a log from a prior Pod is not attributed to the new attempt",
    ),
    deferred_variant(
        CASE,
        "cri_partial_rotation",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: needs a CRI Pod log that rotates and splits a P/F record "
            "across collector polls; not built in this lane"
        ),
        lanes=("eks", "hp"),
        expected="a CRI partial (P) then full (F) record stitches across polls, file rotation and a delivery failure",
    ),
    deferred_variant(
        CASE,
        "missing_flight_recorder",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY/SCALE: the independent low-RDMA hung trigger needs a running "
            "managed NCCL attempt (lane N territory), not a log replay"
        ),
        lanes=("eks", "hp"),
        expected="a live, non-exited process with sustained low RDMA raises the independent hung trigger",
    ),
]
