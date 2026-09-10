"""Pure verdict functions and constants of GF-REGIONAL-COLLECT-018.

The case proves ARCH-G1 and ARCH-G7 on a real regional deployment: a fault-layer
event the control plane rejects *after* the ingress 202 is booked where an
operator can see it (a per-path 4xx counter, a WARNING naming the request, a
``rejected-event:`` collector status, and no silent-collector page for the
node), and a kernel line that carries an ``Xid`` token but no code becomes an
``unparsed_xid_line`` finding with an operator-review notification instead of
being swallowed.

Two injections, both labelled as what they are: the rejected payload is a
synthetic API post through the node's own configured sink (never a kernel
fault), and the code-less line is a user-space write to ``/dev/kmsg``.

Every function is judged against documents the runner wrote -- store reads,
``/metrics`` text, Pod logs -- and touches no cluster.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from scripts.e2e.regional.collector_window_fixture import metric_max, metric_sum

CASE_ID = "GF-REGIONAL-COLLECT-018"
CONFIRMATION = "COLLECT018_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-017"
KERNEL_PATH = "/v1/collector-events/nvidia-kernel"
KERNEL_CHANNEL = "NVIDIA_KERNEL"
REJECTED_PREFIX = "rejected-event: HTTP 422"
REJECTION_LOG = "processor replay rejected a fault-layer event"
UNPARSED_KIND = "unparsed_xid_line"
OPERATOR_REVIEW_CATEGORY = "OPERATOR_REVIEW"
FAULT_REJECTIONS_METRIC = "gpu_fault_processor_fault_rejections_total"
COMPLETIONS_METRIC = "gpu_fault_processor_completions_by_path_status_total"
UNRESOLVED_METRIC = "gpu_fault_ingest_unresolved_fault_signals_total"
SILENT_METRIC = "gpu_fault_collector_silent_nodes"
SILENT_TOP_METRIC = "gpu_fault_collector_silent_top_node"
ERRORING_METRIC = "gpu_fault_collector_erroring_nodes"
# Operations the code-less line must never compile into: it is format drift,
# not a fault, and ARCH-G7 chose COLLECT_EVIDENCE over ESCALATE_OPERATOR so one
# drift cannot cordon a fleet.
FORBIDDEN_OPERATIONS = frozenset(
    {
        "MARK_UNSCHEDULABLE",
        "QUARANTINE_NODE",
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESET_GPU",
        "ESCALATE_SUPPORT",
    }
)
REJECTION_TIMEOUT_SECONDS = 240
# The kernel collector's all-zero health summary refreshes ``last_success_at``
# every 300 s (G4), and a success newer than the rejection is exactly what
# clears the rejected-event entry. A heartbeat landing in the first seconds of
# the poll would erase the evidence before the first read, so the runner posts
# the rejection right *after* a heartbeat when the next one is this close.
KERNEL_HEARTBEAT_INTERVAL_SECONDS = 300
KERNEL_HEARTBEAT_GUARD_SECONDS = 90
KERNEL_HEARTBEAT_BATCH_PREFIX = "kernel-health-"


def kernel_heartbeat_age_seconds(
    statuses: list[dict[str, Any]], *, now: datetime
) -> float | None:
    """Seconds since the kernel channel's last health summary, or None."""

    status = kernel_status(statuses)
    if status is None:
        return None
    if not str(status.get("batch_id") or "").startswith(KERNEL_HEARTBEAT_BATCH_PREFIX):
        return None
    observed = _stamp(status.get("observed_at"))
    if observed is None:
        return None
    return (now - observed).total_seconds()


def heartbeat_is_imminent(age_seconds: float | None) -> bool:
    return (
        age_seconds is not None
        and age_seconds
        >= KERNEL_HEARTBEAT_INTERVAL_SECONDS - KERNEL_HEARTBEAT_GUARD_SECONDS
    )


FINDING_TIMEOUT_SECONDS = 300
RECOVERY_TIMEOUT_SECONDS = 420


def _stamp(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def kernel_status(statuses: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in statuses:
        if record.get("collector") == KERNEL_CHANNEL:
            return record
    return None


def rejected_status_errors(
    statuses: list[dict[str, Any]], *, marker: str | None = None
) -> list[str]:
    """The node's kernel channel must say "data rejected", not "no data".

    ``marker`` is the value the probe put in the forbidden field; the status
    may name the field (pydantic's ``loc``) but never carry the value. The
    first two live runs matched the words "input" and "acceptance" instead
    and read the legitimate ``body.acceptance_unknown_field Extra inputs are
    not permitted`` detail as an echo -- every sample was refused until the
    kernel heartbeat cleared the entry, and the case failed on a clean row.
    """

    status = kernel_status(statuses)
    if status is None:
        return ["the node has no NVIDIA_KERNEL collector status row"]
    errors: list[str] = []
    rejected = [
        item
        for item in status.get("errors") or []
        if str(item).startswith(REJECTED_PREFIX)
    ]
    if not rejected:
        errors.append(
            f"no {REJECTED_PREFIX!r} entry on the kernel collector status: "
            f"{status.get('errors')!r}"
        )
    if not status.get("last_error_at"):
        errors.append("the rejected event did not stamp last_error_at")
    for item in rejected:
        if marker and marker in str(item):
            errors.append("the rejected-event status echoes the payload body")
    return errors


def rejection_metric_errors(before: list[str], after: list[str]) -> list[str]:
    """Both G1 counters must have moved for the kernel path's 4xx class."""

    errors: list[str] = []
    if metric_sum(after, FAULT_REJECTIONS_METRIC) <= metric_sum(
        before, FAULT_REJECTIONS_METRIC
    ):
        errors.append(f"{FAULT_REJECTIONS_METRIC} did not increase")
    where = {"path": KERNEL_PATH, "status_class": "4xx"}
    if metric_sum(after, COMPLETIONS_METRIC, where=where) <= metric_sum(
        before, COMPLETIONS_METRIC, where=where
    ):
        errors.append(
            f"{COMPLETIONS_METRIC}{{path={KERNEL_PATH},status_class=4xx}} did not "
            "increase"
        )
    return errors


def rejection_log_errors(logs: str, record_id: str) -> list[str]:
    """One WARNING names the request; the payload body stays out of the log."""

    lines = [line for line in logs.splitlines() if REJECTION_LOG in line]
    if not lines:
        return [f"no {REJECTION_LOG!r} WARNING in the control-plane logs"]
    errors: list[str] = []
    if not any("request_id=" in line and "status=422" in line for line in lines):
        errors.append("the rejection WARNING does not carry request_id and status=422")
    if any(record_id in line for line in lines):
        errors.append("the rejection WARNING echoes the rejected payload")
    return errors


def silence_errors(
    before: list[str], after: list[str], *, cluster_id: str, node: str
) -> list[str]:
    """Rejection is an error state, never a silence page for this node."""

    errors: list[str] = []
    where = {"cluster_id": cluster_id, "channel": KERNEL_CHANNEL}
    silent_before = metric_max(before, SILENT_METRIC, where=where) or 0.0
    silent_after = metric_max(after, SILENT_METRIC, where=where) or 0.0
    if silent_after > silent_before:
        errors.append(
            f"{SILENT_METRIC} for the kernel channel rose from {silent_before} to "
            f"{silent_after} after a rejection"
        )
    top = metric_max(after, SILENT_TOP_METRIC, where={**where, "node_id": node})
    if top:
        errors.append(f"{SILENT_TOP_METRIC} names {node} as silent")
    if (metric_max(after, ERRORING_METRIC, where=where) or 0.0) < 1:
        errors.append(f"{ERRORING_METRIC} for the kernel channel is not at least 1")
    return errors


def _mentions(value: Any, needle: str) -> bool:
    return needle in str(value)


def unparsed_finding_errors(
    activity: dict[str, Any],
    *,
    marker: str,
    metrics_before: list[str],
    metrics_after: list[str],
) -> list[str]:
    """The code-less line becomes a bounded finding, evidence and a notification."""

    errors: list[str] = []
    # The kind is spelled in the identifiers (`inc-kernel-log-<record>-<kind>`
    # / `finding-<signal>-<kind>`), not in the incident's prose reasons; the
    # first live run matched only the prose and saw no incident at all.
    incidents = [
        item
        for item in activity.get("incidents") or []
        if _mentions(item.get("incident_id"), UNPARSED_KIND)
        or _mentions(item.get("event_id"), UNPARSED_KIND)
        or _mentions(item.get("reasons"), UNPARSED_KIND)
        or _mentions(item.get("event_type"), UNPARSED_KIND)
    ]
    if not incidents:
        errors.append(f"no incident names {UNPARSED_KIND}")
    incident_ids = {str(item.get("incident_id")) for item in incidents}
    workflows = [
        item
        for item in activity.get("workflows") or []
        if str(item.get("incident_id")) in incident_ids
    ]
    if incidents and not workflows:
        errors.append("the unparsed-line incident opened no workflow")
    for workflow in workflows:
        operations = {
            str(step.get("operation")) for step in workflow.get("official_steps") or []
        }
        forbidden = sorted(operations & FORBIDDEN_OPERATIONS)
        if forbidden:
            errors.append(f"the unparsed-line workflow compiles {forbidden}")
        if "FREEZE_EVIDENCE" not in operations:
            errors.append("the unparsed-line workflow does not freeze evidence")
    notifications = [
        item
        for item in activity.get("notifications") or []
        if str(item.get("incident_id")) in incident_ids
        and item.get("category") == OPERATOR_REVIEW_CATEGORY
    ]
    if incidents and not notifications:
        errors.append(f"no {OPERATOR_REVIEW_CATEGORY} notification for the finding")
    evidence = [
        item
        for item in activity.get("evidence") or []
        if _mentions(item.get("payload"), marker)
    ]
    if not evidence:
        errors.append("the code-less kernel line was not kept as raw evidence")
    where = {"kind": UNPARSED_KIND}
    if metric_sum(metrics_after, UNRESOLVED_METRIC, where=where) <= metric_sum(
        metrics_before, UNRESOLVED_METRIC, where=where
    ):
        errors.append(f"{UNRESOLVED_METRIC}{{kind={UNPARSED_KIND}}} did not increase")
    return errors


def recovery_errors(statuses: list[dict[str, Any]]) -> list[str]:
    """One accepted kernel batch clears the erroring state."""

    status = kernel_status(statuses)
    if status is None:
        return ["the node has no NVIDIA_KERNEL collector status row"]
    success = _stamp(status.get("last_success_at"))
    error = _stamp(status.get("last_error_at"))
    if success is None:
        return ["the kernel channel never recorded a success after the rejection"]
    if error is not None and error >= success:
        return [
            "the kernel channel is still erroring: "
            f"last_error_at={error.isoformat()} >= last_success_at={success.isoformat()}"
        ]
    return []


def node_untouched_errors(node: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if node.get("ownership_annotations"):
        errors.append("the node carries workflow ownership after the case")
    if node.get("unschedulable"):
        errors.append("the node is unschedulable after the case")
    return errors


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"
