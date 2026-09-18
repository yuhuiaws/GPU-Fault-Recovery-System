"""Read-only event-time milestone windows over the bounded workflow scan.

The legacy summary is a retained snapshot, not a counter. These gauges instead
count each workflow's first successful milestone in (scan time - 6h, scan time].
Audit successes survive DAG rewrites; completed steps also cover older records.
Neither repeated scrapes nor inherited containment copies create observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import fsum

from gpu_fault.app.metric_scan_cache import WorkflowScan
from gpu_fault.models import (
    BlockedKind,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)

WINDOW_SECONDS = 6 * 60 * 60
MILESTONE_OPERATIONS = {
    WorkflowOperation.MARK_UNSCHEDULABLE: "containment",
    WorkflowOperation.QUARANTINE: "containment",
    WorkflowOperation.VALIDATE_GPU: "validation",
    WorkflowOperation.VALIDATE_HOST: "validation",
    WorkflowOperation.VALIDATE_FABRIC: "validation",
    WorkflowOperation.RESTORE_SCHEDULING: "readmission",
    WorkflowOperation.RESTART_WORKLOAD: "workload_restart",
}
MILESTONES = tuple(sorted(set(MILESTONE_OPERATIONS.values())))


@dataclass(frozen=True)
class MilestoneWindow:
    count: int
    mean_seconds: float | None
    complete: bool


def _aware(value: datetime | None) -> bool:
    return value is not None and value.utcoffset() is not None


def _initial_unexecuted_dag(workflow: WorkflowRequest) -> bool:
    # Health triage is born at revision 1, before any audit event or claim.
    return (
        workflow.dag_enabled
        and workflow.dag_revision == 1
        and workflow.created_at == workflow.updated_at
        and (
            workflow.status is WorkflowStatus.PENDING
            or (
                workflow.status is WorkflowStatus.BLOCKED
                and workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
            )
        )
        and workflow.execution_epoch == 0
        and workflow.merge_revision == 0
        and workflow.execution_owner_id is None
        and workflow.execution_lease_expires_at is None
        and workflow.execution_deadline is None
        and workflow.lifetime_deadline_at is None
        and workflow.pending_failure_step_index is None
        and not (
            workflow.step_executions
            or workflow.completed_operations
            or workflow.completed_step_indexes
            or workflow.inherited_step_indexes
            or workflow.superseded_step_indexes
            or workflow.branch_escalation_counts
            or workflow.exhausted_branch_ids
        )
    )


def _first_successes(
    workflow: WorkflowRequest,
) -> tuple[dict[str, datetime], set[str]]:
    first: dict[str, datetime] = {}
    unknown: set[str] = set()

    def observe(operation: WorkflowOperation | None, at: datetime) -> None:
        milestone = (
            MILESTONE_OPERATIONS.get(operation) if operation is not None else None
        )
        if milestone is None:
            return
        if not _aware(at) or at < workflow.created_at or at > workflow.updated_at:
            unknown.add(milestone)
        else:
            # A Store read can see a success committed after the scan boundary.
            # Retain its timestamp as evidence; window membership is checked later.
            first[milestone] = min(first.get(milestone, at), at)

    for event in workflow.events:
        if (
            event.kind is WorkflowEventKind.STEP_ATTEMPT
            and event.status == WorkflowStepStatus.SUCCEEDED.value
        ):
            observe(event.operation, event.at)
    inherited: set[WorkflowOperation] = set()
    for execution in workflow.step_executions:
        if (
            execution.step_index in workflow.inherited_step_indexes
            or execution.details.get("inherited_from_workflow_id")
            or execution.details.get("preemption_reuse")
        ):
            inherited.add(execution.operation)
            continue
        if execution.status is WorkflowStepStatus.SUCCEEDED:
            observe(execution.operation, execution.updated_at)
    # A completed operation without a timestamp cannot be classified as an
    # empty window. Do not invent an observation at workflow.updated_at.
    for operation in set(workflow.completed_operations) - inherited:
        milestone = MILESTONE_OPERATIONS.get(operation)
        if milestone is not None and milestone not in first:
            unknown.add(milestone)
    return first, unknown


def milestone_windows(
    scan: WorkflowScan, *, retention: timedelta | None = None
) -> dict[str, MilestoneWindow]:
    """None retention means archiving is disabled; otherwise use its actual bound.

    The supported archive path retains terminal control records for at least a
    day. Shorter retention, a short scan window, or a capped scan cannot prove a
    six-hour census. No separate observation ledger or additional Store read is
    needed. The result is anchored to the cached scan, not the scrape's clock.
    """

    observed_at = scan.observed_at
    complete = (
        _aware(observed_at)
        and not scan.truncated
        and (scan.window_seconds == 0 or scan.window_seconds >= WINDOW_SECONDS)
        and (retention is None or retention.total_seconds() >= WINDOW_SECONDS)
    )
    unknown = set() if complete else set(MILESTONES)
    first: dict[tuple[str, str], datetime] = {}
    created: dict[str, datetime] = {}
    if observed_at is not None and _aware(observed_at):
        cutoff = observed_at - timedelta(seconds=WINDOW_SECONDS)
        for workflow in scan.workflows:
            if (
                not _aware(workflow.created_at)
                or not _aware(workflow.updated_at)
                or workflow.updated_at < workflow.created_at
            ):
                unknown.update(MILESTONES)
                continue
            previous_creation = created.setdefault(
                workflow.request_id, workflow.created_at
            )
            if previous_creation != workflow.created_at:
                unknown.update(MILESTONES)
                continue
            if workflow.updated_at <= cutoff or workflow.created_at > observed_at:
                continue
            successes, invalid = _first_successes(workflow)
            unknown.update(invalid)
            history_incomplete = any(
                event.kind is WorkflowEventKind.HISTORY_TRUNCATED
                for event in workflow.events
            ) or (
                workflow.dag_revision > 0
                and not workflow.events
                and not _initial_unexecuted_dag(workflow)
            )
            for milestone in MILESTONES:
                at = successes.get(milestone)
                # An older first success proves this workflow contributes no
                # observation, even if later history was truncated.
                if history_incomplete and (at is None or at > cutoff):
                    unknown.add(milestone)
                if at is not None:
                    key = (workflow.request_id, milestone)
                    first[key] = min(first.get(key, at), at)
        values: dict[str, list[float]] = {name: [] for name in MILESTONES}
        for (request_id, milestone), at in first.items():
            if cutoff < at <= observed_at:
                values[milestone].append((at - created[request_id]).total_seconds())
    else:
        values = {name: [] for name in MILESTONES}
    return {
        milestone: MilestoneWindow(
            count=len(durations),
            mean_seconds=fsum(durations) / len(durations) if durations else None,
            complete=milestone not in unknown,
        )
        for milestone, durations in values.items()
    }


def closed_loop_window_metric_lines(
    scan: WorkflowScan, *, retention: timedelta | None = None
) -> list[str]:
    windows = milestone_windows(scan, retention=retention)
    lines = [
        "# HELP gpu_fault_closed_loop_milestone_window_mean_seconds Mean workflow "
        "creation-to-first-success duration for milestones completed in the six "
        "hours ending at the scan timestamp; NaN when empty or incomplete.",
        "# TYPE gpu_fault_closed_loop_milestone_window_mean_seconds gauge",
        "# HELP gpu_fault_closed_loop_milestone_window_count Deduplicated first "
        "milestone successes in the six-hour event-time window; 0 when empty, "
        "NaN when incomplete. Not a counter.",
        "# TYPE gpu_fault_closed_loop_milestone_window_count gauge",
        "# HELP gpu_fault_closed_loop_milestone_window_complete Whether the "
        "bounded scan, retained history and retention cover the six-hour "
        "milestone window; 0 means unknown, not healthy.",
        "# TYPE gpu_fault_closed_loop_milestone_window_complete gauge",
        "# HELP gpu_fault_closed_loop_window_end_timestamp_seconds Timestamp "
        "captured before the workflow scan; the end of its six-hour event "
        "window, unchanged on cache hits. NaN when unavailable.",
        "# TYPE gpu_fault_closed_loop_window_end_timestamp_seconds gauge",
    ]
    for milestone, window in windows.items():
        labels = f'{{milestone="{milestone}"}}'
        mean = (
            f"{window.mean_seconds:.6f}"
            if window.complete and window.mean_seconds is not None
            else "NaN"
        )
        count = str(window.count) if window.complete else "NaN"
        lines.extend(
            [
                f"gpu_fault_closed_loop_milestone_window_mean_seconds{labels} {mean}",
                f"gpu_fault_closed_loop_milestone_window_count{labels} {count}",
                f"gpu_fault_closed_loop_milestone_window_complete{labels} "
                f"{int(window.complete)}",
            ]
        )
    timestamp = (
        f"{scan.observed_at.timestamp():.6f}"
        if scan.observed_at is not None and _aware(scan.observed_at)
        else "NaN"
    )
    lines.append(f"gpu_fault_closed_loop_window_end_timestamp_seconds {timestamp}")
    return lines
