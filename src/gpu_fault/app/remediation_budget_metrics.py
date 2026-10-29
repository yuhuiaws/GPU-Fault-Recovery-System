"""The remediation budget's Store-derived /metrics families (design §9.5).

Read from the closed-loop workflow scan, which always includes every open
workflow, so every process renders the same values and the aggregation takes
their ``max``. Holders follow the admission rule (``budget_holds``); a waiter is
a row whose last claim the budget refused and that may still claim.

``gpu_fault_remediation_budget_longest_wait_seconds`` is how long the oldest
current wait has lasted, by the type of the scope that last refused it: from
the refusal that began the wait (its ``REMEDIATION_BUDGET_WAIT`` hold event,
``budget_wait_started_at``), so a waiter refused again and again keeps its
age, and an admission ends the wait. Waiting counts never said that: churn
keeps a count above zero for hours without one workflow waiting long, and one
workflow stuck behind its own node's or job's holder looks like any other
waiter. Only scope types with a waiter are rendered.

No family is per cluster or carries a limit: no fixed number caps a cluster's
or the region's remediations any more. Which cluster a correlated fault hits,
and how close it came to holding, is the storm breaker's: its families
(``gpu_fault.remediation_storm``) are one series per cluster that has a
breaker record -- whether it is open, the nodes its window holds (frozen at
the trip while open) against the threshold it last judged, when it tripped,
and how many workflows it holds. A ``region``, ``cluster`` or ``class`` scope
type an older release's RUNNING row still lists is counted in
``gpu_fault_remediation_budget_active_claims`` until that row ends.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from gpu_fault.models import WorkflowRequest, WorkflowStatus
from gpu_fault.remediation_storm import STORM_SCOPE_PREFIX, RemediationStormBreaker
from gpu_fault.store.shared.remediation_budgets import (
    budget_holds,
    budget_wait_started_at,
)

WAITING_STATUSES = frozenset(
    {WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING}
)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class BudgetTally:
    """What one workflow scan says about the remediation budget."""

    active_by_scope_type: Counter[str] = field(default_factory=Counter)
    waiting_by_scope_type: Counter[str] = field(default_factory=Counter)
    longest_wait_by_scope_type: dict[str, float] = field(default_factory=dict)
    storm_held_by_cluster: Counter[str] = field(default_factory=Counter)
    wait_total: int = 0
    waiting: int = 0


def budget_tally(workflows: Iterable[WorkflowRequest], now: datetime) -> BudgetTally:
    rows = list(workflows)
    tally = BudgetTally()
    holding = {holder.request_id for holder, _ in budget_holds(rows)}
    for workflow in rows:
        if workflow.request_id in holding:
            for scope in workflow.remediation_budget_claims:
                tally.active_by_scope_type[scope.split(":", 1)[0]] += 1
        tally.wait_total += workflow.remediation_budget_wait_count
        if (
            workflow.remediation_budget_last_blocked_reason is None
            or workflow.status not in WAITING_STATUSES
        ):
            continue
        tally.waiting += 1
        blocked_scope = workflow.remediation_budget_last_blocked_scope or ""
        if not blocked_scope:
            continue
        scope_type = blocked_scope.split(":", 1)[0]
        tally.waiting_by_scope_type[scope_type] += 1
        if blocked_scope.startswith(STORM_SCOPE_PREFIX):
            cluster_id = blocked_scope.removeprefix(STORM_SCOPE_PREFIX)
            tally.storm_held_by_cluster[cluster_id] += 1
        started = budget_wait_started_at(workflow)
        if started is not None:
            waited = max(0.0, (now - started).total_seconds())
            longest = tally.longest_wait_by_scope_type
            longest[scope_type] = max(longest.get(scope_type, 0.0), waited)
    return tally


def remediation_budget_lines(tally: BudgetTally) -> list[str]:
    """Render the remediation budget families from the scan tally."""

    lines = [
        "# HELP gpu_fault_remediation_budget_active_claims Active durable "
        "remediation claims by scope type.",
        "# TYPE gpu_fault_remediation_budget_active_claims gauge",
    ]
    for scope_type, count in sorted(tally.active_by_scope_type.items()):
        lines.append(
            "gpu_fault_remediation_budget_active_claims"
            f'{{scope_type="{_escape_label(scope_type)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_wait_total Workflow claim "
            "attempts held by a remediation budget.",
            "# TYPE gpu_fault_remediation_budget_wait_total gauge",
            f"gpu_fault_remediation_budget_wait_total {tally.wait_total}",
            "# HELP gpu_fault_remediation_budget_waiting_workflows Workflows "
            "currently waiting for remediation capacity.",
            "# TYPE gpu_fault_remediation_budget_waiting_workflows gauge",
            f"gpu_fault_remediation_budget_waiting_workflows {tally.waiting}",
            "# HELP gpu_fault_remediation_budget_waiting_workflows_by_scope "
            "Waiting workflows by the scope type of the budget that last "
            "refused them.",
            "# TYPE gpu_fault_remediation_budget_waiting_workflows_by_scope gauge",
        ]
    )
    for scope_type, count in sorted(tally.waiting_by_scope_type.items()):
        lines.append(
            "gpu_fault_remediation_budget_waiting_workflows_by_scope"
            f'{{scope_type="{_escape_label(scope_type)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_longest_wait_seconds How long "
            "the longest current remediation budget wait has lasted, by the "
            "scope type of the budget that last refused it, from the refusal "
            "that began the wait.",
            "# TYPE gpu_fault_remediation_budget_longest_wait_seconds gauge",
        ]
    )
    for scope_type, seconds in sorted(tally.longest_wait_by_scope_type.items()):
        lines.append(
            "gpu_fault_remediation_budget_longest_wait_seconds"
            f'{{scope_type="{_escape_label(scope_type)}"}} {seconds:.0f}'
        )
    return lines


def remediation_storm_lines(
    breakers: Iterable[RemediationStormBreaker], tally: BudgetTally
) -> list[str]:
    """The storm breaker families, one series per cluster with a breaker."""

    records = sorted(breakers, key=lambda record: record.cluster_id)
    families: tuple[tuple[str, str, Callable[[RemediationStormBreaker], float]], ...]
    families = (
        (
            "open",
            "1 while the cluster's remediation storm breaker is open (tripped, or "
            "aborted by an operator): its new destructive remediation is held "
            "until an operator releases it.",
            lambda record: int(record.open),
        ),
        (
            "window_nodes",
            "Distinct GPU nodes that entered destructive remediation in the "
            "breaker's 600-s window (frozen at the trip while open).",
            lambda record: (
                len(record.trip_node_ids)
                if record.open
                else len({entry.node_id for entry in record.window})
            ),
        ),
        (
            "threshold_nodes",
            "Nodes the breaker trips at: max(3, ceil(20% of the cluster's GPU "
            "nodes)), as last judged.",
            lambda record: record.trip_threshold if record.open else record.threshold,
        ),
        (
            "tripped_timestamp_seconds",
            "Unix time the open breaker tripped; 0 while closed.",
            lambda record: (
                record.tripped_at.timestamp()
                if record.open and record.tripped_at
                else 0
            ),
        ),
        (
            "held_workflows",
            "Workflows waiting because the cluster's storm breaker refused their "
            "first admission.",
            lambda record: tally.storm_held_by_cluster.get(record.cluster_id, 0),
        ),
    )
    lines: list[str] = []
    for name, help_text, value in families:
        metric = f"gpu_fault_remediation_budget_storm_breaker_{name}"
        lines.extend([f"# HELP {metric} {help_text}", f"# TYPE {metric} gauge"])
        lines.extend(
            f'{metric}{{cluster_id="{_escape_label(record.cluster_id)}"}} '
            f"{value(record):g}"
            for record in records
        )
    return lines
