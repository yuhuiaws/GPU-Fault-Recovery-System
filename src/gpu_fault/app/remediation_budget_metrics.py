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
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.models import WorkflowRequest, WorkflowStatus
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
    active_by_cluster: Counter[str] = field(default_factory=Counter)
    waiting_by_scope_type: Counter[str] = field(default_factory=Counter)
    waiting_by_cluster: Counter[str] = field(default_factory=Counter)
    longest_wait_by_scope_type: dict[str, float] = field(default_factory=dict)
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
                if scope.startswith("cluster:"):
                    tally.active_by_cluster[scope.removeprefix("cluster:")] += 1
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
        if blocked_scope.startswith("cluster:"):
            tally.waiting_by_cluster[blocked_scope.removeprefix("cluster:")] += 1
        started = budget_wait_started_at(workflow)
        if started is not None:
            waited = max(0.0, (now - started).total_seconds())
            longest = tally.longest_wait_by_scope_type
            longest[scope_type] = max(longest.get(scope_type, 0.0), waited)
    return tally


def remediation_budget_lines(
    tally: BudgetTally, *, config: ProductionExecutorConfig | None
) -> list[str]:
    """Render the remediation budget families from the scan tally.

    The per-cluster families exist because the scope-type split cannot say
    *which* cluster is saturated (S1): a correlated whole-cluster fault drains
    at ``cluster_limit`` concurrent remediations, and an operator needs the
    active/limit pair and the waiting count for that cluster to estimate the
    drain time. Labels are the cluster id only -- never a node -- so the
    cardinality is bounded by the fleet, not by the fault.

    The limit gauge is omitted, not zeroed, when there is no executor policy
    to read: a fabricated zero would make the saturation alert's ``limit > 0``
    guard read as "never saturated".
    """

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
            "# HELP gpu_fault_remediation_budget_cluster_active_claims Workflows "
            "holding this cluster's remediation budget claim: RUNNING (any lease) "
            "or reserved for a preempting successor that has not claimed yet.",
            "# TYPE gpu_fault_remediation_budget_cluster_active_claims gauge",
        ]
    )
    for cluster_id, count in sorted(tally.active_by_cluster.items()):
        lines.append(
            "gpu_fault_remediation_budget_cluster_active_claims"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {count}'
        )
    if config is not None:
        lines.extend(
            [
                "# HELP gpu_fault_remediation_budget_cluster_limit Configured "
                "per-cluster remediation concurrency limit "
                "(GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER); a safety "
                "limit, the same for every cluster.",
                "# TYPE gpu_fault_remediation_budget_cluster_limit gauge",
                "gpu_fault_remediation_budget_cluster_limit "
                f"{config.remediation_budget.cluster_limit}",
                "# HELP gpu_fault_remediation_budget_region_limit Configured "
                "region-wide remediation concurrency limit "
                "(GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION); compare with "
                'gpu_fault_remediation_budget_active_claims{scope_type="region"}.',
                "# TYPE gpu_fault_remediation_budget_region_limit gauge",
                "gpu_fault_remediation_budget_region_limit "
                f"{config.remediation_budget.region_limit}",
            ]
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
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_cluster_waiting_workflows "
            "Workflows waiting because this cluster's remediation budget was "
            "full when they last tried to claim.",
            "# TYPE gpu_fault_remediation_budget_cluster_waiting_workflows gauge",
        ]
    )
    for cluster_id, count in sorted(tally.waiting_by_cluster.items()):
        lines.append(
            "gpu_fault_remediation_budget_cluster_waiting_workflows"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {count}'
        )
    return lines
