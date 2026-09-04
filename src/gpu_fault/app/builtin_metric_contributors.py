from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

from gpu_fault.app.metric_scan_cache import metric_scan_cache
from gpu_fault.app.runtime import AppRuntime
from gpu_fault.collector_requirements import agent_is_current
from gpu_fault.fleet_deployment import DeploymentStatus
from gpu_fault.models import (
    NotificationStatus,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store.contracts import ControlPlaneStore


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def remote_command_metric_lines(
    runtime: AppRuntime,
) -> list[str]:
    context = runtime.context
    if not context.regional_mode:
        return []
    remote = context.store.remote_command_stats()
    lines = [
        "# HELP gpu_fault_remote_command_total Remote cluster commands by status.",
        "# TYPE gpu_fault_remote_command_total gauge",
    ]
    for status_value, count in sorted(remote["by_status"].items()):
        lines.append(
            f'gpu_fault_remote_command_total{{status="{status_value}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remote_command_oldest_unclaimed_seconds "
            "Age of the oldest PENDING remote command never claimed "
            "by a cluster executor.",
            "# TYPE gpu_fault_remote_command_oldest_unclaimed_seconds gauge",
        ]
    )
    for cluster_id, age in sorted(
        remote["oldest_unclaimed_age_seconds_by_cluster"].items()
    ):
        lines.append(
            "gpu_fault_remote_command_oldest_unclaimed_seconds"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {age:.6f}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remote_command_executor_internal_errors"
            "_total Retained remote commands failed by an executor-side defect.",
            "# TYPE gpu_fault_remote_command_executor_internal_errors_total gauge",
            "gpu_fault_remote_command_executor_internal_errors_total "
            f"{remote['executor_internal_error_total']}",
            "# HELP "
            "gpu_fault_remote_command_executor_internal_error_last_seen_"
            "timestamp_seconds Unix timestamp of the newest retained "
            "executor-side defect.",
            "# TYPE "
            "gpu_fault_remote_command_executor_internal_error_last_seen_"
            "timestamp_seconds gauge",
            "gpu_fault_remote_command_executor_internal_error_last_seen_"
            "timestamp_seconds "
            f"{remote['executor_internal_error_last_seen_timestamp_seconds']:.6f}",
            "# HELP gpu_fault_remote_command_unclaimed_expired "
            "Remote commands dead-lettered because no executor "
            "claimed them before the deadline.",
            "# TYPE gpu_fault_remote_command_unclaimed_expired gauge",
            "gpu_fault_remote_command_unclaimed_expired "
            f"{remote['unclaimed_expired_total']}",
        ]
    )
    return lines


def fleet_rollout_metric_lines(
    runtime: AppRuntime,
) -> list[str]:
    """Expose how long a fleet rollout has been fencing each cluster.

    ``execution/fleet_preflight.py`` refuses every destructive remediation for a
    cluster while any fleet deployment there is non-terminal. That is correct and
    it is also completely silent: the hold is a ``WARNING`` log line on whichever
    worker happened to poll, and nothing in ``gpu-fault-admin status`` mentions
    fleet deployments at all. A rollout abandoned in ``PLANNED`` fenced a live
    cluster for 36 hours on 2026-09-03 without producing a single signal.

    Age rather than a count, because a count cannot distinguish a rollout that is
    working from one that is stuck -- a healthy regional roll holds a deployment
    open for minutes, and that is exactly what the fence is for.
    """

    context = runtime.context
    if not context.regional_mode:
        return []
    now = datetime.now(timezone.utc)
    # Only the cluster's open deployments, which is an indexed lookup and the
    # same one every agent heartbeat already makes -- not a table scan.
    ages: dict[str, float] = {}
    planned: Counter[str] = Counter()
    for cluster_id in context.store.list_regional_cluster_ids():
        for deployment in context.store.list_active_fleet_deployments(cluster_id):
            age = max(0.0, (now - deployment.created_at).total_seconds())
            ages[cluster_id] = max(ages.get(cluster_id, 0.0), age)
            if deployment.status is DeploymentStatus.PLANNED:
                planned[cluster_id] += 1
    lines = [
        "# HELP gpu_fault_fleet_rollout_fence_age_seconds Age of the oldest "
        "non-terminal fleet deployment, which is how long destructive "
        "remediation has been fenced for the cluster.",
        "# TYPE gpu_fault_fleet_rollout_fence_age_seconds gauge",
    ]
    for cluster_id, age in sorted(ages.items()):
        lines.append(
            "gpu_fault_fleet_rollout_fence_age_seconds"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {age:.6f}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_fleet_rollout_never_started Non-terminal fleet "
            "deployments that have not started a wave, so they fence the "
            "cluster without any rollout being in flight.",
            "# TYPE gpu_fault_fleet_rollout_never_started gauge",
        ]
    )
    for cluster_id in sorted(ages):
        lines.append(
            "gpu_fault_fleet_rollout_never_started"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {planned[cluster_id]}'
        )
    return lines


def policy_metric_lines(runtime: AppRuntime) -> list[str]:
    lines = [
        "# HELP gpu_fault_policy_unknown_product_total "
        "GPU product observations that could not be mapped to a "
        "catalog product family.",
        "# TYPE gpu_fault_policy_unknown_product_total counter",
    ]
    for product, count in sorted(
        runtime.context.policy.unknown_product_counts().items()
    ):
        lines.append(
            "gpu_fault_policy_unknown_product_total"
            f'{{product="{_escape_label(product)}"}} {count}'
        )
    return lines


def orchestration_metric_lines(
    runtime: AppRuntime,
) -> list[str]:
    evidence = runtime.context.orchestrator._evidence_operations
    cache = metric_scan_cache(runtime)
    scan = cache.observation_states()
    snapshot = evidence.ownership_metric_snapshot(
        agents=cache.agents(),
        observation_states=scan.states,
    )
    lines = [
        "# HELP gpu_fault_ambiguous_attempt_ownership_total "
        "Events rejected because more than one active attempt owned "
        "the target node.",
        "# TYPE gpu_fault_ambiguous_attempt_ownership_total counter",
        "gpu_fault_ambiguous_attempt_ownership_total "
        f"{evidence.ambiguous_attempt_ownership_total()}",
        "# HELP gpu_fault_ambiguous_attempt_ownership_current "
        "Fresh active attempts currently claiming one GPU node.",
        "# TYPE gpu_fault_ambiguous_attempt_ownership_current gauge",
    ]
    for (cluster_id, node_id), count in snapshot["current"].items():
        lines.append(
            "gpu_fault_ambiguous_attempt_ownership_current"
            f'{{cluster_id="{_escape_label(cluster_id)}",'
            f'gpu_node="{_escape_label(node_id)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_stale_attempt_observations "
            "Latest active attempt observations older than the ownership "
            "freshness window for one GPU node.",
            "# TYPE gpu_fault_stale_attempt_observations gauge",
        ]
    )
    for (cluster_id, node_id), count in snapshot["stale"].items():
        lines.append(
            "gpu_fault_stale_attempt_observations"
            f'{{cluster_id="{_escape_label(cluster_id)}",'
            f'gpu_node="{_escape_label(node_id)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_attempt_observation_scan_limit Attempt "
            "observations the ownership family is allowed to read per scrape.",
            "# TYPE gpu_fault_attempt_observation_scan_limit gauge",
            f"gpu_fault_attempt_observation_scan_limit {scan.limit}",
            "# HELP gpu_fault_attempt_observation_scan_size Attempt "
            "observations actually read for the ownership family.",
            "# TYPE gpu_fault_attempt_observation_scan_size gauge",
            f"gpu_fault_attempt_observation_scan_size {len(scan.states)}",
            "# HELP gpu_fault_attempt_observation_scan_truncated Set when the "
            "attempt observation table is larger than the scan budget, so the "
            "ownership and staleness families describe only its newest slice.",
            "# TYPE gpu_fault_attempt_observation_scan_truncated gauge",
            f"gpu_fault_attempt_observation_scan_truncated {int(scan.truncated)}",
        ]
    )
    return lines


def closed_loop_metric_lines(runtime: AppRuntime) -> list[str]:
    store = runtime.context.store
    scan = metric_scan_cache(runtime).workflows()
    workflows = scan.workflows
    # The status gauge is a server-side aggregate, so it stays exact even when
    # the detail scan below only covers the newest slice of the table.
    workflow_statuses = {
        status.value: count for status, count in store.workflow_status_counts().items()
    }
    # Also a server-side aggregate, and the one the BLOCKED backlog alert reads:
    # the status gauge counts every workflow ever persisted, so its BLOCKED
    # bucket never falls just because the node came back.
    blocked_unreconciled = store.blocked_workflows_without_verified_restore()
    step_statuses: Counter[tuple[str, str]] = Counter()
    terminal_durations: dict[str, list[float]] = defaultdict(list)
    milestone_durations: dict[str, list[float]] = defaultdict(list)
    budget_scope_types: Counter[str] = Counter()
    budget_wait_total = 0
    budget_waiting = 0
    now = datetime.now(timezone.utc)
    terminal = {
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    }
    milestone_operations = {
        WorkflowOperation.MARK_UNSCHEDULABLE: "containment",
        WorkflowOperation.QUARANTINE: "containment",
        WorkflowOperation.VALIDATE_GPU: "validation",
        WorkflowOperation.VALIDATE_HOST: "validation",
        WorkflowOperation.VALIDATE_FABRIC: "validation",
        WorkflowOperation.RESTORE_SCHEDULING: "readmission",
        WorkflowOperation.RESTART_WORKLOAD: "workload_restart",
    }
    for workflow in workflows:
        if workflow.status in terminal:
            terminal_durations[workflow.status.value].append(
                max(0.0, (workflow.updated_at - workflow.created_at).total_seconds())
            )
        seen_milestones = set()
        for execution in workflow.step_executions:
            step_statuses[
                (
                    execution.operation.value,
                    execution.status.value,
                )
            ] += 1
            milestone = milestone_operations.get(execution.operation)
            if milestone is None or milestone in seen_milestones:
                continue
            if execution.status.value != "SUCCEEDED":
                continue
            seen_milestones.add(milestone)
            milestone_durations[milestone].append(
                max(
                    0.0,
                    (execution.updated_at - workflow.created_at).total_seconds(),
                )
            )
        if (
            workflow.status is WorkflowStatus.RUNNING
            and workflow.execution_lease_expires_at is not None
            and workflow.execution_lease_expires_at > now
        ):
            for scope in workflow.remediation_budget_claims:
                budget_scope_types[scope.split(":", 1)[0]] += 1
        budget_wait_total += workflow.remediation_budget_wait_count
        budget_waiting += int(
            workflow.remediation_budget_last_blocked_reason is not None
            and workflow.status
            in {
                WorkflowStatus.PENDING,
                WorkflowStatus.RUNNING,
                WorkflowStatus.SAFETY_PENDING,
            }
        )

    lines = [
        "# HELP gpu_fault_workflow_total Persisted recovery workflows by status.",
        "# TYPE gpu_fault_workflow_total gauge",
    ]
    for workflow_status in WorkflowStatus:
        lines.append(
            f'gpu_fault_workflow_total{{status="{workflow_status.value}"}} '
            f"{workflow_statuses.get(workflow_status.value, 0)}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_workflow_blocked_unreconciled BLOCKED recovery "
            "workflows with no verified restore successor, so the GPU node is "
            "still out of the training pool.",
            "# TYPE gpu_fault_workflow_blocked_unreconciled gauge",
            f"gpu_fault_workflow_blocked_unreconciled {blocked_unreconciled}",
            "# HELP gpu_fault_workflow_scan_limit Workflows the step, duration "
            "and milestone families are allowed to read per scrape.",
            "# TYPE gpu_fault_workflow_scan_limit gauge",
            f"gpu_fault_workflow_scan_limit {scan.limit}",
            "# HELP gpu_fault_workflow_scan_size Workflows actually read for "
            "the step, duration and milestone families.",
            "# TYPE gpu_fault_workflow_scan_size gauge",
            f"gpu_fault_workflow_scan_size {len(workflows)}",
            "# HELP gpu_fault_workflow_scan_truncated Set when the workflow "
            "table is larger than the scan budget, so the step, duration and "
            "milestone families describe only the newest slice.",
            "# TYPE gpu_fault_workflow_scan_truncated gauge",
            f"gpu_fault_workflow_scan_truncated {int(scan.truncated)}",
            "# HELP gpu_fault_workflow_step_total Persisted workflow step outcomes.",
            "# TYPE gpu_fault_workflow_step_total gauge",
        ]
    )
    for (operation, step_status), count in sorted(step_statuses.items()):
        lines.append(
            "gpu_fault_workflow_step_total"
            f'{{operation="{operation}",status="{step_status}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_workflow_duration_seconds Terminal workflow duration.",
            "# TYPE gpu_fault_workflow_duration_seconds summary",
        ]
    )
    for status_value, values in sorted(terminal_durations.items()):
        _append_summary(
            lines,
            "gpu_fault_workflow_duration_seconds",
            values,
            status=status_value,
        )
    lines.extend(
        [
            "# HELP gpu_fault_closed_loop_milestone_seconds Time from workflow creation to a successful closed-loop milestone.",
            "# TYPE gpu_fault_closed_loop_milestone_seconds summary",
        ]
    )
    for milestone, values in sorted(milestone_durations.items()):
        _append_summary(
            lines,
            "gpu_fault_closed_loop_milestone_seconds",
            values,
            milestone=milestone,
        )
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_active_claims Active durable remediation claims by scope type.",
            "# TYPE gpu_fault_remediation_budget_active_claims gauge",
        ]
    )
    for scope_type, count in sorted(budget_scope_types.items()):
        lines.append(
            "gpu_fault_remediation_budget_active_claims"
            f'{{scope_type="{_escape_label(scope_type)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_wait_total Workflow claim attempts held by a remediation budget.",
            "# TYPE gpu_fault_remediation_budget_wait_total gauge",
            f"gpu_fault_remediation_budget_wait_total {budget_wait_total}",
            "# HELP gpu_fault_remediation_budget_waiting_workflows Workflows currently waiting for remediation capacity.",
            "# TYPE gpu_fault_remediation_budget_waiting_workflows gauge",
            f"gpu_fault_remediation_budget_waiting_workflows {budget_waiting}",
        ]
    )

    lines.extend(_notification_lines(store))

    stale_agents = sum(
        1
        for agent in metric_scan_cache(runtime).agents()
        if getattr(agent.lifecycle_state, "value", agent.lifecycle_state) == "ACTIVE"
        and not agent_is_current(agent, observed_at=now)
    )
    lines.extend(
        [
            "# HELP gpu_fault_stale_agents Agents whose heartbeat lease is absent or expired.",
            "# TYPE gpu_fault_stale_agents gauge",
            f"gpu_fault_stale_agents {stale_agents}",
        ]
    )
    return lines


def _notification_lines(store: ControlPlaneStore) -> list[str]:
    """Render the business notification families from the Store aggregate.

    Read from ``notification_status_counts`` rather than the workflow detail
    scan, so the outbox depth stays exact on a table larger than the scan
    budget.
    """

    statuses = {
        status.value: count
        for status, count in store.notification_status_counts().items()
    }
    lines = [
        "# HELP gpu_fault_notification_total Persisted business notifications by delivery result.",
        "# TYPE gpu_fault_notification_total gauge",
    ]
    for status_value in NotificationStatus:
        lines.append(
            f'gpu_fault_notification_total{{status="{status_value.value}"}} '
            f"{statuses[status_value.value]}"
        )
    pending = sum(
        statuses[status]
        for status in {
            NotificationStatus.QUEUED.value,
            NotificationStatus.FAILED.value,
        }
    )
    lines.extend(
        [
            "# HELP gpu_fault_notification_outbox_depth Notifications without a terminal delivery result.",
            "# TYPE gpu_fault_notification_outbox_depth gauge",
            f"gpu_fault_notification_outbox_depth {pending}",
        ]
    )
    return lines


def _append_summary(
    lines: list[str],
    name: str,
    values: list[float],
    **labels: str,
) -> None:
    if not values:
        return
    rendered = ",".join(
        f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items())
    )
    suffix = f"{{{rendered}}}" if rendered else ""
    lines.extend(
        [
            f"{name}_count{suffix} {len(values)}",
            f"{name}_sum{suffix} {sum(values):.6f}",
            f"{name}_max{suffix} {max(values):.6f}",
        ]
    )
