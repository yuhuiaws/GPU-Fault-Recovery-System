from __future__ import annotations

from gpu_fault.app.runtime import AppRuntime


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
            "_total Remote commands failed by an executor-side defect.",
            "# TYPE gpu_fault_remote_command_executor_internal_errors_total counter",
            "gpu_fault_remote_command_executor_internal_errors_total "
            f"{remote['executor_internal_error_total']}",
            "# HELP gpu_fault_remote_command_unclaimed_expired "
            "Remote commands dead-lettered because no executor "
            "claimed them before the deadline.",
            "# TYPE gpu_fault_remote_command_unclaimed_expired gauge",
            "gpu_fault_remote_command_unclaimed_expired "
            f"{remote['unclaimed_expired_total']}",
        ]
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
    return [
        "# HELP gpu_fault_ambiguous_attempt_ownership_total "
        "Events rejected because more than one active attempt owned "
        "the target node.",
        "# TYPE gpu_fault_ambiguous_attempt_ownership_total counter",
        "gpu_fault_ambiguous_attempt_ownership_total "
        f"{evidence.ambiguous_attempt_ownership_total()}",
    ]
