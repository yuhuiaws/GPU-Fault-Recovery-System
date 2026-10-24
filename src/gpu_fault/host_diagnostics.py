"""Bind read-only validation to the metric and device that requested it."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from gpu_fault.host_health import NodeHealthFinding, NodeHealthPolicy
from gpu_fault.models import RecoveryAction

HOST_DIAGNOSTIC_PROBE = "host_diagnostic_probe"
HOST_DIAGNOSTIC_FAILURE = "host_diagnostic_failure"
FAILURE_PREFIX = "host_diagnostic:"


def diagnostic_probe(finding: NodeHealthFinding) -> dict[str, Any] | None:
    name = finding.metric_name
    if finding.recommended_action is not RecoveryAction.RUN_DIAGNOSTICS or not name:
        return None
    if "inventory" in name or "allocatable" in name:
        return None
    if name in NodeHealthPolicy.SUSTAINED_METRIC_RULES:
        threshold = finding.diagnostic_parameters.get("threshold_percent")
        comparison = finding.diagnostic_parameters.get("comparison")
    elif name in NodeHealthPolicy.METRIC_RULES:
        threshold = NodeHealthPolicy.METRIC_RULES[name][0]
        comparison = "gte"
    else:
        return None
    if not isinstance(threshold, (float, int)) or comparison not in {"gte", "lte"}:
        return None
    return {
        "metric": name,
        "node_id": finding.node_id,
        "device": finding.device,
        "threshold": threshold,
        "comparison": comparison,
        "observed_at": finding.observed_at.isoformat(),
    }


def validate_diagnostic_probe(
    probe: dict[str, Any], latest: list[Any]
) -> tuple[dict[str, Any] | None, str | None]:
    name = probe.get("metric")
    try:
        threshold = float(probe["threshold"])
        since = datetime.fromisoformat(probe["observed_at"])
        comparison = probe["comparison"]
        if not name or comparison not in {"gte", "lte"} or not math.isfinite(threshold):
            raise ValueError("invalid metric comparison")
        matched = [
            item
            for item in latest
            if item.name == name
            and item.device == probe.get("device")
            and item.observed_at >= since
            and math.isfinite(item.value)
        ]
    except (KeyError, TypeError, ValueError):
        return {"invalid_host_diagnostic_probe": True}, None
    if not matched:
        return {
            "missing_diagnostic_metric": [name],
            "device": probe.get("device"),
        }, None
    breached = any(
        item.value >= threshold if comparison == "gte" else item.value <= threshold
        for item in matched
    )
    return None, f"{FAILURE_PREFIX}{name}" if breached else None


def apply_diagnostic_probe(
    probe: Any,
    latest: list[Any],
    node_id: str,
    pending: dict[str, Any],
    failures: dict[str, list[str]],
) -> bool:
    if probe is None:
        return False
    if not isinstance(probe, dict) or (
        probe.get("node_id") is not None and not isinstance(probe["node_id"], str)
    ):
        pending[node_id] = {"invalid_host_diagnostic_probe": True}
        return True
    if probe.get("node_id") not in (None, node_id):
        return False
    probe_pending, failure = validate_diagnostic_probe(probe, latest)
    if probe_pending is not None:
        pending[node_id] = probe_pending
        return True
    if failure is not None:
        failures.setdefault(node_id, []).append(failure)
    return False


def diagnostic_failure_details(
    probe: Any,
    pending: dict[str, Any],
    failures: dict[str, list[str]],
) -> dict[str, bool]:
    if probe is None:
        return {}
    if pending:
        return {HOST_DIAGNOSTIC_FAILURE: True} if not failures else {}
    if failures and all(
        reason.startswith(FAILURE_PREFIX)
        or (isinstance(probe, dict) and reason == probe.get("metric"))
        for reasons in failures.values()
        for reason in reasons
    ):
        return {HOST_DIAGNOSTIC_FAILURE: True}
    return {}


def diagnostic_only_failures(executions: list[Any]) -> bool:
    return bool(executions) and all(
        execution.details.get(HOST_DIAGNOSTIC_FAILURE) is True
        for execution in executions
    )


def validate_host_metrics(
    latest: list[Any], node_id: str, failures: dict[str, list[str]]
) -> dict[str, list[str]] | None:
    names = {item.name for item in latest}
    missing = sorted(
        {"load1_per_cpu", "memory_used_percent", "filesystem_used_percent"} - names
    )
    if missing:
        return {"missing_recent_host_metrics": missing}
    thresholds = {
        "cpu_usage_percent": 98,
        "load1_per_cpu": 2,
        "memory_used_percent": 95,
        "swap_used_percent": 80,
        "filesystem_used_percent": 98,
        "shared_filesystem_unavailable": 1,
        "shared_filesystem_used_percent": 98,
        "disk_io_util_percent": 98,
        "disk_io_await_ms": 100,
        "smart_health_failed": 1,
        "network_link_down": 1,
        "bmc_critical_sensor": 1,
    }
    unhealthy = sorted(
        {
            item.name
            for item in latest
            if item.name in thresholds and item.value >= thresholds[item.name]
        }
    )
    if unhealthy:
        failures.setdefault(node_id, []).extend(unhealthy)
    return None
