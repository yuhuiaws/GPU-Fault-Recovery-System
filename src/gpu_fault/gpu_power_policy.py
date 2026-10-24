from __future__ import annotations

import math
from typing import Any

from gpu_fault.gpu_metric_models import (
    SITE_METRIC_POLICY_VERSION,
    GpuHealthSeverity,
)
from gpu_fault.models import RecoveryAction

NVIDIA_DCGM_POLICY_VERSION = "nvidia-dcgm-health/v4.4.1"
NVIDIA_DCGM_HEALTH_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/"
    "user-guide/feature-overview.html#background-health-checks"
)


def power_violation_decision(
    thresholds: Any,
    batch: Any,
    sample: Any,
    delta: float | None,
    gpu_key: str,
    rate_per_minute: float | None = None,
) -> dict | None:
    if (
        sample.canonical_name != "power_violation_total_us"
        or delta is None
        or delta < thresholds.power_violation_delta_warning_us
        or rate_per_minute is None
        or not 60_000 <= rate_per_minute <= 63_000_000
    ):
        return None
    peers = {
        item.canonical_name: item.value
        for item in batch.samples
        if (item.gpu_uuid or item.pci_bdf or item.gpu_index or "node") == gpu_key
    }
    power = peers.get("power_usage_w")
    limit = peers.get("power_limit_w")
    utilization = peers.get("gpu_utilization_percent")
    if (
        power is None
        or limit is None
        or limit <= 0
        or power / limit < thresholds.power_limit_ratio
        or utilization is None
        or utilization < thresholds.power_correlation_min_utilization_percent
        or not all(math.isfinite(value) for value in (power, limit, utilization))
    ):
        return None
    threshold = thresholds.power_violation_delta_warning_us
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": (
            "GPU power throttling duration increased near the "
            "enforced power limit under high utilization"
        ),
        "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
        "threshold_value": threshold,
        "threshold_source": "CORRELATED_POWER_LIMIT",
        "policy_source": "SITE_GPU_POWER_PERFORMANCE",
        "policy_version": SITE_METRIC_POLICY_VERSION,
        "policy_reference": NVIDIA_DCGM_HEALTH_REFERENCE,
    }
