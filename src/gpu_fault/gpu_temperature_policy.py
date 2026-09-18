from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict

from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.models import RecoveryAction

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricsThresholds

NVIDIA_NVML_TEMPERATURE_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html"
)


class TemperatureDecision(TypedDict):
    severity: GpuHealthSeverity
    reason: str
    automatic_action: str
    threshold_value: float
    threshold_source: str
    policy_source: NotRequired[str]
    policy_reference: NotRequired[str]


def temperature_decision(
    thresholds: GpuMetricsThresholds,
    sample: GpuMetricSample,
    limits: Mapping[str, float],
) -> TemperatureDecision | None:
    """Use the same device-derived action boundaries before and after delivery."""

    name = sample.canonical_name
    if name == "gpu_temperature_c":
        slowdown = limits.get("gpu_slowdown_temperature_c")
        shutdown = limits.get("gpu_shutdown_temperature_c")
        maximum = limits.get("gpu_max_operating_temperature_c")
        critical = (
            slowdown
            or maximum
            or (
                shutdown - thresholds.gpu_temperature_shutdown_margin_c
                if shutdown is not None
                else None
            )
        )
        fallback_warning = thresholds.gpu_temperature_warning_c
        fallback_critical = thresholds.gpu_temperature_critical_c
        margin = thresholds.gpu_temperature_warning_margin_c
        label = "GPU"
    elif name == "memory_temperature_c":
        maximum = limits.get("memory_max_operating_temperature_c")
        critical = maximum
        fallback_warning = thresholds.memory_temperature_warning_c
        fallback_critical = thresholds.memory_temperature_critical_c
        margin = thresholds.memory_temperature_warning_margin_c
        label = "GPU memory"
    else:
        return None
    if critical is not None:
        warning = min(
            critical - margin,
            maximum if maximum is not None else critical,
        )
        source = "NVIDIA_DEVICE_LIMIT"
    else:
        warning, critical, source = (
            fallback_warning,
            fallback_critical,
            "CONFIGURED_FALLBACK",
        )
    severity = (
        GpuHealthSeverity.CRITICAL
        if sample.value >= critical
        else GpuHealthSeverity.WARNING
        if sample.value >= warning
        else None
    )
    if severity is None:
        return None
    result: TemperatureDecision = {
        "severity": severity,
        "reason": f"{label} temperature exceeded {source.lower()} threshold",
        "automatic_action": (
            RecoveryAction.DRAIN.value
            if severity is GpuHealthSeverity.CRITICAL
            else RecoveryAction.RUN_DIAGNOSTICS.value
        ),
        "threshold_value": (
            critical if severity is GpuHealthSeverity.CRITICAL else warning
        ),
        "threshold_source": source,
    }
    if source == "NVIDIA_DEVICE_LIMIT":
        result["policy_source"] = "SITE_NVIDIA_DEVICE_LIMIT_DERIVED"
        result["policy_reference"] = NVIDIA_NVML_TEMPERATURE_REFERENCE
    return result
