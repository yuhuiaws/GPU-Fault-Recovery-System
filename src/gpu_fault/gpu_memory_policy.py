"""Memory telemetry is evidence, not a universal RMA verdict."""

from __future__ import annotations

import re
from typing import Any

from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.models import RecoveryAction

MEMORY_METRICS = frozenset(
    {
        "ecc_sbe_volatile_total",
        "ecc_sbe_aggregate_total",
        "ecc_dbe_volatile_total",
        "ecc_dbe_aggregate_total",
        "row_remap_failure",
        "row_remap_uncorrectable_total",
        "row_remap_correctable_total",
        "retired_pages_sbe_total",
        "retired_pages_dbe_total",
        "row_remap_pending",
        "retired_pages_pending",
    }
)
NVIDIA_GPU_MEMORY_POLICY_VERSION = "nvidia-gpu-memory-error-management/2026-07-26"
NVIDIA_ROW_REMAP_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html"
)
NVIDIA_ROW_REMAP_RMA_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/"
    "rma-policy-thresholds-for-row-remapping.html"
)


def memory_decision(
    thresholds: Any, sample: Any, delta: float | None, product: str | None
) -> dict[str, Any] | None:
    name = sample.canonical_name
    if name not in MEMORY_METRICS or sample.value < 0:
        return None
    model = re.split(
        r"[\s_-]+", (product or sample.labels.get("modelName") or "").upper()
    )
    modern = bool(
        set(model)
        & {
            "A100",
            "A30",
            "H100",
            "H200",
            "H800",
            "H20",
            "B100",
            "B200",
            "B300",
            "GB200",
            "GB300",
        }
    )
    legacy = bool(set(model) & {"P100", "V100", "V100S"})
    increments = {
        "ecc_sbe_volatile_total": thresholds.ecc_sbe_delta_warning,
        "ecc_sbe_aggregate_total": thresholds.ecc_sbe_delta_warning,
        "ecc_dbe_aggregate_total": 1,
        "row_remap_uncorrectable_total": 1,
        "row_remap_correctable_total": thresholds.row_remap_correctable_delta_warning,
        "retired_pages_sbe_total": thresholds.retired_pages_sbe_delta_warning,
        "retired_pages_dbe_total": thresholds.retired_pages_dbe_delta_critical,
    }
    threshold = increments.get(name)
    if threshold is not None:
        if delta is None or delta < threshold:
            return None
        return {
            "severity": GpuHealthSeverity.WARNING,
            "reason": f"{name} increased; inspect repair state and driver recovery action",
            "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
            "threshold_value": threshold,
            "threshold_source": "CONFIGURED_DELTA",
            "policy_source": "SITE_GPU_MEMORY_EVIDENCE",
            "policy_reference": NVIDIA_ROW_REMAP_REFERENCE,
        }
    if sample.value <= 0:
        return None
    if name == "ecc_dbe_volatile_total":
        return {
            "severity": GpuHealthSeverity.CRITICAL,
            "reason": "uncorrectable ECC observed; stop affected work and inspect recovery evidence",
            "automatic_action": RecoveryAction.QUARANTINE.value,
            "policy_source": "SITE_GPU_MEMORY_EVIDENCE",
            "policy_reference": NVIDIA_ROW_REMAP_REFERENCE,
            "official_action": "TERMINATE_JOB_AND_ANALYZE_GPU_HEALTH",
        }
    if name == "row_remap_failure":
        return {
            "severity": GpuHealthSeverity.CRITICAL,
            "reason": "row remap failure; field diagnosis required before any RMA conclusion",
            "automatic_action": (
                RecoveryAction.DRAIN if modern else RecoveryAction.QUARANTINE
            ).value,
            "policy_source": "NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT",
            "policy_version": NVIDIA_GPU_MEMORY_POLICY_VERSION,
            "policy_reference": NVIDIA_ROW_REMAP_RMA_REFERENCE,
            "official_action": (
                "RUN_FIELD_DIAGNOSTIC_FOR_RMA" if modern else "REVIEW_GPU_ARCHITECTURE"
            ),
        }
    supported = modern if name == "row_remap_pending" else legacy
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": (
            "GPU memory repair pending; reset in a service window"
            if supported
            else "memory repair telemetry requires architecture review before reset"
        ),
        "automatic_action": (
            RecoveryAction.RESET_GPU if supported else RecoveryAction.RUN_DIAGNOSTICS
        ).value,
        "policy_source": "NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT",
        "policy_version": NVIDIA_GPU_MEMORY_POLICY_VERSION,
        "policy_reference": NVIDIA_ROW_REMAP_REFERENCE,
        "official_action": (
            "RESET_GPU_IN_SERVICE_WINDOW" if supported else "REVIEW_GPU_ARCHITECTURE"
        ),
    }
