"""Separate operating limits from sustained cooling or power-delivery faults."""

from __future__ import annotations

import math
from typing import Any

from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.models import RecoveryAction

NVML_CLOCK_REFERENCE = "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html"
HW_POWER_BRAKE_MASK = 0x80
THERMAL_REASON_MASK = 0x60
CONFIRMATION_SAMPLES = 3
CONFIRMATION_MAX_GAP_SECONDS = 60
DCGM_ENVIRONMENT_ALIASES = (
    frozenset(
        {"DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"}
    ),
    frozenset({"DCGM_FI_DEV_GPU_TEMP_LIMIT", "DCGM_FI_DEV_GPU_TEMP_MARGIN_CELSIUS"}),
)


def environmental_semantics(value: str) -> bool:
    return value.startswith("CRITICAL:") or value == "NVIDIA_HW_POWER_BRAKE"


def deduplicate_dcgm_aliases(samples: list[Any]) -> list[Any]:
    by_slot: dict[tuple[Any, str], list[Any]] = {}
    for sample in samples:
        key = (
            sample.gpu_uuid or sample.pci_bdf or sample.gpu_index,
            sample.canonical_name,
        )
        by_slot.setdefault(key, []).append(sample)
    result = []
    for group in by_slot.values():
        if (
            len(group) == 2
            and any(
                {item.metric_name for item in group} == aliases
                for aliases in DCGM_ENVIRONMENT_ALIASES
            )
            and group[0].value == group[1].value
            and group[0].labels == group[1].labels
        ):
            result.append(group[0])
        else:
            result.extend(group)
    return result


def valid_clock_mask(value: float) -> bool:
    return math.isfinite(value) and 0 <= value < 2**32 and value.is_integer()


def clock_reason_decision(sample: Any) -> dict[str, Any] | None:
    if sample.canonical_name != "clock_throttle_reasons" or not valid_clock_mask(
        float(sample.value)
    ):
        return None
    mask = int(sample.value)
    if mask & HW_POWER_BRAKE_MASK:
        return {
            "severity": GpuHealthSeverity.CRITICAL,
            "reason": "hardware power brake asserted; inspect external power delivery",
            "automatic_action": RecoveryAction.DRAIN.value,
            "threshold_value": float(HW_POWER_BRAKE_MASK),
            "threshold_source": "NVIDIA_HW_POWER_BRAKE",
            "policy_source": "SITE_NVIDIA_POWER_DELIVERY",
            "policy_reference": NVML_CLOCK_REFERENCE,
            "official_action": "CONTACT_SUPPORT",
        }
    if mask & THERMAL_REASON_MASK:
        return {
            "severity": GpuHealthSeverity.WARNING,
            "reason": "thermal clock limiting; corroborate with device temperature limits",
            "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
            "threshold_value": float(THERMAL_REASON_MASK),
            "threshold_source": "NVIDIA_THROTTLE_REASON_BITMASK",
            "policy_source": "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
            "policy_reference": NVML_CLOCK_REFERENCE,
        }
    return None


def confirm_environment_decision(
    decision: dict[str, Any],
    batch: Any,
    sample: Any,
    previous: Any,
    previous_latest: Any,
    previous_completeness: Any,
    limits: dict[str, float],
) -> dict[str, Any]:
    """Persist the actual consecutive *critical* observations, not all warnings."""
    if decision["severity"] is not GpuHealthSeverity.CRITICAL:
        return decision
    prior = previous.finding if previous is not None else None
    comparable = (
        prior is not None
        and previous_latest is not None
        and previous_latest is not False
        and previous_latest.source == batch.source
        and 0
        < (batch.observed_at - prior.observed_at).total_seconds()
        <= CONFIRMATION_MAX_GAP_SECONDS
        and previous_latest.observed_at == prior.observed_at
        and prior.threshold_source == decision.get("threshold_source")
        and prior.threshold_value == decision.get("threshold_value")
        and previous_latest.sample.pci_bdf == sample.pci_bdf
        and previous_latest.sample.labels.get("boot_id") == sample.labels.get("boot_id")
        and previous_latest.sample.labels.get("source_boot_id")
        == sample.labels.get("source_boot_id")
        and not batch.collection_errors
        and not (
            previous_completeness is not None
            and previous_completeness.finding is not None
        )
    )
    count = prior.confirmation_count + 1 if comparable and prior is not None else 1
    shutdown = limits.get("gpu_shutdown_temperature_c")
    emergency = (
        sample.canonical_name == "gpu_temperature_c"
        and shutdown is not None
        and math.isfinite(shutdown)
        and shutdown > 0
        and sample.value >= shutdown
    )
    count = min(count, CONFIRMATION_SAMPLES)
    result = {**decision, "confirmation_count": count}
    if not emergency and count < CONFIRMATION_SAMPLES:
        result.update(
            severity=GpuHealthSeverity.WARNING,
            automatic_action=RecoveryAction.RUN_DIAGNOSTICS.value,
            reason=f"{decision['reason']}; confirmation {count}/{CONFIRMATION_SAMPLES}",
        )
    return result


def thermal_duration_decision(
    thresholds: Any, sample: Any, delta: Any, rate: Any
) -> dict[str, Any] | None:
    if (
        sample.canonical_name != "thermal_violation_total_us"
        or delta is None
        or delta < thresholds.thermal_violation_delta_warning_us
        or rate is None
        or not 60_000 <= rate <= 63_000_000
    ):
        return None
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": "thermal limiting duration increased; corroborate temperature and cooling",
        "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
        "threshold_value": thresholds.thermal_violation_delta_warning_us,
        "threshold_source": "SITE_DURATION_AND_DUTY_CYCLE",
        "policy_source": "SITE_GPU_THERMAL_PERFORMANCE",
        "policy_reference": NVML_CLOCK_REFERENCE,
    }
