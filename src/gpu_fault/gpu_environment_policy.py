"""Separate operating limits from sustained cooling or power-delivery faults."""

from __future__ import annotations

import math
from typing import Any

from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.models import RecoveryAction

NVML_CLOCK_REFERENCE = "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html"
HW_POWER_BRAKE_MASK = 0x80
THERMAL_REASON_MASK = 0x60
#: Bit 0x08, HW Slowdown, is deliberately not graded on its own (batch 233,
#: decision 3c). nvidia-smi(1) (v610-era page), "Clocks Event Reasons":
#: :data:`NVIDIA_SMI_HW_SLOWDOWN_QUOTE` -- it is the OR of 0x40 (graded with
#: 0x20 as thermal limiting, plus field 1423's duration) and 0x80 (the brake
#: DRAIN, plus field 1424's duration). NVML API Reference (v615),
#: ``nvmlClocksThrottleReasonHwSlowdown``: :data:`NVML_HW_SLOWDOWN_QUOTE` -- the
#: only causes beyond those two are a power-draw protection and a P-state or
#: clock change artefact, and NVIDIA marks the bit for removal, so a WARNING on
#: 0x08 without 0x40/0x80 would rest on a bit with no stable meaning.
HW_SLOWDOWN_MASK = 0x08
NVIDIA_SMI_HW_SLOWDOWN_QUOTE = (
    "This option will be removed a future CUDA release. HW Slowdown is engaged, "
    "reducing the core clocks by a factor of 2 or more. It is active if either "
    "HW Thermal Slowdown or HW Power Brake are active."
)
NVML_HW_SLOWDOWN_QUOTE = (
    "HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. "
    "This is an indicator of: temperature being too high External Power Brake "
    "Assertion is triggered (e.g. by the system power supply) Power draw is too "
    "high and Fast Trigger protection is reducing the clocks May be also "
    "reported during PState or clock change This behavior may be removed in a "
    "later release."
)
#: Clock-event bits NVML v615 adds that the product does not grade: NVML API
#: Reference (v615), "Clocks Event Reasons" (:data:`NVML_CLOCK_REFERENCE`) --
#: :data:`NVML_BOARD_LIMIT_QUOTE`, :data:`NVML_RELIABILITY_QUOTE`; the values
#: from NVIDIA's ``nvidia-ml-py`` 13.615.71 (``nvmlClocksThrottleReasonBoardLimit
#: = 0x0000000000000200``, ``nvmlClocksThrottleReasonReliability =
#: 0x0000000000000400``). A policy limiting the clocks is no fault, and any bit
#: :func:`clock_reason_decision` does not name grades nothing.
NVML_UNGRADED_CLOCK_EVENT_BITS = {0x200: "BoardLimit", 0x400: "Reliability"}
NVML_BOARD_LIMIT_QUOTE = (
    "The board limit (operating) policy is currently limiting the GPU clocks."
)
NVML_RELIABILITY_QUOTE = "The reliability policy is currently limiting the GPU clocks."
CONFIRMATION_SAMPLES = 3
CONFIRMATION_MAX_GAP_SECONDS = 60
DCGM_ENVIRONMENT_ALIASES = (
    frozenset(
        {"DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"}
    ),
    frozenset({"DCGM_FI_DEV_GPU_TEMP_LIMIT", "DCGM_FI_DEV_GPU_TEMP_MARGIN_CELSIUS"}),
    # Field 159, the name before and since DCGM 4.6 (collectors/gpu/shutdown_limit).
    frozenset({"DCGM_FI_DEV_SHUTDOWN_TEMP", "DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS"}),
    # Batch 233: the 4.5.3 names the counters CSV exports and their DCGM 4.6
    # names (DCGM 4.6.0 release notes: "Existing names remain available as
    # deprecated aliases").
    frozenset(
        {"DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION", "DCGM_FI_DEV_GPU_RECOVERY_ACTION"}
    ),
    frozenset({"DCGM_FI_DEV_THRESHOLD_SRM", "DCGM_FI_DEV_SRAM_EXCEEDED"}),
    frozenset(
        {"DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_LOW", "DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW"}
    ),
    frozenset(
        {"DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_NONE", "DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE"}
    ),
    frozenset({"DCGM_FI_DEV_ECC_CURRENT", "DCGM_FI_DEV_ECC_MODE"}),
    frozenset(
        {
            "DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS",
            "DCGM_FI_DEV_PCIE_CORRECTABLE_ERROR_TOTAL",
        }
    ),
)
#: Field 1423 grades like 241 (batch 233): both are thermal limiting time, 1423
#: only the HW slowdown share of it ("Throttling due to temperature being too
#: high (reducing core clocks by a factor of 2 or more) in ns.").
THERMAL_DURATION_NAMES = frozenset(
    {"thermal_violation_total_us", "hw_thermal_slowdown_violation_total_us"}
)
#: Candidate semantics the DCGM edge filter delivers on every tick while they
#: hold, so the control plane sees consecutive samples: the confirmed
#: environmental CRITICALs, the power brake, and a PCIe correctable-error rate
#: above its limit (two consecutive windows grade it, ``gpu_pcie_link_policy``).
PCIE_CORRECTABLE_RATE_SEMANTICS = "PCIE_CORRECTABLE_RATE"


def environmental_semantics(value: str) -> bool:
    return value.startswith("CRITICAL:") or value in {
        "NVIDIA_HW_POWER_BRAKE",
        PCIE_CORRECTABLE_RATE_SEMANTICS,
    }


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
        sample.canonical_name not in THERMAL_DURATION_NAMES
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
