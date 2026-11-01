"""Grading of the DCGM fields added from the 2026-11-01 field audit (batch 233).

One entry point for ``GpuMetricsService._evaluate``: :func:`field_rule_decision`
grades the fields below; the memory ones (346, 388/389, the 393 absolute limit)
stay with ``gpu_memory_policy.memory_decision`` and field 1423 with
``gpu_environment_policy.thermal_duration_decision``. Every field is exported
by the bundled counters CSV and never required.

* 1523 GPU recovery action: ``gpu_recovery_action_policy``.
* 235-238 PCIe link downtrain (the collector's verdict) and 1501 PCIe
  correctable errors: ``gpu_pcie_link_policy``.
* 1424 HW power brake slowdown time, graded as a duty cycle -- a cumulative
  counter of nanoseconds, never "any change". NVML API Reference (v615), Field
  Value Enums: :data:`NVML_CLOCK_EVENT_COUNTERS_QUOTE`; DCGM v4.7.0 and v4.6.1
  (``sdk/nvidia/nvml/dcgm_nvml.h``) define two of the five counters as the
  cumulative perf-policy violation counters (``NVML_FI_DEV_CLOCKS_EVENT_REASON_
  SW_POWER_CAP`` = ``NVML_FI_DEV_PERF_POLICY_POWER``) and query 270/271 for the
  HW thermal / power brake ones, the numbering NVML v580 adopted ("Adjusted
  NVML_FI_DEV_CLOCKS_EVENT_REASON_* ... field value numbering"), so a driver
  580 host reads the right NVML field. The cumulative reading is an assumption
  live data could not settle (0 on every H200 at idle); it fails safe: a value
  that decreases is a counter reset and grades nothing, a duty above 105 % is
  not a duration and grades nothing. Duty > 0 is WARNING ``RUN_DIAGNOSTICS``;
  >= 1 % in two consecutive samples, or any brake time beside clock-event bit
  0x80 in the same scrape, is the existing brake verdict -- CRITICAL ``DRAIN``,
  ``NVIDIA_HW_POWER_BRAKE``, official ``CONTACT_SUPPORT`` (NVML: "HW Power Brake
  Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This
  is an indicator of: External Power Brake Assertion being triggered (e.g. by
  the system power supply)").
* 300/301 ECC mode: ECC disabled blinds every memory rule -- NVML
  ``nvmlDeviceGetMemoryErrorCounter``: :data:`NVML_ECC_REQUIRED_QUOTE` -- so
  current = 0 is WARNING ``ESCALATE_OPERATOR``; a pending mode different from
  the current one (NVML ``nvmlDeviceGetEccMode``: :data:`NVML_ECC_PENDING_QUOTE`)
  is WARNING ``ESCALATE_OPERATOR`` too: it changes at the next reboot.
* InfoROM validity, read by the collector with ``nvidia-smi --query-gpu=
  index,pci.bus_id,uuid,inforom.checksum_validation`` at the inventory cadence
  (``collectors/gpu/inforom``; DCGM field 84 reads the flash on every watch).
  ``invalid`` is WARNING ``ESCALATE_OPERATOR``, official ``FLASH_INFOROM`` (DCGM
  v4.7.0 ``DCGM_FR_CORRUPT_INFOROM_NEXT``: "Flash the InfoROM to clear this
  corruption."), and the finding says the GPU's memory-repair evidence is
  untrusted: Dynamic Page Retirement (docs 615), "Implementation":
  :data:`NVIDIA_INFOROM_RETIREMENT_QUOTE`.

Batch 235 (the 2026-11-01 Blackwell field audit) adds, through the same entry
points: the NVLink5 link-recovery counters 1211-1213 (``gpu_nvlink5_policy``;
the NVLink5 error counters are ``nvlink_<x>_error_total`` and keep the NVLink
family rule), the channel/TPC repair-pending flags of the nvidia-smi XML
(``gpu_memory_repair_policy``), and :func:`field_rule_gate`, which every
decision passes: an NVLink5 increment inside a product window grades nothing
and one an XID 144-150/155 owns is evidence; a memory-repair finding names the
GPU's XID 160/161.

Persistent advisory findings (ECC mode, InfoROM, remap headroom, the load-gated
downtrain, field 393 at DCGM's 512 limit, a pending memory repair) cannot be
re-judged by a validation
and would fail every later VALIDATE_GPU on the GPU:
:data:`ADVISORY_GPU_FINDING_NAMES` and :data:`ADVISORY_THRESHOLD_SOURCES` list
them, and a
finding whose automatic action is ``NO_ACTION`` is evidence only -- neither
opens an incident nor blocks a validation.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.gpu_environment_policy import HW_POWER_BRAKE_MASK, valid_clock_mask
from gpu_fault.gpu_memory_policy import DCGM_ROW_REMAP_LIMIT_SOURCE
from gpu_fault.gpu_memory_repair_policy import (
    MEMORY_UNREPAIRABLE,
    REPAIR_PENDING_NAMES,
    repair_pending_decision,
    with_repair_xid_evidence,
)
from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.gpu_nvlink5_policy import (
    NVLINK5_COUNTERS,
    NVLINK5_LINK_RECOVERY,
    nvlink5_gate,
    recovery_decision,
)
from gpu_fault.gpu_pcie_link_policy import (
    PCIE_CORRECTABLE_ERRORS,
    PCIE_LINK_DOWNTRAIN,
    correctable_decision,
    device_values,
    downtrain_decision,
)
from gpu_fault.gpu_recovery_action_policy import (
    GPU_RECOVERY_ACTION,
    recovery_action_decision,
)
from gpu_fault.models import RecoveryAction

HW_THERMAL_SLOWDOWN = "hw_thermal_slowdown_violation_total_us"
HW_POWER_BRAKE_SLOWDOWN = "hw_power_brake_violation_total_us"
ECC_MODE_CURRENT = "ecc_mode_current"
ECC_MODE_PENDING = "ecc_mode_pending"
INFOROM_CHECKSUM_VALID = "inforom_checksum_valid"
SRAM_THRESHOLD_EXCEEDED = "sram_threshold_exceeded"
ROW_REMAP_BANKS_LOW = "row_remap_banks_low"
ROW_REMAP_BANKS_NONE = "row_remap_banks_none"
#: The duration counters' plausibility bound (``collectors/gpu/dcgm``
#: ``DUTY_CYCLE_MAX_PLAUSIBLE``): more time than elapsed is not a duration.
DUTY_MAX_PLAUSIBLE = 1.05
POWER_BRAKE_DRAIN_DUTY = 0.01
POWER_BRAKE_CONSECUTIVE_MAX_GAP_SECONDS = 60
NVML_FIELD_VALUE_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html"
)
NVML_CLOCK_EVENT_COUNTERS_QUOTE = (
    "Field values for Clock Throttle Reason Counters All counters are in nanoseconds."
)
NVML_DEVICE_QUERIES_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html"
)
NVML_ECC_REQUIRED_QUOTE = "Requires ECC Mode to be enabled."
NVML_ECC_PENDING_QUOTE = (
    "Changing ECC modes requires a reboot. The “pending” ECC mode refers to the "
    "target mode following the next reboot."
)
NVIDIA_INFOROM_RETIREMENT_REFERENCE = (
    "https://docs.nvidia.com/deploy/dynamic-page-retirement/implementation.html"
    "#implementation"
)
NVIDIA_INFOROM_RETIREMENT_QUOTE = "These addresses are stored in the InfoROM."
NVIDIA_SMI_INFOROM_QUOTE = (
    'Inforom checksum validation ("valid", "invalid", "N/A") Only available via '
    "--query-gpu inforom.checksum_validation"
)
#: Names :func:`field_rule_decision` grades (``GpuMetricsService._evaluate``
#: skips the legacy resolvers for them).
FIELD_RULE_NAMES = frozenset(
    {
        GPU_RECOVERY_ACTION,
        PCIE_LINK_DOWNTRAIN,
        PCIE_CORRECTABLE_ERRORS,
        HW_POWER_BRAKE_SLOWDOWN,
        ECC_MODE_CURRENT,
        ECC_MODE_PENDING,
        INFOROM_CHECKSUM_VALID,
        *NVLINK5_LINK_RECOVERY,
        *REPAIR_PENDING_NAMES,
    }
)
#: The counters among the new fields: the control plane's delta and rate.
GPU_FIELD_COUNTERS = frozenset(
    {HW_THERMAL_SLOWDOWN, HW_POWER_BRAKE_SLOWDOWN, PCIE_CORRECTABLE_ERRORS}
    | NVLINK5_COUNTERS
)
#: Persistent WARNINGs a validation cannot re-judge (see the module docstring).
ADVISORY_GPU_FINDING_NAMES = frozenset(
    {
        PCIE_LINK_DOWNTRAIN,
        ECC_MODE_CURRENT,
        ECC_MODE_PENDING,
        INFOROM_CHECKSUM_VALID,
        ROW_REMAP_BANKS_NONE,
        *REPAIR_PENDING_NAMES,
    }
)
#: Persistent WARNING levels of older fields, by ``threshold_source``: field
#: 393 at DCGM's absolute limit (``gpu_memory_policy.row_remap_limit_decision``).
ADVISORY_THRESHOLD_SOURCES = frozenset({DCGM_ROW_REMAP_LIMIT_SOURCE})
#: ``host_health.NodeHealthFinding.fault_class`` for the new findings.
GPU_FIELD_FAULT_CLASSES = {
    GPU_RECOVERY_ACTION: "GPU",
    PCIE_LINK_DOWNTRAIN: "GPU",
    PCIE_CORRECTABLE_ERRORS: "GPU",
    HW_THERMAL_SLOWDOWN: "GPU_THERMAL",
    HW_POWER_BRAKE_SLOWDOWN: "GPU",
    ECC_MODE_CURRENT: "GPU_MEMORY",
    ECC_MODE_PENDING: "GPU_MEMORY",
    INFOROM_CHECKSUM_VALID: "GPU_MEMORY",
    SRAM_THRESHOLD_EXCEEDED: "GPU_MEMORY",
    ROW_REMAP_BANKS_NONE: "GPU_MEMORY",
    **{name: "GPU_FABRIC" for name in NVLINK5_COUNTERS},
    **{name: "GPU_MEMORY" for name in (*REPAIR_PENDING_NAMES, MEMORY_UNREPAIRABLE)},
}


def evidence_only(finding: Any) -> bool:
    """A GPU finding recorded as evidence: no incident, no validation failure."""

    return getattr(finding, "automatic_action", None) == RecoveryAction.NO_ACTION.value


def blocks_validation(finding: Any) -> bool:
    """Whether VALIDATE_GPU may fail on ``finding`` (see the module docstring)."""

    return not evidence_only(finding) and not (
        (
            getattr(finding, "canonical_name", None) in ADVISORY_GPU_FINDING_NAMES
            or getattr(finding, "threshold_source", None) in ADVISORY_THRESHOLD_SOURCES
        )
        and getattr(getattr(finding, "severity", None), "value", None) == "WARNING"
    )


def power_brake_decision(
    batch: Any, sample: Any, rate: float | None, previous: Any
) -> dict[str, Any] | None:
    """Field 1424 graded as a duty cycle (see the module docstring)."""

    if sample.canonical_name != HW_POWER_BRAKE_SLOWDOWN or rate is None:
        return None
    duty = rate / 60_000_000
    if not 0 < duty <= DUTY_MAX_PLAUSIBLE:
        return None
    mask = device_values(batch, sample).get("clock_throttle_reasons")
    with_bit = (
        mask is not None
        and valid_clock_mask(mask)
        and bool(int(mask) & HW_POWER_BRAKE_MASK)
    )
    prior = previous.finding if previous is not None else None
    sustained = (
        duty >= POWER_BRAKE_DRAIN_DUTY
        and prior is not None
        and prior.canonical_name == HW_POWER_BRAKE_SLOWDOWN
        and prior.rate_per_minute is not None
        and prior.rate_per_minute / 60_000_000 >= POWER_BRAKE_DRAIN_DUTY
        and 0
        < (batch.observed_at - prior.observed_at).total_seconds()
        <= POWER_BRAKE_CONSECUTIVE_MAX_GAP_SECONDS
    )
    common = {
        "policy_source": "SITE_NVIDIA_POWER_DELIVERY",
        "policy_reference": NVML_FIELD_VALUE_REFERENCE,
        "threshold_value": POWER_BRAKE_DRAIN_DUTY,
    }
    if with_bit or sustained:
        return {
            **common,
            "severity": GpuHealthSeverity.CRITICAL,
            "reason": (
                f"hardware power brake engaged {duty:.2%} of the interval (DCGM "
                "field 1424)"
                + (
                    " beside clock-event bit 0x80"
                    if with_bit
                    else " in two consecutive samples"
                )
                + "; inspect external power delivery"
            ),
            "automatic_action": RecoveryAction.DRAIN.value,
            "threshold_source": "NVIDIA_HW_POWER_BRAKE",
            "official_action": "CONTACT_SUPPORT",
            "confirmation_count": 2,
        }
    return {
        **common,
        "severity": GpuHealthSeverity.WARNING,
        "reason": (
            f"hardware power brake engaged {duty:.2%} of the interval (DCGM field "
            "1424); corroborate with clock-event bit 0x80 and the power supply"
        ),
        "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
        "threshold_source": "NVIDIA_HW_POWER_BRAKE_DURATION",
        "confirmation_count": 1,
    }


def ecc_mode_decision(batch: Any, sample: Any) -> dict[str, Any] | None:
    """Fields 300/301 (see the module docstring)."""

    name = sample.canonical_name
    common = {
        "severity": GpuHealthSeverity.WARNING,
        "automatic_action": RecoveryAction.ESCALATE_OPERATOR.value,
        "policy_source": "SITE_GPU_MEMORY_EVIDENCE",
        "policy_reference": NVML_DEVICE_QUERIES_REFERENCE,
        "official_action": "ENABLE_ECC_AND_REBOOT",
    }
    if name == ECC_MODE_CURRENT and sample.value == 0:
        return {
            **common,
            "reason": (
                "ECC is disabled on this GPU (DCGM field 300): NVML reads memory "
                "error counters only with ECC mode enabled, so the memory rules "
                "are blind"
            ),
            "threshold_value": 0.0,
            "threshold_source": "NVIDIA_ECC_MODE_DISABLED",
        }
    if name != ECC_MODE_PENDING:
        return None
    current = device_values(batch, sample).get(ECC_MODE_CURRENT)
    if current is None or sample.value == current:
        return None
    return {
        **common,
        "reason": (
            f"ECC mode change pending (current {current:g}, pending "
            f"{sample.value:g}, DCGM fields 300/301): it takes effect at the next "
            "reboot"
        ),
        "threshold_value": current,
        "threshold_source": "NVIDIA_ECC_MODE_PENDING",
        "official_action": "REVIEW_ECC_MODE_CHANGE",
    }


def inforom_decision(sample: Any) -> dict[str, Any] | None:
    """InfoROM checksum validation ``invalid`` (see the module docstring)."""

    if sample.canonical_name != INFOROM_CHECKSUM_VALID or sample.value != 0:
        return None
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": (
            "InfoROM checksum validation reports invalid (nvidia-smi "
            "inforom.checksum_validation): this GPU's memory-repair evidence -- the "
            "retired pages and ECC records the driver keeps in the InfoROM -- is "
            "untrusted until the InfoROM is flashed"
        ),
        "automatic_action": RecoveryAction.ESCALATE_OPERATOR.value,
        "threshold_value": 0.0,
        "threshold_source": "NVIDIA_INFOROM_CHECKSUM_INVALID",
        "policy_source": "SITE_GPU_MEMORY_EVIDENCE",
        "policy_reference": NVIDIA_INFOROM_RETIREMENT_REFERENCE,
        "official_action": "FLASH_INFOROM",
    }


def field_rule_decision(
    service: Any,
    batch: Any,
    sample: Any,
    delta: float | None,
    rate: float | None,
    previous: Any,
) -> dict[str, Any] | None:
    """The decision for one sample of a :data:`FIELD_RULE_NAMES` field."""

    name = sample.canonical_name
    if name not in FIELD_RULE_NAMES:
        return None
    if name == GPU_RECOVERY_ACTION:
        return recovery_action_decision(service.store, batch, sample)
    if name == PCIE_LINK_DOWNTRAIN:
        return downtrain_decision(service.store, batch, sample, previous)
    if name == PCIE_CORRECTABLE_ERRORS:
        return correctable_decision(batch, sample, rate, previous)
    if name == HW_POWER_BRAKE_SLOWDOWN:
        return power_brake_decision(
            batch, sample, rate if delta is not None else None, previous
        )
    if name == INFOROM_CHECKSUM_VALID:
        return inforom_decision(sample)
    if name in NVLINK5_LINK_RECOVERY:
        return recovery_decision(sample, delta, service.thresholds)
    if name in REPAIR_PENDING_NAMES:
        return repair_pending_decision(batch, sample)
    return ecc_mode_decision(batch, sample)


def field_rule_gate(
    service: Any,
    batch: Any,
    sample: Any,
    decision: dict[str, Any] | None,
    previous_latest: Any,
) -> dict[str, Any] | None:
    """What every decision of ``GpuMetricsService._evaluate`` passes (b235):
    the NVLink5 window suppression and XID evidence, the memory-repair XID
    160/161 evidence. Any other decision is returned unchanged."""

    decision = nvlink5_gate(service.store, batch, sample, decision, previous_latest)
    return with_repair_xid_evidence(service.store, batch, sample, decision)
