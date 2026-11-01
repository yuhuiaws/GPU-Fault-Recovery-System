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
        "sram_threshold_exceeded",
        "row_remap_banks_none",
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
#: Field 346 (batch 233): NVIDIA GPU Memory Error Management (docs version 615),
#: "RMA Policy" > "GPU L2 Memory RMA Policy" -- the SRAM threshold flag is an RMA
#: criterion the field diagnostic validates, the same path as a row-remap
#: failure.
NVIDIA_SRAM_RMA_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/"
    "rma-policy-thresholds-for-row-remapping.html#gpu-l2-memory-rma-policy"
)
NVIDIA_SRAM_RMA_QUOTE = (
    "The NVIDIA Field Diagnostic tool will determine whether a GPU qualifies for "
    "RMA. Regarding SRAM uncorrectable errors, the RMA criteria is met for events "
    "outlined below. Any of the following events will trigger the SRAM Threshold "
    "Exceeded flag:"
)
#: Field 389: nvidia-smi(1) "Bank Remap Availability Histogram" -- "None means
#: that no reserved rows are available"; DCGM field identifiers (docs 4.6):
#: "Historical marker of memory banks with no available spare memory rows."
NVIDIA_SMI_BANK_REMAP_QUOTE = (
    "Maximum availability means that all reserved rows are available for "
    "remapping while None means that no reserved rows are available."
)
#: What the next uncorrectable error on such a bank does depends on the
#: family: NVIDIA GPU Memory Error Management (docs version 615), "RMA Policy"
#: > "GPU DRAM Memory RMA Policy".
NVIDIA_DRAM_RMA_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/"
    "rma-policy-thresholds-for-row-remapping.html#gpu-dram-memory-rma-policy"
)
NVIDIA_ROW_REMAP_FAILURE_BANK_QUOTE = (
    "A remapping attempt for an uncorrectable memory error on a bank that already "
    "has eight uncorrectable error rows remapped."
)
NVIDIA_BLACKWELL_CHANNEL_REPAIR_QUOTE = (
    "For Blackwell GPUs, a third remapping attempt for an uncorrectable memory "
    "error on a bank invokes HBM channel repair if there is a spare channel "
    "available. If the channel repair is successful, the hardware will be "
    "recovered and ready to use. Otherwise, the GPU will continue remapping until "
    "Row-Remap Failure flag is set."
)
#: The Blackwell family among the memory policy's Ampere-and-later parts.
BLACKWELL_MODEL_TOKENS = frozenset({"B100", "B200", "B300", "GB200", "GB300"})
#: Field 393's absolute limit: DCGM 4.6.0 release notes, "Background health
#: checks (dcgmi health)", and DCGM v4.7.0 ``common/DcgmGPUHardwareLimits.h``
#: ``#define DCGM_LIMIT_MAX_ROW_REMAP_UNCORRECTABLE 512`` ("This number of
#: uncorrectable row remaps indicates GPU is in a bad state"), compared with
#: ``>=`` in ``DcgmHealthWatch::MonitorMemRowRemapUncorrectable``.
DCGM_ROW_REMAP_UNCORRECTABLE_LIMIT = 512
DCGM_ROW_REMAP_LIMIT_SOURCE = "NVIDIA_DCGM_ROW_REMAP_LIMIT"
DCGM_ROW_REMAP_LIMIT_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html"
)
DCGM_ROW_REMAP_LIMIT_QUOTE = (
    "Health monitoring now warns when total uncorrectable row remaps reach 512 on "
    "Ampere and newer GPUs, allowing administrators to run field diagnostics "
    "before GPU memory health degrades further."
)


def _no_spare_rows_outcome(model: set[str], modern: bool) -> str:
    """What the next uncorrectable error does on a bank without spare rows,
    by family (:data:`NVIDIA_DRAM_RMA_REFERENCE`)."""

    if model & BLACKWELL_MODEL_TOKENS:
        return (
            "on Blackwell the next uncorrectable error on such a bank invokes HBM "
            "channel repair if a spare channel is available, and the GPU keeps "
            "remapping until the row-remap failure flag is set otherwise; diagnose"
        )
    if modern:
        return (
            "the next uncorrectable error on such a bank cannot be remapped and sets "
            "the row-remap failure flag (an RMA criterion); diagnose and plan the "
            "replacement"
        )
    return (
        "the architecture is not recognised, so how it handles the next "
        "uncorrectable error on such a bank is unknown; diagnose"
    )


def row_remap_limit_decision(sample: Any) -> dict[str, Any] | None:
    """Field 393 at or above DCGM's absolute limit, without a new increment.

    A new uncorrectable remap keeps its increment verdict (``memory_decision``);
    between increments the level itself is DCGM's health warning. Remaps are
    permanent, so the level persists and no validation can re-judge it: an
    advisory for VALIDATE_GPU (``gpu_field_rules.blocks_validation``).
    """

    if (
        sample.canonical_name != "row_remap_uncorrectable_total"
        or sample.value < DCGM_ROW_REMAP_UNCORRECTABLE_LIMIT
    ):
        return None
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": (
            f"{sample.value:g} uncorrectable row remaps reach DCGM's limit of "
            f"{DCGM_ROW_REMAP_UNCORRECTABLE_LIMIT}; run the field diagnostic"
        ),
        "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
        "threshold_value": float(DCGM_ROW_REMAP_UNCORRECTABLE_LIMIT),
        "threshold_source": DCGM_ROW_REMAP_LIMIT_SOURCE,
        "policy_source": "NVIDIA_DCGM_HEALTH",
        "policy_reference": DCGM_ROW_REMAP_LIMIT_REFERENCE,
        "official_action": "RUN_FIELD_DIAGNOSTIC_FOR_RMA",
    }


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
            return row_remap_limit_decision(sample)
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
    if name == "row_remap_banks_none":
        return {
            "severity": GpuHealthSeverity.WARNING,
            "reason": (
                f"{sample.value:g} memory bank(s) report no spare rows left for "
                "remapping (DCGM field 389): "
                + _no_spare_rows_outcome(set(model), modern)
            ),
            "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
            "threshold_value": 0.0,
            "threshold_source": "NVIDIA_ROW_REMAP_HEADROOM_NONE",
            "policy_source": "SITE_GPU_MEMORY_EVIDENCE",
            "policy_reference": NVIDIA_DRAM_RMA_REFERENCE,
        }
    if name in {"row_remap_failure", "sram_threshold_exceeded"}:
        sram = name == "sram_threshold_exceeded"
        return {
            "severity": GpuHealthSeverity.CRITICAL,
            "reason": (
                "SRAM threshold exceeded (DCGM field 346, an RMA criterion); field "
                "diagnosis required before any RMA conclusion"
                if sram
                else "row remap failure; field diagnosis required before any RMA "
                "conclusion"
            ),
            "automatic_action": (
                RecoveryAction.DRAIN if modern else RecoveryAction.QUARANTINE
            ).value,
            "policy_source": "NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT",
            "policy_version": NVIDIA_GPU_MEMORY_POLICY_VERSION,
            "policy_reference": (
                NVIDIA_SRAM_RMA_REFERENCE if sram else NVIDIA_ROW_REMAP_RMA_REFERENCE
            ),
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
