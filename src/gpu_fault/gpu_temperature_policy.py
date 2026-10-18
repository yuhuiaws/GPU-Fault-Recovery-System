"""GPU temperature grading, shared by the DCGM edge filter and the control plane.

NVIDIA reports a GPU's temperature limits on one of two scales, and this module
never converts one into the other because NVIDIA documents no conversion:

* Absolute degrees Celsius (``gpu_slowdown_temperature_c`` and the other
  ``ABSOLUTE_TEMPERATURE_LIMITS``), what ``nvidia-smi -q -x`` prints for a GPU
  without the T.Limit sensor. nvidia-smi(1), "GPU Slowdown Temp": "The
  temperature at which a GPU HW will begin optimizing clocks due to thermal
  conditions, in order to cool." They grade the absolute ``gpu_temperature_c``.
* The T.Limit scale (``TLIMIT_THRESHOLDS``), what driver 580.178.04 prints on
  H200 (live: ``<gpu_temp_tlimit>58 C``, shutdown ``-8 C``, slowdown ``-2 C``,
  GPU and memory max operating ``0 C``). nvidia-smi documentation,
  https://docs.nvidia.com/deploy/nvidia-smi/index.html, GPU ATTRIBUTES >
  Temperature (the same text is in the nvidia-smi(1) man page of driver
  580.178.04):

    "T.Limit: The T.Limit sensor measures the current margin in degree Celsius
    to the maximum operating temperature. As such it is not an absolute
    temperature reading rather a relative measurement."
    "When supported, nvidia-smi reports the current T.Limit temperature as a
    signed value that counts down. A T.Limit temperature of 0 C or lower
    indicates that the GPU may optimize its clock based on thermal conditions.
    Further, when the T.Limit sensor is supported, available temperature
    thresholds are also reported relative to T.Limit (see below) instead of
    absolute measurements."
    "GPU Shutdown T.Limit Temp Specification: The T.Limit temperature below
    which a GPU may shutdown. Since shutdown can only triggered by the maximum
    GPU temperature it is possible for the current T.Limit to be more negative
    than this threshold."
    "GPU Slowdown T.Limit Temp Specification: The T.Limit temperature at or
    below which GPU HW may optimize its clocks for thermal conditions."
    "GPU Max Operating T.Limit Temp Specification: The T.Limit temperature
    below which GPU SW will optimize its clock for thermal conditions."

  NVML defines the same thresholds only on that scale (nvml.h of
  cuda-nvml-dev 13.0; https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html):
  "GPU T.Limit temperature thresholds in degree Celsius. These fields are
  supported on Ada and later architectures and supersedes
  nvmlDeviceGetTemperatureThreshold", e.g. NVML_FI_DEV_TEMPERATURE_SLOWDOWN_TLIMIT
  "T.Limit temperature after which GPU may begin HW slowdown".

  An absolute threshold does not follow from these: T.Limit "is not an
  absolute temperature reading", and the shutdown caveat says the current
  T.Limit can pass a threshold without the trigger firing. So the T.Limit
  thresholds grade the current T.Limit reading ``gpu_tlimit_c`` -- DCGM field
  153, which DCGM 4.1.0 added as "a field to track the TLIMIT" (DCGM field
  identifiers: "Thermal margin temperature (distance to nearest slowdown
  threshold) for this GPU"), or ``nvidia-smi --query-gpu=temperature.gpu.tlimit``
  -- and a breach is "at or below", because the scale counts down.

The reading is one margin for the GPU and its memory: the 535.129.03 release
notes (https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-535-129-03/index.html,
1.2 Fixed Issues) fixed an HBM temperature read "above the memory maximum
operating temperature. This produced a negative T.Limit temperature reading."
That agrees with the shutdown and slowdown caveats: those trigger on "the
maximum GPU temperature" only, so the reading can pass them. So the memory max
operating T.Limit threshold (NVML: "T.Limit temperature after which GPU may
begin SW slowdown due to memory temperature") joins the warning boundary, as
the GPU's own SW slowdown point does; critical stays the HW slowdown threshold,
because one reading cannot tell whose margin it is.

Precedence for the GPU core: the T.Limit judgement (a plausible reading plus
at least one GPU T.Limit threshold) replaces the absolute one, since NVML says
the T.Limit fields supersede the absolute interface; otherwise the absolute
device limits; otherwise the configured fallback. ``memory_temperature_c``
keeps the absolute memory limit or the configured fallback: NVIDIA reports the
memory temperature absolute only, and the T.Limit's memory share rests on a
bug-fix note, not on a definition, so the memory safety net stays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict

from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.models import RecoveryAction

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricsThresholds

#: ``nvmlDeviceGetTemperatureThreshold`` (the absolute thresholds). The page
#: moved under ``latest/api/``; the old ``nvml-api/group__...`` URL is a 404.
NVIDIA_NVML_TEMPERATURE_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html"
)
#: nvidia-smi(1), GPU ATTRIBUTES > Temperature: the T.Limit scale.
NVIDIA_SMI_TLIMIT_REFERENCE = "https://docs.nvidia.com/deploy/nvidia-smi/index.html"

ABSOLUTE_TEMPERATURE_LIMITS = frozenset(
    {
        "gpu_slowdown_temperature_c",
        "gpu_shutdown_temperature_c",
        "gpu_max_operating_temperature_c",
        "memory_max_operating_temperature_c",
    }
)
#: The current T.Limit, a signed margin in degrees Celsius that counts down.
TLIMIT_READING = "gpu_tlimit_c"
TLIMIT_GPU_THRESHOLDS = frozenset(
    {"gpu_shutdown_tlimit_c", "gpu_slowdown_tlimit_c", "gpu_max_operating_tlimit_c"}
)
TLIMIT_THRESHOLDS = TLIMIT_GPU_THRESHOLDS | {"memory_max_operating_tlimit_c"}
#: Every sample a temperature decision reads besides the graded one.
TEMPERATURE_LIMIT_CONTEXT = (
    ABSOLUTE_TEMPERATURE_LIMITS | TLIMIT_THRESHOLDS | {TLIMIT_READING}
)
#: The readings a temperature finding is raised on.
TEMPERATURE_FINDING_NAMES = frozenset(
    {"gpu_temperature_c", "memory_temperature_c", TLIMIT_READING}
)
#: Input validation, not an NVIDIA value: a margin further than this from 0 C is
#: a sentinel or a wrapped integer, never the margin of a GPU in operation, and
#: is ignored like "N/A" (a garbage threshold must neither drain every GPU nor
#: switch the grading off).
TLIMIT_PLAUSIBLE_MAGNITUDE_C = 150.0


class TemperatureDecision(TypedDict):
    severity: GpuHealthSeverity
    reason: str
    automatic_action: str
    threshold_value: float
    threshold_source: str
    policy_source: NotRequired[str]
    policy_reference: NotRequired[str]


def plausible_tlimit(value: float | None) -> float | None:
    """A T.Limit value usable for grading, or ``None``."""

    if value is None or not math.isfinite(value):
        return None
    return value if abs(value) <= TLIMIT_PLAUSIBLE_MAGNITUDE_C else None


def tlimit_grades_gpu(limits: Mapping[str, float]) -> bool:
    """Whether the GPU core is judged on the T.Limit scale for this sample."""

    return plausible_tlimit(limits.get(TLIMIT_READING)) is not None and any(
        plausible_tlimit(limits.get(name)) is not None for name in TLIMIT_GPU_THRESHOLDS
    )


def _tlimit_boundaries(
    thresholds: GpuMetricsThresholds, limits: Mapping[str, float]
) -> tuple[float, float] | None:
    """(critical, warning) on the T.Limit scale, mirroring the absolute rule.

    Critical is the HW slowdown threshold, else the max operating one, else
    the shutdown threshold plus the shutdown margin; warning is the warning
    margin before critical, or a SW slowdown point (GPU or memory max
    operating) when that comes first. The scale counts down, so "before" means
    a larger value.
    """

    slowdown = plausible_tlimit(limits.get("gpu_slowdown_tlimit_c"))
    maximum = plausible_tlimit(limits.get("gpu_max_operating_tlimit_c"))
    shutdown = plausible_tlimit(limits.get("gpu_shutdown_tlimit_c"))
    memory = plausible_tlimit(limits.get("memory_max_operating_tlimit_c"))
    if slowdown is not None:
        critical = slowdown
    elif maximum is not None:
        critical = maximum
    elif shutdown is not None:
        critical = shutdown + thresholds.gpu_temperature_shutdown_margin_c
    else:
        return None
    warning = max(
        (
            critical + thresholds.gpu_temperature_warning_margin_c,
            *(point for point in (maximum, memory) if point is not None),
        )
    )
    return critical, warning


def _decision(
    severity: GpuHealthSeverity,
    reason: str,
    threshold_value: float,
    source: str,
) -> TemperatureDecision:
    result: TemperatureDecision = {
        "severity": severity,
        "reason": reason,
        "automatic_action": (
            RecoveryAction.DRAIN.value
            if severity is GpuHealthSeverity.CRITICAL
            else RecoveryAction.RUN_DIAGNOSTICS.value
        ),
        "threshold_value": threshold_value,
        "threshold_source": source,
    }
    if source != "CONFIGURED_FALLBACK":
        result["policy_source"] = "SITE_NVIDIA_DEVICE_LIMIT_DERIVED"
        result["policy_reference"] = (
            NVIDIA_SMI_TLIMIT_REFERENCE
            if source == "NVIDIA_DEVICE_TLIMIT"
            else NVIDIA_NVML_TEMPERATURE_REFERENCE
        )
    return result


def _tlimit_decision(
    thresholds: GpuMetricsThresholds,
    sample: GpuMetricSample,
    limits: Mapping[str, float],
) -> TemperatureDecision | None:
    reading = plausible_tlimit(sample.value)
    boundaries = _tlimit_boundaries(thresholds, limits)
    if reading is None or boundaries is None:
        return None
    critical, warning = boundaries
    if reading <= critical:
        severity, threshold = GpuHealthSeverity.CRITICAL, critical
    elif reading <= warning:
        severity, threshold = GpuHealthSeverity.WARNING, warning
    else:
        return None
    return _decision(
        severity,
        "GPU T.Limit margin fell to the nvidia_device_tlimit threshold",
        threshold,
        "NVIDIA_DEVICE_TLIMIT",
    )


def temperature_decision(
    thresholds: GpuMetricsThresholds,
    sample: GpuMetricSample,
    limits: Mapping[str, float],
) -> TemperatureDecision | None:
    """Use the same device-derived action boundaries before and after delivery."""

    name = sample.canonical_name
    if name == TLIMIT_READING:
        return _tlimit_decision(thresholds, sample, limits)
    if name == "gpu_temperature_c":
        if tlimit_grades_gpu(limits):
            # Judged on its own T.Limit sample; grading the absolute reading
            # against the configured fallback too would report one condition
            # twice, on a boundary that is not the GPU's.
            return None
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
    if sample.value >= critical:
        severity, threshold = GpuHealthSeverity.CRITICAL, critical
    elif sample.value >= warning:
        severity, threshold = GpuHealthSeverity.WARNING, warning
    else:
        return None
    return _decision(
        severity,
        f"{label} temperature exceeded {source.lower()} threshold",
        threshold,
        source,
    )
