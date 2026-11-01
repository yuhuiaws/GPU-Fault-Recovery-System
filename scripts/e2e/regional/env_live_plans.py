"""Reviewed replay plans for GF-ENV-LIVE, built from one device's live limits.

The replay tool (:mod:`tools.gpu_environment_replay`) changes only fields a GPU
already reports and refuses to invent a limit. So the per-case plan this module
builds is decided by what the *live* device publishes on ``nvidia-smi -q -x``,
read on the target node at preflight -- never by the ``tlimit-example.json``
numbers, which the spec warns are a single device's values.

Two temperature scales, never mixed (``gpu_temperature_policy``): the absolute
Celsius thresholds a GPU without the T.Limit sensor prints, and the signed
T.Limit margins (count down to 0) a GPU with it prints. A p5en H200 publishes
only the T.Limit scale (live 2026-10-24: shutdown ``-8``, slowdown ``-2``, GPU
and memory max operating ``0``), so the absolute-core variant is NOT_RUN there
and the T.Limit variant runs. Its one absolute limit, the shutdown temperature,
reaches the product from DCGM field 159 (``collectors/gpu/shutdown_limit.py``),
and the runner merges what the product published into the device's limits:
the absolute-shutdown case runs once that value is there. This module decides
per GPU and names the reason; it never fabricates the missing scale.

Everything here is pure: the device limits come in as a mapping, the plan goes
out as the ``{"steps": [...]}`` object the tool validates. No network, no
device, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from gpu_fault.gpu_temperature_policy import (
    ABSOLUTE_TEMPERATURE_LIMITS,
    TLIMIT_GPU_THRESHOLDS,
    plausible_tlimit,
    shutdown_from_device_field,
    tlimit_grades_gpu,
)

#: The three severe-temperature variants of GF-ENV-LIVE-003/004. Each is run
#: only when the device reports the scale it grades on.
CORE_ABSOLUTE = "core_absolute"
CORE_TLIMIT = "core_tlimit"
MEMORY_ABSOLUTE = "memory_absolute"
TEMPERATURE_VARIANTS = (CORE_ABSOLUTE, CORE_TLIMIT, MEMORY_ABSOLUTE)

#: The Power Brake and thermal clock bits (``gpu_environment_policy``), kept
#: here so a plan names them without importing the mask constants twice.
POWER_BRAKE_BIT = 0x80
SW_THERMAL_BIT = 0x20
HW_THERMAL_BIT = 0x40
#: Non-brake clock reasons 007 proves are not read as a brake; each is a single
#: bit the policy maps to WARNING/NONE, never ``0x80``.
NON_BRAKE_BITS = (0x04, 0x08, 0x10, 0x100)
#: Mixed masks that still carry ``0x80`` and must still read as a brake.
BRAKE_MIXED_MASKS = (0x84, 0xC0, 0x180)


class EnvLivePlanError(ValueError):
    """A reviewed plan cannot be built from the device's live limits."""


@dataclass(frozen=True)
class VariantDecision:
    """Whether one severe-temperature variant runs on this device, and why."""

    variant: str
    runnable: bool
    reason: str
    scale: str | None = None
    threshold_source: str | None = None
    critical_value: float | None = None
    warning_value: float | None = None


@dataclass(frozen=True)
class DevicePlan:
    """The reviewed plans for one physical GPU, with the limits they came from."""

    gpu_uuid: str
    gpu_index: str | None
    product: str | None
    limits: dict[str, float]
    tlimit_reading: float | None
    decisions: dict[str, VariantDecision]
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)

    def runnable_variants(self) -> list[str]:
        return [name for name, item in self.decisions.items() if item.runnable]


def _finite_limit(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def normalize_device_limits(raw: dict[str, Any]) -> dict[str, float]:
    """Keep only the finite numeric temperature limits the policy reads.

    Absolute limits must be positive (a non-positive absolute threshold is a
    sentinel); T.Limit margins may be zero or negative, bounded by the same
    plausibility window the policy uses. ``gpu_tlimit_c`` (the reading) is kept.
    """

    result: dict[str, float] = {}
    for name, value in raw.items():
        number = _finite_limit(value)
        if number is None:
            continue
        if name in ABSOLUTE_TEMPERATURE_LIMITS:
            if number > 0:
                result[name] = number
        elif name == "gpu_tlimit_c" or name.endswith("_tlimit_c"):
            if plausible_tlimit(number) is not None:
                result[name] = number
        elif name in {"memory_temperature_c", "gpu_temperature_c"}:
            result[name] = number
    return result


def _tlimit_boundaries(
    thresholds: GpuMetricsThresholds, limits: dict[str, float]
) -> tuple[float | None, float] | None:
    slowdown = plausible_tlimit(limits.get("gpu_slowdown_tlimit_c"))
    maximum = plausible_tlimit(limits.get("gpu_max_operating_tlimit_c"))
    shutdown = plausible_tlimit(limits.get("gpu_shutdown_tlimit_c"))
    memory = plausible_tlimit(limits.get("memory_max_operating_tlimit_c"))
    if slowdown is not None:
        critical: float | None = slowdown
    elif shutdown is not None:
        critical = shutdown + thresholds.gpu_temperature_shutdown_margin_c
    elif maximum is not None:
        return None, maximum
    else:
        return None
    warning = max(
        (
            critical + thresholds.gpu_temperature_warning_margin_c,
            *(point for point in (maximum, memory) if point is not None),
        )
    )
    return critical, warning


def core_tlimit_decision(
    thresholds: GpuMetricsThresholds, limits: dict[str, float]
) -> VariantDecision:
    reading = plausible_tlimit(limits.get("gpu_tlimit_c"))
    if reading is None:
        return VariantDecision(
            CORE_TLIMIT,
            False,
            "device publishes no plausible T.Limit reading (field 153)",
        )
    if not tlimit_grades_gpu(limits):
        return VariantDecision(
            CORE_TLIMIT,
            False,
            "device publishes no GPU T.Limit threshold "
            f"({sorted(TLIMIT_GPU_THRESHOLDS)})",
        )
    boundaries = _tlimit_boundaries(thresholds, limits)
    if boundaries is None or boundaries[0] is None:
        return VariantDecision(
            CORE_TLIMIT,
            False,
            "device has only a max-operating T.Limit threshold; software "
            "limiting warns but there is no HW slowdown critical boundary",
            scale="tlimit",
        )
    critical, warning = boundaries
    return VariantDecision(
        CORE_TLIMIT,
        True,
        "GPU core graded on the T.Limit scale against its own slowdown margin",
        scale="tlimit",
        threshold_source="NVIDIA_DEVICE_TLIMIT",
        critical_value=critical,
        warning_value=warning,
    )


def core_absolute_decision(
    thresholds: GpuMetricsThresholds, limits: dict[str, float]
) -> VariantDecision:
    if tlimit_grades_gpu(limits):
        return VariantDecision(
            CORE_ABSOLUTE,
            False,
            "device grades the GPU core on the T.Limit scale; the absolute "
            "reading is not graded a second time against a fallback",
            scale="tlimit",
        )
    slowdown = limits.get("gpu_slowdown_temperature_c")
    # A T.Limit GPU's field-159 shutdown limit serves the emergency only (011),
    # never a "shutdown minus margin" boundary: the product does not grade so.
    shutdown = (
        None
        if shutdown_from_device_field(limits)
        else limits.get("gpu_shutdown_temperature_c")
    )
    maximum = limits.get("gpu_max_operating_temperature_c")
    critical = slowdown or (
        shutdown - thresholds.gpu_temperature_shutdown_margin_c
        if shutdown is not None
        else None
    )
    if critical is None and maximum is None:
        return VariantDecision(
            CORE_ABSOLUTE,
            False,
            "device publishes no absolute GPU temperature threshold",
            scale="absolute",
        )
    if critical is None:
        return VariantDecision(
            CORE_ABSOLUTE,
            False,
            "device publishes only an absolute max-operating temperature "
            "(software limiting warns; no HW slowdown critical boundary)",
            scale="absolute",
        )
    warning = min(
        critical - thresholds.gpu_temperature_warning_margin_c,
        maximum if maximum is not None else critical,
    )
    return VariantDecision(
        CORE_ABSOLUTE,
        True,
        "GPU core graded on the absolute Celsius scale against its slowdown threshold",
        scale="absolute",
        threshold_source="NVIDIA_DEVICE_LIMIT",
        critical_value=critical,
        warning_value=warning,
    )


def memory_decision(
    thresholds: GpuMetricsThresholds,
    limits: dict[str, float],
    *,
    allow_fallback: bool,
) -> VariantDecision:
    maximum = limits.get("memory_max_operating_temperature_c")
    if maximum is not None:
        warning = min(maximum - thresholds.memory_temperature_warning_margin_c, maximum)
        return VariantDecision(
            MEMORY_ABSOLUTE,
            True,
            "GPU memory graded on the absolute device memory limit",
            scale="absolute",
            threshold_source="NVIDIA_DEVICE_LIMIT",
            critical_value=maximum,
            warning_value=warning,
        )
    if not allow_fallback:
        return VariantDecision(
            MEMORY_ABSOLUTE,
            False,
            "device publishes no absolute memory temperature limit and the "
            "configured memory fallback was not approved for this run",
            scale="absolute",
        )
    return VariantDecision(
        MEMORY_ABSOLUTE,
        True,
        "GPU memory graded on the site configured absolute fallback (device "
        "publishes no absolute memory limit); recorded as CONFIGURED_FALLBACK",
        scale="absolute",
        threshold_source="CONFIGURED_FALLBACK",
        critical_value=thresholds.memory_temperature_critical_c,
        warning_value=thresholds.memory_temperature_warning_c,
    )


def _reads(count: int, values: dict[str, float | None]) -> dict[str, Any]:
    if count < 1:
        raise EnvLivePlanError("a replay step needs at least one read")
    return {"reads": count, "values": values}


def _temperature_field(variant: str) -> str:
    if variant == MEMORY_ABSOLUTE:
        return "memory_temperature_c"
    if variant == CORE_TLIMIT:
        return "gpu_tlimit_c"
    return "gpu_temperature_c"


def _severe_value(decision: VariantDecision) -> float:
    """A reading one step past the critical boundary, on the right scale.

    The T.Limit scale counts down, so "more severe" is below critical; the
    absolute scales count up, so it is above. The step stays well inside the
    tool's own safety stops (it injects the transformed value; the real reading
    is what the tool guards)."""

    assert decision.critical_value is not None, decision
    return (
        decision.critical_value - 1.0
        if decision.scale == "tlimit"
        else decision.critical_value + 1.0
    )


def _warning_value(decision: VariantDecision) -> float:
    assert decision.warning_value is not None, decision
    if decision.scale == "tlimit":
        # Warning band is critical < T <= warning; a value strictly inside it.
        assert decision.critical_value is not None, decision
        return min(
            decision.warning_value,
            (decision.critical_value + decision.warning_value) / 2,
        )
    return decision.warning_value


def confirmation_plan(
    decision: VariantDecision, *, confirmation_samples: int, warmup_reads: int = 2
) -> dict[str, Any]:
    """GF-ENV-LIVE-004: warm up healthy, then N severe reads to confirm.

    The first ``confirmation_samples - 1`` severe reads must not STOP; the Nth
    crosses the confirmation count. A final healthy block proves the cooled
    reading passes the edge filter and the node stays isolated regardless.
    """

    field = _temperature_field(decision.variant)
    severe = _severe_value(decision)
    steps = [_reads(max(1, warmup_reads), {})]
    steps.append(_reads(confirmation_samples, {field: severe}))
    return {"steps": steps}


def warning_plan(
    decision: VariantDecision, *, confirmation_samples: int, warmup_reads: int = 2
) -> dict[str, Any]:
    """GF-ENV-LIVE-003: warm up healthy, then sustained warning-band reads.

    Warning stays read-only diagnostic; no sample may STOP. The reads outlast a
    confirmation window so a mistaken escalation would show.
    """

    field = _temperature_field(decision.variant)
    warning = _warning_value(decision)
    return {
        "steps": [
            _reads(max(1, warmup_reads), {}),
            _reads(confirmation_samples + 2, {field: warning}),
        ]
    }


def shutdown_plan(limits: dict[str, float]) -> dict[str, Any]:
    """GF-ENV-LIVE-011: one read at absolute shutdown + 1 C.

    Requires the product's device limits to carry an absolute
    ``gpu_shutdown_temperature_c`` from any device source: the XML of a GPU
    without the T.Limit sensor, or DCGM field 159 of a T.Limit GPU, which the
    collector publishes once the GPU's own T.Limit arithmetic agrees and the
    runner reads from the product's records (``env_live_fixture``). One accepted
    sample triggers protection without waiting for three. The read raises
    ``gpu_temperature_c`` -- DCGM_FI_DEV_GPU_TEMP through the proxy, the reading
    the product grades against that limit; the limit is hardware metadata the
    replay tool never changes, and nothing is physically heated. The T.Limit
    shutdown offset is a different scale and never substitutes, so a device
    without an absolute limit is NOT_RUN for this case.
    """

    shutdown = limits.get("gpu_shutdown_temperature_c")
    if shutdown is None:
        raise EnvLivePlanError(
            "the product's device limits carry no absolute "
            "gpu_shutdown_temperature_c from any device source (nvidia-smi XML "
            "or DCGM field 159); GF-ENV-LIVE-011 is NOT_RUN (the T.Limit offset "
            "is a different scale and never substitutes)"
        )
    return {"steps": [_reads(1, {"gpu_temperature_c": shutdown + 1.0})]}


def transient_plans(
    decision: VariantDecision, *, confirmation_samples: int
) -> dict[str, dict[str, Any]]:
    """GF-ENV-LIVE-005: one-then-clear, two-then-clear, warning-then-severe.

    None may confirm: a cleared severe sample resets the count, and warning
    history never counts toward critical. Each clears to healthy and observes.
    """

    field = _temperature_field(decision.variant)
    severe = _severe_value(decision)
    warning = _warning_value(decision)
    clear = max(3, confirmation_samples)
    return {
        "once_then_clear": {"steps": [_reads(1, {field: severe}), _reads(clear, {})]},
        "twice_then_clear": {"steps": [_reads(2, {field: severe}), _reads(clear, {})]},
        "warning_then_severe": {
            "steps": [
                _reads(2, {field: warning}),
                _reads(1, {field: severe}),
                _reads(clear, {}),
            ]
        },
    }


def temperature_field(decision: VariantDecision) -> str:
    """The exporter field a core/memory temperature decision grades on."""

    return _temperature_field(decision.variant)


def warning_reading(decision: VariantDecision) -> float:
    """A reading strictly inside the decision's warning band, on its scale."""

    return _warning_value(decision)


def thermal_stress_plan(
    decision: VariantDecision,
    *,
    confirmation_samples: int,
    baseline_us: float = 0.0,
    step_us: float = 1_000_000.0,
    warmup_reads: int = 3,
    clear_reads: int | None = None,
    reason_bit: int = SW_THERMAL_BIT,
) -> dict[str, Any]:
    """GF-GENERAL-LIVE-024 ``thermal_warning`` (GF-ENV-LIVE-006, warning band).

    Warm up healthy, then hold the core temperature inside its *warning* band
    while the thermal-violation duration rises by ``step_us`` per read under a
    thermal clock reason (``0x20`` software thermal slowdown by default; never
    the ``0x80`` brake) -- on exactly ``confirmation_samples`` reads, one per
    sample the DCGM edge filter needs before it delivers the warning-band
    temperature and the clock reason as ``candidate-confirmed``. The product
    correlates the temperature finding with the duration/throttle findings
    into THERMAL_STRESS, which must stay a WARNING diagnostic. The tail clears
    the reason bit, lets the temperature pass through to the real reading and
    repeats the confirmed read's duration, so the first cleared read
    (``candidate-recovered``) is a zero delta that clears the component
    findings and the composite with them. ``baseline_us`` is the device's real
    counter so the first read is a rise, not a counter reset.

    The tail must not hold a *later* value: only the confirmed read and the
    first cleared read are delivered (a confirmed candidate is no new edge),
    so whatever the cleared read serves is graded against the confirmed read
    (``thermal_duration_decision``: a delta of 10000 us is a fresh WARNING),
    and a rise there is cleared by nothing before VALIDATE_GPU's temperature
    warning grace -- the flat tail is no edge, the pass-through real counter
    is a reset the ingest keeps the finding through (GF-ENV-LIVE-006 live
    2026-10-27, HyperPod). The former shape rose on three more reads than the
    confirmation needed and held the last of them: at a 1e6 us step per 15 s
    read the violation counter is a duty-cycle candidate of its own (0.067 >
    the collector's 0.05) and confirmed one read after the temperature, so
    live 2026-10-26 (both clusters) the product accepted two breach batches
    and the cleared read carried 2e6 us over the last delivered one -- a rise
    the product graded two ways on the two clusters for 006.
    """

    if reason_bit & POWER_BRAKE_BIT:
        raise EnvLivePlanError("a thermal warning plan must not assert the power brake")
    rising = int(confirmation_samples)
    if rising < 1:
        raise EnvLivePlanError("a thermal warning plan needs a confirmation sample")
    field = _temperature_field(decision.variant)
    warning = _warning_value(decision)
    clear = max(3, rising) if clear_reads is None else int(clear_reads)
    if clear < 1:
        raise EnvLivePlanError("a thermal warning plan needs a clearing read")
    steps = [_reads(max(1, warmup_reads), {})]
    for index in range(1, rising + 1):
        steps.append(
            _reads(
                1,
                {
                    field: warning,
                    "thermal_violation_total_us": baseline_us + index * step_us,
                    "clock_throttle_reasons": reason_bit,
                },
            )
        )
    steps.append(
        _reads(
            clear,
            {
                "thermal_violation_total_us": baseline_us + rising * step_us,
                "clock_throttle_reasons": 0,
            },
        )
    )
    return {"steps": steps}


def power_limit_throttling_plan(
    *,
    confirmation_samples: int,
    power_limit_w: float,
    power_limit_ratio: float,
    min_utilization_percent: float,
    baseline_us: float = 0.0,
    step_us: float = 1_000_000.0,
    warmup_reads: int = 3,
    clear_reads: int | None = None,
    serve_limit: bool = False,
) -> dict[str, Any]:
    """GF-GENERAL-LIVE-024 ``power_performance``: an ordinary power cap.

    The power-violation duration rises by ``step_us`` per read while the power
    draw sits between ``power_limit_ratio`` of the enforced limit and the
    limit itself under high utilization -- what DCGM reports for a GPU pinned
    at its power cap -- on exactly ``confirmation_samples`` reads, one per
    sample the DCGM edge filter needs before it delivers the draw-at-the-cap
    correlation as ``candidate-confirmed``. No clock reason bit is served: a
    thermal or brake reason would be a temperature finding, which the product
    deliberately keeps out of POWER_LIMIT_THROTTLING
    (``gpu_metrics._add_power_composite``). The limit is the device's real
    reading unless ``serve_limit`` asks the proxy to serve it too. The tail
    lets draw and utilization pass through again and repeats the confirmed
    read's duration, so the first cleared read (``candidate-recovered``) is a
    zero delta that clears the component whatever the real draw is. The former
    shape held the last of three extra rising reads instead (see
    :func:`thermal_stress_plan`); ``power_violation_decision`` let that rise
    pass only because the pass-through draw of an idle GPU sits under the
    ratio, not because the tail was flat.
    """

    if power_limit_w <= 0:
        raise EnvLivePlanError("a power cap plan needs a positive enforced power limit")
    if not 0 < power_limit_ratio <= 1:
        raise EnvLivePlanError("power_limit_ratio must be in (0, 1]")
    rising = int(confirmation_samples)
    if rising < 1:
        raise EnvLivePlanError("a power cap plan needs a confirmation sample")
    usage = power_limit_w * (power_limit_ratio + (1.0 - power_limit_ratio) / 2)
    utilization = min(100.0, max(min_utilization_percent + 10.0, 90.0))
    clear = max(3, rising) if clear_reads is None else int(clear_reads)
    if clear < 1:
        raise EnvLivePlanError("a power cap plan needs a clearing read")
    steps = [_reads(max(1, warmup_reads), {})]
    for index in range(1, rising + 1):
        values: dict[str, float | None] = {
            "power_violation_total_us": baseline_us + index * step_us,
            "power_usage_w": round(usage, 1),
            "gpu_utilization_percent": utilization,
        }
        if serve_limit:
            values["power_limit_w"] = float(power_limit_w)
        steps.append(_reads(1, values))
    steps.append(
        _reads(clear, {"power_violation_total_us": baseline_us + rising * step_us})
    )
    return {"steps": steps}


def interruption_plan(
    decision: VariantDecision, *, confirmation_samples: int, warmup_reads: int = 2
) -> dict[str, Any]:
    """GF-ENV-LIVE-009: two severe reads, the collection pause, one more, a clear.

    The runner pauses the collector read-synchronously right after the second
    severe read (``pause_read_index``) for longer than the policy's 60 s
    confirmation gap, so the third severe read -- the one that would have
    confirmed -- must be counted ``1/N`` again, never CRITICAL. The final
    healthy block clears the finding before the collector restart is taken at
    a normal temperature. ``confirmation_samples - 1`` severe reads go before
    the pause so the pre-gap count is visibly one short of confirming.
    """

    field = _temperature_field(decision.variant)
    severe = _severe_value(decision)
    before = max(1, confirmation_samples - 1)
    return {
        "steps": [
            _reads(max(1, warmup_reads), {}),
            _reads(before, {field: severe}),
            _reads(1, {field: severe}),
            _reads(max(3, confirmation_samples), {}),
        ]
    }


def pause_read_index(plan: dict[str, Any]) -> int:
    """The plan read after which GF-ENV-LIVE-009 pauses: the end of step 2.

    Counted in the proxy's reads since the arm (``reads - start_reads``): the
    warmup reads plus the first severe block.
    """

    steps = list(plan.get("steps") or [])
    if len(steps) < 3:
        raise EnvLivePlanError("an interruption plan needs warmup, severe and clear")
    return int(steps[0]["reads"]) + int(steps[1]["reads"])


def thermal_duration_plan(
    *,
    confirmation_samples: int,
    baseline_us: float = 0.0,
    step_us: float = 1_000_000.0,
    clear_reads: int | None = None,
) -> dict[str, dict[str, Any]]:
    """GF-ENV-LIVE-006: a rising thermal-violation duration under 0x20 / 0x40.

    Durations are plan micro-seconds; the tool converts them to the exporter's
    nanosecond wire units. The core temperature stays healthy (only the field
    the device already reports is changed), so this exercises the duration /
    composite path, which must stay read-only diagnostic and never DRAIN.

    The duration rises on exactly ``confirmation_samples`` reads -- one per
    sample the DCGM edge filter needs before it delivers the thermal clock
    reason as ``candidate-confirmed`` -- and the tail repeats the last rising
    read's duration with the reason bit cleared. Only two armed reads reach the
    control plane: the confirmed read (the findings and their read-only
    diagnostic workflows) and the first cleared read (``candidate-recovered``,
    delivered at once). A rising read after the confirmed one is never
    delivered by itself (a confirmed candidate is no new edge and a violation
    counter is judged by duty cycle, not by change), so its value would surface
    only on the cleared read -- as a fresh rise the ingest grades into a new
    WARNING (``thermal_duration_decision``) that nothing clears before
    VALIDATE_GPU's temperature-warning grace: the flat tail is no edge and the
    pass-through real counter is a reset the ingest keeps the finding through.
    Live 2026-10-27 (HyperPod, R13g) the four-read ramp did exactly that:
    ``candidate-recovered`` carried 3e6 over the delivered 2e6, VALIDATE_GPU
    failed on the re-minted finding at the grace and both read-only workflows
    ended FAILED (diagnostic inconclusive). With the tail at the confirmed
    read's duration the cleared read is a zero delta and the ingest clears both
    findings before the validation re-checks on that fresh sample.
    """

    rising = int(confirmation_samples)
    if rising < 2:
        raise EnvLivePlanError(
            "a thermal duration plan needs at least two confirmation samples: the "
            "first read repeats the device's counter and only the later reads rise "
            "before the clock reason confirms"
        )
    clear = max(3, rising) if clear_reads is None else int(clear_reads)
    if clear < 1:
        raise EnvLivePlanError("a thermal duration plan needs a clearing read")
    confirmed_us = baseline_us + (rising - 1) * step_us
    result: dict[str, dict[str, Any]] = {}
    for label, bit in (("sw_thermal", SW_THERMAL_BIT), ("hw_thermal", HW_THERMAL_BIT)):
        steps = [
            _reads(
                1,
                {
                    "thermal_violation_total_us": baseline_us + index * step_us,
                    "clock_throttle_reasons": bit,
                },
            )
            for index in range(rising)
        ]
        steps.append(
            _reads(
                clear,
                {
                    "thermal_violation_total_us": confirmed_us,
                    "clock_throttle_reasons": 0,
                },
            )
        )
        result[label] = {"steps": steps}
    return result


def power_brake_plans(*, confirmation_samples: int) -> dict[str, dict[str, Any]]:
    """GF-ENV-LIVE-007: the ``0x80`` brake, non-brake bits and mixed masks.

    ``transient`` is one brake then clear (no isolation); ``sustained`` is N
    brake samples (the same human-review chain as 004); ``non_brake`` and
    ``mixed`` prove the ``mask & 0x80`` discrimination. The temperature stays
    healthy so a thermal warning cannot be read into the brake decision.
    """

    clear = max(3, confirmation_samples)
    plans: dict[str, dict[str, Any]] = {
        "transient": {
            "steps": [
                _reads(1, {"clock_throttle_reasons": POWER_BRAKE_BIT}),
                _reads(clear, {"clock_throttle_reasons": 0}),
            ]
        },
        "sustained": {
            "steps": [
                _reads(
                    confirmation_samples, {"clock_throttle_reasons": POWER_BRAKE_BIT}
                )
            ]
        },
    }
    for bit in NON_BRAKE_BITS:
        plans[f"non_brake_{bit:#x}"] = {
            "steps": [_reads(clear, {"clock_throttle_reasons": bit})]
        }
    for mask in BRAKE_MIXED_MASKS:
        plans[f"mixed_{mask:#x}"] = {
            "steps": [_reads(confirmation_samples, {"clock_throttle_reasons": mask})]
        }
    return plans


def unavailable_field_plans(
    decision: VariantDecision,
) -> dict[str, dict[str, Any]]:
    """GF-ENV-LIVE-008: omit the reason bit (null), a non-integer mask, a blank.

    ``null`` omits the field from the target GPU; a non-integer mask and a
    not-a-number value must read as UNKNOWN / collection error, never as a
    fabricated brake or a healthy sample. The DCGM blank sentinel (>= 9e18)
    is what the collector drops, so it is sent as a large magnitude.
    """

    return {
        "omit_reason": {"steps": [_reads(3, {"clock_throttle_reasons": None})]},
        "non_integer_mask": {"steps": [_reads(3, {"clock_throttle_reasons": 128.5})]},
        "dcgm_blank": {"steps": [_reads(3, {"clock_throttle_reasons": 9.0e18})]},
    }


def build_device_plan(
    *,
    gpu_uuid: str,
    gpu_index: str | None,
    product: str | None,
    raw_limits: dict[str, Any],
    thresholds: GpuMetricsThresholds | None = None,
    confirmation_samples: int = 3,
    allow_memory_fallback: bool = True,
) -> DevicePlan:
    """Decide every severe-temperature variant for one GPU and build its plans."""

    thresholds = thresholds or GpuMetricsThresholds()
    limits = normalize_device_limits(raw_limits)
    decisions = {
        CORE_TLIMIT: core_tlimit_decision(thresholds, limits),
        CORE_ABSOLUTE: core_absolute_decision(thresholds, limits),
        MEMORY_ABSOLUTE: memory_decision(
            thresholds, limits, allow_fallback=allow_memory_fallback
        ),
    }
    plans: dict[str, dict[str, Any]] = {}
    for name, decision in decisions.items():
        if not decision.runnable:
            continue
        plans[f"{name}/severe"] = confirmation_plan(
            decision, confirmation_samples=confirmation_samples
        )
        plans[f"{name}/warning"] = warning_plan(
            decision, confirmation_samples=confirmation_samples
        )
    return DevicePlan(
        gpu_uuid=gpu_uuid,
        gpu_index=gpu_index,
        product=product,
        limits=limits,
        tlimit_reading=limits.get("gpu_tlimit_c"),
        decisions=decisions,
        plans=plans,
    )
