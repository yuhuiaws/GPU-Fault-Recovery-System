"""GF-GENERAL-LIVE-024 thermal and power composites through the environment replay.

``thermal_warning`` and ``power_performance`` are the two 024 composites whose
components are environment fields -- temperature / T.Limit, thermal and power
violation durations, clock reasons, power draw and limit, utilization -- so
they reuse the GF-ENV-LIVE telemetry replay (``general_live_env_replay`` over
``tools/gpu_environment_replay.py`` and ``probes/env_replay_proxy.py``; the DCGM
field tool of the other 024 variants does not carry these fields, and the
environment tool's allow-list is not widened). The reviewed plans come from
``env_live_plans`` and the live device: the warning band is computed from the
GPU's own ``nvidia-smi`` limits exactly as GF-ENV-LIVE-003/006 do, and the
counters start above the device's real reading.

What the product must show (spec 024: "热warning、普通power cap ... 不独立授权
更强动作", "检查实际工作流去重而非只比较规则名"):

* ``thermal_warning`` -- a warning-band temperature with a rising thermal
  violation under the ``0x20`` software thermal reason forms THERMAL_STRESS as a
  WARNING RUN_DIAGNOSTICS composite (``gpu_metrics._add_thermal_composite``:
  CRITICAL/DRAIN only for a critical temperature); its workflow collects
  evidence and validates, isolates nothing, resets nothing.
* ``power_performance`` -- a rising power violation with the draw pinned just
  under the enforced limit at high utilization forms POWER_LIMIT_THROTTLING as a
  WARNING RUN_DIAGNOSTICS composite (``_add_power_composite``); no clock reason
  is served because a thermal or brake reason is a temperature finding the
  product keeps out of the power composite by design.
* both -- one composite incident across the sustained batches, its component
  findings suppressed into it (no component incident minted after it formed),
  the composite built only from its own metrics, and a recovery on flat/clear
  readings without a hardware counter reset.

Verdicts read the product's own records (``general_live_gpu_store``,
``general_live_gpu_verdicts``); the proxy's ``/status`` and the accepted
replayed samples are the raw-source oracle. ``proof_scope`` is ``live-replay``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gpu_fault.gpu_metrics import GpuMetricsThresholds

from scripts.e2e.regional import env_live_plans as plans
from scripts.e2e.regional.env_live_fixture import SAFE_TLIMIT_MARGIN_C
from scripts.e2e.regional.general_live_env_replay import (
    CLEANUP_NAME,
    EnvReplay,
    env_replay,
    plan_reads,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    GpuSurvey,
    activity_now,
    gpu_spec,
    no_hardware,
    result,
    run_gpu_survey,
    state_now,
    utc_now,
    wait_accepted,
    wait_gpu_incident,
)
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    accepted_samples,
    chain_errors,
    component_dedup_errors,
    composite_component_errors,
    composite_findings,
    composite_grade_errors,
    composite_incidents,
    read_only_errors,
)

CASE = "GF-GENERAL-LIVE-024"
THERMAL_RULE = "THERMAL_STRESS"
POWER_RULE = "POWER_LIMIT_THROTTLING"
THERMAL_SIGNALS = ("thermal_violation_total_us", "clock_throttle_reasons")
#: The clock context ``_add_thermal_composite`` adds to its component list.
THERMAL_CONTEXT = ("sm_clock_mhz", "memory_clock_mhz")
POWER_FIELDS = (
    "power_violation_total_us",
    "power_usage_w",
    "power_limit_w",
    "gpu_utilization_percent",
)
#: Clock reasons a GPU must not really carry before a replay: software and
#: hardware thermal slowdown and the hardware power brake.
LIVE_THROTTLE_MASK = 0xE0
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)
DIAGNOSTIC_ACTION = "RUN_DIAGNOSTICS"
WARNING = "WARNING"


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 20 * 60)


def _throttle_mask(survey: GpuSurvey, gpu: str) -> int:
    value = survey.value(gpu, "clock_throttle_reasons")
    return int(value) if value is not None and value >= 0 else 0


# --------------------------------------------------------------------------- #
# Thermal warning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermalTarget:
    """The GPU, scale and warning reading a thermal replay grades on."""

    gpu: str
    decision: plans.VariantDecision
    field: str
    warning: float
    baseline_us: float


def thermal_target(
    survey: GpuSurvey, thresholds: GpuMetricsThresholds
) -> ThermalTarget:
    """The highest-index GPU with a gradable core scale and the thermal fields.

    The scale decision is GF-ENV-LIVE's (``env_live_plans.build_device_plan``):
    the T.Limit scale where the device publishes it (p5en H200), else the
    absolute scale; a device with neither is NOT_RUN(CAPABILITY_ABSENT).
    """

    for gpu in reversed(survey.gpus):
        limits = survey.temperature_limits(gpu)
        if not limits:
            continue
        raw: dict[str, Any] = dict(limits)
        reading = survey.value(gpu, "gpu_tlimit_c")
        if reading is not None:
            raw.setdefault("gpu_tlimit_c", reading)
        device = plans.build_device_plan(
            gpu_uuid=gpu,
            gpu_index=survey.entry(gpu).get("gpu_index"),
            product=survey.model(gpu) or None,
            raw_limits=raw,
            thresholds=thresholds,
            confirmation_samples=survey.confirmation,
        )
        decision = next(
            (
                device.decisions[name]
                for name in (plans.CORE_TLIMIT, plans.CORE_ABSOLUTE)
                if device.decisions[name].runnable
            ),
            None,
        )
        if decision is None:
            continue
        field = plans.temperature_field(decision)
        if not all(survey.has(gpu, name) for name in (field, *THERMAL_SIGNALS)):
            continue
        return ThermalTarget(
            gpu=gpu,
            decision=decision,
            field=field,
            warning=plans.warning_reading(decision),
            baseline_us=survey.value(gpu, "thermal_violation_total_us") or 0.0,
        )
    raise VariantNotRun(
        "CAPABILITY_ABSENT",
        "no GPU publishes a gradable core temperature scale together with the "
        "thermal violation and clock reason fields",
    )


def require_cool(survey: GpuSurvey, target: ThermalTarget) -> None:
    """Refuse to arm on a GPU that is really warm or really throttling.

    Mirrors the GF-ENV-LIVE preflight (a T.Limit reading must sit well above 0;
    the replay tool itself aborts on a real unsafe reading) so the replayed
    warning never lands on top of a real thermal event.
    """

    errors = []
    if target.decision.scale == "tlimit":
        reading = survey.value(target.gpu, "gpu_tlimit_c")
        if reading is None or reading <= SAFE_TLIMIT_MARGIN_C:
            errors.append(
                f"target GPU T.Limit reading {reading} C is not safely above 0"
            )
    else:
        reading = survey.value(target.gpu, target.field)
        if reading is None or reading >= target.warning:
            errors.append(
                f"target GPU {target.field} reading {reading} C is already in the "
                f"warning band (>= {target.warning})"
            )
    mask = _throttle_mask(survey, target.gpu)
    if mask & LIVE_THROTTLE_MASK:
        errors.append(f"the GPU really reports thermal/brake clock reasons {mask:#x}")
    if errors:
        raise VariantNotRun("FIXTURE_NOT_READY", "; ".join(errors) + "; refuse to arm")


def thermal_warning(ctx: VariantContext) -> dict[str, Any]:
    thresholds = GpuMetricsThresholds.from_environment()
    survey = run_gpu_survey(ctx, limits=True)
    target = thermal_target(survey, thresholds)
    require_cool(survey, target)
    plan = plans.thermal_stress_plan(
        target.decision,
        confirmation_samples=survey.confirmation,
        baseline_us=target.baseline_us,
    )
    ctx.evidence.write(
        "thermal-target",
        {
            "gpu_uuid": target.gpu,
            "scale": target.decision.scale,
            "field": target.field,
            "warning_reading": target.warning,
            "critical_value": target.decision.critical_value,
            "warning_value": target.decision.warning_value,
            "threshold_source": target.decision.threshold_source,
            "baseline_us": target.baseline_us,
        },
    )
    return _composite_flow(
        ctx,
        survey,
        plan,
        gpu=target.gpu,
        rule=THERMAL_RULE,
        names=(target.field, *THERMAL_SIGNALS, f"composite:{THERMAL_RULE}"),
        components=(target.field, *THERMAL_SIGNALS),
        allowed_components=(target.field, *THERMAL_SIGNALS, *THERMAL_CONTEXT),
        required_components=(target.field,),
        injected_field=target.field,
        injected_value=target.warning,
        notes=[
            "a thermal warning composite stays WARNING RUN_DIAGNOSTICS; only a "
            "critical temperature makes THERMAL_STRESS CRITICAL/DRAIN",
            f"warning reading {target.warning} on the {target.decision.scale} scale "
            f"from the device's own limits ({target.decision.threshold_source})",
        ],
        extra={"scale": target.decision.scale, "temperature_field": target.field},
    )


# --------------------------------------------------------------------------- #
# Power performance (an ordinary power cap)
# --------------------------------------------------------------------------- #
def power_target(survey: GpuSurvey) -> tuple[str, float]:
    """The highest-index GPU reporting every power field and a positive limit."""

    for gpu in reversed(survey.gpus):
        if not all(survey.has(gpu, name) for name in POWER_FIELDS):
            continue
        limit = survey.value(gpu, "power_limit_w") or 0.0
        if limit <= 0:
            continue
        return gpu, float(limit)
    raise VariantNotRun(
        "CAPABILITY_ABSENT",
        "no GPU publishes power violation, draw, a positive enforced power limit "
        "and utilization",
    )


def power_performance(ctx: VariantContext) -> dict[str, Any]:
    thresholds = GpuMetricsThresholds.from_environment()
    survey = run_gpu_survey(ctx)
    gpu, limit = power_target(survey)
    mask = _throttle_mask(survey, gpu)
    if mask & LIVE_THROTTLE_MASK:
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            f"the GPU really reports thermal/brake clock reasons {mask:#x}; refuse to arm",
        )
    baseline_us = survey.value(gpu, "power_violation_total_us") or 0.0
    plan = plans.power_limit_throttling_plan(
        confirmation_samples=survey.confirmation,
        power_limit_w=limit,
        power_limit_ratio=thresholds.power_limit_ratio,
        min_utilization_percent=thresholds.power_correlation_min_utilization_percent,
        baseline_us=baseline_us,
    )
    usage = next(
        float(step["values"]["power_usage_w"])
        for step in plan["steps"]
        if "power_usage_w" in step["values"]
    )
    ctx.evidence.write(
        "power-target",
        {
            "gpu_uuid": gpu,
            "power_limit_w": limit,
            "served_power_usage_w": usage,
            "power_limit_ratio": thresholds.power_limit_ratio,
            "min_utilization_percent": (
                thresholds.power_correlation_min_utilization_percent
            ),
            "baseline_us": baseline_us,
        },
    )
    return _composite_flow(
        ctx,
        survey,
        plan,
        gpu=gpu,
        rule=POWER_RULE,
        names=(*POWER_FIELDS, f"composite:{POWER_RULE}"),
        components=("power_violation_total_us",),
        allowed_components=POWER_FIELDS,
        required_components=(
            "power_violation_total_us",
            "power_usage_w",
            "power_limit_w",
        ),
        injected_field="power_usage_w",
        injected_value=usage,
        notes=[
            "an ordinary power cap (violation near the enforced limit under load) "
            "stays WARNING RUN_DIAGNOSTICS; no clock reason is served, a thermal or "
            "brake reason would be a temperature finding the power composite excludes",
        ],
        extra={"power_limit_w": limit, "served_power_usage_w": usage},
    )


# --------------------------------------------------------------------------- #
# The shared composite flow
# --------------------------------------------------------------------------- #
def _composite_flow(
    ctx: VariantContext,
    survey: GpuSurvey,
    plan: dict[str, Any],
    *,
    gpu: str,
    rule: str,
    names: Sequence[str],
    components: Sequence[str],
    allowed_components: Sequence[str],
    required_components: Sequence[str],
    injected_field: str,
    injected_value: float,
    notes: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    since = utc_now()
    replay: EnvReplay = env_replay(ctx, plan, gpu, _hold(ctx))
    replay.deploy()
    replay.arm()
    composite_label = f"composite:{rule}"
    # Every wait stays inside the proxy's replay window (``run_env_live``'s
    # rule): the tool passes through when it ends and its unit stops shortly
    # after, so the endpoint must be restored before that, not by the deadman.
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, rule),
        metrics=tuple(names),
        timeout_seconds=min(survey.health_summary + 300, replay.remaining_seconds()),
        name="detection",
    )
    # The raw-source oracle: every armed read (breach and clear tail) really
    # reached the collector and the tool never aborted on a real reading.
    total = plan_reads(plan)
    replay.wait_served(
        total,
        timeout_seconds=min(
            survey.interval * (total + 6) + 120, replay.remaining_seconds()
        ),
        name="served",
    )
    state = state_now(ctx, since, names=names)
    accepted = _accepted_injected(state, injected_field, gpu, injected_value)
    if not accepted:
        raise VariantInconclusive(
            f"the product accepted no replayed {injected_field}={injected_value} "
            f"sample on {gpu}"
        )
    final = activity_now(ctx, since, metrics=tuple(names))
    composites = composite_incidents(final, rule) or composite_incidents(
        detection, rule
    )
    findings = composite_findings(state, rule, gpu=gpu)
    stages = {
        "chain": chain_errors(
            final,
            composites,
            GpuChain(
                label=rule,
                operations=("FREEZE_EVIDENCE",),
                forbidden=RESET,
                actions=frozenset({DIAGNOSTIC_ACTION}),
            ),
        ),
        "read_only": read_only_errors(final, composites, rule),
        "diagnostic": composite_grade_errors(
            findings,
            rule_id=rule,
            severity=WARNING,
            action=DIAGNOSTIC_ACTION,
            required_components=required_components,
        ),
        "own_components": composite_component_errors(
            findings, rule_id=rule, allowed_components=allowed_components
        ),
        "dedup": component_dedup_errors(
            final, composites, components=components, gpu=gpu
        ),
        "hardware": no_hardware(final),
    }
    replay.close()
    ctx.cleanup.discard(CLEANUP_NAME)
    recovered = _recovered(set(names) | {composite_label})
    final_state = wait_accepted(
        ctx,
        since,
        recovered,
        names=names,
        timeout_seconds=max(600, _hold(ctx)),
        name="recovery",
    )
    stages["recovery"] = (
        []
        if recovered(final_state)
        else [
            f"{rule} or its components stayed active after the replay cleared: "
            + ", ".join(
                sorted(
                    str(item.get("canonical_name"))
                    for item in final_state.get("active_findings") or []
                )
            )
        ]
    )
    return result(
        stages,
        notes=notes,
        extra={
            **extra,
            "gpu_uuid": gpu,
            "rule_id": rule,
            "replay": replay.evidence(),
            "accepted_injected_samples": len(accepted),
            "composite_incidents": [
                str(item.get("incident_id")) for item in composites
            ],
        },
    )


def _accepted_injected(
    state: dict[str, Any], field: str, gpu: str, value: float
) -> list[dict[str, Any]]:
    tolerance = 0.01 * max(1.0, abs(value))
    return [
        sample
        for sample in accepted_samples(state, field, gpu)
        if isinstance(sample.get("value"), (int, float))
        and abs(float(sample["value"]) - value) <= tolerance
    ]


def _recovered(names: set[str]) -> Callable[[dict[str, Any]], bool]:
    def accept(state: dict[str, Any]) -> bool:
        active = {
            str(item.get("canonical_name"))
            for item in state.get("active_findings") or []
        }
        return not (active & names) and bool(state.get("batches"))

    return accept


VARIANTS = [
    gpu_spec(
        CASE,
        "thermal_warning",
        "warning-band core temperature with a rising thermal violation under the "
        "0x20 thermal clock reason on one GPU (GF-ENV-LIVE environment replay)",
        "THERMAL_STRESS stays a WARNING RUN_DIAGNOSTICS composite: no DRAIN, "
        "isolation or reset; one composite incident, components de-duplicated into "
        "it; recovers on flat/clear readings",
        thermal_warning,
        max_minutes=25,
    ),
    gpu_spec(
        CASE,
        "power_performance",
        "rising power violation with the draw pinned just under the enforced power "
        "limit at high utilization on one GPU (GF-ENV-LIVE environment replay)",
        "POWER_LIMIT_THROTTLING stays a WARNING RUN_DIAGNOSTICS composite: no "
        "stronger action; one composite incident, the violation component "
        "de-duplicated into it; recovers on a flat counter",
        power_performance,
        max_minutes=25,
    ),
]

__all__ = [
    "CASE",
    "LIVE_THROTTLE_MASK",
    "POWER_FIELDS",
    "POWER_RULE",
    "THERMAL_RULE",
    "THERMAL_SIGNALS",
    "VARIANTS",
    "ThermalTarget",
    "power_performance",
    "power_target",
    "require_cool",
    "thermal_target",
    "thermal_warning",
]
