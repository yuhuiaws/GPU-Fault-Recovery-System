"""The reviewed replay plans come from one device's live limits, no templates."""

from __future__ import annotations

import pytest

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from scripts.e2e.regional import env_live_plans as plans
from tools.gpu_environment_replay import validate_plan

# Live p5en H200 (2026-10-24): only the T.Limit scale is published.
H200_LIMITS = {
    "gpu_tlimit_c": 59.0,
    "gpu_shutdown_tlimit_c": -8.0,
    "gpu_slowdown_tlimit_c": -2.0,
    "gpu_max_operating_tlimit_c": 0.0,
    "memory_max_operating_tlimit_c": 0.0,
}
# A hypothetical absolute-scale device (A100-era), for the other branch.
ABSOLUTE_LIMITS = {
    "gpu_temperature_c": 40.0,
    "gpu_slowdown_temperature_c": 85.0,
    "gpu_shutdown_temperature_c": 92.0,
    "gpu_max_operating_temperature_c": 83.0,
    "memory_max_operating_temperature_c": 95.0,
    "memory_temperature_c": 50.0,
}


def _h200() -> plans.DevicePlan:
    return plans.build_device_plan(
        gpu_uuid="GPU-aaaa",
        gpu_index="0",
        product="NVIDIA H200",
        raw_limits=H200_LIMITS,
    )


def _absolute() -> plans.DevicePlan:
    return plans.build_device_plan(
        gpu_uuid="GPU-bbbb", gpu_index="1", product="A100", raw_limits=ABSOLUTE_LIMITS
    )


def test_normalize_drops_sentinels_and_keeps_signed_tlimit() -> None:
    limits = plans.normalize_device_limits(
        {
            "gpu_slowdown_temperature_c": -5,  # non-positive absolute: sentinel
            "gpu_shutdown_temperature_c": 92,
            "gpu_tlimit_c": -2,  # signed margin is real
            "gpu_slowdown_tlimit_c": float("nan"),
            "gpu_max_operating_tlimit_c": 500,  # implausible magnitude
            "memory_temperature_c": 50,
        }
    )
    assert "gpu_slowdown_temperature_c" not in limits
    assert limits["gpu_shutdown_temperature_c"] == 92
    assert limits["gpu_tlimit_c"] == -2
    assert "gpu_slowdown_tlimit_c" not in limits
    assert "gpu_max_operating_tlimit_c" not in limits


def test_h200_runs_tlimit_and_memory_not_absolute_core_or_shutdown() -> None:
    device = _h200()
    assert device.runnable_variants() == [plans.CORE_TLIMIT, plans.MEMORY_ABSOLUTE]
    core = device.decisions[plans.CORE_ABSOLUTE]
    assert not core.runnable and "T.Limit" in core.reason
    tlimit = device.decisions[plans.CORE_TLIMIT]
    assert (
        tlimit.runnable
        and tlimit.critical_value == -2.0
        and tlimit.warning_value == 3.0
    )
    with pytest.raises(plans.EnvLivePlanError, match="gpu_shutdown_temperature_c"):
        plans.shutdown_plan(device.limits)


def test_absolute_device_runs_core_and_shutdown() -> None:
    device = _absolute()
    assert plans.CORE_ABSOLUTE in device.runnable_variants()
    assert device.decisions[plans.CORE_TLIMIT].runnable is False
    validate_plan(plans.shutdown_plan(device.limits))
    # shutdown+1 is injected on the absolute core field.
    step = plans.shutdown_plan(device.limits)["steps"][0]
    assert step["values"]["gpu_temperature_c"] == 93.0


def test_memory_falls_back_to_configured_threshold_when_device_is_silent() -> None:
    device = _h200()
    memory = device.decisions[plans.MEMORY_ABSOLUTE]
    assert memory.runnable and memory.threshold_source == "CONFIGURED_FALLBACK"
    refused = plans.memory_decision(GpuMetricsThresholds(), {}, allow_fallback=False)
    assert refused.runnable is False and "not approved" in refused.reason


def test_max_operating_only_device_warns_without_a_critical_boundary() -> None:
    decision = plans.core_tlimit_decision(
        GpuMetricsThresholds(), {"gpu_tlimit_c": 50, "gpu_max_operating_tlimit_c": 0}
    )
    assert decision.runnable is False and "max-operating" in decision.reason


def test_every_built_and_auxiliary_plan_validates_against_the_tool() -> None:
    device = _h200()
    for plan in device.plans.values():
        validate_plan(plan)
    decision = device.decisions[plans.CORE_TLIMIT]
    families = [
        plans.transient_plans(decision, confirmation_samples=3),
        plans.thermal_duration_plan(),
        plans.power_brake_plans(confirmation_samples=3),
        plans.unavailable_field_plans(decision),
    ]
    for family in families:
        for plan in family.values():
            validate_plan(plan)


def test_tlimit_severe_is_below_critical_and_warning_sits_inside_the_band() -> None:
    device = _h200()
    decision = device.decisions[plans.CORE_TLIMIT]
    severe_step = device.plans["core_tlimit/severe"]["steps"][1]
    warning_step = device.plans["core_tlimit/warning"]["steps"][1]
    assert severe_step["values"]["gpu_tlimit_c"] == -3.0
    warning = warning_step["values"]["gpu_tlimit_c"]
    assert decision.critical_value < warning <= decision.warning_value


def test_absolute_severe_is_above_critical() -> None:
    device = _absolute()
    decision = device.decisions[plans.CORE_ABSOLUTE]
    severe = device.plans["core_absolute/severe"]["steps"][1]["values"][
        "gpu_temperature_c"
    ]
    assert severe > decision.critical_value


def test_power_brake_plan_covers_transient_sustained_non_brake_and_mixed() -> None:
    family = plans.power_brake_plans(confirmation_samples=3)
    assert "transient" in family and "sustained" in family
    assert any(name.startswith("non_brake_") for name in family), family
    assert any(name.startswith("mixed_") for name in family), family
    # The mixed masks all carry 0x80.
    for mask in plans.BRAKE_MIXED_MASKS:
        assert mask & plans.POWER_BRAKE_BIT


def test_thermal_duration_plan_rises_and_clears_under_both_bits() -> None:
    family = plans.thermal_duration_plan(increments=3, step_us=1_000_000.0)
    for label, bit in (
        ("sw_thermal", plans.SW_THERMAL_BIT),
        ("hw_thermal", plans.HW_THERMAL_BIT),
    ):
        steps = family[label]["steps"]
        durations = [s["values"]["thermal_violation_total_us"] for s in steps]
        assert durations == sorted(durations)
        assert steps[0]["values"]["clock_throttle_reasons"] == bit
        assert steps[-1]["values"]["clock_throttle_reasons"] == 0


def test_unavailable_field_plans_cover_null_non_integer_and_blank() -> None:
    family = plans.unavailable_field_plans(_h200().decisions[plans.CORE_TLIMIT])
    assert family["omit_reason"]["steps"][0]["values"]["clock_throttle_reasons"] is None
    assert (
        family["non_integer_mask"]["steps"][0]["values"]["clock_throttle_reasons"]
        == 128.5
    )
    assert (
        family["dcgm_blank"]["steps"][0]["values"]["clock_throttle_reasons"] >= 9.0e18
    )


def test_thermal_stress_plan_holds_the_warning_band_and_clears() -> None:
    decision = _h200().decisions[plans.CORE_TLIMIT]
    plan = plans.thermal_stress_plan(
        decision, confirmation_samples=3, baseline_us=5_000_000.0
    )
    validate_plan(plan)
    field = plans.temperature_field(decision)
    assert field == "gpu_tlimit_c"
    warning = plans.warning_reading(decision)
    assert decision.critical_value is not None and decision.warning_value is not None
    assert decision.critical_value < warning <= decision.warning_value
    breach = [step for step in plan["steps"] if field in step["values"]]
    assert len(breach) == 6, "confirmation samples plus three"
    assert all(step["values"][field] == warning for step in breach), (
        "every breach read sits at the warning reading"
    )
    assert all(
        step["values"]["clock_throttle_reasons"] == plans.SW_THERMAL_BIT
        for step in breach
    ), "only the software thermal reason is asserted"
    durations = [step["values"]["thermal_violation_total_us"] for step in breach]
    assert durations == sorted(durations) and durations[0] > 5_000_000.0
    assert plan["steps"][0] == {"reads": 3, "values": {}}, "pass-through warm-up"
    tail = plan["steps"][-1]
    assert tail["reads"] == 3
    assert tail["values"] == {
        "thermal_violation_total_us": durations[-1],
        "clock_throttle_reasons": 0,
    }, "the tail is flat with the reason cleared and the real temperature"
    with pytest.raises(plans.EnvLivePlanError):
        plans.thermal_stress_plan(
            decision, confirmation_samples=3, reason_bit=plans.POWER_BRAKE_BIT
        )


def test_power_limit_throttling_plan_pins_draw_under_the_limit_without_reasons() -> (
    None
):
    plan = plans.power_limit_throttling_plan(
        confirmation_samples=3,
        power_limit_w=700.0,
        power_limit_ratio=0.95,
        min_utilization_percent=80.0,
        baseline_us=100.0,
    )
    validate_plan(plan)
    breach = [step for step in plan["steps"] if "power_usage_w" in step["values"]]
    assert len(breach) == 6
    for step in breach:
        values = step["values"]
        assert 700.0 * 0.95 <= values["power_usage_w"] < 700.0, values
        assert values["gpu_utilization_percent"] >= 80.0, values
        assert "clock_throttle_reasons" not in values
        assert "power_limit_w" not in values
    durations = [step["values"]["power_violation_total_us"] for step in breach]
    assert durations == sorted(durations) and durations[0] > 100.0
    assert plan["steps"][-1]["values"] == {"power_violation_total_us": durations[-1]}
    served = plans.power_limit_throttling_plan(
        confirmation_samples=3,
        power_limit_w=700.0,
        power_limit_ratio=0.95,
        min_utilization_percent=80.0,
        serve_limit=True,
    )
    assert all(
        step["values"].get("power_limit_w") == 700.0
        for step in served["steps"]
        if "power_usage_w" in step["values"]
    ), "serve_limit serves the enforced limit with every breach read"
    with pytest.raises(plans.EnvLivePlanError):
        plans.power_limit_throttling_plan(
            confirmation_samples=3,
            power_limit_w=0.0,
            power_limit_ratio=0.95,
            min_utilization_percent=80.0,
        )
