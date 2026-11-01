"""The reviewed replay plans come from one device's live limits, no templates."""

from __future__ import annotations

import pytest

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from scripts.e2e.regional import env_live_expectations as expectations
from scripts.e2e.regional import env_live_plans as plans
from scripts.e2e.regional.env_live_watch import CollectorCadence
from tests.regional._env_live_world import h200_preflight
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


#: The same H200 once the product publishes the absolute shutdown limit DCGM
#: field 159 reports (95 C live, = (28 + 59) - (-8); batch 232).
H200_PUBLISHED = {**H200_LIMITS, "gpu_shutdown_temperature_c": 95.0}


def test_h200_runs_the_shutdown_case_on_the_limit_the_product_published() -> None:
    device = plans.build_device_plan(
        gpu_uuid="GPU-aaaa",
        gpu_index="0",
        product="NVIDIA H200",
        raw_limits=H200_PUBLISHED,
    )

    assert device.runnable_variants() == [plans.CORE_TLIMIT, plans.MEMORY_ABSOLUTE], (
        "the T.Limit rules stay authoritative: no absolute-core variant appears"
    )
    tlimit = device.decisions[plans.CORE_TLIMIT]
    assert (tlimit.critical_value, tlimit.warning_value) == (-2.0, 3.0)
    plan = plans.shutdown_plan(device.limits)
    validate_plan(plan)
    assert plan == {"steps": [{"reads": 1, "values": {"gpu_temperature_c": 96.0}}]}, (
        "one read of DCGM_FI_DEV_GPU_TEMP, the reading graded against the limit"
    )
    without_reading = plans.normalize_device_limits(
        {k: v for k, v in H200_PUBLISHED.items() if k != "gpu_tlimit_c"}
    )
    core = plans.core_absolute_decision(GpuMetricsThresholds(), without_reading)
    assert core.runnable is False, (
        "the field-159 limit is never a shutdown-minus-margin core boundary"
    )
    with pytest.raises(plans.EnvLivePlanError, match="never substitutes"):
        plans.shutdown_plan(_h200().limits)


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
        plans.thermal_duration_plan(confirmation_samples=3),
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
    family = plans.thermal_duration_plan(confirmation_samples=3, step_us=1_000_000.0)
    for label, bit in (
        ("sw_thermal", plans.SW_THERMAL_BIT),
        ("hw_thermal", plans.HW_THERMAL_BIT),
    ):
        steps = family[label]["steps"]
        durations = [s["values"]["thermal_violation_total_us"] for s in steps]
        assert durations == sorted(durations)
        assert steps[0]["values"]["clock_throttle_reasons"] == bit
        assert steps[-1]["values"]["clock_throttle_reasons"] == 0


def test_thermal_duration_plan_tail_repeats_the_confirmed_read() -> None:
    """The edge filter delivers the confirmation read and the first cleared read
    only; a rise between them surfaces on the cleared read as a fresh WARNING
    (live 2026-10-27, HyperPod). One rising read per confirmation sample, the
    tail at that read's duration."""

    for samples in (2, 3, 5):
        family = plans.thermal_duration_plan(confirmation_samples=samples)
        for plan in family.values():
            validate_plan(plan)
            steps = plan["steps"]
            rising = [s for s in steps if s["values"]["clock_throttle_reasons"]]
            assert len(rising) == samples, "one rising read per confirmation sample"
            assert all(step["reads"] == 1 for step in rising), (
                "every rising read is served once"
            )
            (tail,) = [s for s in steps if not s["values"]["clock_throttle_reasons"]]
            assert tail["reads"] == max(3, samples)
            confirmed = rising[-1]["values"]["thermal_violation_total_us"]
            assert confirmed == (samples - 1) * 1_000_000.0, "the ramp still rises"
            assert tail["values"]["thermal_violation_total_us"] == confirmed, (
                "the cleared read is a zero delta against the delivered confirmation"
            )
    with pytest.raises(plans.EnvLivePlanError, match="at least two"):
        plans.thermal_duration_plan(confirmation_samples=1)


def test_duration_expectation_refuses_a_rise_on_the_cleared_read() -> None:
    """``expectation_for`` grades the plan's shape against the node's own
    confirmation count before anything is armed."""

    preflight = h200_preflight()
    good = plans.thermal_duration_plan(confirmation_samples=3)["hw_thermal"]
    expectation = expectations.expectation_for(
        case_id="GF-ENV-LIVE-006",
        plan_name="hw_thermal",
        plan=good,
        preflight=preflight,
        cadence=CollectorCadence(),
        node="worker-1",
    )
    assert expectation.kind == "duration" and expectation.planned_reads == 6
    assert expectation.trigger_values == (float(plans.HW_THERMAL_BIT),)
    # The shape that failed live: one rising read more than the confirmation
    # samples, so the cleared read carried 3e6 over the delivered 2e6.
    live = {
        "steps": [
            *good["steps"][:3],
            {
                "reads": 1,
                "values": {
                    "thermal_violation_total_us": 3_000_000.0,
                    "clock_throttle_reasons": plans.HW_THERMAL_BIT,
                },
            },
            {
                "reads": 3,
                "values": {
                    "thermal_violation_total_us": 3_000_000.0,
                    "clock_throttle_reasons": 0,
                },
            },
        ]
    }
    with pytest.raises(ValueError, match="read 5 clears the clock reason") as caught:
        expectations.expectation_for(
            case_id="GF-ENV-LIVE-006",
            plan_name="hw_thermal",
            plan=live,
            preflight=preflight,
            cadence=CollectorCadence(),
            node="worker-1",
        )
    assert "read 3 (the 3-sample confirmation) with 2e+06" in str(caught.value)
    # A node that confirms on fewer samples than the plan was built for would
    # deliver an earlier read: the same rise, refused the same way.
    with pytest.raises(ValueError, match="2-sample confirmation"):
        expectations.expectation_for(
            case_id="GF-ENV-LIVE-006",
            plan_name="hw_thermal",
            plan=good,
            preflight=preflight,
            cadence=CollectorCadence(edge_confirmation_samples=2),
            node="worker-1",
        )


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
    assert len(breach) == 3, "one rising read per confirmation sample"
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
    }, "the tail repeats the confirmed read with the reason cleared"
    with pytest.raises(plans.EnvLivePlanError):
        plans.thermal_stress_plan(
            decision, confirmation_samples=3, reason_bit=plans.POWER_BRAKE_BIT
        )
    with pytest.raises(plans.EnvLivePlanError, match="confirmation sample"):
        plans.thermal_stress_plan(decision, confirmation_samples=0)


@pytest.mark.parametrize("samples", [1, 2, 3, 5])
def test_ramp_plans_rise_on_the_confirmation_samples_and_hold_the_confirmed_read(
    samples: int,
) -> None:
    """The edge filter delivers the confirmation read and the first cleared read
    only (GF-ENV-LIVE-006 live 2026-10-27): the 024 plans rise once per
    confirmation sample and their tail is a zero delta against that read."""

    decision = _h200().decisions[plans.CORE_TLIMIT]
    thermal = plans.thermal_stress_plan(
        decision, confirmation_samples=samples, baseline_us=2_000_000.0
    )
    power = plans.power_limit_throttling_plan(
        confirmation_samples=samples,
        power_limit_w=700.0,
        power_limit_ratio=0.95,
        min_utilization_percent=80.0,
        baseline_us=100.0,
    )
    for plan, counter, marker in (
        (thermal, "thermal_violation_total_us", "clock_throttle_reasons"),
        (power, "power_violation_total_us", "power_usage_w"),
    ):
        validate_plan(plan)
        rising = [s for s in plan["steps"] if s["values"].get(marker)]
        assert len(rising) == samples, "one rising read per confirmation sample"
        assert all(step["reads"] == 1 for step in rising), (
            "every rising read is served once"
        )
        values = [s["values"][counter] for s in rising]
        assert values == sorted(values) and len(set(values)) == samples, (
            "the ramp still rises"
        )
        (tail,) = [
            s for s in plan["steps"] if s["values"] and not s["values"].get(marker)
        ]
        assert tail["reads"] == max(3, samples)
        assert tail["values"][counter] == values[-1], (
            "the cleared read is a zero delta against the delivered confirmation"
        )
        expectations.require_counter_tail_at_confirmed_read(
            plan,
            samples,
            counter=counter,
            marker=marker,
            case_id="GF-GENERAL-LIVE-024",
            plan_name="shape",
        )


def _old_thermal_shape(baseline_us: float = 0.0) -> dict[str, object]:
    """The shape both 024 plans had live on 2026-10-26: confirmation samples
    plus three rising reads, the tail holding the last of them."""

    return {
        "steps": [
            {"reads": 3, "values": {}},
            *(
                {
                    "reads": 1,
                    "values": {
                        "gpu_tlimit_c": 0.5,
                        "thermal_violation_total_us": baseline_us + index * 1e6,
                        "clock_throttle_reasons": plans.SW_THERMAL_BIT,
                    },
                }
                for index in range(1, 7)
            ),
            {
                "reads": 3,
                "values": {
                    "thermal_violation_total_us": baseline_us + 6e6,
                    "clock_throttle_reasons": 0,
                },
            },
        ]
    }


def test_counter_tail_guard_refuses_a_rise_between_confirmation_and_clear() -> None:
    def refuse(plan: dict[str, object], samples: int, **fields: str) -> str:
        with pytest.raises(ValueError) as caught:
            expectations.require_counter_tail_at_confirmed_read(
                plan, samples, case_id="GF-GENERAL-LIVE-024", plan_name="t", **fields
            )
        return str(caught.value)

    thermal = {
        "counter": "thermal_violation_total_us",
        "marker": "clock_throttle_reasons",
    }
    message = refuse(_old_thermal_shape(), 3, **thermal)
    # Warm-up reads count: the confirmation is read 6, the cleared read 10.
    assert "read 10 clears clock_throttle_reasons and serves 6e+06" in message
    assert "read 6 (the 3-sample confirmation) with 3e+06" in message
    assert "reads [7, 8, 9, 10] change the counter" in message
    # A plan built for three samples on a node that confirms on two: read 5
    # is delivered with 2e6, read 6 raises it.
    decision = _h200().decisions[plans.CORE_TLIMIT]
    good = plans.thermal_stress_plan(decision, confirmation_samples=3)
    message = refuse(good, 2, **thermal)
    assert "read 5 (the 2-sample confirmation) with 2e+06" in message
    assert "reads [6, 7] change the counter" in message
    # Too few candidate reads for the node's count: nothing would be delivered.
    message = refuse(good, 4, **thermal)
    assert "no 4 consecutive reads serve a non-zero clock_throttle_reasons" in message
    # A tail that drops the counter passes the real reset through.
    reset = plans.thermal_stress_plan(decision, confirmation_samples=3)
    reset["steps"][-1]["values"].pop("thermal_violation_total_us")
    message = refuse(reset, 3, **thermal)
    assert "omits it for thermal_violation_total_us" in message
    # The power plan is judged on its draw-at-the-cap marker.
    power = plans.power_limit_throttling_plan(
        confirmation_samples=3,
        power_limit_w=700.0,
        power_limit_ratio=0.95,
        min_utilization_percent=80.0,
    )
    power["steps"].insert(4, {"reads": 1, "values": dict(power["steps"][3]["values"])})
    power["steps"][4]["values"]["power_violation_total_us"] = 4e6
    power["steps"][-1]["values"]["power_violation_total_us"] = 4e6
    message = refuse(
        power, 3, counter="power_violation_total_us", marker="power_usage_w"
    )
    assert "read 8 clears power_usage_w and serves 4e+06" in message
    # Plans that replay neither field are not this guard's business.
    expectations.require_counter_tail_at_confirmed_read(
        plans.power_brake_plans(confirmation_samples=3)["transient"],
        3,
        counter="thermal_violation_total_us",
        marker="power_usage_w",
        case_id="GF-ENV-LIVE-007",
        plan_name="transient",
    )


def test_served_reads_lines_every_field_up_per_read() -> None:
    reads = expectations.served_reads(_old_thermal_shape())
    assert len(reads) == 12 and reads[:3] == [{}, {}, {}]
    assert reads[3]["gpu_tlimit_c"] == 0.5 and reads[8]["clock_throttle_reasons"] == 32
    assert reads[9] == {"thermal_violation_total_us": 6e6, "clock_throttle_reasons": 0}
    assert expectations.served_reads(
        {"steps": [{"reads": 2, "values": {"a": None}}]}
    ) == [{"a": None}, {"a": None}]


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
    assert len(breach) == 3, "one rising read per confirmation sample"
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
