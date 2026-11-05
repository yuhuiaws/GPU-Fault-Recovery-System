"""GF-ENV-LIVE plans: limit normalisation, threshold decisions and plan refusals."""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from scripts.e2e.regional import env_live_plans as plans
from scripts.e2e.regional.env_live_plans import EnvLivePlanError
from tests.regional.test_env_live_plans import H200_LIMITS
from tools.gpu_environment_replay import validate_plan

THRESHOLDS = GpuMetricsThresholds()


def h200_decision() -> plans.VariantDecision:
    return plans.build_device_plan(
        gpu_uuid="GPU-aaaa",
        gpu_index="0",
        product="NVIDIA H200",
        raw_limits=H200_LIMITS,
    ).decisions[plans.CORE_TLIMIT]


def test_normalize_skips_booleans_and_fields_the_policy_does_not_read() -> None:
    assert plans.normalize_device_limits(
        {
            "gpu_slowdown_temperature_c": True,
            "gpu_temperature_c": 40,
            "power_usage_w": 300.0,
            "memory_temperature_c": "51",
        }
    ) == {"gpu_temperature_c": 40.0}


def test_a_shutdown_only_tlimit_device_derives_its_critical_boundary() -> None:
    limits = {"gpu_tlimit_c": 50.0, "gpu_shutdown_tlimit_c": -8.0}

    decision = plans.core_tlimit_decision(THRESHOLDS, limits)

    assert decision.runnable is True
    assert (
        decision.critical_value == -8.0 + THRESHOLDS.gpu_temperature_shutdown_margin_c
    )
    assert decision.warning_value == (
        decision.critical_value + THRESHOLDS.gpu_temperature_warning_margin_c
    )


def test_a_tlimit_reading_without_a_gpu_threshold_is_not_graded() -> None:
    decision = plans.core_tlimit_decision(
        THRESHOLDS, {"gpu_tlimit_c": 50.0, "memory_max_operating_tlimit_c": 0.0}
    )

    assert decision.runnable is False
    assert decision.reason.startswith("device publishes no GPU T.Limit threshold"), (
        decision.reason
    )


def test_an_absolute_max_operating_only_device_has_no_critical_boundary() -> None:
    decision = plans.core_absolute_decision(
        THRESHOLDS, {"gpu_temperature_c": 40.0, "gpu_max_operating_temperature_c": 83.0}
    )

    assert decision.runnable is False
    assert decision.reason.startswith(
        "device publishes only an absolute max-operating temperature"
    ), decision.reason
    assert decision.scale == "absolute"


def test_a_confirmation_plan_needs_at_least_one_severe_read() -> None:
    decision = h200_decision()
    validate_plan(plans.confirmation_plan(decision, confirmation_samples=3))

    with pytest.raises(EnvLivePlanError, match="a replay step needs at least one read"):
        plans.confirmation_plan(decision, confirmation_samples=0)


def test_a_thermal_warning_plan_needs_a_clearing_read() -> None:
    with pytest.raises(
        EnvLivePlanError, match="a thermal warning plan needs a clearing read"
    ):
        plans.thermal_stress_plan(
            h200_decision(), confirmation_samples=3, clear_reads=0
        )


def power_plan(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "confirmation_samples": 3,
        "power_limit_w": 700.0,
        "power_limit_ratio": 0.95,
        "min_utilization_percent": 80.0,
        **overrides,
    }
    return plans.power_limit_throttling_plan(**arguments)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"power_limit_ratio": 0.0}, "power_limit_ratio must be in \\(0, 1\\]"),
        ({"power_limit_ratio": 1.2}, "power_limit_ratio must be in \\(0, 1\\]"),
        ({"confirmation_samples": 0}, "a power cap plan needs a confirmation sample"),
        ({"clear_reads": 0}, "a power cap plan needs a clearing read"),
    ],
)
def test_a_power_cap_plan_refuses_an_impossible_shape(
    overrides: dict[str, Any], message: str
) -> None:
    validate_plan(power_plan(power_limit_ratio=1.0))

    with pytest.raises(EnvLivePlanError, match=message):
        power_plan(**overrides)


def test_a_thermal_duration_plan_needs_a_clearing_read() -> None:
    with pytest.raises(
        EnvLivePlanError, match="a thermal duration plan needs a clearing read"
    ):
        plans.thermal_duration_plan(confirmation_samples=3, clear_reads=0)
