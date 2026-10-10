"""``spec.processor.admissionBatchDelaySeconds``: the ingress admission batch
delay as an AdminConfig knob, omitted while at its default so live digests
hold, rendered to GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_SECONDS."""

from __future__ import annotations

import pytest

from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    ProcessorConfig,
    admin_config_spec,
    default_admin_config,
)
from gpu_fault.admin.config_patch import PRESET_NAMES, apply_patch, preset_admin_config
from gpu_fault_release.regional_release_rendering import (
    admin_config_renderer_environment,
)


def test_the_default_stays_out_of_records_yaml_and_digests() -> None:
    assert ProcessorConfig().admission_batch_delay_seconds == 0.002
    default = default_admin_config()
    assert "admission_batch_delay_seconds" not in default.as_dict()["processor"]
    assert "admissionBatchDelaySeconds" not in admin_config_spec(default)["processor"]
    for name in PRESET_NAMES:
        preset = preset_admin_config(name)
        assert preset.processor.admission_batch_delay_seconds == 0.002, (
            f"preset {name} keeps the release default"
        )
        assert "admission_batch_delay_seconds" not in preset.as_dict()["processor"]
    assert AdminConfig.from_mapping(default.as_dict()) == default


def test_a_non_default_delay_lays_over_a_preset_and_round_trips() -> None:
    config = apply_patch(
        default_admin_config(),
        {
            "capacity": {"preset": "32-enabled"},
            "processor": {"admissionBatchDelaySeconds": 0.05},
        },
    )
    assert config.processor.admission_batch_delay_seconds == 0.05
    assert config.capacity.telemetry_spool.replicas == 3, "the preset still applied"
    assert config.as_dict()["processor"]["admission_batch_delay_seconds"] == 0.05
    assert admin_config_spec(config)["processor"]["admissionBatchDelaySeconds"] == 0.05
    assert AdminConfig.from_mapping(config.as_dict()) == config
    baseline = preset_admin_config("32-enabled")
    assert config.role_sha256() != baseline.role_sha256(), (
        "a changed delay rolls the roles: the processor section is in every role digest"
    )
    assert apply_patch(default_admin_config(), admin_config_spec(config)) == config


@pytest.mark.parametrize("bad", [-0.001, 0.5001, 2])
def test_the_delay_is_bounded(bad: float) -> None:
    with pytest.raises(
        AdminConfigError, match="admissionBatchDelaySeconds must be within"
    ):
        ProcessorConfig(admission_batch_delay_seconds=bad).validate()
    ProcessorConfig(admission_batch_delay_seconds=0).validate()
    ProcessorConfig(admission_batch_delay_seconds=0.5).validate()


def test_the_renderer_environment_carries_the_delay() -> None:
    assert (
        admin_config_renderer_environment(default_admin_config())[
            "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_SECONDS"
        ]
        == "0.002"
    )
    tuned = apply_patch(
        default_admin_config(), {"processor": {"admissionBatchDelaySeconds": 0.05}}
    )
    assert (
        admin_config_renderer_environment(tuned)[
            "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_SECONDS"
        ]
        == "0.05"
    )
