"""``spec.processor.{,fault,evidence}AdmissionFlushConcurrency`` and
``admissionBatchDelayMaxSeconds``: the site-wide caps on concurrent admission
flush transactions and the adaptive coalescing cap as AdminConfig knobs,
omitted while at their defaults so live digests hold, rendered as literals
the ingress divides by GPU_FAULT_INGRESS_WRITER_PROCESSES."""

from __future__ import annotations

import pytest

from gpu_fault.admin.config import (
    INGRESS_PROCESSES_PER_POD,
    INGRESS_REPLICAS,
    AdminConfig,
    AdminConfigError,
    ProcessorConfig,
    admin_config_spec,
    default_admin_config,
    ingress_writer_processes,
)
from gpu_fault.admin.config_patch import PRESET_NAMES, apply_patch, preset_admin_config
from gpu_fault.postgres_capacity import flush_slots_per_process
from gpu_fault_release.regional_release_rendering import (
    admin_config_renderer_environment,
)

DEFAULTS = {
    "admission_flush_concurrency": 24,
    "fault_admission_flush_concurrency": 24,
    "evidence_admission_flush_concurrency": 12,
    "admission_batch_delay_max_seconds": 0.05,
}
SPELLINGS = {
    "admission_flush_concurrency": "admissionFlushConcurrency",
    "fault_admission_flush_concurrency": "faultAdmissionFlushConcurrency",
    "evidence_admission_flush_concurrency": "evidenceAdmissionFlushConcurrency",
    "admission_batch_delay_max_seconds": "admissionBatchDelayMaxSeconds",
}
LITERALS = {
    "admission_flush_concurrency": "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY",
    "fault_admission_flush_concurrency": (
        "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY"
    ),
    "evidence_admission_flush_concurrency": (
        "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_FLUSH_CONCURRENCY"
    ),
    "admission_batch_delay_max_seconds": (
        "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_MAX_SECONDS"
    ),
}
TUNED = {
    "admissionFlushConcurrency": 36,
    "faultAdmissionFlushConcurrency": 12,
    "evidenceAdmissionFlushConcurrency": 6,
    "admissionBatchDelayMaxSeconds": 0.1,
}


def test_the_defaults_stay_out_of_records_yaml_and_digests() -> None:
    default = default_admin_config()
    for name, value in DEFAULTS.items():
        assert getattr(ProcessorConfig(), name) == value
        assert name not in default.as_dict()["processor"]
        assert SPELLINGS[name] not in admin_config_spec(default)["processor"]
    for preset_name in PRESET_NAMES:
        preset = preset_admin_config(preset_name)
        for name, value in DEFAULTS.items():
            assert getattr(preset.processor, name) == value, (
                f"preset {preset_name} keeps the release default for {name}"
            )
            assert name not in preset.as_dict()["processor"]
    assert AdminConfig.from_mapping(default.as_dict()) == default


def test_the_fleet_share_is_the_cap_divided_among_the_ingress_processes() -> None:
    assert ingress_writer_processes() == INGRESS_REPLICAS * INGRESS_PROCESSES_PER_POD
    assert ingress_writer_processes() == 12, "3 api-ha replicas x uvicorn --workers 4"
    assert default_admin_config().processor.admission_flush_slots_per_process() == {
        "routine": 2,
        "fault": 2,
        "evidence": 1,
    }, "24 + 24 + 12 site-wide, not 16 + 8 + 4 per process (336 on 12 processes)"
    assert flush_slots_per_process(24, 12) == 2
    assert flush_slots_per_process(30, 12) == 3, "a cap that does not divide rounds up"
    assert flush_slots_per_process(1, 12) == 1, "every process keeps one slot"
    assert flush_slots_per_process(16, 1) == 16, "a bare process is a fleet of one"
    with pytest.raises(ValueError, match="flush concurrency must be positive"):
        flush_slots_per_process(0, 12)
    with pytest.raises(ValueError, match="writer process count must be positive"):
        flush_slots_per_process(24, 0)


def test_non_default_caps_lay_over_a_preset_and_round_trip() -> None:
    config = apply_patch(
        default_admin_config(),
        {"capacity": {"preset": "32-enabled"}, "processor": TUNED},
    )
    assert config.processor.admission_flush_concurrency == 36
    assert config.processor.fault_admission_flush_concurrency == 12
    assert config.processor.evidence_admission_flush_concurrency == 6
    assert config.processor.admission_batch_delay_max_seconds == 0.1
    assert config.capacity.telemetry_spool.replicas == 3, "the preset still applied"
    assert config.processor.admission_flush_slots_per_process() == {
        "routine": 3,
        "fault": 1,
        "evidence": 1,
    }
    record = config.as_dict()["processor"]
    spec = admin_config_spec(config)["processor"]
    for name, camel in SPELLINGS.items():
        assert record[name] == TUNED[camel]
        assert spec[camel] == TUNED[camel]
    assert AdminConfig.from_mapping(config.as_dict()) == config
    assert apply_patch(default_admin_config(), admin_config_spec(config)) == config
    baseline = preset_admin_config("32-enabled")
    assert config.role_sha256() != baseline.role_sha256(), (
        "changed caps roll the roles: the processor section is in every role digest"
    )


@pytest.mark.parametrize(
    "field",
    [
        "admission_flush_concurrency",
        "fault_admission_flush_concurrency",
        "evidence_admission_flush_concurrency",
    ],
)
@pytest.mark.parametrize("bad", [0, -1, 1025])
def test_the_caps_are_bounded(field: str, bad: int) -> None:
    with pytest.raises(
        AdminConfigError, match=f"{SPELLINGS[field]} must be within 1..1024"
    ):
        ProcessorConfig(**{field: bad}).validate()
    ProcessorConfig(**{field: 1}).validate()
    ProcessorConfig(**{field: 1024}).validate()


@pytest.mark.parametrize("bad", [-0.001, 0.5001, 2])
def test_the_delay_cap_stays_within_the_knob_range(bad: float) -> None:
    with pytest.raises(
        AdminConfigError, match="admissionBatchDelayMaxSeconds must be within 0..0.5"
    ):
        ProcessorConfig(admission_batch_delay_max_seconds=bad).validate()
    ProcessorConfig(admission_batch_delay_max_seconds=0).validate()
    ProcessorConfig(admission_batch_delay_max_seconds=0.5).validate()
    # Raising the base delay alone stays valid: a cap at or below the base
    # means the window is fixed (the runtime takes the larger of the two).
    ProcessorConfig(
        admission_batch_delay_seconds=0.5, admission_batch_delay_max_seconds=0.05
    ).validate()


def test_the_renderer_environment_carries_the_caps_and_the_delay_cap() -> None:
    environment = admin_config_renderer_environment(default_admin_config())
    assert environment[LITERALS["admission_flush_concurrency"]] == "24"
    assert environment[LITERALS["fault_admission_flush_concurrency"]] == "24"
    assert environment[LITERALS["evidence_admission_flush_concurrency"]] == "12"
    assert environment[LITERALS["admission_batch_delay_max_seconds"]] == "0.05"
    tuned = admin_config_renderer_environment(
        apply_patch(default_admin_config(), {"processor": TUNED})
    )
    assert tuned[LITERALS["admission_flush_concurrency"]] == "36"
    assert tuned[LITERALS["fault_admission_flush_concurrency"]] == "12"
    assert tuned[LITERALS["evidence_admission_flush_concurrency"]] == "6"
    assert tuned[LITERALS["admission_batch_delay_max_seconds"]] == "0.1"
