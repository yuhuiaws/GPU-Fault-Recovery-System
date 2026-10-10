"""``spec.processor.{,fault,evidence}AdmissionFlushConcurrency`` (the site-wide
CEILING on concurrent admission flush transactions), its ``...Floor`` (where
every ingress process starts and what moderate load runs at) and
``admissionBatchDelayMaxSeconds`` as AdminConfig knobs: omitted while at their
defaults so live digests hold, validated floor <= ceiling, rendered as literals
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

# Live calibration on the 32-cluster integrated suite: a fixed cap of 24 (2 per
# process) ran at its cap for the whole load start and expired 4,601 requests
# (18.7 % reserved 503s, 517 req/s, 0.59 s per flush); 96/48 admitted the same
# burst at 1,051 req/s with no expiry and 81 ms per flush but was still binding
# (in_flight_max 8 everywhere) against the 1,428-1,542 req/s the site did with
# 16 slots per process before any cap existed, which schema v22 made safe again.
DEFAULTS = {
    "admission_flush_concurrency": 192,
    "admission_flush_concurrency_floor": 48,
    "fault_admission_flush_concurrency": 96,
    "fault_admission_flush_concurrency_floor": 24,
    "evidence_admission_flush_concurrency": 24,
    "evidence_admission_flush_concurrency_floor": 12,
    "admission_batch_delay_max_seconds": 0.05,
}
SPELLINGS = {
    "admission_flush_concurrency": "admissionFlushConcurrency",
    "admission_flush_concurrency_floor": "admissionFlushConcurrencyFloor",
    "fault_admission_flush_concurrency": "faultAdmissionFlushConcurrency",
    "fault_admission_flush_concurrency_floor": "faultAdmissionFlushConcurrencyFloor",
    "evidence_admission_flush_concurrency": "evidenceAdmissionFlushConcurrency",
    "evidence_admission_flush_concurrency_floor": (
        "evidenceAdmissionFlushConcurrencyFloor"
    ),
    "admission_batch_delay_max_seconds": "admissionBatchDelayMaxSeconds",
}
LITERALS = {
    "admission_flush_concurrency": "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY",
    "admission_flush_concurrency_floor": (
        "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_FLOOR"
    ),
    "fault_admission_flush_concurrency": (
        "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY"
    ),
    "fault_admission_flush_concurrency_floor": (
        "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY_FLOOR"
    ),
    "evidence_admission_flush_concurrency": (
        "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_FLUSH_CONCURRENCY"
    ),
    "evidence_admission_flush_concurrency_floor": (
        "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_FLUSH_CONCURRENCY_FLOOR"
    ),
    "admission_batch_delay_max_seconds": (
        "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_MAX_SECONDS"
    ),
}
TUNED = {
    "admissionFlushConcurrency": 36,
    "admissionFlushConcurrencyFloor": 12,
    "faultAdmissionFlushConcurrency": 12,
    "faultAdmissionFlushConcurrencyFloor": 6,
    "evidenceAdmissionFlushConcurrency": 6,
    "evidenceAdmissionFlushConcurrencyFloor": 3,
    "admissionBatchDelayMaxSeconds": 0.1,
}
TIERS = {
    "admission": ("admission_flush_concurrency_floor", "admission_flush_concurrency"),
    "faultAdmission": (
        "fault_admission_flush_concurrency_floor",
        "fault_admission_flush_concurrency",
    ),
    "evidenceAdmission": (
        "evidence_admission_flush_concurrency_floor",
        "evidence_admission_flush_concurrency",
    ),
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


def test_the_fleet_share_is_each_value_divided_among_the_ingress_processes() -> None:
    assert ingress_writer_processes() == INGRESS_REPLICAS * INGRESS_PROCESSES_PER_POD
    assert ingress_writer_processes() == 12, "3 api-ha replicas x uvicorn --workers 4"
    processor = default_admin_config().processor
    assert processor.admission_flush_ranges() == {
        "admission": (48, 192),
        "faultAdmission": (24, 96),
        "evidenceAdmission": (12, 24),
    }
    assert processor.admission_flush_slots_per_process() == {
        "routine": (4, 16),
        "fault": (2, 8),
        "evidence": (1, 2),
    }, "floor and ceiling per process; 16 routine is the pre-cap value v22 made safe"
    assert flush_slots_per_process(192, 12) == 16
    assert flush_slots_per_process(30, 12) == 3, "a cap that does not divide rounds up"
    assert flush_slots_per_process(1, 12) == 1, "every process keeps one slot"
    assert flush_slots_per_process(16, 1) == 16, "a bare process is a fleet of one"
    with pytest.raises(ValueError, match="flush concurrency must be positive"):
        flush_slots_per_process(0, 12)
    with pytest.raises(ValueError, match="writer process count must be positive"):
        flush_slots_per_process(24, 0)


def test_non_default_ranges_lay_over_a_preset_and_round_trip() -> None:
    config = apply_patch(
        default_admin_config(),
        {"capacity": {"preset": "32-enabled"}, "processor": TUNED},
    )
    assert config.processor.admission_flush_ranges() == {
        "admission": (12, 36),
        "faultAdmission": (6, 12),
        "evidenceAdmission": (3, 6),
    }
    assert config.processor.admission_batch_delay_max_seconds == 0.1
    assert config.capacity.telemetry_spool.replicas == 3, "the preset still applied"
    assert config.processor.admission_flush_slots_per_process() == {
        "routine": (1, 3),
        "fault": (1, 1),
        "evidence": (1, 1),
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
        "changed ranges roll the roles: the processor section is in every role digest"
    )


@pytest.mark.parametrize("field", [name for name in SPELLINGS if "flush" in name])
@pytest.mark.parametrize("bad", [0, -1, 1025])
def test_the_ceilings_and_floors_are_bounded(field: str, bad: int) -> None:
    with pytest.raises(
        AdminConfigError, match=f"{SPELLINGS[field]} must be within 1..1024"
    ):
        ProcessorConfig(**{field: bad}).validate()
    stem = field.replace("_floor", "")
    # The extremes are valid as long as the floor stays at or below the ceiling.
    ProcessorConfig(**{stem: 1, f"{stem}_floor": 1}).validate()
    ProcessorConfig(**{stem: 1024, f"{stem}_floor": 1024}).validate()
    ProcessorConfig(**{stem: 1024, f"{stem}_floor": 1}).validate()


@pytest.mark.parametrize("spelling", sorted(TIERS))
def test_a_floor_above_its_ceiling_is_refused(spelling: str) -> None:
    floor_field, ceiling_field = TIERS[spelling]
    with pytest.raises(
        AdminConfigError,
        match=(
            f"spec.processor.{spelling}FlushConcurrencyFloor must not exceed "
            f"{spelling}FlushConcurrency"
        ),
    ):
        ProcessorConfig(**{ceiling_field: 24, floor_field: 25}).validate()
    ProcessorConfig(**{ceiling_field: 24, floor_field: 24}).validate()
    # Lowering a ceiling below the default floor needs the floor lowered too:
    # the parser rejects it instead of silently inverting the range.
    with pytest.raises(AdminConfigError, match="must not exceed"):
        apply_patch(
            default_admin_config(), {"processor": {f"{spelling}FlushConcurrency": 6}}
        )


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


def test_the_renderer_environment_carries_the_ranges_and_the_delay_cap() -> None:
    environment = admin_config_renderer_environment(default_admin_config())
    for name, value in DEFAULTS.items():
        assert environment[LITERALS[name]] == str(value), name
    tuned = admin_config_renderer_environment(
        apply_patch(default_admin_config(), {"processor": TUNED})
    )
    for name, camel in SPELLINGS.items():
        assert tuned[LITERALS[name]] == str(TUNED[camel]), name
