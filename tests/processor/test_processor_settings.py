"""S15: the processor's 44 flat constructor keywords become four typed groups.

``ProcessorCoordinator.__init__`` used to take 45 keyword parameters and the
factory fed it four untyped ``dict``s through ``**kwargs``, so a renamed
setting failed only when the process started. The groups below carry the same
field names and defaults the constructor had, own the validation that belongs
to their fields, and are passed to the coordinator by name so mypy checks each
one statically.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gpu_fault.app.processor_factory import ProcessorFactory
from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
    ProcessorSpoolSettings,
    ProcessorStaleSettings,
)
from tests._builders import build_store

TOKEN = "processor-token-" + "x" * 32
REPLAY_SECRET = "replay-secret-" + "y" * 32


def _coordinator(**groups: object) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        build_store(),
        owner_id="pod-a:1",
        internal_token=TOKEN,
        active_consumers=True,
        **groups,
    )


# --- defaults -------------------------------------------------------------


def test_lease_defaults_match_the_former_constructor_defaults() -> None:
    assert asdict(ProcessorLeaseSettings()) == {
        "lease_seconds": 15,
        "renew_seconds": 3,
        "request_lease_seconds": 120,
        "request_renew_seconds": 5,
        "request_max_execution_seconds": 30,
        "deadline_exceeded_process_threshold": 3,
        "unhealthy_ttl_seconds": 300,
        "retryable_response_max_age_seconds": 300,
        "retry_backoff_seconds": 1,
        "retry_backoff_max_seconds": 30,
        "poll_seconds": 0.1,
        "idle_backoff_max_seconds": 2.0,
        "busy_backoff_max_seconds": 0.4,
        "fault_idle_backoff_max_seconds": 0.5,
        "fault_busy_backoff_max_seconds": 0.1,
        "processor_notification_fallback_seconds": 5.0,
        "processor_notification_shard_count": 8,
    }


def test_pool_defaults_match_the_former_constructor_defaults() -> None:
    assert asdict(ProcessorPoolSettings()) == {
        "worker_count": 4,
        "fault_worker_count": None,
        "observation_worker_count": None,
        "gpu_telemetry_worker_count": None,
        "host_telemetry_worker_count": None,
        "fault_pressure_evidence_workers": 1,
        "routine_starvation_seconds": 30.0,
    }


def test_stale_defaults_match_the_former_constructor_defaults() -> None:
    assert asdict(ProcessorStaleSettings()) == {
        "gpu_inventory_stale_seconds": 180,
        "health_summary_stale_seconds": 420,
        "observation_stale_seconds": 120,
        "training_progress_stale_seconds": 120,
    }


def test_spool_defaults_match_the_former_constructor_defaults() -> None:
    assert asdict(ProcessorSpoolSettings()) == {
        "telemetry_spool_enabled": False,
        "telemetry_spool_workers": 4,
        "telemetry_spool_lease_seconds": 60,
        "telemetry_spool_retry_backoff_seconds": 1.0,
        "telemetry_spool_notification_fallback_seconds": 5.0,
        "telemetry_spool_fault_pressure_workers": 1,
        "telemetry_spool_fault_pressure_poll_seconds": 0.5,
        "telemetry_spool_max_in_flight_bytes": 64 * 1024 * 1024,
        "telemetry_spool_replay_batch_max_items": 64,
        "telemetry_spool_replay_batch_max_bytes": 8 * 1024 * 1024,
    }


# --- from_environment -----------------------------------------------------


def test_lease_settings_read_their_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_LEADER_LEASE_SECONDS", "20")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_LEADER_RENEW_SECONDS", "4")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", "12")
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES", raising=False)

    settings = ProcessorLeaseSettings.from_environment()

    assert settings.lease_seconds == 20
    assert settings.renew_seconds == 4.0
    assert settings.processor_notification_shard_count == 12
    assert settings.poll_seconds == 0.1


def test_pool_settings_read_their_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_WORKERS", "16")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS", "6")
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_FAULT_WORKERS", raising=False)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_ROUTINE_STARVATION_SECONDS", "12")

    settings = ProcessorPoolSettings.from_environment()

    assert settings.worker_count == 16
    assert settings.default_pool == 4
    assert settings.fault_worker_count == 4
    assert settings.gpu_telemetry_worker_count == 6
    assert settings.routine_starvation_seconds == 12.0


def test_stale_settings_read_their_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_HEALTH_SUMMARY_STALE_SECONDS", "900")

    settings = ProcessorStaleSettings.from_environment()

    assert settings.health_summary_stale_seconds == 900.0
    assert settings.gpu_inventory_stale_seconds == 180.0


def test_spool_settings_read_their_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "yes")
    monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL_WORKERS", raising=False)
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_LEASE_SECONDS", "90")

    settings = ProcessorSpoolSettings.from_environment(default_pool=3)

    assert settings.telemetry_spool_enabled is True
    assert settings.telemetry_spool_workers == 6
    assert settings.telemetry_spool_lease_seconds == 90.0


# --- group-local validation -----------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lease_seconds": 4}, "processor leader lease must be at least 5s"),
        ({"renew_seconds": 15}, "processor renew interval must be below its lease"),
        (
            {"request_renew_seconds": 120},
            "processor request renew interval must be below its lease",
        ),
        (
            {"request_max_execution_seconds": 0},
            "processor request maximum execution time must be positive",
        ),
        (
            {"retryable_response_max_age_seconds": 0},
            "processor stale limits must be positive: retryable response",
        ),
        (
            {"retry_backoff_max_seconds": 0.5},
            "processor retry backoff must be positive and not exceed its maximum",
        ),
        (
            {"processor_notification_fallback_seconds": 0},
            "processor notification fallback must be positive",
        ),
        (
            {"processor_notification_shard_count": 0},
            "processor notification shard count must be positive",
        ),
    ],
)
def test_lease_settings_reject_their_own_bad_values(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ProcessorLeaseSettings(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"worker_count": 0}, "processor worker count must be positive"),
        (
            {"routine_starvation_seconds": 0},
            "processor routine starvation bound must be positive",
        ),
        (
            {"fault_worker_count": 0, "gpu_telemetry_worker_count": 2},
            "processor fault workers must be positive and "
            "other pool workers cannot be negative",
        ),
        (
            {"fault_worker_count": 1, "observation_worker_count": -1},
            "processor fault workers must be positive and "
            "other pool workers cannot be negative",
        ),
        (
            {
                "fault_worker_count": 1,
                "gpu_telemetry_worker_count": 2,
                "host_telemetry_worker_count": 4,
                "fault_pressure_evidence_workers": 3,
            },
            "processor fault-pressure evidence workers must be "
            "positive and not exceed a dedicated evidence pool",
        ),
    ],
)
def test_pool_settings_reject_their_own_bad_values(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ProcessorPoolSettings(**overrides)


def test_pool_settings_derive_the_worker_split() -> None:
    assert ProcessorPoolSettings(worker_count=3).worker_counts() == {
        "fault": 3,
        "observation": 0,
        "gpu": 0,
        "host": 0,
    }
    explicit = ProcessorPoolSettings(
        fault_worker_count=1, host_telemetry_worker_count=2
    )
    assert explicit.worker_counts() == {
        "fault": 1,
        "observation": 0,
        "gpu": 0,
        "host": 2,
    }


def test_stale_settings_name_every_non_positive_limit() -> None:
    with pytest.raises(
        ValueError,
        match="processor stale limits must be positive: "
        "GPU inventory, training progress",
    ):
        ProcessorStaleSettings(
            gpu_inventory_stale_seconds=0, training_progress_stale_seconds=-1
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"telemetry_spool_enabled": True, "telemetry_spool_workers": 0},
            "telemetry spool workers must be positive when the spool is enabled",
        ),
        (
            {"telemetry_spool_notification_fallback_seconds": 1},
            "telemetry spool notification fallback must be between 2 and 5 seconds",
        ),
        (
            {"telemetry_spool_fault_pressure_workers": 5},
            "telemetry spool fault-pressure workers must be "
            "positive and not exceed normal spool workers",
        ),
        (
            {"telemetry_spool_fault_pressure_poll_seconds": 0},
            "telemetry spool fault-pressure poll interval must be positive",
        ),
        (
            {"telemetry_spool_max_in_flight_bytes": 0},
            "telemetry spool in-flight byte limit must be positive",
        ),
        (
            {"telemetry_spool_replay_batch_max_items": 65},
            "telemetry spool replay batch item limit must be between 1 and 64",
        ),
        (
            {"telemetry_spool_replay_batch_max_bytes": 65 * 1024 * 1024},
            "telemetry spool replay batch byte limit must be "
            "positive and not exceed the in-flight byte limit",
        ),
    ],
)
def test_spool_settings_reject_their_own_bad_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ProcessorSpoolSettings(**overrides)


# --- the coordinator surface ----------------------------------------------


def test_coordinator_takes_groups_not_a_flat_keyword_surface() -> None:
    parameters = inspect.signature(ProcessorCoordinator.__init__).parameters

    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ), "expected ProcessorCoordinator.__init__ to take no **kwargs"
    assert len(parameters) <= 12
    assert {"lease", "pools", "stale", "spool"} <= set(parameters)
    with pytest.raises(TypeError, match="lease_seconds"):
        _coordinator(lease_seconds=20)


def test_coordinator_keeps_the_groups_and_mirrors_their_fields() -> None:
    lease = ProcessorLeaseSettings(lease_seconds=20, renew_seconds=4)
    pools = ProcessorPoolSettings(fault_worker_count=1, gpu_telemetry_worker_count=2)
    stale = ProcessorStaleSettings(health_summary_stale_seconds=900)
    spool = ProcessorSpoolSettings(telemetry_spool_enabled=True)

    processor = _coordinator(lease=lease, pools=pools, stale=stale, spool=spool)

    assert processor.lease is lease
    assert processor.pools is pools
    assert processor.stale is stale
    assert processor.spool is spool
    assert processor.lease_seconds == 20
    assert processor.renew_seconds == 4
    assert processor.worker_counts == {
        "fault": 1,
        "observation": 0,
        "gpu": 2,
        "host": 0,
    }
    assert processor.worker_count == 3
    assert processor.health_summary_stale_seconds == 900
    assert processor.telemetry_spool_enabled is True


def test_factory_passes_statically_named_groups(monkeypatch) -> None:
    monkeypatch.setenv("POD_UID", "pod-settings")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_WORKERS", "8")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_FAULT_WORKERS", "3")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_HEALTH_SUMMARY_STALE_SECONDS", "900")
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES", raising=False)
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", raising=False)
    monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL_WORKERS", raising=False)
    context = SimpleNamespace(
        store=build_store(),
        execution_token=TOKEN,
        processor_replay_secret=REPLAY_SECRET,
    )

    processor = ProcessorFactory(
        context, mode="active-active", exit_grace_seconds=0
    ).build()

    assert processor is not None
    assert processor.pools == ProcessorPoolSettings.from_environment()
    assert processor.pools.fault_worker_count == 3
    assert processor.worker_counts["fault"] == 3
    assert processor.stale.health_summary_stale_seconds == 900.0
    assert processor.spool.telemetry_spool_workers == 4
    assert processor.lease == ProcessorLeaseSettings.from_environment()
