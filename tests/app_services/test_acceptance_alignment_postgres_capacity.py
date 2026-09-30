from __future__ import annotations

import math

import pytest

from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    CapacityConfig,
    TelemetrySpoolCapacity,
)
from gpu_fault.admin.config_patch import apply_patch, preset_admin_config
from gpu_fault.app.admission_runtime import AdmissionRuntimeFactory
from gpu_fault.app.admission_runtime import PostgresPoolCapacity as RuntimeCapacity
from gpu_fault.postgres_capacity import PostgresPoolCapacity
from scripts.e2e.regional.capacity_connection_budget import unpooled_connection_budget


@pytest.mark.parametrize("role", ["all", "worker", "ingress", "spool-worker"])
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("spool", [False, True])
@pytest.mark.parametrize("dispatcher", [False, True])
@pytest.mark.parametrize("regional", [False, True])
def test_shared_listener_budget_follows_role_and_actual_enabled_consumers(
    role, queued, spool, dispatcher, regional, monkeypatch
) -> None:
    listeners = PostgresPoolCapacity.listener_connections(
        role,
        queued_processor=queued,
        spool_enabled=spool,
        workflow_dispatcher_enabled=dispatcher,
        regional=regional,
    )
    background = role in {"all", "worker"}
    expected = (
        int(queued and background)
        + int(queued and spool and role in {"all", "spool-worker"})
        + 2 * int(background and dispatcher)
        + int(regional)
    )
    assert sum(listeners.values()) == expected
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://isolated.invalid/db")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
    monkeypatch.setenv(
        "GPU_FAULT_PROCESSOR_MODE", "active-active" if queued else "direct"
    )
    monkeypatch.setenv("GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER", str(dispatcher))
    monkeypatch.setenv("GPU_FAULT_DEPLOYMENT_MODE", "regional" if regional else "local")
    runtime = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": spool})
    assert runtime is not None and runtime.unpooled_connections == expected
    assert PostgresPoolCapacity.role_connection_ceiling(
        pool_max=20,
        processes_per_pod=4,
        replicas=3,
        service_role=role,
        queued_processor=queued,
        spool_enabled=spool,
        workflow_dispatcher_enabled=dispatcher,
        regional=regional,
    ) == 12 * (20 + expected)


def test_old_runtime_import_uses_the_same_budget_api() -> None:
    assert RuntimeCapacity is PostgresPoolCapacity
    for role in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ):
        assert unpooled_connection_budget(
            role
        ) == PostgresPoolCapacity.listener_connections(
            {
                "gpu-fault-api-ha": "ingress",
                "gpu-fault-control-worker": "worker",
                "gpu-fault-telemetry-spool-worker": "spool-worker",
            }[role],
            queued_processor=True,
            spool_enabled=True,
            workflow_dispatcher_enabled=True,
            regional=True,
        )


def test_listener_connections_do_not_consume_pool_headroom() -> None:
    capacity = PostgresPoolCapacity(8, {"workers": 6}, 4)
    assert capacity.has_headroom and capacity.headroom == 2
    assert capacity.demand == 6 and capacity.oversubscription_ratio == 0.75
    assert not PostgresPoolCapacity(7, {"workers": 6}, 4).has_headroom, (
        "one spare slot is insufficient"
    )
    assert math.isinf(PostgresPoolCapacity(0, {}, 0).oversubscription_ratio), (
        "zero pool capacity must not look usable"
    )


def test_admin_validated_budget_is_not_raised_to_hide_missing_listeners() -> None:
    baseline = CapacityConfig()
    assert baseline.postgres_connection_ceiling() == 1164
    baseline.validate()
    spool = TelemetrySpoolCapacity(enabled=True, replicas=2)
    oversized = CapacityConfig(telemetry_spool=spool)
    assert oversized.postgres_connection_ceiling() == 1288
    assert oversized.postgres_fleet_connection_budget() == 1240
    with pytest.raises(AdminConfigError, match="1288.*1240"):
        oversized.validate()
    allowed = CapacityConfig(control_worker_replicas=5, telemetry_spool=spool)
    assert allowed.postgres_connection_ceiling() == 1176
    allowed.validate()


def test_old_overbudget_state_is_readable_but_cannot_be_authored_again() -> None:
    old = AdminConfig(
        capacity=CapacityConfig(
            telemetry_spool=TelemetrySpoolCapacity(enabled=True, replicas=3)
        )
    )
    observed = AdminConfig.from_mapping(old.as_dict())
    assert observed == old
    with pytest.raises(AdminConfigError, match="connection ceiling"):
        observed.validate()
    corrected = apply_patch(observed, {"capacity": {"controlWorkerReplicas": 5}})
    corrected.validate()
    for name in ("32-enabled", "50-enabled"):
        preset = preset_admin_config(name)
        assert preset.capacity.control_worker_replicas == 5
        preset.validate()


@pytest.mark.parametrize(
    "update",
    [
        {"pool_max": True},
        {"pool_max": 0},
        {"pool_max": -1},
        {"processes_per_pod": True},
        {"processes_per_pod": 0},
        {"replicas": True},
        {"replicas": -1},
        {"service_role": "unknown"},
    ],
)
def test_invalid_or_unbounded_role_inputs_are_rejected(update) -> None:
    values = {
        "pool_max": 8,
        "processes_per_pod": 4,
        "replicas": 2,
        "service_role": "worker",
        "queued_processor": True,
        "spool_enabled": False,
        "workflow_dispatcher_enabled": True,
        "regional": True,
    }
    with pytest.raises(ValueError):
        PostgresPoolCapacity.role_connection_ceiling(**{**values, **update})
    assert (
        PostgresPoolCapacity.role_connection_ceiling(**{**values, "replicas": 0}) == 0
    )
