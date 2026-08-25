from __future__ import annotations

import os
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, field_validator

from gpu_fault.models import StrictModel
from gpu_fault.telemetry import CollectorKind


class CollectorActiveState(StrEnum):
    ACTIVE = "active"
    ACTIVATING = "activating"
    RELOADING = "reloading"
    DEACTIVATING = "deactivating"
    INACTIVE = "inactive"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class CollectorEnabledState(StrEnum):
    ENABLED = "enabled"
    ENABLED_RUNTIME = "enabled-runtime"
    STATIC = "static"
    INDIRECT = "indirect"
    LINKED = "linked"
    LINKED_RUNTIME = "linked-runtime"
    ALIAS = "alias"
    DISABLED = "disabled"
    MASKED = "masked"
    MASKED_RUNTIME = "masked-runtime"
    GENERATED = "generated"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class CollectorServiceState(StrictModel):
    active: CollectorActiveState
    enabled: CollectorEnabledState

    @field_validator("active", mode="before")
    @classmethod
    def normalize_active(cls, value):
        raw = str(value or "unknown").strip().lower()
        return (
            raw
            if raw in {item.value for item in CollectorActiveState}
            else CollectorActiveState.UNKNOWN
        )

    @field_validator("enabled", mode="before")
    @classmethod
    def normalize_enabled(cls, value):
        raw = str(value or "unknown").strip().lower()
        return (
            raw
            if raw in {item.value for item in CollectorEnabledState}
            else CollectorEnabledState.UNKNOWN
        )

    @property
    def intentionally_disabled(self) -> bool:
        return self.enabled in {
            CollectorEnabledState.DISABLED,
            CollectorEnabledState.MASKED,
            CollectorEnabledState.MASKED_RUNTIME,
        }


COLLECTOR_SYSTEMD_UNITS = {
    CollectorKind.NVIDIA_KERNEL: "gpu-fault-kernel-collector",
    CollectorKind.FABRIC_MANAGER_LOG: ("gpu-fault-fabric-manager-collector"),
    CollectorKind.GPU_INVENTORY: "gpu-fault-metrics-collector",
    CollectorKind.GPU_METRICS: "gpu-fault-metrics-collector",
    CollectorKind.HOST_TELEMETRY: "gpu-fault-host-collector",
    CollectorKind.NODE_LOGS: "gpu-fault-log-collector",
}


def validate_collector_services(value):
    if not isinstance(value, dict):
        return value
    known_units = set(COLLECTOR_SYSTEMD_UNITS.values())
    return {unit: state for unit, state in value.items() if unit in known_units}


CollectorServices = Annotated[
    dict[str, CollectorServiceState],
    BeforeValidator(validate_collector_services),
]
ReportedCollectorServices = dict[str, CollectorServiceState]


def collector_silent_thresholds() -> dict[CollectorKind, float]:
    return {
        CollectorKind.GPU_INVENTORY: float(
            os.getenv("GPU_FAULT_GPU_INVENTORY_SILENT_AFTER_SECONDS", "180")
        ),
        CollectorKind.GPU_METRICS: float(
            os.getenv("GPU_FAULT_GPU_METRICS_SILENT_AFTER_SECONDS", "420")
        ),
        CollectorKind.HOST_TELEMETRY: float(
            os.getenv("GPU_FAULT_HOST_TELEMETRY_SILENT_AFTER_SECONDS", "420")
        ),
        CollectorKind.NVIDIA_KERNEL: float(
            os.getenv("GPU_FAULT_KERNEL_SILENT_AFTER_SECONDS", "900")
        ),
        CollectorKind.FABRIC_MANAGER_LOG: float(
            os.getenv("GPU_FAULT_FABRIC_MANAGER_SILENT_AFTER_SECONDS", "900")
        ),
        CollectorKind.NODE_LOGS: float(
            os.getenv("GPU_FAULT_NODE_LOG_SILENT_AFTER_SECONDS", "900")
        ),
    }


def agent_is_current(agent, *, observed_at: datetime) -> bool:
    if agent.lifecycle_state.value != "ACTIVE":
        return False
    lease_expires_at = getattr(agent, "lease_expires_at", None)
    if lease_expires_at is not None:
        return lease_expires_at >= observed_at
    last_seen_at = getattr(agent, "last_seen_at", None)
    if last_seen_at is None:
        return True
    max_age = float(os.getenv("GPU_FAULT_AGENT_MAX_HEARTBEAT_AGE_SECONDS", "90"))
    return (observed_at - last_seen_at).total_seconds() <= max_age


def required_collectors_for_agent(agent) -> set[CollectorKind]:
    states = getattr(agent, "collector_services", {}) or {}
    require_node_log = (
        os.getenv("GPU_FAULT_REQUIRE_NODE_LOG_COLLECTOR", "false").strip().lower()
        == "true"
    )
    result = set()
    for kind, unit in COLLECTOR_SYSTEMD_UNITS.items():
        state = states.get(unit)
        if state is not None and not isinstance(state, CollectorServiceState):
            state = CollectorServiceState.model_validate(state)
        if state is not None and state.intentionally_disabled:
            continue
        if (
            kind is CollectorKind.NODE_LOGS
            and not require_node_log
            and (state is None or state.enabled is CollectorEnabledState.UNKNOWN)
        ):
            continue
        result.add(kind)
    return result
