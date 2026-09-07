from __future__ import annotations

import os
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, field_validator

# The unit and silence tables are rows of the collector registry; they keep
# their historical import path here.
from gpu_fault.collector_registry import (
    COLLECTOR_SYSTEMD_UNITS as COLLECTOR_SYSTEMD_UNITS,
)
from gpu_fault.collector_registry import (
    collector_silent_thresholds as collector_silent_thresholds,
)
from gpu_fault.env import env_bool
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
    require_node_log = env_bool("GPU_FAULT_REQUIRE_NODE_LOG_COLLECTOR")
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
