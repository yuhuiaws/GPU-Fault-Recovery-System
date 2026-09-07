from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from gpu_fault.collector_requirements import (
    CollectorServices,
    CollectorServiceState,
    required_collectors_for_agent,
)
from gpu_fault.telemetry import CollectorKind


def test_intentionally_disabled_node_log_is_not_required() -> None:
    agent = SimpleNamespace(
        collector_services={
            "gpu-fault-log-collector": CollectorServiceState(
                active="inactive", enabled="disabled"
            )
        }
    )
    required = required_collectors_for_agent(agent)
    assert CollectorKind.NODE_LOGS not in required
    assert CollectorKind.NVIDIA_KERNEL in required


def test_enabled_but_failed_collector_remains_required() -> None:
    agent = SimpleNamespace(
        collector_services={
            "gpu-fault-log-collector": CollectorServiceState(
                active="failed", enabled="enabled"
            )
        }
    )
    assert CollectorKind.NODE_LOGS in required_collectors_for_agent(agent)


@pytest.mark.parametrize("token", ["true", "1", "yes", "ON"])
def test_every_enabled_token_makes_node_logs_required(monkeypatch, token) -> None:
    # `=1` used to read as "not required" (S9): the switch must take every
    # enabled token the validator documents.
    monkeypatch.setenv("GPU_FAULT_REQUIRE_NODE_LOG_COLLECTOR", token)
    agent = SimpleNamespace(collector_services={})

    assert CollectorKind.NODE_LOGS in required_collectors_for_agent(agent)


def test_unknown_enabled_state_keeps_optional_node_log_optional(monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_REQUIRE_NODE_LOG_COLLECTOR", raising=False)
    agent = SimpleNamespace(
        collector_services={
            "gpu-fault-log-collector": CollectorServiceState(
                active="unknown", enabled="unknown"
            )
        }
    )

    assert CollectorKind.NODE_LOGS not in required_collectors_for_agent(agent)


def test_unknown_collector_units_are_discarded_before_validation() -> None:
    services = TypeAdapter(CollectorServices).validate_python(
        {
            "gpu-fault-kernel-collector": {"active": "active", "enabled": "enabled"},
            "gpu-fault-future-collector": {"future_state": "new-protocol"},
        }
    )

    assert set(services) == {"gpu-fault-kernel-collector"}
