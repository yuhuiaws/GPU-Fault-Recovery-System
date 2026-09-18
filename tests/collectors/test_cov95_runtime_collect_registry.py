"""Collector factories, retired producers and heartbeat requirements."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from gpu_fault import collector_registry as registry
from gpu_fault import collector_requirements as requirements
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize("needs_context", [False, True])
def test_descriptor_invokes_the_selected_factory_with_its_declared_context(
    monkeypatch, needs_context
):
    calls = []
    expected = SimpleNamespace(run=lambda: calls.append("run"))

    def factory(*args):
        calls.append(args)
        return expected

    monkeypatch.setitem(
        sys.modules, "cov95_fake_collector_plugin", SimpleNamespace(build=factory)
    )
    descriptor = replace(
        registry.COLLECTOR_REGISTRY["kernel"],
        factory="cov95_fake_collector_plugin:build",
        needs_context=needs_context,
    )
    sink = support.RecordingSink()
    context = support.collector_context()
    args = argparse.Namespace(node_id="node-a")
    built = descriptor.build(sink, context, args)
    built.run()
    assert built is expected
    assert calls == [(sink, context, args) if needs_context else (sink, args), "run"]
    if needs_context:
        with pytest.raises(RuntimeError, match="needs a collector context"):
            descriptor.build(sink, None, args)


@pytest.mark.parametrize(
    "reference",
    ["missing-separator", ":build", "module:", "cov95_fake_collector_plugin:value"],
)
def test_invalid_factory_is_rejected_before_collector_construction(
    monkeypatch, reference
):
    monkeypatch.setitem(
        sys.modules, "cov95_fake_collector_plugin", SimpleNamespace(value=3)
    )
    descriptor = replace(registry.COLLECTOR_REGISTRY["kernel"], factory=reference)
    with pytest.raises(RuntimeError, match="not module:callable|not callable"):
        descriptor.build(
            support.RecordingSink(), support.collector_context(), argparse.Namespace()
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"factory": "not-a-reference"}, "module:callable"),
        ({"channel_paths": ()}, "no channel path"),
        ({"kinds": (registry.CollectorKind.NVIDIA_KERNEL,) * 2}, "repeats"),
        ({"needs_context": False}, "GPU product discovery"),
        ({"runs_in": "cluster"}, "GPU product discovery"),
        (
            {
                "kinds": (
                    registry.CollectorKind.NVIDIA_KERNEL,
                    registry.CollectorKind.HOST_TELEMETRY,
                )
            },
            "systemd units",
        ),
    ],
)
def test_registry_refuses_inconsistent_producer_contracts(updates, message):
    candidate = dict(registry.COLLECTOR_REGISTRY)
    candidate["kernel"] = replace(candidate["kernel"], **updates)
    with pytest.raises(RuntimeError, match=message):
        registry.validate_collector_registry(candidate)


@pytest.mark.parametrize("runtime_field", ["systemd_unit", "silent_threshold"])
def test_retired_kind_cannot_regain_runtime_settings(runtime_field):
    kinds = dict(registry.COLLECTOR_KINDS)
    value = (
        "retired-unit"
        if runtime_field == "systemd_unit"
        else registry.silent_threshold("COV95_TEST_SILENCE_SECONDS", "5")
    )
    kinds[registry.CollectorKind.HMA_NODE] = replace(
        kinds[registry.CollectorKind.HMA_NODE], **{runtime_field: value}
    )
    with pytest.raises(RuntimeError, match="retired collector kind"):
        registry.validate_collector_registry(kinds=kinds)
    assert "kubernetes-hma" not in registry.COLLECTOR_REGISTRY
    assert "sqs-hma" not in registry.COLLECTOR_REGISTRY


@pytest.mark.parametrize("mode", ["wrong-type", "wrong-name", "collision", "valid"])
def test_plugin_discovery_enforces_type_name_and_builtin_ownership(monkeypatch, mode):
    descriptor = replace(
        registry.COLLECTOR_REGISTRY["training-progress"],
        cli_command="cov95-plugin",
        export_name=None,
    )
    name = "cov95-plugin"
    if mode == "wrong-type":
        descriptor = object()
    elif mode == "wrong-name":
        name = "other-plugin"
    elif mode == "collision":
        name = "kernel"
        descriptor = registry.COLLECTOR_REGISTRY["kernel"]
    monkeypatch.setattr(
        registry,
        "discover_plugins",
        lambda group: {name: SimpleNamespace(load=lambda: descriptor)},
    )
    if mode == "valid":
        discovered = registry.collector_registry_with_plugins()
        assert discovered[name] is descriptor
        assert set(discovered) == set(registry.COLLECTOR_REGISTRY) | {name}
    else:
        with pytest.raises(RuntimeError, match="must be|two must agree|collides"):
            registry.collector_registry_with_plugins()


@pytest.mark.parametrize(
    ("agent", "current"),
    [
        (SimpleNamespace(lifecycle_state=SimpleNamespace(value="UPGRADING")), False),
        (
            SimpleNamespace(
                lifecycle_state=SimpleNamespace(value="ACTIVE"),
                lease_expires_at=support.NOW,
            ),
            True,
        ),
        (
            SimpleNamespace(
                lifecycle_state=SimpleNamespace(value="ACTIVE"),
                lease_expires_at=support.NOW - timedelta(microseconds=1),
            ),
            False,
        ),
        (SimpleNamespace(lifecycle_state=SimpleNamespace(value="ACTIVE")), True),
        (
            SimpleNamespace(
                lifecycle_state=SimpleNamespace(value="ACTIVE"),
                last_seen_at=support.NOW - timedelta(seconds=91),
            ),
            False,
        ),
    ],
)
def test_existing_agent_freshness_contract_at_lease_and_legacy_boundaries(
    agent, current
):
    assert requirements.agent_is_current(agent, observed_at=support.NOW) is current


def test_service_reports_filter_unknown_units_and_normalize_unknown_states():
    unit = registry.COLLECTOR_SYSTEMD_UNITS[registry.CollectorKind.GPU_METRICS]
    result = TypeAdapter(requirements.CollectorServices).validate_python(
        {
            unit: {"active": "unrecognized", "enabled": " MASKED-RUNTIME "},
            "unowned-unit": {"active": "active", "enabled": "enabled"},
        }
    )
    assert set(result) == {unit}
    assert result[unit].active is requirements.CollectorActiveState.UNKNOWN
    assert result[unit].intentionally_disabled, (
        "masked service must not be required as running"
    )
    with pytest.raises(ValidationError):
        TypeAdapter(requirements.CollectorServices).validate_python(
            ["not", "a", "mapping"]
        )


@pytest.mark.parametrize("require_logs", [False, True])
def test_required_collectors_exclude_disabled_and_retired_producers(
    monkeypatch, require_logs
):
    monkeypatch.setenv(
        "GPU_FAULT_REQUIRE_NODE_LOG_COLLECTOR", str(require_logs).lower()
    )
    unit = registry.COLLECTOR_SYSTEMD_UNITS[registry.CollectorKind.NVIDIA_KERNEL]
    agent = SimpleNamespace(
        collector_services={unit: {"active": "inactive", "enabled": "disabled"}}
    )
    required = requirements.required_collectors_for_agent(agent)
    assert registry.CollectorKind.NVIDIA_KERNEL not in required
    assert (registry.CollectorKind.NODE_LOGS in required) is require_logs
    assert registry.CollectorKind.HMA_NODE not in required
    assert registry.CollectorKind.HMA_CLOUDWATCH not in required
