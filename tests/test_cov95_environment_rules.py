from __future__ import annotations

import importlib.resources
import json
import logging
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault import env_validation as validation
from gpu_fault.settings import (
    AgentRegistrySettings,
    ControlPlaneSettings,
    StoreSettings,
)
from tests.fleet.test_settings import active_values


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf"])
def test_nonfinite_numeric_configuration_is_rejected_without_echoing_values(
    monkeypatch, value
) -> None:
    key = "GPU_FAULT_UNIT_TIMEOUT"
    monkeypatch.setattr(
        validation, "environment_value_kinds", lambda: {key: ("number", frozenset())}
    )
    monkeypatch.setattr(validation, "environment_value_bounds", lambda: {})
    assert validation.invalid_gpu_fault_environment_values({key: value}) == [
        f"{key} must be finite"
    ], "NaN/infinity cannot disable timeout and capacity bounds"


@pytest.mark.parametrize(
    ("bounds", "value", "message"),
    [
        ((None, 2.5), "3", "at most 2.5"),
        ((1.5, None), "1", "at least 1.5"),
        ((1.0, 2.0), "0", "between 1 and 2"),
        ((1.0, 2.0), "3", "between 1 and 2"),
    ],
)
def test_numeric_bound_diagnostics_preserve_both_limit_directions(
    monkeypatch, bounds, value, message
) -> None:
    key = "GPU_FAULT_UNIT_NUMBER"
    monkeypatch.setattr(
        validation, "environment_value_kinds", lambda: {key: ("number", frozenset())}
    )
    monkeypatch.setattr(validation, "environment_value_bounds", lambda: {key: bounds})
    assert validation.invalid_gpu_fault_environment_values({key: value}) == [
        f"{key} must be {message}"
    ], "diagnostics explain the configured bound without including the input"


def test_untyped_configuration_is_not_assumed_boolean(monkeypatch) -> None:
    monkeypatch.setattr(
        validation,
        "environment_value_kinds",
        lambda: {"GPU_FAULT_UNIT": ("opaque", frozenset())},
    )
    assert (
        validation.invalid_gpu_fault_environment_values({"GPU_FAULT_UNIT": "value"})
        == []
    ), "inferred type checking does not invent constraints for opaque strings"


@pytest.mark.parametrize("policy", ["warn", "ignore", "invalid"])
def test_unknown_name_policy_is_explicit_and_never_logs_the_value(
    caplog, policy
) -> None:
    values = {
        validation.POLICY_ENV: policy,
        "GPU_FAULT_NOT_REGISTERED_UNIT": "example-sensitive-value",
    }
    with caplog.at_level(logging.WARNING):
        if policy == "invalid":
            with pytest.raises(RuntimeError, match="error, warn or ignore"):
                validation.validate_gpu_fault_environment(values, process_name="unit")
        else:
            validation.validate_gpu_fault_environment(values, process_name="unit")
    assert "example-sensitive-value" not in caplog.text, (
        "unknown-setting diagnostics must not echo credentials"
    )
    if policy == "warn":
        assert "GPU_FAULT_NOT_REGISTERED_UNIT" in caplog.text, (
            "warning mode identifies the offending name"
        )


@pytest.mark.parametrize(
    ("document", "function", "message"),
    [
        ([], "environment_inventory", "JSON object"),
        (
            {"variables": [], "value_kinds": [1]},
            "environment_value_kinds",
            "value_kinds",
        ),
        (
            {"variables": [], "value_bounds": [1]},
            "environment_value_bounds",
            "value_bounds",
        ),
    ],
)
def test_packaged_inventory_corruption_is_not_empty_configuration(
    monkeypatch, document, function, message
) -> None:
    resource = SimpleNamespace()
    resource.joinpath = lambda *parts: resource
    resource.read_text = lambda **options: json.dumps(document)
    monkeypatch.setattr(importlib.resources, "files", lambda *args: resource)
    isolated = runpy.run_path(str(Path(validation.__file__)))
    with pytest.raises(RuntimeError, match=message):
        isolated[function]()


@pytest.mark.parametrize(
    "values",
    [
        {"GPU_FAULT_STORE_URL": "unknown://store"},
        {
            "GPU_FAULT_STORE_URL": "postgresql://unit/db",
            "GPU_FAULT_POSTGRES_POOL_MIN_SIZE": "-1",
        },
        {
            "GPU_FAULT_STORE_URL": "postgresql://unit/db",
            "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "0",
        },
        {
            "GPU_FAULT_STORE_URL": "postgresql://unit/db",
            "GPU_FAULT_POSTGRES_POOL_MIN_SIZE": "9",
            "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "8",
        },
    ],
)
def test_store_settings_refuse_invalid_backend_and_capacity(values) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        StoreSettings.from_mapping(values)


@pytest.mark.parametrize(
    "change",
    [
        {"GPU_FAULT_ENABLE_AGENT_REGISTRY": "true"},
        {"GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION": "3"},
        {"GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": "0"},
        {"GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS": "0,2"},
        {
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_AGENT_ENDPOINT_ALLOWED_PORTS": "",
        },
    ],
)
def test_registry_settings_reject_unusable_pins_protocols_and_ports(change) -> None:
    values = {
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
        **change,
    }
    if change == {"GPU_FAULT_ENABLE_AGENT_REGISTRY": "true"}:
        values.pop("GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256")
    with pytest.raises((RuntimeError, ValueError)):
        AgentRegistrySettings.from_mapping(values)


def test_regional_registry_settings_keep_per_cluster_cidrs() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": json.dumps(
                [
                    {
                        "cluster_id": "unit-cluster",
                        "agent_endpoint_allowed_cidrs": ["10.2.0.0/16"],
                    }
                ]
            ),
        }
    )
    configured = ControlPlaneSettings.from_mapping(values)
    assert configured.regional_cluster_values[0]["agent_endpoint_allowed_cidrs"] == [
        "10.2.0.0/16"
    ], "regional settings retain the cluster-local endpoint boundary"


@pytest.mark.parametrize(
    "change,fragment",
    [
        ({"GPU_FAULT_REMOTE_EXECUTION_OWNERS": ""}, "remote execution owners"),
        ({"GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS": "1"}, "retention requires"),
        ({"GPU_FAULT_ENABLE_HYPERPOD_ADAPTER": "true"}, "delegate HyperPod"),
    ],
)
def test_regional_settings_refuse_missing_owners_and_retention_authority(
    change, fragment
) -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": "[]",
            **change,
        }
    )
    with pytest.raises(RuntimeError, match=fragment):
        ControlPlaneSettings.from_mapping(values)


@pytest.mark.parametrize("registry", ["{", "{}", "[null]", ""])
def test_regional_registry_structure_is_validated_before_assembly(registry) -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": registry,
        }
    )
    with pytest.raises(RuntimeError):
        ControlPlaneSettings.from_mapping(values)


def test_regional_mode_refuses_a_single_cluster_override() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": "[]",
            "GPU_FAULT_HYPERPOD_CLUSTER": "single-cluster",
        }
    )
    with pytest.raises(RuntimeError, match="must be unset"):
        ControlPlaneSettings.from_mapping(values)


def test_legacy_sqlite_settings_are_only_a_compatibility_parse() -> None:
    result = StoreSettings.from_mapping(
        {"GPU_FAULT_STORE_URL": "sqlite:///unit/legacy.db"}
    )
    assert result.kind == "sqlite" and result.sqlite_path == "unit/legacy.db", (
        "legacy parsing remains available without opening a database or changing production mode"
    )
