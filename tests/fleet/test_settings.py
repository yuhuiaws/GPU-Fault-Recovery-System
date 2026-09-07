from __future__ import annotations

import json

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.settings import ControlPlaneSettings


def active_values() -> dict[str, str]:
    return {
        "GPU_FAULT_EXECUTOR_MODE": "active",
        "GPU_FAULT_ALLOWED_OPERATIONS": "FREEZE_EVIDENCE",
        "GPU_FAULT_STORE_URL": "postgresql://db/gpu_fault",
        "GPU_FAULT_EXECUTION_TOKEN": "x" * 32,
        "GPU_FAULT_ALLOW_SINGLE_CLUSTER": "true",
        "GPU_FAULT_AGENT_ENDPOINT_ALLOWED_CIDRS": "10.0.0.0/16",
    }


@pytest.mark.parametrize(("token", "expected"), [("on", True), ("off", False)])
def test_settings_switches_accept_on_and_off(token: str, expected: bool) -> None:
    """The settings parser used to stop at ``1/true/yes``; ``on`` was refused."""

    values = active_values()
    values["GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT"] = token

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.store is not None
    assert settings.store.postgres_auto_schema_init is expected


def test_blank_settings_switch_means_its_default() -> None:
    values = active_values()
    values["GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT"] = ""

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.store is not None
    assert settings.store.postgres_auto_schema_init is True


def test_simulation_settings_do_not_require_active_secrets() -> None:
    settings = ControlPlaneSettings.from_mapping({})

    assert settings.executor.enabled is False
    assert settings.store is None
    assert settings.execution_token is None


def test_active_settings_parse_without_dependency_assembly() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_POSTGRES_POOL_MIN_SIZE": "2",
            "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "12",
            "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT": "false",
        }
    )

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.executor.allowed_operations == frozenset(
        {WorkflowOperation.FREEZE_EVIDENCE}
    )
    assert settings.store is not None
    assert settings.store.postgres_pool_min_size == 2
    assert settings.store.postgres_pool_max_size == 12
    assert settings.store.postgres_auto_schema_init is False


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("GPU_FAULT_EXECUTION_TOKEN", "short", "EXECUTION_TOKEN"),
        ("GPU_FAULT_ALLOWED_OPERATIONS", "", "ALLOWED_OPERATIONS"),
    ],
)
def test_active_settings_fail_closed(name: str, value: str, message: str) -> None:
    values = active_values()
    values[name] = value

    with pytest.raises(RuntimeError, match=message):
        ControlPlaneSettings.from_mapping(values)


def test_regional_settings_reject_local_mutation_adapters() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": json.dumps(
                [
                    {
                        "cluster_id": "cluster-a",
                        "region": "us-west-2",
                        "hyperpod_cluster_name": "hp-a",
                        "eks_cluster_arn": "arn:aws:eks:::cluster/a",
                        "token": "token-a",
                    }
                ]
            ),
            "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER": "true",
        }
    )

    with pytest.raises(RuntimeError, match="must not enable.*Kubernetes"):
        ControlPlaneSettings.from_mapping(values)


def test_regional_settings_allow_an_explicit_empty_cluster_registry() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": "[]",
        }
    )

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.regional_mode is True
    assert settings.regional_cluster_values == ()


def test_single_cluster_requires_explicit_canary_acknowledgement() -> None:
    values = active_values()
    values.pop("GPU_FAULT_ALLOW_SINGLE_CLUSTER")

    with pytest.raises(RuntimeError, match="Canary-only"):
        ControlPlaneSettings.from_mapping(values)


def test_single_cluster_rejects_multiple_registered_clusters() -> None:
    values = active_values()
    values["GPU_FAULT_REGIONAL_CLUSTERS_JSON"] = json.dumps(
        [{"cluster_id": "a"}, {"cluster_id": "b"}]
    )

    with pytest.raises(RuntimeError, match="multiple regional"):
        ControlPlaneSettings.from_mapping(values)


def test_agent_registry_pins_are_validated_without_app_startup() -> None:
    values = active_values()
    values["GPU_FAULT_ENABLE_AGENT_REGISTRY"] = "true"

    with pytest.raises(RuntimeError, match="REQUIRED_AGENT_ARTIFACT_SHA256"):
        ControlPlaneSettings.from_mapping(values)


def test_agent_registry_protocol_defaults_to_current_version() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "c" * 64,
        }
    )

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.agent_registry.required_agent_protocol_version == 3
    assert settings.agent_registry.endpoint_require_tls is True


def test_agent_registry_requires_endpoint_cidrs() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "c" * 64,
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": json.dumps(
                [
                    {
                        "cluster_id": "cluster-a",
                        "region": "us-west-2",
                        "hyperpod_cluster_name": "hp-a",
                        "eks_cluster_arn": "arn:aws:eks:::cluster/a",
                        "token": "a" * 32,
                    }
                ]
            ),
        }
    )

    with pytest.raises(RuntimeError, match="agent_endpoint_allowed_cidrs"):
        ControlPlaneSettings.from_mapping(values)


def test_agent_registry_parses_rollout_compatibility_window() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "c" * 64,
            "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": "3",
            "GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS": "2,3",
            "GPU_FAULT_COMPATIBLE_AGENT_ARTIFACT_SHA256S": (f"{'b' * 64},{'a' * 64}"),
            "GPU_FAULT_COMPATIBLE_AGENT_CONFIG_DIGESTS": (f"{'d' * 64},{'c' * 64}"),
        }
    )

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.agent_registry.compatible_agent_protocol_versions == frozenset({2})
    assert settings.agent_registry.compatible_artifact_sha256s == frozenset({"b" * 64})
    assert (
        settings.agent_registry.required_compatibility_digest
        == settings.agent_registry.required_artifact_sha256
    )
    assert settings.agent_registry.compatible_config_digests == frozenset({"d" * 64})


def test_empty_agent_compatibility_digest_falls_back_to_artifact() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": "",
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "c" * 64,
        }
    )

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.agent_registry.required_compatibility_digest == "a" * 64


def test_agent_registry_protocol_must_be_positive() -> None:
    values = active_values()
    values.update(
        {
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "c" * 64,
            "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": "0",
        }
    )

    with pytest.raises(ValueError, match="PROTOCOL_VERSION"):
        ControlPlaneSettings.from_mapping(values)
