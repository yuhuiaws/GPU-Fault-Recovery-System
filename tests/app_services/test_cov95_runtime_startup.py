from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import context as context_module
from gpu_fault.app.context import ApplicationContext
from gpu_fault.store import InMemoryStore
from tests._builders import build_context
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **changes: str
) -> None:
    values = {
        "GPU_FAULT_EXECUTOR_MODE": "active",
        "GPU_FAULT_ALLOW_SINGLE_CLUSTER": "true",
        "GPU_FAULT_STORE_URL": f"sqlite:///{tmp_path / 'runtime.db'}",
        "GPU_FAULT_EXECUTION_TOKEN": "t" * 32,
        "GPU_FAULT_ALLOWED_OPERATIONS": "FREEZE_EVIDENCE",
        "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true",
        "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER": "false",
        "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER": "false",
        "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER": "false",
        "GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER": "false",
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false",
        "AWS_REGION": "us-west-2",
        **changes,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def registry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
        "GPU_FAULT_AGENT_REGISTRATION_SECRET": "m" * 32,
        "GPU_FAULT_NODE_ACTION_SECRET": "m" * 32,
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
        "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION": "2",
    }.items():
        monkeypatch.setenv(name, value)


def providers(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    calls = []

    def lifecycle(config: Any, **kwargs: Any) -> Any:
        calls.append(("hyperpod", config))
        return SimpleNamespace(config=config, store=kwargs.get("store"))

    def kubernetes(**kwargs: Any) -> Any:
        calls.append(("kubernetes", kwargs))
        return SimpleNamespace(core=object())

    def create(name: str) -> Any:
        def build(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, (args, kwargs)))
            return SimpleNamespace(**kwargs)

        return build

    monkeypatch.setattr(context_module, "HyperPodLifecycleAdapter", lifecycle)
    monkeypatch.setattr(context_module, "KubernetesWorkflowAdapter", kubernetes)
    for name in (
        "HyperPodSpareCoordinator",
        "HyperPodSpareHealthController",
        "HyperPodLifecycleStepAdapter",
    ):
        monkeypatch.setattr(context_module, name, create(name))
    return calls


@pytest.mark.parametrize(
    "profile", ["minimal", "node", "managed", "spare", "regional-managed"]
)
def test_active_context_wires_only_the_configured_local_or_regional_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: str
) -> None:
    environment(monkeypatch, tmp_path)
    calls = providers(monkeypatch)
    if profile == "minimal":
        monkeypatch.setenv("GPU_FAULT_MANAGED_RECOVERY_OWNERS", "")
    else:
        registry_settings(monkeypatch)
    if profile == "node":
        monkeypatch.setenv("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", "true")
    elif profile in {"managed", "spare"}:
        monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "unit-hyperpod")
        monkeypatch.setenv("GPU_FAULT_ENABLE_KUBERNETES_ADAPTER", "true")
        monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "true")
        monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER", "true")
        if profile == "spare":
            monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", "true")
    elif profile == "regional-managed":
        monkeypatch.setenv("GPU_FAULT_DEPLOYMENT_MODE", "regional")
        monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER", "true")
        monkeypatch.setenv(
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON",
            json.dumps(
                [
                    {
                        "cluster_id": "cluster-a",
                        "region": "us-west-2",
                        "hyperpod_cluster_name": "unit-hyperpod",
                        "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/unit",
                        "token": "a" * 32,
                        "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
                    }
                ]
            ),
        )
    context = ApplicationContext.from_environment()
    assert context.executor_mode == "active"
    assert context.regional_mode is (profile == "regional-managed")
    assert context.workflow_executor.fleet_registry is context.fleet_registry
    if profile == "minimal":
        assert calls == [] and context.fleet_registry is None
        assert len(context.workflow_executor.adapters) == 3
    elif profile == "node":
        adapter = next(
            adapter
            for adapter in context.workflow_executor.adapters
            if isinstance(adapter, context_module.NodeActionWorkflowAdapter)
        )
        assert adapter.registry is context.fleet_registry
        assert adapter.barriers is context.barrier_coordinator
        assert calls == []
    elif profile == "regional-managed":
        assert context.hyperpod_identity_registry is None
        assert len(context.hyperpod_identity_registries) == 1
        assert set(context.regional_managed_observer.observers) == {"cluster-a"}
        assert [name for name, _ in calls] == ["hyperpod"]
    else:
        assert context.hyperpod_identity_registry is not None
        assert [name for name, _ in calls][:2] == ["kubernetes", "hyperpod"]
        assert (context.spare_health_controller is not None) is (profile == "spare")
    context.store.close()


@pytest.mark.parametrize(
    "failure",
    [
        "observer-no-registry",
        "observer-no-kubernetes",
        "spare-no-kubernetes",
        "spare-no-registry",
    ],
)
def test_active_context_refuses_incomplete_managed_recovery_prerequisites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    environment(monkeypatch, tmp_path)
    providers(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "unit-hyperpod")
    if failure.startswith("observer"):
        monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER", "true")
        if failure.endswith("kubernetes"):
            registry_settings(monkeypatch)
    else:
        monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "true")
        monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", "true")
        if failure.endswith("registry"):
            monkeypatch.setenv("GPU_FAULT_ENABLE_KUBERNETES_ADAPTER", "true")
    with pytest.raises((RuntimeError, ValueError), match="requires"):
        ApplicationContext.from_environment()


def test_active_postgres_configuration_passes_schema_and_archive_contract_without_a_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment(
        monkeypatch,
        tmp_path,
        GPU_FAULT_STORE_URL="postgresql://unit.invalid/unit",
        GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT="false",
        GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS="7",
        GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI="s3://unit-bucket/archive",
    )
    selected = InMemoryStore()
    calls = []
    archives = []
    monkeypatch.setattr(
        context_module,
        "PostgresStore",
        lambda url, **kwargs: calls.append((url, kwargs)) or selected,
    )
    monkeypatch.setattr(
        context_module,
        "ControlRecordArchiver",
        lambda *args, **kwargs: archives.append((args, kwargs)) or object(),
    )
    context = ApplicationContext.from_environment()
    assert context.store is selected
    assert calls[0][1]["initialize_schema"] is False
    assert archives[0][0] == (
        "postgresql://unit.invalid/unit",
        "s3://unit-bucket/archive",
    )
    assert archives[0][1]["store"] is selected
    assert archives[0][1]["retention"].days == 7
    assert context.control_record_archiver is not None


def test_inventory_freshness_and_branch_escalation_require_valid_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_GPU_INVENTORY_MAX_AGE_SECONDS", "119")
    with pytest.raises(ValueError, match="two inventory delivery intervals"):
        build_context()
    monkeypatch.setenv("GPU_FAULT_GPU_INVENTORY_MAX_AGE_SECONDS", "180")
    monkeypatch.setenv("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS", "0")
    with pytest.raises(ValueError, match="at least 1"):
        build_context()


def test_startup_reports_timing_warnings_without_discarding_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        context_module,
        "validate_timing_from_environment",
        lambda values: ["unit timing warning"],
    )
    context = ApplicationContext.from_environment()
    assert "unit timing warning" in caplog.text
    context.store.close()
