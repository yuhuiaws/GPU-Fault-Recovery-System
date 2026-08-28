from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_MODULE = lazy_script_module(
    "regional_release_registry_safety",
    ROOT / "deploy/control-plane/regional/regional_release_registry.py",
)
STATE_MODULE = lazy_script_module(
    "regional_release_state_safety",
    ROOT / "deploy/control-plane/regional/regional_release_state.py",
)
ROLLOUT_MODULE = lazy_script_module(
    "rollout_regional_release_retry_safety",
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py",
)


def cluster_target(tmp_path: Path, *, cidrs: tuple[str, ...]):
    token = tmp_path / "cluster.token"
    token.write_text("t" * 32)
    return REGISTRY_MODULE.ClusterTarget(
        cluster_id="gpu-a",
        context="gpu-a-context",
        executor_irsa_role_arn="arn:aws:iam::1:role/a",
        region="us-east-1",
        hyperpod_cluster_name="hp-gpu-a",
        eks_cluster_arn=("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"),
        token_file=str(token),
        allowed_namespaces=("training",),
        agent_endpoint_allowed_cidrs=cidrs,
    )


def test_registry_entry_requires_and_normalizes_agent_endpoint_cidrs(
    tmp_path: Path,
) -> None:
    entry = REGISTRY_MODULE.registry_entry(
        cluster_target(tmp_path, cidrs=("10.0.1.15/16",))
    )

    assert entry["agent_endpoint_allowed_cidrs"] == ["10.0.0.0/16"]
    assert entry["token"] == "t" * 32

    with pytest.raises(
        REGISTRY_MODULE.ReleaseError, match="requires Agent endpoint CIDRs"
    ):
        REGISTRY_MODULE.registry_entry(cluster_target(tmp_path, cidrs=()))


def test_registry_stage_keeps_the_original_secret_until_commit() -> None:
    stage = inspect.getsource(REGISTRY_MODULE.stage_registry)
    commit = inspect.getsource(REGISTRY_MODULE.commit_registry_update)
    restore = inspect.getsource(REGISTRY_MODULE.restore_registry_backup)

    assert "backup=current if backup is None else backup" in stage
    assert "write_registry(release, current)" in commit
    assert "write_registry(release, backup)" in restore


def test_remote_command_idle_check_has_bounded_retries() -> None:
    source = inspect.getsource(ROLLOUT_MODULE.RegionalRelease._remote_commands_are_idle)

    assert "remote_command_stats()" in source
    assert "for attempt in range(3)" in source
    assert "if attempt == 2" in source


def test_registry_stage_forces_cpu_secret_reload() -> None:
    source = inspect.getsource(ROLLOUT_MODULE.RegionalRelease.upgrade)

    assert "force_restart=registry_staged" in source


def test_cpu_role_config_snapshot_rejects_sensitive_keys() -> None:
    class SnapshotRelease:
        config = SimpleNamespace(namespace="gpu-fault-system")

        @staticmethod
        def _cpu(*args):
            return ["kubectl", *args]

        @staticmethod
        def _get_json(arguments):
            if "deployment" in arguments:
                return {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "envFrom": [
                                            {
                                                "configMapRef": {
                                                    "name": (
                                                        "gpu-fault-api-ha-config-core"
                                                    )
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                }
            return {}

        @staticmethod
        def _config_map_data(_name):
            return {"GPU_FAULT_DATABASE_PASSWORD": "unsafe"}

    with pytest.raises(STATE_MODULE.ReleaseError, match="sensitive-looking keys"):
        STATE_MODULE.cpu_role_config_maps(SnapshotRelease())
