from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
MODULE = lazy_script_module("rollout_regional_release", MODULE_PATH)


def test_upgrade_ensures_schema_before_rolling_cpu() -> None:
    source = inspect.getsource(MODULE.RegionalRelease.upgrade)

    assert source.index("self._upload_release()") < source.index(
        "self._ensure_schema()"
    )
    assert source.index("self._ensure_schema()") < source.index(
        "self._apply_cpu(finalize=False)"
    )


def config_file(tmp_path: Path, *, clusters=None) -> Path:
    wheel = tmp_path / "release.whl"
    bundle = tmp_path / "bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    value = {
        "cpu_kubeconfig": "/secure/cpu.kubeconfig",
        "namespace": "gpu-fault-system",
        "release": {
            "wheel": str(wheel),
            "bundle": str(bundle),
            "agent_config_digest": "a" * 64,
        },
        "clusters": clusters
        or [
            {
                "cluster_id": "gpu-a",
                "context": "gpu-a-context",
                "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
            }
        ],
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value))
    return path


def manifest_config_file(tmp_path: Path) -> Path:
    wheel = tmp_path / "release.whl"
    bundle = tmp_path / "bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    manifest = tmp_path / "current-release.json"
    manifest.write_text(
        json.dumps(
            {
                "wheel": str(wheel),
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "bundle": str(bundle),
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            }
        )
    )
    path = tmp_path / "release-config.json"
    path.write_text(
        json.dumps(
            {
                "cpu_kubeconfig": "/secure/cpu.kubeconfig",
                "release": {"manifest": str(manifest), "agent_config_digest": "a" * 64},
                "clusters": [
                    {
                        "cluster_id": "gpu-a",
                        "context": "gpu-a-context",
                        "executor_irsa_role_arn": ("arn:aws:iam::1:role/a"),
                    }
                ],
            }
        )
    )
    return path


def test_release_config_requires_unique_clusters(tmp_path) -> None:
    cluster = {
        "cluster_id": "gpu-a",
        "context": "gpu-a-context",
        "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
    }

    with pytest.raises(MODULE.ReleaseError, match="unique"):
        MODULE.ReleaseConfig.load(config_file(tmp_path, clusters=[cluster, cluster]))


def test_plan_covers_first_deploy_and_rollback(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    deploy = release.plan("deploy")
    rollback = release.plan("rollback")

    assert any("prerequisites" in step for step in deploy)
    assert any("PostgreSQL schema" in step for step in deploy)
    assert any("previous required pins" in step for step in rollback)
    assert any("installer bundle" in step for step in rollback)


def test_join_cluster_requires_current_release_artifact(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    assert release.wheel_cm.startswith("gpu-fault-control-plane-wheel-0100-")
    assert release.bundle_cm.startswith("gpu-fault-node-installer-0100-")
    assert MODULE.STATE_CONFIG_MAP == ("gpu-fault-regional-release-state")


def test_regional_release_shell_has_valid_syntax() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(ROOT / "deploy/control-plane/regional/rollout-regional-release.sh"),
        ],
        check=True,
    )


def test_gpu_deployment_manifest_is_stamped_with_release_sha(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    document = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "executor"},
        "spec": {
            "template": {
                "metadata": {"annotations": {"gpu-fault.io/artifact-sha256": "old"}}
            }
        },
    }
    rendered = release._stamp_gpu_deployments(json.dumps(document))
    stamped = next(yaml.safe_load_all(rendered))
    annotations = stamped["spec"]["template"]["metadata"]["annotations"]

    assert set(annotations.values()) == {
        release.wheel_sha,
        release.wheel_sha[:12],
        MODULE.DEFAULT_RUNTIME_IMAGE,
    }


def test_executor_iam_boundary_accepts_minimal_role() -> None:
    MODULE.validate_executor_iam_documents(
        "arn:aws:iam::1:role/executor",
        [
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeCluster",
                            "sagemaker:ListClusterNodes",
                            "sagemaker:DescribeClusterNode",
                            "sagemaker:BatchRebootClusterNodes",
                            "s3:PutObject",
                        ],
                    }
                ]
            }
        ],
    )


@pytest.mark.parametrize(
    "action",
    [
        "ses:SendEmail",
        "sagemaker:BatchReplaceClusterNodes",
        "sagemaker:BatchDeleteClusterNodes",
        "sagemaker:*",
    ],
)
def test_executor_iam_boundary_rejects_excess_privilege(action: str) -> None:
    with pytest.raises(
        MODULE.ReleaseError, match="exceeds the regional data-plane boundary"
    ):
        MODULE.validate_executor_iam_documents(
            "arn:aws:iam::1:role/executor",
            [{"Statement": [{"Effect": "Allow", "Action": action}]}],
        )


def test_release_config_loads_content_addressed_manifest(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(manifest_config_file(tmp_path))

    assert config.wheel.name == "release.whl"
    assert config.bundle.name == "bundle.tar.gz"


def test_executor_iam_boundary_rejects_allow_not_action() -> None:
    with pytest.raises(MODULE.ReleaseError, match="Allow/NotAction"):
        MODULE.validate_executor_iam_documents(
            "arn:aws:iam::1:role/executor",
            [{"Statement": [{"Effect": "Allow", "NotAction": "iam:*"}]}],
        )


def test_release_renders_one_runtime_image_across_gpu_roles(
    tmp_path, monkeypatch
) -> None:
    runtime_image = "registry.example/gpu-fault/python@sha256:" + "a" * 64
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", runtime_image)
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class RecordingRunner:
        dry_run = True

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return ""

    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    target = config.clusters[0]

    release._apply_gpu_deployments(target, release.wheel_cm)
    rendered = [
        kwargs["input_text"]
        for _args, kwargs in runner.calls
        if kwargs.get("input_text")
    ]

    assert len(rendered) == 3
    assert all(runtime_image in item for item in rendered)
    assert all(MODULE.DEFAULT_RUNTIME_IMAGE not in item for item in rendered)

    release._deploy_reconciler(
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
    )
    assert runner.calls[-1][1]["env"]["GPU_FAULT_RUNTIME_IMAGE"] == runtime_image


def test_release_rejects_invalid_runtime_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", "registry.example/bad image")
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    with pytest.raises(MODULE.ReleaseError, match="OCI image"):
        MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
