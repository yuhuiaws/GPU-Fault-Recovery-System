from __future__ import annotations

import json
from pathlib import Path

import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    "rollout_regional_release_artifact_pins",
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py",
)
REGION = "us-east-1"


def _config_file(tmp_path: Path) -> Path:
    wheel = tmp_path / "release.whl"
    bundle = tmp_path / "bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "aws_region": REGION,
                "cpu_kubeconfig": "/secure/cpu.kubeconfig",
                "cpu_eks_arn": (
                    "arn:aws:eks:us-east-1:123456789012:cluster/gpu-fault-control-plane"
                ),
                "cpu_hyperpod_cluster_name": "gpu-fault-control-plane",
                "namespace": "gpu-fault-system",
                "runtime_profile": {
                    "source": str(
                        ROOT
                        / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
                    ),
                    "version": "hyperpod-v1",
                    "registration_cluster_id": "gpu-a",
                },
                "release": {
                    "wheel": str(wheel),
                    "bundle": str(bundle),
                    "agent_config_digest": "a" * 64,
                },
                "clusters": [
                    {
                        "cluster_id": "gpu-a",
                        "context": "gpu-a-context",
                        "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
                        "region": REGION,
                        "hyperpod_cluster_name": "hp-gpu-a",
                        "eks_cluster_arn": (
                            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class RecordingRunner:
    dry_run = True

    def __init__(self) -> None:
        self.calls = []

    def run(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return ""


def test_gpu_manifest_can_restore_previous_executor_pin(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(_config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    previous_artifact = "a" * 64
    previous_compatibility = "b" * 64
    document = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "executor"},
        "spec": {"template": {"metadata": {}}},
    }

    rendered = MODULE.stamp_gpu_deployments(
        release,
        json.dumps(document),
        executor_artifact_sha=previous_artifact,
        executor_compatibility_digest=previous_compatibility,
    )

    annotations = next(yaml.safe_load_all(rendered))["spec"]["template"]["metadata"][
        "annotations"
    ]
    assert annotations["gpu-fault.io/artifact-sha256"] == previous_artifact
    assert annotations["gpu-fault.io/executor-wheel-sha256"] == previous_artifact
    assert (
        annotations["gpu-fault.io/executor-compatibility-digest"]
        == previous_compatibility
    )


def test_executor_pin_preflight_accepts_only_staged_artifacts() -> None:
    required = "a" * 64
    candidate = "b" * 64
    compatibility = "c" * 64
    metadata = {
        "required-regional-executor-protocol-version": "2",
        "required-regional-executor-artifact-sha256": required,
        "compatible-regional-executor-artifact-sha256s": candidate,
        "required-regional-executor-compatibility-digest": compatibility,
    }

    assert (
        MODULE.executor_pin_rejection(
            metadata,
            protocol_version=2,
            artifact_sha=candidate,
            compatibility_digest=compatibility,
        )
        is None
    )
    rejection = MODULE.executor_pin_rejection(
        {**metadata, "compatible-regional-executor-artifact-sha256s": ""},
        protocol_version=2,
        artifact_sha=candidate,
        compatibility_digest=compatibility,
    )
    assert rejection is not None
    assert "artifact mismatch" in rejection


def test_gpu_rollouts_use_five_minute_timeout(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(_config_file(tmp_path))
    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)

    MODULE.apply_gpu_deployments(release, config.clusters[0], release.executor_wheel_cm)

    rollout_calls = [
        args for args, _kwargs in runner.calls if "rollout" in args and "status" in args
    ]
    assert rollout_calls, "GPU rollout did not wait for any Deployment"
    assert all("--timeout=5m" in args for args in rollout_calls), (
        "a fast GPU Deployment retained a rollout timeout longer than five minutes"
    )


def test_rollback_uses_previous_executor_and_node_pins(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(_config_file(tmp_path))
    release = MODULE.RegionalRelease(config, RecordingRunner())
    previous_executor = "a" * 64
    previous_executor_compatibility = "b" * 64
    previous_agent = "c" * 64
    previous_agent_compatibility = "d" * 64
    gpu_calls = []
    reconciler_calls = []
    monkeypatch.setattr(release, "_config_map_sha", lambda *_args: "e" * 64)
    monkeypatch.setattr(release, "_verify_gpu_control_plane_endpoint", lambda *_: None)
    monkeypatch.setattr(release, "_apply_gpu_dcgm_exporter", lambda *_: None)
    monkeypatch.setattr(
        release,
        "_apply_gpu_deployments",
        lambda *args, **kwargs: gpu_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        release,
        "_deploy_reconciler",
        lambda *args, **kwargs: reconciler_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(release, "_wait_agents", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release, "_save_state", lambda *_args, **_kwargs: None)
    target = config.clusters[0]
    previous = {
        "cpu_wheel": "old-control-wheel",
        "runtime_profile_version": "hyperpod-v1",
        "metadata": {
            "required-agent-artifact-sha256": previous_agent,
            "required-agent-compatibility-digest": previous_agent_compatibility,
            "required-agent-config-digest": "f" * 64,
            "required-regional-executor-artifact-sha256": previous_executor,
            "required-regional-executor-compatibility-digest": (
                previous_executor_compatibility
            ),
        },
        "clusters": {
            target.cluster_id: {
                "wheel": "old-executor-wheel",
                "wheel_key": "old-executor.whl",
                "reconciler_wheel": "old-executor-wheel",
                "reconciler_wheel_key": "old-executor.whl",
                "bundle": "old-node-bundle",
            }
        },
    }

    release.rollback(state=previous)

    assert gpu_calls[0][1]["executor_artifact_sha"] == previous_executor
    assert (
        gpu_calls[0][1]["executor_compatibility_digest"]
        == previous_executor_compatibility
    )
    assert (
        reconciler_calls[0][1]["node_compatibility_digest"]
        == previous_agent_compatibility
    )
