from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpu_fault.training_submit_cli import render_workload
from gpu_fault_release import regional_release_rendering as RENDERING_MODULE
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]


def test_release_rejects_invalid_runtime_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", "registry.example/bad image")
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    with pytest.raises(MODULE.ReleaseError, match="OCI image"):
        MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))


def test_non_default_runtime_profile_reaches_every_plane(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="hyperpod-v2")
    )

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

    cpu_environment = RENDERING_MODULE.build_cpu_apply_environment(
        release, finalize=False
    )
    assert (
        cpu_environment["GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"] == "hyperpod-v2"
    )
    assert cpu_environment["GPU_FAULT_ADMIN_CONFIG_SHA256"] == (
        release.admin_config_digest
    )

    rendered_manifests = [
        text
        for _deployment, text in RENDERING_MODULE.render_gpu_rollout_manifests(
            release, target, release.wheel_cm
        )
    ]
    resources = [
        document
        for text in rendered_manifests
        for document in yaml.safe_load_all(text)
        if isinstance(document, dict)
    ]
    collector = next(
        item
        for item in resources
        if item.get("kind") == "Deployment"
        and item["metadata"]["name"] == "gpu-fault-kubernetes-node-resource-collector"
    )
    collector_env = {
        item["name"]: item.get("value")
        for item in collector["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert collector_env["GPU_FAULT_RUNTIME_PROFILE_VERSION"] == "hyperpod-v2"
    assert all(
        "REPLACE_WITH_RUNTIME_PROFILE_VERSION" not in item
        for item in rendered_manifests
    ), "rendered GPU manifests retained the Runtime Profile placeholder"

    reconciler_environment = RENDERING_MODULE.build_reconciler_environment(
        release,
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
    )
    assert reconciler_environment["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v2"

    workload = render_workload(
        ROOT / "examples/hyperpod/three-node-pytorchjob.yaml",
        job_id="profile-v2-job",
        attempt_id=None,
        attempt_number=1,
        runtime_profile_version="hyperpod-v2",
        expected_critical_ranks=None,
        training_container="pytorch",
        restart_budget=1,
        namespace="training",
    )
    document = yaml.safe_load(workload.manifest)
    for replica in document["spec"]["pytorchReplicaSpecs"].values():
        annotations = replica["template"]["metadata"]["annotations"]
        assert annotations["gpu-fault.io/runtime-profile-version"] == "hyperpod-v2"


def test_runtime_profile_override_restores_rollback_version(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="hyperpod-v2")
    )

    class RecordingRunner:
        dry_run = True

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return ""

    release = MODULE.RegionalRelease(config, RecordingRunner())
    target = config.clusters[0]
    rendered = RENDERING_MODULE.render_gpu_rollout_manifests(
        release, target, release.wheel_cm, runtime_profile_version="hyperpod-v1"
    )
    resources = [
        document
        for _deployment, text in rendered
        for document in yaml.safe_load_all(text)
        if isinstance(document, dict)
    ]
    collector = next(
        item
        for item in resources
        if item.get("kind") == "Deployment"
        and item["metadata"]["name"] == "gpu-fault-kubernetes-node-resource-collector"
    )
    collector_env = {
        item["name"]: item.get("value")
        for item in collector["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert collector_env["GPU_FAULT_RUNTIME_PROFILE_VERSION"] == "hyperpod-v1"

    environment = RENDERING_MODULE.build_reconciler_environment(
        release,
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
        runtime_profile_version="hyperpod-v1",
    )
    assert environment["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v1"
