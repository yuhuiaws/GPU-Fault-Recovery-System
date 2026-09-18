from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from gpu_fault.admin.config_patch import preset_admin_config
from gpu_fault_release import regional_release_store_preflight as GATE
from gpu_fault_release import rollout as MODULE
from tests.deploy.test_installer_template_identity import template_job

ROOT = Path(__file__).resolve().parents[2]
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
    """Records every command; answers the in-flight install gate's two store
    reads the way a healthy control plane would (one Running Pod per role, a
    probe that ran and found nothing in flight), and ``""`` to everything else.
    Answering ``""`` to the gate too would mean "no Running Pod", which a manual
    rollback correctly refuses -- and the pins are what these tests are about."""

    def probe_output(self, args, **_kwargs):
        assert "get" in args and "cronjob" in args and "--ignore-not-found" in args
        return 0, "", ""

    dry_run = True

    def __init__(self) -> None:
        self.calls = []

    def run(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if GATE.RUNNING_PODS_JSONPATH in args:
            return "cpu-pod-a\ttrue"
        if GATE.PROBE_WRAPPER in args:
            clear = {"inflight": [], "inflight_count": 0, "bounded": True, "scanned": 0}
            return f"{json.dumps(clear)}\n{GATE.PROBE_EXIT_MARKER}=0"
        return ""

    def probe(self, _args, **_kwargs):
        return True


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


def _rollback_started_state(cluster_id: str) -> dict[str, Any]:
    cluster_components = ["executor", "watcher", "collector", "reconciler", "agent"]
    return {
        "execution_plan": {
            "nodes": [
                "registry",
                "cpu-stage",
                *cluster_components,
                "cpu-finalize",
                "verify",
            ]
        },
        "component_progress": {
            "schema_version": 1,
            "global": {
                name: {"status": "STARTED"}
                for name in ("registry", "cpu-stage", "cpu-finalize")
            },
            "clusters": {
                cluster_id: {name: {"status": "STARTED"} for name in cluster_components}
            },
        },
    }


def test_rollback_uses_previous_executor_and_node_pins(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(_config_file(tmp_path))
    release = MODULE.RegionalRelease(config, RecordingRunner())
    previous_executor = "a" * 64
    previous_executor_compatibility = "b" * 64
    previous_agent = "c" * 64
    previous_agent_compatibility = "d" * 64
    previous_runtime_image = "registry.example/runtime@sha256:" + "1" * 64
    previous_installer_image = "registry.example/installer@sha256:" + "2" * 64
    gpu_calls = []
    reconciler_calls = []
    restore_calls = []
    monkeypatch.setattr(release, "_config_map_sha", lambda *_args: "e" * 64)
    monkeypatch.setattr(
        release,
        "_restore_registry_backup",
        lambda: restore_calls.append("registry") or True,
    )
    monkeypatch.setattr(
        release,
        "_publish_restored_registry",
        lambda: restore_calls.append("publish-registry") or {},
    )
    monkeypatch.setattr(
        release,
        "_restore_cpu_role_config_maps",
        lambda snapshots: restore_calls.append(("config-maps", snapshots)) or True,
    )
    monkeypatch.setattr(release, "_verify_gpu_control_plane_endpoint", lambda *_: None)
    monkeypatch.setattr(release, "_apply_gpu_dcgm_exporter", lambda *_: None)
    monkeypatch.setattr(
        release,
        "_apply_gpu_deployments",
        lambda *args, **kwargs: gpu_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        release,
        "_roll_node_runtime",
        lambda *args, **kwargs: reconciler_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(release, "_cancel_active_installer_jobs", lambda *_args: None)
    monkeypatch.setattr(
        release,
        "_fleet_deployment_id",
        lambda *_args, **_kwargs: "candidate-agent-deployment",
    )
    monkeypatch.setattr(
        release, "_fleet_command", lambda *_args, **_kwargs: {"status": "CANCELLED"}
    )
    monkeypatch.setattr(release, "_save_state", lambda *_args, **_kwargs: None)
    target = config.clusters[0]
    release.state = _rollback_started_state(target.cluster_id)
    job = template_job(offline=False)
    pod = job["spec"]["template"]["spec"]
    pod["containers"][0]["image"] = previous_installer_image
    pod["volumes"][-1]["configMap"]["name"] = "old-node-bundle"
    template_text = yaml.safe_dump(job)
    previous_admin_config = preset_admin_config("32-enabled")
    previous = {
        "release_manifest_schema_version": 3,
        "cpu_wheel": "old-control-wheel",
        "aurora_refresh": None,
        "runtime_image": previous_runtime_image,
        "node_installer_image": previous_installer_image,
        "runtime_profile_version": "hyperpod-v1",
        "admin_config": previous_admin_config.as_dict(),
        "cpu_role_config_maps": {
            "gpu-fault-control-worker-config-core": {"GPU_FAULT_SERVICE_ROLE": "worker"}
        },
        "metadata": {
            "required-agent-artifact-sha256": previous_agent,
            "required-agent-compatibility-digest": previous_agent_compatibility,
            "required-agent-config-digest": "f" * 64,
            "required-agent-protocol-version": "3",
            "required-node-action-key-version": "2",
            "required-regional-executor-artifact-sha256": previous_executor,
            "required-regional-executor-compatibility-digest": (
                previous_executor_compatibility
            ),
        },
        "agent_identities": {
            target.cluster_id: {
                "agent_protocol_version": 3,
                "agent_version": "0.9.0",
                "artifact_sha256": previous_agent,
                "compatibility_digest": previous_agent_compatibility,
                "installer_bundle_sha256": None,
                "installer_template_sha256": None,
                "policy_version": "catalog-a",
                "runtime_profile_version": "hyperpod-v1",
                "config_digest": "f" * 64,
                "node_action_key_version": 2,
                "node_ids": ["node-a"],
            }
        },
        "clusters": {
            target.cluster_id: {
                "wheel": "old-executor-wheel",
                "wheel_key": "old-executor.whl",
                "reconciler_wheel": "old-executor-wheel",
                "reconciler_wheel_key": "old-executor.whl",
                "bundle": "old-node-bundle",
                "bundle_sha256": "e" * 64,
                "template": "old-node-template",
                "template_sha256": "3" * 64,
                "template_content_sha256": hashlib.sha256(
                    template_text.encode()
                ).hexdigest(),
            }
        },
    }

    release.rollback(state=previous)

    assert release.state["inflight_installs"]["verdict"] == "clear", (
        "the gate ran for real against the runner's clear snapshot and its "
        "verdict rode into the rollback's state"
    )
    assert restore_calls == [
        "registry",
        "publish-registry",
        "registry",
        "publish-registry",
        ("config-maps", previous["cpu_role_config_maps"]),
    ], "a restored registry backup must be republished to the durable head"
    cpu_applies = [
        kwargs
        for args, kwargs in release.runner.calls
        if args[0] == "bash" and "apply-control-plane-role-split.sh" in args[1]
    ]
    assert len(cpu_applies) == 2
    assert all(
        item["env"]["GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS"] == "true"
        for item in cpu_applies
    ), "rollback CPU stages did not preserve the selected role ConfigMaps"
    assert all(
        item["env"]["GPU_FAULT_FORCE_ROLE_RESTART"] == "true" for item in cpu_applies
    ), "rollback CPU stages did not force a restart"
    assert cpu_applies[0]["env"]["GPU_FAULT_RUNTIME_IMAGE"] == release.runtime_image
    assert cpu_applies[1]["env"]["GPU_FAULT_RUNTIME_IMAGE"] == previous_runtime_image
    cpu_renders = [
        kwargs
        for args, kwargs in release.runner.calls
        if args[0] == "bash" and "render-control-plane-role-split.sh" in args[1]
    ]
    assert len(cpu_renders) == 2
    assert all(
        item["env"]["GPU_FAULT_TELEMETRY_SPOOL"] == "true"
        and item["env"]["GPU_FAULT_TELEMETRY_SPOOL_REPLICAS"] == "3"
        and item["env"]["GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION"] == "128"
        for item in cpu_renders
    ), "rollback CPU stages did not render the previous administrator config"
    core_patches = [
        args
        for args, _kwargs in release.runner.calls
        if "patch" in args and "configmap" in args and "--type=merge" in args
    ]
    assert len(core_patches) == 3
    patch = json.loads(core_patches[0][-1])
    assert patch["data"] == {
        "GPU_FAULT_REQUIRED_AGENT_VERSION": "0.9.0",
        "GPU_FAULT_REQUIRED_POLICY_VERSION": "catalog-a",
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "hyperpod-v1",
    }
    assert gpu_calls[0][1]["executor_artifact_sha"] == previous_executor
    assert (
        gpu_calls[0][1]["executor_compatibility_digest"]
        == previous_executor_compatibility
    )
    assert gpu_calls[0][1]["runtime_image"] == previous_runtime_image
    assert (
        reconciler_calls[0][1]["node_compatibility_digest"]
        == previous_agent_compatibility
    )
    assert reconciler_calls[0][1]["phase"] == "rollback"
    assert reconciler_calls[0][1]["runtime_image"] == release.runtime_image
    assert reconciler_calls[0][1]["steady_runtime_image"] == previous_runtime_image
    assert reconciler_calls[0][1]["node_installer_image"] == previous_installer_image
    assert reconciler_calls[0][1]["bundle_sha256"] == "e" * 64
    assert reconciler_calls[0][1]["template_sha256"] == "3" * 64
    assert reconciler_calls[0][1]["steady_template_config_map"] == "old-node-template"
