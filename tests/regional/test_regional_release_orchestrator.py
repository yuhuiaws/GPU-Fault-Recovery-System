from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile
from gpu_fault.training_submit_cli import render_workload
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
MODULE = lazy_script_module("rollout_regional_release", MODULE_PATH)
RUNTIME_PROFILE_MODULE = lazy_script_module(
    "regional_runtime_profile",
    ROOT / "deploy/control-plane/regional/regional_runtime_profile.py",
)
RENDERING_MODULE = lazy_script_module(
    "regional_release_rendering",
    ROOT / "deploy/control-plane/regional/regional_release_rendering.py",
)
REGION = "us-east-1"
CPU_EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-fault-control-plane"
GPU_EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"


def test_upgrade_ensures_schema_before_rolling_cpu() -> None:
    source = inspect.getsource(MODULE.RegionalRelease.upgrade)

    assert source.index("self._upload_release()") < source.index(
        "self._ensure_schema()"
    )
    assert source.index("self._ensure_schema()") < source.index(
        "self._apply_cpu(finalize=False)"
    )
    assert source.index("self._apply_cpu(finalize=False)") < source.index(
        "ensure_runtime_profile(self)"
    )
    assert source.index("ensure_runtime_profile(self)") < source.index(
        "self._apply_gpu_deployments"
    )


def test_agent_convergence_uses_hyperpod_cluster_name(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    items = [
        {
            "metadata": {
                "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"},
                "annotations": {
                    "gpu-fault.io/installer-state": "Succeeded",
                    "gpu-fault.io/installer-artifact-sha256": hashlib.sha256(
                        b"wheel"
                    ).hexdigest(),
                },
            }
        },
        {
            "metadata": {
                "labels": {"sagemaker.amazonaws.com/cluster-name": "another-hyperpod"},
                "annotations": {},
            }
        },
    ]

    assert MODULE.agents_converged(
        items, config.clusters[0], hashlib.sha256(b"wheel").hexdigest()
    ), "agent convergence ignored the configured HyperPod cluster name"


def config_file(
    tmp_path: Path, *, clusters=None, profile_version: str = "hyperpod-v1"
) -> Path:
    wheel = tmp_path / "release.whl"
    bundle = tmp_path / "bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    value = {
        "aws_region": REGION,
        "cpu_kubeconfig": "/secure/cpu.kubeconfig",
        "cpu_eks_arn": CPU_EKS_ARN,
        "cpu_hyperpod_cluster_name": "gpu-fault-control-plane",
        "namespace": "gpu-fault-system",
        "runtime_profile": {
            "source": str(
                ROOT / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
            ),
            "version": profile_version,
            "registration_cluster_id": "gpu-a",
        },
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
                "region": REGION,
                "hyperpod_cluster_name": "hp-gpu-a",
                "eks_cluster_arn": GPU_EKS_ARN,
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
                "aws_region": REGION,
                "cpu_kubeconfig": "/secure/cpu.kubeconfig",
                "cpu_eks_arn": CPU_EKS_ARN,
                "cpu_hyperpod_cluster_name": "gpu-fault-control-plane",
                "runtime_profile": {
                    "source": str(
                        ROOT
                        / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
                    ),
                    "version": "hyperpod-v1",
                    "registration_cluster_id": "gpu-a",
                },
                "release": {"manifest": str(manifest), "agent_config_digest": "a" * 64},
                "clusters": [
                    {
                        "cluster_id": "gpu-a",
                        "context": "gpu-a-context",
                        "executor_irsa_role_arn": ("arn:aws:iam::1:role/a"),
                        "region": REGION,
                        "hyperpod_cluster_name": "hp-gpu-a",
                        "eks_cluster_arn": GPU_EKS_ARN,
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
        "region": REGION,
        "hyperpod_cluster_name": "hp-gpu-a",
        "eks_cluster_arn": GPU_EKS_ARN,
    }

    with pytest.raises(MODULE.ReleaseError, match="unique"):
        MODULE.ReleaseConfig.load(config_file(tmp_path, clusters=[cluster, cluster]))


def test_release_config_requires_runtime_profile_inputs(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value.pop("runtime_profile")
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="runtime_profile.source"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_requires_registered_profile_anchor(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["registration_cluster_id"] = "missing"
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="configured GPU cluster"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_requires_existing_profile_source(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["source"] = str(tmp_path / "missing-profile.yaml")
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="existing file"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_unsafe_profile_version(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["version"] = "hyperpod-v2&unexpected"
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="Runtime Profile version"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_requires_explicit_matching_region(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["aws_region"] = "REPLACE_WITH_AWS_REGION"
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="aws_region"):
        MODULE.ReleaseConfig.load(path)

    value["aws_region"] = REGION
    value["clusters"][0]["region"] = "us-west-2"
    path.write_text(json.dumps(value))
    with pytest.raises(MODULE.ReleaseError, match="does not match aws_region"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_cross_region_eks_arns(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["clusters"][0]["eks_cluster_arn"] = (
        "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
    )
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="eks_cluster_arn Region"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_cross_region_nlb_certificate(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["nlb"] = {
        "name": "gpu-fault-regional",
        "public_subnets": "subnet-a,subnet-b",
        "security_group": "sg-0123456789abcdef0",
        "certificate_arn": ("arn:aws:acm:us-west-2:123456789012:certificate/example"),
    }
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="certificate_arn Region"):
        MODULE.ReleaseConfig.load(path)


def test_nlb_manifest_uses_explicit_name_and_region_certificate(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["nlb"] = {
        "name": "gpu-fault-regional-test",
        "public_subnets": "subnet-a,subnet-b",
        "security_group": "sg-0123456789abcdef0",
        "certificate_arn": ("arn:aws:acm:us-east-1:123456789012:certificate/example"),
    }
    path.write_text(json.dumps(value))
    config = MODULE.ReleaseConfig.load(path)

    source = (
        ROOT / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
    ).read_text(encoding="utf-8")
    rendered = MODULE.render_nlb_manifest(config, source)

    assert "gpu-fault-regional-test" in rendered
    assert "arn:aws:acm:us-east-1:" in rendered
    assert "REPLACE_WITH" not in rendered


def test_plan_covers_first_deploy_and_rollback(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    deploy = release.plan("deploy")
    rollback = release.plan("rollback")
    join = release.plan("join-cluster")

    assert any("prerequisites" in step for step in deploy)
    assert any("PostgreSQL schema" in step for step in deploy)
    assert any("Runtime Profile" in step for step in deploy), (
        "deploy plan must include Runtime Profile registration"
    )
    assert any("previous required pins" in step for step in rollback)
    assert any("installer bundle" in step for step in rollback)
    assert any("Runtime Profile" in step for step in join), (
        "join plan must verify the shared Runtime Profile"
    )


class RuntimeProfileRunner:
    dry_run = False

    def __init__(self, existing=None) -> None:
        self.existing = existing
        self.posted = []

    @staticmethod
    def _desired(input_text: str) -> dict:
        profile = RuntimeProfile.model_validate(json.loads(input_text))
        return compile_runtime_profile(profile).model_dump(mode="json")

    def run(self, args, **kwargs):
        command = " ".join(args)
        if "get pod" in command:
            return "api-pod"
        if "compile_runtime_profile" in command:
            desired = self._desired(kwargs["input_text"])
            return json.dumps({"desired": desired, "existing": self.existing})
        if "/v1/runtime-profiles" in command:
            desired = self._desired(kwargs["input_text"])
            self.posted.append(json.loads(kwargs["input_text"]))
            return json.dumps(desired)
        raise AssertionError(f"unexpected Runtime Profile command: {args}")


def test_runtime_profile_payload_uses_declared_identity(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    payload = RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)

    assert payload["cluster_id"] == "gpu-a"
    assert payload["profile_version"] == "hyperpod-v1"
    assert payload["cluster_id"] != "hp-gpu-a"


def test_runtime_profile_is_registered_when_missing(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = RuntimeProfileRunner()
    release = MODULE.RegionalRelease(config, runner)

    MODULE.ensure_runtime_profile(release)

    assert len(runner.posted) == 1
    assert runner.posted[0]["cluster_id"] == "gpu-a"
    assert runner.posted[0]["profile_version"] == "hyperpod-v1"


def test_runtime_profile_registration_is_idempotent(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    desired = compile_runtime_profile(
        RuntimeProfile.model_validate(
            RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)
        )
    ).model_dump(mode="json")
    runner = RuntimeProfileRunner(existing=desired)
    release = MODULE.RegionalRelease(config, runner)

    MODULE.ensure_runtime_profile(release)

    assert runner.posted == []


def test_runtime_profile_drift_requires_a_new_version(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    existing = compile_runtime_profile(
        RuntimeProfile.model_validate(
            RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)
        )
    ).model_dump(mode="json")
    existing["capabilities"][0]["mode"] = "OBSERVE"
    runner = RuntimeProfileRunner(existing=existing)
    release = MODULE.RegionalRelease(config, runner)

    with pytest.raises(MODULE.ReleaseError, match="new profile version"):
        MODULE.ensure_runtime_profile(release)

    assert runner.posted == []


class PreflightRunner:
    dry_run = False

    def __init__(
        self, *, gpu_eks_arn: str = GPU_EKS_ARN, gpu_node_recovery: str = "None"
    ) -> None:
        self.gpu_eks_arn = gpu_eks_arn
        self.gpu_node_recovery = gpu_node_recovery

    def run(self, args, **kwargs):
        del kwargs
        if "config" in args and "view" in args:
            return CPU_EKS_ARN if "--kubeconfig" in args else self.gpu_eks_arn
        if args[:3] == ["aws", "sagemaker", "describe-cluster"]:
            cluster_name = args[args.index("--cluster-name") + 1]
            return json.dumps(
                {
                    "EksClusterArn": (
                        GPU_EKS_ARN if cluster_name == "hp-gpu-a" else CPU_EKS_ARN
                    ),
                    "NodeRecovery": (
                        self.gpu_node_recovery
                        if cluster_name == "hp-gpu-a"
                        else "Automatic"
                    ),
                }
            )
        if "--raw=/readyz" in args:
            return "ok"
        raise AssertionError(f"unexpected preflight command: {args}")


def test_preflight_binds_contexts_and_hyperpod_to_config(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, PreflightRunner())
    monkeypatch.setattr(release, "_validate_executor_iam_role", lambda _target: None)

    MODULE.ensure_region_contexts(release)


def test_preflight_rejects_wrong_context_and_managed_gpu_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    wrong_context = MODULE.RegionalRelease(
        config,
        PreflightRunner(
            gpu_eks_arn=("arn:aws:eks:us-east-1:123456789012:cluster/unexpected-gpu")
        ),
    )
    monkeypatch.setattr(
        wrong_context, "_validate_executor_iam_role", lambda _target: None
    )

    with pytest.raises(MODULE.ReleaseError, match="does not match"):
        MODULE.ensure_region_contexts(wrong_context)

    managed_recovery = MODULE.RegionalRelease(
        config, PreflightRunner(gpu_node_recovery="Automatic")
    )
    monkeypatch.setattr(
        managed_recovery, "_validate_executor_iam_role", lambda _target: None
    )
    with pytest.raises(MODULE.ReleaseError, match="NodeRecovery=None"):
        MODULE.ensure_region_contexts(managed_recovery)


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
    assert any(f"value: {REGION}" in item for item in rendered), (
        "GPU manifests did not receive the configured Region"
    )
    assert all("REPLACE_WITH_AWS_REGION" not in item for item in rendered), (
        "GPU manifests retained an unresolved Region placeholder"
    )

    release._deploy_reconciler(
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
    )
    assert runner.calls[-1][1]["env"]["GPU_FAULT_RUNTIME_IMAGE"] == runtime_image
    assert runner.calls[-1][1]["env"]["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v1"
    assert runner.calls[-1][1]["env"]["GPU_FAULT_CLUSTER_ID"] == "gpu-a"
    assert runner.calls[-1][1]["env"]["GPU_FAULT_HYPERPOD_CLUSTER"] == "hp-gpu-a"


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
