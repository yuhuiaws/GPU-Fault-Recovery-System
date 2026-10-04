"""The release engine on a plain EKS (``eks-managed``) GPU cluster.

Every proof here has a HyperPod twin: the HyperPod cluster must render the
selector, Secret keys, manifests and IAM boundary it always rendered, byte for
byte, while the plain EKS cluster gets its own selector, provider keys, watcher
environment and EC2-scoped IAM allowlist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.cluster_scope import (
    CONNECTION_SECRET_EKS_CLUSTER_NAME_KEY,
    CONNECTION_SECRET_NODE_SELECTOR_KEY,
    CONNECTION_SECRET_PROVIDER_KEY,
    NODE_SELECTOR_ENV,
    PROVIDER_ENV,
)
from gpu_fault_release import regional_gpu_bootstrap as GPU_BOOTSTRAP
from gpu_fault_release import regional_release_gpu_rollout as GPU_ROLLOUT
from gpu_fault_release import regional_release_iam as IAM
from gpu_fault_release import regional_release_preflight as PREFLIGHT
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_profile_model import cluster_profiles_digest
from tests.regional._release_orchestrator_support import (
    GPU_EKS_ARN,
    REGION,
    config_file,
)

ACCOUNT = "123456789012"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-eks-cluster"
HYPERPOD_SELECTOR = "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
EKS_SELECTOR = "nvidia.com/gpu.present=true"


def _hyperpod_cluster(tmp_path: Path) -> dict[str, Any]:
    token = tmp_path / "hyperpod.token"
    token.write_text("t" * 64, encoding="utf-8")
    ca = tmp_path / "ca.crt"
    ca.write_text("certificate", encoding="utf-8")
    return {
        "cluster_id": "gpu-a",
        "context": "gpu-a-context",
        "executor_irsa_role_arn": f"arn:aws:iam::{ACCOUNT}:role/a",
        "region": REGION,
        "hyperpod_cluster_name": "hp-gpu-a",
        "eks_cluster_arn": GPU_EKS_ARN,
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
        "token_file": str(token),
        "ca_file": str(ca),
        "control_plane_url": "https://control.example",
        "allowed_namespaces": ["training"],
    }


def _eks_cluster(tmp_path: Path) -> dict[str, Any]:
    value = _hyperpod_cluster(tmp_path)
    value.update(
        {
            "cluster_id": "gpu-eks-cluster",
            "context": "gpu-eks-context",
            "executor_irsa_role_arn": f"arn:aws:iam::{ACCOUNT}:role/eks",
            "provider": "eks-managed",
            "node_selector": EKS_SELECTOR,
            "eks_cluster_arn": EKS_ARN,
            "agent_endpoint_allowed_cidrs": ["10.0.0.0/20", "10.0.16.0/20"],
        }
    )
    value.pop("hyperpod_cluster_name")
    return value


class RecordingRunner:
    dry_run = True

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.secret_files: dict[str, bytes] = {}

    def run(self, args: list[str], **kwargs: Any) -> str:
        self.calls.append((list(args), dict(kwargs)))
        for item in args:
            if str(item).startswith("--from-file="):
                name, _, path = str(item)[len("--from-file=") :].partition("=")
                # Read while the temporary directory still exists.
                self.secret_files[name] = Path(path).read_bytes()
        return ""


@pytest.fixture
def release(tmp_path: Path) -> MODULE.RegionalRelease:
    config = MODULE.ReleaseConfig.load(
        config_file(
            tmp_path, clusters=[_hyperpod_cluster(tmp_path), _eks_cluster(tmp_path)]
        )
    )
    return MODULE.RegionalRelease(config, RecordingRunner())


def _targets(release: MODULE.RegionalRelease) -> tuple[ClusterTarget, ClusterTarget]:
    hyperpod, eks = release.config.clusters
    assert (hyperpod.cluster_id, eks.cluster_id) == ("gpu-a", "gpu-eks-cluster"), (
        "fixture clusters must load in declaration order"
    )
    return hyperpod, eks


# --- node inventory read -------------------------------------------------------


def test_the_shared_node_read_uses_each_cluster_selector(release) -> None:
    hyperpod, eks = _targets(release)
    hyperpod_command = GPU_ROLLOUT.gpu_node_command(release, hyperpod)
    eks_command = GPU_ROLLOUT.gpu_node_command(release, eks)
    assert hyperpod_command[-2:] == ["-l", HYPERPOD_SELECTOR], (
        "a HyperPod cluster must still be read on the SageMaker label text"
    )
    assert eks_command[-2:] == ["-l", EKS_SELECTOR], (
        "a plain EKS cluster must be read on its configured selector"
    )


# --- connection Secret -----------------------------------------------------------


def _secret_files(release: MODULE.RegionalRelease, target: ClusterTarget) -> dict:
    runner: RecordingRunner = release.runner  # type: ignore[assignment]
    runner.secret_files.clear()
    GPU_BOOTSTRAP.ensure_connection_secret(release, target)
    return {name: value.decode() for name, value in runner.secret_files.items()}


def test_hyperpod_connection_secret_keeps_its_keys_and_gains_the_provider_pair(
    release,
) -> None:
    hyperpod, _eks = _targets(release)
    files = _secret_files(release, hyperpod)
    assert files["hyperpod-cluster-name"] == "hp-gpu-a", "HyperPod name key unchanged"
    assert files["hyperpod-confirm-cluster-name"] == "hp-gpu-a", "confirm key unchanged"
    assert files[CONNECTION_SECRET_PROVIDER_KEY] == "hyperpod", "provider key"
    assert files[CONNECTION_SECRET_NODE_SELECTOR_KEY] == HYPERPOD_SELECTOR, (
        "the HyperPod Secret carries the historical label text as its selector"
    )
    assert set(files) == {
        "cluster-token",
        "ca.crt",
        "control-plane-url",
        "cluster-id",
        "allowed-namespaces",
        "hyperpod-cluster-name",
        "hyperpod-confirm-cluster-name",
        CONNECTION_SECRET_PROVIDER_KEY,
        CONNECTION_SECRET_NODE_SELECTOR_KEY,
        CONNECTION_SECRET_EKS_CLUSTER_NAME_KEY,
    }, "the Secret key set must be the historical keys plus the provider keys"
    assert files[CONNECTION_SECRET_EKS_CLUSTER_NAME_KEY] == "", (
        "a HyperPod Secret carries no EKS cluster name for the EKS provider"
    )


def test_eks_managed_connection_secret_is_created_with_the_cluster_id_scope(
    release,
) -> None:
    _hyperpod, eks = _targets(release)
    files = _secret_files(release, eks)
    assert files, "an eks-managed cluster must create its Secret, never skip it"
    assert files["hyperpod-cluster-name"] == "gpu-eks-cluster", (
        "the provider scope keys carry the cluster_id for a plain EKS cluster"
    )
    assert files["hyperpod-confirm-cluster-name"] == "gpu-eks-cluster", (
        "the confirm key must match so the executor's fail-closed start passes"
    )
    assert files[CONNECTION_SECRET_PROVIDER_KEY] == "eks-managed", "provider key"
    assert files[CONNECTION_SECRET_NODE_SELECTOR_KEY] == EKS_SELECTOR, "selector key"
    assert files[CONNECTION_SECRET_EKS_CLUSTER_NAME_KEY] == "gpu-eks-cluster", (
        "the EKS cluster name feeds GPU_FAULT_EKS_CLUSTER_NAME for the EC2 tag check"
    )
    assert files["cluster-id"] == "gpu-eks-cluster", "cluster id key unchanged"


# --- manifests ---------------------------------------------------------------------


def _deployments(release: MODULE.RegionalRelease, target: ClusterTarget) -> dict:
    rendered = RENDERING.render_gpu_rollout_manifests(release, target, release.wheel_cm)
    documents = {}
    for deployment, text in rendered:
        assert "REPLACE_WITH" not in text, f"{deployment} kept a placeholder"
        for document in yaml.safe_load_all(text):
            if isinstance(document, dict) and document.get("kind") == "Deployment":
                documents[document["metadata"]["name"]] = document
    return documents


def _env(deployment: dict, container_index: int = 0) -> dict[str, Any]:
    containers = deployment["spec"]["template"]["spec"]["containers"]
    return {item["name"]: item for item in containers[container_index]["env"]}


def test_watcher_environment_is_eks_for_plain_eks_and_unchanged_for_hyperpod(
    release,
) -> None:
    hyperpod, eks = _targets(release)
    hyperpod_env = _env(_deployments(release, hyperpod)["gpu-fault-completion-watcher"])
    eks_env = _env(_deployments(release, eks)["gpu-fault-completion-watcher"])
    assert hyperpod_env["GPU_FAULT_WATCHER_ENVIRONMENT"]["value"] == "hyperpod-eks", (
        "a HyperPod watcher keeps the hyperpod-eks environment"
    )
    assert eks_env["GPU_FAULT_WATCHER_ENVIRONMENT"]["value"] == "eks", (
        "a plain EKS watcher must run the eks environment"
    )


@pytest.mark.parametrize(
    "deployment", ["gpu-fault-cluster-executor", "gpu-fault-completion-watcher"]
)
def test_rolled_out_deployments_read_the_provider_keys_optionally(
    release, deployment
) -> None:
    hyperpod, eks = _targets(release)
    for target in (hyperpod, eks):
        env = _env(_deployments(release, target)[deployment])
        keys = [
            (PROVIDER_ENV, CONNECTION_SECRET_PROVIDER_KEY),
            (NODE_SELECTOR_ENV, CONNECTION_SECRET_NODE_SELECTOR_KEY),
        ]
        if deployment == "gpu-fault-cluster-executor":
            keys.append(
                ("GPU_FAULT_EKS_CLUSTER_NAME", CONNECTION_SECRET_EKS_CLUSTER_NAME_KEY)
            )
        for name, key in keys:
            reference = env[name]["valueFrom"]["secretKeyRef"]
            assert reference["name"] == "gpu-fault-regional-connection", (
                f"{deployment}: {name} must come from the connection Secret"
            )
            assert reference["key"] == key, f"{deployment}: {name} reads key {key}"
            assert reference["optional"] is True, (
                f"{deployment}: {name} must be optional so an older HyperPod "
                "Secret without the keys still starts the Pod"
            )


def test_hyperpod_manifests_render_identically_to_a_single_cluster_site(
    tmp_path, release
) -> None:
    """Adding a plain EKS sibling to a site must not move the HyperPod render."""

    hyperpod, _eks = _targets(release)
    single_root = tmp_path / "single"
    single_root.mkdir()
    single = MODULE.RegionalRelease(
        MODULE.ReleaseConfig.load(
            config_file(single_root, clusters=[_hyperpod_cluster(tmp_path)])
        ),
        RecordingRunner(),
    )
    mixed = RENDERING.render_gpu_rollout_manifests(release, hyperpod, release.wheel_cm)
    alone = RENDERING.render_gpu_rollout_manifests(
        single, single.config.clusters[0], single.wheel_cm
    )
    assert mixed == alone, "HyperPod manifests must not depend on their siblings"


def test_reconciler_manifest_declares_the_optional_provider_keys() -> None:
    text = (
        Path(__file__).resolve().parents[2]
        / "deploy/dataplane/node-installer-reconciler.yaml"
    ).read_text(encoding="utf-8")
    deployment = next(
        item
        for item in yaml.safe_load_all(text)
        if isinstance(item, dict) and item.get("kind") == "Deployment"
    )
    env = _env(deployment)
    assert (
        env["GPU_FAULT_HYPERPOD_CLUSTER"]["value"] == "REPLACE_WITH_HYPERPOD_CLUSTER"
    ), "the literal provider scope placeholder must stay for the deploy script"
    for name, key in (
        (PROVIDER_ENV, CONNECTION_SECRET_PROVIDER_KEY),
        (NODE_SELECTOR_ENV, CONNECTION_SECRET_NODE_SELECTOR_KEY),
    ):
        reference = env[name]["valueFrom"]["secretKeyRef"]
        assert (reference["key"], reference["optional"]) == (key, True), (
            f"the reconciler must read {name} optionally from the connection Secret"
        )


# --- reconciler environment and product digest ---------------------------------------


def _reconciler_environment(release, target) -> dict[str, str]:
    return RENDERING.build_reconciler_environment(
        release,
        target,
        wheel_cm="wheel-cm",
        bundle_cm="bundle-cm",
        artifact_sha="a" * 64,
        config_digest="b" * 64,
    )


def test_reconciler_environment_carries_the_provider_scope(release) -> None:
    hyperpod, eks = _targets(release)
    hyperpod_env = _reconciler_environment(release, hyperpod)
    eks_env = _reconciler_environment(release, eks)
    assert hyperpod_env["GPU_FAULT_HYPERPOD_CLUSTER"] == "hp-gpu-a", "HyperPod scope"
    assert hyperpod_env[PROVIDER_ENV] == "hyperpod", "HyperPod provider"
    assert hyperpod_env[NODE_SELECTOR_ENV] == HYPERPOD_SELECTOR, (
        "the HyperPod reconciler receives the historical label as its selector"
    )
    assert eks_env["GPU_FAULT_HYPERPOD_CLUSTER"] == "gpu-eks-cluster", (
        "a plain EKS reconciler receives the cluster_id as provider scope"
    )
    assert eks_env[PROVIDER_ENV] == "eks-managed", "plain EKS provider"
    assert eks_env[NODE_SELECTOR_ENV] == EKS_SELECTOR, "plain EKS selector"


def test_product_digest_is_unchanged_for_hyperpod_and_binds_the_eks_selector(
    release,
) -> None:
    hyperpod, eks = _targets(release)
    hyperpod_env = _reconciler_environment(release, hyperpod)
    without_provider = {
        key: value
        for key, value in hyperpod_env.items()
        if key not in {PROVIDER_ENV, NODE_SELECTOR_ENV}
    }
    assert RENDERING.installer_product_inputs_digest(
        hyperpod_env
    ) == RENDERING.installer_product_inputs_digest(without_provider), (
        "a HyperPod site's recorded products must keep their digest"
    )
    eks_env = _reconciler_environment(release, eks)
    moved = {**eks_env, NODE_SELECTOR_ENV: "node-type=gpu"}
    assert RENDERING.installer_product_inputs_digest(
        eks_env
    ) != RENDERING.installer_product_inputs_digest(moved), (
        "changing a plain EKS selector must invalidate the recorded products"
    )


# --- preflight -------------------------------------------------------------------------


class PreflightRunner:
    def __init__(self, *, status: str = "ACTIVE", nodegroup_status: str = "ACTIVE"):
        self.status = status
        self.nodegroup_status = nodegroup_status
        self.calls: list[list[str]] = []

    def run(self, args: list[str], **_kwargs: Any) -> str:
        self.calls.append(list(args))
        if args[1:3] == ["eks", "describe-cluster"]:
            return json.dumps({"Arn": EKS_ARN, "Status": self.status})
        if args[1:3] == ["eks", "list-nodegroups"]:
            return json.dumps({"nodegroups": ["cpu-ng", "gpu-ng"]})
        if args[1:3] == ["eks", "describe-nodegroup"]:
            name = args[args.index("--nodegroup-name") + 1]
            return json.dumps(self.nodegroup_status if name == "gpu-ng" else "ACTIVE")
        raise AssertionError(f"unexpected preflight command {args}")


def test_eks_managed_preflight_requires_active_cluster_and_nodegroups() -> None:
    runner = PreflightRunner()
    PREFLIGHT.validate_eks_managed_cluster(
        runner, region=REGION, expected_eks_arn=EKS_ARN
    )
    assert not any("sagemaker" in call for call in runner.calls), (
        "a plain EKS preflight must not describe a HyperPod cluster"
    )
    with pytest.raises(ReleaseError, match="is not ACTIVE"):
        PREFLIGHT.validate_eks_managed_cluster(
            PreflightRunner(status="UPDATING"), region=REGION, expected_eks_arn=EKS_ARN
        )
    with pytest.raises(ReleaseError, match="node group gpu-ng .* is not ACTIVE"):
        PREFLIGHT.validate_eks_managed_cluster(
            PreflightRunner(nodegroup_status="DEGRADED"),
            region=REGION,
            expected_eks_arn=EKS_ARN,
        )


def test_eks_managed_preflight_refuses_a_different_cluster_arn() -> None:
    with pytest.raises(ReleaseError, match="does not match configured EKS ARN"):
        PREFLIGHT.validate_eks_managed_cluster(
            PreflightRunner(),
            region=REGION,
            expected_eks_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/other",
        )


# --- IAM boundary ------------------------------------------------------------------------

ROLE = f"arn:aws:iam::{ACCOUNT}:role/executor"


def _eks_policy() -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "ec2:DescribeInstances", "Resource": "*"},
            {
                "Effect": "Allow",
                "Action": "ec2:RebootInstances",
                "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*",
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceTag/eks:cluster-name": "gpu-eks-cluster"
                    }
                },
            },
        ],
    }


def test_eks_managed_executor_allowlist_accepts_describe_and_tag_scoped_reboot() -> (
    None
):
    IAM.validate_executor_iam_documents(ROLE, [_eks_policy()], provider="eks-managed")


@pytest.mark.parametrize(
    "statement, reason",
    [
        (
            {"Effect": "Allow", "Action": "sagemaker:DescribeCluster", "Resource": "*"},
            "sagemaker",
        ),
        (
            {
                "Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*",
            },
            "TerminateInstances",
        ),
        (
            {"Effect": "Allow", "Action": "ec2:RebootInstances", "Resource": "*"},
            'Resource "\\*"',
        ),
        ({"Effect": "Allow", "Action": "ec2:*", "Resource": "*"}, "ec2:\\*"),
    ],
)
def test_eks_managed_executor_allowlist_rejects_everything_else(
    statement, reason
) -> None:
    policy = _eks_policy()
    policy["Statement"].append(statement)
    with pytest.raises(ReleaseError, match=reason):
        IAM.validate_executor_iam_documents(ROLE, [policy], provider="eks-managed")


def test_hyperpod_executor_allowlist_is_unchanged() -> None:
    hyperpod_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:DescribeCluster",
                    "sagemaker:ListClusterNodes",
                    "sagemaker:DescribeClusterNode",
                    "sagemaker:BatchRebootClusterNodes",
                ],
                "Resource": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp",
            }
        ],
    }
    IAM.validate_executor_iam_documents(ROLE, [hyperpod_policy])
    # A HyperPod role is still refused the wildcard resource, even for the EC2
    # read a plain EKS role may hold on "*": the exemption is provider-scoped.
    with pytest.raises(ReleaseError, match='Resource "\\*"'):
        IAM.validate_executor_iam_documents(
            ROLE,
            [
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "ec2:DescribeInstances",
                            "Resource": "*",
                        }
                    ]
                }
            ],
        )
    with pytest.raises(ReleaseError, match="BatchReplaceClusterNodes"):
        IAM.validate_executor_iam_documents(
            ROLE,
            [
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "sagemaker:BatchReplaceClusterNodes",
                            "Resource": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp",
                        }
                    ]
                }
            ],
        )


# --- per-cluster runtime profile (mixed-provider site) --------------------------------

ROOT = Path(__file__).resolve().parents[2]
EKS_PROFILE = ROOT / "config/runtime-profile.regional-eks-safe.example.yaml"
EKS_DIGEST = "e" * 64


def _eks_cluster_with_profile(tmp_path: Path) -> dict[str, Any]:
    value = _eks_cluster(tmp_path)
    value["runtime_profile"] = {
        "source": str(EKS_PROFILE),
        "template_source": str(EKS_PROFILE),
        "version": "eks-v1",
        "agent_config_digest": EKS_DIGEST,
    }
    return value


class ProfileAwareRunner(RecordingRunner):
    """Answers the ingress exec probes the profile registration performs."""

    def __init__(self) -> None:
        super().__init__()
        self.registered: list[dict[str, Any]] = []
        self.inspected: list[str] = []
        self.live: dict[str, dict[str, Any]] = {}

    def run(self, args: list[str], **kwargs: Any) -> str:
        from gpu_fault_release.regional_release_probes import probe_source
        from tests.regional._release_orchestrator_support import ingress_pod_list_json

        super().run(args, **kwargs)
        if str(args[0]).endswith("deploy-node-installer-reconciler.sh"):
            # Dry-run preflight: no nodes were listed, so none are covered.
            return json.dumps({"node_count": 0, "status": "PASSED"})
        if "get" in args and "pod" in args:
            return ingress_pod_list_json()
        if "exec" in args:
            payload = json.loads(kwargs.get("input_text") or "{}")
            version = str(payload.get("profile_version"))
            if args[-1] == probe_source("runtime_profile_inspect"):
                self.inspected.append(version)
                desired = {**payload, "warnings": []}
                return json.dumps(
                    {"desired": desired, "existing": self.live.get(version)}
                )
            if args[-1] == probe_source("runtime_profile_register"):
                self.registered.append(payload)
                self.live[version] = {**payload, "warnings": []}
                return json.dumps(self.live[version])
        return ""


@pytest.fixture
def mixed_release(tmp_path: Path) -> MODULE.RegionalRelease:
    config = MODULE.ReleaseConfig.load(
        config_file(
            tmp_path,
            clusters=[_hyperpod_cluster(tmp_path), _eks_cluster_with_profile(tmp_path)],
        )
    )
    release = MODULE.RegionalRelease(config, ProfileAwareRunner())
    release.runner.dry_run = False  # type: ignore[attr-defined]
    return release


def test_every_profile_version_in_use_is_registered_once(mixed_release) -> None:
    runner: ProfileAwareRunner = mixed_release.runner  # type: ignore[assignment]
    from gpu_fault_release.regional_runtime_profile import ensure_runtime_profile

    ensure_runtime_profile(mixed_release)
    assert [item["profile_version"] for item in runner.registered] == [
        "hyperpod-v1",
        "eks-v1",
    ], "the site profile and the cluster's own profile must both be registered"
    eks = next(
        item for item in runner.registered if item["profile_version"] == "eks-v1"
    )
    assert eks["cluster_id"] == "gpu-eks-cluster", (
        "a cluster override is anchored on its own cluster id"
    )
    assert eks["environment"] == "eks", "the EKS example profile is the one registered"
    runner.registered.clear()
    ensure_runtime_profile(mixed_release)
    assert runner.registered == [], "registered profiles are not re-registered"


def test_each_gpu_target_renders_its_own_profile_version(mixed_release) -> None:
    hyperpod, eks = _targets(mixed_release)
    hyperpod_env = _env(
        _deployments(mixed_release, hyperpod)[
            "gpu-fault-kubernetes-node-resource-collector"
        ]
    )
    eks_env = _env(
        _deployments(mixed_release, eks)["gpu-fault-kubernetes-node-resource-collector"]
    )
    assert (
        hyperpod_env["GPU_FAULT_RUNTIME_PROFILE_VERSION"]["value"] == "hyperpod-v1"
    ), "the HyperPod cluster stays on the site profile version"
    assert eks_env["GPU_FAULT_RUNTIME_PROFILE_VERSION"]["value"] == "eks-v1", (
        "the plain EKS cluster renders its own profile version"
    )
    hyperpod_reconciler = _reconciler_environment(mixed_release, hyperpod)
    eks_reconciler = _reconciler_environment(mixed_release, eks)
    assert hyperpod_reconciler["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v1", "site pin"
    assert eks_reconciler["GPU_FAULT_RUNTIME_PROFILE"] == "eks-v1", "cluster pin"
    cpu = RENDERING.build_cpu_apply_environment(mixed_release, finalize=False)
    assert cpu["GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"] == "hyperpod-v1", (
        "the CPU pin keeps meaning the site default"
    )
    assert cpu["GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST"] == "a" * 64, (
        "the CPU Agent config pin keeps meaning the site default"
    )


def test_node_runtime_preflight_pins_the_cluster_profile(mixed_release) -> None:
    from gpu_fault_release import regional_release_node_runtime_rollout as NODE_RUNTIME

    _hyperpod, eks = _targets(mixed_release)
    runner: ProfileAwareRunner = mixed_release.runner  # type: ignore[assignment]
    runner.dry_run = True
    NODE_RUNTIME.preflight_node_runtime(
        mixed_release,
        eks,
        phase="join",
        wheel_cm="wheel-cm",
        bundle_cm="bundle-cm",
        artifact_sha="a" * 64,
        config_digest=mixed_release.config.cluster_agent_config_digest(eks),
    )
    environments = [
        kwargs["env"]
        for _args, kwargs in runner.calls
        if isinstance(kwargs.get("env"), dict)
        and "GPU_FAULT_RUNTIME_PROFILE" in kwargs["env"]
    ]
    assert environments, "the node installer preflight must render the reconciler env"
    assert {env["GPU_FAULT_RUNTIME_PROFILE"] for env in environments} == {"eks-v1"}, (
        "a node candidate without an explicit version takes its cluster's"
    )
    assert {env["GPU_FAULT_INSTALLER_CONFIG_DIGEST"] for env in environments} == {
        EKS_DIGEST
    }, "the cluster's own Agent config digest reaches the installer"


def test_release_state_records_cluster_profiles_and_the_diff_sees_them(
    mixed_release, release
) -> None:
    from gpu_fault_release import regional_release_diff as DIFF

    assert cluster_profiles_digest(release.config) is None, (
        "a site without overrides digests to None so older states compare equal"
    )
    assert cluster_profiles_digest(mixed_release.config), "overrides produce a digest"
    plain_state = {"runtime_profile_version": "hyperpod-v1"}
    assert DIFF.classify_release(release, plain_state).changed >= set(), "classifies"
    assert (
        "cluster_runtime_profiles"
        not in DIFF.classify_release(
            release, {**plain_state, "cluster_runtime_profiles_sha256": None}
        ).changed
    ), "no override on either side is not a change"
    assert (
        "cluster_runtime_profiles"
        in DIFF.classify_release(mixed_release, plain_state).changed
    ), "a new cluster override is a profile change for the release plan"
    plan = DIFF.build_execution_plan(
        DIFF.diff_from_changed({"cluster_runtime_profiles"})
    )
    assert plan.has(DIFF.ReleaseComponent.RUNTIME_PROFILE), "profile registration"
    assert plan.has(DIFF.ReleaseComponent.AGENT), "the data plane Agents roll"
    assert plan.has(DIFF.ReleaseComponent.RECONCILER), "the reconciler rolls"
