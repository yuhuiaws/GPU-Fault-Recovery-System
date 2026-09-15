"""Bootstrap's site contract and GPU-scope tests.

Split from ``test_admin_bootstrap.py``, which keeps the release build and reuse
side. The tests here share one subject: what bootstrap is allowed to touch and
what it must carry forward -- the solution security group, the dedicated public
subnets, the site identifier, the baseline GPU cluster scope, and the identity
and release contract an existing site pins.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_services as admin_bootstrap_services
from gpu_fault.admin import notification_bootstrap as admin_notification_bootstrap
from gpu_fault.admin.bootstrap import _ensure_security_group
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_site import (
    bind_initial_deploy_target,
    bootstrap_gpu_scope,
    discover_bootstrap_scope,
    existing_gpu_context,
    preserve_existing_site_contract,
    recover_verified_site_contract,
    unique_gpu_vpcs,
    validate_existing_cluster_identity,
)
from gpu_fault.admin.bootstrap_site import site_identifier as _site_identifier
from gpu_fault.admin.notifications import NotificationRouting
from tests.admin._bootstrap_support import _cluster


class _NoCommands(CommandRunner):
    """Every step is patched; any command reaching the runner is a leak."""

    def run(self, arguments, **_kwargs):
        raise AssertionError(f"bootstrap ran an unpatched command: {arguments}")


def test_solution_security_group_is_never_shared() -> None:
    class Runner:
        def aws_json(self, *_args, **_kwargs):
            return {
                "SecurityGroups": [
                    {
                        "GroupId": "sg-foreign",
                        "Tags": [{"Key": "Name", "Value": "foreign"}],
                    }
                ]
            }

    with pytest.raises(BootstrapError, match="refusing to share"):
        _ensure_security_group(
            Runner(),
            cluster=_cluster(),
            group_name="gpu-fault-site",
            description="test",
            site_id="site-a",
        )


def test_public_nlb_subnets_are_dedicated_even_if_public_subnets_exist(
    monkeypatch,
) -> None:
    class Runner:
        """Plays a VPC whose only public subnets belong to someone else, and
        answers the create calls with predictable ids."""

        def __init__(self) -> None:
            self.created_in: list[str] = []

        def run(self, _arguments, **_kwargs) -> str:
            return ""

        def aws_text(self, _region, service, operation, *_arguments, **_kwargs):
            if service == "ec2" and operation == "create-route-table":
                return f"rtb-public-{len(self.created_in)}"
            if service == "ec2" and operation == "associate-route-table":
                return f"rtbassoc-public-{len(self.created_in)}"
            raise AssertionError((service, operation))

        def aws_json(self, _region, service, operation, *arguments, **_kwargs):
            if service == "ec2" and operation == "create-subnet":
                zone = arguments[arguments.index("--availability-zone") + 1]
                self.created_in.append(zone)
                return {"Subnet": {"SubnetId": f"subnet-public-{len(self.created_in)}"}}
            if service == "ec2" and operation == "describe-subnets":
                if "--subnet-ids" in arguments:
                    return {
                        "Subnets": [
                            {
                                "SubnetId": "subnet-private-a",
                                "AvailabilityZone": "us-east-1a",
                            },
                            {
                                "SubnetId": "subnet-private-b",
                                "AvailabilityZone": "us-east-1b",
                            },
                        ]
                    }
                return {
                    "Subnets": [
                        {
                            "SubnetId": "subnet-shared-a",
                            "AvailabilityZone": "us-east-1a",
                            "CidrBlock": "10.0.0.0/28",
                            "Tags": [],
                        },
                        {
                            "SubnetId": "subnet-shared-b",
                            "AvailabilityZone": "us-east-1b",
                            "CidrBlock": "10.0.0.16/28",
                            "Tags": [],
                        },
                    ]
                }
            if service == "ec2" and operation == "describe-vpcs":
                return {"Vpcs": [{"CidrBlock": "10.0.0.0/24"}]}
            if service == "ec2" and operation == "describe-route-tables":
                # The VPC main table routes to the internet gateway, so the
                # shared subnets are public -- and still not ours.
                return {
                    "RouteTables": [
                        {
                            "RouteTableId": "rtb-main",
                            "Associations": [{"Main": True}],
                            "Routes": [
                                {
                                    "DestinationCidrBlock": "0.0.0.0/0",
                                    "GatewayId": "igw-external",
                                }
                            ],
                            "Tags": [],
                        }
                    ]
                }
            if service == "ec2" and operation == "describe-internet-gateways":
                return {
                    "InternetGateways": [
                        {"InternetGatewayId": "igw-external", "Tags": []}
                    ]
                }
            raise AssertionError((service, operation, arguments))

    runner = Runner()
    result = admin_bootstrap._ensure_public_subnets(
        runner, cluster=_cluster(), site_id="site-a"
    )

    assert result["public_subnets"] == ["subnet-public-1", "subnet-public-2"], (
        "external public subnets were reused"
    )
    assert runner.created_in == ["us-east-1a", "us-east-1b"], (
        "dedicated subnets were not created in the CPU EKS availability zones"
    )
    assert result["internet_gateway"]["ownership"] == "EXTERNAL", (
        "the VPC-level internet gateway should remain external"
    )


def test_site_identifier_is_independent_of_gpu_membership() -> None:
    cpu = _cluster()
    gpu_a = replace(
        cpu,
        role="gpu",
        hyperpod_arn=("arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a"),
        hyperpod_name="gpu-a",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        eks_name="gpu-a",
        context="gpu-a",
    )
    gpu_b = replace(
        gpu_a,
        hyperpod_arn=("arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-b"),
        hyperpod_name="gpu-b",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        eks_name="gpu-b",
        context="gpu-b",
    )

    assert _site_identifier(cpu, [gpu_a]) == _site_identifier(cpu, [gpu_a, gpu_b]), (
        "adding a GPU cluster changed the control-plane site ID"
    )


def test_new_bootstrap_mutates_only_the_baseline_gpu_cluster() -> None:
    cpu = _cluster()
    gpu_a = replace(
        cpu,
        role="gpu",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_name="gpu-a",
        vpc_id="vpc-a",
        context="gpu-a",
    )
    gpu_b = replace(
        gpu_a,
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        hyperpod_name="gpu-b",
        vpc_id="vpc-b",
        context="gpu-b",
    )
    gpu_c = replace(
        gpu_a,
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-c",
        hyperpod_name="gpu-c",
        vpc_id="vpc-b",
        context="gpu-c",
    )
    existing = {
        "spec": {
            "clusters": [
                {
                    "eksClusterArn": gpu_a.eks_arn,
                    "hyperpodClusterName": gpu_a.hyperpod_name,
                }
            ]
        }
    }

    assert bootstrap_gpu_scope(None, [gpu_a, gpu_b, gpu_c]) == [gpu_a]
    assert bootstrap_gpu_scope(existing, [gpu_a, gpu_b, gpu_c]) == [gpu_a]
    assert unique_gpu_vpcs(cpu, [gpu_a, gpu_b, gpu_c]) == [
        ("us-east-1", "vpc-a"),
        ("us-east-1", "vpc-b"),
    ]


def test_bootstrap_uses_the_baseline_scope_for_all_gpu_mutations(
    tmp_path: Path, monkeypatch
) -> None:
    """Only the baseline cluster is mutated; the rest are reported as pending.

    A first install commits one GPU cluster so a failure has one blast radius,
    and the extra ARNs come back as ``pending_gpu_cluster_arns`` for the join
    flow. Every mutating step therefore has to receive the baseline scope, while
    the site finalizer still sees the full discovered set so it can record what
    is still pending.
    """

    cpu = _cluster()
    gpu_a = replace(
        cpu,
        role="gpu",
        input_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_name="gpu-a",
        context="gpu-a",
    )
    gpu_b = replace(
        gpu_a,
        input_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-b",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        hyperpod_name="gpu-b",
        context="gpu-b",
    )
    scopes: dict[str, tuple[str, ...]] = {}
    namespaced: list[str | None] = []
    order: list[str] = []

    def record(name: str, result=None):
        def step(*_arguments, gpu_clusters=(), **_kwargs):
            scopes[name] = tuple(cluster.hyperpod_name for cluster in gpu_clusters)
            order.append(name)
            return result

        return step

    monkeypatch.setattr(
        admin_bootstrap, "validate_bootstrap_dependencies", lambda: None
    )
    monkeypatch.setattr(
        admin_bootstrap,
        "discover_bootstrap_scope",
        lambda **_kwargs: (None, cpu, [gpu_a, gpu_b]),
    )
    monkeypatch.setattr(admin_bootstrap, "_require_same_scope", lambda *_a, **_k: None)
    monkeypatch.setattr(
        admin_bootstrap, "bootstrap_gpu_scope", lambda *_arguments: [gpu_a]
    )
    # The release-independent task digests are bound to the baseline scope
    # before the release build is started and before any graph reads its
    # checkpoint (see ``bind_foundation_inputs``).
    monkeypatch.setattr(
        admin_bootstrap, "bind_foundation_inputs", record("foundation_bind")
    )
    monkeypatch.setattr(
        admin_bootstrap,
        "prepare_signed_release",
        record(
            "release",
            {
                "manifest": str(tmp_path / "release.json"),
                "images": {"adot": "adot:1", "runtime": "runtime:1"},
            },
        ),
    )
    monkeypatch.setattr(
        admin_notification_bootstrap,
        "notification_routing",
        lambda *_a, **_k: ("admin@example.com", {}),
    )
    kubeconfig_clusters: list[str] = []
    monkeypatch.setattr(
        admin_bootstrap,
        "_update_kubeconfig",
        lambda _runner, *, cluster, path: (
            kubeconfig_clusters.append(cluster.hyperpod_name)
            if cluster.role == "gpu"
            else None
        ),
    )
    monkeypatch.setattr(
        admin_bootstrap,
        "_ensure_namespace",
        lambda *_a, context=None, **_k: namespaced.append(context),
    )
    monkeypatch.setattr(
        admin_bootstrap,
        "_initial_secure_files",
        record("secure_files", ({}, tmp_path / "secure")),
    )
    monkeypatch.setattr(
        admin_bootstrap, "_ensure_base_secrets", lambda *_a, **_k: tmp_path / "master"
    )
    monkeypatch.setattr(
        admin_bootstrap_services, "_ensure_pod_identity_agent", lambda *_a: {}
    )
    monkeypatch.setattr(
        admin_bootstrap, "revalidate_pod_identity_agent", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        admin_bootstrap, "bootstrap_aurora_capacity", lambda _state_dir: None
    )
    # Both graph builders receive the baseline scope; the one run that follows
    # answers for both former phases.
    monkeypatch.setattr(admin_bootstrap, "foundation_task_graph", record("foundation"))
    monkeypatch.setattr(admin_bootstrap, "platform_task_graph", record("platform"))

    def run_tasks(**_keywords):
        order.append("graph")
        return {
            "executor_role:gpu-a": {"role_arn": "arn:aws:iam::1:role/gpu-a"},
            "aurora": {"cluster_id": "c"},
            "aurora_ready": {"master_secret_arn": "arn:new"},
            "monitoring_resources": {},
            "nlb_network": {},
            "pki": {},
        }

    monkeypatch.setattr(admin_bootstrap, "run_bootstrap_tasks", run_tasks)
    documents: dict[str, object] = {}

    def site_document(*_arguments, gpu_clusters=(), aurora=None, **_kwargs):
        scopes["site_document"] = tuple(
            cluster.hyperpod_name for cluster in gpu_clusters
        )
        documents["aurora"] = aurora
        return {}

    monkeypatch.setattr(admin_bootstrap, "_site_document", site_document)
    monkeypatch.setattr(
        admin_bootstrap,
        "finalize_bootstrap_site",
        lambda _file, _generated, _existing, discovered, _state, _write: scopes.update(
            finalize=tuple(cluster.hyperpod_name for cluster in discovered)
        ),
    )

    admin_bootstrap.bootstrap_from_arns(
        admin_bootstrap.BootstrapRequest(
            cpu_cluster_arn=cpu.input_arn,
            gpu_cluster_arns=(gpu_a.input_arn, gpu_b.input_arn),
            repository_root=tmp_path / "repo",
            state_dir=tmp_path / "state",
        ),
        runner=_NoCommands(),
    )
    scopes["kubeconfigs"] = tuple(kubeconfig_clusters)

    # The readiness task's master Secret ARN is merged over the foundation's
    # ``aurora`` record before the site document is built.
    assert documents["aurora"] == {"cluster_id": "c", "master_secret_arn": "arn:new"}
    baseline = (gpu_a.hyperpod_name,)
    assert scopes == {
        "foundation_bind": baseline,
        "release": baseline,
        "kubeconfigs": baseline,
        "secure_files": baseline,
        "foundation": baseline,
        "platform": baseline,
        "site_document": baseline,
        "finalize": (gpu_a.hyperpod_name, gpu_b.hyperpod_name),
    }
    # The release build is submitted only after the bind returned, so the order
    # below is decided by the call sequence, not by thread timing.
    assert order.index("foundation_bind") < order.index("release"), (
        "the release build started before the foundation digests were bound"
    )
    assert order.index("foundation_bind") < order.index("graph"), (
        "the graph read its checkpoint before the foundation digests were bound"
    )
    # The CPU and GPU namespace chains run on their own threads, so only the
    # set is deterministic.
    assert sorted(namespaced, key=str) == sorted([None, gpu_a.context], key=str), (
        "a namespace was created outside the baseline GPU scope"
    )


def test_existing_site_preserves_release_profile_and_cluster_context() -> None:
    generated = {
        "spec": {
            "release": {"manifest": "/repo/dist/current-release.json"},
            "runtimeProfile": {
                "source": "/repo/config/profile.yaml",
                "version": "hyperpod-v1",
            },
            "clusters": [
                {
                    "clusterId": "gpu-a",
                    "context": "new-context",
                    "allowedNamespaces": ["training"],
                }
            ],
        }
    }
    existing = {
        "spec": {
            "release": {"manifest": "dist/current-release.json"},
            "runtimeProfile": {
                "source": "/secure/profiles/hyperpod-v1.yaml",
                "templateSource": "/repo/config/profile.yaml",
                "version": "hyperpod-v1",
                "registrationClusterId": "gpu-a",
            },
            "clusters": [
                {
                    "clusterId": "gpu-a",
                    "context": "stable-context",
                    "allowedNamespaces": ["gpu-fault-system", "training"],
                }
            ],
        }
    }

    result = preserve_existing_site_contract(generated, existing)

    assert result["spec"]["release"] == existing["spec"]["release"]
    assert result["spec"]["runtimeProfile"] == existing["spec"]["runtimeProfile"]
    assert result["spec"]["clusters"][0]["context"] == "stable-context"
    assert result["spec"]["clusters"][0]["allowedNamespaces"] == [
        "gpu-fault-system",
        "training",
    ]


def _render_site(
    tmp_path: Path,
    *,
    gpu_clusters: list[ClusterIdentity],
    roles: dict[str, dict[str, str]],
    token_files: dict[str, Path],
) -> dict[str, Any]:
    """The one call site for the private site renderer the tests below share."""

    return admin_bootstrap._site_document(
        request=BootstrapRequest(
            cpu_cluster_arn=_cluster().input_arn,
            gpu_cluster_arns=tuple(cluster.input_arn for cluster in gpu_clusters),
            repository_root=tmp_path / "repo",
            state_dir=tmp_path / "state",
        ),
        site_id="site-a",
        cpu=_cluster(),
        gpu_clusters=gpu_clusters,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        release={
            "manifest": str(tmp_path / "release.json"),
            "agent_config_digest": "a" * 64,
            "images": {
                "runtime": "r:1",
                "node_installer": "n:1",
                "dcgm_exporter": "d:1",
                "adot": "adot:1",
            },
        },
        nlb={"name": "nlb", "public_subnets": ["subnet-1"], "security_group": "sg"},
        pki={
            "hostname": "gpu-fault.example.com",
            "ca_file": str(tmp_path / "ca.pem"),
            "certificate_arn": "arn:aws:acm:us-east-1:123456789012:certificate/x",
            "hosted_zone_id": "Z1",
        },
        aurora={"cluster_id": "aurora"},
        monitoring={"workspace_id": "ws-1", "sns_topic_arn": "arn:aws:sns:x"},
        roles=roles,
        token_files=token_files,
        fleet_master_file=tmp_path / "fleet-master",
        adot_image="adot:1",
        admin_email="admin@example.com",
        routing=NotificationRouting(
            sender="admin@example.com",
            recipients=("admin@example.com",),
            subject_prefix="[gpu-fault]",
        ),
        grafana_health={},
    )


def test_the_generated_site_names_the_created_adot_writer_role(tmp_path: Path) -> None:
    """A fresh site with a workspace gets the data-plane collector with no hand step.

    The role the ``adot_writer_role:<cluster>`` task created lands on the
    cluster entry as ``adotIrsaRoleArn``; a cluster the task produced no role
    for keeps the key absent, which is the release's "skip the collector" signal.
    """

    gpu_a = replace(_cluster(), role="gpu", hyperpod_name="gpu-a", eks_name="gpu-a")
    gpu_b = replace(_cluster(), role="gpu", hyperpod_name="gpu-b", eks_name="gpu-b")
    document = _render_site(
        tmp_path,
        gpu_clusters=[gpu_a, gpu_b],
        roles={
            "executor_role:gpu-a": {
                "role_arn": "arn:aws:iam::123456789012:role/gpu-a-executor"
            },
            "executor_role:gpu-b": {
                "role_arn": "arn:aws:iam::123456789012:role/gpu-b-executor"
            },
            "adot_writer_role:gpu-a": {
                "role_arn": "arn:aws:iam::123456789012:role/gpu-a-adot-writer"
            },
            "aurora": {"cluster_id": "aurora"},
        },
        token_files={"gpu-a": tmp_path / "a.token", "gpu-b": tmp_path / "b.token"},
    )
    clusters = {item["clusterId"]: item for item in document["spec"]["clusters"]}
    assert clusters["gpu-a"]["adotIrsaRoleArn"] == (
        "arn:aws:iam::123456789012:role/gpu-a-adot-writer"
    ), "the created ADOT writer role did not reach the generated site"
    assert "adotIrsaRoleArn" not in clusters["gpu-b"], (
        "a cluster without a writer role was given a placeholder ARN"
    )
    assert clusters["gpu-a"]["executorIrsaRoleArn"].endswith("gpu-a-executor"), (
        "the executor role ARN must still be written next to the writer ARN"
    )


def test_existing_site_keeps_the_operator_owned_adot_role() -> None:
    """An explicit ``adotIrsaRoleArn`` in site.yaml wins over the created role.

    Operators who made the role by hand before bootstrap created one keep
    theirs; a site that never had the key takes the generated one.
    """

    generated = {
        "spec": {
            "clusters": [
                {"clusterId": "gpu-a", "adotIrsaRoleArn": "arn:created:a"},
                {"clusterId": "gpu-b", "adotIrsaRoleArn": "arn:created:b"},
            ]
        }
    }
    existing = {
        "spec": {
            "clusters": [
                {"clusterId": "gpu-a", "adotIrsaRoleArn": "arn:operator:a"},
                {"clusterId": "gpu-b"},
            ]
        }
    }

    result = preserve_existing_site_contract(deepcopy(generated), existing)

    clusters = {item["clusterId"]: item for item in result["spec"]["clusters"]}
    assert clusters["gpu-a"]["adotIrsaRoleArn"] == "arn:operator:a", (
        "the operator's ADOT writer role was overwritten by the created one"
    )
    assert clusters["gpu-b"]["adotIrsaRoleArn"] == "arn:created:b", (
        "a cluster that never declared a role lost the created one"
    )


def test_existing_site_keeps_the_declared_rollback_policy() -> None:
    # Fail-forward is how the release engine wants non-transactional changes
    # (ADOT manifest, ADOT image, endpoint, cluster registry) deployed, and the
    # only way to declare it is ``spec.autoRollback: false``. Re-running deploy
    # re-renders the site, so a regenerated ``true`` would revert the decision
    # under the very rollout that depends on it.
    generated = {"spec": {"autoRollback": True, "clusters": []}}

    declared_off = preserve_existing_site_contract(
        deepcopy(generated), {"spec": {"autoRollback": False, "clusters": []}}
    )
    assert declared_off["spec"]["autoRollback"] is False

    declared_on = preserve_existing_site_contract(
        deepcopy(generated), {"spec": {"autoRollback": True, "clusters": []}}
    )
    assert declared_on["spec"]["autoRollback"] is True

    # Neither a missing nor a malformed declaration may turn into "false":
    # losing automatic rollback has to be something an operator wrote down.
    for existing in ({"spec": {"clusters": []}}, {"spec": {"autoRollback": "false"}}):
        fell_back = preserve_existing_site_contract(deepcopy(generated), existing)
        assert fell_back["spec"]["autoRollback"] is True, existing


def test_existing_site_recovers_latest_verified_release_contract(
    tmp_path: Path,
) -> None:
    existing = {
        "kind": "RegionalSite",
        "metadata": {"name": "test-site"},
        "spec": {
            "repositoryRoot": "/repo/current",
            "awsRegion": "us-west-2",
            "cpu": {"eksArn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu"},
            "release": {"manifest": "/repo/current/dist/current-release.json"},
            "runtimeProfile": {
                "source": "/repo/current/config/profile.yaml",
                "templateSource": "/repo/current/config/profile.yaml",
                "version": "profile-v1",
            },
            "clusters": [
                {
                    "clusterId": "gpu-a",
                    "eksClusterArn": (
                        "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
                    ),
                    "context": "current-context",
                    "agentEndpointAllowedCidrs": ["192.0.2.0/24"],
                }
            ],
        },
    }
    verified = {
        "kind": "RegionalSite",
        "metadata": {"name": "test-site"},
        "spec": {
            "repositoryRoot": "/repo/previous",
            "awsRegion": "us-west-2",
            "cpu": {"eksArn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu"},
            "release": {"manifest": "dist/current-release.json"},
            "runtimeProfile": {
                "source": "/secure/profiles/profile-v1.yaml",
                "templateSource": "/repo/previous/config/profile.yaml",
                "version": "profile-v1",
            },
            "clusters": [
                {
                    "clusterId": "gpu-a",
                    "eksClusterArn": (
                        "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
                    ),
                    "context": "verified-context",
                    "agentEndpointAllowedCidrs": ["198.51.100.0/24"],
                }
            ],
        },
    }
    release = tmp_path / "release-deploy/release-a"
    release.mkdir(parents=True)
    (release / "state.json").write_text(
        json.dumps({"phase": "COMPLETED", "verification": {"status": "PASSED"}}),
        encoding="utf-8",
    )
    (release / "site.candidate.yaml").write_text(json.dumps(verified), encoding="utf-8")

    recovered = recover_verified_site_contract(tmp_path, existing)

    assert recovered is not None
    assert recovered["spec"]["repositoryRoot"] == "/repo/current"
    assert recovered["spec"]["release"] == verified["spec"]["release"]
    assert recovered["spec"]["runtimeProfile"] == verified["spec"]["runtimeProfile"]
    assert recovered["spec"]["clusters"][0]["context"] == "verified-context"
    assert recovered["spec"]["clusters"][0]["agentEndpointAllowedCidrs"] == [
        "198.51.100.0/24"
    ]


def test_existing_gpu_context_matches_discovered_identity() -> None:
    cluster = replace(
        _cluster(),
        role="gpu",
        hyperpod_name="hp-gpu-a",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
    )
    site = {
        "spec": {
            "clusters": [
                {
                    "clusterId": "hp-gpu-a",
                    "context": "stable-context",
                    "hyperpodClusterName": "hp-gpu-a",
                    "eksClusterArn": cluster.eks_arn,
                }
            ]
        }
    }

    assert existing_gpu_context(site, cluster) == "stable-context"


def test_existing_site_rejects_cluster_identity_changes() -> None:
    cpu = _cluster()
    gpu = replace(
        _cluster(),
        role="gpu",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_name="gpu-a",
    )
    existing = {
        "spec": {
            "cpu": {"eksArn": cpu.eks_arn, "hyperpodClusterName": cpu.hyperpod_name},
            "clusters": [
                {"eksClusterArn": gpu.eks_arn, "hyperpodClusterName": gpu.hyperpod_name}
            ],
        }
    }

    gpu_b = replace(
        gpu,
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        hyperpod_name="gpu-b",
    )

    assert (
        validate_existing_cluster_identity(existing, cpu=cpu, gpu_clusters=[gpu]) == []
    )
    # A strict superset passes and yields the delta the deploy joins afterwards.
    assert validate_existing_cluster_identity(
        existing, cpu=cpu, gpu_clusters=[gpu, gpu_b]
    ) == [gpu_b]

    # A set that omits a managed cluster is a subset, however many new ones it
    # adds: clusters leave a site only through remove-cluster.
    with pytest.raises(BootstrapError, match="remove-cluster"):
        validate_existing_cluster_identity(existing, cpu=cpu, gpu_clusters=[gpu_b])
    with pytest.raises(BootstrapError, match="remove-cluster"):
        validate_existing_cluster_identity(existing, cpu=cpu, gpu_clusters=[])


def test_existing_site_rejects_a_different_cpu_cluster() -> None:
    cpu = _cluster()
    gpu = replace(
        _cluster(),
        role="gpu",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_name="gpu-a",
    )
    existing = {
        "spec": {
            "cpu": {"eksArn": cpu.eks_arn, "hyperpodClusterName": cpu.hyperpod_name},
            "clusters": [
                {"eksClusterArn": gpu.eks_arn, "hyperpodClusterName": gpu.hyperpod_name}
            ],
        }
    }
    other_cpu = replace(
        cpu,
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/control-2",
        hyperpod_name="control-2",
    )

    with pytest.raises(BootstrapError, match="CPU cluster identity differs"):
        validate_existing_cluster_identity(existing, cpu=other_cpu, gpu_clusters=[gpu])


def test_initial_deploy_target_accepts_monotonic_membership_subset(
    tmp_path: Path,
) -> None:
    cpu = _cluster()
    gpu_a = replace(
        cpu,
        input_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        role="gpu",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_name="gpu-a",
    )
    gpu_b = replace(
        gpu_a,
        input_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-b",
        hyperpod_name="gpu-b",
    )
    site = {
        "spec": {
            "cpu": {"eksArn": cpu.eks_arn, "hyperpodClusterName": cpu.hyperpod_name},
            "clusters": [
                {
                    "eksClusterArn": gpu_a.eks_arn,
                    "hyperpodClusterName": gpu_a.hyperpod_name,
                }
            ],
        }
    }
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="test")

    bind_initial_deploy_target(state, site, cpu, [gpu_a, gpu_b])
    reloaded = BootstrapState(state.path, site_id="test")

    assert reloaded.value["resources"]["initial_deploy_target"]["status"] == "PENDING"
    assert bootstrap_gpu_scope(site, [gpu_a, gpu_b]) == [gpu_a]

    with pytest.raises(BootstrapError, match="persisted checkpoint"):
        bind_initial_deploy_target(reloaded, site, cpu, [gpu_a])


def _gpu_identity(cpu: ClusterIdentity, name: str) -> ClusterIdentity:
    return replace(
        cpu,
        input_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{name}",
        role="gpu",
        eks_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{name}",
        hyperpod_arn=f"arn:aws:sagemaker:us-east-1:123456789012:cluster/{name}",
        hyperpod_name=name,
    )


def _site_with(cpu: ClusterIdentity, *gpu_clusters: ClusterIdentity) -> dict[str, Any]:
    return {
        "spec": {
            "cpu": {"eksArn": cpu.eks_arn, "hyperpodClusterName": cpu.hyperpod_name},
            "clusters": [
                {
                    "eksClusterArn": item.eks_arn,
                    "hyperpodClusterName": item.hyperpod_name,
                }
                for item in gpu_clusters
            ],
        }
    }


def test_initial_deploy_target_settles_once_the_site_manages_it(tmp_path: Path) -> None:
    """The first deploy's checkpoint stops binding once the site holds its target.

    It used to stay PENDING until a later deploy requested exactly the birth
    set, so ``deploy --gpu-cluster-arn A --gpu-cluster-arn NEW`` right after
    the first deploy of A was refused against the checkpoint instead of joining
    NEW. Before the site document exists the checkpoint still pins a resume.
    """

    cpu = _cluster()
    gpu_a = _gpu_identity(cpu, "gpu-a")
    gpu_b = _gpu_identity(cpu, "gpu-b")
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="test")

    assert bind_initial_deploy_target(state, None, cpu, [gpu_a]) == [gpu_a]
    assert state.value["resources"]["initial_deploy_target"]["status"] == "PENDING"
    with pytest.raises(BootstrapError, match="persisted checkpoint"):
        bind_initial_deploy_target(
            BootstrapState(state.path, site_id="test"), None, cpu, [gpu_b]
        )

    # The site now manages gpu-a: adding gpu-b is the superset the deploy joins
    # after the release, not a departure from the checkpoint.
    grown = BootstrapState(state.path, site_id="test")
    assert bind_initial_deploy_target(
        grown, _site_with(cpu, gpu_a), cpu, [gpu_a, gpu_b]
    ) == [gpu_a]
    target = grown.value["resources"]["initial_deploy_target"]
    assert target["status"] == "PENDING", target
    assert [item["hyperpod_name"] for item in target["gpu_clusters"]] == [
        "gpu-a",
        "gpu-b",
    ]

    settled = BootstrapState(state.path, site_id="test")
    assert bind_initial_deploy_target(
        settled, _site_with(cpu, gpu_a, gpu_b), cpu, [gpu_a, gpu_b]
    ) == [gpu_a, gpu_b]
    assert settled.value["resources"]["initial_deploy_target"]["status"] == "COMPLETE"


def test_a_cluster_detached_by_remove_cluster_releases_the_initial_target(
    tmp_path: Path,
) -> None:
    """A CPU-only upgrade after remove-cluster is judged by the site document.

    Live 2026-09-15 (BOOT-029 stage 3 rerun): the site had removed its only GPU
    cluster, ``deploy --state-dir`` read ``clusters: []`` and was refused with
    "initial deploy target differs from the persisted checkpoint" because the
    PENDING checkpoint still named the removed cluster. remove-cluster records
    the detachment under ``removed_clusters``; that record releases the
    commitment, the checkpoint settles and the empty site deploys.
    """

    cpu = _cluster()
    gpu_a = _gpu_identity(cpu, "gpu-a")
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="test")
    bind_initial_deploy_target(state, None, cpu, [gpu_a])
    document = json.loads(state.path.read_text(encoding="utf-8"))
    document["removed_clusters"] = {
        "gpu-a": {"removed_at": "2026-09-15T04:21:38+00:00", "vpc_id": "vpc-a"}
    }
    state.path.write_text(json.dumps(document), encoding="utf-8")

    reloaded = BootstrapState(state.path, site_id="test")
    assert bind_initial_deploy_target(reloaded, _site_with(cpu), cpu, []) == []
    target = reloaded.value["resources"]["initial_deploy_target"]
    assert target["status"] == "COMPLETE", target
    assert target["gpu_clusters"] == []


def test_a_site_without_gpu_clusters_still_discovers_its_control_plane(
    tmp_path: Path,
) -> None:
    """After remove-cluster the site deploys with ``clusters: []``; the GPU
    discovery fan-out then has nothing to run, and a zero-worker pool is a
    ValueError (deploy #61, 2026-09-12: "max_workers must be greater than 0")."""

    cpu = _cluster()

    existing, discovered_cpu, gpu_clusters = discover_bootstrap_scope(
        request=BootstrapRequest(
            cpu_cluster_arn=cpu.input_arn,
            gpu_cluster_arns=(),
            repository_root=tmp_path / "repo",
            state_dir=tmp_path / "state",
        ),
        runner=object(),
        discover=lambda _runner, **_keywords: cpu,
        alias=lambda _arn, role, index: f"{role}-{index}",
    )

    assert existing is None, "a fresh state dir has no site to recover"
    assert discovered_cpu.role == "cpu"
    assert gpu_clusters == [], "no GPU ARN means no GPU cluster, not an error"


def test_a_site_left_without_gpu_clusters_still_renders(tmp_path: Path) -> None:
    """Deploy #62 (2026-09-12) died on ``cluster_documents[0]`` for the site
    remove-cluster had left with ``clusters: []``. The managed site keeps its
    own runtimeProfile through ``preserve_existing_site_contract``; the
    generated placeholder only has to exist."""

    document = _render_site(tmp_path, gpu_clusters=[], roles={}, token_files={})
    assert document["spec"]["clusters"] == [], "no GPU cluster means no entry"
    assert document["spec"]["runtimeProfile"]["registrationClusterId"], (
        "the placeholder registration id must be non-empty so the site loads"
    )


def test_existing_site_keeps_its_profile_but_takes_the_current_template_path() -> None:
    """Live 2026-09-12: eight resumes moved repositoryRoot to new snapshots while
    runtimeProfile.templateSource kept naming the first run's snapshot; once the
    pruning removed that tree, `status` refused with "template_source must be an
    existing file". The template is code and follows the tree being deployed."""

    generated = {
        "spec": {
            "repositoryRoot": "/snapshots/new",
            "runtimeProfile": {
                "source": "/snapshots/new/config/profile.yaml",
                "templateSource": "/snapshots/new/config/profile.yaml",
                "version": "hyperpod-v1",
                "registrationClusterId": "gpu-a",
            },
        }
    }
    existing = {
        "spec": {
            "repositoryRoot": "/snapshots/old",
            "runtimeProfile": {
                "source": "/secure/profiles/hyperpod-v1.yaml",
                "templateSource": "/snapshots/old/config/profile.yaml",
                "version": "hyperpod-v1",
                "registrationClusterId": "gpu-a",
            },
        }
    }

    result = preserve_existing_site_contract(generated, existing)

    profile = result["spec"]["runtimeProfile"]
    assert profile["source"] == "/secure/profiles/hyperpod-v1.yaml", (
        "the operator's rendered profile is preserved"
    )
    assert profile["templateSource"] == "/snapshots/new/config/profile.yaml", (
        "the template path follows the tree being deployed"
    )
