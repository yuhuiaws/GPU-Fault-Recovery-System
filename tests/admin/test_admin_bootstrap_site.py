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

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_services as admin_bootstrap_services
from gpu_fault.admin import notification_bootstrap as admin_notification_bootstrap
from gpu_fault.admin.bootstrap import _ensure_security_group
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_site import (
    bind_initial_deploy_target,
    bootstrap_gpu_scope,
    existing_gpu_context,
    preserve_existing_site_contract,
    recover_verified_site_contract,
    unique_gpu_vpcs,
    validate_existing_cluster_identity,
)
from gpu_fault.admin.bootstrap_site import site_identifier as _site_identifier
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

    def record(name: str, result=None):
        def step(*_arguments, gpu_clusters=(), **_kwargs):
            scopes[name] = tuple(cluster.hyperpod_name for cluster in gpu_clusters)
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
    monkeypatch.setattr(
        admin_bootstrap,
        "run_bootstrap_tasks",
        lambda **_keywords: {
            "executor_role:gpu-a": {"role_arn": "arn:aws:iam::1:role/gpu-a"},
            "aurora": {"cluster_id": "c"},
            "aurora_ready": {"master_secret_arn": "arn:new"},
            "monitoring_resources": {},
            "nlb_network": {},
            "pki": {},
        },
    )
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
        "release": baseline,
        "kubeconfigs": baseline,
        "secure_files": baseline,
        "foundation": baseline,
        "platform": baseline,
        "site_document": baseline,
        "finalize": (gpu_a.hyperpod_name, gpu_b.hyperpod_name),
    }
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
