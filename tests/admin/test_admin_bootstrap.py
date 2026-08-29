from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

from gpu_fault import admin_bootstrap, admin_release_artifacts
from gpu_fault.admin_bootstrap import (
    _build_release,
    _ensure_security_group,
    _site_identifier,
    _unused_subnet_cidrs,
)
from gpu_fault.admin_bootstrap_common import (
    Arn,
    BootstrapError,
    BootstrapState,
    ClusterIdentity,
    run_parallel,
)
from gpu_fault.admin_bootstrap_site import (
    existing_gpu_context,
    preserve_existing_site_contract,
)


def test_cluster_arn_parser_accepts_eks_and_hyperpod() -> None:
    eks = Arn.parse("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a")
    hyperpod = Arn.parse("arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a")

    assert eks.service == "eks"
    assert eks.resource_name == "gpu-a"
    assert hyperpod.service == "sagemaker"
    assert hyperpod.resource_name == "gpu-a"


def test_hyperpod_discovery_uses_the_complete_input_arn(monkeypatch) -> None:
    hyperpod_arn = "arn:aws:sagemaker:us-east-1:123456789012:cluster/internal-id"
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    class Runner:
        def aws_json(self, region, service, operation, *arguments, **_kwargs):
            calls.append((service, operation, arguments))
            if service == "sagemaker":
                return {
                    "ClusterArn": hyperpod_arn,
                    "ClusterName": "gpu-a",
                    "NodeRecovery": "None",
                    "Orchestrator": {
                        "Eks": {
                            "ClusterArn": (
                                "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
                            )
                        }
                    },
                }
            return {
                "cluster": {
                    "status": "ACTIVE",
                    "resourcesVpcConfig": {"vpcId": "vpc-a", "subnetIds": ["subnet-a"]},
                }
            }

    monkeypatch.setattr(
        admin_bootstrap,
        "discover_subnet_cidrs",
        lambda *_args, **_kwargs: ("10.0.0.0/24",),
    )

    discovered = admin_bootstrap.discover_cluster(
        Runner(), cluster_arn=hyperpod_arn, role="gpu", context="gpu-a"
    )

    assert discovered.hyperpod_name == "gpu-a"
    assert calls[0] == (
        "sagemaker",
        "describe-cluster",
        ("--cluster-name", hyperpod_arn),
    )


def test_public_subnet_allocator_avoids_existing_ranges() -> None:
    selected = _unused_subnet_cidrs(
        ["10.0.0.0/24"], ["10.0.0.0/28", "10.0.0.16/28"], count=2
    )

    assert selected == ["10.0.0.32/28", "10.0.0.48/28"]


def test_independent_bootstrap_tasks_run_in_parallel(tmp_path) -> None:
    state = BootstrapState(tmp_path / "state.json", site_id="test")

    def task(value: str) -> str:
        time.sleep(0.1)
        return value

    started = time.monotonic()
    result = run_parallel(
        {"a": lambda: task("a"), "b": lambda: task("b"), "c": lambda: task("c")},
        state=state,
    )
    elapsed = time.monotonic() - started

    assert result == {"a": "a", "b": "b", "c": "c"}
    assert elapsed < 0.25


def test_parallel_bootstrap_persists_successes_when_another_task_fails(
    tmp_path,
) -> None:
    state = BootstrapState(tmp_path / "state.json", site_id="test")

    def fail():
        raise BootstrapError("aurora failed")

    with pytest.raises(BootstrapError, match="aurora failed"):
        run_parallel(
            {"aurora": fail, "pki": lambda: {"certificate_arn": "arn:certificate"}},
            state=state,
        )

    reloaded = BootstrapState(tmp_path / "state.json", site_id="test")
    assert reloaded.value["completed_tasks"] == ["pki"]
    assert reloaded.value["resources"]["pki"] == {"certificate_arn": "arn:certificate"}


def test_legacy_foundation_consumes_prebuilt_release(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "dist/current-release.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "deployable": True,
                "release_id": "release-a",
                "delivery": {
                    "images": {
                        name: {
                            "reference": f"registry.example/{name}@sha256:"
                            + character * 64
                        }
                        for name, character in {
                            "runtime": "1",
                            "node_installer": "2",
                            "dcgm_exporter": "3",
                            "adot": "4",
                        }.items()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "compute_agent_config_digest",
        lambda *_args, **_kwargs: "a" * 64,
    )

    result = _build_release(
        type("Runner", (), {"dry_run": False})(),
        repository_root=tmp_path,
        runtime_profile="profile-a",
    )

    assert result["manifest"] == str(manifest)
    assert result["agent_config_digest"] == "a" * 64
    assert result["images"]["runtime"].endswith("1" * 64), (
        "legacy bootstrap did not preserve the signed runtime image digest"
    )


def test_legacy_foundation_rejects_source_only_release(tmp_path) -> None:
    manifest = tmp_path / "dist/current-release.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {"schema_version": 3, "deployable": False, "release_id": "release-a"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(BootstrapError, match="deployable schema v3"):
        _build_release(
            type("Runner", (), {"dry_run": False})(),
            repository_root=tmp_path,
            runtime_profile="profile-a",
        )


def test_legacy_bootstrap_state_revalidates_exclusive_resources(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "site_id": "test",
                "phase": "site-ready",
                "resources": {"nlb_network": {"public_subnets": ["subnet-old"]}},
                "completed_tasks": ["nlb_network"],
            }
        ),
        encoding="utf-8",
    )

    state = BootstrapState(path, site_id="test")

    assert state.value["schema_version"] == 2, "legacy state was not upgraded"
    assert state.value["completed_tasks"] == [], "ownership tasks were not invalidated"


def _cluster() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        role="cpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn=("arn:aws:sagemaker:us-east-1:123456789012:cluster/control"),
        hyperpod_name="control",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        eks_name="control",
        vpc_id="vpc-control",
        subnet_ids=("subnet-private-a", "subnet-private-b"),
        node_recovery="None",
        context="control",
    )


def test_solution_security_group_is_never_shared() -> None:
    class Runner:
        dry_run = False

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
        dry_run = True

        def aws_json(self, _region, service, operation, *arguments, **_kwargs):
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
            if service == "ec2" and operation == "describe-internet-gateways":
                return {
                    "InternetGateways": [
                        {"InternetGatewayId": "igw-external", "Tags": []}
                    ]
                }
            raise AssertionError((service, operation, arguments))

    monkeypatch.setattr(
        admin_bootstrap, "_public_subnet", lambda *_args, **_kwargs: True
    )
    result = admin_bootstrap._ensure_public_subnets(
        Runner(), cluster=_cluster(), site_id="site-a"
    )

    assert result["public_subnets"] == [
        "subnet-dryrun-public-1",
        "subnet-dryrun-public-2",
    ], "external public subnets were reused"
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
