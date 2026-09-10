"""What a deploy does before the resource pools start, and how little it costs.

The serial preamble of ``bootstrap_from_arns`` -- tool checks, discovery,
kubeconfigs, namespaces, the base Secret, the Pod Identity add-on -- used to be
the slowest stretch of a re-deploy that changed nothing. These tests pin the
shape that replaced it: independent chains run together, a persisted hint spares
the region inventory, one route-table read serves a VPC, an unchanged subnet
group is left alone, and every "is it there?" question is answered by the same
read that fetches the resource, with only a not-found code meaning absent.
"""

from __future__ import annotations

import base64
import json
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_dependencies as admin_bootstrap_dependencies
from gpu_fault.admin.bootstrap import (
    RDS_CA_BUNDLE_ENVIRONMENT,
    _aurora_ready,
    _ensure_aurora,
)
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    describe_or_absent,
)
from gpu_fault.admin.config import AuroraCapacityConfig
from tests.admin._bootstrap_support import _cluster

# --- the first-deploy preamble ----------------------------------------------------


class _ClusterAccess:
    """A runner that records which thread ran each command and takes a beat
    per command, so overlap is measurable and ordering is checkable."""

    def __init__(self, *, pause: float = 0.05) -> None:
        self.pause = pause
        self.calls: list[tuple[str, int]] = []
        self.lock = threading.Lock()

    def run(self, arguments, **_kwargs) -> str:
        label = " ".join(str(item) for item in arguments)
        with self.lock:
            self.calls.append((label, threading.get_ident()))
        time.sleep(self.pause)
        if "jsonpath={.data.node-action-secret}" in label:
            return base64.b64encode(b"master").decode()
        return ""

    def aws_json(self, region: str, *arguments: str, **kwargs) -> dict:
        self.run(["aws", *arguments, "--region", region], **kwargs)
        return {}

    def labels(self, fragment: str) -> list[str]:
        return [label for label, _thread in self.calls if fragment in label]

    def thread_of(self, fragment: str) -> int:
        return next(thread for label, thread in self.calls if fragment in label)


def _gpu(name: str):
    return replace(
        _cluster(),
        role="gpu",
        hyperpod_name=name,
        eks_name=name,
        eks_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{name}",
        hyperpod_arn=f"arn:aws:sagemaker:us-east-1:123456789012:cluster/{name}",
        context=name,
    )


def test_cluster_access_runs_its_three_chains_concurrently_and_in_order(
    tmp_path: Path,
) -> None:
    """The CPU kubeconfig/namespace/Secret chain, the GPU kubeconfig/namespace
    chain and the Pod Identity revalidation share nothing but the wait; each
    chain still runs in its own order and the add-on still lands in the
    checkpoint the way `revalidate_pod_identity_agent` always recorded it."""

    runner = _ClusterAccess()
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    ensured: list[str] = []

    def ensure_pod_identity_agent(active_runner, cluster, site_id) -> dict:
        active_runner.run(["aws", "eks", "describe-addon", cluster.eks_name])
        ensured.append(site_id)
        return {"addon": "eks-pod-identity-agent", "ownership": "CREATED"}

    started = time.monotonic()
    cpu_kubeconfig, gpu_kubeconfig, master_file = (
        admin_bootstrap.prepare_cluster_access(
            runner,
            state=state,
            cpu=_cluster(),
            gpu_clusters=[_gpu("gpu-a"), _gpu("gpu-b")],
            state_dir=tmp_path,
            namespace="gpu-fault-system",
            secure_dir=tmp_path / "secure",
            site_id="site-a",
            ensure_pod_identity_agent=ensure_pod_identity_agent,
        )
    )
    elapsed = time.monotonic() - started

    # Ten commands at 50 ms each would be 0.5 s in series; three chains overlap.
    serial = len(runner.calls) * runner.pause
    assert elapsed < serial * 0.75, f"{elapsed:.2f}s for {len(runner.calls)} commands"
    threads = {
        runner.thread_of("--name control"),
        runner.thread_of("--name gpu-a"),
        runner.thread_of("describe-addon"),
    }
    assert len(threads) == 3, "the three chains did not run on their own threads"
    # Each chain keeps its order.
    control = [label for label, _t in runner.calls if "gpu.kubeconfig" not in label]
    assert (
        control.index(next(c for c in control if "--name control" in c))
        < control.index(next(c for c in control if "create namespace" in c))
        < control.index(next(c for c in control if "get secret" in c))
    ), "the control-plane chain ran out of order"
    gpu_writes = [
        "update-kubeconfig" in label for label in runner.labels("gpu.kubeconfig")
    ]
    assert gpu_writes.count(True) == 2 and gpu_writes == sorted(
        gpu_writes, reverse=True
    ), "GPU namespaces were created before every GPU context was written"
    assert cpu_kubeconfig == tmp_path / "cpu.kubeconfig"
    assert gpu_kubeconfig == tmp_path / "gpu.kubeconfig"
    assert master_file.read_text(encoding="utf-8") == "master"
    assert ensured == ["site-a"]
    assert "pod_identity_agent" in state.value["completed_tasks"], (
        "the add-on checkpoint semantics changed"
    )
    assert state.value["resources"]["pod_identity_agent"]["ownership"] == "CREATED"


def test_a_failing_chain_stops_cluster_access_with_its_own_error(
    tmp_path: Path,
) -> None:
    runner = _ClusterAccess(pause=0.0)
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")

    def refuse(_runner, _cluster, _site_id) -> dict:
        raise BootstrapError("eks describe-addon: AccessDenied")

    with pytest.raises(BootstrapError, match="AccessDenied"):
        admin_bootstrap.prepare_cluster_access(
            runner,
            state=state,
            cpu=_cluster(),
            gpu_clusters=[_gpu("gpu-a")],
            state_dir=tmp_path,
            namespace="gpu-fault-system",
            secure_dir=tmp_path / "secure",
            site_id="site-a",
            ensure_pod_identity_agent=refuse,
        )

    assert "pod_identity_agent" not in state.value["completed_tasks"]


def test_deploy_host_tools_are_probed_concurrently() -> None:
    tools = [
        {
            "name": f"tool-{index}",
            "executable": f"tool-{index}",
            "command": [f"tool-{index}", "--version"],
        }
        for index in range(6)
    ]

    def slow_version(arguments, **_kwargs):
        time.sleep(0.1)
        return subprocess.CompletedProcess(arguments, 0, f"{arguments[0]} 1.0", "")

    started = time.monotonic()
    report = admin_bootstrap_dependencies.deploy_host_dependency_report(
        manifest={
            "schema_version": 1,
            "python": {"major": 3, "minor": 12},
            "tools": tools,
        },
        which=lambda name: f"/usr/bin/{name}",
        runner=slow_version,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.6 * 0.75, f"six 100 ms probes took {elapsed:.2f}s"
    assert [item["name"] for item in report["tools"]] == [t["name"] for t in tools], (
        "the report lost the manifest order"
    )
    assert all(item["status"] == "PASS" for item in report["tools"]), (
        "a healthy tool was reported unhealthy"
    )


# --- discovery hints ------------------------------------------------------------------


def _hyperpod(name: str, eks_arn: str) -> dict:
    return {
        "ClusterArn": f"arn:aws:sagemaker:us-east-1:123456789012:cluster/{name}",
        "ClusterName": name,
        "NodeRecovery": "None",
        "Orchestrator": {"Eks": {"ClusterArn": eks_arn}},
    }


class _Region:
    """A region with several HyperPod clusters; records every sagemaker call."""

    def __init__(self, clusters: dict[str, str]) -> None:
        self.clusters = {
            _hyperpod(name, eks)["ClusterArn"]: _hyperpod(name, eks)
            for name, eks in clusters.items()
        }
        self.calls: list[str] = []

    def aws_json(self, region, service, operation, *arguments, **_kwargs):
        self.calls.append(f"{service} {operation}")
        if service == "sagemaker" and operation == "list-clusters":
            return {
                "ClusterSummaries": [
                    {"ClusterName": item["ClusterName"]}
                    for item in self.clusters.values()
                ]
            }
        if service == "sagemaker":
            wanted = arguments[arguments.index("--cluster-name") + 1]
            for arn, item in self.clusters.items():
                if wanted in (arn, item["ClusterName"]):
                    return item
            raise BootstrapError(
                "command failed (254): aws: An error occurred (ResourceNotFound) "
                f"when calling the DescribeCluster operation: {wanted}"
            )
        return {
            "cluster": {
                "status": "ACTIVE",
                "resourcesVpcConfig": {"vpcId": "vpc-a", "subnetIds": ["subnet-a"]},
            }
        }


def test_a_persisted_hint_replaces_the_region_inventory_with_one_describe(
    monkeypatch, tmp_path: Path
) -> None:
    eks = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
    monkeypatch.setattr(
        admin_bootstrap, "discover_subnet_cidrs", lambda *_a, **_k: ("10.0.0.0/24",)
    )
    admin_bootstrap.hyperpod_inventory.cache_clear()
    region = _Region({"gpu-a": eks, "gpu-b": eks.replace("gpu-a", "gpu-b")})
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "bootstrap-state.json").write_text(
        json.dumps(
            {
                "site_id": "x",
                "resources": {
                    admin_bootstrap.HYPERPOD_HINTS: {
                        eks: "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a"
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    discovered = admin_bootstrap.discover_cluster(
        region,
        cluster_arn=eks,
        role="gpu",
        context="gpu-a",
        hyperpod_hints=admin_bootstrap.load_hyperpod_hints(state_dir),
    )

    assert discovered.hyperpod_name == "gpu-a"
    assert "sagemaker list-clusters" not in region.calls, (
        "the hint did not spare the inventory"
    )
    assert region.calls.count("sagemaker describe-cluster") == 1


def test_a_stale_hint_falls_back_to_the_inventory(monkeypatch) -> None:
    """The persisted map is a hint, not an identity: a HyperPod cluster that no
    longer orchestrates this EKS cluster (rebuilt, renamed, gone) is not trusted."""

    eks_a = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
    eks_b = eks_a.replace("gpu-a", "gpu-b")
    monkeypatch.setattr(
        admin_bootstrap, "discover_subnet_cidrs", lambda *_a, **_k: ("10.0.0.0/24",)
    )
    admin_bootstrap.hyperpod_inventory.cache_clear()
    region = _Region({"gpu-a": eks_a, "gpu-b": eks_b})

    # Points at gpu-b for gpu-a's EKS ARN: the describe disagrees, inventory decides.
    rewired = admin_bootstrap.discover_cluster(
        region,
        cluster_arn=eks_a,
        role="gpu",
        context="gpu-a",
        hyperpod_hints={eks_a: region.clusters[list(region.clusters)[1]]["ClusterArn"]},
    )
    assert rewired.hyperpod_name == "gpu-a"
    assert "sagemaker list-clusters" in region.calls, "a stale hint was trusted"

    # Points at a cluster that is gone: not-found is absent, not an error.
    region = _Region({"gpu-a": eks_a})
    admin_bootstrap.hyperpod_inventory.cache_clear()
    gone = admin_bootstrap.discover_cluster(
        region,
        cluster_arn=eks_a,
        role="gpu",
        context="gpu-a",
        hyperpod_hints={eks_a: "arn:aws:sagemaker:us-east-1:123456789012:cluster/old"},
    )
    assert gone.hyperpod_name == "gpu-a"
    assert admin_bootstrap.load_hyperpod_hints(Path("/nonexistent")) == {}


# --- describe_or_absent ------------------------------------------------------------


class _Describe:
    def __init__(self, error: str | None) -> None:
        self.error = error

    def aws_json(self, _region, *_arguments, **_kwargs):
        if self.error is not None:
            raise BootstrapError(f"command failed (254): aws: {self.error}")
        return {"Thing": {"Id": "t-1"}}


def test_describe_or_absent_reads_only_the_named_not_found_codes_as_absent() -> None:
    present = describe_or_absent(
        _Describe(None), "us-east-1", "rds", "describe-x", not_found=("XNotFound",)
    )
    absent = describe_or_absent(
        _Describe("An error occurred (XNotFound) when calling DescribeX"),
        "us-east-1",
        "rds",
        "describe-x",
        not_found=("XNotFound",),
    )

    assert present == {"Thing": {"Id": "t-1"}}
    assert absent is None
    for other in ("Throttling", "AccessDeniedException", "ExpiredToken"):
        with pytest.raises(BootstrapError, match=other):
            describe_or_absent(
                _Describe(f"An error occurred ({other}) when calling DescribeX"),
                "us-east-1",
                "rds",
                "describe-x",
                not_found=("XNotFound",),
            )


# --- Aurora subnet group -----------------------------------------------------------


class _AuroraAccount:
    """RDS, EC2 and the CPU cluster as an already-bootstrapped site sees them:
    every resource exists, tagged for the site, so the ensure should only read."""

    def __init__(self, *, subnet_group_subnets: Sequence[str]) -> None:
        self.subnet_group_subnets = list(subnet_group_subnets)
        self.calls: list[list[str]] = []
        self.mutations: list[str] = []

    def run(self, arguments, **kwargs) -> str:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        if kwargs.get("mutate"):
            self.mutations.append(argv[2] if argv[0] == "aws" else argv[0])
            return ""
        if argv[0] == "kubectl":
            return json.dumps(
                {
                    "items": [
                        {
                            "status": {
                                "addresses": [
                                    {"type": "InternalIP", "address": "10.0.1.5"}
                                ]
                            }
                        }
                    ]
                }
            )
        return ""

    def aws_text(self, region, *arguments, **kwargs) -> str:
        self.run(["aws", *arguments, "--region", region], **kwargs)
        return json.dumps({"username": "u", "password": "p"})

    def aws_json(self, region, *arguments, **kwargs) -> dict:
        self.run(["aws", *arguments, "--region", region], **kwargs)
        operation = arguments[1]
        tags = {"TagList": [{"Key": "gpu-fault:site-id", "Value": "site-a"}]}
        answers = {
            "describe-subnets": {
                "Subnets": [
                    {
                        "SubnetId": "subnet-private-a",
                        "AvailabilityZone": "us-east-1a",
                        "VpcId": "vpc-control",
                    },
                    {
                        "SubnetId": "subnet-private-b",
                        "AvailabilityZone": "us-east-1b",
                        "VpcId": "vpc-control",
                    },
                ]
            },
            "describe-route-tables": {
                "RouteTables": [
                    {
                        "RouteTableId": "rtb-main",
                        "Associations": [{"Main": True}],
                        "Routes": [
                            {
                                "DestinationCidrBlock": "0.0.0.0/0",
                                "NatGatewayId": "nat-1",
                            }
                        ],
                    }
                ]
            },
            "describe-db-subnet-groups": {
                "DBSubnetGroups": [
                    {
                        "DBSubnetGroupArn": "arn:aws:rds:us-east-1:123456789012:subgrp:g",
                        "Subnets": [
                            {"SubnetIdentifier": s} for s in self.subnet_group_subnets
                        ],
                    }
                ]
            },
            "list-tags-for-resource": tags,
            "describe-security-groups": {
                "SecurityGroups": [{"GroupId": "sg-aurora", "Tags": tags["TagList"]}]
            },
            "describe-network-interfaces": {
                "NetworkInterfaces": [{"Groups": [{"GroupId": "sg-nodes"}]}]
            },
            "describe-db-clusters": {
                "DBClusters": [
                    {
                        "DBClusterArn": "arn:aws:rds:us-east-1:123456789012:cluster:c",
                        "EngineVersion": "16.8",
                        "Endpoint": "c.cluster.example",
                        "MasterUserSecret": {
                            "SecretArn": "arn:aws:secretsmanager:x",
                            "KmsKeyId": "k",
                        },
                    }
                ]
            },
        }
        return answers.get(operation, {})


def _aurora(
    account: _AuroraAccount, monkeypatch, tmp_path: Path, *, stub_instances: bool = True
) -> dict:
    monkeypatch.setenv(RDS_CA_BUNDLE_ENVIRONMENT, "/etc/rds/ca.pem")
    monkeypatch.setattr(
        admin_bootstrap.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, "", ""),
    )
    for name in ("reconcile_cluster_diagnostics", "reconcile_existing_capacity"):
        monkeypatch.setattr(admin_bootstrap, name, lambda *_a, **_k: None)
    monkeypatch.setattr(
        admin_bootstrap, "ensure_cluster_parameter_group", lambda *_a, **_k: "pg"
    )
    if stub_instances:
        monkeypatch.setattr(
            admin_bootstrap, "ensure_serverless_writer", lambda *_a, **_k: ["w", "r"]
        )
    return _ensure_aurora(
        account,
        cpu=_cluster(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        capacity=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
    )


def test_the_subnet_group_is_not_modified_when_it_already_holds_the_subnets(
    monkeypatch, tmp_path: Path
) -> None:
    """`modify-db-subnet-group` is a mutation RDS answers slowly even when nothing
    changes; a re-entry against a settled group must only read."""

    settled = _AuroraAccount(
        subnet_group_subnets=["subnet-private-b", "subnet-private-a"]
    )

    result = _aurora(settled, monkeypatch, tmp_path)

    assert "modify-db-subnet-group" not in settled.mutations, settled.mutations
    assert result["subnet_group"] == "gpu-fault-site-a-aurora"
    describes = [argv[2] for argv in settled.calls if argv[0] == "aws"]
    assert describes.count("describe-db-subnet-groups") == 1, (
        "the subnet group was described twice (an existence probe and then the value)"
    )
    assert describes.count("describe-route-tables") == 1, (
        "the route tables were read per subnet instead of once for the VPC"
    )


def test_the_subnet_group_is_modified_when_its_subnets_drifted(
    monkeypatch, tmp_path: Path
) -> None:
    drifted = _AuroraAccount(subnet_group_subnets=["subnet-old-a", "subnet-private-b"])

    _aurora(drifted, monkeypatch, tmp_path)

    assert "modify-db-subnet-group" in drifted.mutations


def test_the_foundation_aurora_task_neither_waits_for_instances_nor_reads_the_secret(
    monkeypatch, tmp_path: Path
) -> None:
    """The instance wait and the credential read moved to ``aurora_ready`` so the
    monitoring and node-key installs no longer sit behind them. What RDS already
    exposes -- the master secret ARN of an existing cluster -- still rides along,
    so a checkpoint reader that knew the old shape keeps working."""

    settled = _AuroraAccount(
        subnet_group_subnets=["subnet-private-a", "subnet-private-b"]
    )

    result = _aurora(settled, monkeypatch, tmp_path, stub_instances=False)

    operations = [argv[2] for argv in settled.calls if argv[0] == "aws"]
    assert "wait" not in operations, "the foundation task waited for an instance"
    assert "get-secret-value" not in operations, "the foundation read the secret"
    assert "kubectl" not in settled.mutations, "the foundation applied the Secret"
    assert result["instance_ids"] == [
        "gpu-fault-site-a-aurora-writer",
        "gpu-fault-site-a-aurora-reader",
    ]
    assert result["master_secret_arn"] == "arn:aws:secretsmanager:x"
    assert result["availability_zones"] == ["us-east-1a", "us-east-1b"], (
        "aurora_ready cannot place the reader without the zones the foundation chose"
    )


class _ReadyAccount(_AuroraAccount):
    """The account as ``aurora_ready`` sees it: both instances available, and
    the applied manifests kept so a test can read what reached kubectl."""

    def __init__(self) -> None:
        super().__init__(subnet_group_subnets=[])
        self.applied: list[str] = []

    def run(self, arguments, **kwargs) -> str:
        argv = [str(item) for item in arguments]
        if argv[0] == "kubectl" and "apply" in argv:
            self.applied.append(str(kwargs.get("input_text")))
        if argv[0] == "kubectl" and "secret" in argv and not kwargs.get("mutate"):
            self.calls.append(argv)
            return "gpu-fault-aurora"
        return super().run(arguments, **kwargs)

    def aws_json(self, region, *arguments, **kwargs) -> dict:
        if arguments[1] == "describe-db-instances":
            self.run(["aws", *arguments, "--region", region], **kwargs)
            return {
                "DBInstances": [
                    {"DBInstanceIdentifier": "w", "DBInstanceStatus": "available"},
                    {"DBInstanceIdentifier": "r", "DBInstanceStatus": "available"},
                ]
            }
        return super().aws_json(region, *arguments, **kwargs)


def _ready(account: _ReadyAccount, tmp_path: Path, *, probe_only: bool) -> dict:
    return _aurora_ready(
        account,
        cpu=_cluster(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        aurora={"cluster_id": "gpu-fault-site-a-aurora", "instance_ids": ["w", "r"]},
        probe_only=probe_only,
    )


def test_aurora_ready_hands_the_control_plane_its_secret_once_both_instances_are_up(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(RDS_CA_BUNDLE_ENVIRONMENT, "/etc/rds/ca.pem")
    account = _ReadyAccount()

    result = _ready(account, tmp_path, probe_only=False)

    assert result == {
        "master_secret_arn": "arn:aws:secretsmanager:x",
        "master_secret_kms_key_arn": "k",
    }
    assert account.mutations == ["kubectl"], (
        "aurora_ready mutated something other than the control-plane Secret"
    )
    manifest = json.loads(account.applied[0])
    assert manifest["metadata"]["name"] == "gpu-fault-aurora"
    assert manifest["stringData"]["master-secret-arn"] == "arn:aws:secretsmanager:x"
    url = manifest["stringData"]["postgres-url"]
    assert url.startswith("postgresql://u:p@c.cluster.example:5432/"), url
    assert "sslmode=verify-full" in url


def test_the_aurora_ready_probe_reads_two_facts_and_mutates_nothing(
    tmp_path: Path,
) -> None:
    account = _ReadyAccount()

    result = _ready(account, tmp_path, probe_only=True)

    assert result == {}
    assert account.mutations == []
    operations = [argv[2] for argv in account.calls if argv[0] == "aws"]
    assert operations == ["describe-db-instances"], operations
    assert any(argv[0] == "kubectl" and "secret" in argv for argv in account.calls), (
        "the probe did not check that the control-plane Secret is present"
    )
