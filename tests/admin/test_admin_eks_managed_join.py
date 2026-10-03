"""Joining a plain EKS GPU cluster (provider ``eks-managed``) through the admin CLI.

The target modelled here is the live ``gpu-eks-cluster``: EKS 1.31, VPC
10.0.0.0/16 with three private /20 subnets, a Spot ``p5en.48xlarge`` managed
node group labelled ``nvidia.com/gpu.present=true`` and tainted
``nvidia.com/gpu=present:NoSchedule``, an ``m7i-flex.xlarge`` CPU node group that
must never be selected, FQDN node names and ``aws:///<zone>/i-...`` provider ids.
No HyperPod cluster orchestrates it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_services
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    ClusterIdentity,
    default_cluster_id,
)
from gpu_fault.admin.bootstrap_discovery import regional_profile_for
from gpu_fault.admin.cluster_join_types import JoinClusterRequest
from gpu_fault.admin.site import load_site
from gpu_fault.cluster_scope import NODE_SELECTOR_ENV, PROVIDER_ENV, node_selector
from tests.admin._cov95_join_support import ACCOUNT, REGION, JoinScenario, target

EKS_NAME = "gpu-eks-cluster"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{EKS_NAME}"
GPU_NODEGROUP = "gpu-p5en-efa-spot-node-group"
CPU_NODEGROUP = "cpu-node-group-20260930"
GPU_SUBNETS = ("subnet-gpu-a", "subnet-gpu-b")
CPU_SUBNETS = ("subnet-cpu-a",)
CONTROL_PLANE_SUBNETS = ("subnet-cp-a", "subnet-cp-b")
CIDRS = {
    "subnet-gpu-a": "10.0.0.0/20",
    "subnet-gpu-b": "10.0.16.0/20",
    "subnet-cpu-a": "10.0.32.0/20",
    "subnet-cp-a": "10.0.48.0/24",
    "subnet-cp-b": "10.0.49.0/24",
}
GPU_NODE = "ip-10-0-33-205.us-west-2.compute.internal"
GPU_INSTANCE = "i-0222042756c3a2a68"


class EksRunner:
    """``aws`` answers for a Region that has no HyperPod cluster at all."""

    def __init__(self, *, status: str = "ACTIVE", nodegroup_status: str = "ACTIVE"):
        self.status = status
        self.nodegroup_status = nodegroup_status
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.hyperpods: list[dict] = []

    def aws_json(self, region, service, operation, *arguments, **_options):
        self.calls.append((service, operation, tuple(arguments)))
        assert region == REGION, "discovery crossed the Region boundary"
        if (service, operation) == ("sagemaker", "list-clusters"):
            return {
                "ClusterSummaries": [
                    {"ClusterName": item["ClusterName"]} for item in self.hyperpods
                ]
            }
        if (service, operation) == ("sagemaker", "describe-cluster"):
            selected = arguments[arguments.index("--cluster-name") + 1]
            return next(
                item
                for item in self.hyperpods
                if selected in {item["ClusterName"], item["ClusterArn"]}
            )
        if (service, operation) == ("eks", "describe-cluster"):
            return {
                "cluster": {
                    "arn": EKS_ARN,
                    "status": self.status,
                    "resourcesVpcConfig": {
                        "vpcId": "vpc-gpu-eks",
                        "subnetIds": list(CONTROL_PLANE_SUBNETS),
                    },
                }
            }
        if (service, operation) == ("eks", "list-nodegroups"):
            return {"nodegroups": [CPU_NODEGROUP, GPU_NODEGROUP]}
        if (service, operation) == ("eks", "describe-nodegroup"):
            name = arguments[arguments.index("--nodegroup-name") + 1]
            gpu = name == GPU_NODEGROUP
            return {
                "nodegroup": {
                    "nodegroupName": name,
                    "status": self.nodegroup_status if gpu else "ACTIVE",
                    "capacityType": "SPOT" if gpu else "ON_DEMAND",
                    "instanceTypes": ["p5en.48xlarge"] if gpu else ["m7i-flex.xlarge"],
                    "subnets": list(GPU_SUBNETS if gpu else CPU_SUBNETS),
                    "labels": (
                        {"node-type": "gpu", "nvidia.com/gpu.present": "true"}
                        if gpu
                        else {"node-type": "cpu"}
                    ),
                }
            }
        if (service, operation) == ("ec2", "describe-subnets"):
            ids = arguments[arguments.index("--subnet-ids") + 1 :]
            return {
                "Subnets": [
                    {"SubnetId": item, "CidrBlock": CIDRS[item]} for item in ids
                ]
            }
        raise AssertionError(f"unexpected discovery call {service}/{operation}")


def test_eks_arn_without_hyperpod_discovers_an_eks_managed_identity() -> None:
    runner = EksRunner()

    identity = admin_bootstrap.discover_cluster(
        runner, cluster_arn=EKS_ARN, role="gpu", context="gpu-eks"
    )

    assert identity.provider == "eks-managed", "a HyperPod-less EKS ARN is eks-managed"
    assert identity.is_eks_managed, "the identity must report itself as plain EKS"
    assert identity.node_selector == "nvidia.com/gpu.present=true", (
        "the default eks-managed selector must be recorded on the identity"
    )
    assert (identity.hyperpod_arn, identity.hyperpod_name, identity.node_recovery) == (
        "",
        "",
        "",
    ), "no HyperPod facts may be invented for a plain EKS cluster"
    assert identity.eks_arn == EKS_ARN and identity.eks_name == EKS_NAME, (
        "the EKS identity must be the ARN given"
    )
    assert identity.subnet_ids == CONTROL_PLANE_SUBNETS, (
        "subnet_ids stays the EKS set that places Aurora and the NLB"
    )
    assert identity.subnet_cidrs == ("10.0.0.0/20", "10.0.16.0/20"), (
        "Agent endpoint CIDRs must come from the GPU node group subnets only"
    )
    assert default_cluster_id(identity) == EKS_NAME, (
        "the default cluster_id of a plain EKS cluster is its EKS name"
    )
    assert node_selector(identity) == "nvidia.com/gpu.present=true", (
        "the selector helper must return the recorded eks-managed selector"
    )
    assert all(
        operation.startswith(("describe", "list")) for _s, operation, _a in runner.calls
    ), "discovery performed a mutation"


def test_gpu_nodegroup_flag_selects_that_node_group_and_its_subnets() -> None:
    runner = EksRunner()

    identity = admin_bootstrap.discover_cluster(
        runner,
        cluster_arn=EKS_ARN,
        role="gpu",
        context="gpu-eks",
        node_selector=f"eks.amazonaws.com/nodegroup={GPU_NODEGROUP}",
    )

    assert identity.node_selector == f"eks.amazonaws.com/nodegroup={GPU_NODEGROUP}", (
        "an explicit node group selector must be recorded verbatim"
    )
    described = [
        arguments[arguments.index("--nodegroup-name") + 1]
        for service, operation, arguments in runner.calls
        if (service, operation) == ("eks", "describe-nodegroup")
    ]
    assert described == [GPU_NODEGROUP], (
        "only the named node group is described when the selector names one"
    )
    assert identity.subnet_cidrs == ("10.0.0.0/20", "10.0.16.0/20"), (
        "CIDRs follow the named node group"
    )


def test_unknown_nodegroup_in_selector_fails_closed() -> None:
    with pytest.raises(BootstrapError, match="no managed node group named"):
        admin_bootstrap.discover_cluster(
            EksRunner(),
            cluster_arn=EKS_ARN,
            role="gpu",
            context="gpu-eks",
            node_selector="eks.amazonaws.com/nodegroup=missing",
        )


def test_eks_managed_discovery_refuses_inactive_cluster_or_nodegroup() -> None:
    with pytest.raises(BootstrapError, match="not ACTIVE"):
        admin_bootstrap.discover_cluster(
            EksRunner(status="UPDATING"),
            cluster_arn=EKS_ARN,
            role="gpu",
            context="gpu-eks",
        )
    with pytest.raises(BootstrapError, match="is not ACTIVE"):
        admin_bootstrap.discover_cluster(
            EksRunner(nodegroup_status="DEGRADED"),
            cluster_arn=EKS_ARN,
            role="gpu",
            context="gpu-eks",
        )


def test_eks_managed_discovery_refuses_a_cpu_role_and_bad_selectors() -> None:
    with pytest.raises(BootstrapError, match="CPU control plane must be a HyperPod"):
        admin_bootstrap.discover_cluster(
            EksRunner(), cluster_arn=EKS_ARN, role="cpu", context="cpu"
        )
    with pytest.raises(BootstrapError, match="invalid --node-selector"):
        admin_bootstrap.discover_cluster(
            EksRunner(),
            cluster_arn=EKS_ARN,
            role="gpu",
            context="gpu-eks",
            node_selector="bad key=1",
        )


def test_cluster_with_no_gpu_nodegroup_names_the_remedy() -> None:
    class CpuOnly(EksRunner):
        def aws_json(self, region, service, operation, *arguments, **options):
            if (service, operation) == ("eks", "list-nodegroups"):
                self.calls.append((service, operation, tuple(arguments)))
                return {"nodegroups": [CPU_NODEGROUP]}
            return super().aws_json(region, service, operation, *arguments, **options)

    with pytest.raises(BootstrapError, match="--gpu-nodegroup"):
        admin_bootstrap.discover_cluster(
            CpuOnly(), cluster_arn=EKS_ARN, role="gpu", context="gpu-eks"
        )


def test_hyperpod_discovery_is_unchanged_when_the_eks_arn_has_an_orchestrator() -> None:
    runner = EksRunner()
    hyperpod_arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-gpu-eks"
    runner.hyperpods.append(
        {
            "ClusterArn": hyperpod_arn,
            "ClusterName": "hp-gpu-eks",
            "NodeRecovery": "None",
            "Orchestrator": {"Eks": {"ClusterArn": EKS_ARN}},
        }
    )

    identity = admin_bootstrap.discover_cluster(
        runner, cluster_arn=EKS_ARN, role="gpu", context="gpu-eks"
    )

    assert identity.provider == "hyperpod", "an orchestrated EKS ARN stays HyperPod"
    assert identity.hyperpod_name == "hp-gpu-eks", "the HyperPod name is recorded"
    assert node_selector(identity) == (
        "sagemaker.amazonaws.com/cluster-name=hp-gpu-eks"
    ), "the HyperPod selector text is byte-identical to the historical label"
    assert not any(
        operation in {"list-nodegroups", "describe-nodegroup"}
        for _s, operation, _a in runner.calls
    ), "HyperPod discovery must not read EKS node groups"


def test_bootstrap_request_derives_the_nodegroup_selector_and_refuses_both() -> None:
    request = BootstrapRequest(
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        gpu_cluster_arns=(EKS_ARN,),
        repository_root=__import__("pathlib").Path("/repo"),
        state_dir=__import__("pathlib").Path("/state"),
        gpu_nodegroup=GPU_NODEGROUP,
    )
    assert request.requested_node_selector == (
        f"eks.amazonaws.com/nodegroup={GPU_NODEGROUP}"
    ), "--gpu-nodegroup must derive the node group selector"
    both = replace(request, node_selector="a=b")
    with pytest.raises(BootstrapError, match="exclusive"):
        both.requested_node_selector


def test_regional_profile_follows_the_gpu_providers(tmp_path) -> None:
    hyperpod = target("b")
    eks = replace(hyperpod, provider="eks-managed", hyperpod_arn="", hyperpod_name="")
    assert regional_profile_for(tmp_path, [eks]) == (
        str(tmp_path / "config/runtime-profile.regional-eks-safe.example.yaml"),
        "eks-v1",
    ), "an eks-managed-only site starts from the EKS profile"
    assert regional_profile_for(tmp_path, [hyperpod, eks])[1] == "hyperpod-v1", (
        "a site with any HyperPod cluster keeps the HyperPod profile"
    )
    assert regional_profile_for(tmp_path, [])[1] == "hyperpod-v1", (
        "a site without GPU clusters keeps the HyperPod profile"
    )


def test_eks_managed_executor_policy_is_ec2_only_and_tag_scoped() -> None:
    identity = eks_target()
    document = bootstrap_services.executor_policy_document(
        hyperpod_arn="", provider=identity.provider, eks_cluster_arn=identity.eks_arn
    )
    actions = {
        action
        for statement in document["Statement"]
        for action in (
            statement["Action"]
            if isinstance(statement["Action"], list)
            else [statement["Action"]]
        )
    }
    assert actions == {"ec2:DescribeInstances", "ec2:RebootInstances"}, (
        "an eks-managed executor holds exactly describe and reboot"
    )
    reboot = next(
        item
        for item in document["Statement"]
        if item["Action"] == "ec2:RebootInstances"
    )
    assert reboot["Condition"] == {
        "StringEquals": {"aws:ResourceTag/eks:cluster-name": "gpu-e"}
    }, "reboot must be conditioned on the EKS cluster-name tag"
    assert reboot["Resource"] == f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/*", (
        "reboot is scoped to this account and Region's instances"
    )
    assert not any(action.startswith("sagemaker:") for action in actions), (
        "no sagemaker action may reach a plain EKS executor"
    )
    with pytest.raises(BootstrapError, match="HyperPod ARN"):
        bootstrap_services.executor_policy_document(hyperpod_arn="")


def eks_target() -> ClusterIdentity:
    """The plain EKS cluster as ``discover_cluster`` returns it."""

    return ClusterIdentity(
        input_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-e",
        role="gpu",
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn="",
        hyperpod_name="",
        eks_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-e",
        eks_name="gpu-e",
        vpc_id="vpc-gpu-e",
        subnet_ids=("subnet-e",),
        subnet_cidrs=("10.0.0.0/20", "10.0.16.0/20"),
        node_recovery="",
        context="gpu-e",
        provider="eks-managed",
        node_selector="nvidia.com/gpu.present=true",
    )


def _eks_node(name: str = GPU_NODE) -> dict:
    return {
        "kind": "Node",
        "metadata": {
            "name": name,
            "uid": "uid-" + name,
            "resourceVersion": "1",
            "labels": {
                "nvidia.com/gpu.present": "true",
                "node-type": "gpu",
                "eks.amazonaws.com/nodegroup": GPU_NODEGROUP,
                "topology.kubernetes.io/zone": "us-west-2c",
                "node.kubernetes.io/instance-type": "p5en.48xlarge",
            },
        },
        "spec": {
            "providerID": f"aws:///us-west-2c/{GPU_INSTANCE}",
            "taints": [
                {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
            ],
        },
    }


class EksJoinScenario(JoinScenario):
    """The HyperPod join world plus one plain EKS cluster, ``gpu-e``."""

    def __init__(self, tmp_path, monkeypatch):
        super().__init__(tmp_path, monkeypatch, suffixes=("b",))
        self.targets[eks_target().eks_arn] = eks_target()
        self.node_names["e"] = GPU_NODE
        self.node_key_environments: list[dict[str, str]] = []
        monkeypatch.setattr(join, "provision_node_action_keys", self.eks_node_keys)

    def eks_node_keys(self, runner, *, cluster, cluster_id, **_options):
        # Record what bootstrap_services would hand the provisioning script.
        self.node_key_environments.append(
            {
                "GPU_FAULT_HYPERPOD_CLUSTER": cluster.hyperpod_name or cluster_id,
                PROVIDER_ENV: cluster.provider,
                NODE_SELECTOR_ENV: node_selector(cluster),
            }
        )
        return self.node_keys(runner, cluster=cluster, cluster_id=cluster_id)

    def run(self, arguments, **options):
        context = (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )
        if "get" in arguments and "nodes" in arguments and context == "gpu-e":
            self.commands.append((list(arguments), dict(options)))
            if any(str(item).startswith("go-template=") for item in arguments):
                return json.dumps([GPU_NODE])
            return json.dumps({"items": [_eks_node()]})
        return super().run(arguments, **options)

    def eks_request(self, **changes):
        return JoinClusterRequest(
            site=load_site(self.path), gpu_cluster_arn=eks_target().eks_arn, **changes
        )

    def join_eks(self, **changes):
        return join.join_cluster(self.eks_request(**changes), runner=self)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return EksJoinScenario(tmp_path, monkeypatch)


def _node_selectors(scenario, context: str) -> set[str]:
    return {
        arguments[arguments.index("-l") + 1]
        for arguments, _options in scenario.commands
        if "nodes" in arguments
        and "-l" in arguments
        and "--context" in arguments
        and arguments[arguments.index("--context") + 1] == context
    }


def test_full_eks_managed_join_completes_through_the_state_machine(scenario) -> None:
    result = scenario.join_eks()

    assert result["phase"] == "COMPLETED", result
    state = json.loads(
        (scenario.path.parent / "join-cluster")
        .glob("*/state.json")
        .__next__()
        .read_text()
    )
    assert state["phase"] == "COMPLETED", "the eks-managed join must complete"
    assert {
        "LOCAL_INPUTS_READY",
        "NODE_NAMES_VERIFIED",
        "NODE_KEYS_STARTED",
        "JOINED",
        "VERIFIED",
        "SITE_UPDATED",
        "REGISTRY_UPDATED",
        "ACTIVATED",
        "FINAL_VERIFIED",
    }.issubset(state["completed_steps"]), "a persistent join barrier was skipped"
    assert state["evidence"]["DISCOVERED"]["cluster_id"] == "gpu-e", (
        "the default cluster_id of a plain EKS cluster is its EKS name"
    )
    assert state["evidence"]["DISCOVERED"]["target"]["provider"] == "eks-managed", (
        "the journaled identity must record the provider"
    )
    assert state["evidence"]["NODE_NAMES_VERIFIED"]["nodes"] == [GPU_NODE], (
        "the FQDN GPU node must be the verified node set"
    )
    assert scenario.cluster_states == {"gpu-a": "ACTIVE", "gpu-e": "ACTIVE"}, (
        "the registry must end with the new member ACTIVE"
    )

    document = yaml.safe_load(scenario.path.read_text())
    entry = next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == "gpu-e"
    )
    assert entry["provider"] == "eks-managed", "site.yaml must record the provider"
    assert entry["nodeSelector"] == "nvidia.com/gpu.present=true", (
        "site.yaml must record the GPU node selector"
    )
    assert "hyperpodClusterName" not in entry, "no HyperPod name may be written"
    assert entry["agentEndpointAllowedCidrs"] == ["10.0.0.0/20", "10.0.16.0/20"], (
        "Agent endpoint CIDRs come from the GPU node group subnets"
    )
    hyperpod_entry = next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == "gpu-a"
    )
    assert "provider" not in hyperpod_entry and "nodeSelector" not in hyperpod_entry, (
        "the existing HyperPod entry must be left byte for byte as it was"
    )

    assert _node_selectors(scenario, "gpu-e") == {"nvidia.com/gpu.present=true"}, (
        "every node read of the plain EKS cluster must use its selector"
    )
    assert _node_selectors(scenario, "gpu-a") == {
        "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
    }, "the HyperPod sibling keeps the historical label selector"

    recorded = {item.resource_key for item in scenario.registry.live.resources}
    assert "cluster/gpu-e/eks" in recorded, "the EKS cluster must be registered"
    assert "aws/iam/executor/gpu-e/role" in recorded, "the executor role is recorded"
    assert not any(key.startswith("cluster/gpu-e/hyperpod") for key in recorded), (
        "a plain EKS cluster records no gpu_hyperpod resource"
    )
    assert scenario.node_key_environments == [
        {
            "GPU_FAULT_HYPERPOD_CLUSTER": "gpu-e",
            PROVIDER_ENV: "eks-managed",
            NODE_SELECTOR_ENV: "nvidia.com/gpu.present=true",
        }
    ], "node-key provisioning must receive the cluster_id as provider scope"
    assert "gpu-e" in scenario.events[0] or any(
        event.startswith("discover:gpu-e") for event in scenario.events
    ), "discovery must have run for the plain EKS target"


def test_eks_managed_join_honours_an_explicit_cluster_id(scenario) -> None:
    result = scenario.join_eks(cluster_id="Spot GPU Pool")
    assert result["cluster_id"] == "spot-gpu-pool", (
        "--cluster-id must be normalised and win over the EKS name"
    )


def test_eks_managed_join_is_idempotent_and_dedupes_on_the_eks_arn(scenario) -> None:
    first = scenario.join_eks()
    assert first["phase"] == "COMPLETED", first
    replay = scenario.join_eks()
    assert replay["phase"] == "COMPLETED", "a replayed join must stay completed"
    other = join.JoinClusterRequest(
        site=load_site(scenario.path),
        gpu_cluster_arn=eks_target().eks_arn,
        cluster_id="another-id",
    )
    # The journal for this ARN already binds cluster_id gpu-e; a different id
    # is refused before discovery (and would be refused by the site dedupe on
    # the EKS ARN after it). Either way the second identity never lands.
    with pytest.raises(
        BootstrapError, match="conflicts on requested_cluster_id|already managed"
    ):
        join.join_cluster(other, runner=scenario)


def test_two_plain_eks_clusters_are_not_mistaken_for_one_another(scenario) -> None:
    """Two eks-managed clusters share an empty HyperPod name; the empty names
    must never read as the same HyperPod cluster."""

    scenario.join_eks()
    second = replace(
        eks_target(),
        input_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-f",
        eks_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-f",
        eks_name="gpu-f",
        vpc_id="vpc-gpu-f",
        context="gpu-f",
    )
    scenario.targets[second.eks_arn] = second
    scenario.node_names["f"] = "ip-10-0-40-7.us-west-2.compute.internal"
    request = join.JoinClusterRequest(
        site=load_site(scenario.path), gpu_cluster_arn=second.eks_arn
    )
    # ``_validate_target`` runs during discovery; a false HyperPod-name match
    # would raise "GPU HyperPod cluster is already managed" here.
    discovered, cluster_id, existing = join._discover_join_target(request, scenario)
    assert (discovered.eks_name, cluster_id, existing) == ("gpu-f", "gpu-f", None), (
        "a second plain EKS cluster must be a new, distinct join target"
    )


def test_hyperpod_join_still_renders_the_historical_selector(scenario) -> None:
    """The HyperPod path is the fully validated one and must not move."""

    result = scenario.join("b")
    assert result["phase"] == "COMPLETED", result
    assert _node_selectors(scenario, "gpu-b") == {
        "sagemaker.amazonaws.com/cluster-name=hp-gpu-b"
    }, "HyperPod node reads must keep the exact historical label text"
    document = yaml.safe_load(scenario.path.read_text())
    entry = next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == "hp-gpu-b"
    )
    assert entry["hyperpodClusterName"] == "hp-gpu-b", "HyperPod name must be written"
    assert "provider" not in entry and "nodeSelector" not in entry, (
        "a HyperPod entry gains no provider keys"
    )
    recorded = {item.resource_key for item in scenario.registry.live.resources}
    assert "cluster/hp-gpu-b/hyperpod" in recorded, (
        "a HyperPod join still records the gpu_hyperpod resource"
    )
