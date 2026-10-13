"""Joining a plain EKS GPU cluster (provider ``eks-managed``) through the admin CLI.

The target modelled here is the live ``gpu-eks-cluster``: EKS 1.31, VPC
10.0.0.0/16 with three private /20 subnets, a Spot ``p5en.48xlarge`` managed
node group labelled ``nvidia.com/gpu.present=true`` and tainted
``nvidia.com/gpu=present:NoSchedule``, an ``m7i-flex.xlarge`` CPU node group that
must never be selected, FQDN node names and ``aws:///<zone>/i-...`` provider ids.
No HyperPod cluster orchestrates it.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_services, cluster_join_profile, profile_approval
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    ClusterIdentity,
    default_cluster_id,
    profile_node_reboot_owner,
    require_profile_matches_provider,
    require_single_provider,
)
from gpu_fault.admin.bootstrap_discovery import regional_profile_for
from gpu_fault.admin.cluster_join_types import JoinClusterRequest
from gpu_fault.admin.site import load_site
from gpu_fault.cluster_scope import (
    NODE_SELECTOR_ENV,
    PROVIDER_ENV,
    PROVIDER_NODE_REBOOT_OWNERS,
    PROVIDER_PROFILE_SOURCES,
    node_selector,
)
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
GPU_INSTANCE = "i-00000000000000001"


SITE_ID = "us-east-1-control-985fd370"
OWNERSHIP = "gpu-fault.io/control-plane"
EKS_OWNERSHIP_SELECTOR = f"{OWNERSHIP}={SITE_ID}"


class EksRunner:
    """``aws`` answers for a Region that has no HyperPod cluster at all.

    The GPU node group carries the ownership label for ``SITE_ID`` unless a
    test says otherwise (``ownership``); the CPU node group never does unless a
    test labels it (``cpu_ownership``) to model an operator mistake.
    """

    def __init__(
        self,
        *,
        status: str = "ACTIVE",
        nodegroup_status: str = "ACTIVE",
        node_repair: bool | None = None,
        ownership: str | None = SITE_ID,
        cpu_ownership: str | None = None,
    ):
        self.status = status
        self.nodegroup_status = nodegroup_status
        # EKS node auto repair of the GPU node group: None leaves the opt-in
        # field out of the description, as EKS does for a group never configured.
        self.node_repair = node_repair
        self.ownership = ownership
        self.cpu_ownership = cpu_ownership
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.hyperpods: list[dict] = []

    def nodegroup_labels(self, gpu: bool) -> dict[str, str]:
        if gpu:
            labels = {"node-type": "gpu", "nvidia.com/gpu.present": "true"}
            if self.ownership is not None:
                labels[OWNERSHIP] = self.ownership
            return labels
        labels = {"node-type": "cpu"}
        if self.cpu_ownership is not None:
            labels[OWNERSHIP] = self.cpu_ownership
        return labels

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
                    "labels": self.nodegroup_labels(gpu),
                    **(
                        {"nodeRepairConfig": {"enabled": self.node_repair}}
                        if gpu and self.node_repair is not None
                        else {}
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


def _discover(runner, **overrides):
    arguments = dict(
        cluster_arn=EKS_ARN, role="gpu", context="gpu-eks", site_id=SITE_ID
    )
    arguments.update(overrides)
    return admin_bootstrap.discover_cluster(runner, **arguments)


def test_eks_arn_without_hyperpod_discovers_an_eks_managed_identity() -> None:
    runner = EksRunner()

    identity = _discover(runner)

    assert identity.provider == "eks-managed", "a HyperPod-less EKS ARN is eks-managed"
    assert identity.is_eks_managed, "the identity must report itself as plain EKS"
    assert identity.node_selector == EKS_OWNERSHIP_SELECTOR, (
        "the ownership label selector must be recorded on the identity"
    )
    assert identity.node_selector_source == "label-convention", (
        "a cluster joining now records where its selector came from"
    )
    assert (identity.hyperpod_arn, identity.hyperpod_name, identity.node_recovery) == (
        "",
        "",
        "",
    ), "no HyperPod facts may be invented for a plain EKS cluster"
    assert identity.node_repair == "unset", (
        "a node group without nodeRepairConfig has the opt-in feature off; "
        "record that it is unset rather than inventing disabled"
    )
    assert identity.eks_arn == EKS_ARN and identity.eks_name == EKS_NAME, (
        "the EKS identity must be the ARN given"
    )
    assert identity.subnet_ids == CONTROL_PLANE_SUBNETS, (
        "subnet_ids stays the EKS set that places Aurora and the NLB"
    )
    assert identity.subnet_cidrs == ("10.0.0.0/20", "10.0.16.0/20"), (
        "Agent endpoint CIDRs must come from the labelled node group subnets only"
    )
    assert default_cluster_id(identity) == EKS_NAME, (
        "the default cluster_id of a plain EKS cluster is its EKS name"
    )
    assert node_selector(identity) == EKS_OWNERSHIP_SELECTOR, (
        "the selector helper must return the recorded eks-managed selector"
    )
    described = [
        arguments[arguments.index("--nodegroup-name") + 1]
        for service, operation, arguments in runner.calls
        if (service, operation) == ("eks", "describe-nodegroup")
    ]
    assert described == [CPU_NODEGROUP, GPU_NODEGROUP], (
        "every node group is described: the label is compared on all of them"
    )
    assert all(
        operation.startswith(("describe", "list")) for _s, operation, _a in runner.calls
    ), "discovery performed a mutation"


def test_a_cluster_without_a_labelled_node_group_is_refused_with_the_evidence() -> None:
    runner = EksRunner(ownership=None)

    with pytest.raises(BootstrapError) as refusal:
        _discover(runner)

    message = str(refusal.value)
    assert f"{OWNERSHIP}={SITE_ID}" in message, "the exact label and site id are named"
    assert "gpu-fault-admin site-id" in message, "the verb that prints the id is named"
    assert (
        f"{GPU_NODEGROUP} instanceTypes=p5en.48xlarge {OWNERSHIP}=absent" in message
    ), "the describe-nodegroup evidence of every group is printed"
    assert (
        f"{CPU_NODEGROUP} instanceTypes=m7i-flex.xlarge {OWNERSHIP}=absent" in message
    )
    assert (
        f"aws eks update-nodegroup-config --cluster-name {EKS_NAME} "
        f"--nodegroup-name {GPU_NODEGROUP} --labels addOrUpdateLabels={{{OWNERSHIP}={SITE_ID}}}"
    ) in message, "the remedy names the GPU node group"
    assert not any(
        operation.startswith(("create", "update", "put", "tag"))
        for _s, operation, _a in runner.calls
    ), "a refusal happens before any mutation"


def test_a_node_group_labelled_for_another_site_is_refused() -> None:
    with pytest.raises(BootstrapError, match="name another site") as refusal:
        _discover(EksRunner(ownership="us-east-1-other-site-00000000"))
    message = str(refusal.value)
    assert (
        f"{GPU_NODEGROUP} carries {OWNERSHIP}=us-east-1-other-site-00000000" in message
    )
    assert f"this site is {OWNERSHIP}={SITE_ID}" in message


def test_a_labelled_node_group_outside_the_gpu_inventory_is_refused() -> None:
    with pytest.raises(
        BootstrapError, match="outside the GPU instance inventory"
    ) as refusal:
        _discover(EksRunner(cpu_ownership=SITE_ID))
    message = str(refusal.value)
    assert f"{CPU_NODEGROUP} runs m7i-flex.xlarge" in message, (
        "the node group and its instance types are named"
    )
    assert "p5en.48xlarge" in message, "the inventory is listed"


def test_the_label_is_not_consulted_without_a_site_id() -> None:
    with pytest.raises(BootstrapError, match="needs the site id"):
        _discover(EksRunner(), site_id=None)


def test_an_already_managed_cluster_keeps_its_recorded_selector() -> None:
    """A legacy entry (the retired --gpu-nodegroup flag) is the runtime truth:
    discovery of a managed cluster never re-derives membership, even when the
    node groups are labelled now -- that is node-scope --converge."""

    runner = EksRunner()
    recorded = f"eks.amazonaws.com/nodegroup={GPU_NODEGROUP}"
    entries = (
        {
            "clusterId": EKS_NAME,
            "provider": "eks-managed",
            "eksClusterArn": EKS_ARN,
            "nodeSelector": recorded,
        },
    )

    identity = _discover(runner, existing_clusters=entries)

    assert identity.node_selector == recorded, "the recorded selector is kept verbatim"
    assert identity.node_selector_source == "flag-legacy", (
        "a selector recorded by the retired flags says so until converged"
    )
    assert identity.subnet_cidrs == ("10.0.0.0/20", "10.0.16.0/20"), (
        "CIDRs follow the node group the recorded selector names"
    )
    unlabelled = _discover(
        EksRunner(ownership=None),
        existing_clusters=(
            {
                "cluster_id": EKS_NAME,
                "provider": "eks-managed",
                "eks_cluster_arn": EKS_ARN,
                "node_selector": "nvidia.com/gpu.present=true",
            },
        ),
    )
    assert unlabelled.node_selector == "nvidia.com/gpu.present=true", (
        "a legacy site without any label keeps deploying with its recorded default"
    )
    assert unlabelled.subnet_cidrs == ("10.0.0.0/20", "10.0.16.0/20"), (
        "the retired default meant the GPU inventory node groups"
    )


def test_a_recorded_selector_covering_no_node_group_fails_closed() -> None:
    with pytest.raises(BootstrapError, match="covers none of its managed node groups"):
        _discover(
            EksRunner(),
            existing_clusters=(
                {
                    "clusterId": EKS_NAME,
                    "provider": "eks-managed",
                    "eksClusterArn": EKS_ARN,
                    "nodeSelector": "eks.amazonaws.com/nodegroup=missing",
                },
            ),
        )


def test_eks_managed_discovery_refuses_inactive_cluster_or_nodegroup() -> None:
    with pytest.raises(BootstrapError, match="not ACTIVE"):
        _discover(EksRunner(status="UPDATING"))
    with pytest.raises(BootstrapError, match="is not ACTIVE"):
        _discover(EksRunner(nodegroup_status="DEGRADED"))


def test_eks_managed_discovery_refuses_node_auto_repair_like_node_recovery() -> None:
    """The plain-EKS counterpart of NodeRecovery=None: a GPU node group that
    repairs its own nodes is refused before anything is recorded; an explicit
    false is recorded as disabled."""

    with pytest.raises(BootstrapError, match="has node auto repair enabled") as refusal:
        _discover(EksRunner(node_repair=True))
    assert GPU_NODEGROUP in str(refusal.value) and "NodeRecovery=None" in str(
        refusal.value
    ), "name the node group and the HyperPod rule it mirrors"
    identity = _discover(EksRunner(node_repair=False))
    assert identity.node_repair == "disabled"


def test_eks_managed_discovery_refuses_a_cpu_role() -> None:
    with pytest.raises(BootstrapError, match="CPU control plane must be a HyperPod"):
        admin_bootstrap.discover_cluster(
            EksRunner(), cluster_arn=EKS_ARN, role="cpu", context="cpu"
        )


def test_identity_only_discovery_needs_no_site_id_and_resolves_no_scope() -> None:
    """remove-cluster and rotate-token only need the ARNs and names."""

    identity = admin_bootstrap.discover_cluster(
        EksRunner(ownership=None),
        cluster_arn=EKS_ARN,
        role="gpu",
        context="gpu-eks",
        resolve_node_scope=False,
    )
    assert identity.eks_arn == EKS_ARN and identity.is_eks_managed
    assert (identity.node_selector, identity.node_selector_source) == ("", ""), (
        "an identity-only read records no node scope"
    )


def _orchestrated_hyperpod(runner, *, groups):
    hyperpod_arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-gpu-eks"
    runner.hyperpods.append(
        {
            "ClusterArn": hyperpod_arn,
            "ClusterName": "hp-gpu-eks",
            "NodeRecovery": "None",
            "Orchestrator": {"Eks": {"ClusterArn": EKS_ARN}},
            "InstanceGroups": groups,
        }
    )


def _instance_group(name, instance_type, labels=None, desired=None):
    group = {"InstanceGroupName": name, "InstanceType": instance_type}
    kubernetes = {}
    if labels is not None:
        kubernetes["CurrentLabels"] = labels
    if desired is not None:
        kubernetes["DesiredLabels"] = desired
    if kubernetes:
        group["KubernetesConfig"] = kubernetes
    return group


def test_hyperpod_discovery_selects_the_labelled_instance_groups() -> None:
    runner = EksRunner()
    _orchestrated_hyperpod(
        runner,
        groups=[
            _instance_group("spot-p5en", "ml.p5en.48xlarge", {OWNERSHIP: SITE_ID}),
            _instance_group("accelerated-g6", "ml.g6.12xlarge"),
        ],
    )

    identity = _discover(runner)

    assert identity.provider == "hyperpod", "an orchestrated EKS ARN stays HyperPod"
    assert identity.hyperpod_name == "hp-gpu-eks", "the HyperPod name is recorded"
    assert identity.node_selector == (
        f"{OWNERSHIP}={SITE_ID},sagemaker.amazonaws.com/cluster-name=hp-gpu-eks"
    ), "the selector is the ownership label cross-checked with the cluster label"
    assert identity.node_selector_source == "label-convention"
    assert not any(
        operation in {"list-nodegroups", "describe-nodegroup"}
        for _s, operation, _a in runner.calls
    ), "HyperPod discovery must not read EKS node groups"


def test_hyperpod_cluster_without_a_labelled_instance_group_is_refused() -> None:
    runner = EksRunner()
    _orchestrated_hyperpod(
        runner,
        groups=[
            _instance_group(
                "spot-p5en", "ml.p5en.48xlarge", labels={}, desired={OWNERSHIP: SITE_ID}
            ),
            _instance_group("accelerated-g6", "ml.g6.12xlarge"),
        ],
    )

    with pytest.raises(BootstrapError) as refusal:
        _discover(runner)

    message = str(refusal.value)
    assert (
        f"HyperPod cluster hp-gpu-eks has no instance group labelled {OWNERSHIP}={SITE_ID}"
        in message
    )
    assert (
        f"spot-p5en instanceTypes=ml.p5en.48xlarge {OWNERSHIP}=absent "
        f"(desired {SITE_ID}, not applied yet)"
    ) in message, "a label update-cluster is still applying is reported as pending"
    assert f"accelerated-g6 instanceTypes=ml.g6.12xlarge {OWNERSHIP}=absent" in message
    assert "aws sagemaker update-cluster --cluster-name hp-gpu-eks" in message
    assert "KubernetesConfig" in message


def test_hyperpod_instance_group_refusals_name_the_group() -> None:
    foreign = EksRunner()
    _orchestrated_hyperpod(
        foreign,
        groups=[
            _instance_group("spot-p5en", "ml.p5en.48xlarge", {OWNERSHIP: "elsewhere"})
        ],
    )
    with pytest.raises(BootstrapError, match="spot-p5en carries .*=elsewhere"):
        _discover(foreign)
    non_gpu = EksRunner()
    _orchestrated_hyperpod(
        non_gpu,
        groups=[
            _instance_group("spot-p5en", "ml.p5en.48xlarge", {OWNERSHIP: SITE_ID}),
            _instance_group("accelerated-g6", "ml.g6.12xlarge", {OWNERSHIP: SITE_ID}),
        ],
    )
    with pytest.raises(BootstrapError, match="accelerated-g6 runs ml.g6.12xlarge"):
        _discover(non_gpu)


def test_a_managed_hyperpod_cluster_keeps_the_historical_selector() -> None:
    """The HyperPod path of every site deployed before the convention: the
    entry records no selector, the cluster-name label stays the runtime truth
    whatever the instance groups are labelled today."""

    runner = EksRunner()
    _orchestrated_hyperpod(
        runner,
        groups=[_instance_group("spot-p5en", "ml.p5en.48xlarge", {OWNERSHIP: SITE_ID})],
    )
    entries = ({"clusterId": "hp-gpu-eks", "hyperpodClusterName": "hp-gpu-eks"},)

    identity = _discover(runner, existing_clusters=entries)

    assert identity.node_selector == "", "no selector is recorded for a legacy entry"
    assert identity.node_selector_source == "provider-default-legacy"
    assert node_selector(identity) == (
        "sagemaker.amazonaws.com/cluster-name=hp-gpu-eks"
    ), "the HyperPod selector text is byte-identical to the historical label"


def test_bootstrap_request_carries_no_selector_inputs() -> None:
    request = BootstrapRequest(
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        gpu_cluster_arns=(EKS_ARN,),
        repository_root=__import__("pathlib").Path("/repo"),
        state_dir=__import__("pathlib").Path("/state"),
    )
    assert not hasattr(request, "node_selector") and not hasattr(
        request, "gpu_nodegroup"
    ), "the node scope is declared on the node groups, not passed to deploy"
    assert not hasattr(JoinClusterRequest, "requested_node_selector"), (
        "join-cluster takes no selector either"
    )


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
    assert actions == {
        "ec2:DescribeInstances",
        "ec2:RebootInstances",
        "eks:DescribeCluster",
        "eks:DescribeNodegroup",
    }, "an eks-managed executor holds EC2 describe/reboot and EKS describe reads"
    describe_eks = next(
        item for item in document["Statement"] if item["Sid"] == "DescribeEksCluster"
    )
    assert describe_eks["Resource"] == [
        identity.eks_arn,
        f"arn:aws:eks:{REGION}:{ACCOUNT}:nodegroup/gpu-e/*/*",
    ], "EKS reads are scoped to this cluster and its node groups"
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
    """The join world rebuilt as a plain EKS site: ``gpu-a`` is eks-managed and
    the join target ``gpu-e`` is a second plain EKS cluster.

    A site's GPU clusters must share one provider (the Runtime Profile is
    site-wide), so the existing HyperPod ``gpu-a`` of the shared fixture is
    rewritten as eks-managed before any join runs.
    """

    EKS_CONTEXTS = frozenset({"gpu-a", "gpu-e", "gpu-f"})

    def __init__(self, tmp_path, monkeypatch):
        super().__init__(tmp_path, monkeypatch, suffixes=("b",))
        document = yaml.safe_load(self.path.read_text())
        (existing,) = document["spec"]["clusters"]
        existing.pop("hyperpodClusterName")
        existing["provider"] = "eks-managed"
        existing["nodeSelector"] = "nvidia.com/gpu.present=true"
        self.path.write_text(yaml.safe_dump(document, sort_keys=False))
        self.site = load_site(self.path)
        self.targets[eks_target().eks_arn] = eks_target()
        self.node_names["a"] = "ip-10-0-32-11.us-west-2.compute.internal"
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
        if "get" in arguments and "nodes" in arguments and context in self.EKS_CONTEXTS:
            self.commands.append((list(arguments), dict(options)))
            name = self.node_names[context.rsplit("-", 1)[-1]]
            if any(str(item).startswith("go-template=") for item in arguments):
                return json.dumps([name])
            return json.dumps({"items": [_eks_node(name)]})
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
    sibling = next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == "gpu-a"
    )
    assert sibling["provider"] == "eks-managed", (
        "the existing plain EKS entry must be left as it was"
    )

    assert _node_selectors(scenario, "gpu-e") == {"nvidia.com/gpu.present=true"}, (
        "every node read of the plain EKS cluster must use its selector"
    )
    assert _node_selectors(scenario, "gpu-a") == {"nvidia.com/gpu.present=true"}, (
        "the sibling plain EKS cluster is read on its own selector"
    )

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
    # A false HyperPod-name match would refuse the second cluster as "already
    # managed" during discovery; instead it joins as its own member.
    result = join.join_cluster(request, runner=scenario)
    assert (result["phase"], result["cluster_id"]) == ("COMPLETED", "gpu-f"), (
        "a second plain EKS cluster must be a new, distinct join target"
    )
    assert scenario.cluster_states == {
        "gpu-a": "ACTIVE",
        "gpu-e": "ACTIVE",
        "gpu-f": "ACTIVE",
    }, "both plain EKS clusters must be registered side by side"


def test_a_first_deploy_still_requires_one_provider() -> None:
    """A first deploy writes one site profile; the other provider joins later
    with its own reviewed profile version (see the approval tests below)."""

    require_single_provider(["hyperpod", None, ""])
    require_single_provider(["eks-managed"], joining="eks-managed")
    with pytest.raises(BootstrapError, match="join the others with join-cluster"):
        require_single_provider(["hyperpod"], joining="eks-managed")


def test_hyperpod_join_still_renders_the_historical_selector(
    tmp_path, monkeypatch
) -> None:
    """The HyperPod path is the fully validated one and must not move."""

    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",))
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


# --- the site profile must be able to reboot the provider being joined ----------

HYPERPOD_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
)
EKS_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "config/runtime-profile.regional-eks-safe.example.yaml"
)


def test_owner_map_matches_the_example_profiles_and_the_adapters() -> None:
    from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
    from gpu_fault.eks_provider import EKS_ADAPTER_OWNER

    assert (
        profile_node_reboot_owner(HYPERPOD_PROFILE)
        == (PROVIDER_NODE_REBOOT_OWNERS["hyperpod"])
    ), "the HyperPod example profile must name the HyperPod adapter"
    assert (
        profile_node_reboot_owner(EKS_PROFILE)
        == (PROVIDER_NODE_REBOOT_OWNERS["eks-managed"])
    ), "the EKS example profile must name the EKS adapter"
    assert PROVIDER_NODE_REBOOT_OWNERS["eks-managed"] == EKS_ADAPTER_OWNER, (
        "the shared owner map must agree with the runtime EKS adapter"
    )
    assert (
        inspect.signature(HyperPodLifecycleStepAdapter.__init__)
        .parameters["owner"]
        .default
        == PROVIDER_NODE_REBOOT_OWNERS["hyperpod"]
    ), "the shared owner map must agree with the HyperPod adapter default"


@pytest.mark.parametrize(
    ("profile", "provider", "other_owner"),
    [
        (HYPERPOD_PROFILE, "eks-managed", "gpu-fault-hyperpod-adapter"),
        (EKS_PROFILE, "hyperpod", "gpu-fault-eks-adapter"),
    ],
)
def test_profile_owned_by_the_other_adapter_is_refused_with_the_remedy(
    profile, provider, other_owner
) -> None:
    with pytest.raises(BootstrapError) as failure:
        require_profile_matches_provider(profile, version="site-v1", provider=provider)
    message = str(failure.value)
    assert "site-v1" in message and other_owner in message, message
    assert PROVIDER_NODE_REBOOT_OWNERS[provider] in message, "names the expected owner"
    assert "--approve-profile-plan" in message and "new version name" in message, (
        "the remedy must name the profile switch and the approval flow"
    )
    assert PROVIDER_PROFILE_SOURCES[provider] in message, "names the matching profile"


def test_profile_matching_the_provider_or_without_a_reboot_owner_passes(
    tmp_path,
) -> None:
    require_profile_matches_provider(
        EKS_PROFILE, version="eks-v1", provider="eks-managed"
    )
    require_profile_matches_provider(
        HYPERPOD_PROFILE, version="hyperpod-v1", provider="hyperpod"
    )
    observe_only = tmp_path / "observe.yaml"
    observe_only.write_text(
        "claims:\n  - capability: nodeReboot\n    mode: OBSERVE\n"
        "    owner: gpu-fault-hyperpod-adapter\n",
        encoding="utf-8",
    )
    assert profile_node_reboot_owner(observe_only) is None, "OBSERVE is no owner"
    require_profile_matches_provider(observe_only, version="o", provider="eks-managed")
    no_claims = tmp_path / "empty.yaml"
    no_claims.write_text("claims: []\n", encoding="utf-8")
    require_profile_matches_provider(no_claims, version="e", provider="hyperpod")
    with pytest.raises(BootstrapError, match="cannot read runtime profile"):
        require_profile_matches_provider(
            tmp_path / "missing.yaml", version="m", provider="hyperpod"
        )


def _install_example_profiles(scenario) -> None:
    """Copy both example profiles into the scenario's repository root."""

    root = Path(yaml.safe_load(scenario.path.read_text())["spec"]["repositoryRoot"])
    (root / "config").mkdir(exist_ok=True)
    for source in (HYPERPOD_PROFILE, EKS_PROFILE):
        (root / "config" / source.name).write_text(source.read_text(encoding="utf-8"))


def _use_site_profile(scenario, name: str, version: str) -> None:
    document = yaml.safe_load(scenario.path.read_text())
    document["spec"]["runtimeProfile"]["source"] = f"config/{name}"
    document["spec"]["runtimeProfile"]["templateSource"] = f"config/{name}"
    document["spec"]["runtimeProfile"]["version"] = version
    scenario.path.write_text(yaml.safe_dump(document, sort_keys=False))


def _approve_pending_plan(state_dir: Path, plan_sha256: str) -> dict:
    return profile_approval.approve_profile(
        state_dir,
        reference="CHG-eks-1",
        expected_plan_sha256=plan_sha256,
        approver_identity="tester@fixture",
    )


def test_eks_cluster_joining_a_hyperpod_site_gets_its_own_profile_after_review(
    tmp_path, monkeypatch
) -> None:
    """Exactly the live case: a hyperpod-v1 site joining gpu-eks-cluster."""

    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",))
    _install_example_profiles(scenario)
    _use_site_profile(scenario, HYPERPOD_PROFILE.name, "hyperpod-v1")
    scenario.targets[eks_target().eks_arn] = eks_target()
    scenario.node_names["e"] = GPU_NODE
    monkeypatch.setattr(
        cluster_join_profile,
        "compute_agent_config_digest",
        lambda _runner, *, repository_root, runtime_profile_version: "e" * 64,
    )
    original_run = scenario.run

    def run(arguments, **options):
        context = (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )
        if "get" in arguments and "nodes" in arguments and context == "gpu-e":
            scenario.commands.append((list(arguments), dict(options)))
            if any(str(item).startswith("go-template=") for item in arguments):
                return json.dumps([GPU_NODE])
            return json.dumps({"items": [_eks_node()]})
        return original_run(arguments, **options)

    monkeypatch.setattr(scenario, "run", run)
    request = JoinClusterRequest(
        site=load_site(scenario.path), gpu_cluster_arn=eks_target().eks_arn
    )
    state_dir = scenario.path.parent

    # 1. The join stops for review: the EKS cluster needs eks-v1, a new version.
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        join.join_cluster(request, runner=scenario)
    plan = json.loads(
        profile_approval.profile_plan_path(state_dir).read_text(encoding="utf-8")
    )
    assert plan["change_kind"] == "CLUSTER_PROFILE_NEW", plan
    assert (plan["current_version"], plan["desired_version"]) == (
        "hyperpod-v1",
        "eks-v1",
    )
    assert plan["registration_cluster_id"] == "gpu-e", "anchored on the joining cluster"
    assert plan["approval_required"] is True, "a new version always needs review"
    assert plan["plan_sha256"] in str(stop.value), "the stop names the plan digest"
    assert "--approve-profile-plan" in str(stop.value), "and the rerun flag"
    assert not any(event.startswith("executor-role") for event in scenario.events), (
        "the stop must come before any AWS mutation"
    )
    assert not (state_dir / "profiles" / "eks-v1.yaml").exists(), (
        "no snapshot is written before the approval"
    )

    # 2. A rerun without approval stops again with the same plan.
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        join.join_cluster(request, runner=scenario)

    # 3. Approve (what join-cluster --approve-profile-plan SHA --reference does)
    #    and rerun: the join completes with the cluster's own profile.
    _approve_pending_plan(state_dir, plan["plan_sha256"])
    result = join.join_cluster(request, runner=scenario)
    assert result["phase"] == "COMPLETED", result
    snapshot = state_dir / "profiles" / "eks-v1.yaml"
    assert snapshot.is_file(), "the approved profile snapshot is written"
    snapshot_document = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert snapshot_document["profile_version"] == "eks-v1", "snapshot version"
    assert snapshot_document["cluster_id"] == "gpu-e", "snapshot anchor"
    document = yaml.safe_load(scenario.path.read_text())
    entry = next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == "gpu-e"
    )
    assert entry["runtimeProfile"] == {
        "source": str(snapshot),
        "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
        "version": "eks-v1",
        "agentConfigDigest": "e" * 64,
    }, "the cluster entry pins its own profile and Agent config digest"
    assert "runtimeProfile" not in document["spec"]["clusters"][0], (
        "the HyperPod cluster stays on the site profile"
    )
    assert not profile_approval.profile_plan_path(state_dir).exists(), (
        "the approved plan is consumed"
    )
    consumed = list(
        (state_dir / "release-deploy" / "profile-approvals").glob("*/consumed.json")
    )
    assert len(consumed) == 1, "the approval is archived as consumed"
    assert json.loads(consumed[0].read_text())["release_id"] == "join-gpu-e", (
        "the consumption names the join"
    )
    site = load_site(scenario.path)
    eks_entry = next(
        item
        for item in site.release_config["clusters"]
        if item["cluster_id"] == "gpu-e"
    )
    assert eks_entry["runtime_profile"]["version"] == "eks-v1", (
        "rendered for the release"
    )


def test_a_second_cluster_reuses_an_approved_profile_snapshot(
    tmp_path, monkeypatch
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",))
    _install_example_profiles(scenario)
    _use_site_profile(scenario, HYPERPOD_PROFILE.name, "hyperpod-v1")
    site = load_site(scenario.path)
    override = cluster_join_profile.cluster_profile_override(
        site, eks_target(), "gpu-e"
    )
    assert override is not None and override.version == "eks-v1", "override needed"
    state_dir = scenario.path.parent
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        cluster_join_profile.ensure_cluster_profile_approved(
            site, override, rerun_command="gpu-fault-admin join-cluster ..."
        )
    plan = json.loads(profile_approval.profile_plan_path(state_dir).read_text())
    _approve_pending_plan(state_dir, plan["plan_sha256"])
    snapshot = cluster_join_profile.ensure_cluster_profile_approved(
        site, override, rerun_command="x"
    )
    assert snapshot.is_file(), "approval writes the snapshot"
    second = cluster_join_profile.cluster_profile_override(
        site, replace(eks_target(), eks_name="gpu-f"), "gpu-f"
    )
    assert second is not None, "the second plain EKS cluster needs the same version"
    assert (
        cluster_join_profile.ensure_cluster_profile_approved(
            site, second, rerun_command="x"
        )
        == snapshot
    ), "an approved version is reused without a second review"
    assert not profile_approval.profile_plan_path(state_dir).exists(), "no new plan"


def test_hyperpod_cluster_joining_an_eks_site_needs_the_hyperpod_profile(
    tmp_path, monkeypatch
) -> None:
    scenario = EksJoinScenario(tmp_path, monkeypatch)
    _install_example_profiles(scenario)
    _use_site_profile(scenario, EKS_PROFILE.name, "eks-v1")
    site = load_site(scenario.path)
    override = cluster_join_profile.cluster_profile_override(
        site, target("b"), "hp-gpu-b"
    )
    assert override is not None, "a HyperPod cluster cannot run the EKS profile"
    assert (override.version, override.owner) == (
        "hyperpod-v1",
        "gpu-fault-hyperpod-adapter",
    ), "it gets the HyperPod example profile under its declared version"
    assert (
        cluster_join_profile.cluster_profile_override(site, eks_target(), "gpu-e")
        is None
    ), "a plain EKS cluster on an EKS site needs no override"


def test_join_accepts_a_matching_profile(tmp_path, monkeypatch) -> None:
    scenario = EksJoinScenario(tmp_path, monkeypatch)
    _install_example_profiles(scenario)
    _use_site_profile(scenario, EKS_PROFILE.name, "eks-v1")
    assert scenario.join_eks()["phase"] == "COMPLETED", "a matching profile joins"
    document = yaml.safe_load(scenario.path.read_text())
    entry = next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == "gpu-e"
    )
    assert "runtimeProfile" not in entry, "no override when the site profile fits"
    assert not profile_approval.profile_plan_path(scenario.path.parent).exists(), (
        "nothing to review"
    )


# --- allowed workload namespaces must exist on the joining cluster ---------------


def test_join_refuses_early_when_an_allowed_namespace_is_missing(
    tmp_path, monkeypatch
) -> None:
    """Trial 11 (2026-10-04): gpu-eks-cluster had no `training` namespace; the
    release's server dry-run of the per-namespace Role failed deep inside the
    join-cluster mode and rolled everything back. The join now refuses before
    it creates anything on the cluster, naming the namespace and the remedy."""

    scenario = EksJoinScenario(tmp_path, monkeypatch)
    scenario.missing_namespaces = {"training"}
    with pytest.raises(BootstrapError) as refusal:
        scenario.join_eks()
    message = str(refusal.value)
    assert "allowed workload namespace(s) do not exist on the cluster: training" in (
        message
    ), message
    assert "--allowed-namespace" in message and "create namespace" in message, (
        "the remedy names both options"
    )
    assert not [event for event in scenario.events if event.startswith("namespace:")], (
        "nothing was created on the cluster before the refusal"
    )
    assert not any(event.startswith("executor-role") for event in scenario.events), (
        "no AWS mutation either"
    )


def test_join_persists_the_requested_allowed_namespaces_without_forcing_training(
    tmp_path, monkeypatch
) -> None:
    """`--allowed-namespace ml` joins a cluster that has `ml` but no `training`:
    the cluster entry lists the control-plane namespace plus the requested ones."""

    scenario = EksJoinScenario(tmp_path, monkeypatch)
    scenario.missing_namespaces = {"training"}
    request = JoinClusterRequest(
        site=load_site(scenario.path),
        gpu_cluster_arn=eks_target().eks_arn,
        allowed_namespaces=("ml",),
    )
    result = join.join_cluster(request, runner=scenario)
    assert result["phase"] == "COMPLETED", result
    document = yaml.safe_load(scenario.path.read_text())
    entry = next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == "gpu-e"
    )
    assert entry["allowedNamespaces"] == ["gpu-fault-system", "ml"], (
        "the historical default `training` is not forced onto the cluster"
    )
    probed = [
        call[call.index("namespace") + 1]
        for call, _options in scenario.commands
        if "get" in call and "namespace" in call
    ]
    assert "ml" in probed and "training" not in probed, (
        "only the requested workload namespaces are checked for existence"
    )


def test_cli_join_defaults_allowed_namespaces_to_the_sites_list(
    tmp_path, monkeypatch
) -> None:
    from gpu_fault.admin.cluster_join_types import site_allowed_namespaces

    # The scenario patches join's drivers and discovery; a bare
    # pytest.MonkeyPatch() here was never undone and the custody join tests
    # that ran after this one saw the fakes ("join rollout preceded custody
    # completion").
    site = load_site(JoinScenario(tmp_path, monkeypatch, suffixes=("b",)).path)
    assert site_allowed_namespaces(site) == tuple(
        sorted(site.release_config["clusters"][0]["allowed_namespaces"])
    ), "a joining cluster inherits the list the site's clusters share"


# --- deploy keeps, refreshes or restores a cluster's own profile ---------------


def _deploy_document(tmp_path, *, eks_profile):
    """A generated site document: HyperPod site profile, one HyperPod cluster,
    one plain EKS cluster with (or without) its override."""

    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True, exist_ok=True)
    for source in (HYPERPOD_PROFILE, EKS_PROFILE):
        (repo / "config" / source.name).write_text(source.read_text(encoding="utf-8"))
    eks = {
        "clusterId": "gpu-eks-cluster",
        "provider": "eks-managed",
        "eksClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-eks-cluster",
    }
    if eks_profile is not None:
        eks["runtimeProfile"] = eks_profile
    return repo, {
        "spec": {
            "runtimeProfile": {"source": str(repo / "config" / HYPERPOD_PROFILE.name)},
            "clusters": [{"clusterId": "hp-a"}, eks],
        }
    }


def _approved_snapshot(state_dir, repo):
    snapshot = state_dir / "profiles" / "eks-v1.yaml"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_load((repo / "config" / EKS_PROFILE.name).read_text())
    document["cluster_id"] = "gpu-eks-cluster"
    document["profile_version"] = "eks-v1"
    snapshot.write_text(yaml.safe_dump(document, sort_keys=False))
    return snapshot


def test_deploy_restores_a_lost_override_from_the_approved_snapshot(
    tmp_path, monkeypatch
) -> None:
    """The live site after the 10:44 deploy: the plain EKS entry had lost its
    runtimeProfile. The approved snapshot still exists, so the deploy restores
    the override and computes this release's Agent config digest."""

    repo, document = _deploy_document(tmp_path, eks_profile=None)
    state_dir = tmp_path / "state"
    snapshot = _approved_snapshot(state_dir, repo)
    monkeypatch.setattr(
        cluster_join_profile,
        "compute_agent_config_digest",
        lambda _runner, *, repository_root, runtime_profile_version: (
            runtime_profile_version[:1] * 64
        ),
    )
    cluster_join_profile.reconcile_cluster_runtime_profiles(
        object(), document, repository_root=repo, state_dir=state_dir
    )
    hyperpod, eks = document["spec"]["clusters"]
    assert "runtimeProfile" not in hyperpod, "the HyperPod cluster follows the site"
    assert eks["runtimeProfile"] == {
        "source": str(snapshot),
        "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
        "version": "eks-v1",
        "agentConfigDigest": "e" * 64,
    }


def test_deploy_keeps_an_existing_override_and_refreshes_its_digest(
    tmp_path, monkeypatch
) -> None:
    repo, document = _deploy_document(
        tmp_path,
        eks_profile={
            "source": "/elsewhere/eks-v1.yaml",
            "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
            "version": "eks-v1",
            "agentConfigDigest": "1" * 64,  # computed by the join's release
        },
    )
    state_dir = tmp_path / "state"
    snapshot = _approved_snapshot(state_dir, repo)
    monkeypatch.setattr(
        cluster_join_profile,
        "compute_agent_config_digest",
        lambda _runner, *, repository_root, runtime_profile_version: "2" * 64,
    )
    cluster_join_profile.reconcile_cluster_runtime_profiles(
        object(), document, repository_root=repo, state_dir=state_dir
    )
    eks = document["spec"]["clusters"][1]["runtimeProfile"]
    assert eks["version"] == "eks-v1" and eks["source"] == str(snapshot)
    assert eks["agentConfigDigest"] == "2" * 64, (
        "the digest follows this release, as the site-level one does"
    )


def test_deploy_reviews_a_needed_profile_that_was_never_approved(tmp_path) -> None:
    """A member without its own profile and without an approved snapshot (a
    batch join completed before batches planned profiles) is planned by the
    deploy for explicit review -- nothing is pinned or written before the
    approval -- instead of a refusal naming a join that has nothing to join."""

    repo, document = _deploy_document(tmp_path, eks_profile=None)
    document["metadata"] = {"name": "staging"}
    document["spec"]["awsRegion"] = "us-west-2"
    document["spec"]["cpu"] = {
        "eksArn": "arn:aws:eks:us-west-2:123456789012:cluster/control-plane"
    }
    document["spec"]["runtimeProfile"]["version"] = "hyperpod-v1"
    state_dir = tmp_path / "state"
    with pytest.raises(
        cluster_join_profile.JoinProfileApprovalRequired,
        match="the deploy stopped for review",
    ):
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), document, repository_root=repo, state_dir=state_dir
        )
    plan = json.loads(profile_approval.profile_plan_path(state_dir).read_text())
    assert (plan["registration_cluster_id"], plan["desired_version"]) == (
        "gpu-eks-cluster",
        "eks-v1",
    ), plan
    assert plan["current_version"] == "hyperpod-v1", "reviewed against the site"
    assert "runtimeProfile" not in document["spec"]["clusters"][1], (
        "nothing is pinned before the approval"
    )
    assert not (state_dir / "profiles" / "eks-v1.yaml").exists(), (
        "no snapshot before the approval"
    )


def test_deploy_fails_closed_when_a_pinned_version_lost_its_snapshot(tmp_path) -> None:
    repo, document = _deploy_document(
        tmp_path,
        eks_profile={
            "source": "/elsewhere/eks-v1.yaml",
            "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
            "version": "eks-v1",
            "agentConfigDigest": "1" * 64,
        },
    )
    with pytest.raises(BootstrapError, match="needs Runtime Profile eks-v1.*approve"):
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), document, repository_root=repo, state_dir=tmp_path / "state"
        )
