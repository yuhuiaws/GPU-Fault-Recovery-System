"""The executor's page and the deploy host's ``status`` read one verdict.

Both judge a GPU cluster whose selector selects no node with
``gpu_fault.zero_node_capacity``, but each reads the provider its own way: the
deploy host through the AWS CLI (``describe-cluster`` projected by
``HYPERPOD_CONTEXT_QUERY``) and ``kubectl get nodepools.karpenter.sh``, the
executor through boto3 (the full API documents) and its Kubernetes client
(``karpenter.sh/v1``). Fed the same provider state, the two must reach the same
proof -- state, reason and groups -- or the alert and ``status`` would disagree
about the same cluster; a read that fails on either side is unavailable with
the one unreadable reason.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault import zero_node_capacity as capacity
from gpu_fault.cluster_executor.zero_node_probe import ZeroNodeCapacityProbe
from gpu_fault.cluster_scope import ClusterProvider
from gpu_fault_release import regional_parked_cluster as parked_cluster
from tests.execution.test_cluster_executor_zero_node_probe import (
    ApiError,
    Core,
    Custom,
    Eks,
    SageMaker,
    instance_group,
)
from tests.regional._parked_cluster_support import (
    CPU_GROUP,
    EKS_ARN,
    GROUP,
    LIVE_NODEPOOLS,
    OWNERSHIP,
    SITE,
    cpu_nodegroup,
    nodegroup,
    nodepool,
)

NO_CRD = 'error: the server doesn\'t have a resource type "nodepools"'
#: The instance-group fields ``HYPERPOD_CONTEXT_QUERY`` projects.
PROJECTED = (
    "InstanceGroupName",
    "InstanceType",
    "CurrentCount",
    "TargetCount",
    "Status",
    "KubernetesConfig",
    "ScheduledUpdateConfig",
    "AutoPatchConfig",
)


class Cli:
    """The deploy host's provider reads, answered from one provider state."""

    def __init__(
        self,
        *,
        nodegroups: dict[str, Any] | Exception,
        nodepools: dict[str, Any] | Exception | None,
        groups: list[dict[str, Any]] | Exception,
    ) -> None:
        self.nodegroups = nodegroups
        self.nodepools = nodepools
        self.groups = groups

    def run(self, command: list[str], **_options: Any) -> str:
        if command[:3] == ["aws", "eks", "list-nodegroups"]:
            if isinstance(self.nodegroups, Exception):
                raise self.nodegroups
            return json.dumps({"nodegroups": sorted(self.nodegroups)})
        if command[:3] == ["aws", "eks", "describe-nodegroup"]:
            assert isinstance(self.nodegroups, dict), (
                "a failing list is never described"
            )
            name = command[command.index("--nodegroup-name") + 1]
            return json.dumps(self.nodegroups[name])
        if command[:3] == ["aws", "sagemaker", "describe-cluster"]:
            assert command[command.index("--query") + 1] == (
                parked_cluster.HYPERPOD_CONTEXT_QUERY
            )
            if isinstance(self.groups, Exception):
                raise self.groups
            return json.dumps(
                {
                    "InstanceGroups": [
                        {key: group[key] for key in PROJECTED if key in group}
                        for group in self.groups
                    ]
                }
            )
        raise AssertionError(f"unexpected provider read {command[:4]}")

    def probe_output(
        self, command: list[str], *, timeout_seconds: float
    ) -> tuple[int, str, str]:
        assert "nodepools.karpenter.sh" in command, command
        if isinstance(self.nodepools, Exception):
            return 1, "", str(self.nodepools)
        if self.nodepools is None:
            return 1, "", NO_CRD
        return 0, json.dumps(self.nodepools), ""


def deploy_host(provider: str, state: dict[str, Any]) -> dict[str, Any]:
    target = SimpleNamespace(
        cluster_id="gpu-x",
        context="gpu-x",
        provider=provider,
        node_selector=OWNERSHIP if provider == "eks-managed" else "",
        hyperpod_cluster_name="hp-a" if provider == "hyperpod" else "",
        eks_cluster_arn=EKS_ARN,
        region="us-east-1",
    )
    release = SimpleNamespace(
        runner=Cli(**state),
        _gpu=lambda _target, *arguments: ["kubectl", "--context", "gpu-x", *arguments],
    )
    return parked_cluster.provider_proof(release, target)


def executor(provider: str, state: dict[str, Any]) -> dict[str, Any]:
    nodepools = state["nodepools"]
    probe = ZeroNodeCapacityProbe(
        Core([]),
        provider=ClusterProvider(provider),
        node_selector=(
            OWNERSHIP
            if provider == "eks-managed"
            else "sagemaker.amazonaws.com/cluster-name=hp-a"
        ),
        hyperpod_cluster="hp-a",
        eks_cluster="gpu-eks-cluster",
        custom=Custom(
            nodepools
            if isinstance(nodepools, Exception) or nodepools is None
            else {"v1": nodepools}
        ),
        eks_client=Eks(state["nodegroups"], page=1),
        sagemaker_client=SageMaker(state["groups"]),
    )
    return probe.proof()


def provider_state(**overrides: Any) -> dict[str, Any]:
    return {
        "nodegroups": {GROUP: nodegroup(), CPU_GROUP: cpu_nodegroup()},
        "nodepools": LIVE_NODEPOOLS,
        "groups": [instance_group()],
        **overrides,
    }


SAME_PROOF = {
    "eks-parked-live-nodepools": ("eks-managed", provider_state()),
    "eks-parked-no-karpenter": ("eks-managed", provider_state(nodepools=None)),
    "eks-group-wants-nodes": (
        "eks-managed",
        provider_state(nodegroups={GROUP: nodegroup(desired=2)}),
    ),
    "eks-group-updating": (
        "eks-managed",
        provider_state(nodegroups={GROUP: nodegroup(status="UPDATING")}),
    ),
    "eks-selector-covers-no-group": (
        "eks-managed",
        provider_state(nodegroups={CPU_GROUP: cpu_nodegroup()}),
    ),
    "eks-nodepool-supplies": (
        "eks-managed",
        provider_state(
            nodepools={
                "items": [
                    nodepool("gpu-owned", labels={"gpu-fault.io/control-plane": SITE})
                ]
            }
        ),
    ),
    "hyperpod-parked": ("hyperpod", provider_state()),
    "hyperpod-target-wants-instances": (
        "hyperpod",
        provider_state(groups=[instance_group(target=4)]),
    ),
    "hyperpod-updating": (
        "hyperpod",
        provider_state(groups=[instance_group(status="Updating")]),
    ),
    "hyperpod-still-has-instances": (
        "hyperpod",
        provider_state(groups=[instance_group(current=1)]),
    ),
}


@pytest.mark.parametrize(("provider", "state"), SAME_PROOF.values(), ids=SAME_PROOF)
def test_the_same_provider_state_reads_the_same_proof(
    provider: str, state: dict[str, Any]
) -> None:
    on_host = deploy_host(provider, state)
    on_executor = executor(provider, state)

    assert on_executor == on_host, (on_executor, on_host)


FAILED_READS = {
    "eks-node-groups-denied": (
        "eks-managed",
        provider_state(nodegroups=RuntimeError("AccessDenied")),
        capacity.UNREADABLE_REASON,
    ),
    "eks-nodepools-forbidden": (
        "eks-managed",
        provider_state(nodepools=ApiError(403)),
        "the Karpenter NodePools could not be read: ",
    ),
    "hyperpod-describe-fails": (
        "hyperpod",
        provider_state(groups=RuntimeError("ThrottlingException")),
        capacity.UNREADABLE_REASON,
    ),
}


@pytest.mark.parametrize(
    ("provider", "state", "prefix"), FAILED_READS.values(), ids=FAILED_READS
)
def test_a_failed_read_is_unavailable_on_both_sides_with_one_reason(
    provider: str, state: dict[str, Any], prefix: str
) -> None:
    """The error text is each reader's own; the verdict and its reason are not."""

    on_host = deploy_host(provider, state)
    on_executor = executor(provider, state)

    for proof in (on_host, on_executor):
        assert capacity.state_of(proof) == capacity.UNAVAILABLE, proof
        assert proof["reason"].startswith(prefix), proof["reason"]
    assert on_host["groups"] == on_executor["groups"], (on_host, on_executor)
