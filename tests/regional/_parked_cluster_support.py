"""A site with one GPU cluster parked at zero nodes, for the parked-cluster tests.

The shapes are the live ones: the eks-managed cluster's only GPU managed node
group labelled for the site and scaled to min 0 / max 2 / desired 0, its CPU
node group beside it, and three Karpenter NodePools that carry no ownership
label (``cpu``, ``gpu``, ``neuron``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

from gpu_fault.cluster_scope import node_selector
from tests.regional._cov95_release_support import ResourceRelease

SITE = "us-west-2-site-0123abcd"
OWNERSHIP = f"gpu-fault.io/control-plane={SITE}"
EKS = "gpu-eks-cluster"
HYPERPOD = "gpu-a"
EKS_ARN = f"arn:aws:eks:us-east-1:123456789012:cluster/{EKS}"
GROUP = "gpu-efa-spot-node-group"
CPU_GROUP = "cpu-node-group"


def nodegroup(
    name: str = GROUP,
    *,
    desired: int = 0,
    status: str = "ACTIVE",
    labelled: bool = True,
) -> dict[str, Any]:
    """A ``describe-nodegroup`` document of a GPU managed node group."""

    labels = {
        "nvidia.com/gpu.present": "true",
        "vpc.amazonaws.com/efa.present": "true",
        "gpu-fault.io/dcgm-hostengine": "host",
        "node-type": "gpu",
        "instance-family": "p5en",
    }
    if labelled:
        labels["gpu-fault.io/control-plane"] = SITE
    return {
        "nodegroupName": name,
        "status": status,
        "instanceTypes": ["p5en.48xlarge"],
        "labels": labels,
        "scalingConfig": {"minSize": 0, "maxSize": 2, "desiredSize": desired},
    }


def cpu_nodegroup() -> dict[str, Any]:
    return {
        "nodegroupName": CPU_GROUP,
        "status": "ACTIVE",
        "instanceTypes": ["m7i.xlarge"],
        "labels": {"node-type": "cpu"},
        "scalingConfig": {"minSize": 2, "maxSize": 4, "desiredSize": 2},
    }


def nodepool(
    name: str,
    *,
    labels: dict[str, str] | None = None,
    requirements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {"name": name},
        "spec": {
            "template": {
                "metadata": {
                    "labels": {"launch-type": "karpenter", "node-class": name}
                    if labels is None
                    else labels
                },
                "spec": {"requirements": requirements or []},
            }
        },
    }


#: The live pools: no ownership label in a template or a requirement.
LIVE_NODEPOOLS = {
    "items": [
        nodepool(
            "cpu",
            requirements=[
                {
                    "key": "karpenter.sh/capacity-type",
                    "operator": "In",
                    "values": ["spot"],
                },
                {"key": "kubernetes.io/arch", "operator": "In", "values": ["amd64"]},
                {
                    "key": "karpenter.k8s.aws/instance-category",
                    "operator": "In",
                    "values": ["c", "m"],
                },
            ],
        ),
        nodepool(
            "gpu",
            requirements=[
                {
                    "key": "karpenter.sh/capacity-type",
                    "operator": "In",
                    "values": ["spot"],
                },
                {
                    "key": "karpenter.k8s.aws/instance-family",
                    "operator": "In",
                    "values": ["g6e"],
                },
                {
                    "key": "karpenter.k8s.aws/instance-size",
                    "operator": "In",
                    "values": ["xlarge"],
                },
            ],
        ),
        nodepool("neuron"),
    ]
}


def node(name: str, labels: dict[str, str], *, ready: bool = True) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "labels": labels,
            "annotations": {"gpu-fault.io/installer-state": "Succeeded"},
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "addresses": [{"type": "InternalIP", "address": "10.0.1.10"}],
        },
    }


def agent(cluster_id: str, node_id: str, **fields: Any) -> dict[str, Any]:
    """One ``active_agent_identities`` record."""

    return {
        "cluster_id": cluster_id,
        "node_id": node_id,
        "agent_protocol_version": 4,
        "agent_version": "0.10.0",
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "a" * 64,
        "installer_bundle_sha256": "b" * 64,
        "installer_template_sha256": "c" * 64,
        "policy_version": "policy-v1",
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "d" * 64,
        "node_action_key_version": 1,
        **fields,
    }


class ProviderRunner:
    """Answers the provider reads of the parked proof and records them."""

    dry_run = False

    def __init__(
        self,
        nodegroups: dict[str, dict[str, Any]],
        nodepools: dict[str, Any] | None | Exception,
    ) -> None:
        self.nodegroups = nodegroups
        self.nodepools = nodepools
        self.calls: list[list[str]] = []

    def run(self, arguments: list[str], **_kwargs: Any) -> str:
        self.calls.append(list(arguments))
        if arguments[:3] == ["aws", "eks", "list-nodegroups"]:
            return json.dumps({"nodegroups": sorted(self.nodegroups)})
        if arguments[:3] == ["aws", "eks", "describe-nodegroup"]:
            name = arguments[arguments.index("--nodegroup-name") + 1]
            return json.dumps(self.nodegroups[name])
        if arguments[:3] == ["aws", "eks", "describe-cluster"]:
            return json.dumps({"Arn": EKS_ARN, "Status": "ACTIVE"})
        raise AssertionError(f"unexpected provider read {arguments[:4]}")

    def probe_output(
        self, arguments: list[str], *, timeout_seconds: float
    ) -> tuple[int, str, str]:
        self.calls.append(list(arguments))
        assert "nodepools.karpenter.sh" in arguments, arguments
        assert timeout_seconds > 0, "the NodePool read is bounded"
        if isinstance(self.nodepools, Exception):
            return 1, "", str(self.nodepools)
        if self.nodepools is None:
            return 1, "", 'error: the server doesn\'t have a resource type "nodepools"'
        return 0, json.dumps(self.nodepools), ""


class ParkedSite(ResourceRelease):
    """``gpu-a`` (HyperPod) with one node and its Agent; ``gpu-eks-cluster``
    (eks-managed) with no node, its GPU node group at ``desired``."""

    def __init__(
        self,
        *,
        desired: int = 0,
        nodepools: dict[str, Any] | None | Exception = LIVE_NODEPOOLS,
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__((HYPERPOD, EKS))
        hyperpod, eks = self.config.clusters
        self.config.clusters = (
            hyperpod,
            replace(
                eks,
                provider="eks-managed",
                hyperpod_cluster_name="",
                node_selector=OWNERSHIP,
                eks_cluster_arn=EKS_ARN,
            ),
        )
        self.config.site_name = SITE
        self.provider = ProviderRunner(
            {GROUP: nodegroup(desired=desired), CPU_GROUP: cpu_nodegroup()}, nodepools
        )
        self.runner = self.provider  # type: ignore[assignment]
        hyperpod_selector = node_selector(hyperpod)
        key, _, value = hyperpod_selector.partition("=")
        self.documents[(HYPERPOD, "nodes", "-l " + hyperpod_selector)] = {
            "items": [node("node-a", {key: value})]
        }
        self.documents[(EKS, "nodes", "-l " + OWNERSHIP)] = {"items": []}
        self.records = (
            [agent(HYPERPOD, "node-a")] if records is None else copy.deepcopy(records)
        )
        self.saved: list[tuple[str, dict[str, Any]]] = []

    @property
    def eks(self) -> Any:
        return self.config.clusters[1]

    def _target_node_names(self, target: Any) -> tuple[str, ...]:
        names = tuple(
            sorted(
                str(item["metadata"]["name"])
                for item in self._get_json(
                    self._gpu(target, "get", "nodes", "-l", node_selector(target))
                )["items"]
            )
        )
        assert names, "the engine never asks a parked cluster for its fleet nodes"
        return names

    def _capture_previous(self, plan: Any = None) -> dict[str, Any]:
        from gpu_fault_release import regional_release_agent_set as agent_set

        # The wiring of ``regional_release_state._capture_previous`` (pinned by
        # its own test): a plan-less capture is sync-state's.
        return {
            "live_runtime_image": self.runtime_image,
            "agent_identities": agent_set.capture_agent_identities_matching_nodes(
                self, accept_parked=plan is None
            ),
        }

    def _save_state(self, phase: str, **values: Any) -> None:
        self.saved.append((phase, values))
