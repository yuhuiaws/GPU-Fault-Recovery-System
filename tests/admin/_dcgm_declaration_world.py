"""A plain EKS cluster whose node groups declare a DCGM host engine mode.

Shared by the declaration tests of ``status``, ``node-scope`` and the deploy's
NOOP shortcut: ``NodeGroupWorld`` answers the provider reads bootstrap
discovery makes for the cluster, ``eks_site`` writes the shared test site plus
one ``eks-managed`` entry with a recorded mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.cluster_scope import DCGM_HOSTENGINE_LABEL, OWNERSHIP_LABEL
from tests.admin.test_admin_site import site_file

REGION = "us-east-1"
ACCOUNT = "123456789012"
EKS_NAME = "gpu-eks-cluster"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{EKS_NAME}"
#: The shared test site's ``metadata.name``, the ownership label's value.
SITE_ID = "test-site"
LABEL_SELECTOR = f"{OWNERSHIP_LABEL}={SITE_ID}"
CPU_GROUP = "cpu-node-group"
CIDRS = {"subnet-gpu-a": "10.0.0.0/20", "subnet-gpu-b": "10.0.16.0/20"}
NODE_GROUP_READS = ("list-nodegroups", "describe-nodegroup")


class NodeGroupWorld:
    """``aws`` answers for one plain EKS cluster with a CPU and N GPU node groups.

    ``groups`` maps each GPU node group to its ``gpu-fault.io/dcgm-hostengine``
    value (``None``: absent); every GPU group carries the site's ownership
    label unless named in ``unowned``. ``calls`` records ``(service,
    operation, node group)``; with ``failure`` every call raises it.
    """

    def __init__(
        self,
        groups: dict[str, str | None],
        *,
        unowned: tuple[str, ...] = (),
        failure: Exception | None = None,
    ) -> None:
        self.groups = groups
        self.unowned = unowned
        self.failure = failure
        self.calls: list[tuple[str, str, str]] = []

    def _labels(self, name: str) -> dict[str, str]:
        labels = {"node-type": "gpu"}
        if name not in self.unowned:
            labels[OWNERSHIP_LABEL] = SITE_ID
        value = self.groups[name]
        if value is not None:
            labels[DCGM_HOSTENGINE_LABEL] = value
        return labels

    def node_group_calls(self) -> list[tuple[str, str, str]]:
        return [item for item in self.calls if item[1] in NODE_GROUP_READS]

    def aws_json(
        self, region: str, service: str, operation: str, *arguments: str, **_options
    ) -> dict[str, Any]:
        name = (
            arguments[arguments.index("--nodegroup-name") + 1]
            if "--nodegroup-name" in arguments
            else ""
        )
        self.calls.append((service, operation, name))
        if self.failure is not None:
            raise self.failure
        assert region == REGION, "the read crossed the Region boundary"
        if (service, operation) == ("sagemaker", "list-clusters"):
            return {"ClusterSummaries": []}
        if (service, operation) == ("eks", "describe-cluster"):
            return {
                "cluster": {
                    "arn": EKS_ARN,
                    "status": "ACTIVE",
                    "resourcesVpcConfig": {
                        "vpcId": "vpc-gpu-eks",
                        "subnetIds": ["subnet-cp-a"],
                    },
                }
            }
        if (service, operation) == ("eks", "list-nodegroups"):
            return {"nodegroups": [CPU_GROUP, *self.groups]}
        if (service, operation) == ("eks", "describe-nodegroup"):
            if name == CPU_GROUP:
                return {
                    "nodegroup": {
                        "nodegroupName": name,
                        "status": "ACTIVE",
                        "instanceTypes": ["m7i-flex.xlarge"],
                        "subnets": ["subnet-cpu-a"],
                        "labels": {"node-type": "cpu"},
                    }
                }
            return {
                "nodegroup": {
                    "nodegroupName": name,
                    "status": "ACTIVE",
                    "instanceTypes": ["p5en.48xlarge"],
                    "subnets": ["subnet-gpu-a", "subnet-gpu-b"],
                    "labels": self._labels(name),
                }
            }
        if (service, operation) == ("ec2", "describe-subnets"):
            ids = arguments[arguments.index("--subnet-ids") + 1 :]
            return {
                "Subnets": [
                    {"SubnetId": item, "CidrBlock": CIDRS[item]} for item in ids
                ]
            }
        raise AssertionError(f"unexpected provider call {service}/{operation}")


def unreadable_world() -> NodeGroupWorld:
    return NodeGroupWorld(
        {"gpu-a": None},
        failure=BootstrapError("command failed (254): aws eks list-nodegroups"),
    )


def eks_entry(recorded: str | None, *, selector: str = LABEL_SELECTOR) -> dict:
    """The ``eks-managed`` entry fields; ``recorded`` None is an entry written
    before the mode existed (read as embedded/default)."""

    entry: dict[str, Any] = {
        "clusterId": EKS_NAME,
        "context": EKS_NAME,
        "provider": "eks-managed",
        "nodeSelector": selector,
        "nodeSelectorSource": "label-convention",
        "eksClusterArn": EKS_ARN,
    }
    if recorded is not None:
        entry["dcgmHostEngine"] = recorded
        entry["dcgmHostEngineSource"] = (
            "node-group-label" if recorded == "host" else "default"
        )
    return entry


def eks_site(tmp_path: Path, recorded: str | None = "embedded", **overrides) -> Path:
    """The shared test site (one HyperPod cluster) plus the plain EKS cluster."""

    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = dict(document["spec"]["clusters"][0])
    entry.pop("hyperpodClusterName", None)
    entry.update(eks_entry(recorded), **overrides)
    document["spec"]["clusters"].append(entry)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def eks_site_entry(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return next(
        item for item in document["spec"]["clusters"] if item["clusterId"] == EKS_NAME
    )
