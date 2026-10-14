"""The EC2 facts behind a plain-EKS node the node-key sync wants a key for.

The control plane keys a node it can place in the cluster; a plain EKS node
that has never been installed gives it nothing to place it by (no Agent, no
telemetry, no provider inventory -- ``gpu_fault.node_key_attestation``). The
executor's role reads EC2 and EKS already (``ec2:DescribeInstances`` and
``eks:DescribeNodegroup``, granted for the reboot path by
``bootstrap_iam_policies.executor_policy_document``), so it states what the
instance behind each such Node is. A statement is built only when every
binding holds; otherwise the node gets a refusal reason and no request at all:

* the Node carries an EC2 ``spec.providerID`` and an InternalIP;
* ``DescribeInstances`` knows the instance, it is running, it is tagged for
  this EKS cluster, and its private DNS name and address are the Node's;
* the instance was launched by an EKS managed node group, the one the Node's
  own ``eks.amazonaws.com/nodegroup`` label names, and that node group declares
  the site's ownership label (``OWNERSHIP_LABEL``) -- the declaration a kubelet
  cannot forge on itself, unlike a label on its own Node.

So a node outside the site's node groups, one whose instance EC2 does not
know, or one a cluster's legacy selector cannot tie to the site, keeps waiting
for a deploy, exactly as before. Errors of the EC2/EKS calls themselves fail
the sync pass; the next one retries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    OWNERSHIP_LABEL,
    ec2_instance_id_from_provider_id,
    selector_ownership_value,
)
from gpu_fault.eks_provider import (
    DESCRIBE_INSTANCES_CHUNK,
    EKS_CLUSTER_TAG,
    EKS_NODEGROUP_TAG,
    Ec2Client,
    EksClient,
)
from gpu_fault.node_key_attestation import INSTANCE_RUNNING, EksNodeAttestation


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


@dataclass(frozen=True)
class _NodeFacts:
    name: str
    uid: str
    labels: Mapping[str, Any]
    instance_id: str
    address: str


def _internal_ip(node: Any) -> str:
    addresses = _value(_value(node, "status", {}) or {}, "addresses", []) or []
    for address in addresses:
        if _value(address, "type") == "InternalIP":
            return str(_value(address, "address") or "")
    return ""


def _tags(instance: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(tag.get("Key")): str(tag.get("Value", ""))
        for tag in instance.get("Tags") or []
        if isinstance(tag, dict) and tag.get("Key")
    }


class EksNodeEvidence:
    """Attests plain-EKS nodes from EC2 and EKS, with the executor's own grants."""

    def __init__(
        self,
        *,
        eks_cluster_name: str,
        node_selector: str,
        ec2_client: Ec2Client | None = None,
        eks_client: EksClient | None = None,
        region_name: str | None = None,
    ) -> None:
        cluster = eks_cluster_name.strip()
        if not cluster:
            raise ValueError(
                "the EKS cluster name is unset (GPU_FAULT_EKS_CLUSTER_NAME), so "
                "no instance can be tied to this cluster"
            )
        ownership = selector_ownership_value(node_selector)
        if ownership is None:
            raise ValueError(
                f"the node selector {node_selector!r} names no ownership label "
                f"({OWNERSHIP_LABEL}=<site id>), so no node group can be tied to "
                "the site; converge it with gpu-fault-admin node-scope --converge"
            )
        self.eks_cluster_name = cluster
        self.ownership = ownership
        self.region_name = region_name
        self._ec2 = ec2_client
        self._eks = eks_client

    @classmethod
    def from_environment(cls, node_selector: str) -> EksNodeEvidence:
        return cls(
            eks_cluster_name=os.getenv("GPU_FAULT_EKS_CLUSTER_NAME", ""),
            node_selector=node_selector,
            region_name=(
                os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
            ),
        )

    @staticmethod
    def _create_client(service: str, region_name: str | None) -> Any:
        import boto3

        return boto3.client(service, region_name=region_name)

    @property
    def ec2(self) -> Ec2Client:
        # Built on first use: a cluster whose replacements are all keyed by
        # deploy never needs the client, and a HyperPod process never builds one.
        if self._ec2 is None:
            self._ec2 = self._create_client("ec2", self.region_name)
        return self._ec2

    @property
    def eks(self) -> EksClient:
        if self._eks is None:
            self._eks = self._create_client("eks", self.region_name)
        return self._eks

    def attest(
        self, nodes: Sequence[Any]
    ) -> tuple[dict[str, EksNodeAttestation], dict[str, str]]:
        """``(attestations, refusals)`` for ``nodes``, both keyed by node name."""

        facts: list[_NodeFacts] = []
        refusals: dict[str, str] = {}
        for node in nodes:
            metadata = _value(node, "metadata", {}) or {}
            name = str(_value(metadata, "name", "") or "")
            spec = _value(node, "spec", {}) or {}
            provider_id = _value(spec, "providerID", None)
            if provider_id is None:
                provider_id = _value(spec, "provider_id", None)
            instance_id = ec2_instance_id_from_provider_id(provider_id)
            address = _internal_ip(node)
            if not instance_id:
                refusals[name] = "the node has no EC2 providerID (aws:///<zone>/i-...)"
            elif not address:
                refusals[name] = "the node reports no InternalIP yet"
            else:
                facts.append(
                    _NodeFacts(
                        name=name,
                        uid=str(_value(metadata, "uid", "") or ""),
                        labels=dict(_value(metadata, "labels", {}) or {}),
                        instance_id=instance_id,
                        address=address,
                    )
                )
        instances = self._describe_instances([item.instance_id for item in facts])
        declared: dict[str, str | None] = {}
        attestations: dict[str, EksNodeAttestation] = {}
        for item in facts:
            instance = instances.get(item.instance_id)
            if instance is None:
                refusals[item.name] = (
                    f"no EC2 instance {item.instance_id} answers for the node "
                    "(unknown, or terminated and gone)"
                )
                continue
            refusal = self._refusal(item, instance, declared)
            if refusal is not None:
                refusals[item.name] = refusal
                continue
            try:
                attestations[item.name] = EksNodeAttestation(
                    node_id=item.name,
                    node_uid=item.uid,
                    instance_id=item.instance_id,
                    instance_state=INSTANCE_RUNNING,
                    private_dns_name=str(instance.get("PrivateDnsName") or ""),
                    private_ip=item.address,
                    eks_cluster_name=self.eks_cluster_name,
                    nodegroup=_tags(instance)[EKS_NODEGROUP_TAG],
                )
            except ValueError:
                refusals[item.name] = "the node identity is malformed"
        return attestations, refusals

    def _refusal(
        self,
        item: _NodeFacts,
        instance: Mapping[str, Any],
        declared: dict[str, str | None],
    ) -> str | None:
        tags = _tags(instance)
        cluster = tags.get(EKS_CLUSTER_TAG, "")
        state = str((instance.get("State") or {}).get("Name") or "").lower()
        nodegroup = tags.get(EKS_NODEGROUP_TAG, "")
        if cluster != self.eks_cluster_name:
            return (
                f"instance {item.instance_id} is tagged for EKS cluster "
                f"{cluster or '<none>'}, not {self.eks_cluster_name}"
            )
        if state != INSTANCE_RUNNING:
            return f"instance {item.instance_id} is {state or 'in no state'}"
        if str(instance.get("PrivateDnsName") or "").lower() != item.name:
            return f"the node is not named after instance {item.instance_id}'s private DNS name"
        if str(instance.get("PrivateIpAddress") or "") != item.address:
            return f"the node's InternalIP is not instance {item.instance_id}'s private address"
        if not nodegroup:
            return (
                f"instance {item.instance_id} was not launched by an EKS managed "
                "node group"
            )
        if str(item.labels.get(EKS_NODEGROUP_LABEL) or "") != nodegroup:
            return (
                f"the node group label {EKS_NODEGROUP_LABEL} does not name the "
                f"instance's node group {nodegroup}"
            )
        if nodegroup not in declared:
            declared[nodegroup] = self._declared_ownership(nodegroup)
        if declared[nodegroup] != self.ownership:
            return (
                f"node group {nodegroup} does not declare "
                f"{OWNERSHIP_LABEL}={self.ownership}"
            )
        return None

    def _declared_ownership(self, nodegroup: str) -> str | None:
        response = self.eks.describe_nodegroup(
            clusterName=self.eks_cluster_name, nodegroupName=nodegroup
        )
        group = response.get("nodegroup") if isinstance(response, dict) else None
        labels = group.get("labels") if isinstance(group, dict) else None
        value = labels.get(OWNERSHIP_LABEL) if isinstance(labels, dict) else None
        return str(value) if value else None

    def _describe_instances(self, instance_ids: list[str]) -> dict[str, dict[str, Any]]:
        """``DescribeInstances`` by filter: a terminated, purged id is absent."""

        found: dict[str, dict[str, Any]] = {}
        unique = sorted(set(instance_ids))
        for start in range(0, len(unique), DESCRIBE_INSTANCES_CHUNK):
            chunk = unique[start : start + DESCRIBE_INSTANCES_CHUNK]
            token: str | None = None
            while True:
                request: dict[str, Any] = {
                    "Filters": [{"Name": "instance-id", "Values": chunk}]
                }
                if token:
                    request["NextToken"] = token
                response = self.ec2.describe_instances(**request)
                for reservation in response.get("Reservations") or []:
                    for instance in reservation.get("Instances") or []:
                        instance_id = instance.get("InstanceId")
                        if isinstance(instance_id, str) and instance_id:
                            found[instance_id] = instance
                token = response.get("NextToken")
                if not token:
                    break
        return found
