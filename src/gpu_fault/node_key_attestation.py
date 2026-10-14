"""The EC2 identity a plain EKS node brings when it asks for a Node Action key.

``gpu-fault-admin deploy``/``join-cluster`` key the nodes a GPU cluster has at
that time (``deploy/node/provision_node_action_keys.py``); a node that joins
later is keyed by the executor's node-key sync
(``cluster_executor/node_key_sync.py``), which asks
``POST /v1/regional/node-action-keys`` for the names its Secret lacks. The
control plane derives a key only for a node it can place in the authenticated
cluster: a HyperPod node by the SageMaker identity it lists itself, a node that
ran the product before by its Agent record or telemetry. A plain EKS
replacement has neither: the control plane holds no EC2 grant, and every
node-level report comes from software the installer puts on the node -- an
installer that cannot start without the key. Live: a spot replacement's
installer Pod stayed in ``FailedMount ... references non-existent secret key``
while the sync was refused with 404 every minute, until the next deploy.

So the executor, whose role already reads EC2 and EKS for the reboot path,
states what the instance behind the Kubernetes Node is
(``EksNodeAttestation``), and the control plane checks the statement against
what it registered for the cluster (``eks_attestation_refusal``): the EKS
cluster, the node subnets, the instance's own name and address. That trusts the
executor no further than before -- the same cluster token already reports node
telemetry, which is evidence too -- but a statement that does not bind to the
registration keys nothing.

Only the control plane and the executor import this module; the node runtime
does not, so a change here rolls no node Agent.
"""

from __future__ import annotations

import ipaddress
import re

from pydantic import Field, field_validator, model_validator

from gpu_fault.cluster_scope import ClusterProvider, cluster_provider
from gpu_fault.fleet_endpoint import node_name_address, parse_endpoint_networks
from gpu_fault.models import StrictModel
from gpu_fault.regional import (
    NODE_ACTION_KEY_REQUEST_MAX_NODES,
    RegionalClusterRegistration,
    RemoteNodeActionKeyRequest,
)

#: A Kubernetes Node name (an RFC 1123 subdomain): the node-key Secret is keyed
#: by it, and the deploy-host helper keys exactly these names.
NODE_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")
#: An EC2 instance id as ``spec.providerID`` and ``DescribeInstances`` spell it.
INSTANCE_ID_PATTERN = re.compile(r"^i-[0-9a-f]{8,17}$")
#: The EC2 state of the instance behind a live node.
INSTANCE_RUNNING = "running"
_NODE_UID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EKS_CLUSTER_ARN = re.compile(
    r"^arn:aws(?:-[a-z]+)*:eks:[a-z0-9-]+:[0-9]{12}:cluster/"
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)$"
)


class EksNodeAttestation(StrictModel):
    """One plain-EKS node as Kubernetes and EC2 describe it to the executor.

    ``node_id``/``node_uid`` are the Kubernetes Node's name and UID,
    ``instance_id`` is the id in its ``spec.providerID``; the rest is what
    ``DescribeInstances`` answered for that id: the instance state, its private
    DNS name and address, and the ``eks:cluster-name``/``eks:nodegroup-name``
    tags EKS puts on every managed node group instance. No key material.
    """

    node_id: str
    node_uid: str
    instance_id: str
    instance_state: str = Field(min_length=1, max_length=32)
    private_dns_name: str = Field(min_length=1, max_length=253)
    private_ip: str
    eks_cluster_name: str = Field(min_length=1, max_length=100)
    nodegroup: str = Field(min_length=1, max_length=63)

    @field_validator("node_id")
    @classmethod
    def _validate_node_id(cls, value: str) -> str:
        if NODE_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("node_id must be a Kubernetes Node name")
        return value

    @field_validator("node_uid")
    @classmethod
    def _validate_node_uid(cls, value: str) -> str:
        if _NODE_UID_PATTERN.fullmatch(value) is None:
            raise ValueError("node_uid must be a Kubernetes object UID")
        return value

    @field_validator("instance_id")
    @classmethod
    def _validate_instance_id(cls, value: str) -> str:
        if INSTANCE_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("instance_id must be an EC2 instance id")
        return value

    @field_validator("private_ip")
    @classmethod
    def _validate_private_ip(cls, value: str) -> str:
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            raise ValueError("private_ip must be an IP address") from None


class AttestedNodeActionKeyRequest(RemoteNodeActionKeyRequest):
    """A key request that may carry the EC2 identity of the nodes it names.

    ``nodes`` is left out of the body by every executor with nothing to attest,
    so a control plane that predates the field keeps answering the plain
    request: the request models forbid unknown fields.
    """

    nodes: list[EksNodeAttestation] = Field(
        default_factory=list, max_length=NODE_ACTION_KEY_REQUEST_MAX_NODES
    )

    @model_validator(mode="after")
    def _attest_requested_nodes_once(self) -> AttestedNodeActionKeyRequest:
        names = [item.node_id for item in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError("a node is attested at most once")
        if not set(names) <= set(self.node_ids):
            raise ValueError("node attestations must name requested node_ids")
        return self


def _private_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
    )


def eks_attestation_refusal(
    registration: RegionalClusterRegistration, attestation: EksNodeAttestation
) -> str | None:
    """Why ``attestation`` does not place its node in the registered cluster.

    ``None`` when every binding holds: the cluster is ``eks-managed`` (a
    HyperPod node is placed by its SageMaker identity, never by a statement);
    the instance is running and tagged for the registered EKS cluster; the
    Kubernetes Node is named after the instance's private DNS name; and the
    instance address -- the one the name encodes, when it encodes one -- is a
    private address inside the cluster's registered node subnets, the networks
    an Agent of this cluster must advertise its endpoint in anyway.
    """

    if cluster_provider(registration.provider) is not ClusterProvider.EKS_MANAGED:
        return (
            "only a node of an eks-managed cluster is admitted by its EC2 "
            "identity; a HyperPod node is placed by its SageMaker identity"
        )
    registered = _EKS_CLUSTER_ARN.fullmatch(registration.eks_cluster_arn)
    if registered is None or attestation.eks_cluster_name != registered.group("name"):
        return (
            f"instance {attestation.instance_id} is tagged for EKS cluster "
            f"{attestation.eks_cluster_name}, not the registered one"
        )
    if attestation.instance_state != INSTANCE_RUNNING:
        return (
            f"instance {attestation.instance_id} is "
            f"{attestation.instance_state}, not {INSTANCE_RUNNING}"
        )
    if attestation.private_dns_name.lower() != attestation.node_id:
        return (
            f"node {attestation.node_id} is not named after its instance's "
            f"private DNS name {attestation.private_dns_name}"
        )
    address = ipaddress.ip_address(attestation.private_ip)
    encoded = node_name_address(attestation.node_id)
    if encoded is not None and encoded != str(address):
        return f"node name {attestation.node_id} encodes another address than {address}"
    networks = parse_endpoint_networks(
        ",".join(registration.agent_endpoint_allowed_cidrs)
    )
    if not _private_address(address) or not any(
        address in network for network in networks
    ):
        return (
            f"instance address {address} is outside the cluster's registered "
            "node subnets"
        )
    return None
