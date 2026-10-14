"""The executor's EC2 statement about a plain-EKS node it wants a key for.

``EksNodeEvidence`` reads, with the executor's own grants (``ec2:DescribeInstances``
and ``eks:DescribeNodegroup``, held for the reboot path), the instance behind each
Kubernetes Node's ``spec.providerID`` and the managed node group that launched
it. It attests a node only when every binding holds; anything else is a refusal
with a reason and no request to the control plane at all:

* an EC2 provider id and an InternalIP on the Node;
* an instance that exists, runs, is tagged for this EKS cluster and carries the
  Node's own name and address;
* a managed node group that is the Node's own and that declares the site's
  ownership label -- the declaration a kubelet cannot forge on itself.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from gpu_fault.eks_node_evidence import EksNodeEvidence
from tests.eks_provider._support import (
    EKS_CLUSTER,
    NODEGROUP,
    FakeEc2,
    FakeEks,
    ec2_instance,
    gpu_node,
)

SITE = "us-west-2-hp-cluster-0a1b2c3d"
SELECTOR = f"{OWNERSHIP_LABEL}={SITE}"
NODE = "ip-10-0-36-21.us-west-2.compute.internal"
INSTANCE = "i-0b000021"
ADDRESS = "10.0.36.21"


def _node(
    name: str = NODE,
    instance_id: str = INSTANCE,
    address: str | None = ADDRESS,
    **overrides: Any,
) -> dict[str, Any]:
    node = gpu_node(name, instance_id, labels={OWNERSHIP_LABEL: SITE}, **overrides)
    node["status"]["addresses"] = (
        [{"type": "InternalIP", "address": address}] if address else []
    )
    return node


def _instance(
    instance_id: str = INSTANCE,
    name: str = NODE,
    address: str = ADDRESS,
    **overrides: Any,
) -> dict[str, Any]:
    instance = ec2_instance(instance_id, private_dns=name, **overrides)
    instance["PrivateIpAddress"] = address
    return instance


def _owned_groups(**groups: dict[str, str]) -> FakeEks:
    declared = groups or {NODEGROUP: {OWNERSHIP_LABEL: SITE}}
    return FakeEks(
        nodegroups={
            name: {"status": "ACTIVE", "labels": labels}
            for name, labels in declared.items()
        }
    )


def _evidence(ec2: FakeEc2, eks: FakeEks) -> EksNodeEvidence:
    return EksNodeEvidence(
        eks_cluster_name=EKS_CLUSTER,
        node_selector=SELECTOR,
        ec2_client=ec2,
        eks_client=eks,
    )


def test_a_node_of_an_owned_managed_node_group_is_attested() -> None:
    ec2, eks = FakeEc2([_instance()]), _owned_groups()

    attestations, refusals = _evidence(ec2, eks).attest([_node()])

    assert refusals == {}
    attestation = attestations[NODE]
    assert attestation.model_dump() == {
        "node_id": NODE,
        "node_uid": f"{NODE}-uid",
        "instance_id": INSTANCE,
        "instance_state": "running",
        "private_dns_name": NODE,
        "private_ip": ADDRESS,
        "eks_cluster_name": EKS_CLUSTER,
        "nodegroup": NODEGROUP,
    }
    assert ec2.describe_calls == [
        {"Filters": [{"Name": "instance-id", "Values": [INSTANCE]}]}
    ], "instances are read by filter: a terminated, purged id is absent, not an error"
    assert eks.described_nodegroups == [(EKS_CLUSTER, NODEGROUP)]


@pytest.mark.parametrize(
    ("node", "instances", "groups", "reason"),
    [
        (_node(provider_id=None), [_instance()], None, "no EC2 providerID"),
        (_node(address=None), [_instance()], None, "no InternalIP"),
        (_node(), [], None, "no EC2 instance"),
        (_node(), [_instance(cluster="another-eks")], None, "tagged for EKS cluster"),
        (_node(), [_instance(state="stopped")], None, "is stopped"),
        (
            _node(),
            [_instance(name="ip-10-0-36-22.us-west-2.compute.internal")],
            None,
            "private DNS name",
        ),
        (_node(), [_instance(address="10.0.36.22")], None, "private address"),
        (_node(), [_instance(nodegroup=None)], None, "managed node group"),
        (_node(nodegroup="cpu-node-group-1"), [_instance()], None, "node group label"),
        (_node(), [_instance()], {NODEGROUP: {}}, "does not declare"),
        (
            _node(),
            [_instance()],
            {NODEGROUP: {OWNERSHIP_LABEL: "another-site"}},
            "does not declare",
        ),
    ],
)
def test_a_node_that_does_not_bind_is_refused_with_a_reason(
    node: dict[str, Any],
    instances: list[dict[str, Any]],
    groups: dict[str, dict[str, str]] | None,
    reason: str,
) -> None:
    eks = _owned_groups(**groups) if groups is not None else _owned_groups()

    attestations, refusals = _evidence(FakeEc2(instances), eks).attest([node])

    assert attestations == {}
    assert reason in refusals[NODE], refusals


def test_one_node_group_is_described_once_per_pass() -> None:
    other = "ip-10-0-40-17.us-west-2.compute.internal"
    ec2 = FakeEc2([_instance(), _instance("i-0b000022", other, "10.0.40.17")])
    eks = _owned_groups()

    attestations, refusals = _evidence(ec2, eks).attest(
        [_node(), _node(other, "i-0b000022", "10.0.40.17")]
    )

    assert refusals == {}
    assert sorted(attestations) == [NODE, other]
    assert eks.described_nodegroups == [(EKS_CLUSTER, NODEGROUP)]


def test_a_malformed_node_identity_is_refused_not_raised() -> None:
    node = _node()
    node["metadata"]["uid"] = "not a uid"

    attestations, refusals = _evidence(FakeEc2([_instance()]), _owned_groups()).attest(
        [node]
    )

    assert attestations == {}
    assert "malformed" in refusals[NODE]


def test_an_ec2_failure_fails_the_pass() -> None:
    class DeniedEc2(FakeEc2):
        def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("AccessDenied")

    with pytest.raises(RuntimeError, match="AccessDenied"):
        _evidence(DeniedEc2([]), _owned_groups()).attest([_node()])


@pytest.mark.parametrize(
    ("cluster", "selector", "message"),
    [
        ("", SELECTOR, "EKS cluster name"),
        (EKS_CLUSTER, "nvidia.com/gpu.present=true", "ownership label"),
        (EKS_CLUSTER, "eks.amazonaws.com/nodegroup=gpu-ng", "ownership label"),
    ],
)
def test_an_attester_that_cannot_bind_a_node_is_never_built(
    cluster: str, selector: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EksNodeEvidence(
            eks_cluster_name=cluster,
            node_selector=selector,
            ec2_client=FakeEc2([]),
            eks_client=FakeEks(),
        )
