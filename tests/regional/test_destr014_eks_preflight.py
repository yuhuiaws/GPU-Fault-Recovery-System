"""DESTR-014's preflight on a plain-EKS cluster.

Live (gpu-eks-cluster): both nodes were Ready and uncordoned and carried only
the node group's standing ``nvidia.com/gpu=present:NoSchedule`` taint, and the
sibling Agent recorded the EC2 instance id behind the Node's providerID; the
preflight refused all three as isolation and identity faults. The taint rule is
the shared per-provider one and the identity is what the provider's installer
records.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import run_destr014_branch_exhaustion as destr014
from tests.regional.test_destr014_branch_exhaustion import (
    FAULT,
    SIBLING,
    _happy_preflight_kwargs,
    _warm_node,
)


def _eks_node(name: str, instance_id: str) -> dict[str, Any]:
    """The warm-spare view of a plain-EKS node: the node group's standing GPU
    taint, the group label, the EC2 providerID, no HyperPod labels."""

    return {
        "name": name,
        "uid": f"uid-{name}",
        "ready": "True",
        "gpu_allocatable": 8,
        "unschedulable": False,
        "provider_id": f"aws:///us-west-2c/{instance_id}",
        "taints": [
            {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
        ],
        "labels": {
            "gpu-fault.io/spare": None,
            "eks.amazonaws.com/nodegroup": "gpu-p5en-efa-spot-node-group",
            "node.kubernetes.io/instance-type": "p5en.48xlarge",
        },
        "annotations": {},
    }


def _eks_preflight_kwargs() -> dict[str, Any]:
    kwargs = _happy_preflight_kwargs()
    kwargs["fault"] = _eks_node(FAULT, "i-00000000000000001")
    kwargs["sibling"] = _eks_node(SIBLING, "i-00000000000000002")
    kwargs["sibling_agent"]["node_instance_id"] = "i-00000000000000002"
    kwargs["cluster"] = {
        "provider": "eks-managed",
        "status": "ACTIVE",
        "node_recovery": "not-applicable",
    }
    return kwargs


def test_eks_nodes_pass_the_preflight_with_their_standing_gpu_taint() -> None:
    """Live (gpu-eks-cluster): both nodes were Ready and uncordoned, carried only
    ``nvidia.com/gpu=present:NoSchedule``, and the sibling Agent recorded the EC2
    instance id; the preflight refused all three as isolation/identity faults."""

    assert destr014.preflight_errors(**_eks_preflight_kwargs()) == []


@pytest.mark.parametrize(
    "taint",
    [
        {"key": "gpu-fault.io/quarantined", "value": "x", "effect": "NoSchedule"},
        {"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule"},
        {"key": "vendor.example/maintenance", "effect": "NoExecute"},
    ],
)
def test_eks_nodes_are_still_refused_for_isolating_taints(
    taint: dict[str, str],
) -> None:
    kwargs = _eks_preflight_kwargs()
    kwargs["sibling"]["taints"].append(taint)
    errors = destr014.preflight_errors(**kwargs)
    assert f"{SIBLING} is not Ready and schedulable" in errors, errors


def test_a_hyperpod_node_still_refuses_any_taint() -> None:
    kwargs = _happy_preflight_kwargs()
    kwargs["fault"]["taints"] = [
        {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
    ]
    errors = destr014.preflight_errors(**kwargs)
    assert f"{FAULT} is not Ready and schedulable" in errors, (
        "a HyperPod node carries no taint in steady state"
    )


def test_the_eks_sibling_identity_is_the_ec2_instance_behind_the_node() -> None:
    from scripts.e2e.regional import destr014_preflight

    assert destr014_preflight.agent_node_identity(
        _eks_node(SIBLING, "i-00000000000000002")
    ) == ("i-00000000000000002")
    assert destr014_preflight.agent_node_identity(_warm_node(SIBLING)) == (
        f"uid-{SIBLING}"
    )
    kwargs = _eks_preflight_kwargs()
    kwargs["sibling_agent"]["node_instance_id"] = f"uid-{SIBLING}"
    errors = destr014.preflight_errors(**kwargs)
    assert any(
        item.startswith("sibling Node Agent recovery identity is incomplete")
        and "is not the node's 'i-00000000000000002'" in item
        for item in errors
    ), errors
    kwargs = _eks_preflight_kwargs()
    kwargs["sibling"]["provider_id"] = None
    errors = destr014.preflight_errors(**kwargs)
    assert (
        "sibling Node Agent recovery identity is incomplete or mismatched" in errors
    ), "an eks node without a providerID has no identity to match -- fail-closed"
