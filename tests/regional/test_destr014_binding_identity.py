"""The DESTR-014 host recovery binding accepts the node's provider identity.

Live (eks-managed attempt 11): the preflight accepted both nodes by the
provider-aware rule, then ``make_binding`` compared the sibling Agent's
``node_instance_id`` (the EC2 instance id on eks) with the Kubernetes Node UID
and stopped the run with "Agent and Node UID differ".
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import destr014_recovery as recovery
from scripts.e2e.regional.destr014_preflight import agent_node_identity
from scripts.e2e.regional.regional_commands import RegionalFixtureError

INSTANCE = "i-00000000000000001"
UID = "4c3ab07e-0000-4000-8000-000000000014"
SCOPE: dict[str, Any] = {
    "run_id": "destr014-test-a1",
    "release_id": "release-test",
    "plan_sha256": "1" * 64,
    "cluster_id": "gpu-eks-cluster",
    "node": "ip-10-0-41-164.us-west-2.compute.internal",
    "node_uid": UID,
    "boot_id": "boot-sibling",
}


def _agent(node_instance_id: str | None) -> dict[str, Any]:
    return {
        "node_instance_id": node_instance_id,
        "artifact_sha256": "2" * 64,
        "installer_bundle_sha256": "3" * 64,
        "runtime_profile_version": "profile-v1",
    }


def _binding(agent: dict[str, Any], node_identity: str | None) -> dict[str, Any]:
    return recovery.make_binding(
        scope=SCOPE,
        owner="owner",
        agent=agent,
        restore_at=1_000,
        expires_at=1_000 + recovery.RECOVERY_SECONDS,
        node_identity=node_identity,
    )


def test_an_eks_agent_binds_by_the_ec2_instance_behind_the_node() -> None:
    node = {
        "labels": {"eks.amazonaws.com/nodegroup": "gpu-p5en-efa-spot-node-group"},
        "provider_id": f"aws:///us-west-2c/{INSTANCE}",
        "uid": UID,
    }
    identity = agent_node_identity(node)
    assert identity == INSTANCE
    binding = _binding(_agent(INSTANCE), identity)
    assert binding["node_uid"] == UID, "the host binding still keys on the Node UID"
    with pytest.raises(RegionalFixtureError, match="provider identity"):
        _binding(_agent(UID), identity)


def test_a_hyperpod_agent_still_binds_by_the_node_uid() -> None:
    node = {
        "labels": {"sagemaker.amazonaws.com/cluster-name": "hp"},
        "provider_id": f"aws:///us-west-2c/{INSTANCE}",
        "uid": UID,
    }
    identity = agent_node_identity(node)
    assert identity == UID
    assert _binding(_agent(UID), identity)["node_uid"] == UID
    with pytest.raises(RegionalFixtureError, match="provider identity"):
        _binding(_agent(INSTANCE), identity)


def test_an_unknown_node_identity_fails_closed() -> None:
    assert agent_node_identity({"labels": {"eks.amazonaws.com/nodegroup": "g"}}) is None
    with pytest.raises(RegionalFixtureError, match="expects None"):
        _binding(_agent(INSTANCE), None)


def test_the_plain_node_projection_never_yields_an_identity() -> None:
    """Live eks-managed attempt 12: the binding was fed ``regional.node_snapshot``
    (no labels, no providerID), the helper fell back to the Node UID and the
    execute path refused the Agent that correctly recorded the instance id."""
    projection = {
        "name": SCOPE["node"],
        "uid": UID,
        "boot_id": "boot-sibling",
        "ready": "True",
        "taints": [],
        "ownership_annotations": {},
    }
    assert agent_node_identity(projection) is None, (
        "a view that cannot tell its provider must not guess the Node UID"
    )
    warm_view = {
        **projection,
        "labels": {
            "eks.amazonaws.com/nodegroup": "gpu-p5en-efa-spot-node-group",
            "node.kubernetes.io/instance-type": "p5en.48xlarge",
        },
        "provider_id": f"aws:///us-west-2c/{INSTANCE}",
    }
    assert agent_node_identity(warm_view) == INSTANCE
    binding = _binding(_agent(INSTANCE), agent_node_identity(warm_view))
    assert binding["node_uid"] == UID
