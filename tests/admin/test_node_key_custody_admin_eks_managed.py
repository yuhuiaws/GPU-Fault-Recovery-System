"""Node-key custody on a plain EKS (eks-managed) cluster selects its own nodes.

Normal provisioning hands the key helper the provider scope -- the HyperPod
name, or the ``cluster_id`` of a plain EKS cluster, which has no other
provider-side name -- with the provider and the cluster's node selector
(``bootstrap_services.provision_node_action_keys``). The custody path handed it
only the HyperPod name, which is empty on plain EKS: the custody binding could
not even be built (its ``hyperpod_cluster`` must be non-empty), the helper fell
back to the HyperPod cluster-name label (no plain EKS node carries it), and a
rotation's activation target was a HyperPod target without a name.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import replace
from typing import Any, cast

import pytest

from gpu_fault.admin.node_key_custody_activation_io import CustodyActivationIO
from gpu_fault.admin.node_key_custody_admin import CustodyPreparationRequired
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import Chain
from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_CLUSTER_LABEL,
    OWNERSHIP_LABEL,
    node_selector,
)
from gpu_fault_release.regional_release_node_preflight import validate_target_node_state
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.admin.test_node_key_custody_admin import provision
from tests.admin.test_node_key_custody_admin_rotation import authorized_rotation
from tests.regional._cov95_identity_support import offline_guard as offline_guard

SELECTOR = f"{OWNERSHIP_LABEL}=custody-site"
CLUSTER_ID = "gpu"


def eks_managed(world: AdminWorld) -> AdminWorld:
    """The custody world as a plain EKS GPU cluster: no HyperPod name, the GPU
    node group labelled for the site, and a CPU node of the same cluster
    outside the selector."""

    world.gpu = replace(
        world.gpu,
        provider="eks-managed",
        hyperpod_arn="",
        hyperpod_name="",
        node_selector=SELECTOR,
        node_selector_source="label-convention",
    )
    world.cluster_id = CLUSTER_ID
    items = world.api.state["nodes"]["items"]
    for item in items:
        item["metadata"]["labels"] = {
            OWNERSHIP_LABEL: "custody-site",
            EKS_NODEGROUP_LABEL: "gpu-p5en",
        }
        item["status"] = {"conditions": [{"type": "Ready", "status": "True"}]}
    cpu_node = copy.deepcopy(items[0])
    cpu_node["metadata"].update(
        name="node-cpu", uid=str(uuid.uuid4()), labels={EKS_NODEGROUP_LABEL: "cpu-m5"}
    )
    items.append(cpu_node)
    return world


@pytest.fixture
def world(tmp_path, monkeypatch):
    return eks_managed(AdminWorld(tmp_path, monkeypatch))


def _node_reads(world: AdminWorld) -> set[str]:
    return {
        call[call.index("-l") + 1]
        for call in world.api.state["calls"]
        if "nodes" in call and "-l" in call
    }


def test_custody_install_on_an_eks_managed_cluster_selects_its_nodes(world) -> None:
    world.configure()
    world.release_ready = True
    world.access()
    with pytest.raises(CustodyPreparationRequired, match="awaits independent"):
        provision(world)
    chain_path = world.authorize()

    result = provision(world)

    binding = parse(Chain, chain_path.read_bytes()).transactions[0]
    site = binding.authorization.statement.binding.site
    nodes = binding.authorization.statement.binding.nodes
    assert set(nodes) == {"node-a", "node-b"}, "custody bound the CPU node"
    assert site.hyperpod_cluster == CLUSTER_ID, (
        "a plain EKS cluster's provider scope is its cluster_id"
    )
    assert set(world.api.state["secrets"]["gpu"]["data"]) == {"node-a", "node-b"}
    assert _node_reads(world) == {SELECTOR}, (
        "every node read must use the cluster's own selector, never "
        f"{HYPERPOD_CLUSTER_LABEL}"
    )
    assert result["cluster_id"] == CLUSTER_ID
    assert world.helper_calls == 1


def test_custody_rotation_on_an_eks_managed_cluster_rotates_one_node(
    tmp_path, monkeypatch
) -> None:
    world, _, _ = authorized_rotation(
        eks_managed(AdminWorld(tmp_path, monkeypatch)), monkeypatch
    )
    old = dict(world.api.state["secrets"]["gpu"]["data"])
    world.configure()

    result = provision(world)

    current = world.api.state["secrets"]["gpu"]["data"]
    assert current["node-a"] != old["node-a"], "node-a was not rotated"
    assert current["node-b"] == old["node-b"], "the sibling key changed"
    assert set(current) == {"node-a", "node-b"}
    assert result["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"
    assert _node_reads(world) == {SELECTOR}
    assert world.helper_calls == 2


def test_rotation_activation_targets_the_eks_managed_cluster_by_its_selector(
    tmp_path, monkeypatch
) -> None:
    """The activation's pre-mutation gate (and its convergence read) select the
    nodes through the target; a HyperPod target without a name selects none."""

    world, _, _ = authorized_rotation(
        eks_managed(AdminWorld(tmp_path, monkeypatch)), monkeypatch
    )
    authorization = world.activation_io.authorization

    activation = CustodyActivationIO(world, world.context(), authorization)

    assert activation.target.provider == "eks-managed"
    assert node_selector(activation.target) == SELECTOR
    validate_target_node_state(
        cast(Any, activation), activation.target, tuple(authorization.binding.nodes)
    )
