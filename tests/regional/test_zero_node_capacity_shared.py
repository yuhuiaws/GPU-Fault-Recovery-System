"""The zero-node rule is one module.

``gpu_fault.zero_node_capacity`` decides whether a GPU cluster whose selector
selects no node is parked or unavailable. The deploy host
(``gpu_fault_release.regional_parked_cluster``, ``admin.bootstrap_ownership``)
keeps its names for it and reads the provider through the CLI. Two copies of
the rule could drift apart, and a copy in the node runtime would make every
change to it an Agent-wave release.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault import zero_node_capacity as capacity
from gpu_fault.admin import bootstrap_ownership
from gpu_fault_release import regional_parked_cluster as parked_cluster
from tests.regional._parked_cluster_support import (
    CPU_GROUP,
    GROUP,
    OWNERSHIP,
    cpu_nodegroup,
    nodegroup,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import component_wheels  # noqa: E402

DEPLOY_HOST_NAMES = (
    "EKS_MANAGED",
    "HYPERPOD",
    "PARKED",
    "UNAVAILABLE",
    "NodePoolRead",
    "eks_parked",
    "hyperpod_parked",
    "nodepools_supplying",
    "state_of",
)
OWNERSHIP_NAMES = (
    "NodeGroupFacts",
    "groups_for_selector",
    "hyperpod_groups_for_selector",
    "hyperpod_instance_group_facts",
)


@pytest.mark.parametrize("name", DEPLOY_HOST_NAMES)
def test_the_release_engine_names_are_the_shared_rule(name: str) -> None:
    assert getattr(parked_cluster, name) is getattr(capacity, name), (
        f"regional_parked_cluster.{name} must be the shared rule, not a copy"
    )


@pytest.mark.parametrize("name", OWNERSHIP_NAMES)
def test_the_ownership_names_are_the_shared_node_group_facts(name: str) -> None:
    assert getattr(bootstrap_ownership, name) is getattr(capacity, name), (
        f"bootstrap_ownership.{name} must be the shared definition, not a copy"
    )


def test_the_shared_rule_never_reaches_the_node_runtime_or_the_control_plane() -> None:
    """In the node runtime an edit to the rule would roll every GPU node's
    Agent; the control plane never reads a provider's capacity."""

    for component in ("node_runtime", "control_plane"):
        modules = component_wheels.component_modules(component)
        assert "gpu_fault.zero_node_capacity" not in modules, component


def _raise() -> dict[str, object]:
    raise RuntimeError("AccessDenied:\n  eks:ListNodegroups")


@pytest.mark.parametrize("provider", ["eks-managed", "hyperpod"])
def test_a_provider_that_cannot_be_read_is_unavailable_with_one_reason(
    provider: str,
) -> None:
    proof = capacity.capacity_proof(
        provider,
        selector=OWNERSHIP,
        cluster_name="hp-a",
        nodegroups=_raise,
        instance_groups=_raise,  # type: ignore[arg-type]
        nodepools=lambda: None,
    )

    assert proof == {
        "parked": False,
        "provider": provider,
        "reason": capacity.UNREADABLE_REASON + "AccessDenied: eks:ListNodegroups",
        "groups": [],
    }, proof
    assert capacity.state_of(proof) == capacity.UNAVAILABLE


def test_the_deploy_host_reads_the_documents_and_hands_them_to_the_rule() -> None:
    """``provider_proof`` with documents already read is the shared verdict."""

    described = {GROUP: nodegroup(), CPU_GROUP: cpu_nodegroup()}
    target = SimpleNamespace(
        provider="eks-managed", node_selector=OWNERSHIP, hyperpod_cluster_name=""
    )
    no_crd = 'error: the server doesn\'t have a resource type "nodepools"'
    release = SimpleNamespace(
        runner=SimpleNamespace(probe_output=lambda *_a, **_k: (1, "", no_crd)),
        _gpu=lambda _target, *arguments: ["kubectl", *arguments],
    )

    proof = parked_cluster.provider_proof(release, target, described=described)

    assert proof == capacity.eks_parked(
        described, selector=OWNERSHIP, nodepools=lambda: None
    ), proof
    assert proof["parked"] is True, proof
