"""The training cases derive their Master/Worker shape from the cluster.

Three nodes were the fixtures' shape, not the cases' requirement: a two-node
plain-EKS cluster trained the fixture at 16 ranks (live 2026-10-05). The
topology is derived from the Ready, schedulable, isolation-taint-free GPU
nodes -- at most three, at least two -- and every check follows it; a
three-node HyperPod cluster derives the native shape and keeps the fixtures
byte for byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_workload_acceptance as workload
from scripts.e2e.regional import training_topology as topology

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT / "scripts/e2e/regional/manifests/training/xid11-three-node-pytorchjob.yaml"
)


def _node(
    name: str, *, ready: str = "True", unschedulable: bool = False, gpus=8, taints=()
):
    return {
        "name": name,
        "uid": f"uid-{name}",
        "ready": ready,
        "unschedulable": unschedulable,
        "taints": list(taints),
        "gpu_allocatable": gpus,
    }


def test_three_schedulable_eight_gpu_nodes_derive_the_native_topology() -> None:
    nodes = [_node("c"), _node("a"), _node("b"), _node("spare", unschedulable=True)]
    derived = topology.derive_training_topology(nodes)
    assert derived.is_native, derived
    assert derived.candidate_nodes == ("a", "b", "c"), (
        "sorted, the cordoned spare excluded"
    )
    assert derived.world_size == 24 and derived.ranks == {0, 1, 2}
    assert derived.all_reduce_marker() == "all_reduce=300.0"
    assert topology.render_topology_manifest(FIXTURE, derived) == FIXTURE, (
        "the native topology uses the checked-in fixture itself"
    )


def test_two_nodes_derive_a_master_and_one_worker_at_sixteen_ranks(
    tmp_path: Path,
) -> None:
    nodes = [
        _node("ip-10-0-33-205", gpus="8"),
        _node("ip-10-0-41-164", gpus="8"),
        _node("quarantined", taints=[{"key": "gpu-fault.io/quarantined"}]),
    ]
    derived = topology.derive_training_topology(nodes)
    assert not derived.is_native, "two nodes are not the fixtures' native shape"
    assert (derived.node_count, derived.gpus_per_node, derived.world_size) == (2, 8, 16)
    assert derived.ranks == {0, 1}
    assert derived.all_reduce_marker() == "all_reduce=136.0"
    assert derived.evidence()["candidate_nodes"] == ["ip-10-0-33-205", "ip-10-0-41-164"]

    rendered = topology.render_topology_manifest(
        FIXTURE, derived, tmp_path / "two.yaml"
    )
    assert rendered == tmp_path / "two.yaml"
    document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    replicas = document["spec"]["pytorchReplicaSpecs"]
    assert replicas["Master"]["replicas"] == 1 and replicas["Worker"]["replicas"] == 1
    for replica in replicas.values():
        (container,) = replica["template"]["spec"]["containers"]
        assert "--nproc-per-node=8" in container["command"][-1]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "8"
        assert container["resources"]["limits"]["vpc.amazonaws.com/efa"] == "16", (
            "eight GPUs per node keep the fixture's EFA request"
        )
    assert FIXTURE.read_text(encoding="utf-8") == FIXTURE.read_text(encoding="utf-8")


def test_fewer_gpus_per_node_scale_the_requests_and_the_efa_devices(
    tmp_path: Path,
) -> None:
    derived = topology.derive_training_topology(
        [_node("a", gpus=4), _node("b", gpus=4)]
    )
    assert (derived.gpus_per_node, derived.efa_per_node, derived.world_size) == (
        4,
        8,
        8,
    )
    rendered = topology.render_topology_manifest(
        FIXTURE, derived, tmp_path / "small.yaml"
    )
    container = yaml.safe_load(rendered.read_text(encoding="utf-8"))["spec"][
        "pytorchReplicaSpecs"
    ]["Worker"]["template"]["spec"]["containers"][0]
    assert "--nproc-per-node=4" in container["command"][-1]
    assert container["resources"]["requests"] == {
        **container["resources"]["requests"],
        "nvidia.com/gpu": "4",
        "vpc.amazonaws.com/efa": "8",
    }


def test_the_refusals_name_the_cluster_facts() -> None:
    with pytest.raises(topology.TrainingTopologyError, match="found 1 \\(only\\)"):
        topology.derive_training_topology(
            [_node("only"), _node("cordoned", unschedulable=True)]
        )
    with pytest.raises(topology.TrainingTopologyError, match="different GPU counts"):
        topology.derive_training_topology([_node("a", gpus=8), _node("b", gpus=4)])
    unreported = topology.derive_training_topology(
        [_node("a", gpus=None), _node("b", gpus=None)]
    )
    assert unreported.gpus_per_node == 8, (
        "a projection without allocatable keeps the fixture"
    )
    prefiltered = topology.derive_training_topology(
        [{"name": "x"}, {"name": "y"}], schedulable_only=False
    )
    assert prefiltered.candidate_nodes == ("x", "y"), (
        "a preflight's candidates are taken as is"
    )
    with pytest.raises(topology.TrainingTopologyError, match="not a PyTorchJob"):
        topology.render_topology_manifest(
            ROOT / "scripts/e2e/regional/manifests/training/xid11-single-node-job.yaml",
            prefiltered,
        )


class _Regional:
    def __init__(self, nodes: list[dict[str, Any]], cluster_id: str) -> None:
        self.nodes = nodes
        self.settings = type("S", (), {"cluster_id": cluster_id})()
        self.reads = 0

    def gpu_nodes(self) -> list[dict[str, Any]]:
        self.reads += 1
        return list(self.nodes)


def test_the_workload_runner_derives_once_per_cluster_and_prewarms_its_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workload, "TOPOLOGIES", {})
    two = _Regional([_node("a"), _node("b")], "gpu-eks-cluster")
    three = _Regional([_node("a"), _node("b"), _node("c")], "hp-cluster")
    assert workload.prewarm_nodes(two) == ["a", "b"]
    assert workload.prewarm_nodes(two) == ["a", "b"] and two.reads == 1, (
        "the inventory is read once per cluster"
    )
    assert workload.topology_for(two).world_size == 16
    assert workload.topology_for(three).is_native and three.reads == 1
    with pytest.raises(topology.TrainingTopologyError, match="at least 2"):
        workload.prewarm_nodes(_Regional([_node("only")], "small"))


def test_destr009_derives_from_the_preflight_candidates(tmp_path: Path) -> None:
    preflight = {
        "candidate_nodes": [
            {"name": "gpu-a", "uid": "a"},
            {"name": "gpu-b", "uid": "b"},
        ]
    }
    derived = destr009.case_topology(preflight)
    assert derived.node_count == 2 and derived.candidate_nodes == ("gpu-a", "gpu-b")
    settings = type("Settings", (), {"manifest": FIXTURE})()
    rendered = destr009.topology_manifest(settings, derived, tmp_path)
    assert rendered == tmp_path / "topology-workload.yaml"
    native = destr009.case_topology({"candidate_nodes": [{"name": n} for n in "abc"]})
    assert destr009.topology_manifest(settings, native, tmp_path) == FIXTURE
