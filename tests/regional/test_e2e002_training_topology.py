"""E2E-002 / ISO-007 derive each cluster's training shape from its own nodes.

Live 2026-10-05 on the mixed site (three-node HyperPod + two-node plain EKS):
both pair runners refused with ``both clusters require at least three GPU
nodes`` (ISO-007 additionally ``E2E-002 cluster-pair evidence is unreadable``,
a consequence). The two-node work had covered WORKLOAD-001/002, E2E-001 and
DESTR-009/012 but not the pair cases, which still carried the fixtures' native
shape (three Pods, 24 GPUs) as a requirement of both clusters.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e
from scripts.e2e.regional import training_topology

HYPERPOD = "hyperpod-cluster"
EKS = "gpu-eks-cluster"


def _node(name: str, gpus: int = 8) -> dict[str, Any]:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "gpu_allocatable": gpus,
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "labels": {},
    }


def _settings(tmp_path: Path, fixtures: dict[str, Any]) -> Any:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("synthetic\n", encoding="ascii")
    targets = {
        cluster_id: SimpleNamespace(cluster_id=cluster_id) for cluster_id in fixtures
    }
    return e2e.Settings(
        multi=SimpleNamespace(
            cluster_a=targets[HYPERPOD],
            cluster_b=targets[EKS],
            regional=lambda target: fixtures[target.cluster_id],
        ),
        site_file=site_file,
        job_id="job-test",
        attempt_id="attempt-test",
        predecessor_path=tmp_path / "predecessor.json",
    )


def _preflight(hyperpod_nodes: int = 3, eks_nodes: int = 2) -> dict[str, Any]:
    return {
        "nodes_a": [_node(f"hyperpod-i-{index}") for index in range(hyperpod_nodes)],
        "nodes_b": [_node(f"ip-10-0-{index}") for index in range(eks_nodes)],
    }


def test_each_cluster_derives_its_own_shape(tmp_path: Path) -> None:
    settings = _settings(tmp_path, {HYPERPOD: object(), EKS: object()})

    topologies = e2e.cluster_topologies(settings, _preflight())

    assert topologies[HYPERPOD].is_native, "three eight-GPU nodes: the fixture verbatim"
    assert (topologies[EKS].node_count, topologies[EKS].world_size) == (2, 16)
    assert e2e.topology_evidence(topologies)[EKS]["candidate_nodes"] == [
        "ip-10-0-0",
        "ip-10-0-1",
    ]


def test_the_preflight_names_the_cluster_that_cannot_train(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Regional:
        def __init__(self, cluster_id: str, nodes: list[dict[str, Any]]) -> None:
            self.cluster_id = cluster_id
            self.nodes = nodes

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "release-test", "cluster_id": self.cluster_id}

        def gpu_nodes(self) -> list[dict[str, Any]]:
            return list(self.nodes)

        def gpu_workloads(self) -> list[Any]:
            return []

        def ready_pods(self, *args: Any) -> list[Any]:
            return []

    registrations = [
        {"cluster_id": HYPERPOD, "enabled": True, "synthetic": False},
        {"cluster_id": EKS, "enabled": True, "synthetic": False},
    ]
    monkeypatch.setattr(e2e, "registration_snapshot", lambda *args: registrations)
    monkeypatch.setattr(
        e2e, "registrations_are_distinct_physical_clusters", lambda items: True
    )
    monkeypatch.setattr(e2e, "predecessor_evidence", lambda *a, **k: {"valid": True})
    monkeypatch.setattr(e2e, "focused_tests", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(e2e, "control_plane_container_statuses", lambda *a: {})
    monkeypatch.setattr(e2e, "executor_identities", lambda fixture: ["executor"])

    def preflight(hyperpod_nodes: int, eks_nodes: int) -> dict[str, Any]:
        preflight = _preflight(hyperpod_nodes, eks_nodes)
        fixtures = {
            HYPERPOD: Regional(HYPERPOD, preflight["nodes_a"]),
            EKS: Regional(EKS, preflight["nodes_b"]),
        }
        return e2e.read_only_preflight(_settings(tmp_path, fixtures), tmp_path)

    mixed = preflight(3, 2)
    assert mixed["errors"] == [], mixed["errors"]
    assert mixed["training_topology"][HYPERPOD]["native"] is True
    assert mixed["training_topology"][EKS] == {
        "node_count": 2,
        "gpus_per_node": 8,
        "efa_per_node": 16,
        "world_size": 16,
        "ranks": [0, 1],
        "candidate_nodes": ["ip-10-0-0", "ip-10-0-1"],
        "native": False,
    }

    single = preflight(3, 1)
    assert single["errors"] == [
        f"{EKS}: at least 2 schedulable GPU nodes are required for distributed "
        "training; found 1 (ip-10-0-0)"
    ], "the refusal names the cluster and the fact, not a three-node rule"
    assert EKS not in single["training_topology"]
    assert single["training_topology"][HYPERPOD]["native"] is True


def test_each_workload_takes_its_clusters_shape_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: dict[str, Any] = {}

    class Managed:
        def __init__(self, fixture: Any, settings: Any) -> None:
            built[fixture.cluster_id] = settings

    monkeypatch.setattr(e2e, "ManagedWorkloadFixture", Managed)
    fixtures = {
        HYPERPOD: SimpleNamespace(cluster_id=HYPERPOD),
        EKS: SimpleNamespace(cluster_id=EKS),
    }
    settings = _settings(tmp_path, fixtures)
    topologies = e2e.cluster_topologies(settings, _preflight())
    case_dir = tmp_path / "cases" / e2e.CASE_ID

    for target in (settings.multi.cluster_a, settings.multi.cluster_b):
        e2e.managed_workload(
            settings,
            target,
            topology=topologies[target.cluster_id],
            case_dir=case_dir,
            restart_budget=1,
        )

    native, two_node = built[HYPERPOD], built[EKS]
    assert (native.expected_pods, native.expected_gpu_count) == (3, 24)
    assert native.manifest == e2e.MANIFEST, "HyperPod never sees a rewritten copy"
    assert (two_node.expected_pods, two_node.expected_gpu_count) == (2, 16)
    assert two_node.manifest == case_dir / EKS / "topology-workload.yaml"
    document = yaml.safe_load(two_node.manifest.read_text(encoding="utf-8"))
    replicas = document["spec"]["pytorchReplicaSpecs"]
    assert replicas["Worker"]["replicas"] == 1, "a Master and one Worker"
    container = replicas["Master"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"][training_topology.GPU_RESOURCE] == "8"


def test_the_plan_binds_both_shapes_and_names_them_in_the_stop_condition(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, {HYPERPOD: object(), EKS: object()})
    topologies = e2e.cluster_topologies(settings, _preflight())
    preflight = {
        **_preflight(),
        "training_topology": e2e.topology_evidence(topologies),
        "predecessor": {"valid": True},
        "release_id": "release-test",
        "focused_tests": {"passed": True, "command": [], "summary": ""},
    }

    details = e2e.plan_details(settings, preflight)

    assert details["training_topology"] == preflight["training_topology"]
    assert details["stop_conditions"][2] == (
        "either workload or its derived-topology GPU observation fails "
        f"({EKS}: 16 GPUs on 2 nodes, {HYPERPOD}: 24 GPUs on 3 nodes)"
    )


def test_branches_fall_back_to_the_native_shape_without_a_derivation(
    tmp_path: Path,
) -> None:
    branches = e2e.RecoveryBranches(
        settings=SimpleNamespace(job_id="job", attempt_id="attempt"),
        targets=(SimpleNamespace(cluster_id="a"), SimpleNamespace(cluster_id="b")),
        fixtures=[],
        workloads=[],
        prewarms=[],
        submission_started=[False, False],
        injection_contexts=[None, None],
        case_dir=tmp_path,
        expect_a_budget_denial=False,
        sources=[],
    )

    assert branches.topology(1) is training_topology.NATIVE_TOPOLOGY
    branches.topologies["b"] = training_topology.TrainingTopology(2, 8, ("n1", "n2"))
    assert branches.topology(1).world_size == 16
