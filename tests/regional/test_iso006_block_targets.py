"""ISO-006 blocks every cluster B node that talks to the control plane.

Live on the mixed site: the probe installed its REJECT (OUTPUT and FORWARD)
on cluster B's GPU nodes, and cluster B's executor kept claiming -- on
eks-managed the executor Deployment does not tolerate the GPU node taint, so
it runs on a CPU node the GPU-node block never touched, and the runner refused
"cluster B still reaches the control plane after the block". The block is now
aimed at the GPU nodes *and* the nodes hosting Ready executor Pods; the
evidence names each target's roles and each receipt's node and rules.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.e2e.regional import run_iso006_cluster_offline as runner

GPU_NODES = [
    {"name": "gpu-1", "uid": "uid-gpu-1", "ready": "True", "gpu_allocatable": 8},
    {"name": "gpu-2", "uid": "uid-gpu-2", "ready": "True", "gpu_allocatable": 8},
]


def _node_document(name: str, *, ready: str = "True") -> dict[str, Any]:
    return {
        "metadata": {"name": name, "uid": f"uid-{name}"},
        "status": {"conditions": [{"type": "Ready", "status": ready}]},
    }


def test_executor_nodes_outside_the_gpu_set_become_targets_with_their_role() -> None:
    reads: list[str] = []

    def read_node(name: str) -> dict[str, Any]:
        reads.append(name)
        return _node_document(name)

    targets = runner.block_targets(
        GPU_NODES,
        [
            {"name": "executor-a", "uid": "p1", "node": "cpu-7"},
            {"name": "executor-b", "uid": "p2", "node": "gpu-2"},
        ],
        read_node,
    )

    assert [item["name"] for item in targets["nodes"]] == ["cpu-7", "gpu-1", "gpu-2"]
    by_name = {item["name"]: item for item in targets["nodes"]}
    assert by_name["cpu-7"] == {
        "name": "cpu-7",
        "uid": "uid-cpu-7",
        "ready": "True",
        "roles": [runner.ROLE_EXECUTOR_NODE],
    }
    assert by_name["gpu-2"]["roles"] == [
        runner.ROLE_GPU_NODE,
        runner.ROLE_EXECUTOR_NODE,
    ]
    assert by_name["gpu-1"]["roles"] == [runner.ROLE_GPU_NODE]
    assert reads == ["cpu-7"], "only the node the GPU listing did not already carry"
    assert targets["executor_pods"] == [
        {"name": "executor-a", "node": "cpu-7"},
        {"name": "executor-b", "node": "gpu-2"},
    ]
    assert targets["executor_pods_without_node"] == []


def test_a_pod_without_a_node_is_recorded_not_blocked() -> None:
    targets = runner.block_targets(
        GPU_NODES, [{"name": "executor-x", "uid": "p"}], lambda name: pytest.fail(name)
    )

    assert [item["name"] for item in targets["nodes"]] == ["gpu-1", "gpu-2"]
    assert targets["executor_pods_without_node"] == ["executor-x"]


def test_the_preflight_aims_the_block_at_the_executor_node_when_it_is_known(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from tests.regional._cov95_identity_isolation_support import preflight_environment

    settings, regions, _registrations, _reads = preflight_environment(
        monkeypatch, tmp_path, runner
    )
    _a, b = regions

    silent = runner.read_only_preflight(settings, tmp_path)

    assert [item["roles"] for item in silent["block_targets_b"]["nodes"]] == [
        [runner.ROLE_GPU_NODE]
    ] * 3
    assert silent["block_targets_b"]["executor_pods_without_node"] == [
        "gpu-fault-cluster-executor"
    ], "a Pod that names no node is recorded; the cut proof still judges the block"

    original_kubectl = b.kubectl

    def kubectl(*args: str) -> str:
        if args[1:3] == ("get", "node"):
            return json.dumps(_node_document(args[3]))
        return original_kubectl(*args)

    monkeypatch.setattr(b, "kubectl", kubectl)
    monkeypatch.setattr(
        b,
        "ready_pods",
        lambda plane, app: [{"name": app, "uid": "uid-" + app, "node": "cpu-system-1"}],
    )

    aimed = runner.read_only_preflight(settings, tmp_path)

    assert aimed["block_targets_b"]["executor_pods_without_node"] == []
    names = [item["name"] for item in aimed["block_targets_b"]["nodes"]]
    assert "cpu-system-1" in names and len(names) == 4
    assert next(
        item
        for item in aimed["block_targets_b"]["nodes"]
        if item["name"] == "cpu-system-1"
    )["roles"] == [runner.ROLE_EXECUTOR_NODE]
