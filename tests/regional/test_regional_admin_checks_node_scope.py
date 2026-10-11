"""Agent coverage and ``status --full`` read the selected node set, re-checked.

The ownership convention makes the recorded selector the one definition of a
cluster's nodes. The coverage comparison and the per-cluster health check
therefore confirm locally that every node the API server returned satisfies
the selector (as every other node read does), and the full report shows the
selector, its source and the HyperPod nodes the selector leaves to the
provider.
"""

from __future__ import annotations

from types import SimpleNamespace

from gpu_fault_release import regional_admin_checks_coverage as agent_coverage
from gpu_fault_release.regional_admin_checks_coverage import check_agent_coverage

SITE_ID = "us-west-2-site-985fd370"
HP_SELECTOR = "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
HP_LABEL_SELECTOR = f"gpu-fault.io/control-plane={SITE_ID},{HP_SELECTOR}"


def _node(name: str, labels: dict[str, str], ready: bool = True) -> dict:
    return {
        "metadata": {"name": name, "labels": labels},
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }


def _hp_labels(labelled: bool, instance_type: str = "ml.p5en.48xlarge") -> dict:
    labels = {
        "sagemaker.amazonaws.com/cluster-name": "hp-gpu-a",
        "node.kubernetes.io/instance-type": instance_type,
    }
    if labelled:
        labels["gpu-fault.io/control-plane"] = SITE_ID
    return labels


class Release:
    """A release whose ``kubectl get nodes -l`` answers ignore the selector:
    the shape of a proxy or cache that returns the wrong set."""

    def __init__(self, target, nodes):
        self.config = SimpleNamespace(clusters=[target], namespace="gpu-fault-system")
        self.nodes = nodes
        self.commands = []

    def _gpu(self, target, *arguments):
        return ["kubectl", "--context", target.context, *arguments]

    def _get_json(self, command):
        self.commands.append(command)
        return {"items": list(self.nodes)}


def _target(**overrides):
    base = dict(
        cluster_id="gpu-a",
        context="gpu-a",
        provider="hyperpod",
        hyperpod_cluster_name="hp-gpu-a",
        node_selector="",
        node_selector_source="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_coverage_node_reads_are_re_checked_against_the_selector() -> None:
    nodes = {
        "kind": "List",
        "items": [
            _node("hyperpod-i-1", _hp_labels(True)),
            _node("hyperpod-i-2", _hp_labels(True), ready=False),
            # Returned by a proxy or cache although the selector excludes it.
            _node("hyperpod-i-9", _hp_labels(False, "ml.g6.12xlarge")),
        ],
    }

    checked = agent_coverage.selected_node_items(nodes, HP_LABEL_SELECTOR)

    assert [item["metadata"]["name"] for item in checked["items"]] == [
        "hyperpod-i-1",
        "hyperpod-i-2",
    ], "a node the selector excludes never enters the coverage comparison"
    assert checked["kind"] == "List", "the rest of the document is kept"
    ready = [
        item["metadata"]["name"]
        for item in checked["items"]
        if item["status"]["conditions"][0]["status"] == "True"
    ]
    assert ready == ["hyperpod-i-1"], (
        "only Ready nodes that satisfy the selector are expected to carry an Agent"
    )


def test_agent_coverage_judges_the_filtered_set() -> None:
    """A non-GPU node of a converged HyperPod cluster has no Agent and is no drift."""

    cluster = {
        "expected_node_ids": ["hyperpod-i-1"],
        "kubernetes_node_ids": ["hyperpod-i-1"],
        "agents": [{"node_id": "hyperpod-i-1", "lifecycle_state": "ACTIVE"}],
    }
    assert [item["node_id"] for item in check_agent_coverage("gpu-a", cluster)] == [
        "hyperpod-i-1"
    ]


def test_full_status_shows_the_selector_source_and_the_nodes_left_out() -> None:
    target = _target(
        node_selector=HP_LABEL_SELECTOR, node_selector_source="label-convention"
    )
    answers = {
        HP_LABEL_SELECTOR: [_node("hyperpod-i-1", _hp_labels(True))],
        HP_SELECTOR: [
            _node("hyperpod-i-1", _hp_labels(True)),
            _node("hyperpod-i-9", _hp_labels(False, "ml.g6.12xlarge")),
        ],
    }

    class StatusRelease(Release):
        def _get_json(self, command):
            self.commands.append(command)
            return {"items": answers[command[command.index("-l") + 1]]}

    release = StatusRelease(target, [])
    details = agent_coverage.node_scope_details(
        release, target, HP_LABEL_SELECTOR, {"items": answers[HP_LABEL_SELECTOR]}
    )

    assert details == {
        "selector": HP_LABEL_SELECTOR,
        "source": "label-convention",
        "selected_nodes": 1,
        "cluster_nodes_outside_selector": ["hyperpod-i-9"],
    }, "the report names the HyperPod nodes the selector leaves to the provider"
    assert release.commands == [
        ["kubectl", "--context", "gpu-a", "get", "nodes", "-l", HP_SELECTOR]
    ]

    legacy = _target()
    legacy_details = agent_coverage.node_scope_details(
        StatusRelease(legacy, []), legacy, HP_SELECTOR, {"items": answers[HP_SELECTOR]}
    )
    assert legacy_details == {
        "selector": HP_SELECTOR,
        "source": "provider-default-legacy",
        "selected_nodes": 2,
    }, "a legacy cluster selects every cluster node, so there is nothing to compare"

    eks = _target(
        cluster_id="gpu-eks",
        context="gpu-eks",
        provider="eks-managed",
        hyperpod_cluster_name="",
        node_selector="eks.amazonaws.com/nodegroup=gpu-ng",
    )
    eks_details = agent_coverage.node_scope_details(
        StatusRelease(eks, []), eks, eks.node_selector, {"items": []}
    )
    assert eks_details == {
        "selector": "eks.amazonaws.com/nodegroup=gpu-ng",
        "source": "flag-legacy",
        "selected_nodes": 0,
    }
