from __future__ import annotations

from gpu_fault.failure_domains import (
    FAILURE_DOMAIN_LABELS,
    failure_domain_map,
    node_failure_domain,
)


def _node(name: str, **labels: str) -> dict:
    return {"metadata": {"name": name, "labels": labels}}


def test_instance_group_outranks_the_zone_inside_a_single_az_cluster() -> None:
    # Every node of a HyperPod cluster shares one zone, so the zone label
    # cannot separate two of its nodes; the instance group can.
    assert FAILURE_DOMAIN_LABELS[0] == "sagemaker.amazonaws.com/instance-group-name"
    labels = {
        "topology.kubernetes.io/zone": "us-west-2a",
        "sagemaker.amazonaws.com/instance-group-name": "worker-group-3",
    }

    assert node_failure_domain(labels) == "worker-group-3"


def test_zone_is_the_fallback_when_no_finer_label_exists() -> None:
    assert (
        node_failure_domain({"topology.kubernetes.io/zone": "us-west-2a"})
        == "us-west-2a"
    )
    assert node_failure_domain({"unrelated": "x"}) is None
    assert node_failure_domain({"topology.kubernetes.io/zone": "  "}) is None


def test_operator_supplied_label_keys_replace_the_default_priority() -> None:
    labels = {
        "sagemaker.amazonaws.com/instance-group-name": "worker-group-3",
        "topology.k8s.aws/network-node-layer-3": "nn-leaf-17",
    }

    assert (
        node_failure_domain(
            labels, label_keys=("topology.k8s.aws/network-node-layer-3",)
        )
        == "nn-leaf-17"
    )


def test_failure_domain_map_is_keyed_by_cluster_and_skips_unlabelled_nodes() -> None:
    items = [
        _node(
            "hyperpod-i-0001",
            **{"sagemaker.amazonaws.com/instance-group-name": "group-a"},
        ),
        _node(
            "hyperpod-i-0002",
            **{"sagemaker.amazonaws.com/instance-group-name": "group-a"},
        ),
        _node("hyperpod-i-0003"),
        {"metadata": {"labels": {"x": "y"}}},
    ]

    assert failure_domain_map("cluster-a", items) == {
        "cluster-a": {"hyperpod-i-0001": "group-a", "hyperpod-i-0002": "group-a"}
    }
