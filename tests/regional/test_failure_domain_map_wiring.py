"""The failure-domain map reaches the executor by exactly the path the product renders.

The release engine writes one ConfigMap during the control-plane apply; the
control-worker mounts it and points `GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP`
at the file. If either side drifts the domain budget tier goes silently inert
again, which is the defect this wiring exists to close. The fleet rollout reads
the same node labels for its per-domain wave cap, so the two readings are pinned
against each other here too.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from gpu_fault.execution.remediation_budget import FAILURE_DOMAIN_MAP_ENV
from gpu_fault.failure_domains import (
    FAILURE_DOMAIN_CONFIGMAP,
    FAILURE_DOMAIN_LABELS,
    FAILURE_DOMAIN_MOUNT_DIR,
    UNKNOWN_FAILURE_DOMAIN,
    failure_domain_map,
)
from gpu_fault_release import regional_release_fleet_rollout as FLEET_MODULE

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "deploy/control-plane/regional/generated"


def _worker() -> dict:
    return yaml.safe_load(
        (GENERATED / "gpu-fault-control-worker.yaml").read_text(encoding="utf-8")
    )


def test_control_worker_mounts_the_optional_failure_domain_map() -> None:
    pod = _worker()["spec"]["template"]["spec"]
    (api,) = pod["containers"]

    env = {item["name"]: item for item in api["env"]}
    assert env[FAILURE_DOMAIN_MAP_ENV]["valueFrom"]["configMapKeyRef"] == {
        "name": FAILURE_DOMAIN_CONFIGMAP,
        "key": "map-path",
        "optional": True,
    }
    mount = next(
        item for item in api["volumeMounts"] if item["name"] == "failure-domain-map"
    )
    assert mount == {
        "name": "failure-domain-map",
        "mountPath": FAILURE_DOMAIN_MOUNT_DIR,
        "readOnly": True,
    }
    volume = next(
        item for item in pod["volumes"] if item["name"] == "failure-domain-map"
    )
    assert volume["configMap"] == {"name": FAILURE_DOMAIN_CONFIGMAP, "optional": True}


def _node(name: str, **labels: str) -> dict:
    return {"metadata": {"name": name, "labels": labels}}


def _release(items: list[dict], label_keys=FAILURE_DOMAIN_LABELS) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(failure_domain_labels=label_keys),
        _gpu=lambda _target, *args: ["kubectl", *args],
        _get_json=lambda _args: {"items": items},
    )


def test_fleet_rollout_and_remediation_map_read_one_failure_domain_per_node() -> None:
    """Two readers of "what is a failure domain" must not be able to disagree.

    The remediation map omits unlabelled nodes (they consume node, cluster and
    region quota only); the fleet rollout must name a domain for every node it
    rolls, so it substitutes the UNKNOWN sentinel for exactly the same nodes.
    Every labelled node carries the identical domain in both shapes.
    """

    items = [
        _node("n-1", **{"sagemaker.amazonaws.com/instance-group-name": "group-a"}),
        _node("n-2", **{"topology.kubernetes.io/zone": "us-east-1a"}),
        _node(
            "n-3",
            **{
                "sagemaker.amazonaws.com/instance-group-name": "group-b",
                "topology.kubernetes.io/zone": "us-east-1a",
            },
        ),
        _node("n-4"),
        _node("n-5", **{"unrelated": "label"}),
    ]
    target = SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a")
    names = tuple(item["metadata"]["name"] for item in items)

    fleet = FLEET_MODULE.target_node_failure_domains(_release(items), target, names)
    remediation = failure_domain_map("gpu-a", items)["gpu-a"]

    assert set(fleet) == set(names), "the fleet names a domain for every node"
    assert {
        node: domain
        for node, domain in fleet.items()
        if domain != UNKNOWN_FAILURE_DOMAIN
    } == remediation, "labelled nodes read identically on both paths"
    assert {
        node for node, domain in fleet.items() if domain == UNKNOWN_FAILURE_DOMAIN
    } == set(names) - set(remediation), "the unlabelled set is the same set"
    assert UNKNOWN_FAILURE_DOMAIN not in remediation.values()


def test_fleet_rollout_reads_the_site_configured_label_keys() -> None:
    items = [
        _node(
            "n-1",
            **{
                "sagemaker.amazonaws.com/instance-group-name": "group-a",
                "topology.k8s.aws/network-node-layer-3": "rack-7",
            },
        )
    ]
    target = SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a")
    keys = ("topology.k8s.aws/network-node-layer-3",)

    fleet = FLEET_MODULE.target_node_failure_domains(
        _release(items, keys), target, ("n-1",)
    )

    assert fleet == {"n-1": "rack-7"}
    assert failure_domain_map("gpu-a", items, label_keys=keys)["gpu-a"] == fleet


def test_unknown_topology_still_clamps_the_wave_to_one_node() -> None:
    policy = FLEET_MODULE.node_rollout_policy(
        SimpleNamespace(config=SimpleNamespace(upgrade_max_unavailable=0)),
        {"n-1": UNKNOWN_FAILURE_DOMAIN, "n-2": "zone-a", "n-3": "zone-a"},
        phase="upgrade",
    )

    assert policy.max_unavailable == 1, "an unknown topology has no blast radius"
