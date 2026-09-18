"""Positive coverage and cleanup gates for low-utilization notification phases."""

from __future__ import annotations

import copy
from typing import Any


def low_utilization_manifest(
    *,
    name: str,
    nodes: tuple[str, ...],
    image: str,
) -> dict[str, Any]:
    command = [
        "/bin/bash",
        "-ceu",
        (
            "python - <<'PY'\n"
            "import multiprocessing\n"
            "import time\n"
            "def burn():\n"
            "    value = 1\n"
            "    while True:\n"
            "        value = (value * 1103515245 + 12345) & 0x7fffffff\n"
            "workers = [multiprocessing.Process(target=burn, daemon=True) "
            "for _ in range(8)]\n"
            "for worker in workers: worker.start()\n"
            "time.sleep(1800)\n"
            "PY"
        ),
    ]
    container = {
        "name": "pytorch",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "resources": {
            "requests": {"cpu": "8", "memory": "2Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "16", "memory": "4Gi", "nvidia.com/gpu": "1"},
        },
    }
    if len(nodes) == 1:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "restartPolicy": "Never",
                        "nodeName": nodes[0],
                        "containers": [container],
                    },
                }
            },
        }
    affinity = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "In",
                                "values": list(nodes),
                            }
                        ]
                    }
                ],
            }
        },
        "podAntiAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {"matchLabels": {"app": name}},
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        },
    }
    template = {
        "metadata": {"labels": {"app": name}},
        "spec": {
            "restartPolicy": "Never",
            "affinity": affinity,
            "containers": [container],
        },
    }
    # Independent templates prevent YAML aliases and double metadata injection.
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": name},
        "spec": {
            "runPolicy": {"cleanPodPolicy": "All"},
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "restartPolicy": "Never",
                    "template": copy.deepcopy(template),
                },
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "Never",
                    "template": copy.deepcopy(template),
                },
            },
        },
    }


def node_errors(node: dict[str, Any]) -> list[str]:
    errors = []
    if not node.get("uid") or node.get("ready") != "True":
        errors.append("node identity or Ready state is unknown")
    if node.get("unschedulable") is not False or node.get("ownership_annotations"):
        errors.append("node is cordoned or owned by another workflow")
    if str(node.get("gpu_allocatable")) != "8":
        errors.append("the aggregation fixture requires an eight-GPU node")
    return errors


def phase_checks(observation: dict[str, Any]) -> dict[str, bool]:
    nodes = set(observation["nodes"])
    records = observation["notifications"]
    reported = {node for item in records for node in item["matched_nodes"]}
    return {
        "replica_metadata_injected_per_role": not observation["metadata_errors"],
        "workload_on_selected_nodes": {pod["node"] for pod in observation["pods"]}
        == nodes,
        "count_at_most_node_count": len(records) <= len(nodes),
        "every_selected_node_reported_in_its_phase": reported == nodes,
        "every_notification_is_node_scoped": all(
            len(item["matched_nodes"]) == 1 for item in records
        ),
        "notifications_aggregate_multiple_gpu_devices": bool(records)
        and all(1 < len(item["gpu_devices"]) <= 8 for item in records),
        "every_notification_sent": bool(records)
        and all(
            item.get("status") == "SENT"
            and item.get("provider_message_id_present") is True
            for item in records
        ),
    }
