from __future__ import annotations

import copy


def converged_deployment(value):
    value = copy.deepcopy(value)
    value["metadata"].setdefault("generation", 1)
    desired = value["spec"]["replicas"]
    value["spec"].setdefault(
        "selector", {"matchLabels": {"app": value["metadata"]["name"]}}
    )
    value["status"] = {
        "observedGeneration": value["metadata"]["generation"],
        **{
            key: desired
            for key in (
                "replicas",
                "updatedReplicas",
                "readyReplicas",
                "availableReplicas",
            )
        },
    }
    return value


def owned_pod_documents(deployment, marker="initial"):
    meta = deployment["metadata"]
    name, namespace = meta["name"], meta["namespace"]
    rs_name = name + "-" + marker[:8]
    rs_uid = rs_name + "-uid"
    replicasets = {
        "items": [
            {
                "apiVersion": "apps/v1",
                "kind": "ReplicaSet",
                "metadata": {
                    "name": rs_name,
                    "namespace": namespace,
                    "uid": rs_uid,
                    "ownerReferences": [
                        {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "name": name,
                            "uid": meta["uid"],
                            "controller": True,
                        }
                    ],
                },
            }
        ]
    }
    template = deployment["spec"]["template"]
    pods = {
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": rs_name + "-pod-" + str(index),
                    "uid": rs_uid + "-pod-" + str(index),
                    "namespace": namespace,
                    "labels": {"app": name},
                    "annotations": copy.deepcopy(
                        template["metadata"].get("annotations", {})
                    ),
                    "ownerReferences": [
                        {
                            "apiVersion": "apps/v1",
                            "kind": "ReplicaSet",
                            "name": rs_name,
                            "uid": rs_uid,
                            "controller": True,
                        }
                    ],
                },
                "spec": copy.deepcopy(template["spec"]),
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [
                        {
                            "name": container["name"],
                            "ready": True,
                            "containerID": "containerd://"
                            + rs_uid
                            + "-"
                            + container["name"],
                            "state": {"running": {"startedAt": "2026-09-15T00:00:00Z"}},
                        }
                        for container in template["spec"]["containers"]
                    ],
                },
            }
            for index in range(deployment["spec"]["replicas"])
        ]
    }
    return replicasets, pods
