"""Bind credential-consumer processes to their actual Deployment and ReplicaSet."""

from __future__ import annotations

from typing import Any

from gpu_fault.admin.node_key_custody_models import CustodyError
from gpu_fault.container_env_snapshot import ROLE_DEPLOYMENTS


def cpu_role_container(deployment_name: str, spec: dict[str, Any]) -> str:
    """Use the actual single application container of a known rendered CPU role."""
    containers = spec.get("containers")
    if (
        deployment_name not in ROLE_DEPLOYMENTS
        or not isinstance(containers, list)
        or len(containers) != 1
        or not isinstance(containers[0], dict)
        or not isinstance(containers[0].get("name"), str)
        or not containers[0]["name"]
    ):
        raise CustodyError(
            "node key activation CPU role container is ambiguous or unknown"
        )
    return str(containers[0]["name"])


def controller_owner(metadata: dict[str, Any], kind: str) -> tuple[str, str]:
    owners = [
        item
        for item in metadata.get("ownerReferences", [])
        if item.get("controller") is True
    ]
    if (
        len(owners) != 1
        or any(
            not isinstance(owners[0].get(key), str) or not owners[0][key]
            for key in ("uid", "name")
        )
        or owners[0].get("kind") != kind
        or owners[0].get("apiVersion") != "apps/v1"
    ):
        raise CustodyError("credential consumer has no unique bound controller")
    return owners[0]["uid"], owners[0]["name"]


def _items(document: dict[str, Any]) -> list[dict[str, Any]]:
    items = document.get("items")
    if (
        not isinstance(items, list)
        or (document.get("metadata") or {}).get("continue")
        or any(not isinstance(item, dict) for item in items)
    ):
        raise CustodyError("credential consumer inventory is incomplete")
    return items


def bound_consumer_pods(
    deployment: dict[str, Any],
    replicasets: dict[str, Any],
    pods: dict[str, Any],
    *,
    namespace: str,
) -> list[dict[str, Any]]:
    """A complete Ready census, rejecting same-label foreign Pods and old images."""

    try:
        metadata, spec, status = (
            deployment["metadata"],
            deployment["spec"],
            deployment["status"],
        )
        desired, generation = spec["replicas"], metadata["generation"]
        selector = spec["selector"]
        if (
            deployment.get("kind") != "Deployment"
            or deployment.get("apiVersion") != "apps/v1"
            or metadata.get("namespace") != namespace
            or not metadata.get("uid")
            or not metadata.get("name")
            or metadata.get("deletionTimestamp")
            or spec.get("paused", False) is not False
            or type(desired) is not int
            or desired < 0
            or type(generation) is not int
            or generation < 1
            or status.get("observedGeneration") != generation
            or not selector.get("matchLabels")
            or selector.get("matchExpressions")
            or any(
                type(status.get(key)) is not int or status[key] != desired
                for key in (
                    "replicas",
                    "updatedReplicas",
                    "readyReplicas",
                    "availableReplicas",
                )
            )
            or status.get("unavailableReplicas", 0) != 0
            or status.get("terminatingReplicas", 0) != 0
        ):
            raise CustodyError("credential consumer Deployment is not fully converged")
        owned: dict[str, dict[str, Any]] = {}
        for item in _items(replicasets):
            uid = item.get("metadata", {}).get("uid")
            if not isinstance(uid, str) or not uid or uid in owned:
                raise CustodyError(
                    "credential consumer ReplicaSet identity is incomplete"
                )
            owned[uid] = item
        expected_containers = {
            item["name"]: item["image"]
            for item in spec["template"]["spec"]["containers"]
        }
        result = _items(pods)
        if len(result) != desired:
            raise CustodyError("credential consumer Pods do not cover every replica")
        seen: set[str] = set()
        for pod in result:
            pod_meta, pod_spec, pod_status = pod["metadata"], pod["spec"], pod["status"]
            rs_uid, rs_name = controller_owner(pod_meta, "ReplicaSet")
            rs = owned.get(rs_uid)
            if (
                rs is None
                or rs.get("kind") != "ReplicaSet"
                or rs.get("apiVersion") != "apps/v1"
                or rs["metadata"].get("namespace") != namespace
                or rs["metadata"].get("name") != rs_name
                or rs["metadata"].get("deletionTimestamp")
                or controller_owner(rs["metadata"], "Deployment")
                != (metadata["uid"], metadata["name"])
                or pod.get("apiVersion") != "v1"
                or pod.get("kind") != "Pod"
                or pod_meta.get("namespace") != namespace
                or not pod_meta.get("uid")
                or pod_meta["uid"] in seen
                or not pod_meta.get("name")
                or pod_meta.get("deletionTimestamp")
                or any(
                    pod_meta.get("labels", {}).get(key) != value
                    for key, value in selector["matchLabels"].items()
                )
                or pod_status.get("phase") != "Running"
                or [
                    row.get("status")
                    for row in pod_status.get("conditions", [])
                    if row.get("type") == "Ready"
                ]
                != ["True"]
            ):
                raise CustodyError("credential consumer Pod is not owned and Ready")
            containers = {row["name"]: row["image"] for row in pod_spec["containers"]}
            states = {row["name"]: row for row in pod_status["containerStatuses"]}
            if (
                not expected_containers
                or containers != expected_containers
                or len(containers) != len(pod_spec["containers"])
                or len(states) != len(pod_status["containerStatuses"])
                or set(states) != set(containers)
                or any(
                    row.get("ready") is not True
                    or not row.get("state", {}).get("running")
                    for row in states.values()
                )
            ):
                raise CustodyError(
                    "credential consumer container image or process differs"
                )
            seen.add(pod_meta["uid"])
        return result
    except (KeyError, TypeError, AttributeError):
        raise CustodyError("credential consumer binding is incomplete") from None
