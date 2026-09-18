"""Read-only membership publication proof, including its legitimate worker roll."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gpu_fault.admin.execution import run_command
from gpu_fault.admin.failure_domain_map import (
    CONTROL_WORKER_DEPLOYMENT,
    build_failure_domain_map,
    failure_domain_configmap,
    verify_failure_domain_publication,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.failure_domains import failure_domain_map_sha256
from gpu_fault_release.regional_deployment_inventory import CPU_RUNTIME_DEPLOYMENTS
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records


def _role_pods(site: RenderedSite, deployment: dict[str, Any]) -> list[dict[str, Any]]:
    selector = deployment["spec"].get("selector") or {}
    labels = selector.get("matchLabels")
    if not isinstance(labels, dict) or not labels or selector.get("matchExpressions"):
        raise ValueError("CPU role has an unsupported Pod selector")
    result = run_command(
        [
            "kubectl",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
            "-n",
            str(site.release_config["namespace"]),
            "get",
            "pod,replicaset",
            "-l",
            ",".join(f"{key}={value}" for key, value in sorted(labels.items())),
            "-o",
            "json",
        ],
        timeout_seconds=120,
    )
    if result.returncode:
        raise ValueError("CPU Pod ownership query failed")
    value = json.loads(result.stdout)
    sets: dict[str, dict[str, Any]] = {}
    pods: list[dict[str, Any]] = []
    for item in value["items"]:
        if item.get("kind") == "ReplicaSet":
            owners = [
                owner
                for owner in item["metadata"].get("ownerReferences", [])
                if owner.get("controller") is True
            ]
            if (
                len(owners) == 1
                and owners[0].get("kind") == "Deployment"
                and owners[0].get("uid") == deployment["metadata"]["uid"]
                and owners[0].get("name") == deployment["metadata"]["name"]
                and owners[0].get("apiVersion") == "apps/v1"
                and isinstance(item["metadata"].get("uid"), str)
                and item["metadata"]["uid"]
            ):
                sets[item["metadata"]["uid"]] = item
        elif item.get("kind") == "Pod":
            pods.append(item)
    ready = ready_pod_records({"items": pods})
    if len(ready) != deployment["spec"]["replicas"]:
        raise ValueError("CPU role has an incomplete ready Pod set")
    by_uid = {item["metadata"]["uid"]: item for item in pods}
    result_pods = []
    for record in ready:
        pod = by_uid[record["uid"]]
        owners = [
            owner
            for owner in pod["metadata"].get("ownerReferences", [])
            if owner.get("controller") is True
        ]
        if (
            len(owners) != 1
            or owners[0].get("kind") != "ReplicaSet"
            or owners[0].get("uid") not in sets
        ):
            raise ValueError("CPU Pod is not owned by the observed Deployment")
        owner_set = sets[owners[0]["uid"]]
        if (
            owners[0].get("name") != owner_set["metadata"]["name"]
            or owners[0].get("apiVersion") != "apps/v1"
        ):
            raise ValueError("CPU Pod controller reference differs")
        containers = []
        for status in pod["status"]["containerStatuses"]:
            if (
                type(status.get("restartCount")) is not int
                or status["restartCount"] < 0
                or not isinstance(status.get("containerID"), str)
                or not status["containerID"]
            ):
                raise ValueError("CPU container process identity is missing")
            containers.append(
                {
                    "name": status["name"],
                    "container_id": status["containerID"],
                    "restart_count": status["restartCount"],
                }
            )
        result_pods.append(
            {**record, "containers": sorted(containers, key=lambda item: item["name"])}
        )
    return result_pods


def cpu_observation(site: RenderedSite) -> dict[str, Any]:
    result = run_command(
        [
            "kubectl",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
            "-n",
            str(site.release_config["namespace"]),
            "get",
            "deployment",
            "-o",
            "json",
        ],
        timeout_seconds=120,
    )
    if result.returncode:
        raise ValueError("CPU role inventory query failed")
    document = json.loads(result.stdout)
    deployments: dict[str, Any] = {}
    for item in document["items"]:
        meta, spec, status = item["metadata"], item["spec"], item.get("status") or {}
        name = meta["name"]
        if name not in CPU_RUNTIME_DEPLOYMENTS:
            continue
        desired = spec.get("replicas")
        generation = meta.get("generation")
        if (
            name in deployments
            or not meta.get("uid")
            or meta.get("deletionTimestamp")
            or type(generation) is not int
            or generation < 1
            or type(desired) is not int
            or desired < 0
            or status.get("observedGeneration") != generation
            or any(
                status.get(key, 0) != desired
                for key in (
                    "replicas",
                    "updatedReplicas",
                    "readyReplicas",
                    "availableReplicas",
                )
            )
            or status.get("unavailableReplicas", 0) != 0
        ):
            raise ValueError("CPU role observation is incomplete or not converged")
        deployments[name] = {
            "uid": meta["uid"],
            "generation": generation,
            "replicas": desired,
            "template_sha256": hashlib.sha256(
                json.dumps(
                    spec["template"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "pods": _role_pods(site, item),
        }
    if set(deployments) != set(CPU_RUNTIME_DEPLOYMENTS):
        raise ValueError("CPU role observation omitted a required role")
    return deployments


def membership_observation(site: RenderedSite) -> dict[str, Any]:
    deployments = cpu_observation(site)
    expected = build_failure_domain_map(site)
    manifest = failure_domain_configmap(
        expected.mapping, namespace=str(site.release_config["namespace"])
    )
    proof = verify_failure_domain_publication(
        site,
        digest=failure_domain_map_sha256(manifest),
        worker_uid=deployments[CONTROL_WORKER_DEPLOYMENT]["uid"],
    )
    return {"deployments": deployments, "publication": proof}


def membership_transition_errors(
    before: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    old, new = before["deployments"], after["deployments"]
    if set(old) != set(CPU_RUNTIME_DEPLOYMENTS) or set(new) != set(old):
        return ["membership CPU role inventory changed"]
    errors = []
    changed_map = (
        before["publication"]["map_sha256"] != after["publication"]["map_sha256"]
    )
    for name in old:
        if old[name]["uid"] != new[name]["uid"]:
            errors.append(f"{name} Deployment was replaced")
        if name == CONTROL_WORKER_DEPLOYMENT and changed_map:
            if new[name]["generation"] <= old[name]["generation"]:
                errors.append("new failure-domain map did not roll the worker")
        elif old[name] != new[name]:
            errors.append(f"membership unexpectedly rolled {name}")
    return errors
