#!/usr/bin/env python3
"""Collect live CPU/GPU installed resources for regional cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sync_installed_resource_registry import (
    DEFAULT_INVENTORY,
    Kubectl,
    synchronize,
)

NAMESPACED_DISCOVERY_KINDS = (
    "deployment,daemonset,statefulset,cronjob,"
    "poddisruptionbudget,role,rolebinding,"
    "serviceaccount,service"
)
CLUSTER_DISCOVERY_KINDS = "clusterrole,clusterrolebinding"


def _identity(resource: dict[str, Any]) -> tuple[str, str]:
    return resource["kind"], resource["name"]


def discover_unregistered(
    kubectl: Kubectl,
    *,
    plane: str,
    context: str,
    namespace: str,
    registered: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    found = []
    namespaced = kubectl.run(
        [
            "get",
            NAMESPACED_DISCOVERY_KINDS,
            "-A",
            "-o",
            "json",
        ]
    )
    for item in json.loads(namespaced.stdout)["items"]:
        metadata = item.get("metadata") or {}
        name = metadata.get("name", "")
        resource_namespace = metadata.get("namespace", "")
        kind = str(item.get("kind", "")).lower()
        if not name.startswith("gpu-fault-"):
            continue
        if resource_namespace != namespace and not resource_namespace.startswith(
            "gf-regional-"
        ):
            continue
        if (kind, name) in registered:
            continue
        found.append(
            {
                "plane": plane,
                "context": context,
                "kind": kind,
                "name": name,
                "scope": "namespaced",
                "namespace": resource_namespace,
            }
        )
    cluster = kubectl.run(
        [
            "get",
            CLUSTER_DISCOVERY_KINDS,
            "-o",
            "json",
        ]
    )
    for item in json.loads(cluster.stdout)["items"]:
        metadata = item.get("metadata") or {}
        name = metadata.get("name", "")
        kind = str(item.get("kind", "")).lower()
        if not name.startswith("gpu-fault-") or (kind, name) in registered:
            continue
        found.append(
            {
                "plane": plane,
                "context": context,
                "kind": kind,
                "name": name,
                "scope": "cluster",
                "namespace": None,
            }
        )
    namespaces = kubectl.run(["get", "namespace", "-o", "json"])
    for item in json.loads(namespaces.stdout)["items"]:
        name = (item.get("metadata") or {}).get("name", "")
        if not name.startswith("gf-regional-"):
            continue
        found.append(
            {
                "plane": plane,
                "context": context,
                "kind": "namespace",
                "name": name,
                "scope": "cluster",
                "namespace": None,
            }
        )
    return found


def collect(
    config_path: Path,
    *,
    inventory_path: Path,
    apply: bool,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = json.loads(inventory_path.read_text(encoding="utf-8"))
    namespace = config.get("namespace", "gpu-fault-system")
    release = config.get("release") or {}
    release_id = str(
        release.get("id") or Path(str(release.get("manifest") or "unknown")).stem
    )
    cpu_kubectl = Kubectl(
        kubeconfig=config["cpu_kubeconfig"],
        context=None,
        reuse_exec_credential=True,
    )
    cpu = synchronize(
        cpu_kubectl,
        plane="cpu",
        namespace=namespace,
        inventory_path=inventory_path,
        release_id=release_id,
        apply=apply,
    )
    unregistered = discover_unregistered(
        cpu_kubectl,
        plane="cpu",
        context=config["cpu_kubeconfig"],
        namespace=namespace,
        registered={_identity(resource) for resource in cpu["resources"]},
    )
    gpu_documents = []
    for cluster in config["clusters"]:
        gpu_kubectl = Kubectl(
            kubeconfig=None,
            context=cluster["context"],
            reuse_exec_credential=True,
        )
        document = synchronize(
            gpu_kubectl,
            plane="gpu",
            namespace=namespace,
            inventory_path=inventory_path,
            release_id=release_id,
            apply=apply,
        )
        gpu_documents.append(document)
        unregistered.extend(
            discover_unregistered(
                gpu_kubectl,
                plane="gpu",
                context=cluster["context"],
                namespace=namespace,
                registered={_identity(resource) for resource in document["resources"]},
            )
        )
    gpu_resources: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    for document in gpu_documents:
        for resource in document["resources"]:
            identity = _identity(resource)
            existing = gpu_resources.get(identity)
            if existing is not None and existing != resource:
                raise RuntimeError(
                    f"GPU installed registries disagree for {identity[0]}/{identity[1]}"
                )
            gpu_resources[identity] = resource

    result = {
        "schema_version": 1,
        "generated_by": (
            "deploy/control-plane/tools/collect_installed_resource_registry.py"
        ),
        "cpu": {
            **{
                key: value for key, value in source["cpu"].items() if key != "resources"
            },
            "resources": cpu["resources"],
        },
        "gpu": {
            **{
                key: value for key, value in source["gpu"].items() if key != "resources"
            },
            "resources": list(gpu_resources.values()),
        },
        "unregistered_resources": unregistered,
    }
    cpu_names = {
        item["name"]
        for item in result["cpu"]["resources"]
        if item["kind"] == "deployment"
    }
    result["cpu"]["database_pod_preference"] = [
        name for name in source["cpu"]["database_pod_preference"] if name in cpu_names
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.config,
                inventory_path=args.inventory,
                apply=args.apply,
            ),
            indent=2,
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
