#!/usr/bin/env python3
"""Collect live CPU/GPU installed resources for regional cleanup."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import subprocess
from typing import Any

from sync_installed_resource_registry import (
    DEFAULT_INVENTORY,
    Kubectl,
    RegistryError,
    ResourceIdentity,
    resource_identity,
    resource_list,
    stamp_provenance,
    synchronize,
)

NAMESPACED_DISCOVERY_KINDS = (
    "deployment,daemonset,statefulset,cronjob,job,"
    "poddisruptionbudget,role,rolebinding,"
    "serviceaccount,service"
)
CLUSTER_DISCOVERY_KINDS = "clusterrole,clusterrolebinding"


def attach_workload_rbac(
    kubectl: Kubectl,
    config: dict[str, Any],
    target: dict[str, Any],
    document: dict[str, Any],
) -> None:
    """Add live-proven renderer grants to this context's cleanup snapshot."""
    from gpu_fault.admin.cluster_removal_rbac import inspect_workload_namespace_rbac

    prefix = ["kubectl", "--context", target["context"]]

    def run(
        arguments: list[str], *, timeout_seconds: float = 20
    ) -> subprocess.CompletedProcess[str]:
        if arguments[: len(prefix)] != prefix:
            raise RegistryError("workload RBAC proof crossed its selected context")
        return kubectl.run(
            arguments[len(prefix) :], check=False, timeout_seconds=timeout_seconds
        )

    proof = inspect_workload_namespace_rbac(config, target, run=run, kubectl=prefix)
    if not proof:
        return
    existing = {
        resource_identity(resource, config["namespace"])
        for resource in document["resources"]
    }
    for item in proof["documents"]:
        metadata = item["metadata"]
        resource = {
            "kind": item["kind"].lower(),
            "name": metadata["name"],
            "namespace": metadata["namespace"],
            "scope": "namespaced",
            "phase": "support",
            "order": 20 if item["kind"] == "RoleBinding" else 30,
            "clean": "delete",
            "guarded_delete": "workload-rbac",
        }
        identity = resource_identity(resource, config["namespace"])
        if identity in existing:
            raise RegistryError("workload RBAC duplicates a registered resource")
        existing.add(identity)
        document["resources"].append(stamp_provenance(resource))
    document["workload_rbac"] = proof


def _scoped_registered(
    registered: set[ResourceIdentity] | set[tuple[str, str]], namespace: str
) -> set[ResourceIdentity]:
    result: set[ResourceIdentity] = set()
    for identity in registered:
        if len(identity) == 2:
            # Legacy callers have only the selected namespace's registrations.
            # Their name-only keys must never match another namespace.
            kind, name = identity
            cluster_scope = kind in {*CLUSTER_DISCOVERY_KINDS.split(","), "namespace"}
            result.add(
                (
                    "cluster" if cluster_scope else "namespaced",
                    kind,
                    None if cluster_scope else namespace,
                    name,
                )
            )
        else:
            result.add(identity)
    return result


def discover_unregistered(
    kubectl: Kubectl,
    *,
    plane: str,
    context: str,
    namespace: str,
    registered: set[ResourceIdentity] | set[tuple[str, str]],
) -> list[dict[str, Any]]:
    found = []
    scoped_registered = _scoped_registered(registered, namespace)
    namespaced = kubectl.run(
        [
            "get",
            NAMESPACED_DISCOVERY_KINDS,
            "-A",
            "-o",
            "json",
            "--request-timeout=15s",
        ]
    )
    items = resource_list(namespaced.stdout)
    for item in items:
        metadata = item["metadata"]
        if (
            item["kind"].lower() not in NAMESPACED_DISCOVERY_KINDS.split(",")
            or not isinstance(metadata.get("namespace"), str)
            or not metadata["namespace"]
        ):
            raise RegistryError(
                "namespaced discovery returned an invalid scope or kind"
            )
    registered_cronjobs = {
        (
            item.get("apiVersion"),
            item["metadata"]["name"],
            item["metadata"].get("uid"),
            item["metadata"].get("namespace"),
        )
        for item in items
        if item.get("kind") == "CronJob"
        and isinstance(item.get("apiVersion"), str)
        and item["apiVersion"]
        and isinstance(item.get("metadata"), dict)
        and (
            "namespaced",
            "cronjob",
            item["metadata"].get("namespace"),
            item["metadata"].get("name"),
        )
        in scoped_registered
        and item["metadata"].get("namespace") == namespace
        and isinstance(item["metadata"].get("uid"), str)
        and item["metadata"]["uid"]
    }
    for item in items:
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
        if kind == "job":
            owners = metadata.get("ownerReferences") or []
            if (
                isinstance(owners, list)
                and len(owners) == 1
                and isinstance(owners[0], dict)
                and owners[0].get("kind") == "CronJob"
                and all(
                    isinstance(owners[0].get(key), str) and owners[0][key]
                    for key in ("apiVersion", "name", "uid")
                )
                and (
                    owners[0].get("apiVersion"),
                    owners[0].get("name"),
                    owners[0].get("uid"),
                    resource_namespace,
                )
                in registered_cronjobs
            ):
                # Exact parent UID covers transient Jobs through garbage collection.
                # Orphans, recreated parents and multiple owners remain visible.
                continue
        if ("namespaced", kind, resource_namespace, name) in scoped_registered:
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
            "--request-timeout=15s",
        ]
    )
    for item in resource_list(cluster.stdout):
        metadata = item.get("metadata") or {}
        name = metadata.get("name", "")
        kind = str(item.get("kind", "")).lower()
        if (
            kind not in CLUSTER_DISCOVERY_KINDS.split(",")
            or metadata.get("namespace") is not None
        ):
            raise RegistryError("cluster discovery returned an invalid scope or kind")
        if (
            not name.startswith("gpu-fault-")
            or ("cluster", kind, None, name) in scoped_registered
        ):
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
    namespaces = kubectl.run(
        ["get", "namespace", "-o", "json", "--request-timeout=15s"]
    )
    for item in resource_list(namespaces.stdout):
        name = (item.get("metadata") or {}).get("name", "")
        if item["kind"] != "Namespace" or item["metadata"].get("namespace") is not None:
            raise RegistryError("namespace discovery returned an invalid scope or kind")
        if (
            not name.startswith("gf-regional-")
            or ("cluster", "namespace", None, name) in scoped_registered
        ):
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
    clusters = config.get("clusters")
    if (
        not isinstance(clusters, list)
        or any(
            not isinstance(cluster, dict)
            or not isinstance(cluster.get("context"), str)
            or not cluster["context"]
            for cluster in clusters
        )
        or len({cluster["context"] for cluster in clusters}) != len(clusters)
    ):
        raise RegistryError("GPU registry contexts are missing or ambiguous")
    namespace = config.get("namespace", "gpu-fault-system")
    config["namespace"] = namespace
    release = config.get("release") or {}
    release_id = str(
        release.get("id") or Path(str(release.get("manifest") or "unknown")).stem
    )
    with closing(
        Kubectl(
            kubeconfig=config["cpu_kubeconfig"],
            context=None,
            reuse_exec_credential=True,
        )
    ) as cpu_kubectl:
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
            registered={
                resource_identity(resource, namespace) for resource in cpu["resources"]
            },
        )
    gpu_documents: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        with closing(
            Kubectl(
                kubeconfig=None,
                context=cluster["context"],
                reuse_exec_credential=True,
            )
        ) as gpu_kubectl:
            document = synchronize(
                gpu_kubectl,
                plane="gpu",
                namespace=namespace,
                inventory_path=inventory_path,
                release_id=release_id,
                apply=apply,
            )
            attach_workload_rbac(gpu_kubectl, config, cluster, document)
            gpu_documents[cluster["context"]] = document
            unregistered.extend(
                discover_unregistered(
                    gpu_kubectl,
                    plane="gpu",
                    context=cluster["context"],
                    namespace=namespace,
                    registered={
                        resource_identity(resource, namespace)
                        for resource in document["resources"]
                    },
                )
            )
    gpu_resources: dict[
        ResourceIdentity,
        dict[str, Any],
    ] = {}
    for document in gpu_documents.values():
        for resource in document["resources"]:
            identity = resource_identity(resource, namespace)
            existing = gpu_resources.get(identity)
            if existing is not None and existing != resource:
                raise RegistryError(
                    "GPU installed registries disagree for a scoped identity"
                )
            gpu_resources[identity] = resource

    result: dict[str, Any] = {
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
            "by_context": gpu_documents,
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
    try:
        raise SystemExit(main())
    except RegistryError as exc:
        raise SystemExit(f"installed registry collection refused: {exc}") from None
