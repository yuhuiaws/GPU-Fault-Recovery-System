#!/usr/bin/env python3
"""Synchronize the live installed-resource registry ConfigMap."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY = (
    ROOT / "deploy" / "control-plane" / "regional" / "cleanup-inventory.json"
)
REGISTRY_NAME = "gpu-fault-installed-resources"


class RegistryError(RuntimeError):
    pass


class Kubectl:
    def __init__(
        self,
        *,
        kubeconfig: str | None,
        context: str | None,
    ) -> None:
        if kubeconfig and context:
            raise RegistryError("--kubeconfig and --context are mutually exclusive")
        self.prefix = ["kubectl"]
        if kubeconfig:
            self.prefix.extend(["--kubeconfig", kubeconfig])
        elif context:
            self.prefix.extend(["--context", str(context)])

    def run(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [*self.prefix, *arguments],
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
        if check and completed.returncode:
            raise RegistryError(
                completed.stderr.strip() or f"kubectl exited {completed.returncode}"
            )
        return completed


def _registry(
    kubectl: Kubectl,
    namespace: str,
) -> dict[str, Any] | None:
    completed = kubectl.run(
        [
            "-n",
            namespace,
            "get",
            "configmap",
            REGISTRY_NAME,
            "-o",
            "json",
        ],
        check=False,
    )
    if completed.returncode:
        if "NotFound" in completed.stderr:
            return None
        raise RegistryError(completed.stderr.strip())
    config_map = json.loads(completed.stdout)
    raw = (config_map.get("data") or {}).get("inventory.json")
    if not raw:
        raise RegistryError(f"{REGISTRY_NAME} lacks inventory.json")
    value = json.loads(raw)
    if value.get("schema_version") != 1:
        raise RegistryError("installed registry schema is unsupported")
    return value


def _exists(
    kubectl: Kubectl,
    namespace: str,
    resource: dict[str, Any],
) -> bool:
    arguments = []
    if resource["scope"] == "namespaced":
        arguments.extend(["-n", namespace])
    arguments.extend(
        [
            "get",
            resource["kind"],
            resource["name"],
            "-o",
            "name",
        ]
    )
    completed = kubectl.run(arguments, check=False)
    if completed.returncode == 0:
        return True
    if "NotFound" in completed.stderr:
        return False
    if "the server doesn't have a resource type" in completed.stderr:
        return False
    raise RegistryError(completed.stderr.strip())


def _identity(resource: dict[str, Any]) -> tuple[str, str]:
    return resource["kind"], resource["name"]


def synchronize(
    kubectl: Kubectl,
    *,
    plane: str,
    namespace: str,
    inventory_path: Path,
    release_id: str,
    apply: bool,
) -> dict[str, Any]:
    source_text = inventory_path.read_text(encoding="utf-8")
    source = json.loads(source_text)
    candidates = {
        _identity(resource): resource for resource in source[plane]["resources"]
    }
    existing_document = _registry(kubectl, namespace)
    if existing_document is not None and existing_document.get("plane") != plane:
        raise RegistryError(f"installed registry plane is not {plane}")
    existing = {
        _identity(resource): resource
        for resource in ((existing_document or {}).get("resources") or [])
    }
    identities = [
        *candidates,
        *sorted(set(existing) - set(candidates)),
    ]
    installed = []
    for identity in identities:
        resource = candidates.get(identity) or existing[identity]
        if _exists(kubectl, namespace, resource):
            installed.append(resource)
    document = {
        "schema_version": 1,
        "plane": plane,
        "namespace": namespace,
        "release_id": release_id,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "resources": installed,
    }
    if not apply:
        return document
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": REGISTRY_NAME,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "gpu-fault",
                "gpu-fault.io/resource-registry": "installed",
            },
        },
        "data": {
            "schema-version": "1",
            "plane": plane,
            "inventory.json": json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    }
    kubectl.run(
        ["apply", "-f", "-"],
        input_text=json.dumps(config_map),
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plane", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--namespace", default="gpu-fault-system")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--kubeconfig")
    target.add_argument("--context")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
    )
    parser.add_argument("--release-id", default="unknown")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    document = synchronize(
        Kubectl(
            kubeconfig=args.kubeconfig,
            context=args.context,
        ),
        plane=args.plane,
        namespace=args.namespace,
        inventory_path=args.inventory,
        release_id=args.release_id,
        apply=not args.dry_run,
    )
    print(
        json.dumps(
            {
                "plane": args.plane,
                "resources": len(document["resources"]),
                "registry": REGISTRY_NAME,
                "written": not args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
