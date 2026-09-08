#!/usr/bin/env python3
"""Synchronize the live installed-resource registry ConfigMap."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
        reuse_exec_credential: bool = False,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if kubeconfig and context:
            raise RegistryError("--kubeconfig and --context are mutually exclusive")
        self.runner = runner
        self.session_directory: tempfile.TemporaryDirectory[str] | None = None
        self.prefix = ["kubectl"]
        if kubeconfig:
            self.prefix.extend(["--kubeconfig", kubeconfig])
        elif context:
            self.prefix.extend(["--context", str(context)])
        if reuse_exec_credential:
            self._reuse_exec_credential()

    def _reuse_exec_credential(self) -> None:
        completed = self.runner(
            [*self.prefix, "config", "view", "--raw", "--minify", "-o", "json"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RegistryError(
                completed.stderr.strip() or "cannot read the selected kubeconfig"
            )
        try:
            document = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return
        users = document.get("users") or []
        if len(users) != 1:
            return
        user = users[0].get("user") or {}
        command = user.get("exec")
        if not isinstance(command, dict):
            return
        executable = str(command.get("command") or "").strip()
        if not executable:
            raise RegistryError("kubeconfig exec credential has no command")
        environment = dict(os.environ)
        for item in command.get("env") or []:
            name = str(item.get("name") or "")
            if name:
                environment[name] = str(item.get("value") or "")
        credential = self.runner(
            [executable, *[str(item) for item in command.get("args") or []]],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if credential.returncode:
            raise RegistryError(
                credential.stderr.strip() or "kubeconfig exec credential failed"
            )
        value = json.loads(credential.stdout)
        token = str((value.get("status") or {}).get("token") or "")
        if not token:
            raise RegistryError("kubeconfig exec credential returned no token")
        users[0]["user"] = {"token": token}
        self.session_directory = tempfile.TemporaryDirectory(
            prefix="gpu-fault-kube-session-"
        )
        path = Path(self.session_directory.name) / "config"
        path.write_text(json.dumps(document), encoding="utf-8")
        path.chmod(0o600)
        self.prefix = ["kubectl", "--kubeconfig", str(path)]

    def run(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = self.runner(
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


def _listed_names(
    kubectl: Kubectl,
    namespace: str,
    *,
    scope: str,
    kind: str,
) -> set[str]:
    arguments: list[str] = []
    if scope == "namespaced":
        arguments.extend(["-n", namespace])
    arguments.extend(["get", kind, "-o", "json"])
    completed = kubectl.run(arguments, check=False)
    if completed.returncode:
        if "the server doesn't have a resource type" in completed.stderr:
            return set()
        raise RegistryError(completed.stderr.strip())
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise RegistryError(f"kubectl list {kind} returned an invalid JSON document")
    return {
        str((item.get("metadata") or {}).get("name") or "")
        for item in value.get("items", [])
    }


def _live_identities(
    kubectl: Kubectl,
    namespace: str,
    resources: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for resource in resources:
        grouped[(resource["scope"], resource["kind"])].append(resource)
    live: set[tuple[str, str]] = set()
    for (scope, kind), members in grouped.items():
        names = _listed_names(
            kubectl,
            namespace,
            scope=scope,
            kind=kind,
        )
        live.update(
            _identity(resource) for resource in members if resource["name"] in names
        )
    return live


def _identity(resource: dict[str, Any]) -> tuple[str, str]:
    return resource["kind"], resource["name"]


PROVENANCE_KEY = "provenance_sha256"


def _canonical_payload(resource: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in resource.items() if key != PROVENANCE_KEY}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _provenance_digest(resource: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(resource)).hexdigest()


def stamp_provenance(resource: dict[str, Any]) -> dict[str, Any]:
    """Return the resource carrying a provenance digest over its full payload.

    The digest covers every field except the digest itself, so any later edit
    to a recorded entry -- an injected resource, a flipped `clean` verb, a
    renamed target -- no longer matches and the entry stops being authoritative.
    """

    stamped = {key: value for key, value in resource.items() if key != PROVENANCE_KEY}
    stamped[PROVENANCE_KEY] = _provenance_digest(stamped)
    return stamped


def provenance_is_verified(resource: dict[str, Any]) -> bool:
    recorded = resource.get(PROVENANCE_KEY)
    return bool(recorded) and recorded == _provenance_digest(resource)


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
    # Candidates come from the design inventory, whose integrity the document's
    # `source_sha256` already covers, so each is stamped with a provenance digest
    # over its own payload as it is admitted.
    candidates = {
        _identity(resource): stamp_provenance(resource)
        for resource in source[plane]["resources"]
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
    resources = []
    for identity in identities:
        candidate = candidates.get(identity)
        if candidate is not None:
            resources.append(candidate)
            continue
        # The design inventory no longer names this resource, so the only record
        # of it is the mutable live ConfigMap. Retain it as authoritative only
        # when its recorded provenance re-verifies against its own payload;
        # otherwise a hand-edited or injected entry would be trusted without
        # provenance. Drop it (rather than raise) so a forged entry cannot wedge
        # every subsequent sync.
        retained = existing[identity]
        if provenance_is_verified(retained):
            resources.append(retained)
    live = _live_identities(kubectl, namespace, resources)
    installed = [resource for resource in resources if _identity(resource) in live]
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
            reuse_exec_credential=True,
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
