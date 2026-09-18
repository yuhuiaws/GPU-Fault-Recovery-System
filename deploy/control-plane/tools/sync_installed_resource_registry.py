#!/usr/bin/env python3
"""Synchronize the live installed-resource registry ConfigMap."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from gpu_fault.admin.execution import deadline_scope, remaining_timeout, run_command
from gpu_fault_release.regional_resource_probe import (
    ProbeState,
    ResourceRef,
    probe_resource,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY = (
    ROOT / "deploy" / "control-plane" / "regional" / "cleanup-inventory.json"
)
REGISTRY_NAME = "gpu-fault-installed-resources"
COMMAND_TIMEOUT_SECONDS = 30
CREDENTIAL_REFRESH_SECONDS = 60
ResourceIdentity = tuple[str, str, str | None, str]


class RegistryError(RuntimeError):
    pass


def json_object(text: str, description: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        raise RegistryError(f"{description} is not a JSON object") from None
    if not isinstance(value, dict):
        raise RegistryError(f"{description} is not a JSON object")
    return cast(dict[str, Any], value)


class Kubectl:
    """Bound client calls; only plain EKS get-token credentials are cached here."""

    def __init__(
        self,
        *,
        kubeconfig: str | None,
        context: str | None,
        reuse_exec_credential: bool = False,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        if kubeconfig and context:
            raise RegistryError("--kubeconfig and --context are mutually exclusive")
        self.runner = runner or run_command
        self.session_directory: tempfile.TemporaryDirectory[str] | None = None
        self.exec_document: dict[str, Any] | None = None
        self.credential_refresh_at: float | None = None
        self.credential_expires_at: float | None = None
        self.prefix = ["kubectl"]
        if kubeconfig:
            self.prefix.extend(["--kubeconfig", kubeconfig])
        elif context:
            self.prefix.extend(["--context", str(context)])
        if reuse_exec_credential:
            self._reuse_exec_credential()

    def _execute(
        self,
        arguments: list[str],
        *,
        description: str,
        input_text: str | None = None,
        environment: dict[str, str] | None = None,
        timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        try:
            with deadline_scope(description, timeout_seconds) as deadline:
                completed = self.runner(
                    arguments,
                    input_text=input_text,
                    environment=environment,
                    capture=True,
                    timeout_seconds=remaining_timeout(timeout_seconds),
                )
                deadline.remaining()
        except (subprocess.TimeoutExpired, TimeoutError):
            raise RegistryError(f"{description} exceeded its deadline") from None
        except (subprocess.SubprocessError, OSError):
            raise RegistryError(f"{description} could not be executed") from None
        return completed

    def _reuse_exec_credential(self) -> None:
        completed = self._execute(
            [
                *self.prefix,
                "config",
                "view",
                "--raw",
                "--minify",
                "--flatten",
                "-o",
                "json",
            ],
            description="selected kubeconfig read",
        )
        if completed.returncode:
            raise RegistryError("cannot read the selected kubeconfig")
        document = json_object(completed.stdout, "selected kubeconfig")
        users = document.get("users") or []
        if not isinstance(users, list):
            raise RegistryError("selected kubeconfig users are invalid")
        if not users:
            return
        if len(users) != 1:
            raise RegistryError("selected kubeconfig must have one user")
        if not isinstance(users[0], dict):
            raise RegistryError("selected kubeconfig user is invalid")
        user = users[0].get("user") or {}
        if not isinstance(user, dict):
            raise RegistryError("selected kubeconfig user is invalid")
        command = user.get("exec")
        if command is None:
            return
        if not isinstance(command, dict):
            raise RegistryError("kubeconfig exec credential is invalid")
        arguments = command.get("args")
        if (
            Path(str(command.get("command") or "")).name != "aws"
            or not isinstance(arguments, list)
            or not any(
                arguments[index : index + 2] == ["eks", "get-token"]
                for index in range(len(arguments) - 1)
            )
            or command.get("provideClusterInfo", False) is not False
            or command.get("interactiveMode") not in (None, "Never", "IfAvailable")
            or command.get("apiVersion")
            not in (
                "client.authentication.k8s.io/v1",
                "client.authentication.k8s.io/v1beta1",
            )
        ):
            # Native kubectl owns the full exec protocol (KUBERNETES_EXEC_INFO,
            # interactiveMode, cluster info and certificate credentials).
            return
        self.exec_document = document

    def _refresh_exec_credential(self) -> None:
        assert self.exec_document is not None
        command = self.exec_document["users"][0]["user"]["exec"]
        executable = command.get("command")
        arguments = command.get("args", [])
        if (
            not isinstance(executable, str)
            or not executable.strip()
            or not isinstance(arguments, list)
            or any(not isinstance(item, str) for item in arguments)
        ):
            raise RegistryError("kubeconfig exec credential command is invalid")
        environment = dict(os.environ)
        entries = command.get("env") or []
        if not isinstance(entries, list):
            raise RegistryError("kubeconfig exec credential environment is invalid")
        for item in entries:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"]
                or not isinstance(item.get("value"), str)
            ):
                raise RegistryError("kubeconfig exec credential environment is invalid")
            environment[item["name"]] = item["value"]
        credential = self._execute(
            [executable, *arguments],
            description="kubeconfig exec credential",
            environment=environment,
        )
        if credential.returncode:
            raise RegistryError("kubeconfig exec credential failed")
        value = json_object(credential.stdout, "kubeconfig exec credential")
        status = value.get("status")
        if not isinstance(status, dict):
            raise RegistryError("kubeconfig exec credential status is invalid")
        token = status.get("token")
        if not isinstance(token, str) or not token:
            raise RegistryError("kubeconfig exec credential returned no token")
        refresh_at = None
        expires_at = None
        if "expirationTimestamp" in status:
            try:
                expires = datetime.fromisoformat(status["expirationTimestamp"])
                if expires.tzinfo is None:
                    raise ValueError
                expires_at = expires.timestamp()
                remaining = expires_at - time.time()
            except (TypeError, ValueError, OverflowError):
                raise RegistryError(
                    "kubeconfig exec credential expiry is invalid"
                ) from None
            if remaining <= COMMAND_TIMEOUT_SECONDS:
                raise RegistryError("kubeconfig exec credential expires too soon")
            refresh_at = time.monotonic() + max(
                0, remaining - CREDENTIAL_REFRESH_SECONDS
            )
        # Without an expiry the plugin must run again for each command.
        document = {
            **self.exec_document,
            "users": [
                {
                    "name": self.exec_document["users"][0].get("name"),
                    "user": {"token": token},
                }
            ],
        }
        if self.session_directory is None:
            self.session_directory = tempfile.TemporaryDirectory(
                prefix="gpu-fault-kube-session-"
            )
        path = Path(self.session_directory.name) / "config"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.session_directory.name,
                delete=False,
            ) as stream:
                json.dump(document, stream)
                temporary_path = Path(stream.name)
            temporary_path.replace(path)
        except OSError:
            raise RegistryError(
                "cannot write the private kubeconfig credential"
            ) from None
        self.credential_refresh_at = refresh_at
        self.credential_expires_at = expires_at
        self.prefix = ["kubectl", "--kubeconfig", str(path)]

    def close(self) -> None:
        if self.session_directory is not None:
            self.session_directory.cleanup()
            self.session_directory = None
        self.credential_refresh_at = None
        self.credential_expires_at = None

    def run(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout_seconds: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        timeout = (
            COMMAND_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        with deadline_scope(
            "installed registry kubectl",
            timeout,
        ):
            if self.exec_document is not None and (
                self.credential_refresh_at is None
                or time.monotonic() >= self.credential_refresh_at
                or self.credential_expires_at is not None
                and time.time()
                >= self.credential_expires_at - CREDENTIAL_REFRESH_SECONDS
            ):
                self._refresh_exec_credential()
            completed = self._execute(
                [*self.prefix, *arguments],
                description="installed registry kubectl",
                input_text=input_text,
                timeout_seconds=timeout,
            )
        if check and completed.returncode:
            raise RegistryError(
                f"installed registry kubectl exited {completed.returncode}"
            )
        return completed

    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        completed = self.run(args, check=False, timeout_seconds=timeout_seconds)
        return (
            completed.returncode,
            completed.stdout,
            "installed registry resource read failed" if completed.returncode else "",
        )


def _registry(
    kubectl: Kubectl,
    namespace: str,
) -> dict[str, Any] | None:
    observation = probe_resource(
        kubectl, [], ResourceRef("configmap", "ConfigMap", REGISTRY_NAME, namespace)
    )
    if observation.state is ProbeState.ERROR:
        raise RegistryError("cannot verify installed registry identity or absence")
    if observation.state is ProbeState.ABSENT:
        return None
    config_map = observation.document or {}
    data = config_map.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("inventory.json"), str):
        raise RegistryError(f"{REGISTRY_NAME} lacks inventory.json")
    value = json_object(data["inventory.json"], "installed registry inventory")
    if value.get("schema_version") != 1:
        raise RegistryError("installed registry schema is unsupported")
    return value


def resource_list(output: str) -> list[dict[str, Any]]:
    value = json_object(output, "Kubernetes resource list")
    items = value.get("items")
    if not isinstance(items, list):
        raise RegistryError("Kubernetes resource list has no items")
    for item in items:
        if not isinstance(item, dict):
            raise RegistryError("Kubernetes resource list has an invalid item")
        metadata = item.get("metadata")
        if (
            not isinstance(item.get("kind"), str)
            or not item["kind"]
            or not isinstance(metadata, dict)
            or not isinstance(metadata.get("name"), str)
            or not metadata["name"]
        ):
            raise RegistryError("Kubernetes resource list has an invalid identity")
    return cast(list[dict[str, Any]], items)


def _listed_names(
    kubectl: Kubectl,
    namespace: str | None,
    *,
    scope: str,
    kind: str,
) -> set[str]:
    if kind == "prometheusrule":
        observation = probe_resource(
            kubectl,
            [],
            ResourceRef(
                "customresourcedefinition",
                "CustomResourceDefinition",
                "prometheusrules.monitoring.coreos.com",
                None,
            ),
        )
        if observation.state is ProbeState.ERROR:
            raise RegistryError("cannot verify the optional PrometheusRule API")
        if observation.state is ProbeState.ABSENT:
            return set()
    arguments: list[str] = []
    if scope == "namespaced":
        assert namespace is not None
        arguments.extend(["-n", namespace])
    arguments.extend(["get", kind, "-o", "json", "--request-timeout=15s"])
    completed = kubectl.run(arguments, check=False)
    if completed.returncode:
        raise RegistryError(f"cannot list installed registry resource kind {kind}")
    items = resource_list(completed.stdout)
    if any(
        item["kind"].lower() != kind or item["metadata"].get("namespace") != namespace
        for item in items
    ):
        raise RegistryError(
            f"kubectl list {kind} returned a different kind or namespace"
        )
    return {item["metadata"]["name"] for item in items}


def _live_identities(
    kubectl: Kubectl,
    namespace: str,
    resources: list[dict[str, Any]],
) -> set[ResourceIdentity]:
    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for resource in resources:
        scope, kind, resource_namespace, _ = resource_identity(resource, namespace)
        grouped[(scope, kind, resource_namespace)].append(resource)
    live: set[ResourceIdentity] = set()
    for (scope, kind, resource_namespace), members in grouped.items():
        names = _listed_names(
            kubectl,
            resource_namespace,
            scope=scope,
            kind=kind,
        )
        live.update(
            resource_identity(resource, namespace)
            for resource in members
            if resource["name"] in names
        )
    return live


def resource_identity(resource: dict[str, Any], namespace: str) -> ResourceIdentity:
    scope = resource.get("scope")
    kind, name = resource.get("kind"), resource.get("name")
    if (
        not isinstance(scope, str)
        or scope not in {"namespaced", "cluster"}
        or not isinstance(kind, str)
        or not kind
        or not isinstance(name, str)
        or not name
    ):
        raise RegistryError("installed resource identity is invalid")
    resource_namespace = resource.get(
        "namespace", namespace if scope == "namespaced" else None
    )
    if (
        scope == "namespaced"
        and (not isinstance(resource_namespace, str) or not resource_namespace)
        or scope == "cluster"
        and resource_namespace is not None
    ):
        raise RegistryError("installed resource namespace does not match its scope")
    return scope, kind, resource_namespace, name


PROVENANCE_KEY = "provenance_sha256"


def _canonical_payload(resource: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in resource.items() if key != PROVENANCE_KEY}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _provenance_digest(resource: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(resource)).hexdigest()


def stamp_provenance(resource: dict[str, Any]) -> dict[str, Any]:
    """Seal the payload for integrity diagnostics, not deletion authorization."""

    stamped = {key: value for key, value in resource.items() if key != PROVENANCE_KEY}
    stamped[PROVENANCE_KEY] = _provenance_digest(stamped)
    return stamped


def provenance_is_verified(resource: dict[str, Any]) -> bool:
    """Check payload integrity only; an unkeyed digest does not prove ownership."""
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
    source = json_object(source_text, "source inventory")
    if source.get("schema_version") != 1 or plane not in {"cpu", "gpu"}:
        raise RegistryError("source inventory schema or plane is unsupported")
    section = source.get(plane)
    if not isinstance(section, dict) or not isinstance(section.get("resources"), list):
        raise RegistryError("source inventory resources are invalid")
    # Only the trusted release inventory authorizes cleanup, including explicit
    # retired rows. A mutable ConfigMap and its self-hash cannot grant authority.
    candidates: dict[ResourceIdentity, dict[str, Any]] = {}
    for resource in section["resources"]:
        if not isinstance(resource, dict):
            raise RegistryError("source inventory resource is invalid")
        identity = resource_identity(resource, namespace)
        if identity[2] not in {None, namespace}:
            raise RegistryError("source inventory targets a different namespace")
        if identity in candidates:
            raise RegistryError(
                "source inventory contains duplicate resource identities"
            )
        candidates[identity] = stamp_provenance({**resource, "namespace": identity[2]})
    existing_document = _registry(kubectl, namespace)
    if existing_document is not None:
        if existing_document.get("plane") != plane:
            raise RegistryError(f"installed registry plane is not {plane}")
        if existing_document.get("namespace") != namespace:
            raise RegistryError(
                "installed registry namespace does not match the target"
            )
        existing = existing_document.get("resources")
        if not isinstance(existing, list):
            raise RegistryError("installed registry resources are invalid")
        seen: set[ResourceIdentity] = set()
        for index, resource in enumerate(existing):
            if not isinstance(resource, dict):
                raise RegistryError("installed registry resource is invalid")
            identity = resource_identity(resource, namespace)
            if identity in seen:
                raise RegistryError(
                    "installed registry contains duplicate resource identities"
                )
            seen.add(identity)
            if identity not in candidates:
                raise RegistryError(
                    f"installed registry resource at index {index} has no current "
                    "source authorization; registry left unchanged; review and add "
                    "an explicit retired entry to the trusted release inventory"
                )
    resources = list(candidates.values())
    live = _live_identities(kubectl, namespace, resources)
    installed = [
        resource
        for resource in resources
        if resource_identity(resource, namespace) in live
    ]
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
    with closing(
        Kubectl(
            kubeconfig=args.kubeconfig,
            context=args.context,
            reuse_exec_credential=True,
        )
    ) as kubectl:
        document = synchronize(
            kubectl,
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
    try:
        raise SystemExit(main())
    except RegistryError as exc:
        raise SystemExit(f"installed registry sync refused: {exc}") from None
