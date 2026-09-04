from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from regional_release_config import ClusterTarget, ReleaseError

REGISTRY_SECRET = "gpu-fault-regional-clusters"
REGISTRY_CURRENT_KEY = "clusters.json"
REGISTRY_BACKUP_KEY = "previous-clusters.json"


def registry_config_digest(clusters: tuple[ClusterTarget, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                {
                    "cluster_id": item.cluster_id,
                    "region": item.region,
                    "hyperpod_cluster_name": item.hyperpod_cluster_name,
                    "eks_cluster_arn": item.eks_cluster_arn,
                    "allowed_namespaces": sorted(item.allowed_namespaces),
                    "agent_endpoint_allowed_cidrs": sorted(
                        item.agent_endpoint_allowed_cidrs
                    ),
                }
                for item in sorted(clusters, key=lambda target: target.cluster_id)
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def registry_payloads(
    release: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    exists = release.runner.probe(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "secret",
            REGISTRY_SECRET,
        ),
    )
    if not exists:
        return [], None
    value = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "secret",
            REGISTRY_SECRET,
        )
    )
    return _registry_payloads_from_secret(value)


def _registry_payloads_from_secret(
    value: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    data = value.get("data") or {}

    def decode(key: str) -> list[dict[str, Any]] | None:
        encoded = data.get(key)
        if not encoded:
            return None
        decoded = json.loads(base64.b64decode(encoded))
        if not isinstance(decoded, list) or not all(
            isinstance(item, dict) for item in decoded
        ):
            raise ReleaseError(f"{REGISTRY_SECRET}/{key} is invalid")
        return [dict(item) for item in decoded]

    return decode(REGISTRY_CURRENT_KEY) or [], decode(REGISTRY_BACKUP_KEY)


def registry(release: Any) -> list[dict[str, Any]]:
    current, _backup = registry_payloads(release)
    return current


def registry_entry(target: ClusterTarget) -> dict[str, Any]:
    if not all(
        (
            target.token_file,
            target.hyperpod_cluster_name,
            target.eks_cluster_arn,
        )
    ):
        raise ReleaseError(
            "regional cluster registry requires token_file, "
            "hyperpod_cluster_name, and eks_cluster_arn"
        )
    try:
        endpoint_cidrs = sorted(
            {
                str(ipaddress.ip_network(value, strict=False))
                for value in target.agent_endpoint_allowed_cidrs
            }
        )
    except ValueError as exc:
        raise ReleaseError(
            f"{target.cluster_id} Agent endpoint CIDRs are invalid"
        ) from exc
    if not endpoint_cidrs:
        raise ReleaseError(f"{target.cluster_id} requires Agent endpoint CIDRs")
    token = Path(target.token_file).read_text().strip()
    if len(token) < 32:
        raise ReleaseError("cluster token is too short")
    return {
        "cluster_id": target.cluster_id,
        "region": target.region,
        "hyperpod_cluster_name": target.hyperpod_cluster_name,
        "eks_cluster_arn": target.eks_cluster_arn,
        "token": token,
        "allowed_namespaces": list(target.allowed_namespaces),
        "agent_endpoint_allowed_cidrs": endpoint_cidrs,
    }


def desired_registry(release: Any) -> list[dict[str, Any]]:
    return sorted(
        (registry_entry(target) for target in release.config.clusters),
        key=lambda item: str(item["cluster_id"]),
    )


def write_registry(
    release: Any,
    registrations: list[dict[str, Any]],
    *,
    backup: list[dict[str, Any]] | None = None,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        current_path = Path(directory) / REGISTRY_CURRENT_KEY
        current_path.write_text(
            json.dumps(registrations, indent=2),
            encoding="utf-8",
        )
        current_path.chmod(0o600)
        arguments = release._cpu(
            "-n",
            release.config.namespace,
            "create",
            "secret",
            "generic",
            REGISTRY_SECRET,
            f"--from-file={REGISTRY_CURRENT_KEY}={current_path}",
        )
        if backup is not None:
            backup_path = Path(directory) / REGISTRY_BACKUP_KEY
            backup_path.write_text(
                json.dumps(backup, indent=2),
                encoding="utf-8",
            )
            backup_path.chmod(0o600)
            arguments.append(f"--from-file={REGISTRY_BACKUP_KEY}={backup_path}")
        arguments.extend(("--dry-run=client", "-o", "yaml"))
        rendered = release.runner.run(
            arguments,
            capture=True,
            sensitive=True,
        )
        release.runner.run(
            release._cpu("apply", "-f", "-"),
            input_text=rendered,
            sensitive=True,
        )
    if release.runner.dry_run:
        return
    current, persisted_backup = registry_payloads(release)
    if current != registrations or persisted_backup != backup:
        raise ReleaseError("regional cluster registry write did not persist")


def initialize_registry(
    release: Any,
    *,
    load: Callable[
        [Any],
        tuple[list[dict[str, Any]], list[dict[str, Any]] | None],
    ] = registry_payloads,
    desired: Callable[[Any], list[dict[str, Any]]] = desired_registry,
    write: Callable[..., None] = write_registry,
) -> None:
    current, backup = load(release)
    if backup is not None:
        raise ReleaseError("regional cluster registry has an unfinished release backup")
    target = desired(release)
    if current != target:
        write(release, target)


def stage_registry(release: Any) -> bool:
    current, backup = registry_payloads(release)
    desired = desired_registry(release)
    if current == desired and backup is None:
        return False
    write_registry(
        release,
        desired,
        backup=current if backup is None else backup,
    )
    return True


def commit_registry_update(release: Any) -> None:
    current, backup = registry_payloads(release)
    if backup is not None:
        write_registry(release, current)


def restore_registry_backup(release: Any) -> bool:
    _current, backup = registry_payloads(release)
    if backup is None:
        return False
    write_registry(release, backup)
    return True


def update_registry(
    release: Any,
    target: ClusterTarget,
    *,
    remove: bool,
) -> None:
    for attempt in range(8):
        secret = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "secret",
                REGISTRY_SECRET,
            )
        )
        registrations, backup = _registry_payloads_from_secret(secret)
        if backup is not None:
            raise ReleaseError(
                "regional cluster registry has an unfinished release backup"
            )
        remaining = [
            item
            for item in registrations
            if item.get("cluster_id") != target.cluster_id
        ]
        desired = (
            remaining
            if remove
            else sorted(
                [*remaining, registry_entry(target)],
                key=lambda item: str(item["cluster_id"]),
            )
        )
        if registrations == desired:
            return
        metadata = secret.get("metadata") or {}
        resource_version = str(metadata.get("resourceVersion") or "")
        if not resource_version:
            raise ReleaseError(
                "regional cluster registry Secret has no resourceVersion"
            )
        data = secret.get("data") or {}
        operation = "replace" if REGISTRY_CURRENT_KEY in data else "add"
        patch = [
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": resource_version,
            },
            {
                "op": operation,
                "path": f"/data/{REGISTRY_CURRENT_KEY}",
                "value": base64.b64encode(
                    json.dumps(
                        desired,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).decode(),
            },
        ]
        try:
            release.runner.run(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "patch",
                    "secret",
                    REGISTRY_SECRET,
                    "--type=json",
                    "-p",
                    json.dumps(patch, separators=(",", ":")),
                ),
                sensitive=True,
            )
        except ReleaseError as exc:
            if attempt == 7:
                raise ReleaseError(
                    "regional cluster registry changed repeatedly during update"
                ) from exc
            continue
        return
