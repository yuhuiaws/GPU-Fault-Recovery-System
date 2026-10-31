"""The release's backups of the Secrets a rollback restores.

Before the first mutation the upgrade copies the CPU email Secret and every GPU
cluster's connection Secret into labelled ``-rollback-<release id>`` backups;
the rollback restores from them, a verified rollback deletes them, and the
commit sweeps every labelled backup the committed release does not reference.
Moved out of ``regional_release_fleet_rollout`` (which re-exports every name)
unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

# Every release Secret backup carries this label; the commit-time sweep lists
# by it so nothing but our own backups can ever be deleted.
RELEASE_SECRET_BACKUP_LABEL = "gpu-fault.io/release-secret-backup"


def backup_secret(
    release: Any,
    kubectl: list[str],
    *,
    source: str,
    backup: str,
    required: bool,
) -> str | None:
    value = probe_resource(
        release.runner,
        kubectl,
        ResourceRef("secret", "Secret", source, release.config.namespace),
    ).require_readable()
    if value is None:
        if required:
            raise ReleaseError(f"required Secret is missing: {source}")
        return None
    data = value.get("data") or {}
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in data.items()
    ):
        raise ReleaseError("Secret backup source data is invalid")
    document = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": backup,
            "namespace": release.config.namespace,
            "labels": {
                RELEASE_SECRET_BACKUP_LABEL: "true",
            },
            "annotations": {
                "gpu-fault.io/source-secret": source,
                "gpu-fault.io/release-id": release.release_id,
            },
        },
        "type": value.get("type") or "Opaque",
        "data": dict(data),
    }
    release.runner.run(
        kubectl + ["apply", "-f", "-"],
        input_text=json.dumps(document),
        sensitive=True,
    )
    return backup


def restore_secret(
    release: Any,
    kubectl: list[str],
    *,
    source: str,
    backup: str,
) -> dict[str, str]:
    """Put the backup's data back under ``source``; return that data."""

    value = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "secret",
            backup,
        ]
    )
    if not value.get("data"):
        raise ReleaseError(f"release Secret backup is missing: {backup}")
    data = {str(key): str(item) for key, item in (value.get("data") or {}).items()}
    document = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": source,
            "namespace": release.config.namespace,
        },
        "type": value.get("type") or "Opaque",
        "data": data,
    }
    release.runner.run(
        kubectl + ["apply", "-f", "-"],
        input_text=json.dumps(document),
        sensitive=True,
    )
    return data


def backup_release_secrets(release: Any) -> dict[str, Any]:
    created: list[tuple[list[str], str]] = []
    try:
        cpu_backup = release._backup_secret(
            release._cpu(),
            source="gpu-fault-email",
            backup=f"gpu-fault-email-rollback-{release.release_id}",
            required=False,
        )
        if cpu_backup:
            created.append((release._cpu(), cpu_backup))
        clusters: dict[str, dict[str, str]] = {}
        for target in release.config.clusters:
            backup = release._backup_secret(
                release._gpu(target),
                source="gpu-fault-regional-connection",
                backup=("gpu-fault-regional-connection-rollback-" + release.release_id),
                required=True,
            )
            if backup is None:
                raise ReleaseError(
                    f"{target.cluster_id} connection Secret backup failed"
                )
            created.append((release._gpu(target), backup))
            clusters[target.cluster_id] = {
                "source": "gpu-fault-regional-connection",
                "backup": backup,
            }
        return {
            "cpu": (
                {
                    "source": "gpu-fault-email",
                    "backup": cpu_backup,
                }
                if cpu_backup
                else None
            ),
            "clusters": clusters,
        }
    except Exception:
        for kubectl, name in created:
            release.runner.run(
                kubectl
                + [
                    "-n",
                    release.config.namespace,
                    "delete",
                    "secret",
                    name,
                    "--ignore-not-found",
                ],
                sensitive=True,
            )
        raise


def delete_release_secret_backups(
    release: Any,
    previous: dict[str, Any],
) -> None:
    """Delete the backups ``previous`` references (after a verified rollback).

    Only the clusters still in the site are contacted, so a cluster removed
    since the backup was taken cannot fail the cleanup.
    """

    backups = previous.get("secret_backups") or {}
    entries: list[tuple[list[str], str]] = []
    cpu = backups.get("cpu") or {}
    if cpu.get("backup"):
        entries.append((release._cpu(), str(cpu["backup"])))
    by_cluster = backups.get("clusters") or {}
    for target in release.config.clusters:
        item = by_cluster.get(target.cluster_id) or {}
        if item.get("backup"):
            entries.append((release._gpu(target), str(item["backup"])))
    for kubectl, name in entries:
        release.runner.run(
            kubectl
            + [
                "-n",
                release.config.namespace,
                "delete",
                "secret",
                name,
                "--ignore-not-found",
            ],
            sensitive=True,
        )


def referenced_secret_backups(previous: dict[str, Any]) -> set[str] | None:
    """Backup Secret names the ``previous`` snapshot still needs.

    ``None`` when the snapshot recorded no backups at all: without a reference
    there is no way to tell a needed backup from a stale one.
    """

    backups = previous.get("secret_backups")
    if not isinstance(backups, dict):
        return None
    names: set[str] = set()
    cpu = backups.get("cpu")
    if isinstance(cpu, dict) and cpu.get("backup"):
        names.add(str(cpu["backup"]))
    clusters = backups.get("clusters")
    for item in (clusters or {}).values() if isinstance(clusters, dict) else ():
        if isinstance(item, dict) and item.get("backup"):
            names.add(str(item["backup"]))
    return names


def delete_stale_release_secret_backups(
    release: Any,
    previous: dict[str, Any],
) -> list[str]:
    """Commit-time sweep: keep what ``previous`` references, delete the rest.

    A committed release keeps its own backups so that ``deploy --rollback``
    can still restore the CPU email and GPU connection Secrets after the
    commit; what goes is every *other* labelled backup -- the release before
    it, or a transaction that never committed. The referenced names are kept
    in every context (a CPU and a GPU context may be one cluster), and only
    the clusters currently in the site are contacted. Returns the names
    deleted, in the order they were removed.
    """

    keep = referenced_secret_backups(previous)
    if keep is None:
        return []
    contexts = [release._cpu()] + [
        release._gpu(target) for target in release.config.clusters
    ]
    deleted: list[str] = []
    for kubectl in contexts:
        listing = release._get_json(
            kubectl
            + [
                "-n",
                release.config.namespace,
                "get",
                "secret",
                "-l",
                f"{RELEASE_SECRET_BACKUP_LABEL}=true",
            ]
        )
        stale = sorted(
            {
                name
                for item in listing.get("items") or []
                if isinstance(item, dict)
                and (name := str((item.get("metadata") or {}).get("name") or ""))
                and name not in keep
            }
        )
        if not stale:
            continue
        release.runner.run(
            kubectl
            + [
                "-n",
                release.config.namespace,
                "delete",
                "secret",
                *stale,
                "--ignore-not-found",
            ],
            sensitive=True,
        )
        deleted.extend(stale)
    return deleted
