from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

from gpu_fault.cluster_scope import is_eks_managed
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

REGISTRY_SECRET = "gpu-fault-regional-clusters"
REGISTRY_CURRENT_KEY = "clusters.json"
REGISTRY_BACKUP_KEY = "previous-clusters.json"
#: The per-cluster Runtime Profile pins of a registration (``_cluster_profile_pins``).
PROFILE_PIN_KEYS = ("runtime_profile_version", "agent_config_digest")


def _cluster_profile_pins(target: ClusterTarget) -> dict[str, str]:
    """The per-cluster Runtime Profile pins a registration carries.

    Only a cluster that overrides the site profile carries them; a cluster on
    the site profile keeps the registration shape it always had and the
    control plane falls back to its site-wide required pins.
    """

    if not getattr(target, "overrides_runtime_profile", False):
        return {}
    return {
        "runtime_profile_version": target.runtime_profile_version,
        "agent_config_digest": target.agent_config_digest,
    }


def _membership_rows(
    clusters: tuple[ClusterTarget, ...], *, with_pins: bool
) -> list[dict[str, Any]]:
    # Attribute reads tolerate the partial release doubles the diff tests use;
    # a real ClusterTarget always carries every field.
    return [
        {
            "cluster_id": item.cluster_id,
            "region": getattr(item, "region", ""),
            "hyperpod_cluster_name": getattr(item, "hyperpod_cluster_name", ""),
            "eks_cluster_arn": getattr(item, "eks_cluster_arn", ""),
            "allowed_namespaces": sorted(getattr(item, "allowed_namespaces", ())),
            "agent_endpoint_allowed_cidrs": sorted(
                getattr(item, "agent_endpoint_allowed_cidrs", ())
            ),
            **({"provider": "eks-managed"} if is_eks_managed(item) else {}),
            **(_cluster_profile_pins(item) if with_pins else {}),
        }
        for item in sorted(clusters, key=lambda target: target.cluster_id)
    ]


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def registry_config_digest(clusters: tuple[ClusterTarget, ...]) -> str:
    """The cluster *membership* digest the release diff calls ``clusters``.

    Membership and identity only -- ids, names, ARNs, namespaces, CIDRs and the
    provider. A cluster's Runtime Profile pins are release attributes tracked by
    the rollbackable ``cluster_runtime_profiles`` component; digesting them here
    (as the first per-cluster-profile release did) turned every Agent config
    change on a mixed site into a non-transactional ``clusters`` change.
    """

    return _rows_digest(_membership_rows(clusters, with_pins=False))


def legacy_registry_config_digest(clusters: tuple[ClusterTarget, ...]) -> str:
    """The digest recorded by releases that still folded the profile pins in;
    a recorded state carrying it describes the same membership."""

    return _rows_digest(_membership_rows(clusters, with_pins=True))


def registry_payloads(
    release: RegionalRelease,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    value = probe_resource(
        release.runner,
        release._cpu(),
        ResourceRef("secret", "Secret", REGISTRY_SECRET, release.config.namespace),
    ).require_readable()
    if value is None:
        return [], None
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


def registry(release: RegionalRelease) -> list[dict[str, Any]]:
    current, _backup = registry_payloads(release)
    return current


def registry_entry(target: ClusterTarget) -> dict[str, Any]:
    if not all(
        (
            target.token_file,
            target.hyperpod_cluster_name or is_eks_managed(target),
            target.eks_cluster_arn,
        )
    ):
        raise ReleaseError(
            "regional cluster registry requires token_file, "
            "hyperpod_cluster_name (HyperPod clusters), and eks_cluster_arn"
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
    if not target.token_file:
        raise ReleaseError(f"{target.cluster_id} requires a cluster token file")
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
        # Additive: a HyperPod entry keeps the key set it always had.
        **({"provider": "eks-managed"} if is_eks_managed(target) else {}),
        **_cluster_profile_pins(target),
    }


def desired_registry(release: RegionalRelease) -> list[dict[str, Any]]:
    return sorted(
        (registry_entry(target) for target in release.config.clusters),
        key=lambda item: str(item["cluster_id"]),
    )


def write_registry(
    release: RegionalRelease,
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
    release: RegionalRelease,
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


def stage_registry(release: RegionalRelease) -> bool:
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


def commit_registry_update(release: RegionalRelease) -> None:
    current, backup = registry_payloads(release)
    if backup is not None:
        write_registry(release, current)


def restore_registry_backup(release: RegionalRelease) -> bool:
    """Put the backup back as the registry; whether a restore awaits its publish.

    The backup stays beside the registry it restored until
    ``complete_registry_restore`` drops it, which the rollback does only once
    the restore's publish converged. Dropping it with the restore write left a
    rollback whose publish failed with nothing to find when resumed: the
    registry looked restored, it was never published, and the control plane
    kept serving the candidate's from the durable head. Now the resumed
    rollback finds the backup -- the registry already restored, nothing to
    write -- and publishes it. ``restore_profile_pins`` leaves the same mark.
    While it is there join and remove refuse (``update_registry``), as they
    do while a release has the registry staged.
    """

    current, backup = registry_payloads(release)
    if backup is None:
        return False
    if current != backup:
        write_registry(release, backup, backup=backup)
    return True


def complete_registry_restore(release: RegionalRelease) -> None:
    """Drop the backup a rollback's registry restore kept, once its publish
    converged (``restore_registry_backup``).

    Only a registry that is still the one the restore wrote is completed:
    nothing writes the registry while the backup is there, so a different one
    keeps the backup, and the rollback stops; resumed, it restores and
    publishes the backup again.
    """

    if release.runner.dry_run:
        return
    current, backup = registry_payloads(release)
    if backup is None:
        return
    if current != backup:
        raise ReleaseError(
            "regional cluster registry changed while the rollback published its "
            "restore; the backup is kept for a resumed rollback to restore again"
        )
    write_registry(release, current)


def registry_with_profile_pins(
    registrations: list[dict[str, Any]],
    profiles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """``registrations`` with each cluster's Runtime Profile pins from ``profiles``.

    ``profiles`` is a release's ``cluster_runtime_profiles`` record --
    ``{cluster_id: {version, agent_config_digest, ...}}`` for the clusters that
    ran their own profile. Such a cluster's entry carries that version and
    Agent config digest, as ``registry_entry`` registered it; every other
    cluster ran the site profile and carries none. Nothing else changes:
    membership is not a rollback's to change (``clusters`` is
    non-transactional) and a rotated token stays rotated.
    """

    restored = []
    for registration in registrations:
        entry = {
            key: value
            for key, value in registration.items()
            if key not in PROFILE_PIN_KEYS
        }
        profile = profiles.get(str(registration.get("cluster_id")))
        if isinstance(profile, Mapping):
            entry["runtime_profile_version"] = str(profile.get("version") or "")
            entry["agent_config_digest"] = str(profile.get("agent_config_digest") or "")
        restored.append(entry)
    return restored


def restore_profile_pins(
    release: RegionalRelease,
    profiles: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Register each cluster with the pins of ``profiles`` when no backup can.

    The commit drops the registry backup (``commit_registry_update``), so a
    ``deploy --rollback`` of a committed release has nothing for
    ``restore_registry_backup`` to put back: a cluster whose own Runtime
    Profile the release moved stayed registered with the candidate's pins, and
    the control plane judged its rolled-back Agents by them until the next
    deploy. ``profiles`` -- the restored release's -- gives the pins instead.
    Returns whether the registry was rewritten: never while a backup is there
    to restore, nor when the pins are already the registered ones (a release
    that moved no profile, or a stage that restored and published them
    already).

    The restored registry is written as its own backup, the mark
    ``restore_registry_backup`` leaves too, until the rollback has published it
    (``complete_registry_restore``). A rollback whose publish failed otherwise
    found the pins registered when resumed and never published them; with
    the mark the resumed rollback restores from that backup and publishes.
    """

    current, backup = registry_payloads(release)
    if backup is not None:
        return False
    restored = registry_with_profile_pins(current, profiles)
    if restored == current:
        return False
    write_registry(release, restored, backup=restored)
    return True


def update_registry(
    release: RegionalRelease,
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
