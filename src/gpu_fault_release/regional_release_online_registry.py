from __future__ import annotations

import hashlib
import json
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_registry import registry
from gpu_fault_release.regional_release_runtime_identity import exec_cpu_ingress_command


def _request(
    release: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # The mutating helper for every method: this drives the registry API, and a
    # GET here is only ever a step of a revision publish, so re-running one on a
    # different replica would read a generation the caller did not write.
    output = exec_cpu_ingress_command(
        release,
        arguments=(
            "python3",
            "-c",
            probe_source("registry_client"),
            method,
            path,
        ),
        failure="a registry update",
        input_text=json.dumps(payload or {}, separators=(",", ":")),
        sensitive=True,
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ReleaseError("regional registry API returned a non-object")
    return value


def current_registrations(
    release: Any,
    lifecycle_overrides: dict[str, str],
) -> list[dict[str, Any]]:
    """The bootstrap Secret's entries as the registry API takes them.

    Plaintext ``token`` values become ``token_sha256`` here; ``rotate-token``
    reads this to publish a revision that differs from the Secret in exactly
    one cluster's digests.
    """

    result = []
    for source in registry(release):
        item = dict(source)
        token = str(item.pop("token", ""))
        if token:
            item["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
        if "token_sha256" not in item:
            raise ReleaseError("regional registry entry has no token digest source")
        item["lifecycle_state"] = lifecycle_overrides.get(
            str(item["cluster_id"]),
            "ACTIVE",
        )
        result.append(item)
    return sorted(result, key=lambda item: str(item["cluster_id"]))


def _registration(
    release: Any,
    cluster_id: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in current_registrations(release, {})
        if str(item["cluster_id"]) == cluster_id
    ]
    if len(matches) != 1:
        raise ReleaseError(
            f"regional registry candidate has no unique cluster {cluster_id}"
        )
    return matches[0]


def publish_registry_revision(
    release: Any,
    *,
    path: str,
    payload: dict[str, Any],
    use_current_generation: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    output = exec_cpu_ingress_command(
        release,
        arguments=("python3", "-c", probe_source("registry_publish_converge")),
        failure="a registry revision publish",
        input_text=json.dumps(
            {
                "path": path,
                "payload": payload,
                "use_current_generation": use_current_generation,
                "timeout_seconds": timeout_seconds,
            },
            separators=(",", ":"),
        ),
        sensitive=True,
        timeout_seconds=timeout_seconds + 60,
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ReleaseError("regional registry client returned a non-object")
    return value


def publish_current_registry(
    release: Any,
    *,
    reason: str,
    lifecycle_overrides: dict[str, str] | None = None,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    return publish_registry_revision(
        release,
        path="/v1/regional/registry/revisions",
        payload={
            "registrations": current_registrations(
                release,
                lifecycle_overrides or {},
            ),
            "reason": reason,
        },
        use_current_generation=True,
        timeout_seconds=timeout_seconds,
    )


def publish_staged_registry(release: Any) -> dict[str, Any]:
    """Make the registry an upgrade just staged into the Secret durable.

    The running control plane reads the durable Aurora head and ignores the
    Secret once a head exists, so a release that only rewrote the Secret never
    reached the fleet (review H3). This is the same publish-and-converge path
    join and remove use; it runs inside the REGISTRY component so a failure
    rolls the transaction back and a resume repeats it.
    """

    if release.runner.dry_run:
        return {}
    return publish_current_registry(
        release,
        reason=f"release {release.release_id} registry staged",
    )


def publish_restored_registry(release: Any) -> dict[str, Any]:
    """Republish the Secret's restored backup after a rollback restored it."""

    if release.runner.dry_run:
        return {}
    return publish_current_registry(
        release,
        reason=f"release {release.release_id} registry restored by rollback",
    )


def transition_join_registry(
    release: Any,
    cluster_id: str,
    lifecycle_state: str,
    *,
    reason: str,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    return publish_registry_revision(
        release,
        path=f"/v1/regional/registry/clusters/{cluster_id}/transition",
        payload={
            "registration": _registration(release, cluster_id),
            "lifecycle_state": lifecycle_state,
            "reason": reason,
        },
        use_current_generation=False,
        timeout_seconds=timeout_seconds,
    )


def prepare_join_registry(release: Any, cluster_id: str) -> None:
    transition_join_registry(
        release,
        cluster_id,
        "PENDING",
        reason=f"join {cluster_id} pending",
    )


def activate_join_registry(release: Any, cluster_id: str) -> None:
    transition_join_registry(
        release,
        cluster_id,
        "ACTIVE",
        reason=f"join {cluster_id} active",
    )


def fail_join_registry(release: Any, cluster_id: str) -> None:
    transition_join_registry(
        release,
        cluster_id,
        "FAILED",
        reason=f"join {cluster_id} failed",
    )


def rollback_join_registry(release: Any, cluster_id: str) -> None:
    transition_join_registry(
        release,
        cluster_id,
        "ROLLED_BACK",
        reason=f"join {cluster_id} rolled back",
    )


def drain_registry_cluster(release: Any, cluster_id: str) -> None:
    release._target(cluster_id)
    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    publish_current_registry(
        release,
        reason=f"remove {cluster_id} draining",
        lifecycle_overrides={cluster_id: "DRAINING"},
    )


def revoke_registry_cluster(release: Any, cluster_id: str) -> None:
    publish_current_registry(
        release,
        reason=f"remove {cluster_id} revoked",
        lifecycle_overrides={cluster_id: "REVOKED"},
    )


def purge_registry_cluster(release: Any, cluster_id: str) -> None:
    publish_current_registry(
        release,
        reason=f"remove {cluster_id} purged",
    )
