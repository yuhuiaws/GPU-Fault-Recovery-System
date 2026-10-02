from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    RegionalRegistryPublishRequest,
    RegionalRegistryStatus,
    regional_registry_content_sha256,
)
from gpu_fault.regional_registry import regional_registry_config_sha256
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_registry import registry
from gpu_fault_release.regional_release_runtime_identity import exec_cpu_ingress_command


def _request(
    release: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    response_list: bool = False,
) -> Any:
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
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        raise ReleaseError("regional registry API returned invalid JSON") from None
    if response_list:
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            raise ReleaseError("regional registry API returned a non-object list")
    elif not isinstance(value, dict):
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


def describe_unconverged_publish(release: Any, *, timeout_seconds: float) -> None:
    """Name the control-plane members that held a publish past its window.

    Every currently active CPU process must ACK the exact revision, including
    arrivals after the immutable publish-time membership snapshot. A required
    member with a stale row stops blocking a live fleet once its persisted
    heartbeat no longer authorizes traffic (live 2026-09-15: a control-worker
    terminating under the remove-cluster roll was captured at publish and held
    the join's generation for its whole window); a missing row remains
    unresolved. A known fleet needs a live ACK, not universal heartbeat expiry.
    Only an empty required set with no member rows converges without any ACK.
    GPU executors and agents are not CPU registry members.

    The probe's exit line names only the generation. Read status once more
    for the unresolved members and their reported readiness/errors. This is
    best effort and never masks the original publish failure.
    """

    try:
        status = _request(release, "GET", "/v1/regional/registry/status")
    except ReleaseError as exc:
        print(
            "regional registry status is unavailable after the publish ran out "
            f"its {timeout_seconds:.0f}s convergence window: {diagnostic_text(str(exc))}",
            file=sys.stderr,
            flush=True,
        )
        return
    members = {
        str(item.get("member_id")): item
        for item in status.get("members") or []
        if isinstance(item, dict)
    }
    lines = [
        f"regional registry generation {status.get('generation')} did not "
        f"converge within {timeout_seconds:.0f}s: "
        f"required={status.get('required_member_ids')} "
        f"acked={status.get('acked_member_ids')} "
        f"active={status.get('active_member_ids')}"
    ]
    for member_id in status.get("missing_member_ids") or []:
        member = members.get(str(member_id)) or {}
        lines.append(
            f"  missing {member_id}: role={member.get('service_role')} "
            f"ready={member.get('ready')} generation={member.get('generation')} "
            f"last_seen_at={member.get('last_seen_at')} error={member.get('error')}"
        )
    print(diagnostic_text("\n".join(lines)), file=sys.stderr, flush=True)


def publish_registry_revision(
    release: Any,
    *,
    path: str,
    payload: dict[str, Any],
    use_current_generation: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
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
    except ReleaseError:
        # The probe is sensitive, so its last words are not forwarded; a
        # failure that took the whole convergence window is the probe giving
        # up on a member, and that member is worth a second, read-only exec.
        # A refusal (identity, generation) fails in seconds and gets none.
        if time.monotonic() - started >= timeout_seconds:
            describe_unconverged_publish(release, timeout_seconds=timeout_seconds)
        raise
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
    if lifecycle_overrides:
        request = _lifecycle_registry_request(release, lifecycle_overrides, reason)
        result = publish_registry_revision(
            release,
            path="/v1/regional/registry/revisions",
            payload=request.model_dump(mode="json", exclude={"required_member_ids"}),
            use_current_generation=False,
            timeout_seconds=timeout_seconds,
        )
        try:
            status = RegionalRegistryStatus.model_validate(result)
        except ValueError:
            raise ReleaseError(
                "regional registry publish evidence is invalid"
            ) from None
        if (
            status.generation != request.expected_generation + 1
            or status.content_sha256
            != regional_registry_content_sha256(request.registrations)
            or status.cluster_states
            != {item.cluster_id: item.lifecycle_state for item in request.registrations}
            or not status.converged
            or status.missing_member_ids
        ):
            raise ReleaseError(
                "regional registry did not converge on the lifecycle update"
            )
        return result
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


def durable_registrations(
    release: Any,
    *,
    retiring_digests: Mapping[str, str] | None = None,
) -> tuple[RegionalRegistryStatus, list[RegionalClusterRegistration]]:
    """The durable registry head, its redacted digests restored from the Secret.

    The API redacts ``token_sha256``/``retiring_token_sha256``; they are taken
    from the CPU Secret and the revision's content digest proves the bytes
    agree. Raises when the Secret cannot reconstruct the head -- membership,
    lifecycle, digest presence or length, content or config digest differ --
    which is also what a rotation's overlap window (a retiring digest the
    Secret knows nothing about) looks like. ``retiring_digests`` lets a caller
    that does know the retiring digest of a cluster (``rotate-token``'s
    journal) supply it for exactly that case; the content digest still has to
    prove it, so a wrong candidate fails the same way as any other drift.
    """

    try:
        status = RegionalRegistryStatus.model_validate(
            _request(release, "GET", "/v1/regional/registry/status")
        )
        configured = [
            RegionalClusterRegistration.model_validate(item)
            for item in current_registrations(release, {})
        ]
        by_id = {item.cluster_id: item for item in configured}
        if len(by_id) != len(configured) or set(by_id) != set(status.cluster_states):
            raise ReleaseError("regional registry membership differs from its Secret")
        durable: list[RegionalClusterRegistration] = []
        for source in _request(
            release, "GET", "/v1/regional/clusters", response_list=True
        ):
            item = dict(source)
            candidate = by_id[item["cluster_id"]]
            # The API redacts digests. Restore them only from the CPU Secret
            # (or the caller's retiring candidate); the full revision digest
            # below proves the actual bytes agree.
            for field in ("token_sha256", "retiring_token_sha256"):
                digest = getattr(candidate, field)
                present = item.pop(f"{field}_present")
                length = item.pop(f"{field}_length")
                supplied = (retiring_digests or {}).get(str(item["cluster_id"]))
                if field == "retiring_token_sha256" and digest is None and supplied:
                    if present is True:
                        digest = supplied
                        candidate = candidate.model_copy(update={field: supplied})
                        by_id[candidate.cluster_id] = candidate
                if (
                    type(present) is not bool
                    or type(length) is not int
                    or present != bool(digest)
                    or length != len(digest or "")
                ):
                    raise ReleaseError("regional registry credential identity differs")
                item[field] = digest
            durable.append(RegionalClusterRegistration.model_validate(item))
        if (
            len(durable) != len(by_id)
            or {item.cluster_id: item.lifecycle_state for item in durable}
            != status.cluster_states
            or regional_registry_content_sha256(durable) != status.content_sha256
            or regional_registry_config_sha256(durable)
            != regional_registry_config_sha256(list(by_id.values()))
        ):
            raise ReleaseError("regional registry snapshot identity differs")
        return status, durable
    except (KeyError, TypeError, ValueError):
        raise ReleaseError("regional registry lifecycle evidence is invalid") from None


def _lifecycle_registry_request(
    release: Any, overrides: dict[str, str], reason: str
) -> RegionalRegistryPublishRequest:
    """Bind a lifecycle-only update to the complete durable registry snapshot."""

    status, durable = durable_registrations(release)
    if not set(overrides).issubset({item.cluster_id for item in durable}):
        raise ReleaseError("regional registry lifecycle target is missing")
    try:
        observed = datetime.now(timezone.utc)
        for registration in durable:
            if registration.cluster_id not in overrides:
                continue
            desired = RegionalClusterLifecycle(overrides[registration.cluster_id])
            if (
                desired is RegionalClusterLifecycle.DRAINING
                and registration.lifecycle_state
                not in {
                    RegionalClusterLifecycle.ACTIVE,
                    RegionalClusterLifecycle.DRAINING,
                }
            ):
                raise ReleaseError(
                    "regional registry cluster is not active or draining"
                )
            if registration.lifecycle_state != desired:
                registration.lifecycle_state = desired
                registration.updated_at = observed
        return RegionalRegistryPublishRequest(
            expected_generation=status.generation,
            registrations=durable,
            reason=reason,
        )
    except (KeyError, TypeError, ValueError):
        raise ReleaseError("regional registry lifecycle evidence is invalid") from None


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
        release, cluster_id, "FAILED", reason=f"join {cluster_id} failed"
    )


def rollback_join_registry(release: Any, cluster_id: str) -> None:
    transition_join_registry(
        release, cluster_id, "ROLLED_BACK", reason=f"join {cluster_id} rolled back"
    )


def purge_failed_join(release: Any, target: Any) -> None:
    """Purge only after the identity-bound rollback transition succeeds."""

    rollback_join_registry(release, target.cluster_id)
    release._update_registry(target, remove=True)
    purge_registry_cluster(release, target.cluster_id)


def drain_registry_clusters(release: Any, cluster_ids: Sequence[str]) -> None:
    """Drain every selected cluster in one CAS-protected, fleet-ACKed revision.

    ``current_registrations`` publishes every cluster it does not override as
    ACTIVE, so draining several clusters one revision at a time would flip each
    earlier one back to ACTIVE and only the last would stay DRAINING; an
    uninstall drains all of a site's GPU clusters through this one publish.
    Fail-closed like remove-cluster: refused while any remote command is open.
    """

    if isinstance(cluster_ids, (str, bytes)) or any(
        not isinstance(item, str) or not item.strip() for item in cluster_ids
    ):
        raise ReleaseError("cluster IDs must be non-empty strings")
    ordered = list(dict.fromkeys(cluster_ids))
    if not ordered:
        raise ReleaseError("--cluster-id is required")
    for cluster_id in ordered:
        release._target(cluster_id)
    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    publish_current_registry(
        release,
        reason=f"remove {', '.join(ordered)} draining",
        lifecycle_overrides={cluster_id: "DRAINING" for cluster_id in ordered},
    )


def drain_registry_cluster(release: Any, cluster_id: str) -> None:
    drain_registry_clusters(release, [cluster_id])


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
