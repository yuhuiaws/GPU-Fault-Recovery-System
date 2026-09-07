from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_agent_convergence import (
    agent_heartbeats_converged as agent_heartbeats_converged,
)
from gpu_fault_release.regional_release_agent_convergence import (
    wait_agents as wait_agents,
)
from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ClusterTarget,
    ReleaseError,
)
from gpu_fault_release.regional_release_gpu_rollout import gpu_node_items
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_rendering import build_reconciler_environment
from gpu_fault_release.regional_release_rollout_wait import wait_deployment_rollout
from gpu_fault_release.regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    exec_cpu_ingress_command,
    exec_cpu_ingress_probe,
)
from gpu_fault_release.regional_release_timing import record_rollback_wave_event

from gpu_fault.failure_domains import (
    FAILURE_DOMAIN_LABELS as SHARED_FAILURE_DOMAIN_LABELS,
)

ROOT = Path(__file__).resolve().parents[2]
# One definition of "failure domain" for the whole system: the remediation
# budget's per-domain tier reads the same priority, so a rollout wave and a
# repair cap cannot disagree about which nodes share a fate. Instance group
# leads because every node of a HyperPod cluster shares one zone.
FAILURE_DOMAIN_LABELS = SHARED_FAILURE_DOMAIN_LABELS
CANDIDATE_CPU_HEARTBEAT_TIMEOUT_SECONDS = 75.0
ROLLOUT_AGENT_GATE_TIMEOUT_SECONDS = 60.0
ROLLOUT_AGENT_POLL_SECONDS = 5.0
ROLLOUT_AGENT_LEASE_MARGIN_SECONDS = 40
ROLLOUT_AGENT_POST_LEASE_MARGIN_SECONDS = 30
MAX_UPGRADE_UNAVAILABLE = 32
INSTALLER_WAVE_CONFIG_MAP_ENV = "GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP"
INSTALLER_BUNDLE_ENV = "GPU_FAULT_INSTALLER_BUNDLE_SHA256"
INSTALLER_TEMPLATE_ENV = "GPU_FAULT_INSTALLER_TEMPLATE_SHA256"


@dataclass(frozen=True)
class NodeRolloutPolicy:
    max_unavailable: int
    first_wave_max_unavailable: int
    max_unavailable_per_failure_domain: int


@dataclass(frozen=True)
class FleetWaveContext:
    phase: str
    deployment_id: str
    node_names: tuple[str, ...]
    paused_identity: tuple[str, str]
    wheel_cm: str
    bundle_cm: str
    artifact_sha: str
    config_digest: str
    expected_profile: str
    executor_wheel_filename: str | None
    expected_compatibility: str
    desired_bundle: str
    desired_template: str
    template_config_map: str | None
    max_unavailable: int
    runtime_image: str | None
    node_installer_image: str | None
    allow_legacy_identity: bool
    agent_identity: dict[str, Any] | None


def backup_secret(
    release: Any,
    kubectl: list[str],
    *,
    source: str,
    backup: str,
    required: bool,
) -> str | None:
    exists = release.runner.probe(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "secret",
            source,
        ],
    )
    if not exists:
        if required:
            raise ReleaseError(f"required Secret is missing: {source}")
        return None
    value = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "secret",
            source,
        ]
    )
    document = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": backup,
            "namespace": release.config.namespace,
            "labels": {
                "gpu-fault.io/release-secret-backup": "true",
            },
            "annotations": {
                "gpu-fault.io/source-secret": source,
                "gpu-fault.io/release-id": release.release_id,
            },
        },
        "type": value.get("type") or "Opaque",
        "data": dict(value.get("data") or {}),
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
) -> None:
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
    document = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": source,
            "namespace": release.config.namespace,
        },
        "type": value.get("type") or "Opaque",
        "data": dict(value.get("data") or {}),
    }
    release.runner.run(
        kubectl + ["apply", "-f", "-"],
        input_text=json.dumps(document),
        sensitive=True,
    )


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


def target_node_names(
    release: Any,
    target: ClusterTarget,
) -> tuple[str, ...]:
    names = tuple(
        sorted(
            str(item.get("metadata", {}).get("name") or "")
            for item in gpu_node_items(release, target)
            if (
                item.get("metadata", {})
                .get("labels", {})
                .get("sagemaker.amazonaws.com/cluster-name")
                == target.hyperpod_cluster_name
                and item.get("metadata", {}).get("name")
            )
        )
    )
    if not names:
        raise ReleaseError(
            f"{target.cluster_id} has no HyperPod nodes for fleet rollout"
        )
    return names


def target_node_failure_domains(
    release: Any,
    target: ClusterTarget,
    node_names: tuple[str, ...],
) -> dict[str, str]:
    expected = set(node_names)
    result = {}
    for item in gpu_node_items(release, target):
        metadata = item.get("metadata", {})
        node_id = str(metadata.get("name") or "")
        if node_id not in expected:
            continue
        labels = metadata.get("labels") or {}
        result[node_id] = next(
            (
                str(labels[label])
                for label in FAILURE_DOMAIN_LABELS
                if str(labels.get(label) or "").strip()
            ),
            "UNKNOWN",
        )
    if set(result) != expected:
        raise ReleaseError(
            f"{target.cluster_id} failure-domain inventory is incomplete"
        )
    return result


def fleet_command(
    release: Any,
    operation: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    # Through the mutating helper for every operation, including the read-only
    # ``get``: the wave loop interleaves them with ``next-wave``, which takes a
    # lease, so one shared memoised Pod name is what keeps the whole sequence
    # talking to the same replica.
    raw = exec_cpu_ingress_command(
        release,
        arguments=(
            CONTROL_PLANE_PYTHON,
            "-c",
            probe_source("fleet_deployment_command"),
        ),
        failure="fleet rollout",
        input_text=json.dumps({"operation": operation, **payload}),
    )
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ReleaseError("fleet rollout command returned a non-object")
    return result


def fleet_deployment_id(
    release: Any,
    target: ClusterTarget,
    *,
    phase: str,
    artifact_sha: str,
    bundle_sha: str | None,
    template_sha: str | None,
    config_digest: str,
    runtime_profile_version: str,
) -> str:
    inputs: dict[str, Any] = {
        "phase": phase,
        "release_id": release.release_id,
        "cluster_id": target.cluster_id,
        "artifact_sha256": artifact_sha,
        "bundle_sha256": bundle_sha,
        "template_sha256": template_sha,
        "config_digest": config_digest,
        "runtime_profile_version": runtime_profile_version,
    }
    # One rollout record per upgrade transaction. The same candidate applied
    # again after its rollback must not resume the FAILED record that rollback
    # left behind; a resume of the same transaction keeps the same nonce and so
    # the same id. A state without the nonce (written before it existed) keeps
    # the historical id, so an in-flight legacy transaction still finds its
    # own deployment.
    transaction = str(
        (getattr(release, "state", None) or {}).get("fleet_rollout_transaction") or ""
    )
    if transaction:
        inputs["transaction"] = transaction
    identity = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"release-{phase}-{release.release_id}-{identity}"


def nodes_have_legacy_installer_identity(
    release: Any,
    target: ClusterTarget,
    *,
    node_names: tuple[str, ...],
    artifact_sha: str,
    config_digest: str,
) -> bool:
    expected = set(node_names)
    selected = [
        item
        for item in gpu_node_items(release, target, fresh=True)
        if str(item.get("metadata", {}).get("name") or "") in expected
    ]
    if len(selected) != len(expected):
        return False
    for node in selected:
        metadata = node.get("metadata", {})
        annotations = metadata.get("annotations") or {}
        if (
            annotations.get("gpu-fault.io/installer-artifact-sha256") != artifact_sha
            or annotations.get("gpu-fault.io/installer-config-digest") != config_digest
            or annotations.get("gpu-fault.io/installer-node-uid") != metadata.get("uid")
            or annotations.get("gpu-fault.io/installer-state") != "Succeeded"
        ):
            return False
    return True


def finish_legacy_node_runtime_rollback(
    release: Any,
    target: ClusterTarget,
    *,
    enabled: bool,
    node_names: tuple[str, ...],
    paused_identity: tuple[str, str],
    wheel_cm: str,
    bundle_cm: str,
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str,
    executor_wheel_filename: str | None,
    node_compatibility_digest: str,
    template_config_map: str | None,
    runtime_image: str | None,
    steady_runtime_image: str | None,
    steady_template_config_map: str | None,
    node_installer_image: str | None,
    max_unavailable: int,
) -> tuple[str, str] | None:
    if not enabled or not nodes_have_legacy_installer_identity(
        release,
        target,
        node_names=node_names,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
    ):
        return None
    desired_bundle, desired_template = paused_identity
    final_identity = release._deploy_reconciler(
        target,
        wheel_cm=wheel_cm,
        bundle_cm=bundle_cm,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        node_compatibility_digest=node_compatibility_digest,
        bundle_sha256=desired_bundle,
        template_sha256=desired_template,
        template_config_map=steady_template_config_map or template_config_map,
        allowed_node_names=None,
        max_unavailable=max_unavailable,
        runtime_image=steady_runtime_image or runtime_image,
        node_installer_image=node_installer_image,
    )
    if final_identity != paused_identity:
        raise ReleaseError(f"{target.cluster_id} legacy installer identity changed")
    return final_identity


def node_rollout_policy(
    release: Any,
    failure_domains: dict[str, str] | int,
    *,
    phase: str,
) -> NodeRolloutPolicy:
    if phase == "rollback":
        raise ReleaseError("rollback rollout policy requires rollback policy")
    if isinstance(failure_domains, int):
        failure_domains = {
            f"node-{index}": f"domain-{index}" for index in range(failure_domains)
        }
    node_count = len(failure_domains)
    domains = set(failure_domains.values())
    if not node_count:
        raise ReleaseError("upgrade rollout policy has no nodes")
    if "UNKNOWN" in domains:
        effective = 1
        per_domain = 1
    else:
        size_cap = (
            min(node_count, 4)
            if node_count < 64
            else 8
            if node_count < 256
            else 16
            if node_count < 512
            else MAX_UPGRADE_UNAVAILABLE
        )
        configured = int(release.config.upgrade_max_unavailable)
        # 0 is "auto": take the cap this function already derived from the node
        # count. A site that configured 1 paid a full safety, Reconciler and
        # convergence round trip per node -- four waves for four nodes -- while
        # the cap said all four could move together.
        effective = min(
            configured if configured > 0 else size_cap,
            size_cap,
            MAX_UPGRADE_UNAVAILABLE,
        )
        per_domain = (
            effective if len(domains) == 1 else ceil(effective / max(1, len(domains)))
        )
    return NodeRolloutPolicy(
        max_unavailable=max(1, effective),
        first_wave_max_unavailable=1,
        max_unavailable_per_failure_domain=max(1, per_domain),
    )


def rollback_node_rollout_policy(
    release: Any,
    failure_domains: dict[str, str],
) -> NodeRolloutPolicy:
    node_count = len(failure_domains)
    domains = set(failure_domains.values())
    if not node_count:
        raise ReleaseError("rollback rollout policy has no nodes")
    if "UNKNOWN" in domains:
        effective = 1
        per_domain = 1
    else:
        size_cap = 1 if node_count < 6 else 2 if node_count < 32 else 4
        effective = min(release.config.rollback_max_unavailable, size_cap)
        per_domain = effective if len(domains) == 1 else 1
    return NodeRolloutPolicy(
        max_unavailable=max(1, effective),
        first_wave_max_unavailable=1,
        max_unavailable_per_failure_domain=max(1, per_domain),
    )


def next_deployment_wave(deployment: dict[str, Any]) -> tuple[str, ...]:
    statuses = {
        str(item.get("node_id") or ""): str(item.get("status") or "")
        for item in deployment.get("nodes") or []
    }
    for raw_wave in deployment.get("waves") or []:
        wave = tuple(str(node_id) for node_id in raw_wave)
        if any(statuses.get(node_id) != "READY" for node_id in wave):
            return tuple(
                node_id
                for node_id in wave
                if statuses.get(node_id) in {"PENDING", "INSTALLING"}
            )
    return ()


def deployment_wave_position(deployment: dict[str, Any], wave: tuple[str, ...]) -> str:
    """Where this wave sits in the plan, as ``index/total``.

    A wave is named by its nodes, and a rollout of one-node waves therefore
    narrates a different opaque instance id every time; the position is what says
    whether the release is a quarter or three quarters through the fleet. The
    match is by containment because the wave actually leased is the still-pending
    subset of a planned wave.
    """

    waves = [
        tuple(str(node_id) for node_id in raw) for raw in deployment.get("waves") or []
    ]
    for index, planned in enumerate(waves, start=1):
        if wave and set(wave) <= set(planned):
            return f"{index}/{len(waves)}"
    return f"?/{len(waves)}"


def deployment_node_progress(deployment: dict[str, Any]) -> str:
    """How much of the fleet is already on the new identity, as ``ready/total``."""

    nodes = deployment.get("nodes") or []
    ready = sum(1 for item in nodes if str(item.get("status") or "") == "READY")
    return f"{ready}/{len(nodes)}"


def candidate_agent_pin_identity(release: Any) -> dict[str, str]:
    """The Agent identity this release's pin window narrows to on finalize.

    These are the four values ``build_cpu_apply_environment`` writes into
    ``gpu-fault-release-metadata`` as ``required-agent-*`` when ``finalize`` is
    true, so an Agent reporting all four is one the finalized control plane
    still accepts.
    """
    config = release.config
    return {
        "artifact_sha256": str(release.node_wheel_sha),
        "compatibility_digest": str(
            config.component_digests.get("node_runtime") or release.node_wheel_sha
        ),
        "agent_protocol_version": str(config.agent_protocol_version),
        "config_digest": str(config.agent_config_digest),
    }


def wait_candidate_cpu_agent_heartbeats(
    release: Any,
    agent_identities: dict[str, Any],
    *,
    required_identity: dict[str, str] | None = None,
    timeout_seconds: float = CANDIDATE_CPU_HEARTBEAT_TIMEOUT_SECONDS,
    poll_seconds: float = ROLLOUT_AGENT_POLL_SECONDS,
    minimum_lease_remaining_seconds: int = ROLLOUT_AGENT_LEASE_MARGIN_SECONDS,
) -> None:
    if release.runner.dry_run:
        return
    expected_by_cluster = {
        str(cluster_id): sorted(
            {
                str(node_id)
                for node_id in (identity or {}).get("node_ids", [])
                if str(node_id)
            }
        )
        for cluster_id, identity in agent_identities.items()
        if isinstance(identity, dict)
    }
    expected_by_cluster = {
        cluster_id: node_ids
        for cluster_id, node_ids in expected_by_cluster.items()
        if node_ids
    }
    if not expected_by_cluster:
        return
    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("agent_heartbeat_barrier"),
        failure="release safety checks",
        input_text=json.dumps(
            {
                "expected_by_cluster": expected_by_cluster,
                "timeout_seconds": timeout_seconds,
                "poll_seconds": poll_seconds,
                "minimum_lease_remaining_seconds": (minimum_lease_remaining_seconds),
                "required_identity": required_identity,
            }
        ),
        timeout_seconds=max(30, int(timeout_seconds) + 30),
        # The in-Pod barrier already spends the full window; never pay it twice.
        retries=0,
    )
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseError(
            "candidate CPU Agent heartbeat barrier returned invalid evidence"
        ) from exc
    if not isinstance(result, dict) or result.get("status") != "PASSED":
        diagnostics = {
            "expected_count": result.get("expected_count"),
            "refreshed_count": result.get("refreshed_count"),
            "blocker_count": result.get("blocker_count"),
            "blockers": result.get("blockers"),
        }
        prefix = (
            "fleet has not reached the candidate Agent pin, so the compatibility "
            "window stays open: "
            if required_identity is not None
            else "candidate CPU did not observe refreshed Agent heartbeats: "
        )
        raise ReleaseError(prefix + json.dumps(diagnostics, sort_keys=True))


def capture_active_agent_node_sets(release: Any) -> dict[str, dict[str, Any]]:
    if release.runner.dry_run:
        return {}
    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("active_agent_node_sets"),
        failure="release safety checks",
        sensitive=True,
        interactive=False,
    )
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseError("active Agent node inventory is invalid") from exc
    if not isinstance(value, dict):
        raise ReleaseError("active Agent node inventory is not an object")
    expected = {target.cluster_id for target in release.config.clusters}
    result = {
        str(cluster_id): {
            "node_ids": sorted({str(node_id) for node_id in node_ids if str(node_id)})
        }
        for cluster_id, node_ids in value.items()
        if isinstance(node_ids, list)
    }
    missing = sorted(
        cluster_id
        for cluster_id in expected
        if not (result.get(cluster_id) or {}).get("node_ids")
    )
    if missing:
        raise ReleaseError(
            "active Agent node inventory is missing clusters: " + ", ".join(missing)
        )
    return {cluster_id: result[cluster_id] for cluster_id in sorted(expected)}


def rollout_wave_safety_snapshot(
    release: Any,
    target: ClusterTarget,
    *,
    wave: tuple[str, ...],
    node_names: tuple[str, ...],
    minimum_lease_remaining_seconds: int,
) -> dict[str, Any]:
    executor = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.GPU_EXECUTOR_DEPLOYMENT,
        )
    )
    desired = int((executor.get("spec") or {}).get("replicas") or 0)
    ready = int((executor.get("status") or {}).get("readyReplicas") or 0)
    if desired < 1 or ready != desired:
        raise ClusterLocalReleaseError(
            f"{target.cluster_id} executor coverage is not fully Ready "
            f"(ready={ready}, desired={desired})"
        )
    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("rollout_wave_safety"),
        failure="release safety checks",
        input_text=json.dumps(
            {
                "cluster_id": target.cluster_id,
                "wave": list(wave),
                "nodes": list(node_names),
                "minimum_lease_remaining_seconds": (minimum_lease_remaining_seconds),
            }
        ),
    )
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseError("fleet wave safety check returned invalid evidence") from exc
    if not isinstance(result, dict):
        raise ReleaseError("fleet wave safety check returned non-object evidence")
    return result


def ensure_rollout_wave_safe(
    release: Any,
    target: ClusterTarget,
    *,
    wave: tuple[str, ...],
    node_names: tuple[str, ...],
    timeout_seconds: float = ROLLOUT_AGENT_GATE_TIMEOUT_SECONDS,
    poll_seconds: float = ROLLOUT_AGENT_POLL_SECONDS,
    minimum_lease_remaining_seconds: int = ROLLOUT_AGENT_LEASE_MARGIN_SECONDS,
) -> dict[str, Any]:
    """Block until the wave is safe to take down, and return the evidence.

    The returned snapshot is the observation the decision was made on, so a
    caller that needs the same question answered against a wider margin can
    re-decide it from this evidence -- see ``wave_lease_margin_holds`` -- rather
    than paying for a second identical exec into the cluster.
    """

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        snapshot = rollout_wave_safety_snapshot(
            release,
            target,
            wave=wave,
            node_names=node_names,
            minimum_lease_remaining_seconds=(minimum_lease_remaining_seconds),
        )
        open_remote = {
            str(status): int(count)
            for status, count in (snapshot.get("open_remote") or {}).items()
            if int(count)
        }
        if open_remote:
            raise ReleaseError(
                "fleet wave safety blocked by remote commands: "
                + json.dumps(open_remote, sort_keys=True)
            )
        destructive_count = int(snapshot.get("destructive_workflow_count") or 0)
        if destructive_count:
            raise ReleaseError(
                "fleet wave safety blocked by active destructive workflows: "
                + json.dumps(
                    {
                        "count": destructive_count,
                        "request_ids": snapshot.get("destructive_workflows") or [],
                    },
                    sort_keys=True,
                )
            )
        blockers = list(snapshot.get("agent_blockers") or [])
        blocker_count = int(snapshot.get("agent_blocker_count") or len(blockers))
        if not blocker_count:
            return snapshot
        if time.monotonic() >= deadline:
            raise ClusterLocalReleaseError(
                f"{target.cluster_id} fleet wave Agent safety did not converge: "
                + json.dumps(
                    {
                        "blocker_count": blocker_count,
                        "blockers": blockers[:100],
                        "minimum_lease_remaining_seconds": (
                            minimum_lease_remaining_seconds
                        ),
                    },
                    sort_keys=True,
                )
            )
        time.sleep(max(0.0, poll_seconds))


def wave_lease_margin_holds(
    snapshot: dict[str, Any] | None,
    *,
    elapsed_seconds: float,
    required_seconds: int = ROLLOUT_AGENT_POST_LEASE_MARGIN_SECONDS,
) -> bool:
    """Does the safety evidence still prove the post-lease Agent lease margin?

    Taking the wave lease costs one control-plane call, and the margin the
    Agents must retain *after* it is wider than the one the safety gate itself
    requires. That used to be a second exec of the same probe about five
    seconds later -- the same leases, re-read, per wave. A lease shrinks by
    exactly the time that passed, so the first probe's reported minimum minus
    the elapsed time answers it. Anything the probe did not report (an older
    probe, or a wave that is the whole cluster and leaves no node outside it)
    is not evidence, so it fails closed and the caller re-probes.
    """

    if not isinstance(snapshot, dict):
        return False
    minimum = snapshot.get("minimum_lease_remaining_seconds")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        return False
    return float(minimum) - max(0.0, elapsed_seconds) >= required_seconds


def reconciler_container_env(
    release: Any,
    target: ClusterTarget,
    *,
    deployment: dict[str, Any] | None = None,
) -> dict[str, str]:
    """The Reconciler container's environment, read once per caller.

    ``deployment`` lets a caller that already holds the Deployment -- because it
    just waited for it to roll out -- hand it over instead of paying a second
    identical read.
    """

    if deployment is None:
        deployment = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                inventory.GPU_RECONCILER_DEPLOYMENT,
            )
        )
    containers = (
        deployment.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    container = next(
        (item for item in containers if item.get("name") == "reconciler"),
        None,
    )
    if container is None:
        raise ReleaseError("Reconciler Deployment has no reconciler container")
    return {
        str(item.get("name") or ""): str(item.get("value") or "")
        for item in container.get("env") or []
    }


def reconciler_installer_identity(environment: dict[str, str]) -> tuple[str, str]:
    bundle = environment.get(INSTALLER_BUNDLE_ENV, "")
    template = environment.get(INSTALLER_TEMPLATE_ENV, "")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (bundle, template)):
        raise ReleaseError("Reconciler installer identity is invalid")
    return bundle, template


def installer_wave_generation(
    artifact_sha: str,
    max_unavailable: int,
    wave: tuple[str, ...],
) -> str:
    digest = hashlib.sha256(",".join(sorted(wave)).encode()).hexdigest()[:12]
    return f"{artifact_sha[:12]}-{max_unavailable}-{digest}"


def hand_wave_to_reconciler(
    release: Any,
    target: ClusterTarget,
    context: FleetWaveContext,
    wave: tuple[str, ...],
) -> tuple[str, str] | None:
    """Hand one wave to the already running Reconciler through its ConfigMap.

    The Reconciler Deployment uses the Recreate strategy because its installer
    identity lives in its environment, so re-deploying it once per wave took the
    only installer controller of the cluster down and back up for every wave of
    the fleet rollout. The allowed-node set is not part of that identity: it is
    read from `GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP` on every reconcile pass, so
    a wave is a ConfigMap patch against an unchanged Deployment.

    Returns the live installer identity, or ``None`` when the deployed
    Reconciler predates the wave ConfigMap and the caller must fall back to the
    per-wave redeploy. Every barrier of the redeploy path is preserved: the
    Deployment must be fully rolled out, its installer identity must still equal
    the paused identity, in-flight installer Jobs are cancelled before the
    allowed set changes, and the patched wave is read back before any node is
    allowed to move.
    """

    # A wave may only be handed to a healthy, fully rolled out Reconciler: the
    # Pod that reads the ConfigMap has to be the one this release paused.
    rollout = wait_deployment_rollout(
        release,
        target,
        inventory.GPU_RECONCILER_DEPLOYMENT,
        timeout_seconds=600,
    )
    # The identity is read out of the Deployment the wait just proved rolled out,
    # rather than from a second `get deployment` of the same object: one read per
    # wave less, and the identity provably belongs to the generation that passed
    # the barrier instead of to whatever the next read happens to return.
    environment = reconciler_container_env(
        release, target, deployment=rollout.get("object")
    )
    config_map = environment.get(INSTALLER_WAVE_CONFIG_MAP_ENV, "")
    if not config_map:
        return None
    identity = reconciler_installer_identity(environment)
    if identity != context.paused_identity:
        raise ReleaseError(
            f"{target.cluster_id} installer identity changed between waves"
        )
    release._settle_installer_jobs(target)
    desired = {
        "allowed-nodes": ",".join(sorted(wave)),
        "max-unavailable": str(context.max_unavailable),
        "generation": installer_wave_generation(
            context.artifact_sha,
            context.max_unavailable,
            wave,
        ),
    }
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "patch",
            "configmap",
            config_map,
            "--type=merge",
            "-p",
            json.dumps({"data": desired}, sort_keys=True),
        )
    )
    observed = (
        release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "configmap",
                config_map,
            )
        ).get("data")
        or {}
    )
    if {key: str(observed.get(key) or "") for key in desired} != desired:
        raise ReleaseError(
            f"{target.cluster_id} installer wave ConfigMap {config_map} did not "
            "accept the wave"
        )
    return identity


def run_fleet_waves(
    release: Any,
    target: ClusterTarget,
    context: FleetWaveContext,
    deployment: dict[str, Any],
) -> None:
    while deployment.get("status") != "SUCCEEDED":
        if deployment.get("status") == "FAILED":
            raise ReleaseError(f"{target.cluster_id} fleet deployment failed")
        expected_wave = next_deployment_wave(deployment)
        if not expected_wave:
            raise ReleaseError(
                f"{target.cluster_id} fleet deployment has no pending wave"
            )
        wave_started = time.monotonic()
        # A wave is the unit of work of this loop, and until it finishes nothing
        # is checkpointed: `data-plane-progress` is written once per cluster, so
        # on a fleet of one-node waves the release goes minutes per node without
        # a line. This says which wave, which nodes, and how much of the fleet is
        # already done -- the answer to "is it stuck or is it working".
        narrate_step(
            "fleet-wave",
            cluster=target.cluster_id,
            phase=context.phase,
            wave=deployment_wave_position(deployment, expected_wave),
            ready=deployment_node_progress(deployment),
            nodes=",".join(sorted(expected_wave)),
        )
        if context.phase == "rollback":
            record_rollback_wave_event(
                release,
                cluster_id=target.cluster_id,
                wave=expected_wave,
                event="safety_started",
            )
        try:
            # A second clock on purpose: `wave_started` measures the whole safety
            # stage for the narration below, and also covers the wave line and --
            # in a rollback -- a control-plane write. This one measures how much
            # the lease evidence has aged, which starts at the probe.
            probe_started = time.monotonic()
            safety = ensure_rollout_wave_safe(
                release,
                target,
                wave=expected_wave,
                node_names=context.node_names,
            )
            lease = release._fleet_command(
                "next-wave",
                {"deployment_id": context.deployment_id},
            )
            wave = tuple(str(item) for item in lease.get("node_ids", []))
            if not wave:
                raise ReleaseError(
                    f"{target.cluster_id} fleet deployment returned an empty wave"
                )
            if wave != expected_wave:
                raise ReleaseError(
                    f"{target.cluster_id} fleet wave changed after safety validation"
                )
            if not wave_lease_margin_holds(
                safety,
                elapsed_seconds=time.monotonic() - probe_started,
            ):
                ensure_rollout_wave_safe(
                    release,
                    target,
                    wave=wave,
                    node_names=context.node_names,
                    timeout_seconds=0,
                    minimum_lease_remaining_seconds=(
                        ROLLOUT_AGENT_POST_LEASE_MARGIN_SECONDS
                    ),
                )
            safety_seconds = time.monotonic() - wave_started
            if context.phase == "rollback":
                record_rollback_wave_event(
                    release,
                    cluster_id=target.cluster_id,
                    wave=expected_wave,
                    event="safety_completed",
                )
            handoff_started = time.monotonic()
            identity = hand_wave_to_reconciler(release, target, context, wave)
            if identity is None:
                # Compatibility path: a Reconciler deployed before the wave
                # ConfigMap only learns its allowed set from its environment.
                identity = release._deploy_reconciler(
                    target,
                    wheel_cm=context.wheel_cm,
                    bundle_cm=context.bundle_cm,
                    artifact_sha=context.artifact_sha,
                    config_digest=context.config_digest,
                    runtime_profile_version=context.expected_profile,
                    executor_wheel_filename=context.executor_wheel_filename,
                    node_compatibility_digest=context.expected_compatibility,
                    bundle_sha256=context.desired_bundle,
                    template_sha256=context.desired_template,
                    template_config_map=context.template_config_map,
                    allowed_node_names=wave,
                    max_unavailable=context.max_unavailable,
                    sync_registry=False,
                    runtime_image=context.runtime_image,
                    node_installer_image=context.node_installer_image,
                )
            if identity != context.paused_identity:
                raise ReleaseError(
                    f"{target.cluster_id} installer identity changed between waves"
                )
            if context.phase == "rollback":
                record_rollback_wave_event(
                    release,
                    cluster_id=target.cluster_id,
                    wave=wave,
                    event="reconciler_applied",
                )
            install_started = time.monotonic()
            release._wait_agents(
                target,
                context.artifact_sha,
                bundle_sha=context.desired_bundle,
                template_sha=context.desired_template,
                config_digest=context.config_digest,
                runtime_profile_version=context.expected_profile,
                node_names=wave,
                legacy_identity=context.allow_legacy_identity,
                agent_identity=context.agent_identity,
            )
            install_seconds = time.monotonic() - install_started
            if context.phase == "rollback":
                record_rollback_wave_event(
                    release,
                    cluster_id=target.cluster_id,
                    wave=wave,
                    event="agents_converged",
                )
            deployment = release._fleet_command(
                "get",
                {"deployment_id": context.deployment_id},
            )
            # Split three ways because the three are fixed by different things:
            # `safety` by the Agent leases and open commands, `handoff` by the
            # Reconciler rollout and the Jobs of the previous wave, `install` by
            # the node itself. Only the last one is the cluster doing real work.
            narrate_step(
                "fleet-wave-done",
                cluster=target.cluster_id,
                wave=deployment_wave_position(deployment, wave),
                ready=deployment_node_progress(deployment),
                safety=f"{safety_seconds:.1f}s",
                handoff=f"{install_started - handoff_started:.1f}s",
                install=f"{install_seconds:.1f}s",
                elapsed=f"{time.monotonic() - wave_started:.1f}s",
            )
            if context.phase == "rollback":
                record_rollback_wave_event(
                    release,
                    cluster_id=target.cluster_id,
                    wave=wave,
                    event="completed",
                )
        except Exception as exc:
            if context.phase == "rollback":
                record_rollback_wave_event(
                    release,
                    cluster_id=target.cluster_id,
                    wave=expected_wave,
                    event="failed",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            raise


def deploy_reconciler(
    release: Any,
    target: ClusterTarget,
    *,
    wheel_cm: str,
    bundle_cm: str,
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    node_compatibility_digest: str | None = None,
    bundle_sha256: str | None = None,
    template_sha256: str | None = None,
    template_config_map: str | None = None,
    allowed_node_names: tuple[str, ...] | None = None,
    max_unavailable: int | None = None,
    sync_registry: bool = True,
    runtime_image: str | None = None,
    node_installer_image: str | None = None,
) -> tuple[str, str]:
    release._settle_installer_jobs(target)
    # The Reconciler's own knob is a count of nodes it may install on at once,
    # so it can never be the "auto" sentinel: a caller that does not hand a
    # wave policy (the Reconciler-only component path) gets the conservative
    # one-at-a-time the site default used to spell out.
    max_unavailable = max_unavailable or release.config.upgrade_max_unavailable or 1
    environment = build_reconciler_environment(
        release,
        target,
        wheel_cm=wheel_cm,
        bundle_cm=bundle_cm,
        artifact_sha=artifact_sha,
        config_digest=config_digest,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        node_compatibility_digest=node_compatibility_digest,
        bundle_sha256=bundle_sha256,
        template_sha256=template_sha256,
        template_config_map=template_config_map,
        allowed_node_names=allowed_node_names,
        max_unavailable=max_unavailable,
        sync_registry=sync_registry,
        runtime_image=runtime_image,
        node_installer_image=node_installer_image,
    )
    environment["GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT"] = "false"
    release.runner.run(
        [str(ROOT / "deploy/node/deploy-node-installer-reconciler.sh")],
        env=environment,
        sensitive=bool(target.fleet_master_file),
        timeout_seconds=660,
    )
    wait_deployment_rollout(
        release,
        target,
        inventory.GPU_RECONCILER_DEPLOYMENT,
        timeout_seconds=600,
    )
    if release.runner.dry_run:
        return (
            bundle_sha256 or release.bundle_sha,
            template_sha256 or release.node_template_sha,
        )
    return reconciler_installer_identity(reconciler_container_env(release, target))
