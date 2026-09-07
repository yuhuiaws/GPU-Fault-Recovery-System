from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault.hyperpod_spares import (
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    SparePoolState,
)
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_gpu_rollout import gpu_node_items
from gpu_fault_release.regional_release_rendering import build_reconciler_environment

ROOT = Path(__file__).resolve().parents[2]
CORDON_TAINT = "node.kubernetes.io/unschedulable"
BLOCKING_TAINT_KEYS = frozenset(
    {
        "gpu-fault.io/quarantined",
        CORDON_TAINT,
    }
)
HYPERPOD_HEALTH_TAINT = "sagemaker.amazonaws.com/node-health-status"


@dataclass(frozen=True)
class NodeMutationPreflight:
    phase: str
    node_names: tuple[str, ...]
    wheel_cm: str
    bundle_cm: str
    artifact_sha: str
    config_digest: str
    runtime_profile_version: str
    executor_wheel_filename: str | None
    node_compatibility_digest: str
    bundle_sha256: str | None
    template_sha256: str | None
    template_config_map: str | None
    max_unavailable: int
    runtime_image: str | None
    node_installer_image: str | None


def is_parked_warm_spare(metadata: dict[str, Any]) -> bool:
    """Is this node cordoned only because it is waiting in the warm-spare pool?

    A warm spare has to be cordoned: `HyperPodSpareCoordinator` rejects an
    unreserved candidate whose node is schedulable ("unreserved spare is
    schedulable"), because a spare that accepts ordinary Pods cannot be handed
    to a failover intact. So the pool's steady state and this gate's original
    reading of a cordon -- "somebody is draining or repairing this node, keep
    the release off it" -- disagree: with a pool declared, every upgrade failed
    the barrier and rolled back, which means a site could keep spares or keep
    receiving fixes, but not both.

    A parked spare is safe to mutate, and has to be: it holds no workloads,
    and it is only allocatable while its node agent stays fleet-ready on the
    current node runtime. The installer already tolerates the cordon taint
    (`regional_gpu_bootstrap.py`) and nothing in the rollout uncordons a node,
    so the pool invariant survives the wave. Only the cordon is forgiven --
    a quarantine taint, an unready node, a deletion or an active installer
    still blocks, and so does a spare that is reserved or in any pool state
    other than AVAILABLE, because those mean an allocation is in flight.
    """

    annotations = metadata.get("annotations") or {}
    if str(annotations.get(SPARE_RESERVATION_ANNOTATION) or ""):
        return False
    return str(annotations.get(SPARE_POOL_STATE_ANNOTATION) or "") == (
        SparePoolState.AVAILABLE.value
    )


def validate_target_node_state(
    release: Any,
    target: ClusterTarget,
    node_names: tuple[str, ...],
) -> None:
    expected = set(node_names)
    # A safety gate never reuses a pinned inventory; a node that dropped out of
    # the HyperPod label selector shows up as "missing" and blocks the wave.
    selected = {
        str((item.get("metadata") or {}).get("name") or ""): item
        for item in gpu_node_items(release, target, fresh=True)
        if str((item.get("metadata") or {}).get("name") or "") in expected
    }
    blockers: list[dict[str, Any]] = []
    uids: set[str] = set()
    for node_name in sorted(expected):
        item = selected.get(node_name)
        if item is None:
            blockers.append({"node_id": node_name, "reason": "missing"})
            continue
        metadata = item.get("metadata") or {}
        spec = item.get("spec") or {}
        status = item.get("status") or {}
        uid = str(metadata.get("uid") or "")
        if not uid:
            blockers.append({"node_id": node_name, "reason": "uid-missing"})
        elif uid in uids:
            blockers.append({"node_id": node_name, "reason": "uid-duplicate"})
        else:
            uids.add(uid)
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in status.get("conditions") or []
            if isinstance(condition, dict)
        )
        if not ready:
            blockers.append({"node_id": node_name, "reason": "not-ready"})
        if metadata.get("deletionTimestamp"):
            blockers.append({"node_id": node_name, "reason": "deleting"})
        parked_spare = is_parked_warm_spare(metadata)
        if bool(spec.get("unschedulable")) and not parked_spare:
            blockers.append({"node_id": node_name, "reason": "cordoned"})
        for taint in spec.get("taints") or []:
            if not isinstance(taint, dict):
                continue
            key = str(taint.get("key") or "")
            value = str(taint.get("value") or "")
            if key == CORDON_TAINT and parked_spare:
                continue
            if key in BLOCKING_TAINT_KEYS or (
                key == HYPERPOD_HEALTH_TAINT and value == "Unschedulable"
            ):
                blockers.append(
                    {
                        "node_id": node_name,
                        "reason": "blocking-taint",
                        "taint_key": key,
                    }
                )
        installer_state = str(
            (metadata.get("annotations") or {}).get("gpu-fault.io/installer-state")
            or ""
        )
        if installer_state in {"Installing", "Retrying"}:
            blockers.append(
                {
                    "node_id": node_name,
                    "reason": "installer-active",
                    "installer_state": installer_state,
                }
            )
    if blockers:
        raise ReleaseError(
            f"{target.cluster_id} pre-node-mutation node state is unsafe: "
            + json.dumps(blockers[:100], sort_keys=True)
        )


def run_node_installer_preflight(
    release: Any,
    target: ClusterTarget,
    candidate: NodeMutationPreflight,
) -> None:
    environment = build_reconciler_environment(
        release,
        target,
        wheel_cm=candidate.wheel_cm,
        bundle_cm=candidate.bundle_cm,
        artifact_sha=candidate.artifact_sha,
        config_digest=candidate.config_digest,
        runtime_profile_version=candidate.runtime_profile_version,
        executor_wheel_filename=candidate.executor_wheel_filename,
        node_compatibility_digest=candidate.node_compatibility_digest,
        bundle_sha256=candidate.bundle_sha256,
        template_sha256=candidate.template_sha256,
        template_config_map=candidate.template_config_map,
        allowed_node_names=candidate.node_names,
        max_unavailable=candidate.max_unavailable,
        sync_registry=False,
        runtime_image=candidate.runtime_image,
        node_installer_image=candidate.node_installer_image,
    )
    environment.update(
        {
            "GPU_FAULT_RECONCILER_PREFLIGHT_ONLY": "true",
            "GPU_FAULT_REQUIRE_ROLLBACK_SLOT": str(
                candidate.phase not in {"bootstrap", "join"}
            ).lower(),
            "GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT": "false",
        }
    )
    raw = release.runner.run(
        [str(ROOT / "deploy/node/deploy-node-installer-reconciler.sh")],
        env=environment,
        capture=True,
        sensitive=bool(target.fleet_master_file),
        timeout_seconds=max(600, len(candidate.node_names) * 300),
    )
    lines = [line for line in str(raw).splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ReleaseError(
            f"{target.cluster_id} node preflight returned invalid evidence"
        ) from exc
    if result.get("status") != "PASSED" or int(result.get("node_count") or 0) != len(
        candidate.node_names
    ):
        raise ReleaseError(
            f"{target.cluster_id} node preflight did not cover every target node"
        )


def ensure_node_candidate_preflight(
    release: Any,
    target: ClusterTarget,
    candidate: NodeMutationPreflight,
) -> None:
    validate_target_node_state(release, target, candidate.node_names)
    run_node_installer_preflight(release, target, candidate)


def ensure_pre_node_mutation_barrier(
    release: Any,
    target: ClusterTarget,
    candidate: NodeMutationPreflight,
    *,
    ensure_runtime_safe: Callable[..., None],
) -> None:
    validate_target_node_state(release, target, candidate.node_names)
    ensure_runtime_safe(
        release,
        target,
        wave=(
            () if candidate.phase in {"upgrade", "rollback"} else candidate.node_names
        ),
        node_names=candidate.node_names,
    )
