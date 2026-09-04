from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
from regional_admin_checks import (
    # The same tolerant wrapper the checks use: a release object that predates
    # the read cache, or a test double standing in for one, has no
    # `_read_snapshot` and gets a `nullcontext` instead of an AttributeError.
    _read_snapshot as read_snapshot,
)
from regional_admin_checks import (
    build_health_report,
)
from regional_release_config import ReleaseError
from regional_release_diff import (
    ReleaseChangeKind,
    ReleaseDiff,
    build_execution_plan,
    classify_release,
    diff_from_changed,
)
from regional_release_reporting import build_release_status

STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
ROOT = Path(__file__).resolve().parents[3]
EXPECTED_STATE_SHA256_ENV = "GPU_FAULT_EXPECTED_RELEASE_STATE_SHA256"
RESUMABLE_PHASES = frozenset(
    {
        "preflight",
        "uploaded",
        "schema-ready",
        "registry-staged",
        "cpu-staged",
        "profile-ready",
        "endpoint-ready",
        "observability-ready",
        "data-plane-progress",
        "data-converged",
        "cpu-finalized",
        "verified",
        "failed",
        "partial-convergence",
    }
)
ROLLBACK_PHASES = frozenset(
    {
        "rollback-started",
        "rollback-controller-staging",
        "rollback-controller-staged",
        "rollback-observability-restoring",
        "rollback-observability-restored",
        "rollback-endpoint-restoring",
        "rollback-endpoint-restored",
        "rollback-data-restoring",
        "rollback-data-progress",
        "rollback-data-restored",
        "rollback-rollout-cleaning",
        "rollback-rollout-cleaned",
        "rollback-cpu-restoring",
        "rollback-cpu-restored",
        "rollback-restored",
        "rollback-verifying",
        "rollback-verified",
        "rollback-failed",
    }
)
BOOTSTRAP_PHASES = frozenset(
    {
        "bootstrap-started",
        "bootstrap-cpu-ready",
        "bootstrap-endpoint-ready",
        "bootstrap-data-plane-progress",
        "bootstrap-failed",
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-failed",
        "bootstrap-cleaned",
    }
)


def stored_release_diff(state: dict[str, Any]) -> ReleaseDiff | None:
    value = state.get("release_diff")
    if not isinstance(value, dict):
        return None
    try:
        kind = ReleaseChangeKind(str(value["kind"]))
        changed = frozenset(str(item) for item in value["changed"])
    except (KeyError, TypeError, ValueError):
        return None
    return ReleaseDiff(kind=kind, changed=changed)


def release_state_sha256(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rollback_pending(state: dict[str, Any]) -> bool:
    phase = str(state.get("phase") or "")
    return phase in ROLLBACK_PHASES or (
        phase == "rolled-back" and state.get("rollback_cleanup_completed") is False
    )


def _commit_cleanup_pending(state: dict[str, Any]) -> bool:
    return (
        state.get("phase") == "complete"
        and state.get("transaction_committed") is True
        and state.get("commit_cleanup_completed") is False
    )


def _upgrade_resume_required(state: dict[str, Any]) -> bool:
    phase = str(state.get("phase") or "")
    return phase in RESUMABLE_PHASES or (
        phase == "complete" and state.get("transaction_committed") is False
    )


def next_deploy(release: Any, state: dict[str, Any]) -> dict[str, Any]:
    if _rollback_pending(state):
        return {
            **retry_release_diff(release, state).as_dict(),
            "resume": True,
            "action": "rollback",
        }
    if _commit_cleanup_pending(state):
        return {
            **retry_release_diff(release, state).as_dict(),
            "resume": True,
            "action": "commit",
        }
    phase = str(state.get("phase") or "")
    if _upgrade_resume_required(state) or phase == "rolled-back":
        return {
            **retry_release_diff(release, state).as_dict(),
            "resume": _upgrade_resume_required(state),
            "action": "upgrade",
        }
    return classify_release(release, state).as_dict()


def _require_expected_state(state: dict[str, Any]) -> None:
    expected = os.getenv(EXPECTED_STATE_SHA256_ENV, "").strip()
    if expected and release_state_sha256(state) != expected:
        raise ReleaseError(
            "regional release state changed after the deployment diff was calculated"
        )


def retry_release_diff(release: Any, state: dict[str, Any]) -> ReleaseDiff:
    persisted = stored_release_diff(state)
    changed = set(persisted.changed if persisted is not None else ())
    changed.update(classify_release(release, state).changed)
    previous = state.get("previous")
    if not isinstance(previous, dict):
        return diff_from_changed(changed)

    metadata = previous.get("metadata") or {}
    if previous.get("cpu_wheel") != release.wheel_cm:
        changed.add("control_plane_wheel")
    if (
        metadata.get("required-agent-artifact-sha256")
        and metadata.get("required-agent-artifact-sha256") != release.node_wheel_sha
    ):
        changed.add("node_runtime_wheel")
    if (
        metadata.get("required-regional-executor-artifact-sha256")
        and metadata.get("required-regional-executor-artifact-sha256")
        != release.executor_wheel_sha
    ):
        changed.add("executor_wheel")

    clusters = previous.get("clusters") or {}
    for target in release.config.clusters:
        old = clusters.get(target.cluster_id) or {}
        if any(
            old.get(name) and old.get(name) != release.executor_wheel_cm
            for name in ("wheel", "reconciler_wheel")
        ):
            changed.add("executor_wheel")
        if old.get("bundle") and old.get("bundle") != release.bundle_cm:
            changed.add("node_bundle")
    return diff_from_changed(changed)


def ensure_schema(release: Any) -> None:
    release.runner.run(
        [str(ROOT / "deploy/control-plane/tools/ensure-postgres-schema.sh")],
        env={
            **os.environ,
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": release.config.cpu_kubeconfig,
            "GPU_FAULT_NAMESPACE": release.config.namespace,
            "GPU_FAULT_WHEEL_CONFIGMAP": release.wheel_cm,
            "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
        },
    )


def bootstrap_cpu_is_current(release: Any) -> bool:
    try:
        metadata = release._config_map_data("gpu-fault-release-metadata")
        expected = {
            "required-agent-artifact-sha256": release.node_wheel_sha,
            "required-agent-compatibility-digest": (
                release.config.component_digests.get("node_runtime")
                or release.node_wheel_sha
            ),
            "required-regional-executor-artifact-sha256": (release.executor_wheel_sha),
            "required-regional-executor-compatibility-digest": (
                release.config.component_digests.get("executor")
                or release.executor_wheel_sha
            ),
            "required-agent-config-digest": release.config.agent_config_digest,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if (
            release._deployment_wheel(
                release._cpu(),
                inventory.CPU_INGRESS_DEPLOYMENT,
            )
            != release.wheel_cm
        ):
            return False
        for deployment in inventory.CPU_DEPLOYMENTS:
            value = release._get_json(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "get",
                    "deployment",
                    deployment,
                )
            )
            desired = int((value.get("spec") or {}).get("replicas") or 0)
            status = value.get("status") or {}
            metadata_value = value.get("metadata") or {}
            if int(status.get("observedGeneration") or 0) < int(
                metadata_value.get("generation") or 0
            ):
                return False
            if any(
                int(status.get(field) or 0) != desired
                for field in (
                    "readyReplicas",
                    "updatedReplicas",
                    "availableReplicas",
                )
            ):
                return False
        return True
    except (ReleaseError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_deploy(release: Any) -> None:
    state_exists = release.runner.probe(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            STATE_CONFIG_MAP,
        ),
    )
    expected_state = os.getenv(EXPECTED_STATE_SHA256_ENV, "").strip()
    if expected_state and not state_exists:
        raise ReleaseError(
            "regional release state disappeared after the deployment diff was calculated"
        )
    state = release._load_state() if state_exists else None
    bootstrap_required = not state_exists or (
        state is not None and state.get("phase") in BOOTSTRAP_PHASES
    )
    if bootstrap_required:
        if not release.config.clusters:
            raise ReleaseError(
                "initial regional bootstrap requires at least one GPU cluster; "
                "an empty cluster set is only valid after a completed deployment"
            )
        release.bootstrap()
        return
    assert state is not None
    _require_expected_state(state)
    phase = str(state.get("phase") or "")
    if _rollback_pending(state):
        release.rollback()
        raise ReleaseError(
            "rollback recovery completed; rerun deploy to start a new transaction"
        )
    if _commit_cleanup_pending(state):
        release.commit_release()
        return
    if _upgrade_resume_required(state) or phase == "rolled-back":
        release.upgrade(
            resume=_upgrade_resume_required(state),
            diff=retry_release_diff(release, state),
        )
        return
    diff = classify_release(release, state)
    if diff.kind == ReleaseChangeKind.NOOP:
        release.noop(diff)
        return
    release.upgrade(diff=diff)


def build_release_diff(release: Any) -> dict[str, Any]:
    state = release._load_state()
    return {
        "mode": "release-diff",
        "state_sha256": release_state_sha256(state),
        "next_deploy": next_deploy(release, state),
    }


def stage_noop_release(release: Any) -> None:
    state = release._load_state()
    _require_expected_state(state)
    diff = classify_release(release, state)
    if diff.kind is not ReleaseChangeKind.NOOP:
        raise ReleaseError(
            f"stage-noop requires a current NOOP release diff; got {diff.kind.value}"
        )
    release._ensure_contexts()
    release._require_cpu_secrets()
    release.state = dict(state)
    release._save_state(
        "complete",
        transaction_committed=False,
        previous=None,
        release_diff=diff.as_dict(),
        execution_plan=build_execution_plan(diff).as_dict(),
        completed_phases=[],
        completed_cluster_ids=[],
    )


def run_resume(release: Any) -> None:
    state = release._load_state()
    phase = str(state.get("phase") or "")
    if _rollback_pending(state):
        release.rollback()
        return
    if _commit_cleanup_pending(state):
        release.commit_release()
        return
    if not _upgrade_resume_required(state):
        raise ReleaseError(
            "resume requires an incomplete upgrade or rollback transaction; "
            f"current phase is {phase or 'unknown'}"
        )
    release.upgrade(
        resume=True,
        diff=retry_release_diff(release, state),
    )


def build_release_summary(release: Any) -> dict[str, Any]:
    result: dict[str, Any]
    try:
        result = build_release_status(release)
    except Exception as exc:
        result = {
            "site_name": release.config.site_name,
            "release_status_error": str(exc),
        }
    result["mode"] = "release-summary"
    try:
        state = release._load_state()
        result["live_release"] = {
            "release_id": state.get("release_id"),
            "phase": state.get("phase"),
            "transaction_committed": state.get("transaction_committed") is True,
            "release_lifecycle": state.get("release_lifecycle"),
            "state_sha256": release_state_sha256(state),
        }
        result["next_deploy"] = next_deploy(release, state)
    except Exception as exc:
        result["next_deploy_error"] = str(exc)
    return result


def build_full_status(release: Any) -> dict[str, Any]:
    # One snapshot for both halves. `build_health_report` opens its own, and
    # everything `build_release_summary` reads -- the release state ConfigMap,
    # the live Deployments behind `next_deploy` -- the health checks have already
    # read inside it, so without this the summary re-issued each of those
    # `kubectl get` calls after the health report's snapshot had been torn down.
    # It also means the summary describes the same observation as the checks
    # reported beside it, which is the whole point of a status output.
    with read_snapshot(release):
        health = build_health_report(release, mode="status")
        result = build_release_summary(release)
    result["mode"] = "status"
    result["healthy"] = health["healthy"]
    result["health"] = health
    return result
