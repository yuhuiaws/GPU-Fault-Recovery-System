from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
from regional_admin_checks import build_health_report
from regional_release_config import ReleaseError
from regional_release_diff import (
    ReleaseChangeKind,
    ReleaseDiff,
    classify_release,
    diff_from_changed,
)
from regional_release_reporting import build_release_status

STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
ROOT = Path(__file__).resolve().parents[3]
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
    }
)
RETRY_PHASES = frozenset({*RESUMABLE_PHASES, "rolled-back"})
BOOTSTRAP_PHASES = frozenset(
    {
        "bootstrap-started",
        "bootstrap-cpu-ready",
        "bootstrap-endpoint-ready",
        "bootstrap-data-plane-progress",
        "bootstrap-failed",
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
    state_exists = (
        subprocess.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "configmap",
                STATE_CONFIG_MAP,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
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
    if state.get("phase") in RETRY_PHASES:
        release.upgrade(
            resume=state.get("phase") in RESUMABLE_PHASES,
            diff=retry_release_diff(release, state),
        )
        return
    diff = classify_release(release, state)
    if diff.kind == ReleaseChangeKind.NOOP:
        release.noop(diff)
        return
    release.upgrade(diff=diff)


def run_resume(release: Any) -> None:
    state = release._load_state()
    phase = str(state.get("phase") or "")
    if phase not in RESUMABLE_PHASES:
        raise ReleaseError(
            "resume requires an incomplete upgrade transaction; "
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
        retry_diff = (
            retry_release_diff(release, state)
            if state.get("phase") in RETRY_PHASES
            else None
        )
        result["next_deploy"] = (
            {
                **retry_diff.as_dict(),
                "resume": state.get("phase") in RESUMABLE_PHASES,
            }
            if retry_diff is not None
            else classify_release(release, state).as_dict()
        )
    except Exception as exc:
        result["next_deploy_error"] = str(exc)
    return result


def build_full_status(release: Any) -> dict[str, Any]:
    health = build_health_report(release, mode="status")
    result = build_release_summary(release)
    result["mode"] = "status"
    result["healthy"] = health["healthy"]
    result["health"] = health
    return result
