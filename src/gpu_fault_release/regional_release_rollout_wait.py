from __future__ import annotations

import time
from typing import Any

from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError

FATAL_WAITING_REASONS = frozenset(
    {
        "CrashLoopBackOff",
        "CreateContainerConfigError",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
    }
)
CAPACITY_MESSAGES = (
    "insufficient cpu",
    "insufficient memory",
    "insufficient nvidia.com/gpu",
)


def _kubectl_timeout(timeout_seconds: float) -> str:
    whole = int(timeout_seconds)
    if timeout_seconds == whole and whole % 60 == 0:
        return f"{whole // 60}m"
    return f"{whole}s"


def _selector(deployment: dict[str, Any]) -> str:
    labels = (deployment.get("spec") or {}).get("selector", {}).get("matchLabels", {})
    if not isinstance(labels, dict) or not labels:
        raise ReleaseError("Deployment selector has no matchLabels")
    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _rollout_progress(deployment: dict[str, Any]) -> tuple[int, int, int, int]:
    status = deployment.get("status") or {}
    return (
        int(status.get("observedGeneration") or 0),
        int(status.get("updatedReplicas") or 0),
        int(status.get("readyReplicas") or 0),
        int(status.get("availableReplicas") or 0),
    )


def _rollout_complete(deployment: dict[str, Any]) -> bool:
    metadata = deployment.get("metadata") or {}
    spec = deployment.get("spec") or {}
    status = deployment.get("status") or {}
    desired = int(spec.get("replicas") or 0)
    generation = int(metadata.get("generation") or 0)
    observed, updated, ready, available = _rollout_progress(deployment)
    unavailable = int(status.get("unavailableReplicas") or 0)
    return (
        observed >= generation
        and updated == desired
        and ready == desired
        and available == desired
        and unavailable == 0
    )


def _deployment_failure(deployment: dict[str, Any]) -> str | None:
    for condition in (deployment.get("status") or {}).get("conditions") or []:
        if (
            condition.get("type") == "Progressing"
            and condition.get("status") == "False"
        ):
            return str(
                condition.get("reason")
                or condition.get("message")
                or "Deployment stopped progressing"
            )
    return None


def _pod_failure(
    pods: dict[str, Any],
    *,
    no_progress_seconds: float,
    capacity_grace_seconds: float,
) -> str | None:
    for pod in pods.get("items") or []:
        name = str((pod.get("metadata") or {}).get("name") or "unknown")
        status = pod.get("status") or {}
        for container in status.get("containerStatuses") or []:
            waiting = (container.get("state") or {}).get("waiting") or {}
            reason = str(waiting.get("reason") or "")
            if reason in FATAL_WAITING_REASONS:
                return f"Pod {name} container is {reason}"
        for condition in status.get("conditions") or []:
            if (
                condition.get("type") != "PodScheduled"
                or condition.get("status") != "False"
                or condition.get("reason") != "Unschedulable"
            ):
                continue
            message = str(condition.get("message") or "unschedulable")
            capacity_only = any(value in message.lower() for value in CAPACITY_MESSAGES)
            if not capacity_only or no_progress_seconds >= capacity_grace_seconds:
                return f"Pod {name} is unschedulable: {message}"
    return None


def wait_deployment_rollout(
    release: Any,
    target: ClusterTarget,
    deployment_name: str,
    *,
    timeout_seconds: float = 300,
    no_progress_timeout_seconds: float = 90,
    capacity_grace_seconds: float = 60,
    poll_seconds: float = 2,
) -> dict[str, Any]:
    if release.runner.dry_run:
        release.runner.run(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "rollout",
                "status",
                f"deployment/{deployment_name}",
                f"--timeout={_kubectl_timeout(timeout_seconds)}",
            ),
            timeout_seconds=timeout_seconds + 30,
        )
        return {"deployment": deployment_name, "dry_run": True}
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_progress_at = started
    last_progress: tuple[int, int, int, int] | None = None
    while time.monotonic() < deadline:
        deployment = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                deployment_name,
            )
        )
        failure = _deployment_failure(deployment)
        if failure is not None:
            raise ReleaseError(
                f"{target.cluster_id} Deployment {deployment_name} failed: {failure}"
            )
        progress = _rollout_progress(deployment)
        observed_at = time.monotonic()
        if progress != last_progress:
            last_progress = progress
            last_progress_at = observed_at
        if _rollout_complete(deployment):
            return {
                "deployment": deployment_name,
                "duration_seconds": max(0.0, observed_at - started),
                "progress": list(progress),
                # The generation that satisfied the barrier, so a caller that
                # needs to read something off the Deployment -- its container
                # environment, say -- does not have to fetch it again and hope it
                # is still the same one.
                "object": deployment,
            }
        pods = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "pods",
                "-l",
                _selector(deployment),
            )
        )
        no_progress = observed_at - last_progress_at
        failure = _pod_failure(
            pods,
            no_progress_seconds=no_progress,
            capacity_grace_seconds=capacity_grace_seconds,
        )
        if failure is not None:
            raise ReleaseError(
                f"{target.cluster_id} Deployment {deployment_name} failed: {failure}"
            )
        if no_progress >= no_progress_timeout_seconds:
            raise ReleaseError(
                f"{target.cluster_id} Deployment {deployment_name} made no "
                f"progress for {int(no_progress)} seconds; progress={progress}"
            )
        time.sleep(poll_seconds)
    raise ReleaseError(
        f"{target.cluster_id} Deployment {deployment_name} exceeded "
        f"{int(timeout_seconds)} seconds"
    )
