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


def bounded_kubectl_wait(release: Any, wait_args: list[str], *, seconds: float) -> bool:
    """Spend at most ``seconds`` on a ``kubectl wait``/``rollout status``.

    A poll loop used to sleep a fixed interval between reads, so a condition that
    held one second into the interval was noticed at its end. This runs the
    wait the caller built with ``--timeout`` set to the interval -- never to the
    release's whole deadline -- so the loop wakes the moment the condition holds
    and, at the latest, when the interval it would have slept has passed.

    The answer is advisory. ``True`` says kubectl saw the condition; ``False``
    covers the timeout and every other non-zero exit, a transient API error
    included. Either way the caller re-reads the object exactly as it did
    before, so nothing here changes what is judged or when it fails. A wait
    that gives up early without the condition still costs the full interval:
    the next read then happens no sooner than the old loop's would have, and an
    API server that answered one call with an error gets the same pause it used
    to get before the next one. Intervals under a second sleep instead --
    ``--timeout=0s`` means "check once" to kubectl, and a fraction cannot be
    expressed.
    """

    if seconds < 1:
        if seconds > 0:
            time.sleep(seconds)
        return False
    whole = int(seconds)
    started = time.monotonic()
    try:
        met = bool(
            release.runner.probe(
                [*wait_args, f"--timeout={whole}s"],
                timeout_seconds=whole + 30,
            )
        )
    except ReleaseError:
        # A kubectl that outlives its own --timeout is killed by the runner and
        # reported as an error. The old sleep could not fail here, so neither
        # does this: it is "not yet", and the caller's next read decides.
        met = False
    if met:
        return True
    shortfall = seconds - (time.monotonic() - started)
    if shortfall > 0:
        time.sleep(shortfall)
    return False


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
    kubectl_saw_completion = False
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
        # The interval between inspections is spent inside `rollout status`
        # rather than a timer, so a rollout that completes early in it is read
        # back at once; the Pod inspection above still runs every interval, so a
        # Pod that cannot start is caught exactly when it used to be.
        interval = min(poll_seconds, max(0.0, deadline - time.monotonic()))
        if kubectl_saw_completion:
            # kubectl already called the rollout complete and the read above
            # disagreed (`rollout status` stops at "available"; this barrier also
            # wants every replica Ready and none unavailable). Asking again would
            # answer at once and spin, so this interval is a plain sleep.
            time.sleep(interval)
            kubectl_saw_completion = False
            continue
        kubectl_saw_completion = bounded_kubectl_wait(
            release,
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "rollout",
                "status",
                f"deployment/{deployment_name}",
            ),
            seconds=interval,
        )
    raise ReleaseError(
        f"{target.cluster_id} Deployment {deployment_name} exceeded "
        f"{int(timeout_seconds)} seconds"
    )
