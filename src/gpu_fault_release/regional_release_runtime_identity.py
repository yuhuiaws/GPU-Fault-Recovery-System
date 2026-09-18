from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from gpu_fault.admin.deploy_limits import DEPLOY_CONCURRENCY
from gpu_fault.admin.command_log import FAILURE_EXIT_CODE_ATTRIBUTE
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource

BASE_RUNTIME_PATH = (
    "/opt/app-root/bin:/opt/app-root/src/.local/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
CONTROL_PLANE_PATH = f"/opt/gpu-fault/control-plane/bin:{BASE_RUNTIME_PATH}"
EXECUTOR_PATH = f"/opt/gpu-fault/executor/bin:{BASE_RUNTIME_PATH}"
CONTROL_PLANE_PYTHON = "python"
EXECUTOR_PYTHON = "python"
EXECUTOR_READINESS = "gpu-fault-cluster-executor-readiness"
MODULE_DIGEST_SCRIPT = "from gpu_fault import module_digest; print(module_digest())"
MAX_RUNTIME_IDENTITY_WORKERS = DEPLOY_CONCURRENCY.read_only_checks
CPU_INGRESS_POD_ATTRIBUTE = "_cpu_ingress_pod"


def _cpu_ingress_pod_command(release: Any) -> list[str]:
    return release._cpu(
        "-n",
        release.config.namespace,
        "get",
        "pod",
        "-l",
        f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    )


def _cpu_ingress_pods_command(release: Any) -> list[str]:
    command: list[str] = release._cpu(
        "-n",
        release.config.namespace,
        "get",
        "pod",
        "-l",
        f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
        "--field-selector=status.phase=Running",
        "-o",
        "json",
    )
    return command


def _ingress_pod_ready_not_terminating(pod: dict[str, Any]) -> bool:
    """Whether ``pod`` can safely carry a long-lived in-Pod probe.

    A Pod the rollout is about to reap either already carries a
    ``deletionTimestamp`` (it is Terminating, still ``status.phase=Running``)
    or is not yet Ready. Handing the heartbeat barrier to either one is how a
    forward FULL release earns an ``exec`` killed with exit 137 mid-window.
    """

    if (pod.get("metadata") or {}).get("deletionTimestamp"):
        return False
    conditions = (pod.get("status") or {}).get("conditions") or []
    for condition in conditions:
        if isinstance(condition, dict) and condition.get("type") == "Ready":
            return str(condition.get("status")) == "True"
    return False


def resolve_cpu_ingress_pod(release: Any, *, failure: str) -> str:
    """Return a CPU ingress Pod name that can carry an in-Pod probe.

    The control plane rolls one replica at a time, so during a FULL release
    the outgoing and incoming ReplicaSets both have Running Pods for a window.
    Picking the first ``status.phase=Running`` Pod can hand a long barrier to
    an outgoing Pod the rollout then kills (exit 137), which -- for the
    ``retries=0`` barrier -- fails the whole release. Prefer a Pod that is
    Ready and not already Terminating; only when none qualifies (a first
    bootstrap window, or every Pod mid-restart) fall back to any Running Pod,
    which preserves the previous behaviour and its "no Pod" failure.
    """

    raw = release.runner.run(_cpu_ingress_pods_command(release), capture=True)
    pods: list[dict[str, Any]] = []
    if raw:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        items = payload.get("items") if isinstance(payload, dict) else None
        if isinstance(items, list):
            pods = [item for item in items if isinstance(item, dict)]
    for candidate in pods:
        if _ingress_pod_ready_not_terminating(candidate):
            name = str((candidate.get("metadata") or {}).get("name") or "")
            if name:
                return name
    for candidate in pods:
        name = str((candidate.get("metadata") or {}).get("name") or "")
        if name:
            return name
    raise ReleaseError(f"no running CPU ingress Pod for {failure}")


def cpu_ingress_pod(
    release: Any,
    *,
    failure: str,
    refresh: bool = False,
) -> str:
    """Return the CPU ingress Pod name, memoised for read-only probes.

    Wave safety barriers and heartbeat convergence checks re-resolve this Pod
    every few seconds even though its name only changes when the control plane
    itself rolls, so the lookup is cached per release run. Read-only callers
    must retry with ``refresh=True`` once so a replaced Pod is re-resolved
    instead of failing the release; mutating callers resolve it fresh.
    """

    if not refresh:
        cached = str(getattr(release, CPU_INGRESS_POD_ATTRIBUTE, "") or "")
        if cached:
            return cached
    pod = resolve_cpu_ingress_pod(release, failure=failure)
    setattr(release, CPU_INGRESS_POD_ATTRIBUTE, pod)
    return pod


def cpu_ingress_pod_if_running(release: Any) -> str:
    """The memoised ingress Pod name, or ``""`` when the fleet has none.

    For the one caller whose question the absence already answers: with no
    Running ingress Pod there is nothing left dispatching remote commands. The
    resolvers above raise instead, because for every other caller a missing
    control plane means the check could not be made at all.
    """

    cached = str(getattr(release, CPU_INGRESS_POD_ATTRIBUTE, "") or "")
    if cached:
        return cached
    pod = str(release.runner.run(_cpu_ingress_pod_command(release), capture=True) or "")
    if pod:
        setattr(release, CPU_INGRESS_POD_ATTRIBUTE, pod)
    return pod


def cpu_ingress_deployment_installed(release: Any) -> bool:
    """Whether ingress is installed, never a proof that durable work is absent."""
    document = probe_resource(
        release.runner,
        release._cpu(),
        ResourceRef(
            "deployment",
            "Deployment",
            inventory.CPU_INGRESS_DEPLOYMENT,
            release.config.namespace,
        ),
    ).require_readable()
    if document is None:
        return False
    spec = document.get("spec")
    replicas = spec.get("replicas", 1) if isinstance(spec, dict) else None
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 0:
        raise ReleaseError("CPU ingress Deployment replica count is invalid")
    return replicas > 0


def forget_cpu_ingress_pod(release: Any) -> None:
    setattr(release, CPU_INGRESS_POD_ATTRIBUTE, "")


def exec_cpu_ingress(
    release: Any,
    *,
    arguments: Sequence[str],
    failure: str,
    input_text: str | None = None,
    sensitive: bool = False,
    timeout_seconds: float | None = None,
    interactive: bool = True,
    retries: int = 1,
    retry_replaced: bool = True,
) -> str:
    """Exec ``arguments`` in the memoised ingress Pod.

    ``arguments`` is everything after ``kubectl exec <pod> --``, so a caller can
    prefix ``env`` assignments or pass a different interpreter; the two wrappers
    below are what call sites should reach for.

    With ``retries`` above zero a failure re-resolves the Pod and runs the
    command again, which is only sound for a read-only probe. A failed attempt
    always drops the memoised name so the next caller re-resolves it.

    A Pod that no longer exists, or one the rollout killed mid-exec, is not a
    failed attempt when ``retry_replaced`` allows it. The release rolls the CPU
    ingress Deployment itself, so a memoised name can be replaced between two
    probes -- the barrier that runs during a FULL release's CPU roll hits
    exactly that. ``kubectl exec`` then either reports ``NotFound`` without the
    command ever starting, or the running container is reaped and exits 137
    (SIGKILL) / 143 (SIGTERM). Either way the Pod is going away, so it buys one
    more attempt against a freshly resolved Pod -- one the new resolver prefers
    to pick Ready and not Terminating -- even for the callers that set
    ``retries=0`` because their in-Pod barrier already spends its whole window.
    A vanished Pod is looked up rather than the failure text matched, so that
    branch also holds for commands whose output is sensitive and must not be
    inspected; the kill branch reads the child's own exit code, never its
    output.
    """

    error: Exception | None = None
    attempts_left = retries + 1
    replacement_attempts_left = 1 if retry_replaced else 0
    refresh = False
    while attempts_left > 0:
        pod = cpu_ingress_pod(release, failure=failure, refresh=refresh)
        command = release._cpu(
            "-n",
            release.config.namespace,
            "exec",
            *(("-i",) if interactive else ()),
            pod,
            "--",
            *arguments,
        )
        try:
            output = release.runner.run(
                command,
                input_text=input_text,
                capture=True,
                sensitive=sensitive,
                timeout_seconds=timeout_seconds,
            )
            return str(output or "")
        except ReleaseError as exc:
            error = exc
            # Only read-only probes may retry a container killed mid-exec.
            killed_in_flight = getattr(exc, FAILURE_EXIT_CODE_ATTRIBUTE, None) in (
                137,
                143,
            )
            replaced = replacement_attempts_left > 0 and (
                killed_in_flight
                or not probe_resource(
                    release.runner,
                    release._cpu(),
                    ResourceRef("pod", "Pod", pod, release.config.namespace),
                ).exists()
            )
            forget_cpu_ingress_pod(release)
            refresh = True
            attempts_left -= 1
            if attempts_left == 0 and replaced:
                replacement_attempts_left -= 1
                attempts_left = 1
    assert error is not None
    raise error


def exec_cpu_ingress_probe(
    release: Any,
    *,
    script: str,
    failure: str,
    input_text: str | None = None,
    sensitive: bool = False,
    timeout_seconds: float | None = None,
    interactive: bool = True,
    retries: int = 1,
) -> str:
    """Run a read-only in-Pod probe against the memoised ingress Pod.

    Only read-only probes may use this: a failure here re-runs the script, which
    would be unsafe for a mutating command. Those use
    :func:`exec_cpu_ingress_command`.
    """

    return exec_cpu_ingress(
        release,
        arguments=(CONTROL_PLANE_PYTHON, "-c", script),
        failure=failure,
        input_text=input_text,
        sensitive=sensitive,
        timeout_seconds=timeout_seconds,
        interactive=interactive,
        retries=retries,
    )


def exec_cpu_ingress_command(
    release: Any,
    *,
    arguments: Sequence[str],
    failure: str,
    input_text: str | None = None,
    sensitive: bool = False,
    timeout_seconds: float | None = None,
    interactive: bool = True,
) -> str:
    """Run a mutating in-Pod command against the memoised ingress Pod, once.

    Mutating callers get the memoisation -- which is the whole point, a release
    resolved this Pod nineteen times in one run purely to build an exec command
    -- but never a second attempt. Neither retry the probe path grants is sound
    here: re-running is a second mutation, and the vanished-Pod allowance rests
    on "the command never started", which a ``get pod`` issued *after* the
    failure cannot establish. A Pod deleted midway through a write looks
    identical to one that was already gone.

    So a failure drops the memoised name and propagates. The cost of the
    memoisation is that an exec can now land on a name replaced since the last
    call and fail where a fresh resolution would have succeeded; the release
    aborts with a readable error and is resumable, and the CPU ingress
    Deployment only rolls in a phase these callers do not run in.
    """

    return exec_cpu_ingress(
        release,
        arguments=arguments,
        failure=failure,
        input_text=input_text,
        sensitive=sensitive,
        timeout_seconds=timeout_seconds,
        interactive=interactive,
        retries=0,
        retry_replaced=False,
    )


def _running_pods(
    release: Any,
    kubectl: list[str],
    deployment: str,
    expected_image: str | None = None,
) -> list[str]:
    value = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment,
        ]
    )
    replicas = int((value.get("spec") or {}).get("replicas") or 0)
    pods = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "pods",
            "-l",
            f"app={deployment}",
            "--field-selector=status.phase=Running",
        ]
    )
    names = sorted(
        str((item.get("metadata") or {}).get("name") or "")
        for item in pods.get("items", [])
        if not (item.get("metadata") or {}).get("deletionTimestamp")
    )
    names = [name for name in names if name]
    if expected_image is not None:
        for item in pods.get("items", []):
            if (item.get("metadata") or {}).get("deletionTimestamp"):
                continue
            containers = (item.get("spec") or {}).get("containers") or []
            if not containers or containers[0].get("image") != expected_image:
                raise ReleaseError(
                    f"{deployment} Pod OCI image differs from the release"
                )
    if len(names) != replicas:
        raise ReleaseError(
            f"{deployment} runtime identity has {len(names)}/{replicas} Running Pods"
        )
    return names


def _pod_digest(
    release: Any,
    kubectl: list[str],
    deployment: str,
    pod: str,
    *,
    python: str,
    path: str,
    expected: str,
) -> str:
    digest = release.runner.run(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "exec",
            pod,
            "--",
            "env",
            f"PATH={path}",
            python,
            "-c",
            MODULE_DIGEST_SCRIPT,
        ],
        capture=True,
    ).strip()
    if digest != expected:
        raise ReleaseError(
            f"{deployment}/{pod} module digest {digest!r} "
            f"does not match release component {expected}"
        )
    return str(digest)


def validate_runtime_component_identity(release: Any) -> dict[str, Any]:
    control_digest = str(release.config.component_digests.get("control_plane") or "")
    executor_digest = str(release.config.component_digests.get("executor") or "")
    if len(control_digest) != 64 or len(executor_digest) != 64:
        raise ReleaseError("release component module digests are unavailable")

    targets = [
        (
            "control-plane",
            deployment,
            release._cpu(),
            CONTROL_PLANE_PYTHON,
            CONTROL_PLANE_PATH,
            control_digest,
        )
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
    ]
    for target in release.config.clusters:
        targets.extend(
            (
                target.cluster_id,
                deployment,
                release._gpu(target),
                EXECUTOR_PYTHON,
                EXECUTOR_PATH,
                executor_digest,
            )
            for deployment in (
                *inventory.DEPLOYMENTS,
                inventory.GPU_RECONCILER_DEPLOYMENT,
            )
        )
    results: dict[tuple[str, str], dict[str, str]] = {
        (scope, deployment): {} for scope, deployment, *_rest in targets
    }
    errors: dict[str, str] = {}
    # Listings and Pod execs share one pool; nested per-Deployment pools would
    # multiply the bound as clusters and replicas grow.
    with ThreadPoolExecutor(max_workers=MAX_RUNTIME_IDENTITY_WORKERS) as executor:
        listings = {
            executor.submit(
                _running_pods,
                release,
                kubectl,
                deployment,
                (
                    release.runtime_image
                    if scope == "control-plane"
                    else release.executor_image
                )
                if release.config.release_manifest_schema_version >= 4
                else None,
            ): (
                scope,
                deployment,
                kubectl,
                python,
                path,
                expected,
            )
            for scope, deployment, kubectl, python, path, expected in targets
        }
        probes = {}
        for listing in as_completed(listings):
            scope, deployment, kubectl, python, path, expected = listings[listing]
            try:
                pods = listing.result()
            except Exception as exc:
                errors[f"{scope}/{deployment}"] = f"{type(exc).__name__}: {exc}"
                continue
            for pod in pods:
                probes[
                    executor.submit(
                        _pod_digest,
                        release,
                        kubectl,
                        deployment,
                        pod,
                        python=python,
                        path=path,
                        expected=expected,
                    )
                ] = (scope, deployment, pod)
        for probe in as_completed(probes):
            scope, deployment, pod = probes[probe]
            try:
                results[scope, deployment][pod] = probe.result()
            except Exception as exc:
                errors[f"{scope}/{deployment}/{pod}"] = f"{type(exc).__name__}: {exc}"
    if errors:
        details = "; ".join(
            f"{scope}: {error}" for scope, error in sorted(errors.items())
        )
        raise ReleaseError(f"runtime component identity validation failed: {details}")
    return {
        "control_plane": {
            "expected": control_digest,
            "deployments": {
                deployment: dict(sorted(results["control-plane", deployment].items()))
                for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
            },
        },
        "executor": {
            "expected": executor_digest,
            "clusters": {
                target.cluster_id: {
                    deployment: dict(
                        sorted(results[target.cluster_id, deployment].items())
                    )
                    for deployment in (
                        *inventory.DEPLOYMENTS,
                        inventory.GPU_RECONCILER_DEPLOYMENT,
                    )
                }
                for target in release.config.clusters
            },
        },
    }
