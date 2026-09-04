from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import regional_deployment_inventory as inventory
from regional_release_config import ReleaseError

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
MAX_RUNTIME_IDENTITY_WORKERS = 8
CPU_INGRESS_POD_ATTRIBUTE = "_cpu_ingress_pod"


def resolve_cpu_ingress_pod(release: Any, *, failure: str) -> str:
    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError(f"no running CPU ingress Pod for {failure}")
    return str(pod)


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


def forget_cpu_ingress_pod(release: Any) -> None:
    setattr(release, CPU_INGRESS_POD_ATTRIBUTE, "")


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

    Only read-only probes may use this: with ``retries`` above zero a failure
    re-resolves the Pod and runs the script again, which would be unsafe for a
    mutating command. A failed attempt always drops the memoised name so the
    next caller re-resolves it.

    A Pod that no longer exists is not a failed attempt. The release rolls the
    CPU ingress Deployment itself, so a memoised name can be replaced between
    two probes -- the barrier that runs straight after CPU finalize hits exactly
    that. ``kubectl exec`` then reports ``NotFound`` without the script ever
    starting, which is why a vanished Pod always buys one more attempt against a
    freshly resolved Pod, even for the callers that set ``retries=0`` because
    their in-Pod barrier already spends its whole window: nothing was spent when
    the exec never ran. The check is positive -- the Pod is looked up rather
    than the failure text matched -- so it also holds for probes whose output is
    sensitive and must not be inspected.
    """

    error: Exception | None = None
    attempts_left = retries + 1
    replacement_attempts_left = 1
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
            CONTROL_PLANE_PYTHON,
            "-c",
            script,
        )
        try:
            return release.runner.run(
                command,
                input_text=input_text,
                capture=True,
                sensitive=sensitive,
                timeout_seconds=timeout_seconds,
            )
        except ReleaseError as exc:
            error = exc
            replaced = not release.runner.probe(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "get",
                    "pod",
                    pod,
                )
            )
            forget_cpu_ingress_pod(release)
            refresh = True
            attempts_left -= 1
            if attempts_left == 0 and replaced and replacement_attempts_left > 0:
                replacement_attempts_left -= 1
                attempts_left = 1
    assert error is not None
    raise error


def _running_pods(
    release: Any,
    kubectl: list[str],
    deployment: str,
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
    if len(names) != replicas:
        raise ReleaseError(
            f"{deployment} runtime identity has {len(names)}/{replicas} Running Pods"
        )
    return names


def _deployment_digests(
    release: Any,
    kubectl: list[str],
    deployment: str,
    *,
    python: str,
    path: str,
    expected: str,
) -> dict[str, str]:
    pods = _running_pods(release, kubectl, deployment)

    def read_digest(pod: str) -> tuple[str, str]:
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
        return pod, digest

    digests: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(
        max_workers=min(MAX_RUNTIME_IDENTITY_WORKERS, max(1, len(pods)))
    ) as executor:
        futures = {executor.submit(read_digest, pod): pod for pod in pods}
        for future in as_completed(futures):
            pod = futures[future]
            try:
                name, digest = future.result()
            except Exception as exc:
                errors[pod] = f"{type(exc).__name__}: {exc}"
            else:
                digests[name] = digest
    if errors:
        details = "; ".join(
            f"{deployment}/{pod}: {error}" for pod, error in sorted(errors.items())
        )
        raise ReleaseError(f"runtime component identity failed: {details}")
    return dict(sorted(digests.items()))


def validate_runtime_component_identity(release: Any) -> dict[str, Any]:
    control_digest = str(release.config.component_digests.get("control_plane") or "")
    executor_digest = str(release.config.component_digests.get("executor") or "")
    if len(control_digest) != 64 or len(executor_digest) != 64:
        raise ReleaseError("release component module digests are unavailable")

    errors: dict[str, str] = {}
    cpu = {}
    for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
        try:
            cpu[deployment] = _deployment_digests(
                release,
                release._cpu(),
                deployment,
                python=CONTROL_PLANE_PYTHON,
                path=CONTROL_PLANE_PATH,
                expected=control_digest,
            )
        except Exception as exc:
            errors[f"control-plane/{deployment}"] = f"{type(exc).__name__}: {exc}"
    gpu = {}
    for target in release.config.clusters:
        deployments = {}
        for deployment in (
            *inventory.DEPLOYMENTS,
            inventory.GPU_RECONCILER_DEPLOYMENT,
        ):
            try:
                deployments[deployment] = _deployment_digests(
                    release,
                    release._gpu(target),
                    deployment,
                    python=EXECUTOR_PYTHON,
                    path=EXECUTOR_PATH,
                    expected=executor_digest,
                )
            except Exception as exc:
                errors[f"executor/{target.cluster_id}/{deployment}"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        gpu[target.cluster_id] = deployments
    if errors:
        details = "; ".join(
            f"{scope}: {error}" for scope, error in sorted(errors.items())
        )
        raise ReleaseError(f"runtime component identity validation failed: {details}")
    return {
        "control_plane": {"expected": control_digest, "deployments": cpu},
        "executor": {"expected": executor_digest, "clusters": gpu},
    }
