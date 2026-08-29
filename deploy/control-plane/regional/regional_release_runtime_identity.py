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
