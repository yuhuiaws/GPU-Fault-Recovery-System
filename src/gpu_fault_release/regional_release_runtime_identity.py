from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ReleaseError

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


def resolve_cpu_ingress_pod(release: Any, *, failure: str) -> str:
    pod = release.runner.run(_cpu_ingress_pod_command(release), capture=True)
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
    """Whether the CPU ingress Deployment exists at all.

    A first bootstrap runs its preflight before the control plane is
    installed: the namespace holds only the bootstrap's Secrets and Jobs, so
    "no Running ingress Pod" means "nothing installed", not "outage". Callers
    that need a Pod to answer a question use this to tell the two apart: with
    the Deployment absent the answer is decided by the absence; with it present
    but no Running Pod the resolvers above still raise.
    """

    returncode, _stdout, stderr = release.runner.probe_output(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.CPU_INGRESS_DEPLOYMENT,
            "-o",
            "name",
        )
    )
    if returncode == 0:
        return True
    if "NotFound" in str(stderr):
        return False
    raise ReleaseError(
        "could not read the CPU ingress Deployment: "
        + (str(stderr).strip() or f"kubectl exited {returncode}")
    )


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

    A Pod that no longer exists is not a failed attempt, when ``retry_replaced``
    allows it. The release rolls the CPU ingress Deployment itself, so a
    memoised name can be replaced between two probes -- the barrier that runs
    straight after CPU finalize hits exactly that. ``kubectl exec`` then reports
    ``NotFound`` without the command ever starting, which is why a vanished Pod
    always buys one more attempt against a freshly resolved Pod, even for the
    callers that set ``retries=0`` because their in-Pod barrier already spends
    its whole window: nothing was spent when the exec never ran. The check is
    positive -- the Pod is looked up rather than the failure text matched -- so
    it also holds for commands whose output is sensitive and must not be
    inspected.
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
            replaced = replacement_attempts_left > 0 and not release.runner.probe(
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
