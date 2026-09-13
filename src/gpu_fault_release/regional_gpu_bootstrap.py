from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import repository_root
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_rendering import (
    DATAPLANE_ADOT_DEPLOYMENT,
    dataplane_adot_skip_reason,
    render_dataplane_adot_for_target,
    render_dcgm_exporter_manifest,
)
from gpu_fault_release.regional_release_rollout_wait import bounded_kubectl_wait

ROOT = repository_root()
ENDPOINT_CHECK_POLL_SECONDS = 5.0
INSTALLER_JOB_SELECTOR = "gpu-fault.io/node-installer=true"
TERMINAL_JOB_CONDITIONS = frozenset({"Complete", "Failed"})


def _render_gpu_dcgm_exporter(
    release: Any,
    target: ClusterTarget,
    *,
    image: str | None = None,
) -> tuple[str, str]:
    counters = ROOT / "deploy/dataplane/dcgm-counters.csv"
    rendered_config_map = release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "create",
            "configmap",
            "gpu-fault-dcgm-counters",
            f"--from-file=gpu-fault-counters.csv={counters}",
            "--dry-run=client",
            "-o",
            "yaml",
        ),
        capture=True,
    )
    # The same renderer the plan payload uses, so an approver reads the text
    # this function is about to apply -- including the instance-type list.
    manifest = render_dcgm_exporter_manifest(
        namespace=release.config.namespace,
        image=image or release.dcgm_exporter_image,
    )
    return rendered_config_map, manifest


def preflight_gpu_dcgm_exporter(
    release: Any,
    target: ClusterTarget,
    *,
    image: str | None = None,
) -> None:
    for manifest in _render_gpu_dcgm_exporter(release, target, image=image):
        release.runner.run(
            release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
            input_text=manifest,
        )


def apply_gpu_dcgm_exporter(
    release: Any,
    target: ClusterTarget,
    *,
    image: str | None = None,
) -> None:
    rendered_config_map, manifest = _render_gpu_dcgm_exporter(
        release,
        target,
        image=image,
    )
    for candidate in (rendered_config_map, manifest):
        release.runner.run(
            release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
            input_text=candidate,
        )
    release.runner.run(
        release._gpu(target, "apply", "-f", "-"),
        input_text=rendered_config_map,
    )
    release.runner.run(
        release._gpu(target, "apply", "-f", "-"),
        input_text=manifest,
    )
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "rollout",
            "status",
            "daemonset/gpu-fault-dcgm-exporter",
            "--timeout=10m",
        )
    )


def _render_gpu_adot_collector(
    release: Any,
    target: ClusterTarget,
    *,
    image: str | None = None,
) -> str | None:
    """The collector manifest for ``target``, or ``None`` after saying why not.

    The skip is printed rather than logged: this runs inside a release whose
    stdout the operator is reading, and the outcome of a silent skip is the
    inert watcher alerts the component exists to fix.
    """

    reason = dataplane_adot_skip_reason(release, target)
    if reason:
        print(
            f"{target.cluster_id}: data-plane ADOT collector not applied "
            f"({reason}); the Completion Watcher alerts stay inert for this "
            "cluster",
            file=sys.stderr,
            flush=True,
        )
        return None
    return render_dataplane_adot_for_target(release, target, image=image)


def preflight_gpu_adot_collector(
    release: Any,
    target: ClusterTarget,
    *,
    image: str | None = None,
) -> None:
    manifest = _render_gpu_adot_collector(release, target, image=image)
    if manifest is None:
        return
    release.runner.run(
        release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
        input_text=manifest,
    )


def apply_gpu_adot_collector(
    release: Any,
    target: ClusterTarget,
    *,
    image: str | None = None,
) -> None:
    """Apply the per-cluster metrics collector and wait for it to come up.

    Same shape as :func:`apply_gpu_dcgm_exporter`, and meant to be wired the
    same way: the release engine calls it wherever it calls the DCGM apply
    (bootstrap, join, the OBSERVABILITY node of an upgrade, and the
    previous-image rollback fallback).

    The skip branch is not read-only: removing a cluster's role (or the site's
    workspace) is a release -- the inputs are in the observability digest -- and
    the OBSERVABILITY node then lands here for that cluster. A collector the
    previous release applied would otherwise keep running with credentials that
    no longer exist (sigv4 refused, zero series, and the per-cluster absence
    rule gone with the role), so the skip scales it to zero with the same lever
    ``remove_cluster`` uses. The preflight's skip stays a dry run.
    """

    manifest = _render_gpu_adot_collector(release, target, image=image)
    if manifest is None:
        release._scale_if_present(release._gpu(target), DATAPLANE_ADOT_DEPLOYMENT, 0)
        return
    release.runner.run(
        release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
        input_text=manifest,
    )
    release.runner.run(
        release._gpu(target, "apply", "-f", "-"),
        input_text=manifest,
    )
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "rollout",
            "status",
            f"deployment/{DATAPLANE_ADOT_DEPLOYMENT}",
            "--timeout=5m",
        )
    )


def rollback_gpu_adot_collectors(release: Any, previous: dict[str, Any]) -> None:
    """Put every configured cluster's collector on the previous ``adot_image``.

    The FALLBACK compensation, for a previous-state snapshot captured before the
    per-cluster collector snapshot existed (``regional_dataplane_observability``
    decides, and records which path ran). It renders the CANDIDATE manifest with
    the previous image, so only the image is undone; a state that carries the
    snapshot never takes this path. Clusters the release skips (no IRSA role)
    print the same skip as the apply and have any leftover collector scaled to
    zero.
    """

    image = str(previous.get("adot_image") or "")
    if not image:
        raise ReleaseError("previous ADOT image is unavailable for the collectors")
    for target in release.config.clusters:
        release._apply_gpu_adot_collector(target, image=image)


def _installer_jobs(release: Any, target: ClusterTarget) -> list[dict[str, Any]]:
    return list(
        release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "jobs",
                "-l",
                INSTALLER_JOB_SELECTOR,
            )
        ).get("items", [])
    )


def _true_conditions(item: dict[str, Any]) -> set[str]:
    return {
        str(condition.get("type") or "")
        for condition in (item.get("status") or {}).get("conditions") or []
        if condition.get("status") == "True"
    }


def _installer_job_names(
    jobs: list[dict[str, Any]], *, matching: Callable[[set[str]], bool]
) -> list[str]:
    names = [
        str((item.get("metadata") or {}).get("name") or "")
        for item in jobs
        if matching(_true_conditions(item))
    ]
    return sorted(name for name in names if name)


def _active_installer_jobs(jobs: list[dict[str, Any]]) -> list[str]:
    return _installer_job_names(
        jobs, matching=lambda conditions: not conditions & TERMINAL_JOB_CONDITIONS
    )


def _failed_installer_jobs(jobs: list[dict[str, Any]]) -> list[str]:
    return _installer_job_names(
        jobs, matching=lambda conditions: "Failed" in conditions
    )


def _delete_installer_job(release: Any, target: ClusterTarget, name: str) -> None:
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "delete",
            "job",
            name,
            "--wait=true",
        )
    )


def _cancel_installer_jobs(
    release: Any, target: ClusterTarget, jobs: list[dict[str, Any]]
) -> None:
    """Delete every in-flight installer Job in ``jobs``, then prove none is left.

    The re-read is the fail-closed half of the cancellation and stays, but it
    only runs when something was actually deleted: a pass that deleted nothing
    has already proved the fleet is quiet with the listing it was handed, and
    re-reading it was a second identical `get jobs` on every wave of every
    rollout.
    """

    active = _active_installer_jobs(jobs)
    if not active:
        return
    for name in active:
        _delete_installer_job(release, target, name)
    remaining = _active_installer_jobs(_installer_jobs(release, target))
    if remaining:
        raise ReleaseError(
            f"{target.cluster_id} active installer Jobs remain after cancellation: "
            + ", ".join(remaining)
        )


def cancel_active_installer_jobs(release: Any, target: ClusterTarget) -> None:
    _cancel_installer_jobs(release, target, _installer_jobs(release, target))


def settle_installer_jobs(release: Any, target: ClusterTarget) -> None:
    """Clear the installer Jobs standing between this cluster and a new wave.

    Both things have to happen before the allowed-node set changes: an in-flight
    Job would install the previous wave's identity onto a node the new wave has
    not cleared, and a Job left behind as ``Failed`` would keep the Reconciler
    from creating the replacement this wave is waiting for.

    They used to be two functions that each listed the Jobs for themselves, and
    the cancelling one listed twice -- three identical `get jobs` calls against
    one label selector per wave, so twelve on a four-wave rollout, all of them
    inside the stretch of a release that prints nothing. One listing decides
    both, in the same order as before: cancel first and verify, then clear the
    failures.
    """

    jobs = _installer_jobs(release, target)
    _cancel_installer_jobs(release, target, jobs)
    for name in _failed_installer_jobs(jobs):
        _delete_installer_job(release, target, name)


def ensure_gpu_namespace(release: Any, target: ClusterTarget) -> None:
    rendered = release.runner.run(
        release._gpu(
            target,
            "create",
            "namespace",
            release.config.namespace,
            "--dry-run=client",
            "-o",
            "yaml",
        ),
        capture=True,
    )
    release.runner.run(
        release._gpu(target, "apply", "-f", "-"),
        input_text=rendered,
    )


def ensure_connection_secret(release: Any, target: ClusterTarget) -> None:
    if not all(
        (
            target.token_file,
            target.ca_file,
            target.control_plane_url,
            target.hyperpod_cluster_name,
        )
    ):
        release.runner.run(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "secret",
                "gpu-fault-regional-connection",
            ),
            capture=True,
        )
        return
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = {
            "cluster-token": (
                Path(target.token_file).read_text(encoding="utf-8").strip().encode()
            ),
            "ca.crt": Path(target.ca_file).read_bytes(),
            "control-plane-url": target.control_plane_url.encode(),
            "cluster-id": target.cluster_id.encode(),
            "allowed-namespaces": ",".join(target.allowed_namespaces).encode(),
            "hyperpod-cluster-name": (target.hyperpod_cluster_name or "").encode(),
            "hyperpod-confirm-cluster-name": (
                target.hyperpod_cluster_name or ""
            ).encode(),
        }
        arguments = release._gpu(
            target,
            "-n",
            release.config.namespace,
            "create",
            "secret",
            "generic",
            "gpu-fault-regional-connection",
        )
        for name, content in files.items():
            path = root / name
            path.write_bytes(content)
            path.chmod(0o600)
            arguments.append(f"--from-file={name}={path}")
        arguments.extend(["--dry-run=client", "-o", "yaml"])
        rendered = release.runner.run(arguments, capture=True, sensitive=True)
        release.runner.run(
            release._gpu(target, "apply", "-f", "-"),
            input_text=rendered,
            sensitive=True,
        )


def verify_gpu_control_plane_endpoint(release: Any, target: ClusterTarget) -> None:
    if release.runner.dry_run:
        return
    name = "gpu-fault-control-plane-endpoint-check"
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": release.config.namespace,
        },
        "spec": {
            "restartPolicy": "Never",
            # The probe waits up to 240 s for a brand-new endpoint (DNS, then
            # the NLB actually forwarding); the Pod outlives that wait.
            "activeDeadlineSeconds": 420,
            "tolerations": [
                {
                    "key": "gpu-fault.io/quarantined",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
                {
                    "key": "node.kubernetes.io/unschedulable",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
            ],
            "containers": [
                {
                    "name": "check",
                    "image": release.runtime_image,
                    "command": ["python", "-c", probe_source("gpu_endpoint_gate")],
                    "env": [
                        {
                            "name": "CONTROL_PLANE_URL",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "gpu-fault-regional-connection",
                                    "key": "control-plane-url",
                                }
                            },
                        },
                        {
                            "name": "EXPECTED_HOSTNAME",
                            "value": str(release.config.dns.hostname or ""),
                        },
                        {
                            "name": "PROBE_INCIDENT_ID",
                            # A synthetic id the control plane will not find, so
                            # the probe reads 200-with-unknown-owner and stays
                            # side effect free. Only the auth verdict matters.
                            "value": f"gpu-fault-endpoint-gate-{target.cluster_id}",
                        },
                    ],
                    "volumeMounts": [
                        {
                            "name": "tls",
                            "mountPath": "/tls",
                            "readOnly": True,
                        },
                        {
                            "name": "auth",
                            "mountPath": "/auth",
                            "readOnly": True,
                        },
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "tls",
                    "secret": {
                        "secretName": "gpu-fault-regional-connection",
                        "items": [{"key": "ca.crt", "path": "ca.crt"}],
                    },
                },
                {
                    "name": "auth",
                    # Mounted rather than passed through env so the token stays
                    # out of the Pod spec and out of anything that dumps env.
                    "secret": {
                        "secretName": "gpu-fault-regional-connection",
                        "items": [
                            {"key": "cluster-token", "path": "cluster-token"},
                            {"key": "cluster-id", "path": "cluster-id"},
                        ],
                    },
                },
            ],
        },
    }
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "delete",
            "pod",
            name,
            "--ignore-not-found",
            "--wait=true",
        )
    )
    release.runner.run(
        release._gpu(target, "apply", "-f", "-"),
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )
    deadline = time.monotonic() + 300
    try:
        while time.monotonic() < deadline:
            phase = release.runner.run(
                release._gpu(
                    target,
                    "-n",
                    release.config.namespace,
                    "get",
                    "pod",
                    name,
                    "-o",
                    "jsonpath={.status.phase}",
                ),
                capture=True,
            )
            if phase == "Succeeded":
                evidence = release.runner.run(
                    release._gpu(
                        target,
                        "-n",
                        release.config.namespace,
                        "logs",
                        name,
                    ),
                    capture=True,
                )
                print(
                    f"{target.cluster_id}: GPU DNS/TLS gate passed: {evidence}",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if phase == "Failed":
                logs = release.runner.run(
                    release._gpu(
                        target,
                        "-n",
                        release.config.namespace,
                        "logs",
                        name,
                    ),
                    capture=True,
                )
                raise ReleaseError(
                    f"{target.cluster_id}: GPU DNS/TLS check failed: {logs}"
                )
            # Wake the moment the probe succeeds instead of up to 5s later. A
            # Failed probe is not what this waits for, so it is still noticed
            # by the phase read above within the same interval as before.
            bounded_kubectl_wait(
                release,
                release._gpu(
                    target,
                    "-n",
                    release.config.namespace,
                    "wait",
                    f"pod/{name}",
                    "--for=jsonpath={.status.phase}=Succeeded",
                ),
                seconds=min(ENDPOINT_CHECK_POLL_SECONDS, deadline - time.monotonic()),
            )
        raise ReleaseError(f"{target.cluster_id}: GPU DNS/TLS check timed out")
    finally:
        release.runner.run(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "delete",
                "pod",
                name,
                "--ignore-not-found",
                "--wait=false",
            )
        )


def quiesce_gpu_executor(release: Any, target: ClusterTarget) -> None:
    exists = release.runner.probe(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.GPU_EXECUTOR_DEPLOYMENT,
        ),
    )
    if not exists:
        return
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "scale",
            f"deployment/{inventory.GPU_EXECUTOR_DEPLOYMENT}",
            "--replicas=0",
        )
    )
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "rollout",
            "status",
            f"deployment/{inventory.GPU_EXECUTOR_DEPLOYMENT}",
            "--timeout=5m",
        )
    )
