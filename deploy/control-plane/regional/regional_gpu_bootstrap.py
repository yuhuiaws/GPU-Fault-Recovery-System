from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
import yaml
from regional_release_config import ClusterTarget, ReleaseError
from regional_release_probes import probe_source
from regional_release_rendering import DEFAULT_DCGM_EXPORTER_IMAGE

ROOT = Path(__file__).resolve().parents[3]


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
    manifest = (ROOT / "deploy/dataplane/hyperpod-dcgm-exporter.yaml").read_text(
        encoding="utf-8"
    )
    manifest = manifest.replace(
        "namespace: gpu-fault-system",
        f"namespace: {release.config.namespace}",
    ).replace(
        DEFAULT_DCGM_EXPORTER_IMAGE,
        image or release.dcgm_exporter_image,
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


def retry_failed_installer_jobs(release: Any, target: ClusterTarget) -> None:
    jobs = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "jobs",
            "-l",
            "gpu-fault.io/node-installer=true",
        )
    ).get("items", [])
    for item in jobs:
        conditions = (item.get("status") or {}).get("conditions") or []
        failed = any(
            condition.get("type") == "Failed" and condition.get("status") == "True"
            for condition in conditions
        )
        if not failed:
            continue
        name = str((item.get("metadata") or {}).get("name") or "")
        if not name:
            continue
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


def cancel_active_installer_jobs(release: Any, target: ClusterTarget) -> None:
    command = release._gpu(
        target,
        "-n",
        release.config.namespace,
        "get",
        "jobs",
        "-l",
        "gpu-fault.io/node-installer=true",
    )
    jobs = release._get_json(command).get("items", [])
    for item in jobs:
        conditions = (item.get("status") or {}).get("conditions") or []
        terminal = any(
            condition.get("type") in {"Complete", "Failed"}
            and condition.get("status") == "True"
            for condition in conditions
        )
        if terminal:
            continue
        name = str((item.get("metadata") or {}).get("name") or "")
        if not name:
            continue
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
    remaining = []
    for item in release._get_json(command).get("items", []):
        conditions = (item.get("status") or {}).get("conditions") or []
        terminal = any(
            condition.get("type") in {"Complete", "Failed"}
            and condition.get("status") == "True"
            for condition in conditions
        )
        if not terminal:
            name = str((item.get("metadata") or {}).get("name") or "")
            if name:
                remaining.append(name)
    if remaining:
        raise ReleaseError(
            f"{target.cluster_id} active installer Jobs remain after cancellation: "
            + ", ".join(sorted(remaining))
        )


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
            "activeDeadlineSeconds": 300,
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
            time.sleep(5)
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
