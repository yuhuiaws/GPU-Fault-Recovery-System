from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gpu_fault.admin.artifact_configmaps import (
    COMPRESSED_ARTIFACT_SUFFIX,
    artifact_binary_sha,
    compress_artifact,
)
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseDiff


def require_cpu_secrets(release: Any, *, include_registry: bool = True) -> None:
    names = [
        "gpu-fault-aurora",
        "gpu-fault-control-plane-active",
        "gpu-fault-node-action-keys",
    ]
    if include_registry:
        names.append("gpu-fault-regional-clusters")
    release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "secret",
            *names,
            "-o",
            "name",
        ),
        capture=True,
    )


def upload_config_map(
    release: Any,
    kubectl: list[str],
    name: str,
    key: str,
    path: Path,
    expected_sha: str,
    *,
    compress: bool = False,
) -> None:
    command = kubectl + [
        "-n",
        release.config.namespace,
        "get",
        "configmap",
        name,
        "-o",
        "json",
    ]
    returncode, stdout, stderr = release.runner.probe_output(command)
    value: dict[str, Any] | None = None
    if returncode == 0:
        value = json.loads(stdout)
    elif "NotFound" in stderr:
        with tempfile.TemporaryDirectory(prefix="gpu-fault-artifact-") as scratch:
            source, stored_key = path, key
            if compress:
                stored_key = key + COMPRESSED_ARTIFACT_SUFFIX
                source = compress_artifact(path, Path(scratch) / stored_key)
            release.runner.run(
                kubectl
                + [
                    "-n",
                    release.config.namespace,
                    "create",
                    "configmap",
                    name,
                    f"--from-file={stored_key}={source}",
                ]
            )
    else:
        raise ReleaseError(f"cannot inspect ConfigMap {name}: {stderr.strip()}")
    if release.runner.dry_run:
        return
    if value is None:
        value = release._get_json(
            kubectl
            + [
                "-n",
                release.config.namespace,
                "get",
                "configmap",
                name,
            ]
        )
    found = artifact_binary_sha(value.get("binaryData") or {}, key)
    if found is None:
        raise ReleaseError(f"{name}/{key} is missing")
    stored_key, digest = found
    if digest != expected_sha:
        raise ReleaseError(f"{name}/{stored_key} digest mismatch")


def upload_release(release: Any, diff: ReleaseDiff | None = None) -> None:
    if diff is None or diff.has("control_plane_wheel"):
        upload_config_map(
            release,
            release._cpu(),
            release.wheel_cm,
            release.config.wheel.name,
            release.config.wheel,
            release.wheel_sha,
            compress=True,
        )

    def upload_target(target: ClusterTarget) -> None:
        kubectl = release._gpu(target)
        if diff is None or diff.has("executor_wheel"):
            upload_config_map(
                release,
                kubectl,
                release.executor_wheel_cm,
                release.config.executor_wheel.name,
                release.config.executor_wheel,
                release.executor_wheel_sha,
                compress=True,
            )
        if diff is None or diff.has("node_runtime_wheel", "node_bundle"):
            upload_config_map(
                release,
                kubectl,
                release.bundle_cm,
                release.config.bundle.name,
                release.config.bundle,
                release.bundle_sha,
            )

    with ThreadPoolExecutor(
        max_workers=min(4, max(1, len(release.config.clusters)))
    ) as executor:
        futures = [
            executor.submit(upload_target, target) for target in release.config.clusters
        ]
        for future in futures:
            future.result()
