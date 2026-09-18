from __future__ import annotations

import json
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

from gpu_fault.admin.artifact_configmaps import (
    COMPRESSED_ARTIFACT_SUFFIX,
    artifact_binary_sha,
    compress_artifact,
)
from gpu_fault.admin.execution import PROOFS, ProofKey, ProofSubject
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseDiff


def require_cpu_secrets(
    release: RegionalRelease, *, include_registry: bool = True
) -> None:
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
    release: RegionalRelease,
    kubectl: list[str],
    name: str,
    key: str,
    path: Path,
    expected_sha: str,
    *,
    compress: bool = False,
) -> None:
    reference = ResourceRef("configmap", "ConfigMap", name, release.config.namespace)
    value = probe_resource(release.runner, kubectl, reference).require_readable()
    if value is None:
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
    raw_binary = value.get("binaryData") or {}
    if not isinstance(raw_binary, dict) or any(
        not isinstance(name, str) or not isinstance(encoded, str)
        for name, encoded in raw_binary.items()
    ):
        raise ReleaseError(f"{name} binaryData is invalid")
    binary_data = cast(dict[str, str], raw_binary)

    def verify() -> None:
        found = artifact_binary_sha(binary_data, key)
        if found is None:
            raise ReleaseError(f"{name}/{key} is missing")
        stored_key, digest = found
        if digest != expected_sha:
            raise ReleaseError(f"{name}/{stored_key} digest mismatch")

    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ReleaseError(f"{name} metadata is invalid")
    version = str(metadata.get("resourceVersion") or "")
    uid = str(metadata.get("uid") or "")
    if not version or not uid:
        verify()
        return
    PROOFS.verify(
        ProofKey(
            ProofSubject.ARTIFACT,
            json.dumps([kubectl, release.config.namespace, name, uid]),
            hashlib.sha256(
                json.dumps([key, expected_sha, binary_data], sort_keys=True).encode()
            ).hexdigest(),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            version,
        ),
        verify,
        max_age=60,
    )


def upload_release(release: RegionalRelease, diff: ReleaseDiff | None = None) -> None:
    if diff is None or diff.has(
        "control_plane_wheel",
        "aurora_refresh_manifests",
        "aurora_refresh_drift",
    ):
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
