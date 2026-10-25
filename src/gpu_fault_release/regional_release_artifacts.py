from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

from gpu_fault.admin.artifact_configmaps import (
    COMPRESSED_ARTIFACT_SUFFIX,
    CONFIG_MAP_PAYLOAD_LIMIT,
    ArtifactLayoutError,
    ArtifactObject,
    ArtifactPlan,
    ArtifactTooLarge,
    artifact_binary_sha,
    artifact_manifest,
    plan_artifact_config_maps,
    read_artifact_digest,
    write_artifact_object,
)
from gpu_fault.admin.execution import PROOFS, ProofKey, ProofSubject
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_resource_probe import ResourceRef, probe_resource
from gpu_fault_release.regional_release_config import (
    CheckFailure,
    ClusterTarget,
    ReleaseError,
)
from gpu_fault_release.regional_release_diff import ReleaseDiff

__all__ = [
    "COMPRESSED_ARTIFACT_SUFFIX",
    "ArtifactTransportReport",
    "ReleaseArtifact",
    "artifact_binary_sha",
    "artifact_transport_report",
    "check_artifact_config_maps",
    "config_map_artifact_sha",
    "release_artifacts",
    "require_cpu_secrets",
    "upload_config_map",
    "upload_release",
]

#: An artifact a Pod reads from its mount is refused, not split, past the
#: limit; this close to it the operator should hear about it first.
MOUNT_ARTIFACT_WARN_RATIO = 0.9


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


@dataclass(frozen=True)
class ReleaseArtifact:
    """One artifact the release ships through a ConfigMap set."""

    component: str
    label: str
    name: str
    path: Path
    sha256: str
    #: Store the xz stream (wheels: nothing reads them from a mount).
    compress: bool
    #: May become a chunked set. False for the node installer bundle, which the
    #: installer Job reads straight from its mount.
    splittable: bool

    @property
    def key(self) -> str:
        return self.path.name

    def plan(self) -> ArtifactPlan:
        return plan_artifact_config_maps(
            self.path,
            name=self.name,
            key=self.key,
            compress=self.compress,
            splittable=self.splittable,
            label=self.label,
        )


def release_artifacts(release: RegionalRelease) -> tuple[ReleaseArtifact, ...]:
    config = release.config
    return (
        ReleaseArtifact(
            "control_plane_wheel",
            "control-plane wheel",
            release.wheel_cm,
            config.wheel,
            release.wheel_sha,
            compress=True,
            splittable=True,
        ),
        ReleaseArtifact(
            "executor_wheel",
            "Executor wheel",
            release.executor_wheel_cm,
            config.executor_wheel,
            release.executor_wheel_sha,
            compress=True,
            splittable=True,
        ),
        ReleaseArtifact(
            "node_bundle",
            "node installer bundle",
            release.bundle_cm,
            config.bundle,
            release.bundle_sha,
            compress=False,
            splittable=False,
        ),
    )


@dataclass(frozen=True)
class ArtifactTransportReport:
    status: str
    summary: str
    details: dict[str, Any]


def artifact_transport_report(release: RegionalRelease) -> ArtifactTransportReport:
    """Whether every artifact of this release fits its ConfigMap set.

    Computed from the local files alone, before any object is created: this is
    the preflight the 2026-10-25 release lacked, when every gate had passed and
    the first ``kubectl create configmap`` of the deploy died on the 1 MiB
    ceiling. A wheel over the limit is fine (it ships as a chunked set); the
    bundle is not, because the installer Job reads it from its mount.
    """

    details: dict[str, Any] = {"config_map_limit_bytes": CONFIG_MAP_PAYLOAD_LIMIT}
    failures: list[str] = []
    warnings: list[str] = []
    shipped: list[str] = []
    for artifact in release_artifacts(release):
        try:
            plan = artifact.plan()
        except ArtifactTooLarge as exc:
            failures.append(str(exc))
            details[artifact.component] = {
                "config_map": artifact.name,
                "artifact": artifact.key,
                "splittable": artifact.splittable,
                "error": str(exc),
            }
            continue
        details[artifact.component] = {
            "config_map": artifact.name,
            "artifact": artifact.key,
            "bytes": plan.size,
            "payload_bytes": plan.payload_size,
            "encoding": plan.encoding,
            "layout": plan.layout,
            "objects": [item.name for item in plan.objects],
            "limit_ratio": round(plan.limit_ratio, 3),
            "splittable": artifact.splittable,
        }
        shipped.append(
            f"{artifact.label} {plan.payload_size:,} B -> "
            f"{len(plan.objects)} object{'s' if plan.chunked else ''}"
        )
        if not artifact.splittable and plan.limit_ratio >= MOUNT_ARTIFACT_WARN_RATIO:
            warnings.append(
                f"{artifact.label} {artifact.key} uses {plan.limit_ratio:.0%} of the "
                f"{CONFIG_MAP_PAYLOAD_LIMIT:,}-byte ConfigMap limit and cannot be "
                "split; shrink it before it blocks a release"
            )
    if failures:
        return ArtifactTransportReport("FAIL", "; ".join(failures), details)
    if warnings:
        return ArtifactTransportReport("WARN", "; ".join(warnings), details)
    return ArtifactTransportReport(
        "PASS",
        "release artifacts fit their ConfigMap sets: " + "; ".join(shipped),
        details,
    )


def check_artifact_config_maps(release: RegionalRelease) -> CheckValue:
    """The deploy preflight's ``artifact_config_maps`` check.

    Decided from the local wheels and bundle before anything is created, so a
    release that cannot ship stops in the preflight report instead of in the
    deploy's first ``kubectl create configmap``. FAIL carries the per-artifact
    details with it; PASS and WARN name the layout each artifact will take.
    """

    report = artifact_transport_report(release)
    if report.status == "FAIL":
        raise CheckFailure(report.summary, report.details)
    return CheckValue(report.summary, report.details, status=report.status)


def _config_map_fetcher(
    release: RegionalRelease, kubectl: list[str]
) -> Callable[[str], dict[str, Any]]:
    def fetch(part: str) -> dict[str, Any]:
        return release._get_json(
            kubectl + ["-n", release.config.namespace, "get", "configmap", part]
        )

    return fetch


def _artifact_digest(
    release: RegionalRelease,
    kubectl: list[str],
    name: str,
    key: str,
    document: dict[str, Any],
) -> tuple[str, str] | None:
    try:
        return read_artifact_digest(
            document, key, _config_map_fetcher(release, kubectl), name=name
        )
    except ArtifactLayoutError as exc:
        raise ReleaseError(f"{name}: {exc}") from None


def config_map_artifact_sha(
    release: RegionalRelease, kubectl: list[str], name: str, key: str
) -> str:
    """The sha256 of the artifact ConfigMap ``name`` stores, in any layout.

    What a rollback pins the previous control plane and Executor to
    (``GPU_FAULT_WHEEL_SHA256``, the Executor artifact pin): a set written
    before chunking existed reads exactly as it did, and a chunked set is
    reassembled and re-hashed part by part.
    """

    found = _artifact_digest(
        release, kubectl, name, key, _config_map_fetcher(release, kubectl)(name)
    )
    if found is None:
        raise ReleaseError(f"{name} has no binaryData")
    return found[1]


def _create_object(
    release: RegionalRelease,
    kubectl: list[str],
    item: ArtifactObject,
    directory: Path,
) -> None:
    source = write_artifact_object(item, directory)
    release.runner.run(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "create",
            "configmap",
            item.name,
            f"--from-file={item.key}={source}",
        ]
    )


def _create_parts(
    release: RegionalRelease,
    kubectl: list[str],
    plan: ArtifactPlan,
    directory: Path,
) -> None:
    """Create the parts a chunked set is missing, before its manifest.

    A part that already exists (an upload interrupted after it was created)
    is kept when it holds exactly the planned slice and refused otherwise: a
    content-addressed name must never change meaning, and a slice boundary
    that moved between engine versions is an operator decision, not a silent
    overwrite.
    """

    for part in plan.parts:
        reference = ResourceRef(
            "configmap", "ConfigMap", part.name, release.config.namespace
        )
        existing = probe_resource(release.runner, kubectl, reference).require_readable()
        if existing is None:
            _create_object(release, kubectl, part, directory)
            continue
        binary = existing.get("binaryData")
        encoded = binary.get(part.key) if isinstance(binary, dict) else None
        if encoded != base64.b64encode(part.payload).decode():
            raise ReleaseError(
                f"{part.name} exists with content other than {plan.name}'s "
                f"{part.key}; delete the stale part set before releasing"
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
    splittable: bool = False,
    label: str = "artifact",
) -> None:
    """Make ConfigMap ``name`` hold ``path`` and prove it holds ``expected_sha``.

    The layout is decided from the local file first (one object or a chunked
    set; see ``gpu_fault.admin.artifact_configmaps``), so an artifact that
    cannot ship fails here with its numbers before any object exists. An
    existing ConfigMap is never rewritten: whichever layout it has, its content
    is read back, re-hashed and compared to the artifact digest.
    """

    try:
        plan = plan_artifact_config_maps(
            path,
            name=name,
            key=key,
            compress=compress,
            splittable=splittable,
            label=label,
        )
    except (ArtifactTooLarge, ArtifactLayoutError) as exc:
        raise ReleaseError(str(exc)) from None
    if plan.sha256 != expected_sha:
        raise ReleaseError(
            f"{name}/{key} digest mismatch: {path} hashes to {plan.sha256}, the "
            f"release manifest records {expected_sha}"
        )
    reference = ResourceRef("configmap", "ConfigMap", name, release.config.namespace)
    value = probe_resource(release.runner, kubectl, reference).require_readable()
    if value is None:
        with tempfile.TemporaryDirectory(prefix="gpu-fault-artifact-") as scratch:
            if plan.chunked:
                _create_parts(release, kubectl, plan, Path(scratch))
            _create_object(release, kubectl, plan.primary, Path(scratch))
    if release.runner.dry_run:
        return
    if value is None:
        value = _config_map_fetcher(release, kubectl)(name)
    raw_binary = value.get("binaryData") or {}
    if not isinstance(raw_binary, dict) or any(
        not isinstance(name, str) or not isinstance(encoded, str)
        for name, encoded in raw_binary.items()
    ):
        raise ReleaseError(f"{name} binaryData is invalid")
    binary_data = cast(dict[str, str], raw_binary)
    raw_data = value.get("data") or {}
    if not isinstance(raw_data, dict):
        raise ReleaseError(f"{name} data is invalid")
    document: dict[str, Any] = {"data": raw_data, "binaryData": binary_data}

    def verify() -> None:
        found = _artifact_digest(release, kubectl, name, key, document)
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
    # A chunked set's bytes live in objects the primary's resourceVersion says
    # nothing about, so a proof keyed on the primary cannot vouch for them:
    # every look re-reads and re-hashes the parts.
    if not version or not uid or artifact_manifest(document) is not None:
        verify()
        return
    PROOFS.verify(
        ProofKey(
            ProofSubject.ARTIFACT,
            json.dumps([kubectl, release.config.namespace, name, uid]),
            hashlib.sha256(
                json.dumps([key, expected_sha, document], sort_keys=True).encode()
            ).hexdigest(),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            version,
        ),
        verify,
        max_age=60,
    )


def _upload(
    release: RegionalRelease, kubectl: list[str], artifact: ReleaseArtifact
) -> None:
    upload_config_map(
        release,
        kubectl,
        artifact.name,
        artifact.key,
        artifact.path,
        artifact.sha256,
        compress=artifact.compress,
        splittable=artifact.splittable,
        label=artifact.label,
    )


def upload_release(release: RegionalRelease, diff: ReleaseDiff | None = None) -> None:
    artifacts = {item.component: item for item in release_artifacts(release)}
    if diff is None or diff.has(
        "control_plane_wheel",
        "aurora_refresh_manifests",
        "aurora_refresh_drift",
    ):
        _upload(release, release._cpu(), artifacts["control_plane_wheel"])

    def upload_target(target: ClusterTarget) -> None:
        kubectl = release._gpu(target)
        if diff is None or diff.has("executor_wheel"):
            _upload(release, kubectl, artifacts["executor_wheel"])
        if diff is None or diff.has("node_runtime_wheel", "node_bundle"):
            _upload(release, kubectl, artifacts["node_bundle"])

    with ThreadPoolExecutor(
        max_workers=min(4, max(1, len(release.config.clusters)))
    ) as executor:
        futures = [
            executor.submit(upload_target, target) for target in release.config.clusters
        ]
        for future in futures:
            future.result()
