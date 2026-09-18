"""Explicit legacy/shared and split image identities for release compensation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import re

from gpu_fault_release.regional_release_config import DIGEST_IMAGE_PATTERN, ReleaseError


class NodeDependencyTarget(StrEnum):
    """Select the owning release independently of bundle content reuse."""

    CANDIDATE = "candidate"
    PREVIOUS = "previous"

    @classmethod
    def for_phase(cls, phase: str) -> NodeDependencyTarget:
        if phase == "rollback":
            return cls.PREVIOUS
        if phase in {"bootstrap", "join", "upgrade"}:
            return cls.CANDIDATE
        raise ReleaseError("node dependency target requires a known rollout phase")


def require_digest_pinned_image(description: str, image: str | None) -> str:
    """Require an immutable image at every release, resume and rollback sink."""
    text = str(image or "").strip()
    if not DIGEST_IMAGE_PATTERN.fullmatch(text):
        raise ReleaseError(
            f"{description} image is not digest-pinned "
            f"(expected ...@sha256:<64 hex chars>): {text or '<empty>'}"
        )
    return text


def require_consistent_images(description: str, images: dict[str, str | None]) -> str:
    missing = sorted(name for name, image in images.items() if not image)
    if missing:
        raise ReleaseError(
            f"cannot capture previous {description} image from: " + ", ".join(missing)
        )
    distinct = {str(image) for image in images.values()}
    if len(distinct) != 1:
        raise ReleaseError(
            f"previous {description} images are inconsistent across: "
            + ", ".join(sorted(images))
        )
    return distinct.pop()


def capture_previous_image_identity(
    state: Mapping[str, object],
    *,
    cpu_images: dict[str, str | None],
    executor_images: dict[str, str | None],
    capture_gpu: bool,
) -> tuple[str, str, str]:
    live = require_consistent_images("runtime", cpu_images)
    version = previous_release_schema_version(state)
    if capture_gpu and executor_images:
        executor = require_consistent_images("Executor", executor_images)
    elif version >= 4:
        executor = previous_executor_image(state)
    else:
        executor = live
    runtime = live
    adopted = str(state.get("adopted_live_runtime_image") or "").strip()
    rollback = str(state.get("runtime_image") or "").strip()
    if adopted and version < 4:
        if live != adopted:
            raise ReleaseError(
                "live runtime image drifted after legacy release-state adoption"
            )
        if not DIGEST_IMAGE_PATTERN.fullmatch(rollback):
            raise ReleaseError(
                "legacy release-state adoption has no immutable rollback runtime image"
            )
        runtime = rollback
        if executor != adopted:
            raise ReleaseError("legacy Executor image drifted after adoption")
        executor = rollback
    elif version < 4 and executor != live:
        raise ReleaseError("legacy CPU and Executor runtime images disagree")
    elif version >= 4 and (
        runtime != state.get("runtime_image")
        or executor != previous_executor_image(state)
    ):
        raise ReleaseError("live split runtime images drifted before snapshot")
    return live, runtime, executor


def previous_executor_image(previous: Mapping[str, object]) -> str:
    version = previous_release_schema_version(previous)
    value = previous.get("executor_image")
    if version < 4:
        value = value or previous.get("runtime_image")
    if not isinstance(value, str) or not re.fullmatch(
        r"[^\s@]+@sha256:[0-9a-f]{64}", value
    ):
        raise ReleaseError("previous release has no immutable Executor image identity")
    return value


def node_dependency_environment(*, identity: object, required: bool) -> dict[str, str]:
    if (identity is None or identity == {}) and not required:
        return {
            "GPU_FAULT_NODE_DEPENDENCY_IMAGE": "",
            "GPU_FAULT_NODE_WHEELHOUSE_SHA256": "",
        }
    if not isinstance(identity, Mapping):
        raise ReleaseError("offline node dependency identity is incomplete")
    image = identity.get("reference")
    inventory = identity.get("wheelhouse_sha256")
    if (
        not isinstance(image, str)
        or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image)
        or not isinstance(inventory, str)
        or not re.fullmatch(r"[0-9a-f]{64}", inventory)
    ):
        raise ReleaseError("offline node dependency identity is incomplete")
    return {
        "GPU_FAULT_NODE_DEPENDENCY_IMAGE": image,
        "GPU_FAULT_NODE_WHEELHOUSE_SHA256": inventory,
    }


def previous_release_schema_version(previous: Mapping[str, object]) -> int:
    raw_version = previous.get("release_manifest_schema_version")
    try:
        version = int(str(3 if raw_version is None else raw_version))
    except ValueError as exc:
        raise ReleaseError(
            "previous release manifest schema version is invalid"
        ) from exc
    if version not in {1, 2, 3, 4}:
        raise ReleaseError("previous release manifest schema version is unsupported")
    return version


def previous_node_dependency_environment(
    previous: object,
    *,
    cluster_id: str,
    bundle_cm: str,
    bundle_sha256: str | None,
) -> dict[str, str]:
    """Bind rollback dependencies to the captured target cluster and bundle."""
    if not isinstance(previous, Mapping) or not previous:
        raise ReleaseError("rollback has no captured previous release identity")
    version = previous_release_schema_version(previous)
    clusters = previous.get("clusters")
    cluster = clusters.get(cluster_id) if isinstance(clusters, Mapping) else None
    if not isinstance(cluster, Mapping) or cluster.get("bundle") != bundle_cm:
        raise ReleaseError("rollback bundle has no captured node dependency identity")
    captured_sha = cluster.get("bundle_sha256")
    # Legacy snapshots may omit the hash; v4 must prove the complete binding.
    if (
        version >= 4
        and (
            not isinstance(captured_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", captured_sha)
            or bundle_sha256 != captured_sha
        )
        or captured_sha is not None
        and bundle_sha256 is not None
        and captured_sha != bundle_sha256
    ):
        raise ReleaseError(
            "rollback bundle disagrees with captured node dependency identity"
        )
    return node_dependency_environment(
        identity=previous.get("node_dependencies"), required=version >= 4
    )


def previous_node_template_environment(
    previous: object,
    *,
    cluster_id: str,
    template_config_map: str | None,
    template_sha256: str | None,
) -> dict[str, str]:
    """Select a captured content pin, never a hash of the mutable cluster copy."""
    if not isinstance(previous, Mapping):
        raise ReleaseError("rollback has no captured previous release identity")
    clusters = previous.get("clusters")
    cluster = clusters.get(cluster_id) if isinstance(clusters, Mapping) else None
    if not isinstance(cluster, Mapping):
        raise ReleaseError("rollback template has no captured cluster binding")
    captured_template = cluster.get("template")
    if (
        not isinstance(captured_template, str)
        or not captured_template
        or template_config_map is not None
        and captured_template != template_config_map
    ):
        raise ReleaseError("rollback template has no captured cluster binding")
    content = cluster.get("template_content_sha256")
    source = cluster.get("template_sha256")
    bundle = cluster.get("bundle_sha256")
    if not isinstance(content, str) or not re.fullmatch(r"[0-9a-f]{64}", content):
        raise ReleaseError("rollback template has no captured trusted content pin")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{64}", source):
        raise ReleaseError("rollback template has no captured source identity")
    if not isinstance(bundle, str) or not re.fullmatch(r"[0-9a-f]{64}", bundle):
        raise ReleaseError("rollback template has no captured bundle identity")
    if template_sha256 is not None and template_sha256 != source:
        raise ReleaseError("rollback template disagrees with captured source identity")
    return {
        "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP": captured_template,
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": source,
        "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256": content,
        "GPU_FAULT_INSTALLER_BUNDLE_SHA256": bundle,
    }


def node_dependency_pin(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ReleaseError("node dependency image identity must be an object")
    return node_dependency_environment(identity=value, required=True)


def previous_node_installer_image(
    *,
    capture_gpu: bool,
    images: dict[str, str | None],
    recorded: str,
    configured: str,
) -> str:
    """Keep the recorded installer identity when no GPU cluster remains."""

    if capture_gpu and images:
        return require_consistent_images("Node Installer", images)
    return recorded or configured
