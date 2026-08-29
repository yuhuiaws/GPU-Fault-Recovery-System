from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import copy
from typing import Any

import yaml


DIGEST_IMAGE_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_COMPONENT_DISTRIBUTIONS = {
    "control_plane": "gpu-fault-control-plane",
    "executor": "gpu-fault-cluster-executor",
}
IDENTITY_CONFIG = Path("config/release-identity.yaml")


class ReleaseIdentityError(ValueError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseIdentityError(f"{field} must be a mapping")
    return dict(value)


def _patterns(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ReleaseIdentityError(f"{field} must be a non-empty string list")
    return tuple(value)


def _component_patterns(value: object) -> dict[str, tuple[str, ...]]:
    raw = _mapping(value, "component_inputs")
    expected = {
        "collector",
        "cpu",
        "dcgm",
        "endpoint",
        "executor",
        "node",
        "observability",
        "schema",
        "watcher",
    }
    if set(raw) != expected:
        raise ReleaseIdentityError(
            "component_inputs must define exactly: " + ", ".join(sorted(expected))
        )
    return {
        name: _patterns(raw[name], f"component_inputs.{name}") for name in sorted(raw)
    }


def _expanded_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise ReleaseIdentityError(
                f"release identity input matched no files: {pattern}"
            )
        for path in matches:
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ReleaseIdentityError(
                    f"release identity input leaves repository: {path}"
                ) from exc
            found[relative] = resolved
    return [found[name] for name in sorted(found)]


def file_set_identity(
    root: Path,
    patterns: tuple[str, ...],
) -> dict[str, Any]:
    files = _expanded_files(root, patterns)
    entries: dict[str, dict[str, object]] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        entries[relative] = {
            "mode": f"{path.stat().st_mode & 0o777:04o}",
            "sha256": sha256_bytes(path.read_bytes()),
        }
    return {
        "sha256": canonical_sha256(entries),
        "file_count": len(entries),
        "files": entries,
    }


def load_image_lock(path: Path) -> dict[str, dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ReleaseIdentityError("release image lock schema_version must be 1")
    raw_images = _mapping(value.get("images"), "release image lock images")
    expected = {"runtime", "node_installer", "dcgm_exporter", "adot"}
    if set(raw_images) != expected:
        raise ReleaseIdentityError(
            "release image lock must define exactly: " + ", ".join(sorted(expected))
        )
    images: dict[str, dict[str, str]] = {}
    for name, raw in raw_images.items():
        item = _mapping(raw, f"release image lock {name}")
        reference = str(item.get("reference") or "")
        source = str(item.get("source") or "")
        if not DIGEST_IMAGE_PATTERN.fullmatch(reference):
            raise ReleaseIdentityError(
                f"release image {name} must use an immutable sha256 reference"
            )
        if not source:
            raise ReleaseIdentityError(f"release image {name} source is required")
        images[name] = {
            "reference": reference,
            "source": source,
            "digest": reference.rsplit("@sha256:", 1)[1],
        }
    return dict(sorted(images.items()))


def build_release_identity(
    root: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    path = (config_path or root / IDENTITY_CONFIG).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ReleaseIdentityError("release identity schema_version must be 1")
    dependency_lock = root / str(value.get("dependency_lock") or "")
    images_lock = root / str(value.get("images_lock") or "")
    if not dependency_lock.is_file():
        raise ReleaseIdentityError("release dependency lock is missing")
    if not images_lock.is_file():
        raise ReleaseIdentityError("release image lock is missing")
    identity = {
        "schema_version": 1,
        "runtime_prebuilt": False,
        "schema_rollback_compatible": bool(
            value.get("schema_rollback_compatible", False)
        ),
        "dependency_lock": {
            "path": dependency_lock.relative_to(root).as_posix(),
            "sha256": sha256_bytes(dependency_lock.read_bytes()),
        },
        "images": load_image_lock(images_lock),
        "manifest_inputs": file_set_identity(
            root,
            _patterns(value.get("manifest_inputs"), "manifest_inputs"),
        ),
        "renderer_inputs": file_set_identity(
            root,
            _patterns(value.get("renderer_inputs"), "renderer_inputs"),
        ),
        "rendered_manifests": file_set_identity(
            root,
            _patterns(
                value.get("rendered_manifest_inputs"),
                "rendered_manifest_inputs",
            ),
        ),
        "runtime_image_inputs": file_set_identity(
            root,
            _patterns(
                value.get("runtime_image_inputs"),
                "runtime_image_inputs",
            ),
        ),
        "node_template_inputs": file_set_identity(
            root,
            _patterns(value.get("node_template_inputs"), "node_template_inputs"),
        ),
        "components": {
            name: file_set_identity(root, patterns)
            for name, patterns in _component_patterns(
                value.get("component_inputs")
            ).items()
        },
    }
    identity["sha256"] = canonical_sha256(identity)
    return identity


def bind_runtime_image(
    root: Path,
    identity: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    if descriptor.get("schema_version") != 2:
        raise ReleaseIdentityError("runtime image descriptor schema_version must be 2")
    source_identity = str(descriptor.get("source_identity_sha256") or "")
    if source_identity != identity.get("sha256"):
        raise ReleaseIdentityError(
            "runtime image descriptor does not match the release source identity"
        )
    reference = str(descriptor.get("reference") or "")
    if not DIGEST_IMAGE_PATTERN.fullmatch(reference):
        raise ReleaseIdentityError(
            "runtime image descriptor must contain an immutable registry reference"
        )
    expected_files = {
        "dockerfile_sha256": root / "deploy/image/Dockerfile",
        "dependency_lock_sha256": root / "requirements/runtime.lock",
    }
    for field, path in expected_files.items():
        if descriptor.get(field) != sha256_bytes(path.read_bytes()):
            raise ReleaseIdentityError(
                f"runtime image descriptor {field} does not match the checkout"
            )
    image_inputs = descriptor.get("image_inputs")
    image_input_sha256 = str(descriptor.get("image_input_sha256") or "")
    if (
        not isinstance(image_inputs, dict)
        or image_inputs.get("schema_version") != 1
        or not SHA256_PATTERN.fullmatch(image_input_sha256)
        or canonical_sha256(image_inputs) != image_input_sha256
    ):
        raise ReleaseIdentityError(
            "runtime image descriptor image input identity is invalid"
        )
    if (
        image_inputs.get("dockerfile_sha256") != descriptor.get("dockerfile_sha256")
        or image_inputs.get("dependency_lock_sha256")
        != descriptor.get("dependency_lock_sha256")
        or image_inputs.get("platform") != descriptor.get("platform")
        or image_inputs.get("components") != descriptor.get("components")
    ):
        raise ReleaseIdentityError(
            "runtime image descriptor image inputs do not match its metadata"
        )
    base_images = image_inputs.get("base_images")
    if (
        not isinstance(base_images, list)
        or not base_images
        or any(
            not isinstance(item, str) or not DIGEST_IMAGE_PATTERN.fullmatch(item)
            for item in base_images
        )
    ):
        raise ReleaseIdentityError(
            "runtime image descriptor base images are not immutable"
        )
    if not isinstance(image_inputs.get("build_args"), dict):
        raise ReleaseIdentityError("runtime image descriptor build args are invalid")
    if descriptor.get("deployable") is not True:
        raise ReleaseIdentityError(
            "runtime image descriptor is local-only and cannot back a release"
        )
    raw_components = descriptor.get("components")
    if not isinstance(raw_components, dict) or set(raw_components) != set(
        RUNTIME_COMPONENT_DISTRIBUTIONS
    ):
        raise ReleaseIdentityError(
            "runtime image descriptor component identities are incomplete"
        )
    components: dict[str, dict[str, str]] = {}
    for name, distribution in RUNTIME_COMPONENT_DISTRIBUTIONS.items():
        value = raw_components.get(name)
        if not isinstance(value, dict) or value.get("distribution") != distribution:
            raise ReleaseIdentityError(
                f"runtime image descriptor {name} distribution is invalid"
            )
        wheel_sha256 = str(value.get("wheel_sha256") or "")
        module_digest = str(value.get("module_digest") or "")
        if not SHA256_PATTERN.fullmatch(wheel_sha256) or not SHA256_PATTERN.fullmatch(
            module_digest
        ):
            raise ReleaseIdentityError(
                f"runtime image descriptor {name} digests are invalid"
            )
        components[name] = {
            "distribution": distribution,
            "wheel_sha256": wheel_sha256,
            "module_digest": module_digest,
        }
    bound = copy.deepcopy(identity)
    bound.pop("sha256", None)
    digest = reference.rsplit("@sha256:", 1)[1]
    bound["images"]["runtime"] = {
        "reference": reference,
        "source": str(descriptor.get("repository") or ""),
        "digest": digest,
        "source_identity_sha256": source_identity,
        "image_input_sha256": image_input_sha256,
        "components": components,
    }
    bound["runtime_prebuilt"] = True
    bound["sha256"] = canonical_sha256(bound)
    return bound


def rendered_deployment_sha256(
    release_identity: dict[str, Any],
    deployment_inputs: dict[str, Any],
) -> str:
    expected = str(release_identity.get("sha256") or "")
    if not SHA256_PATTERN.fullmatch(expected):
        raise ReleaseIdentityError("release identity has no valid sha256")
    return canonical_sha256(
        {
            "release_identity_sha256": expected,
            "deployment_inputs": deployment_inputs,
        }
    )
