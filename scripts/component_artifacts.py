from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import sysconfig
import tarfile
from typing import Any

if __package__:
    from scripts.component_wheels import COMPONENTS, component_source_digest
    from scripts.release_identity import build_release_identity
else:
    from component_wheels import COMPONENTS, component_source_digest
    from release_identity import build_release_identity


class ComponentArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentArtifactSet:
    manifest: dict[str, Any]
    wheels: dict[str, Path]
    bundle: Path
    module_digests: dict[str, str]
    module_counts: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_build_identity(root: Path) -> str:
    root = root.resolve()
    release_identity = build_release_identity(root)
    payload = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "cache_tag": sys.implementation.cache_tag,
            "platform": sysconfig.get_platform(),
        },
        "build_inputs": {
            relative: _sha256(root / relative)
            for relative in (
                "LICENSE",
                "pyproject.toml",
                "requirements/build.lock",
                "scripts/component_artifacts.py",
                "scripts/component_wheels.py",
            )
        },
        "components": {
            name: component_source_digest(name) for name in sorted(COMPONENTS)
        },
        "node_bundle_inputs_sha256": release_identity["node_template_inputs"]["sha256"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact_path(root: Path, value: object, description: str) -> Path:
    path = Path(str(value or ""))
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ComponentArtifactError(f"{description} leaves repository") from exc
    if not resolved.is_file():
        raise ComponentArtifactError(f"{description} is missing: {resolved}")
    return resolved


def _verify_node_bundle(bundle: Path, expected_wheel_sha256: str) -> None:
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            wheels = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".whl")
            ]
            if len(wheels) != 1:
                raise ComponentArtifactError(
                    "component artifact node bundle must contain exactly one wheel"
                )
            source = archive.extractfile(wheels[0])
            if source is None:
                raise ComponentArtifactError(
                    "component artifact node bundle wheel is unreadable"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except tarfile.TarError as exc:
        raise ComponentArtifactError(
            "component artifact node bundle is invalid"
        ) from exc
    if digest.hexdigest() != expected_wheel_sha256:
        raise ComponentArtifactError(
            "component artifact node bundle does not contain the node runtime wheel"
        )


def load_component_artifacts(
    root: Path,
    manifest_path: Path,
    *,
    artifact_root: Path | None = None,
    require_source_only: bool = True,
    require_delivery_identity: bool = True,
) -> ComponentArtifactSet:
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    manifest_path = manifest_path.resolve()
    try:
        manifest_path.relative_to(artifact_root)
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ComponentArtifactError(
            f"component artifact manifest is invalid: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
        raise ComponentArtifactError("component artifact manifest must use schema v3")
    if require_source_only and manifest.get("deployable") is not False:
        raise ComponentArtifactError(
            "component artifact manifest must be a source-only release"
        )
    release_id = str(manifest.get("release_id") or "")
    immutable = artifact_root / "dist" / release_id / "release.json"
    if not release_id or not immutable.is_file() or immutable.read_bytes() != raw:
        raise ComponentArtifactError(
            "component artifact manifest differs from its content-addressed copy"
        )
    if manifest.get("component_build_identity_sha256") != component_build_identity(
        root
    ):
        raise ComponentArtifactError(
            "component artifacts do not match current component build inputs"
        )
    current_identity = build_release_identity(root)
    delivery = manifest.get("delivery")
    if (
        not isinstance(delivery, dict)
        or delivery.get("runtime_prebuilt") is not False
        or (
            require_delivery_identity
            and delivery.get("sha256") != current_identity.get("sha256")
        )
    ):
        raise ComponentArtifactError(
            "component artifacts do not match the current release source identity"
        )
    raw_components = manifest.get("components")
    if not isinstance(raw_components, dict):
        raise ComponentArtifactError("component artifact identities are missing")
    wheels: dict[str, Path] = {}
    module_digests: dict[str, str] = {}
    module_counts: dict[str, int] = {}
    for name in COMPONENTS:
        raw_component = raw_components.get(name)
        if not isinstance(raw_component, dict):
            raise ComponentArtifactError(
                f"component artifact identity is missing: {name}"
            )
        wheel = _artifact_path(
            artifact_root,
            raw_component.get("wheel"),
            f"{name} wheel",
        )
        expected_sha = str(raw_component.get("wheel_sha256") or "")
        module_digest = str(raw_component.get("module_digest") or "")
        if _sha256(wheel) != expected_sha:
            raise ComponentArtifactError(f"{name} wheel SHA-256 does not match")
        if module_digest != component_source_digest(name):
            raise ComponentArtifactError(
                f"{name} module digest does not match current source"
            )
        module_count = raw_component.get("module_count")
        if not isinstance(module_count, int) or module_count < 1:
            raise ComponentArtifactError(f"{name} module count is invalid")
        wheels[name] = wheel
        module_digests[name] = module_digest
        module_counts[name] = module_count
    bundle = _artifact_path(
        artifact_root,
        manifest.get("bundle"),
        "node installer bundle",
    )
    if _sha256(bundle) != manifest.get("bundle_sha256"):
        raise ComponentArtifactError("node installer bundle SHA-256 does not match")
    _verify_node_bundle(
        bundle,
        str(raw_components["node_runtime"]["wheel_sha256"]),
    )
    return ComponentArtifactSet(
        manifest=manifest,
        wheels=wheels,
        bundle=bundle,
        module_digests=module_digests,
        module_counts=module_counts,
    )
