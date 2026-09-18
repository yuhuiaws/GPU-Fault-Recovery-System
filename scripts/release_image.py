from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from scripts.component_artifacts import load_component_artifacts
    from scripts.component_wheels import COMPONENTS, build_component
    from scripts.release_identity import build_release_identity
else:
    from component_artifacts import load_component_artifacts
    from component_wheels import COMPONENTS, build_component
    from release_identity import build_release_identity


DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
DIGEST_IMAGE_PATTERN = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
BUILD_ARG_PATTERN = re.compile(r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?\s*$")
FROM_PATTERN = re.compile(
    r"^\s*FROM(?:\s+--platform=\S+)?\s+(\S+)",
    re.IGNORECASE,
)
ARG_REFERENCE_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)
MISSING_IMAGE_MESSAGES = (
    "manifest unknown",
    "manifest not found",
    "name unknown",
    "no such manifest",
    "manifest unknown: manifest unknown",
    "manifest unknown: requested image not found",
    "name unknown: repository name not known to registry",
)
IMAGE_INPUT_SCHEMA_VERSION = 1
RUNTIME_COMPONENTS = ("control_plane", "executor")


class ReleaseImageError(RuntimeError):
    pass


def normalize_repository(repository: str) -> str:
    value = repository.strip().rstrip(":")
    if (
        not value
        or any(character.isspace() for character in value)
        or "@" in value
        or ":" in value.rsplit("/", 1)[-1]
    ):
        raise ReleaseImageError("runtime image repository is invalid")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalize_platform(platform: str) -> str:
    value = platform.strip().lower()
    parts = value.split("/")
    if len(parts) not in {2, 3} or any(
        not part or not re.fullmatch(r"[a-z0-9_.-]+", part) for part in parts
    ):
        raise ReleaseImageError("runtime image platform must be os/architecture")
    return value


def _effective_build_args(
    dockerfile: str,
    overrides: Mapping[str, str],
) -> dict[str, str | None]:
    declared: dict[str, str | None] = {}
    for line in dockerfile.splitlines():
        match = BUILD_ARG_PATTERN.fullmatch(line)
        if match:
            declared[match.group(1)] = (
                match.group(2).strip() if match.group(2) is not None else None
            )
    unknown = sorted(set(overrides) - set(declared))
    if unknown:
        raise ReleaseImageError(
            "runtime image build args are not declared by the Dockerfile: "
            + ", ".join(unknown)
        )
    return {
        name: str(overrides[name]) if name in overrides else default
        for name, default in sorted(declared.items())
    }


def _expand_build_args(
    value: str,
    build_args: Mapping[str, str | None],
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        resolved = build_args.get(name)
        if resolved is None:
            raise ReleaseImageError(
                f"runtime image base reference uses unset build arg {name}"
            )
        return resolved

    return ARG_REFERENCE_PATTERN.sub(replace, value)


def _base_images(
    dockerfile: str,
    build_args: Mapping[str, str | None],
) -> list[str]:
    images = []
    for line in dockerfile.splitlines():
        match = FROM_PATTERN.match(line)
        if not match:
            continue
        reference = _expand_build_args(match.group(1), build_args)
        if not DIGEST_IMAGE_PATTERN.fullmatch(reference):
            raise ReleaseImageError(
                "runtime image base references must use immutable sha256 digests"
            )
        images.append(reference)
    if not images:
        raise ReleaseImageError("runtime image Dockerfile has no base image")
    return images


def runtime_image_inputs(
    root: Path,
    *,
    platform: str,
    build_args: Mapping[str, str],
    components: Mapping[str, Mapping[str, str]],
    dockerfile_name: str = "Dockerfile",
) -> dict[str, Any]:
    dockerfile_path = root / "deploy/image" / dockerfile_name
    dependency_lock_path = root / "requirements/runtime.lock"
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    effective_args = _effective_build_args(dockerfile, build_args)
    return {
        "schema_version": IMAGE_INPUT_SCHEMA_VERSION,
        "platform": platform,
        "build_args": effective_args,
        "base_images": _base_images(dockerfile, effective_args),
        "dockerfile_sha256": _sha256(dockerfile_path),
        "dependency_lock_sha256": _sha256(dependency_lock_path),
        "components": {
            name: dict(sorted(component.items()))
            for name, component in sorted(components.items())
        },
    }


def _image_labels(
    image_inputs: Mapping[str, Any],
    image_input_sha256: str,
) -> dict[str, str]:
    labels = {
        "gpu-fault.image-input.schema-version": str(image_inputs["schema_version"]),
        "gpu-fault.image-input.sha256": image_input_sha256,
        "gpu-fault.runtime.platform": str(image_inputs["platform"]),
        "gpu-fault.runtime.dockerfile-sha256": str(image_inputs["dockerfile_sha256"]),
        "gpu-fault.runtime.dependency-lock-sha256": str(
            image_inputs["dependency_lock_sha256"]
        ),
        "gpu-fault.runtime.base-images-sha256": canonical_sha256(
            image_inputs["base_images"]
        ),
        "gpu-fault.runtime.build-args-sha256": canonical_sha256(
            image_inputs["build_args"]
        ),
    }
    for name, component in sorted(image_inputs["components"].items()):
        labels[f"gpu-fault.component.{name}.wheel-sha256"] = str(
            component["wheel_sha256"]
        )
        labels[f"gpu-fault.component.{name}.module-digest"] = str(
            component["module_digest"]
        )
    return labels


def _missing_registry_image(stderr: str, *, tag: str) -> bool:
    message = stderr.strip().removeprefix("ERROR: ")
    try:
        value = json.loads(message)
    except json.JSONDecodeError:
        if message in {f"{tag}: not found", f"no such manifest: {tag}"}:
            return True
        return message.removeprefix(f"{tag}: ").lower() in MISSING_IMAGE_MESSAGES
    if not isinstance(value, dict):
        return False
    errors = value.get("errors")
    return (
        isinstance(errors, list)
        and bool(errors)
        and all(
            isinstance(error, dict)
            and error.get("code") in ("MANIFEST_UNKNOWN", "NAME_UNKNOWN")
            for error in errors
        )
    )


def _registry_failure_hint(stderr: str) -> str:
    lowered = stderr.lower()
    if "credential" in lowered:
        return "check Docker credential helper installation and configuration"
    if any(
        marker in lowered
        for marker in ("unauthorized", "denied", "authentication", "insufficient_scope")
    ):
        return "check registry login and repository read permissions"
    if any(marker in lowered for marker in ("x509", "certificate", "tls")):
        return "check registry TLS trust and certificates"
    if any(
        marker in lowered
        for marker in ("timeout", "timed out", "connection", "no such host")
    ):
        return "check registry connectivity and request time limits"
    return "check Docker Buildx, registry authentication and connectivity"


def _platform_image(
    image: object,
    platform: str,
) -> dict[str, Any]:
    if not isinstance(image, dict):
        raise ReleaseImageError("registry image inspection has no image config")
    selected = image.get(platform)
    if selected is None and {"os", "architecture"} <= set(image):
        selected = image
    if not isinstance(selected, dict):
        raise ReleaseImageError(
            f"registry image does not contain required platform {platform}"
        )
    expected = platform.split("/")
    actual = [str(selected.get("os") or ""), str(selected.get("architecture") or "")]
    if len(expected) == 3:
        actual.append(str(selected.get("variant") or ""))
    if actual != expected:
        raise ReleaseImageError(
            f"registry image platform mismatch: expected {platform}"
        )
    return selected


def inspect_registry_image(
    root: Path,
    *,
    tag: str,
    platform: str,
    expected_labels: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str | None:
    command = [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        "--format",
        "{{json .}}",
        tag,
    ]
    try:
        completed = runner(
            command,
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, TimeoutError):
        raise ReleaseImageError(
            "runtime registry inspection timed out; "
            "check registry connectivity and request time limits"
        ) from None
    except OSError:
        raise ReleaseImageError(
            "runtime registry inspection could not execute Docker; "
            "check Docker Buildx installation and executable permissions"
        ) from None
    except UnicodeError:
        raise ReleaseImageError(
            "runtime registry inspection returned invalid text; check Docker Buildx"
        ) from None
    if completed.returncode:
        if completed.returncode == 1 and _missing_registry_image(
            completed.stderr or "", tag=tag
        ):
            return None
        raise ReleaseImageError(
            f"runtime registry inspection failed with status {completed.returncode}; "
            f"{_registry_failure_hint(completed.stderr or '')} "
            "(registry tool output withheld)"
        )
    try:
        value = json.loads(completed.stdout or "")
    except json.JSONDecodeError:
        raise ReleaseImageError(
            "runtime registry inspection returned invalid JSON; check Docker Buildx"
        ) from None
    if not isinstance(value, dict):
        raise ReleaseImageError(
            "runtime registry inspection returned an invalid JSON object; "
            "check Docker Buildx"
        )
    manifest = value.get("manifest")
    digest = str(manifest.get("digest") if isinstance(manifest, dict) else "")
    if not DIGEST_PATTERN.fullmatch(digest):
        raise ReleaseImageError("runtime registry image has no manifest digest")
    selected = _platform_image(value.get("image"), platform)
    config = selected.get("config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        raise ReleaseImageError("runtime registry image has no config labels")
    mismatches = [
        name
        for name, expected in sorted(expected_labels.items())
        if labels.get(name) != expected
    ]
    if mismatches:
        raise ReleaseImageError(
            "runtime registry image labels do not match image inputs: "
            + ", ".join(mismatches)
        )
    return digest


def _descriptor(
    *,
    source_sha: str,
    repository: str,
    tag: str,
    digest: str | None,
    platform: str,
    image_inputs: dict[str, Any],
    image_input_sha256: str,
    components: dict[str, dict[str, str]],
    deployable: bool,
    registry_reused: bool,
    cache_from: Sequence[str],
    cache_to: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "deployable": deployable,
        "repository": repository,
        "tag": tag,
        "reference": f"{repository}@{digest}" if deployable and digest else None,
        "digest": digest,
        "platform": platform,
        "source_identity_sha256": source_sha,
        "image_input_sha256": image_input_sha256,
        "image_inputs": image_inputs,
        "registry_reused": registry_reused,
        "build_cache": {
            "from_configured": bool(cache_from),
            "to_configured": bool(cache_to),
        },
        "dockerfile_sha256": image_inputs["dockerfile_sha256"],
        "dependency_lock_sha256": image_inputs["dependency_lock_sha256"],
        "components": components,
    }


def publish_context(
    root: Path,
    *,
    context: Path,
    repository: str,
    platform: str,
    image_inputs: dict[str, Any],
    push: bool,
    build_args: Mapping[str, str],
    cache_from: Sequence[str],
    cache_to: Sequence[str],
    reuse_registry_image: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    """Publish a context and verify the registry, including an immutable-tag race."""
    repository = normalize_repository(repository)
    image_input_sha256 = canonical_sha256(image_inputs)
    tag = f"{repository}:build-{image_input_sha256}"
    labels = _image_labels(image_inputs, image_input_sha256)
    if "wheelhouse_sha256" in image_inputs:
        labels["gpu-fault.node-wheelhouse.sha256"] = image_inputs["wheelhouse_sha256"]

    def inspect() -> str | None:
        return inspect_registry_image(
            root, tag=tag, platform=platform, expected_labels=labels, runner=runner
        )

    digest = inspect() if push and reuse_registry_image else None
    reused = digest is not None
    if digest is None:
        metadata_path = context.parent / "metadata.json"
        command = [
            "docker",
            "buildx",
            "build",
            "--file",
            str(context / "Dockerfile"),
            "--platform",
            platform,
            "--tag",
            tag,
            "--metadata-file",
            str(metadata_path),
        ]
        for name, value in sorted(build_args.items()):
            command.extend(("--build-arg", f"{name}={value}"))
        for name, value in sorted(labels.items()):
            command.extend(("--label", f"{name}={value}"))
        for value in cache_from:
            command.extend(("--cache-from", value))
        for value in cache_to:
            command.extend(("--cache-to", value))
        command.extend(("--push" if push else "--load", str(context)))
        completed = runner(command, cwd=root, check=False, text=True)
        if completed.returncode:
            digest = inspect() if push else None
            if digest is None:
                raise ReleaseImageError(
                    f"runtime image build failed with status {completed.returncode}"
                )
            reused = True
        else:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise ReleaseImageError(
                    "runtime image build metadata is invalid"
                ) from exc
            metadata_digest = str(metadata.get("containerimage.digest") or "")
            if push:
                if not DIGEST_PATTERN.fullmatch(metadata_digest):
                    raise ReleaseImageError(
                        "pushed runtime image has no OCI manifest digest"
                    )
                digest = inspect()
                if digest is None or digest != metadata_digest:
                    raise ReleaseImageError(
                        "runtime image metadata and registry digest do not match"
                    )
            elif DIGEST_PATTERN.fullmatch(metadata_digest):
                digest = metadata_digest
    return _descriptor(
        source_sha=str(build_release_identity(root)["sha256"]),
        repository=repository,
        tag=tag,
        digest=digest,
        platform=platform,
        image_inputs=image_inputs,
        image_input_sha256=image_input_sha256,
        components=image_inputs["components"],
        deployable=push,
        registry_reused=reused,
        cache_from=cache_from,
        cache_to=cache_to,
    )


def build_runtime_image(
    root: Path,
    *,
    repository: str,
    platform: str = "linux/amd64",
    push: bool,
    build_args: Mapping[str, str] | None = None,
    cache_from: Sequence[str] = (),
    cache_to: Sequence[str] = (),
    component_artifacts: Path | None = None,
    reuse_registry_image: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    component: str | None = None,
) -> dict[str, Any]:
    repository = normalize_repository(repository)
    platform = _normalize_platform(platform)
    build_args = dict(sorted((build_args or {}).items()))
    selected: tuple[str, ...] = RUNTIME_COMPONENTS
    dockerfile_name = "Dockerfile"
    if component is not None:
        if component not in RUNTIME_COMPONENTS:
            raise ReleaseImageError("unknown runtime image component")
        if {"COMPONENT", "DISTRIBUTION"} & build_args.keys():
            raise ReleaseImageError("component build arguments cannot be overridden")
        selected = (component,)
        dockerfile_name = "Dockerfile.component"
        build_args.update(
            COMPONENT=component.replace("_", "-"),
            DISTRIBUTION=COMPONENTS[component].distribution,
        )
    with tempfile.TemporaryDirectory(prefix="gpu-fault-image-") as directory:
        temporary = Path(directory)
        context = temporary / "context"
        wheels = context / "wheels"
        requirements = context / "requirements"
        wheels.mkdir(parents=True)
        requirements.mkdir()
        shutil.copy2(root / "deploy/image" / dockerfile_name, context / "Dockerfile")
        shutil.copy2(
            root / "requirements/runtime.lock",
            requirements / "runtime.lock",
        )
        components: dict[str, dict[str, str]] = {}
        cached = (
            load_component_artifacts(root, component_artifacts)
            if component_artifacts is not None
            else None
        )
        for name in selected:
            if cached is None:
                wheel, module_digest, _modules = build_component(
                    python=sys.executable,
                    name=name,
                    build_root=temporary / "components",
                    output=wheels,
                )
            else:
                source = cached.wheels[name]
                wheel = Path(shutil.copy2(source, wheels / source.name))
                module_digest = cached.module_digests[name]
            components[name] = {
                "distribution": COMPONENTS[name].distribution,
                "wheel_sha256": _sha256(wheel),
                "module_digest": module_digest,
            }
        image_inputs = runtime_image_inputs(
            root,
            platform=platform,
            build_args=build_args,
            components=components,
            dockerfile_name=dockerfile_name,
        )
        return publish_context(
            root,
            context=context,
            repository=repository,
            platform=platform,
            image_inputs=image_inputs,
            push=push,
            build_args=build_args,
            cache_from=cache_from,
            cache_to=cache_to,
            reuse_registry_image=reuse_registry_image,
            runner=runner,
        )
