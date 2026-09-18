"""Build independently reusable CPU, Executor and node dependency OCI images."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from scripts.node_wheelhouse import (
        NodeImageCache,
        build_wheelhouse,
        sha256,
        validate_build_host,
        validate_wheelhouse_inventory,
    )
    from scripts.release_identity import build_release_identity, load_image_lock
    from scripts.release_image import (
        DIGEST_PATTERN,
        ReleaseImageError,
        _descriptor,
        _image_labels,
        build_runtime_image,
        canonical_sha256,
        inspect_registry_image,
        publish_context,
        normalize_repository,
    )
else:
    from node_wheelhouse import (
        NodeImageCache,
        build_wheelhouse,
        sha256,
        validate_build_host,
        validate_wheelhouse_inventory,
    )
    from release_identity import build_release_identity, load_image_lock
    from release_image import (
        DIGEST_PATTERN,
        ReleaseImageError,
        _descriptor,
        _image_labels,
        build_runtime_image,
        canonical_sha256,
        inspect_registry_image,
        publish_context,
        normalize_repository,
    )


_CACHE_PATH_PART = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_CACHE_REFERENCE = re.compile(
    r"(?P<repository>(?:(?:[a-z0-9.-]+|\[[a-fA-F0-9:]+\])(?::[0-9]+)?/)?"
    + _CACHE_PATH_PART
    + rf"(?:/{_CACHE_PATH_PART})*)"
    + r"(?::(?P<tag>[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}))?"
    + r"(?:@(?P<digest>sha256:[0-9a-f]{64}))?"
)
_CACHE_CSV_FIELD = r'(?:[^",\r\n]*|"(?:[^"]|"")*")'
_CACHE_EXPORT_OPTIONS = {
    "type",
    "mode",
    "image-manifest",
    "oci-mediatypes",
    "compression",
    "compression-level",
    "force-compression",
    "ignore-error",
}


def _cache_fields(value: str) -> list[str]:
    if (
        not value
        or any(ord(character) < 32 for character in value)
        or not re.fullmatch(rf"{_CACHE_CSV_FIELD}(?:,{_CACHE_CSV_FIELD})*", value)
    ):
        raise ReleaseImageError("BuildKit cache options must be valid single-row CSV")
    try:
        return next(csv.reader([value], strict=True))
    except csv.Error:
        raise ReleaseImageError("BuildKit cache options contain invalid CSV") from None


def _cache_options(fields: Sequence[str]) -> dict[str, str]:
    if len(fields) == 1 and "=" not in fields[0]:
        return {"type": "registry", "ref": fields[0]}
    options: dict[str, str] = {}
    for field in fields:
        name, separator, argument = field.partition("=")
        name = name.lower()
        if (
            not separator
            or not re.fullmatch(r"[a-z][a-z0-9.-]*", name)
            or not argument
            or name in options
        ):
            raise ReleaseImageError(
                "BuildKit cache options require unique non-empty key=value fields"
            )
        options[name] = argument
    if "type" not in options:
        raise ReleaseImageError("BuildKit cache options require an explicit type")
    return options


def _cache_spec(options: Mapping[str, str]) -> str:
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerow(
        f"{name}={value}" for name, value in options.items()
    )
    return output.getvalue().removesuffix("\n")


def _scoped_cache(
    root: Path,
    options: Mapping[str, str],
    component: str,
    *,
    export: bool,
) -> dict[str, str] | None:
    kind = options["type"]
    if kind not in {"registry", "local"}:
        if export:
            raise ReleaseImageError(
                "split image cache exports support only registry ref or local dest"
            )
        return None
    field = "ref" if kind == "registry" else "dest" if export else "src"
    target = options.get(field)
    if not target:
        raise ReleaseImageError(f"BuildKit {kind} cache requires {field}")
    if export and set(options) - (_CACHE_EXPORT_OPTIONS | {field}):
        raise ReleaseImageError("unsupported split image cache export options")
    # Import selectors or backend-specific options must keep their original meaning.
    if not export and set(options) - {"type", field}:
        return None
    if kind == "registry":
        match = _CACHE_REFERENCE.fullmatch(target)
        if match is None:
            raise ReleaseImageError("BuildKit registry cache reference is invalid")
        if match["digest"] is not None:
            if export:
                raise ReleaseImageError(
                    "BuildKit registry cache exports cannot use digest references"
                )
            return None
        tag = f"{match['tag'] or 'latest'}-{component}"
        if len(tag) > 128:
            if not export:
                return None
            raise ReleaseImageError(
                "component BuildKit cache tag exceeds 128 characters"
            )
        scoped = f"{match['repository']}:{tag}"
    else:
        path = Path(target)
        scoped = str(path / component)
        if export:
            try:
                destination = (root / scoped).resolve()
                expected = (root / path).resolve() / component
            except (OSError, RuntimeError):
                raise ReleaseImageError(
                    "cannot resolve component BuildKit cache destination"
                ) from None
            if destination != expected:
                raise ReleaseImageError(
                    "component BuildKit cache destination must not be a symlink"
                )
    return {**options, field: scoped}


def _cache_export_target(root: Path, options: Mapping[str, str]) -> tuple[str, str]:
    if options["type"] == "registry":
        match = _CACHE_REFERENCE.fullmatch(options["ref"])
        assert match is not None
        repository = match["repository"]
        registry, slash, path = repository.partition("/")
        if not slash or not (
            "." in registry or ":" in registry or registry == "localhost"
        ):
            registry, path = "docker.io", repository
        if registry == "index.docker.io":
            registry = "docker.io"
        if registry == "docker.io" and "/" not in path:
            path = f"library/{path}"
        return "registry", f"{registry}/{path}:{match['tag'] or 'latest'}"
    try:
        return "local", str((root / options["dest"]).resolve())
    except (OSError, RuntimeError):
        raise ReleaseImageError(
            "cannot resolve BuildKit cache export destination"
        ) from None


def _component_build_cache(
    root: Path,
    *,
    component: str,
    cache_from: Sequence[str],
    cache_to: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    component = component.replace("_", "-")
    imports: list[str] = []
    exports: list[str] = []
    targets: set[tuple[str, str]] = set()
    shared_targets: set[tuple[str, str]] = set()
    for value in cache_to:
        if "=" not in value:
            raise ReleaseImageError("BuildKit cache exports require an explicit type")
        shared = _cache_options(_cache_fields(value))
        options = _scoped_cache(root, shared, component, export=True)
        assert options is not None
        target = _cache_export_target(root, options)
        if target in targets:
            raise ReleaseImageError("duplicate component BuildKit cache export target")
        targets.add(target)
        shared_targets.add(_cache_export_target(root, shared))
        exports.append(_cache_spec(options))
    if targets & shared_targets:
        raise ReleaseImageError(
            "component BuildKit cache export overlaps a shared target"
        )
    for value in cache_from:
        # Buildx also accepts a comma-separated list of shorthand registry imports.
        fields = _cache_fields(value)
        sources = (
            [{"type": "registry", "ref": field} for field in fields]
            if all("=" not in field for field in fields)
            else [_cache_options(fields)]
        )
        for source in sources:
            options = _scoped_cache(root, source, component, export=False)
            if options is not None:
                imports.append(_cache_spec(options))
        imports.append(value)
    return tuple(dict.fromkeys(imports)), tuple(exports)


def _node_image_inputs(root: Path) -> dict[str, Any]:
    base = load_image_lock(root / "config/release-images.lock.json")["node_installer"][
        "reference"
    ]
    return {
        "schema_version": 1,
        "platform": "linux/amd64",
        "build_args": {"NODE_INSTALLER_BASE_IMAGE": base},
        "base_images": [base],
        "dockerfile_sha256": sha256(root / "deploy/image/Dockerfile.node-dependencies"),
        "dependency_lock_sha256": sha256(root / "requirements/node-runtime.lock"),
        "tools_lock_sha256": sha256(root / "requirements/node-tools.lock"),
        "wheelhouse_builder_sha256": sha256(root / "scripts/node_wheelhouse.py"),
        "components": {},
    }


def _node_cache_identity(root: Path, repository: str, inputs: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "repository": repository,
            "image_inputs": inputs,
            "image_set_builder_sha256": sha256(root / "scripts/release_image_set.py"),
            "publisher_sha256": sha256(root / "scripts/release_image.py"),
        }
    )


def _reuse_node_image(
    root: Path,
    *,
    cached: Mapping[str, Any],
    inputs: Mapping[str, Any],
    repository: str,
    source_sha: str,
    cache_from: Sequence[str],
    cache_to: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any] | None:
    cached_inputs = cached.get("image_inputs")
    digest = cached.get("digest")
    if not isinstance(cached_inputs, dict):
        raise ReleaseImageError("cached node image inputs are invalid")
    wheelhouse_sha = cached_inputs.get("wheelhouse_sha256")
    if (
        not isinstance(digest, str)
        or not DIGEST_PATTERN.fullmatch(digest)
        or not isinstance(wheelhouse_sha, str)
        or not DIGEST_PATTERN.fullmatch(f"sha256:{wheelhouse_sha}")
        or cached_inputs != {**inputs, "wheelhouse_sha256": wheelhouse_sha}
    ):
        raise ReleaseImageError("cached node image identity differs from its inputs")
    if validate_wheelhouse_inventory(root, cached.get("inventory")) != wheelhouse_sha:
        raise ReleaseImageError("cached node image inventory digest differs")
    input_sha = canonical_sha256(cached_inputs)
    labels = _image_labels(cached_inputs, input_sha)
    labels["gpu-fault.node-wheelhouse.sha256"] = wheelhouse_sha
    # The receipt is a lookup hint; locks and a pinned OCI proof authorize reuse.
    observed = inspect_registry_image(
        root,
        tag=f"{repository}@{digest}",
        platform="linux/amd64",
        expected_labels=labels,
        runner=runner,
    )
    if observed is None:
        return None
    if observed != digest:
        raise ReleaseImageError("cached node image registry digest differs")
    descriptor = _descriptor(
        source_sha=source_sha,
        repository=repository,
        tag=f"{repository}:build-{input_sha}",
        digest=digest,
        platform="linux/amd64",
        image_inputs=cached_inputs,
        image_input_sha256=input_sha,
        components={},
        deployable=True,
        registry_reused=True,
        cache_from=cache_from,
        cache_to=cache_to,
    )
    descriptor["wheelhouse_sha256"] = wheelhouse_sha
    return descriptor


def build_node_dependency_image(
    root: Path,
    *,
    repository: str,
    push: bool,
    cache_from: Sequence[str] = (),
    cache_to: Sequence[str] = (),
    reuse_registry_image: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    validate_build_host()
    repository = normalize_repository(repository)
    cache = NodeImageCache()
    if cache.directory.resolve().is_relative_to(root.resolve()):
        raise ReleaseImageError("node image cache must be outside the source checkout")
    source_sha = str(build_release_identity(root)["sha256"])
    inputs = _node_image_inputs(root)
    cache_identity = _node_cache_identity(root, repository, inputs)

    def check_source() -> None:
        if (
            _node_cache_identity(root, repository, _node_image_inputs(root))
            != cache_identity
            or str(build_release_identity(root)["sha256"]) != source_sha
        ):
            raise ReleaseImageError("release inputs changed during node image build")

    if push and reuse_registry_image:
        cached = cache.load(cache_identity)
        if cached is not None:
            reused = _reuse_node_image(
                root,
                cached=cached,
                inputs=inputs,
                repository=repository,
                source_sha=source_sha,
                cache_from=cache_from,
                cache_to=cache_to,
                runner=runner,
            )
            check_source()
            if reused is not None:
                return reused
    with tempfile.TemporaryDirectory(prefix="gpu-fault-node-image-") as directory:
        context = Path(directory) / "context"
        context.mkdir()
        dockerfile = root / "deploy/image/Dockerfile.node-dependencies"
        shutil.copyfile(dockerfile, context / "Dockerfile")
        if sha256(context / "Dockerfile") != inputs["dockerfile_sha256"]:
            raise ReleaseImageError("node image Dockerfile changed during copy")
        wheelhouse_sha = build_wheelhouse(
            root,
            context / "wheelhouse",
            runner=runner,
            expected_locks={
                "node-runtime.lock": inputs["dependency_lock_sha256"],
                "node-tools.lock": inputs["tools_lock_sha256"],
            },
        )
        check_source()
        inventory = json.loads((context / "wheelhouse/inventory.json").read_bytes())
        if validate_wheelhouse_inventory(root, inventory) != wheelhouse_sha:
            raise ReleaseImageError("node image inventory changed before publication")
        image_inputs = {**inputs, "wheelhouse_sha256": wheelhouse_sha}
        descriptor = publish_context(
            root,
            context=context,
            repository=repository,
            platform="linux/amd64",
            image_inputs=image_inputs,
            push=push,
            build_args=inputs["build_args"],
            cache_from=cache_from,
            cache_to=cache_to,
            reuse_registry_image=reuse_registry_image,
            runner=runner,
        )
        check_source()
        descriptor["wheelhouse_sha256"] = wheelhouse_sha
        if push and reuse_registry_image:
            cache.store(
                cache_identity,
                {
                    "digest": descriptor["digest"],
                    "image_inputs": image_inputs,
                    "inventory": inventory,
                },
            )
        return descriptor


def build_image_set(
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
) -> dict[str, Any]:
    if platform != "linux/amd64":
        raise ReleaseImageError("node dependency image set supports linux/amd64 only")
    repository = normalize_repository(repository)
    caches = {
        name: _component_build_cache(
            root, component=name, cache_from=cache_from, cache_to=cache_to
        )
        for name in ("control_plane", "executor", "node_dependencies")
    }
    source_identity = str(build_release_identity(root)["sha256"])
    # Bound independent builds; every started mutation completes before returning.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            name: pool.submit(
                build_runtime_image,
                root,
                repository=repository,
                platform=platform,
                push=push,
                build_args=build_args,
                cache_from=caches[name][0],
                cache_to=caches[name][1],
                component_artifacts=component_artifacts,
                reuse_registry_image=reuse_registry_image,
                runner=runner,
                component=name,
            )
            for name in ("control_plane", "executor")
        }
        futures["node_dependencies"] = pool.submit(
            build_node_dependency_image,
            root,
            repository=repository,
            push=push,
            cache_from=caches["node_dependencies"][0],
            cache_to=caches["node_dependencies"][1],
            reuse_registry_image=reuse_registry_image,
            runner=runner,
        )
        images = {name: future.result() for name, future in futures.items()}
    if any(
        item["source_identity_sha256"] != source_identity for item in images.values()
    ):
        raise ReleaseImageError("release inputs changed during image builds")
    return {
        "schema_version": 3,
        "deployable": push,
        "source_identity_sha256": source_identity,
        "images": images,
        "components": {
            name: images[name]["components"][name]
            for name in ("control_plane", "executor")
        },
    }
