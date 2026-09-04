"""Build one content-addressed wheel and node bundle release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from component_artifact_cache import (
    find_cached_manifest,
    store_component_artifacts,
)
from component_artifacts import (
    ComponentArtifactError,
    component_build_identity,
    load_component_artifacts,
)
from component_wheels import build_component
from release_identity import bind_runtime_image, build_release_identity


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DIST = ROOT / "dist"
PRESERVED_DIST_DIRECTORIES = {"ci-domains"}


@dataclass(frozen=True)
class PreparedArtifacts:
    wheels: dict[str, Path]
    bundle: Path
    module_digests: dict[str, str]
    module_counts: dict[str, int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_version() -> str:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(document["project"]["version"])


def validate_runtime_components(
    descriptor: dict[str, object],
    *,
    hashes: dict[str, str],
    module_digests: dict[str, str],
) -> None:
    raw_components = descriptor.get("components")
    if not isinstance(raw_components, dict):
        raise RuntimeError("runtime image descriptor has no component identities")
    for name in ("control_plane", "executor"):
        value = raw_components.get(name)
        if not isinstance(value, dict):
            raise RuntimeError(f"runtime image descriptor has no {name} identity")
        if value.get("wheel_sha256") != hashes[name]:
            raise RuntimeError(
                f"runtime image {name} wheel does not match release artifact"
            )
        if value.get("module_digest") != module_digests[name]:
            raise RuntimeError(
                f"runtime image {name} module digest does not match release artifact"
            )


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )


def resolve_python_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable is None:
        raise RuntimeError(f"Python executable was not found: {value}")
    return str(Path(executable).absolute())


def _write_synced(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as destination:
        destination.write(content)
        destination.flush()
        os.fsync(destination.fileno())


def _existing_release(
    release_dir: Path,
    *,
    release_id: str,
    hashes: dict[str, str],
    delivery_sha256: str,
    staging_only: bool,
) -> tuple[dict[str, object], str] | None:
    manifest_path = release_dir / "release.json"
    if not manifest_path.is_file():
        return None
    content = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(content)
    components = manifest.get("components") or {}
    expected = {
        "control_plane": hashes["control_plane"],
        "executor": hashes["executor"],
        "node_runtime": hashes["node_runtime"],
    }
    if manifest.get("release_id") != release_id or any(
        (components.get(name) or {}).get("wheel_sha256") != digest
        for name, digest in expected.items()
    ):
        raise RuntimeError(f"content-addressed release collision: {release_id}")
    if manifest.get("bundle_sha256") != hashes["node_bundle"]:
        raise RuntimeError(f"content-addressed bundle collision: {release_id}")
    if (manifest.get("delivery") or {}).get("sha256") != delivery_sha256:
        raise RuntimeError(f"content-addressed delivery collision: {release_id}")
    if manifest.get("staging_only", False) is not staging_only:
        raise RuntimeError(f"content-addressed release tier collision: {release_id}")
    for name, digest in expected.items():
        wheel = ROOT / str((components[name] or {})["wheel"])
        if not wheel.is_file() or sha256(wheel) != digest:
            raise RuntimeError(f"existing {name} artifact is incomplete: {release_id}")
    bundle = ROOT / str(manifest["bundle"])
    if not bundle.is_file() or sha256(bundle) != hashes["node_bundle"]:
        raise RuntimeError(f"existing node bundle is incomplete: {release_id}")
    return manifest, content


def _publish_release(
    staging: Path,
    *,
    release_id: str,
    hashes: dict[str, str],
    delivery_sha256: str,
    staging_only: bool,
    manifest: dict[str, object],
    content: str,
) -> tuple[dict[str, object], str]:
    release_dir = DIST / release_id
    existing = _existing_release(
        release_dir,
        release_id=release_id,
        hashes=hashes,
        delivery_sha256=delivery_sha256,
        staging_only=staging_only,
    )
    staged_release = staging / release_id
    if existing is None:
        os.replace(staged_release, release_dir)
    else:
        manifest, content = existing
        shutil.rmtree(staged_release)

    staged_current = staging / "current-release.json"
    _write_synced(staged_current, content)
    os.replace(staged_current, DIST / "current-release.json")
    prune_dist(release_dir)
    return manifest, content


def prune_dist(release_dir: Path) -> None:
    for path in sorted(DIST.iterdir()):
        if (
            path.is_dir()
            and not path.name.startswith(".")
            and path.name not in PRESERVED_DIST_DIRECTORIES
            and path != release_dir
        ):
            shutil.rmtree(path)


def prune_abandoned_staging() -> None:
    """Drop staging directories a killed build left inside dist.

    A build stages the wheels and the node bundle under ``dist/.build-XXXXXXXX``
    and renames the finished release into place. Every ordinary exit removes
    that directory, a failed build included, which is why ``prune_dist`` leaves
    dot-directories alone -- the staging directory of the build calling it is
    live. A signal that kills the process outright skips the cleanup: an
    interrupted deploy, an OOM kill, a host reboot.

    The leftover is not inert. ``artifact-check`` counts release artifacts with
    ``dist.rglob("*.whl")``, so the next build fails the gate with "6 wheels,
    expected 3" and names the release it just built correctly -- an operator
    reads that as a broken build rather than as the corpse of the one before it.
    Concurrent builds in one tree are already unsupported, since ``prune_dist``
    deletes every release directory but its own, so clearing these costs nothing
    a caller could have been relying on.
    """

    if not DIST.is_dir():
        return
    for path in sorted(DIST.glob(".build-*")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def prepare_component_artifacts(
    python: str,
    *,
    staging: Path,
    reuse_artifacts_from: Path | None,
) -> PreparedArtifacts:
    if reuse_artifacts_from is not None:
        resolved_manifest = reuse_artifacts_from.resolve()
        external = not resolved_manifest.is_relative_to((ROOT / "dist").resolve())
        cached = load_component_artifacts(
            ROOT,
            reuse_artifacts_from,
            artifact_root=(resolved_manifest.parent.parent if external else ROOT),
            require_delivery_identity=not external,
        )
        wheels = {
            name: Path(shutil.copy2(path, staging / path.name))
            for name, path in cached.wheels.items()
        }
        return PreparedArtifacts(
            wheels=wheels,
            bundle=Path(shutil.copy2(cached.bundle, staging / cached.bundle.name)),
            module_digests=cached.module_digests,
            module_counts=cached.module_counts,
        )

    built = {
        name: build_component(
            python=python,
            name=name,
            build_root=BUILD / "components",
            output=staging,
        )
        for name in ("control_plane", "executor", "node_runtime")
    }
    node_wheel = built["node_runtime"][0]
    run(
        [
            str(ROOT / "deploy/node/build-node-installer-bundle.sh"),
            str(staging),
        ],
        env={**os.environ, "GPU_FAULT_NODE_WHEEL": str(node_wheel)},
    )
    bundles = sorted(staging.glob("gpu-fault-node-installer-*.tar.gz"))
    if len(bundles) != 1:
        raise RuntimeError(
            f"bundle build produced {len(bundles)} bundles; expected one"
        )
    return PreparedArtifacts(
        wheels={name: value[0] for name, value in built.items()},
        bundle=bundles[0],
        module_digests={name: value[1] for name, value in built.items()},
        module_counts={name: len(value[2]) for name, value in built.items()},
    )


def resolve_current_component_artifacts(
    *,
    staging_only: bool,
    component_cache_root: Path | None,
) -> tuple[dict[str, object] | None, Path | None]:
    try:
        cached = load_component_artifacts(
            ROOT,
            DIST / "current-release.json",
        )
    except ComponentArtifactError:
        cached_manifest = (
            find_cached_manifest(ROOT, component_cache_root)
            if component_cache_root is not None
            else None
        )
        return None, cached_manifest
    if cached.manifest.get("staging_only", False) is not staging_only:
        return None, None
    if component_cache_root is not None:
        store_component_artifacts(
            ROOT,
            component_cache_root,
            DIST / "current-release.json",
        )
    return cached.manifest, None


def build(
    python: str,
    *,
    runtime_image_descriptor: Path | None = None,
    staging_only: bool = False,
    reuse_artifacts_from: Path | None = None,
    reuse_if_current: bool = False,
    component_cache_root: Path | None = None,
) -> dict[str, object]:
    python = resolve_python_executable(python)
    # Before the reuse shortcut below, so that a build which republishes the
    # current release still clears a leftover the artifact gate would trip on.
    prune_abandoned_staging()
    delivery = build_release_identity(ROOT)
    if runtime_image_descriptor is None and reuse_if_current:
        current, cached_manifest = resolve_current_component_artifacts(
            staging_only=staging_only,
            component_cache_root=component_cache_root,
        )
        if current is not None:
            return current
        reuse_artifacts_from = reuse_artifacts_from or cached_manifest
    runtime_descriptor: dict[str, object] | None = None
    if runtime_image_descriptor is not None:
        runtime_descriptor = json.loads(
            runtime_image_descriptor.read_text(encoding="utf-8")
        )
        delivery = bind_runtime_image(ROOT, delivery, runtime_descriptor)
    shutil.rmtree(BUILD, ignore_errors=True)
    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=DIST) as directory:
        staging = Path(directory)
        artifacts = prepare_component_artifacts(
            python,
            staging=staging,
            reuse_artifacts_from=reuse_artifacts_from,
        )
        control_wheel = artifacts.wheels["control_plane"]
        executor_wheel = artifacts.wheels["executor"]
        node_wheel = artifacts.wheels["node_runtime"]
        bundle = artifacts.bundle
        control_digest = artifacts.module_digests["control_plane"]
        executor_digest = artifacts.module_digests["executor"]
        node_digest = artifacts.module_digests["node_runtime"]
        module_counts = artifacts.module_counts

        hashes = {
            "control_plane": sha256(control_wheel),
            "executor": sha256(executor_wheel),
            "node_runtime": sha256(node_wheel),
            "node_bundle": sha256(bundle),
        }
        module_digests = {
            "control_plane": control_digest,
            "executor": executor_digest,
            "node_runtime": node_digest,
        }
        if runtime_descriptor is not None:
            validate_runtime_components(
                runtime_descriptor,
                hashes=hashes,
                module_digests=module_digests,
            )
        release_identity = {
            "artifacts": hashes,
            "delivery": delivery["sha256"],
        }
        if staging_only:
            release_identity["staging_only"] = True
        release_id = hashlib.sha256(
            json.dumps(
                release_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
        release_dir = staging / release_id
        release_dir.mkdir()
        control_wheel = Path(
            shutil.move(str(control_wheel), release_dir / control_wheel.name)
        )
        executor_wheel = Path(
            shutil.move(str(executor_wheel), release_dir / executor_wheel.name)
        )
        node_wheel = Path(shutil.move(str(node_wheel), release_dir / node_wheel.name))
        bundle = Path(shutil.move(str(bundle), release_dir / bundle.name))

        schema_version = subprocess.run(
            [
                python,
                "-c",
                (
                    "from gpu_fault.schema_migrations import "
                    "LATEST_POSTGRES_SCHEMA_VERSION; "
                    "print(LATEST_POSTGRES_SCHEMA_VERSION)"
                ),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        protocols = json.loads(
            subprocess.run(
                [
                    python,
                    "-c",
                    (
                        "import json;"
                        "from gpu_fault.fleet import CURRENT_AGENT_PROTOCOL_VERSION;"
                        "from gpu_fault.regional_compatibility import "
                        "CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION;"
                        "print(json.dumps({"
                        "'agent':CURRENT_AGENT_PROTOCOL_VERSION,"
                        "'executor':CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION}))"
                    ),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
        published = Path("dist") / release_id
        manifest: dict[str, object] = {
            "schema_version": 3,
            "component_build_identity_sha256": component_build_identity(ROOT),
            "deployable": bool(delivery["runtime_prebuilt"]),
            "staging_only": staging_only,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": project_version(),
            "release_id": release_id,
            # Compatibility aliases for older tooling.
            "wheel": (published / control_wheel.name).as_posix(),
            "wheel_sha256": hashes["control_plane"],
            "bundle": (published / bundle.name).as_posix(),
            "bundle_sha256": hashes["node_bundle"],
            "module_digest": control_digest,
            "database_schema_version": int(schema_version),
            "protocol_versions": protocols,
            "delivery": delivery,
            "database": {
                "schema_version": int(schema_version),
                "rollback_compatible": delivery["schema_rollback_compatible"],
            },
            "components": {
                "control_plane": {
                    "wheel": (published / control_wheel.name).as_posix(),
                    "wheel_sha256": hashes["control_plane"],
                    "module_digest": control_digest,
                    "module_count": module_counts["control_plane"],
                },
                "executor": {
                    "wheel": (published / executor_wheel.name).as_posix(),
                    "wheel_sha256": hashes["executor"],
                    "module_digest": executor_digest,
                    "module_count": module_counts["executor"],
                },
                "node_runtime": {
                    "wheel": (published / node_wheel.name).as_posix(),
                    "wheel_sha256": hashes["node_runtime"],
                    "module_digest": node_digest,
                    "module_count": module_counts["node_runtime"],
                },
                "node_bundle": {
                    "bundle": (published / bundle.name).as_posix(),
                    "bundle_sha256": hashes["node_bundle"],
                    "template_sha256": delivery["node_template_inputs"]["sha256"],
                },
            },
        }
        content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        _write_synced(release_dir / "release.json", content)
        manifest, _content = _publish_release(
            staging,
            release_id=release_id,
            hashes=hashes,
            delivery_sha256=str(delivery["sha256"]),
            staging_only=staging_only,
            manifest=manifest,
            content=content,
        )
        if runtime_image_descriptor is None and component_cache_root is not None:
            store_component_artifacts(
                ROOT,
                component_cache_root,
                DIST / "current-release.json",
            )
    shutil.rmtree(BUILD, ignore_errors=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument(
        "--runtime-image-descriptor",
        type=Path,
    )
    parser.add_argument("--reuse-artifacts-from", type=Path)
    parser.add_argument("--reuse-if-current", action="store_true")
    parser.add_argument("--component-cache-root", type=Path)
    parser.add_argument("--staging-only", action="store_true")
    args = parser.parse_args()
    manifest = build(
        args.python,
        runtime_image_descriptor=(
            args.runtime_image_descriptor.resolve()
            if args.runtime_image_descriptor is not None
            else None
        ),
        staging_only=args.staging_only,
        reuse_artifacts_from=(
            args.reuse_artifacts_from.resolve()
            if args.reuse_artifacts_from is not None
            else None
        ),
        reuse_if_current=args.reuse_if_current,
        component_cache_root=(
            args.component_cache_root.resolve()
            if args.component_cache_root is not None
            else None
        ),
    )
    print(
        "release_id={release_id}\n"
        "control_plane_wheel={wheel}\n"
        "control_plane_wheel_sha256={wheel_sha256}\n"
        "executor_wheel={components[executor][wheel]}\n"
        "node_runtime_wheel={components[node_runtime][wheel]}\n"
        "bundle={bundle}\n"
        "bundle_sha256={bundle_sha256}\n"
        "module_digest={module_digest}".format(**manifest)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
