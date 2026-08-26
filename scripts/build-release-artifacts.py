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
from datetime import datetime, timezone
from pathlib import Path

from component_wheels import build_component


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DIST = ROOT / "dist"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_version() -> str:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(document["project"]["version"])


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )


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
    manifest: dict[str, object],
    content: str,
) -> tuple[dict[str, object], str]:
    release_dir = DIST / release_id
    existing = _existing_release(
        release_dir,
        release_id=release_id,
        hashes=hashes,
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
    for path in sorted(DIST.iterdir()):
        if path.is_dir() and not path.name.startswith(".") and path != release_dir:
            shutil.rmtree(path)
    return manifest, content


def build(python: str) -> dict[str, object]:
    python = str(Path(python).resolve())
    shutil.rmtree(BUILD, ignore_errors=True)
    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=DIST) as directory:
        staging = Path(directory)
        control_wheel, control_digest, control_modules = build_component(
            python=python,
            name="control_plane",
            build_root=BUILD / "components",
            output=staging,
        )
        executor_wheel, executor_digest, executor_modules = build_component(
            python=python,
            name="executor",
            build_root=BUILD / "components",
            output=staging,
        )
        node_wheel, node_digest, node_modules = build_component(
            python=python,
            name="node_runtime",
            build_root=BUILD / "components",
            output=staging,
        )
        env = {
            **os.environ,
            "GPU_FAULT_NODE_WHEEL": str(node_wheel),
        }
        run(
            [
                str(ROOT / "deploy/node/build-node-installer-bundle.sh"),
                str(staging),
            ],
            env=env,
        )
        bundles = sorted(staging.glob("gpu-fault-node-installer-*.tar.gz"))
        if len(bundles) != 1:
            raise RuntimeError(
                f"bundle build produced {len(bundles)} bundles; expected one"
            )
        bundle = bundles[0]

        hashes = {
            "control_plane": sha256(control_wheel),
            "executor": sha256(executor_wheel),
            "node_runtime": sha256(node_wheel),
            "node_bundle": sha256(bundle),
        }
        release_id = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
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
            "schema_version": 2,
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
            "components": {
                "control_plane": {
                    "wheel": (published / control_wheel.name).as_posix(),
                    "wheel_sha256": hashes["control_plane"],
                    "module_digest": control_digest,
                    "module_count": len(control_modules),
                },
                "executor": {
                    "wheel": (published / executor_wheel.name).as_posix(),
                    "wheel_sha256": hashes["executor"],
                    "module_digest": executor_digest,
                    "module_count": len(executor_modules),
                },
                "node_runtime": {
                    "wheel": (published / node_wheel.name).as_posix(),
                    "wheel_sha256": hashes["node_runtime"],
                    "module_digest": node_digest,
                    "module_count": len(node_modules),
                },
                "node_bundle": {
                    "bundle": (published / bundle.name).as_posix(),
                    "bundle_sha256": hashes["node_bundle"],
                },
            },
        }
        content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        _write_synced(release_dir / "release.json", content)
        manifest, _content = _publish_release(
            staging,
            release_id=release_id,
            hashes=hashes,
            manifest=manifest,
            content=content,
        )
    shutil.rmtree(BUILD, ignore_errors=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    args = parser.parse_args()
    manifest = build(args.python)
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
