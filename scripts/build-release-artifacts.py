"""Build one content-addressed wheel and node bundle release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path


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


def build(python: str) -> dict[str, object]:
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir()

    run([python, "-m", "build", "--wheel", "--outdir", str(DIST)])
    wheels = sorted(DIST.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"wheel build produced {len(wheels)} wheels; expected one")
    wheel = wheels[0]
    wheel_sha = sha256(wheel)
    release_id = wheel_sha[:12]
    release_dir = DIST / release_id
    release_dir.mkdir()
    wheel = Path(shutil.move(str(wheel), release_dir / wheel.name))

    env = {
        **os.environ,
        "GPU_FAULT_WHEEL": str(wheel),
    }
    run(
        [
            str(ROOT / "deploy/node/build-node-installer-bundle.sh"),
            str(release_dir),
        ],
        env=env,
    )
    bundles = sorted(release_dir.glob("gpu-fault-node-installer-*.tar.gz"))
    if len(bundles) != 1:
        raise RuntimeError(
            f"bundle build produced {len(bundles)} bundles; expected one"
        )
    bundle = bundles[0]

    module_digest = subprocess.run(
        [
            python,
            "-c",
            "from gpu_fault import module_digest; print(module_digest())",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": project_version(),
        "release_id": release_id,
        "wheel": wheel.relative_to(ROOT).as_posix(),
        "wheel_sha256": wheel_sha,
        "bundle": bundle.relative_to(ROOT).as_posix(),
        "bundle_sha256": sha256(bundle),
        "module_digest": module_digest,
    }
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (release_dir / "release.json").write_text(
        content,
        encoding="utf-8",
    )
    (DIST / "current-release.json").write_text(
        content,
        encoding="utf-8",
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
        "wheel={wheel}\n"
        "wheel_sha256={wheel_sha256}\n"
        "bundle={bundle}\n"
        "bundle_sha256={bundle_sha256}\n"
        "module_digest={module_digest}".format(**manifest)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
