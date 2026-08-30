from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from deploy_host_bundle import (
    BUNDLE_ROOT,
    bundle_platform_id,
    host_compatibility,
    sha256_file,
    write_bundle_manifest,
    write_deterministic_archive,
    write_sha256_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]


def _run(
    arguments: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(arguments, cwd=cwd, env=env, check=True)


def _project_version() -> str:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(document["project"]["version"])


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _python_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable is None:
        raise RuntimeError(f"Python executable was not found: {value}")
    return str(Path(executable).absolute())


def _copy_files(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _build_wheelhouse(
    python: str,
    destination: Path,
    *,
    source: Path | None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if source is not None:
        _copy_files(source, destination)
    else:
        for lock in (
            ROOT / "requirements/build.lock",
            ROOT / "requirements/deploy-host.lock",
        ):
            _run(
                [
                    python,
                    "-m",
                    "pip",
                    "wheel",
                    "--require-hashes",
                    "--wheel-dir",
                    str(destination),
                    "--requirement",
                    str(lock),
                ]
            )
    for existing in destination.glob("gpu_fault_control_plane-*.whl"):
        existing.unlink()
    source_date_epoch = _git_output("show", "-s", "--format=%ct", "HEAD")
    build_environment = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "SOURCE_DATE_EPOCH": source_date_epoch,
    }
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="gpu-fault-deploy-host-build-",
            dir=destination.parent,
        ) as directory:
            build_venv = Path(directory)
            _run([python, "-m", "venv", str(build_venv)])
            build_python = build_venv / "bin/python"
            for lock in (
                ROOT / "requirements/build.lock",
                ROOT / "requirements/deploy-host.lock",
            ):
                _run(
                    [
                        str(build_python),
                        "-m",
                        "pip",
                        "install",
                        "--no-index",
                        "--find-links",
                        str(destination),
                        "--require-hashes",
                        "--requirement",
                        str(lock),
                    ]
                )
            _run(
                [
                    str(build_python),
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(destination),
                ],
                env=build_environment,
            )
    finally:
        shutil.rmtree(ROOT / "build", ignore_errors=True)
    wheels = sorted(destination.glob("gpu_fault_control_plane-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"deploy-host bundle requires one project wheel, found {len(wheels)}"
        )
    return wheels[0]


def build_bundle(
    *,
    python: str,
    output: Path,
    wheelhouse: Path | None,
    tools_dir: Path | None,
    allow_dirty: bool,
) -> dict[str, str]:
    status = _git_output("status", "--porcelain", "--untracked-files=normal")
    if status and not allow_dirty:
        raise RuntimeError("deploy-host bundle requires a clean source tree")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-deploy-host-") as directory:
        staging = Path(directory) / BUNDLE_ROOT
        requirements = staging / "requirements"
        requirements.mkdir(parents=True)
        for name in ("build.lock", "deploy-host.lock"):
            shutil.copy2(ROOT / "requirements" / name, requirements / name)
        tool_manifest = staging / "deploy-host-tools.json"
        shutil.copy2(
            ROOT / "src/gpu_fault/data/deploy-host-tools.json",
            tool_manifest,
        )
        project_wheel = _build_wheelhouse(
            python,
            staging / "wheelhouse",
            source=wheelhouse,
        )
        if tools_dir is not None:
            _copy_files(tools_dir, staging / "tools")
        compatibility = host_compatibility()
        write_bundle_manifest(
            staging,
            metadata={
                "compatibility": compatibility,
                "platform_id": bundle_platform_id(compatibility),
                "project_version": _project_version(),
                "project_wheel": project_wheel.relative_to(staging).as_posix(),
                "python": "3.12",
                "requirements": {
                    "build": "requirements/build.lock",
                    "deploy_host": "requirements/deploy-host.lock",
                },
                "source": {
                    "git_commit": _git_output("rev-parse", "HEAD"),
                    "dirty": bool(status),
                },
                "tool_manifest": tool_manifest.relative_to(staging).as_posix(),
            },
        )
        write_deterministic_archive(staging, output)
    sidecar = write_sha256_sidecar(output)
    return {
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "sha256_file": str(sidecar),
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a signed-ready offline deployment-host bundle"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "dist" / f"gpu-fault-deploy-host-{bundle_platform_id()}.tar.gz"
        ),
    )
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--tools-dir", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    options = parser.parse_args(arguments)
    try:
        result = build_bundle(
            python=_python_executable(options.python),
            output=options.output.resolve(),
            wheelhouse=(
                options.wheelhouse.resolve() if options.wheelhouse is not None else None
            ),
            tools_dir=(
                options.tools_dir.resolve() if options.tools_dir is not None else None
            ),
            allow_dirty=options.allow_dirty,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"build-deploy-host-bundle: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
