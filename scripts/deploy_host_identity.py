from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import sysconfig
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from scripts.component_wheels import component_source_digest
else:
    from component_wheels import component_source_digest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_INPUTS = (
    "LICENSE",
    "config/admin-config.example.yaml",
    "pyproject.toml",
    "requirements/build.lock",
    "requirements/deploy-host.lock",
    "scripts/build-deploy-host-bundle.py",
    "scripts/component_wheels.py",
    "scripts/deploy_host_bundle.py",
    "scripts/deploy_host_component.py",
    "scripts/deploy_host_identity.py",
    "src/gpu_fault/data/deploy-host-tools.json",
)


class DeployHostIdentityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_identity(root: Path, relative: str) -> dict[str, object]:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DeployHostIdentityError(
            f"deploy-host identity input leaves repository: {relative}"
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise DeployHostIdentityError(
            f"deploy-host identity input is missing or unsupported: {relative}"
        )
    return {
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "sha256": _sha256(path),
    }


def _tools_identity(tools_dir: Path | None) -> dict[str, object] | None:
    if tools_dir is None:
        return None
    root = tools_dir.resolve()
    if not root.is_dir():
        raise DeployHostIdentityError(f"deploy-host tools directory is missing: {root}")
    files: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DeployHostIdentityError(
                f"deploy-host tools must not contain symlinks: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "mode": f"{path.stat().st_mode & 0o777:04o}",
            "sha256": _sha256(path),
        }
    if not files:
        raise DeployHostIdentityError("deploy-host tools directory is empty")
    return {
        "file_count": len(files),
        "files": files,
        "sha256": _canonical_sha256(files),
    }


def host_compatibility() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "os": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "python_implementation": platform.python_implementation().lower(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_cache_tag": sys.implementation.cache_tag,
        "sysconfig_platform": sysconfig.get_platform(),
        "libc": {
            "name": libc_name or "unknown",
            "version": libc_version or "unknown",
        },
    }


def build_identity(
    root: Path = ROOT,
    *,
    compatibility: dict[str, Any] | None = None,
    tools_dir: Path | None = None,
) -> dict[str, Any]:
    resolved = root.resolve()
    inputs = {
        relative: _file_identity(resolved, relative) for relative in BUNDLE_INPUTS
    }
    identity: dict[str, Any] = {
        "schema_version": 1,
        "compatibility": compatibility or host_compatibility(),
        "deploy_host_module_sha256": component_source_digest("deploy_host"),
        "inputs": inputs,
        "tools": _tools_identity(tools_dir),
    }
    identity["sha256"] = _canonical_sha256(identity)
    return identity


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tools-dir", type=Path)
    options = parser.parse_args(arguments)
    try:
        value = build_identity(
            options.root,
            tools_dir=(
                options.tools_dir.expanduser().resolve()
                if options.tools_dir is not None
                else None
            ),
        )
    except (DeployHostIdentityError, OSError, RuntimeError, ValueError) as exc:
        print(f"deploy-host-identity: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
