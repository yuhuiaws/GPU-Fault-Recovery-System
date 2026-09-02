from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import sys
import sysconfig
import tarfile
from pathlib import Path
from typing import Any, Mapping


BUNDLE_ROOT = "gpu-fault-deploy-host"
MANIFEST_NAME = "manifest.json"
BUNDLE_SCHEMA_VERSION = 1


class DeployHostBundleError(RuntimeError):
    pass


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


def bundle_platform_id(compatibility: Mapping[str, Any] | None = None) -> str:
    value = dict(compatibility or host_compatibility())
    return "-".join(
        (
            str(value["os"]),
            str(value["architecture"]),
            str(value["python_cache_tag"]),
        )
    ).replace("_", "-")


def validate_host_compatibility(manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected = manifest.get("compatibility")
    if not isinstance(expected, dict):
        raise DeployHostBundleError(
            "deploy-host bundle compatibility metadata is missing"
        )
    actual = host_compatibility()
    fields = (
        "os",
        "architecture",
        "python_implementation",
        "python_version",
        "python_cache_tag",
        "sysconfig_platform",
    )
    mismatches = [
        f"{field}={expected.get(field)!r}, host={actual.get(field)!r}"
        for field in fields
        if expected.get(field) != actual.get(field)
    ]
    expected_libc = expected.get("libc")
    actual_libc = actual.get("libc")
    if (
        isinstance(expected_libc, dict)
        and isinstance(actual_libc, dict)
        and expected_libc.get("name") != actual_libc.get("name")
    ):
        mismatches.append(
            f"libc={expected_libc.get('name')!r}, host={actual_libc.get('name')!r}"
        )
    if mismatches:
        raise DeployHostBundleError(
            "deploy-host bundle is incompatible with this host: "
            + "; ".join(mismatches)
        )
    return actual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_identity(
    root: Path,
    compatibility: Mapping[str, Any],
) -> str:
    payload = {
        "compatibility": dict(compatibility),
        "requirements": {
            name: sha256_file(root / "requirements" / name)
            for name in ("build.lock", "deploy-host.lock")
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _payload_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    ]


def write_bundle_manifest(
    root: Path,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    for path in _payload_files(root):
        path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
    files = {
        path.relative_to(root).as_posix(): {
            "mode": f"{path.stat().st_mode & 0o777:04o}",
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in _payload_files(root)
    }
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        **dict(metadata),
        "files": files,
    }
    target = root / MANIFEST_NAME
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    target.chmod(0o644)
    return target


def verify_bundle_tree(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeployHostBundleError("deploy-host bundle manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DeployHostBundleError("deploy-host bundle schema_version must be 1")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise DeployHostBundleError("deploy-host bundle file inventory is missing")
    actual = {path.relative_to(root).as_posix() for path in _payload_files(root)}
    if actual != set(expected):
        raise DeployHostBundleError("deploy-host bundle file inventory does not match")
    for relative, raw in expected.items():
        if not isinstance(relative, str) or not isinstance(raw, dict):
            raise DeployHostBundleError("deploy-host bundle file entry is invalid")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise DeployHostBundleError(
                "deploy-host bundle path leaves its root"
            ) from exc
        if not path.is_file():
            raise DeployHostBundleError(
                f"deploy-host bundle file is missing: {relative}"
            )
        if sha256_file(path) != raw.get("sha256"):
            raise DeployHostBundleError(
                f"deploy-host bundle file digest mismatch: {relative}"
            )
        if path.stat().st_size != raw.get("size"):
            raise DeployHostBundleError(
                f"deploy-host bundle file size mismatch: {relative}"
            )
        if f"{path.stat().st_mode & 0o777:04o}" != raw.get("mode"):
            raise DeployHostBundleError(
                f"deploy-host bundle file mode mismatch: {relative}"
            )
    return manifest


def write_deterministic_archive(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                paths = [root, *sorted(root.rglob("*"))]
                for path in paths:
                    relative = path.relative_to(root)
                    arcname = (
                        BUNDLE_ROOT
                        if not relative.parts
                        else f"{BUNDLE_ROOT}/{relative.as_posix()}"
                    )
                    info = archive.gettarinfo(str(path), arcname)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                    else:
                        archive.addfile(info)


def extract_verified_bundle(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or not path.parts
                    or path.parts[0] != BUNDLE_ROOT
                    or ".." in path.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise DeployHostBundleError(
                        f"unsafe deploy-host bundle member: {member.name}"
                    )
                target = (destination_root / path).resolve()
                try:
                    target.relative_to(destination_root)
                except ValueError as exc:
                    raise DeployHostBundleError(
                        f"deploy-host bundle member leaves destination: {member.name}"
                    ) from exc
            archive.extractall(destination_root, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise DeployHostBundleError("cannot extract deploy-host bundle") from exc
    root = destination_root / BUNDLE_ROOT
    verify_bundle_tree(root)
    return root


def write_sha256_sidecar(path: Path) -> Path:
    target = path.with_suffix(path.suffix + ".sha256")
    target.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    os.chmod(target, 0o644)
    return target
