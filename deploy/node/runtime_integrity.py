"""Validate immutable node environments without importing their application code.

Run with the selected host Python's -I -S flags. Site initialization and .pth
execution must not precede validation of an existing environment.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import marshal
import os
import platform
import re
import shlex
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

BOOTSTRAP_PACKAGES = frozenset({"pip", "setuptools", "wheel"})
PROJECT = "gpu-fault-node-runtime"
LEGACY_PROJECT = "gpu-fault-control-plane"
LAYER_PTH = "gpu-fault-node-dependencies.pth"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def base_python() -> Path:
    return Path(getattr(sys, "_base_executable", None) or sys.executable).resolve(
        strict=True
    )


def dependency_identity(lock: Path) -> str:
    identity = {
        "schema_version": 2,
        "lock_sha256": digest(lock),
        "implementation": sys.implementation.name,
        "python": list(sys.version_info[:3]),
        "platform": sysconfig.get_platform(),
        "abi": sysconfig.get_config_var("SOABI"),
        "libc": platform.libc_ver(),
        "python_sha256": digest(base_python()),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def site_path(venv: Path) -> Path:
    return Path(
        sysconfig.get_path(
            "purelib", scheme="venv", vars={"base": str(venv), "platbase": str(venv)}
        )
    )


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def verify_bytecode(path: Path, recorded: set[Path]) -> None:
    source = Path(importlib.util.source_from_cache(str(path)))
    if source not in recorded or not source.is_file():
        raise ValueError("bytecode has no recorded source")
    optimization = re.search(r"\.opt-([012])\.pyc$", path.name)
    level = int(optimization[1]) if optimization else 0
    raw = path.read_bytes()
    if raw[:4] != importlib.util.MAGIC_NUMBER or marshal.loads(raw[16:]) != compile(
        source.read_bytes(), str(source), "exec", dont_inherit=True, optimize=level
    ):
        raise ValueError("installed bytecode differs from its source")


def verify_records(
    venv: Path, *, layer: Path | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    site = site_path(venv)
    versions: dict[str, str] = {}
    records: dict[str, str] = {}
    files: set[Path] = set()
    bytecode: set[Path] = set()
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = canonical_name(str(distribution.metadata["Name"]))
        if name in versions:
            raise ValueError("duplicate installed distribution")
        versions[name] = distribution.version
        record = distribution.read_text("RECORD")
        if not record:
            raise ValueError("installed distribution has no RECORD")
        records[name] = hashlib.sha256(record.encode()).hexdigest()
        for row in csv.reader(io.StringIO(record)):
            if len(row) != 3:
                raise ValueError("invalid RECORD row")
            relative, encoded, size = row
            path = Path(os.path.abspath(str(distribution.locate_file(relative))))
            resolved = path.resolve(strict=True)
            if (
                not resolved.is_relative_to(venv)
                or path.is_symlink()
                or not path.is_file()
            ):
                raise ValueError("RECORD escapes its environment")
            if path != resolved:
                raise ValueError("symlink in installed distribution")
            files.add(resolved)
            if not encoded:
                if path.suffix == ".pyc":
                    bytecode.add(resolved)
                elif path.name != "RECORD" or not path.parent.name.endswith(
                    ".dist-info"
                ):
                    raise ValueError("unhashed installed file")
                continue
            algorithm, expected = encoded.split("=", 1)
            if algorithm not in {"sha256", "sha384", "sha512"}:
                raise ValueError("weak or unknown RECORD digest")
            with path.open("rb") as stream:
                actual = (
                    base64.urlsafe_b64encode(
                        hashlib.file_digest(stream, algorithm).digest()
                    )
                    .decode()
                    .rstrip("=")
                )
            if actual != expected or path.stat().st_size != int(size):
                raise ValueError("installed file differs from RECORD")
    pth = site / LAYER_PTH
    if layer is not None:
        if pth.is_symlink() or pth.read_text() != str(site_path(layer)) + "\n":
            raise ValueError("dependency layer reference differs")
        files.add(pth)
    for path in site.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in installed site-packages")
        if not path.is_file() or path in files:
            continue
        if path.suffix == ".pyc":
            bytecode.add(path)
        else:
            raise ValueError("unrecorded installed file")
    for path in bytecode:
        verify_bytecode(path, files)
    return versions, records


def locked_versions(lock: Path, trusted_site: Path) -> dict[str, str]:
    # pip is required by the installer and its files have already been verified.
    # Reuse its requirement/marker parsers, without running site or .pth files.
    sys.path.insert(0, str(trusted_site))
    from pip._internal.req.req_file import break_args_options, preprocess
    from pip._vendor.packaging.requirements import Requirement

    content = lock.read_text()
    if "${" in content:
        raise ValueError("environment expansion is forbidden in node locks")
    versions: dict[str, str] = {}
    for _, line in preprocess(content):
        requirement, options = break_args_options(line)
        hashes = shlex.split(options)
        if not hashes or any(
            not re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", value) for value in hashes
        ):
            raise ValueError("node lock requires only pinned hashes")
        parsed = Requirement(requirement)
        pins = list(parsed.specifier)
        if (
            parsed.url
            or parsed.extras
            or len(pins) != 1
            or pins[0].operator != "=="
            or "*" in pins[0].version
        ):
            raise ValueError("node lock must contain exact package versions")
        if parsed.marker is not None and not parsed.marker.evaluate():
            continue
        name = canonical_name(parsed.name)
        if name in versions:
            raise ValueError("duplicate applicable node lock requirement")
        versions[name] = pins[0].version
    return versions


def project_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise ValueError("candidate wheel metadata is ambiguous")
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    if canonical_name(str(metadata["Name"])) != PROJECT:
        raise ValueError("candidate is not the node runtime wheel")
    return str(metadata["Version"])


def validate_wheelhouse(
    directory: Path, expected: str, *, lock: Path | None, bundle: Path | None
) -> None:
    inventory = directory / "inventory.json"
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected)
        or inventory.is_symlink()
        or inventory.stat().st_size > 1024 * 1024
        or digest(inventory) != expected
    ):
        raise ValueError("offline node wheelhouse inventory digest differs")
    data = json.loads(inventory.read_bytes())
    expected_platform = data.get("platform") or {}
    libc, version = platform.libc_ver()
    if (
        data.get("schema_version") != 1
        or expected_platform.get("os") != sys.platform
        or expected_platform.get("machine") != platform.machine()
        or expected_platform.get("implementation") != sys.implementation.name
        or expected_platform.get("python") != list(sys.version_info[:2])
        or libc != "glibc"
        or tuple(int(part) for part in version.split(".")[:2])
        < tuple(expected_platform.get("minimum_glibc") or (999, 0))
    ):
        raise ValueError("offline node wheelhouse platform is incompatible")
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("offline node wheelhouse inventory is empty")
    if {path.name for path in directory.iterdir()} != {*files, "inventory.json"}:
        raise ValueError("offline node wheelhouse file set differs")
    for name, entry in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(entry, dict)
        ):
            raise ValueError("offline node wheelhouse file entry is invalid")
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry.get("size")
            or digest(path) != entry.get("sha256")
        ):
            raise ValueError(f"offline node wheelhouse file differs: {name}")
    locks = data.get("locks") or {}
    if set(locks) != {"node-runtime.lock", "node-tools.lock"}:
        raise ValueError("offline node wheelhouse lock set differs")
    for name, expected_lock in locks.items():
        if digest(directory / name) != expected_lock:
            raise ValueError("offline node wheelhouse lock digest differs")
    if bundle is not None:
        with tarfile.open(bundle, "r:gz") as archive:
            for name, expected_lock in locks.items():
                members = [
                    item
                    for item in archive.getmembers()
                    if item.name.endswith("/requirements/" + name) and item.isfile()
                ]
                if len(members) != 1 or members[0].size > 1024 * 1024:
                    raise ValueError(
                        "candidate bundle dependency lock is missing or ambiguous"
                    )
                stream = archive.extractfile(members[0])
                if (
                    stream is None
                    or hashlib.sha256(stream.read()).hexdigest() != expected_lock
                ):
                    raise ValueError("wheelhouse does not match candidate bundle locks")
    elif lock is None or digest(lock) != locks["node-runtime.lock"]:
        raise ValueError("wheelhouse does not match candidate runtime lock")


def validate_interpreter(venv: Path) -> None:
    venv = venv.resolve(strict=True)
    interpreter = venv / "bin/python"
    trusted = base_python()
    if digest(interpreter.resolve(strict=True)) != digest(trusted):
        raise ValueError("environment interpreter differs from selected Python")
    # venv records the launch directory, which can precede the final binary in
    # an alternatives/symlink chain. Accept only directories on that same chain.
    homes = {trusted.parent}
    visited: set[Path] = set()
    link = interpreter
    while True:
        if link in visited:
            raise ValueError("cyclic environment interpreter reference")
        visited.add(link)
        if not link.is_relative_to(venv):
            homes.add(link.parent.resolve())
        if not link.is_symlink():
            break
        target = Path(os.readlink(link))
        link = Path(
            os.path.abspath(target if target.is_absolute() else link.parent / target)
        )
    config = configparser.ConfigParser()
    config.read_string("[venv]\n" + (venv / "pyvenv.cfg").read_text())
    if config["venv"].get("include-system-site-packages") != "false":
        raise ValueError("system site-packages are not isolated")
    if Path(config["venv"]["home"]).resolve() not in homes or config["venv"][
        "version"
    ] != ".".join(str(value) for value in sys.version_info[:3]):
        raise ValueError("environment base interpreter configuration differs")


def validate(
    venv: Path,
    lock: Path,
    layer: Path | None,
    wheel: Path | None,
    seal: Path | None = None,
) -> str:
    venv = venv.resolve(strict=True)
    validate_interpreter(venv)
    versions, records = verify_records(venv, layer=layer)
    records_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()
    if seal is not None and seal.read_text().strip() != records_sha256:
        raise ValueError("installed RECORD seal differs")
    if layer is not None:
        expected = {PROJECT: project_version(wheel)} if wheel is not None else {}
    else:
        expected = locked_versions(lock, site_path(venv))
    actual = {
        name: version
        for name, version in versions.items()
        if name in expected or name not in BOOTSTRAP_PACKAGES
    }
    if actual != expected:
        raise ValueError("installed package versions differ from the locked closure")
    if any(versions.get(name) != version for name, version in expected.items()):
        raise ValueError("installed package version differs from its pin")
    return records_sha256


def validate_dependency_closure(sites: list[Path], versions: dict[str, str]) -> None:
    pip_site = next((site for site in sites if (site / "pip").is_dir()), None)
    if pip_site is None:
        raise ValueError("rollback environment has no verified dependency parser")
    sys.path.insert(0, str(pip_site))
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version

    for site in sites:
        for distribution in importlib.metadata.distributions(path=[str(site)]):
            for value in distribution.requires or []:
                requirement = Requirement(value)
                if requirement.marker is not None and not requirement.marker.evaluate(
                    {"extra": ""}
                ):
                    continue
                version = versions.get(canonical_name(requirement.name))
                if (
                    requirement.url
                    or version is None
                    or Version(version) not in requirement.specifier
                ):
                    raise ValueError("rollback dependency closure is inconsistent")


def validate_previous_runtime(
    root: Path, expected_artifact: str | None
) -> dict[str, str]:
    root = root.resolve(strict=True)
    current = root / "current"
    if current.is_symlink():
        slot = current.resolve(strict=True)
        if not slot.is_relative_to(root / "releases"):
            raise ValueError("rollback slot escapes the runtime releases directory")
    elif current.exists():
        raise ValueError("runtime current is not an atomic link")
    elif (root / "venv").is_dir() and not (root / "venv").is_symlink():
        slot = root
    else:
        raise ValueError("host has no rollback runtime slot")
    marker = (
        slot / "artifact.sha256" if slot != root else root / "runtime-artifact-sha256"
    )
    artifact = marker.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", artifact) or (
        expected_artifact is not None and artifact != expected_artifact
    ):
        raise ValueError("rollback artifact identity differs")
    if slot != root and not (slot / ".complete").is_file():
        raise ValueError("rollback slot is incomplete")
    venv = slot / "venv"
    validate_interpreter(venv)
    pth = site_path(venv) / LAYER_PTH
    layer: Path | None = None
    layer_versions: dict[str, str] = {}
    if pth.exists() or pth.is_symlink():
        raw = pth.read_text()
        reference = Path(raw.strip())
        if (
            pth.is_symlink()
            or not reference.is_absolute()
            or "\n" in raw.strip()
            or not reference.is_relative_to(root / "dependencies")
        ):
            raise ValueError("rollback dependency reference is invalid")
        layer = reference.parents[2]
        directory = layer.parent
        dependency = (slot / "dependency.sha256").read_text().strip()
        if (
            directory.parent != root / "dependencies"
            or directory.name != dependency
            or not re.fullmatch(r"[0-9a-f]{64}", dependency)
            or not (directory / ".complete").is_file()
            or (directory / "dependency.sha256").read_text().strip() != dependency
        ):
            raise ValueError("rollback dependency layer identity differs")
        validate_interpreter(layer)
        layer_versions, layer_records = verify_records(layer)
        layer_seal = hashlib.sha256(
            json.dumps(layer_records, sort_keys=True).encode()
        ).hexdigest()
        if (directory / "record.sha256").read_text().strip() != layer_seal:
            raise ValueError("rollback dependency RECORD seal differs")
        stored_lock = directory / "dependency.lock"
        if stored_lock.is_file():
            if dependency_identity(stored_lock) != dependency:
                raise ValueError("rollback dependency lock identity differs")
            validate(layer, stored_lock, None, None, directory / "record.sha256")
    versions, records = verify_records(venv, layer=layer)
    projects = {PROJECT, LEGACY_PROJECT}.intersection(versions)
    if len(projects) != 1:
        raise ValueError("rollback runtime distribution is ambiguous")
    project = next(iter(projects))
    seal = slot / "record.sha256"
    records_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()
    if seal.is_file():
        expected_seal = (
            records_sha256
            if (slot / "dependency.sha256").is_file()
            else records[project]
        )
        if seal.read_text().strip() != expected_seal:
            raise ValueError("rollback runtime RECORD seal differs")
    elif slot != root:
        raise ValueError("rollback runtime RECORD seal is missing")
    for name, version in layer_versions.items():
        if name in versions and name not in BOOTSTRAP_PACKAGES:
            raise ValueError("rollback distributions shadow the shared layer")
        versions.setdefault(name, version)
    sites = [site_path(venv), *([site_path(layer)] if layer is not None else [])]
    validate_dependency_closure(sites, versions)
    commands = ("gpu-fault-node-agent", "gpu-fault-restore-gpu-services")
    for command in commands:
        if not os.access(venv / "bin" / command, os.X_OK):
            raise ValueError("rollback runtime entrypoint is unavailable")
    if not (root / "installed-units.txt").read_text().strip():
        raise ValueError("rollback unit inventory is unavailable")
    # Only after file/metadata verification may the old interpreter initialize
    # site and import entrypoint modules. No command body or GPU action is run.
    code = (
        "import importlib, importlib.metadata as m; "
        f"d=m.distribution({project!r}); "
        f"names={commands!r}; "
        "entries={ep.name:ep for ep in d.entry_points if ep.group=='console_scripts'}; "
        "[importlib.import_module(entries[name].module) for name in names]"
    )
    result = subprocess.run(
        [str(venv / "bin/python"), "-I", "-B", "-c", code],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise ValueError("rollback runtime entrypoint import failed")
    return {
        "artifact_sha256": artifact,
        "record_sha256": records_sha256,
        "slot": str(slot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("identity", "site", "validate", "previous", "wheelhouse")
    )
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--layer", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--expected-artifact")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--expected-inventory")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "wheelhouse":
            validate_wheelhouse(
                args.wheelhouse,
                args.expected_inventory,
                lock=args.lock,
                bundle=args.bundle,
            )
        elif args.command == "identity":
            print(dependency_identity(args.lock))
        elif args.command == "site":
            print(site_path(args.venv))
        elif args.command == "previous":
            print(
                json.dumps(validate_previous_runtime(args.root, args.expected_artifact))
            )
        else:
            print(validate(args.venv, args.lock, args.layer, args.wheel, args.seal))
    except (
        OSError,
        ValueError,
        KeyError,
        EOFError,
        ImportError,
        configparser.Error,
        subprocess.SubprocessError,
    ) as exc:
        raise SystemExit(
            f"node environment integrity validation failed: {exc}"
        ) from None


if __name__ == "__main__":
    main()
