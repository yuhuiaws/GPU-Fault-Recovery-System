"""Prepare the isolated tool environment using pins supplied by Make."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

if TYPE_CHECKING or __package__:
    from scripts.deploy_host_bundle import sha256_file
else:
    from deploy_host_bundle import sha256_file

PACKAGE_PROBE = """\
import importlib.metadata as metadata
import json
import sys
versions = {}
for name in sys.argv[1:]:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "prefix": sys.prefix, "base_prefix": sys.base_prefix, "versions": versions
}))
"""


class SupplyChainSetupError(RuntimeError):
    pass


def run(arguments: Sequence[str], *, phase: str, timeout: int = 600) -> str:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "COSIGN_PASSWORD"):
        environment.pop(name, None)
    try:
        result = subprocess.run(
            list(arguments),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, UnicodeError, subprocess.SubprocessError):
        raise SupplyChainSetupError(f"{phase} failed") from None
    if result.returncode:
        raise SupplyChainSetupError(f"{phase} failed (exit {result.returncode})")
    return result.stdout.strip()


def installed_versions(venv: Path, packages: Sequence[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(
            run(
                [str(venv / "bin/python"), "-I", "-c", PACKAGE_PROBE, *packages],
                phase="tool environment probe",
                timeout=30,
            )
        )
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("prefix"), str)
            or not isinstance(value.get("base_prefix"), str)
            or Path(value["prefix"]).resolve() != venv.resolve()
            or Path(value["prefix"]).resolve() == Path(value["base_prefix"]).resolve()
            or not isinstance(value.get("versions"), dict)
        ):
            return None
        return dict(value["versions"])
    except (SupplyChainSetupError, ValueError):
        return None


def python_tools_ready(venv: Path, packages: dict[str, str]) -> bool:
    if installed_versions(venv, tuple(packages)) != packages:
        return False
    try:
        audit = run(
            [str(venv / "bin/pip-audit"), "--version"],
            phase="pip-audit version probe",
            timeout=30,
        )
        cyclone = run(
            [str(venv / "bin/cyclonedx-py"), "--version"],
            phase="CycloneDX version probe",
            timeout=30,
        )
        if audit != f"pip-audit {packages['pip-audit']}" or cyclone not in {
            packages["cyclonedx-bom"],
            f"cyclonedx-py {packages['cyclonedx-bom']}",
        }:
            return False
        run(
            [str(venv / "bin/python"), "-m", "pip", "check"],
            phase="tool dependency check",
            timeout=30,
        )
    except SupplyChainSetupError:
        return False
    return True


def prepare_python_tools(venv: Path, *, python: str, packages: dict[str, str]) -> bool:
    if installed_versions(venv, tuple(packages)) is None:
        run([python, "-m", "venv", str(venv)], phase="tool venv creation")
    if installed_versions(venv, tuple(packages)) is None:
        raise SupplyChainSetupError("tool Python is not isolated in the requested venv")
    if python_tools_ready(venv, packages):
        return True
    tool_python = str(venv / "bin/python")
    run(
        [
            tool_python,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-input",
            "--upgrade",
            "pip",
        ],
        phase="tool pip preparation",
    )
    run(
        [
            tool_python,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-input",
            "--force-reinstall",
            *(f"{name}=={version}" for name, version in packages.items()),
        ],
        phase="pinned tool package installation",
    )
    if not python_tools_ready(venv, packages):
        raise SupplyChainSetupError("pinned Python tools failed verification")
    return False


def matches_digest(path: Path, digest: str) -> bool:
    try:
        return path.is_file() and sha256_file(path) == digest
    except OSError:
        return False


def prepare_archive(venv: Path, *, version: str, digest: str) -> tuple[Path, bool]:
    archive = venv / "promtool.tar.gz"
    if not archive.is_symlink() and matches_digest(archive, digest):
        return archive, True
    with tempfile.TemporaryDirectory(prefix=".promtool-download-", dir=venv) as work:
        downloaded = Path(work) / "promtool.tar.gz"
        run(
            [
                "curl",
                "-sSfL",
                "--retry",
                "3",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "-o",
                str(downloaded),
                "https://github.com/prometheus/prometheus/releases/download/"
                f"v{version}/prometheus-{version}.linux-amd64.tar.gz",
            ],
            phase="pinned Prometheus archive download",
        )
        if not matches_digest(downloaded, digest):
            raise SupplyChainSetupError(
                "Prometheus archive SHA-256 differs from the Make pin"
            )
        downloaded.replace(archive)
    return archive, False


def prepare_promtool(archive: Path, *, target: Path, version: str) -> bool:
    with tarfile.open(archive, "r:gz") as source:
        try:
            member = source.getmember(f"prometheus-{version}.linux-amd64/promtool")
        except KeyError:
            raise SupplyChainSetupError(
                "pinned archive does not contain promtool"
            ) from None
        if not member.isfile():
            raise SupplyChainSetupError("pinned promtool member must be a regular file")
        stream = source.extractfile(member)
        if stream is None:
            raise SupplyChainSetupError("pinned promtool member is unreadable")
        with stream:
            digest = hashlib.file_digest(
                cast(tarfile.ExFileObject, stream), "sha256"
            ).hexdigest()
            if matches_digest(target, digest) and os.access(target, os.X_OK):
                return True
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".promtool-install-", dir=target.parent
            ) as work:
                staged = Path(work) / "promtool"
                stream.seek(0)
                with staged.open("wb") as destination:
                    shutil.copyfileobj(stream, destination)
                staged.chmod(0o755)
                staged.replace(target)
    return False


def promtool_target(candidate: str) -> Path:
    if not candidate.strip():
        raise SupplyChainSetupError("PROMTOOL must not be empty")
    if "/" not in candidate:
        found = shutil.which(candidate)
        if found is None:
            raise SupplyChainSetupError(
                "PROMTOOL must name an existing PATH tool or an explicit install path"
            )
        candidate = found
    return Path(candidate).expanduser().absolute()


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--host-venv", type=Path, required=True)
    parser.add_argument("--tools-python", type=Path, required=True)
    parser.add_argument("--pip-audit-version", required=True)
    parser.add_argument("--cyclonedx-bom-version", required=True)
    parser.add_argument("--promtool", required=True)
    parser.add_argument("--promtool-version", required=True)
    parser.add_argument("--promtool-sha256", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate pins and path isolation without preparing either environment",
    )
    options = parser.parse_args(arguments)
    try:
        packages = {
            "pip-audit": options.pip_audit_version,
            "cyclonedx-bom": options.cyclonedx_bom_version,
        }
        if (
            any(
                re.fullmatch(r"\d+\.\d+\.\d+", version) is None
                for version in (*packages.values(), options.promtool_version)
            )
            or re.fullmatch(r"[0-9a-f]{64}", options.promtool_sha256) is None
        ):
            raise SupplyChainSetupError(
                "tool versions and archive SHA-256 must be explicit pins"
            )
        venv = options.venv.expanduser().absolute()
        host = options.host_venv.expanduser().resolve()
        protected = [host]
        if sys.prefix != sys.base_prefix:
            protected.append(Path(sys.prefix).resolve())
        if any(
            venv.resolve().is_relative_to(prefix)
            or prefix.is_relative_to(venv.resolve())
            for prefix in protected
        ) or Path(sys.base_prefix).resolve().is_relative_to(venv.resolve()):
            raise SupplyChainSetupError("host and supply-chain venvs must be separate")
        if (
            options.tools_python.expanduser().absolute().parent.resolve()
            != (venv / "bin").resolve()
        ):
            raise SupplyChainSetupError(
                "SUPPLY_CHAIN_PYTHON must be in SUPPLY_CHAIN_TOOLS_VENV/bin"
            )
        target = promtool_target(options.promtool)
        if any(
            target.resolve().is_relative_to(prefix)
            or prefix.is_relative_to(target.resolve())
            for prefix in protected
        ):
            raise SupplyChainSetupError(
                "PROMTOOL must be outside the host and running Python environments"
            )
        if options.validate_only:
            print(json.dumps({"status": "validated"}, sort_keys=True))
            return 0
        venv.mkdir(parents=True, exist_ok=True)
        print(
            "setup-supply-chain-tools: preparing Python tools and Prometheus archive",
            file=sys.stderr,
            flush=True,
        )
        # Only the download overlaps pip; extraction into bin waits for both.
        # The executor also waits for an in-flight peer before returning failure.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool-setup") as pool:
            python_future = pool.submit(
                prepare_python_tools, venv, python=options.python, packages=packages
            )
            archive_future = pool.submit(
                prepare_archive,
                venv,
                version=options.promtool_version,
                digest=options.promtool_sha256,
            )
            python_reused = python_future.result()
            archive, archive_reused = archive_future.result()
        binary_reused = prepare_promtool(
            archive, target=target, version=options.promtool_version
        )
        run(
            [
                options.python,
                str(Path(__file__).resolve().with_name("check-alert-rules.py")),
                "--promtool",
                str(target),
                "--version",
                options.promtool_version,
                "--tool-only",
            ],
            phase="pinned promtool version verification",
            timeout=30,
        )
    except (SupplyChainSetupError, OSError, ValueError, tarfile.TarError) as exc:
        print(f"setup-supply-chain-tools: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ready",
                "venv": str(venv),
                "python_tools_reused": python_reused,
                "archive_reused": archive_reused,
                "promtool_reused": binary_reused,
                "promtool": str(target),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
