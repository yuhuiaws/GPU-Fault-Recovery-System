from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from scripts.deploy_host_bundle import (
        DeployHostBundleError,
        extract_verified_bundle,
        sha256_file,
        validate_host_compatibility,
    )
else:
    from deploy_host_bundle import (
        DeployHostBundleError,
        extract_verified_bundle,
        sha256_file,
        validate_host_compatibility,
    )


ROOT = Path(__file__).resolve().parents[1]
STATE_NAME = "deploy-host-state.json"


class DeployHostSetupError(RuntimeError):
    pass


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=False,
        text=True,
        capture_output=capture,
    )
    if completed.returncode:
        detail = (completed.stderr or "").strip() if capture else ""
        raise DeployHostSetupError(
            f"command failed ({completed.returncode}): {arguments[0]}"
            + (f": {detail}" if detail else "")
        )
    return (completed.stdout or "").strip() if capture else ""


def verify_bundle_signature(
    archive: Path,
    *,
    signature_bundle: Path | None,
    cosign_key: str | None,
    certificate_identity: str | None,
    certificate_oidc_issuer: str | None,
    allow_unsigned: bool,
) -> None:
    if allow_unsigned:
        return
    if signature_bundle is None:
        raise DeployHostSetupError(
            "signed deployment-host setup requires --signature-bundle"
        )
    command = ["cosign", "verify-blob", "--bundle", str(signature_bundle)]
    if cosign_key:
        command.extend(["--key", cosign_key])
    else:
        if not certificate_identity or not certificate_oidc_issuer:
            raise DeployHostSetupError(
                "keyless deployment-host verification requires certificate "
                "identity and OIDC issuer"
            )
        command.extend(
            [
                "--certificate-identity",
                certificate_identity,
                "--certificate-oidc-issuer",
                certificate_oidc_issuer,
            ]
        )
    command.append(str(archive))
    _run(command, capture=True)


def validate_bundle_source(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    allow_source_mismatch: bool,
) -> dict[str, Any]:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise DeployHostSetupError("deploy-host bundle source identity is missing")
    expected_commit = str(source.get("git_commit") or "")
    expected_dirty = source.get("dirty")
    current_commit = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture=True,
    )
    current_dirty = bool(
        _run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            capture=True,
        )
    )
    if not allow_source_mismatch and (
        expected_dirty is not False
        or current_dirty
        or expected_commit != current_commit
    ):
        raise DeployHostSetupError(
            "deploy-host bundle source does not match a clean current checkout"
        )
    return {
        "expected_commit": expected_commit,
        "expected_dirty": expected_dirty,
        "actual_commit": current_commit,
        "actual_dirty": current_dirty,
        "matched": (
            expected_dirty is False
            and not current_dirty
            and expected_commit == current_commit
        ),
    }


def _python_path(venv: Path) -> Path:
    return venv / "bin/python"


def _isolated_python_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _pip_install(
    python: Path,
    *,
    requirements: Path,
    wheelhouse: Path | None = None,
) -> None:
    command = [str(python), "-m", "pip", "install"]
    if wheelhouse is not None:
        command.extend(["--no-index", "--find-links", str(wheelhouse)])
    command.extend(["--require-hashes", "--requirement", str(requirements)])
    _run(command, env=_isolated_python_environment())


def _install_bundled_tools(bundle_root: Path, venv: Path) -> None:
    source = bundle_root / "tools"
    if not source.is_dir():
        return
    destination = venv / "share/gpu-fault/tools"
    shutil.copytree(source, destination)
    binaries = destination / "bin"
    if binaries.is_dir():
        for path in sorted(binaries.iterdir()):
            if not path.is_file() or not os.access(path, os.X_OK):
                continue
            target = venv / "bin" / path.name
            if target.exists():
                raise DeployHostSetupError(
                    f"bundled tool conflicts with venv executable: {path.name}"
                )
            target.symlink_to(os.path.relpath(path, target.parent))


def _dependency_report(venv: Path) -> dict[str, Any]:
    env = _isolated_python_environment()
    env["PATH"] = f"{venv / 'bin'}:{env.get('PATH', '')}"
    output = _run(
        [
            str(_python_path(venv)),
            "-m",
            "gpu_fault.admin_bootstrap_dependencies",
            "--output",
            "json",
        ],
        env=env,
        capture=True,
    )
    try:
        report = json.loads(output)
    except json.JSONDecodeError as exc:
        raise DeployHostSetupError(
            "deployment-host dependency check returned invalid JSON"
        ) from exc
    if not isinstance(report, dict) or report.get("healthy") is not True:
        raise DeployHostSetupError("deployment-host dependency report is not healthy")
    return report


def _check_venv(venv: Path) -> dict[str, Any]:
    python = _python_path(venv)
    admin = venv / "bin/gpu-fault-admin"
    state_path = venv / STATE_NAME
    if not python.is_file() or not admin.is_file() or not state_path.is_file():
        raise DeployHostSetupError(f"deployment-host venv is incomplete: {venv}")
    _run([str(admin), "--help"], capture=True)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeployHostSetupError("deployment-host state is invalid") from exc
    return {
        "schema_version": 1,
        "healthy": True,
        "venv": str(venv),
        "state": state,
        "dependencies": _dependency_report(venv),
    }


def _install_from_bundle(
    bundle_root: Path,
    *,
    venv: Path,
) -> dict[str, Any]:
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    requirements = dict(manifest["requirements"])
    wheelhouse = bundle_root / "wheelhouse"
    python = _python_path(venv)
    _pip_install(
        python,
        requirements=bundle_root / str(requirements["build"]),
        wheelhouse=wheelhouse,
    )
    _pip_install(
        python,
        requirements=bundle_root / str(requirements["deploy_host"]),
        wheelhouse=wheelhouse,
    )
    project_wheel = bundle_root / str(manifest["project_wheel"])
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--no-deps",
            str(project_wheel),
        ],
        env=_isolated_python_environment(),
    )
    _install_bundled_tools(bundle_root, venv)
    return manifest


def _install_online(
    *,
    repo_root: Path,
    venv: Path,
) -> dict[str, Any]:
    python = _python_path(venv)
    for name in ("build.lock", "deploy-host.lock"):
        _pip_install(
            python,
            requirements=repo_root / "requirements" / name,
        )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "--editable",
            str(repo_root),
        ],
        env=_isolated_python_environment(),
    )
    commit = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture=True,
    )
    dirty = bool(
        _run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            capture=True,
        )
    )
    return {
        "project_version": _run(
            [
                str(python),
                "-c",
                "import importlib.metadata as m; "
                "print(m.version('gpu-fault-control-plane'))",
            ],
            capture=True,
        ),
        "source": {"git_commit": commit, "dirty": dirty},
    }


def _activate_venv(version: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.previous")
    next_link = target.with_name(f".{target.name}.next")
    if backup.is_symlink() or backup.is_file():
        backup.unlink()
    elif backup.exists():
        shutil.rmtree(backup)
    next_link.unlink(missing_ok=True)
    next_link.symlink_to(
        os.path.relpath(version, target.parent),
        target_is_directory=True,
    )
    moved_directory = target.exists() and not target.is_symlink()
    if moved_directory:
        os.replace(target, backup)
    try:
        os.replace(next_link, target)
    except Exception:
        next_link.unlink(missing_ok=True)
        if moved_directory and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def setup_deploy_host(
    *,
    repo_root: Path,
    venv: Path,
    python: str,
    archive: Path | None,
    signature_bundle: Path | None,
    cosign_key: str | None,
    certificate_identity: str | None,
    certificate_oidc_issuer: str | None,
    allow_unsigned: bool,
    allow_network: bool,
    allow_source_mismatch: bool,
) -> dict[str, Any]:
    if archive is None and not allow_network:
        raise DeployHostSetupError(
            "provide a signed --bundle or explicitly set --allow-network"
        )
    if archive is not None and allow_network:
        raise DeployHostSetupError(
            "--bundle and --allow-network are mutually exclusive"
        )
    if allow_source_mismatch and not allow_unsigned:
        raise DeployHostSetupError(
            "--allow-source-mismatch is only valid with --allow-unsigned"
        )
    if Path(sys.executable).resolve().is_relative_to(venv.resolve()):
        raise DeployHostSetupError(
            "run setup with system Python, not the venv being replaced"
        )
    archive_sha = sha256_file(archive) if archive is not None else None
    if archive is not None:
        verify_bundle_signature(
            archive,
            signature_bundle=signature_bundle,
            cosign_key=cosign_key,
            certificate_identity=certificate_identity,
            certificate_oidc_issuer=certificate_oidc_issuer,
            allow_unsigned=allow_unsigned,
        )
        if venv.is_dir():
            state_path = venv / STATE_NAME
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("bundle_sha256") == archive_sha:
                    validate_host_compatibility(
                        {"compatibility": state.get("compatibility")}
                    )
                    validate_bundle_source(
                        {"source": state.get("source")},
                        repo_root=repo_root,
                        allow_source_mismatch=allow_source_mismatch,
                    )
                    try:
                        result = _check_venv(venv)
                    except DeployHostSetupError as exc:
                        if "deployment-host venv is incomplete" not in str(exc):
                            raise
                    else:
                        result["reused"] = True
                        return result
    versions = venv.parent / f".{venv.name}.versions"
    versions.mkdir(mode=0o700, parents=True, exist_ok=True)
    if archive_sha is not None:
        staged = versions / f"bundle-{archive_sha[:24]}"
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir()
    else:
        staged = Path(tempfile.mkdtemp(prefix="online-", dir=versions))
    try:
        _run([python, "-m", "venv", str(staged)])
        with tempfile.TemporaryDirectory(
            prefix=f".{venv.name}-bundle-",
            dir=venv.parent,
        ) as directory:
            if archive is not None:
                bundle_root = extract_verified_bundle(
                    archive,
                    Path(directory),
                )
                bundle_manifest = json.loads(
                    (bundle_root / "manifest.json").read_text(encoding="utf-8")
                )
                compatibility = validate_host_compatibility(bundle_manifest)
                source_binding = validate_bundle_source(
                    bundle_manifest,
                    repo_root=repo_root,
                    allow_source_mismatch=allow_source_mismatch,
                )
                metadata = _install_from_bundle(bundle_root, venv=staged)
                mode = "bundle"
            else:
                compatibility = None
                source_binding = None
                metadata = _install_online(repo_root=repo_root, venv=staged)
                mode = "online"
        dependencies = _dependency_report(staged)
        state = {
            "schema_version": 1,
            "mode": mode,
            "bundle_sha256": archive_sha,
            "compatibility": compatibility,
            "platform_id": metadata.get("platform_id"),
            "project_version": metadata.get("project_version"),
            "source": metadata.get("source"),
            "source_binding": source_binding,
            "dependencies": dependencies,
        }
        state_path = staged / STATE_NAME
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state_path.chmod(0o600)
        _activate_venv(staged, venv)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    result = _check_venv(venv)
    result["reused"] = False
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or validate the GPU fault deployment-host venv"
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--signature-bundle", type=Path)
    parser.add_argument("--cosign-key")
    parser.add_argument("--certificate-identity")
    parser.add_argument("--certificate-oidc-issuer")
    parser.add_argument("--allow-unsigned", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    repo_root = options.repo_root.expanduser().resolve()
    venv = options.venv.expanduser()
    if not venv.is_absolute():
        venv = repo_root / venv
    venv = venv.absolute()
    try:
        result = (
            _check_venv(venv)
            if options.check
            else setup_deploy_host(
                repo_root=repo_root,
                venv=venv,
                python=options.python,
                archive=(
                    options.bundle.expanduser().resolve()
                    if options.bundle is not None
                    else None
                ),
                signature_bundle=(
                    options.signature_bundle.expanduser().resolve()
                    if options.signature_bundle is not None
                    else None
                ),
                cosign_key=options.cosign_key,
                certificate_identity=options.certificate_identity,
                certificate_oidc_issuer=options.certificate_oidc_issuer,
                allow_unsigned=options.allow_unsigned,
                allow_network=options.allow_network,
                allow_source_mismatch=options.allow_source_mismatch,
            )
        )
    except (
        DeployHostBundleError,
        DeployHostSetupError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"setup-deploy-host: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
