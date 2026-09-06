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
        dependency_identity,
        extract_verified_bundle,
        sha256_file,
        validate_host_compatibility,
    )
else:
    from deploy_host_bundle import (
        DeployHostBundleError,
        dependency_identity,
        extract_verified_bundle,
        sha256_file,
        validate_host_compatibility,
    )


ROOT = Path(__file__).resolve().parents[1]
STATE_NAME = "deploy-host-state.json"
DEPENDENCY_STATE_NAME = "deploy-host-dependency-state.json"
DEPENDENCY_ENTRYPOINTS = ("ruff",)


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
    payload_identity = str(source.get("payload_identity_sha256") or "")
    if payload_identity:
        current_dirty = bool(
            _run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repo_root,
                capture=True,
            )
        )
        output = _run(
            [
                sys.executable,
                str(repo_root / "scripts/deploy_host_identity.py"),
                "--root",
                str(repo_root),
            ],
            cwd=repo_root,
            capture=True,
        )
        try:
            actual_identity = str(json.loads(output)["sha256"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DeployHostSetupError(
                "current deploy-host payload identity is invalid"
            ) from exc
        matched = (
            len(payload_identity) == 64
            and not current_dirty
            and payload_identity == actual_identity
        )
        if not allow_source_mismatch and not matched:
            raise DeployHostSetupError(
                "deploy-host bundle payload does not match a clean current checkout"
            )
        return {
            "expected_payload_identity_sha256": payload_identity,
            "actual_payload_identity_sha256": actual_identity,
            "actual_dirty": current_dirty,
            "matched": matched,
        }
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


def install_admin_config_template(source: Path, venv: Path) -> Path:
    if not source.is_file():
        raise DeployHostSetupError(
            f"administrator config template is missing: {source}"
        )
    destination = venv / "share/gpu-fault/admin-config.example.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o644)
    return destination


def _dependency_report(
    venv: Path,
    *,
    dependency_venv: Path | None = None,
) -> dict[str, Any]:
    env = _isolated_python_environment()
    paths = [str(venv / "bin")]
    if dependency_venv is not None:
        paths.append(str(dependency_venv / "bin"))
    paths.append(env.get("PATH", ""))
    env["PATH"] = ":".join(paths)
    output = _run(
        [
            str(_python_path(venv)),
            "-m",
            "gpu_fault.admin.bootstrap_dependencies",
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


def deploy_host_project_report(
    venv: Path,
    state: Mapping[str, Any],
) -> dict[str, str] | None:
    distribution = str(state.get("project_distribution") or "")
    expected_digest = str(state.get("project_module_digest") or "")
    if not distribution and not expected_digest:
        return None
    if not distribution or len(expected_digest) != 64:
        raise DeployHostSetupError(f"deployment-host venv is incomplete: {venv}")
    output = _run(
        [
            str(_python_path(venv)),
            "-c",
            (
                "import importlib.metadata as m,json;"
                "from gpu_fault import module_digest;"
                f"print(json.dumps({{'distribution':{distribution!r},"
                f"'version':m.version({distribution!r}),"
                "'module_digest':module_digest()}))"
            ),
        ],
        env=_isolated_python_environment(),
        capture=True,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise DeployHostSetupError(
            "deployment-host project identity is invalid"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("distribution") != distribution
        or value.get("module_digest") != expected_digest
    ):
        raise DeployHostSetupError(
            "deployment-host project identity does not match bundle"
        )
    return {key: str(item) for key, item in value.items()}


def check_deploy_host_venv(venv: Path) -> dict[str, Any]:
    python = _python_path(venv)
    admin = venv / "bin/gpu-fault-admin"
    state_path = venv / STATE_NAME
    if not python.is_file() or not admin.is_file() or not state_path.is_file():
        raise DeployHostSetupError(f"deployment-host venv is incomplete: {venv}")
    admin_config_template = venv / "share/gpu-fault/admin-config.example.yaml"
    if not admin_config_template.is_file():
        raise DeployHostSetupError(
            f"deployment-host administrator config template is missing: {venv}"
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeployHostSetupError("deployment-host state is invalid") from exc
    dependency_path = state.get("dependency_venv")
    dependency_venv = Path(str(dependency_path)).resolve() if dependency_path else None
    if dependency_venv is not None and not dependency_venv.is_dir():
        raise DeployHostSetupError(f"deployment-host venv is incomplete: {venv}")
    dependency_identity_value = state.get("dependency_identity_sha256")
    if dependency_venv is not None:
        dependency_identity = str(dependency_identity_value or "")
        if len(dependency_identity) != 64:
            raise DeployHostSetupError(f"deployment-host venv is incomplete: {venv}")
        _check_dependency_venv(dependency_venv, dependency_identity)
        for name in DEPENDENCY_ENTRYPOINTS:
            source = dependency_venv / "bin" / name
            target = venv / "bin" / name
            if (
                not source.is_file()
                or not target.is_symlink()
                or target.resolve() != source.resolve()
            ):
                raise DeployHostSetupError(
                    f"deployment-host venv is incomplete: {venv}"
                )
    elif dependency_identity_value is not None:
        raise DeployHostSetupError(f"deployment-host venv is incomplete: {venv}")
    _run(
        [str(admin), "--help"],
        env=_isolated_python_environment(),
        capture=True,
    )
    return {
        "schema_version": 1,
        "healthy": True,
        "venv": str(venv),
        "state": state,
        "admin_config_template": str(admin_config_template),
        "project": deploy_host_project_report(venv, state),
        "dependencies": _dependency_report(
            venv,
            dependency_venv=dependency_venv,
        ),
    }


def _venv_site_paths(venv: Path) -> tuple[Path, ...]:
    output = _run(
        [
            str(_python_path(venv)),
            "-c",
            (
                "import json,sysconfig;"
                "paths=sysconfig.get_paths();"
                "print(json.dumps([paths['purelib'],paths['platlib']]))"
            ),
        ],
        capture=True,
    )
    values = json.loads(output)
    if not isinstance(values, list) or not values:
        raise DeployHostSetupError("deployment-host venv site paths are invalid")
    return tuple(dict.fromkeys(Path(str(value)) for value in values))


def install_dependency_entrypoints(venv: Path, dependency_venv: Path) -> None:
    for name in DEPENDENCY_ENTRYPOINTS:
        source = dependency_venv / "bin" / name
        target = venv / "bin" / name
        if not source.is_file():
            raise DeployHostSetupError(
                f"deployment-host dependency entrypoint is missing: {source}"
            )
        if target.is_symlink():
            if target.resolve() == source.resolve():
                continue
            raise DeployHostSetupError(
                f"deployment-host dependency entrypoint conflicts: {target}"
            )
        if target.exists():
            raise DeployHostSetupError(
                f"deployment-host dependency entrypoint conflicts: {target}"
            )
        target.symlink_to(os.path.relpath(source, target.parent))


def _check_dependency_venv(venv: Path, expected_identity: str) -> None:
    state_path = venv / DEPENDENCY_STATE_NAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeployHostSetupError(
            f"deployment-host dependency venv is incomplete: {venv}"
        ) from exc
    if state.get("dependency_identity_sha256") != expected_identity:
        raise DeployHostSetupError(
            "deployment-host dependency venv identity does not match"
        )
    _run(
        [
            str(_python_path(venv)),
            "-c",
            (
                "import importlib.metadata as m;"
                "names=('build','boto3','kubernetes','psycopg','pytest','ruff','PyYAML');"
                "print(','.join(m.version(name) for name in names))"
            ),
        ],
        capture=True,
    )


def _ensure_dependency_venv(
    bundle_root: Path,
    *,
    target_venv: Path,
    base_python: str,
    manifest: Mapping[str, Any],
) -> tuple[Path, bool]:
    expected = str(manifest.get("dependency_identity_sha256") or "")
    actual = dependency_identity(
        bundle_root,
        dict(manifest["compatibility"]),
    )
    if len(expected) != 64 or expected != actual:
        raise DeployHostSetupError(
            "deploy-host bundle dependency identity does not match"
        )
    versions = target_venv.parent / f".{target_venv.name}.dependencies"
    versions.mkdir(mode=0o700, parents=True, exist_ok=True)
    dependency_venv = versions / expected[:24]
    if dependency_venv.is_dir():
        _check_dependency_venv(dependency_venv, expected)
        return dependency_venv, True

    staged = Path(tempfile.mkdtemp(prefix=".dependency-", dir=versions))
    try:
        _run([base_python, "-m", "venv", str(staged)])
        requirements = dict(manifest["requirements"])
        wheelhouse = bundle_root / "wheelhouse"
        for name in ("build", "deploy_host"):
            _pip_install(
                _python_path(staged),
                requirements=bundle_root / str(requirements[name]),
                wheelhouse=wheelhouse,
            )
        state_path = staged / DEPENDENCY_STATE_NAME
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dependency_identity_sha256": expected,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        state_path.chmod(0o600)
        _check_dependency_venv(staged, expected)
        try:
            os.replace(staged, dependency_venv)
        except OSError:
            if not dependency_venv.is_dir():
                raise
            shutil.rmtree(staged, ignore_errors=True)
            _check_dependency_venv(dependency_venv, expected)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return dependency_venv, False


def install_from_bundle(
    bundle_root: Path,
    *,
    venv: Path,
    dependency_venv: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    requirements = dict(manifest["requirements"])
    wheelhouse = bundle_root / "wheelhouse"
    python = _python_path(venv)
    if dependency_venv is None:
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
    else:
        dependency_paths = _venv_site_paths(dependency_venv)
        for site_path in _venv_site_paths(venv):
            site_path.mkdir(parents=True, exist_ok=True)
            (site_path / "gpu-fault-deploy-host-dependencies.pth").write_text(
                "\n".join(str(path) for path in dependency_paths) + "\n",
                encoding="utf-8",
            )
        install_dependency_entrypoints(venv, dependency_venv)
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
    install_admin_config_template(
        bundle_root / str(manifest["admin_config_template"]),
        venv,
    )
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
    install_admin_config_template(
        repo_root / "config/admin-config.example.yaml",
        venv,
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


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def activate_venv(version: Path, target: Path) -> Path | None:
    backup = target.with_name(f".{target.name}.previous")
    next_link = target.with_name(f".{target.name}.next")
    _remove_path(backup)
    next_link.unlink(missing_ok=True)
    next_link.symlink_to(
        os.path.relpath(version, target.parent),
        target_is_directory=True,
    )
    had_target = target.is_symlink() or target.exists()
    if had_target:
        os.replace(target, backup)
    try:
        os.replace(next_link, target)
    except Exception:
        next_link.unlink(missing_ok=True)
        if had_target and (backup.is_symlink() or backup.exists()):
            os.replace(backup, target)
        raise
    return backup if had_target else None


def restore_venv_activation(target: Path, backup: Path | None) -> None:
    _remove_path(target)
    if backup is not None and (backup.is_symlink() or backup.exists()):
        os.replace(backup, target)


def finalize_venv_activation(backup: Path | None) -> None:
    if backup is not None:
        _remove_path(backup)


def prune_venv_versions(venv: Path, *, lock_fd: int) -> tuple[Path, ...]:
    """Delete installed venv versions nothing can still activate.

    Every deploy-host change installs a fresh tree under ``.<venv>.versions`` and
    nothing removed the old ones, so a long-lived state directory reached 12 GB.
    Kept: the version the symlink points at, the activation backup a crashed
    setup may have left (the only way back), and the newest
    ``GPU_FAULT_DEPLOY_HOST_VENV_VERSIONS_RETAINED`` (3) by modification time.
    The dependency venv named by the state file needs no protection: it lives
    under ``.<venv>.dependencies`` and is never a candidate here.

    ``lock_fd`` is the site operation lock the caller holds, and it is required:
    a version another process is installing into cannot be told from one it
    abandoned, so nothing here may run unlocked. The hand-run
    ``scripts/setup-deploy-host.sh`` has no lock to give and therefore never
    prunes -- the deploy does, under the lock it took before classifying.
    """

    try:
        os.fstat(lock_fd)
    except OSError as exc:
        raise DeployHostSetupError(
            "deploy-host venv versions are prunable only while the site operation "
            "lock is held"
        ) from exc
    versions = venv.parent / f".{venv.name}.versions"
    if not versions.is_dir():
        return ()
    raw = os.getenv("GPU_FAULT_DEPLOY_HOST_VENV_VERSIONS_RETAINED", "").strip()
    retained = 3
    if raw:
        try:
            retained = int(raw)
        except ValueError as exc:
            raise DeployHostSetupError(
                "GPU_FAULT_DEPLOY_HOST_VENV_VERSIONS_RETAINED must be a positive "
                "integer"
            ) from exc
        if retained < 1:
            raise DeployHostSetupError(
                "GPU_FAULT_DEPLOY_HOST_VENV_VERSIONS_RETAINED must be a positive "
                f"integer, not {raw}"
            )
    keep: set[Path] = set()
    for link in (venv, venv.with_name(f".{venv.name}.previous")):
        if not (link.is_symlink() or link.exists()):
            continue
        try:
            keep.add(link.resolve())
        except OSError:
            continue
    candidates = sorted(
        (
            path
            for path in versions.iterdir()
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    removed: list[Path] = []
    for position, candidate in enumerate(candidates):
        if position < retained or candidate in keep or candidate.resolve() in keep:
            continue
        shutil.rmtree(candidate, ignore_errors=True)
        removed.append(candidate)
    return tuple(removed)


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
                    dependency_path = state.get("dependency_venv")
                    if dependency_path:
                        install_dependency_entrypoints(
                            venv,
                            Path(str(dependency_path)).resolve(),
                        )
                    try:
                        result = check_deploy_host_venv(venv)
                    except DeployHostSetupError as exc:
                        if "deployment-host venv is incomplete" not in str(exc):
                            raise
                    else:
                        result["reused"] = True
                        return result
    versions = venv.parent / f".{venv.name}.versions"
    versions.mkdir(mode=0o700, parents=True, exist_ok=True)
    if archive_sha is not None:
        canonical = versions / f"bundle-{archive_sha[:24]}"
        current = venv.resolve() if venv.is_symlink() or venv.exists() else None
        if canonical.exists() and current == canonical.resolve():
            staged = Path(
                tempfile.mkdtemp(
                    prefix=f".bundle-{archive_sha[:24]}-",
                    dir=versions,
                )
            )
        else:
            shutil.rmtree(canonical, ignore_errors=True)
            canonical.mkdir()
            staged = canonical
    else:
        staged = Path(tempfile.mkdtemp(prefix="online-", dir=versions))
    activation_backup: Path | None = None
    activated = False
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
                dependency_venv = None
                dependency_reused = None
                if bundle_manifest.get("dependency_identity_sha256"):
                    dependency_venv, dependency_reused = _ensure_dependency_venv(
                        bundle_root,
                        target_venv=venv,
                        base_python=python,
                        manifest=bundle_manifest,
                    )
                metadata = install_from_bundle(
                    bundle_root,
                    venv=staged,
                    dependency_venv=dependency_venv,
                )
                mode = "bundle"
            else:
                compatibility = None
                source_binding = None
                dependency_venv = None
                dependency_reused = None
                metadata = _install_online(repo_root=repo_root, venv=staged)
                mode = "online"
        dependencies = _dependency_report(
            staged,
            dependency_venv=dependency_venv,
        )
        state = {
            "schema_version": 1,
            "mode": mode,
            "bundle_sha256": archive_sha,
            "compatibility": compatibility,
            "platform_id": metadata.get("platform_id"),
            "project_version": metadata.get("project_version"),
            "project_distribution": metadata.get("project_distribution"),
            "project_module_digest": metadata.get("project_module_digest"),
            "source": metadata.get("source"),
            "source_binding": source_binding,
            "dependency_identity_sha256": metadata.get("dependency_identity_sha256"),
            "dependency_venv": (
                str(dependency_venv) if dependency_venv is not None else None
            ),
            "dependency_reused": dependency_reused,
            "dependencies": dependencies,
        }
        state_path = staged / STATE_NAME
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state_path.chmod(0o600)
        check_deploy_host_venv(staged)
        activation_backup = activate_venv(staged, venv)
        activated = True
        result = check_deploy_host_venv(venv)
    except Exception:
        if activated:
            restore_venv_activation(venv, activation_backup)
        shutil.rmtree(staged, ignore_errors=True)
        raise
    finalize_venv_activation(activation_backup)
    # Versions are not pruned here. This entry point is also hand-run without any
    # site lock, and deleting a tree a concurrent install is writing into is the
    # failure that costs more than the disk: ``staging_deploy`` prunes instead,
    # under the lock it holds across the whole deploy.
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
            check_deploy_host_venv(venv)
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
