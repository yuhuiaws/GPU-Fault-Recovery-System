from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence, cast

import yaml

from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK_FD_ENV,
    site_operation_lock,
)

if __package__:
    from scripts.setup_deploy_host import prune_venv_versions
    from scripts.staging_gate_caches import (
        public_release_verdict_cache,
        tool_cache_environment,
    )
    from scripts.staging_live_evidence import (
        LiveEvidenceError,
        collect_live_deploy_evidence,
        successful_source_live_matches,
    )
    from scripts.staging_state_hygiene import (
        DeployHostArtifacts,
        SigningMaterial,
        SourceCheckout,
        StagingDeployError,
        deploy_host_artifacts,
        deploy_host_wheelhouse_cache,
        private_state_file,
        prune_source_snapshots,
        public_state_file,
        record_source_deploy_state,
        restore_site_file,
        source_success_paths,
    )
else:
    from setup_deploy_host import prune_venv_versions
    from staging_gate_caches import (
        public_release_verdict_cache,
        tool_cache_environment,
    )
    from staging_live_evidence import (
        LiveEvidenceError,
        collect_live_deploy_evidence,
        successful_source_live_matches,
    )
    from staging_state_hygiene import (
        DeployHostArtifacts,
        SigningMaterial,
        SourceCheckout,
        StagingDeployError,
        deploy_host_artifacts,
        deploy_host_wheelhouse_cache,
        private_state_file,
        prune_source_snapshots,
        public_state_file,
        record_source_deploy_state,
        restore_site_file,
        source_success_paths,
    )


ROOT = Path(__file__).resolve().parents[1]


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
    pass_fds: tuple[int, ...] = (),
) -> str:
    print("+ " + " ".join(arguments), file=sys.stderr, flush=True)
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=False,
        text=True,
        capture_output=capture,
        pass_fds=pass_fds,
    )
    if completed.returncode:
        detail = (completed.stderr or "").strip() if capture else ""
        raise StagingDeployError(
            f"command failed ({completed.returncode}): {arguments[0]}"
            + (f": {detail}" if detail else "")
        )
    return (completed.stdout or "").strip() if capture else ""


def _git_output(repository_root: Path, *arguments: str) -> str:
    return _run(
        ["git", *arguments],
        cwd=repository_root,
        capture=True,
    )


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    print("+ git " + " ".join(arguments), file=sys.stderr, flush=True)
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise StagingDeployError(
            f"git command failed ({completed.returncode}): "
            + completed.stderr.decode(errors="replace").strip()
        )
    return completed.stdout


def _repository_head(repository_root: Path) -> str:
    if sys.version_info[:2] != (3, 12):
        raise StagingDeployError("staging deploy requires Python 3.12")
    expected_root = Path(
        _git_output(repository_root, "rev-parse", "--show-toplevel")
    ).resolve()
    if expected_root != repository_root.resolve():
        raise StagingDeployError(
            f"repository root mismatch: expected {expected_root}, got {repository_root}"
        )
    return _git_output(repository_root, "rev-parse", "HEAD")


def validate_source_checkout(
    repository_root: Path,
    *,
    verdict_cache: Path | None = None,
) -> None:
    """Refuse to deploy a tree that carries live-environment identity.

    ``verdict_cache`` lets the second call in a deploy -- the prepared snapshot,
    whose content is a copy of the live tree the first call already cleared --
    prove it is scanning identical bytes instead of scanning them again. The gate
    still walks and hashes every public file each time; only the pattern matching
    is skipped, and only on an exact content match.
    """

    _run(
        [
            sys.executable,
            str(repository_root / "scripts/check-public-release.py"),
            "--root",
            str(repository_root),
            *(("--verdict-cache", str(verdict_cache)) if verdict_cache else ()),
        ],
        cwd=repository_root,
    )


def _worktree_status(repository_root: Path) -> str:
    return _git_output(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=normal",
    )


def _untracked_files(repository_root: Path) -> tuple[Path, ...]:
    raw = _git_bytes(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    return tuple(Path(os.fsdecode(value)) for value in raw.split(b"\0") if value)


def _tracked_files(repository_root: Path) -> tuple[Path, ...]:
    raw = _git_bytes(repository_root, "ls-files", "-z")
    return tuple(Path(os.fsdecode(value)) for value in raw.split(b"\0") if value)


def _prepared_tree_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in sorted(
        _tracked_files(repository_root),
        key=lambda value: value.as_posix(),
    ):
        if relative.is_absolute() or ".." in relative.parts:
            raise StagingDeployError(f"prepared source leaves repository: {relative}")
        source = repository_root.resolve() / relative
        digest.update(b"\0path\0")
        digest.update(relative.as_posix().encode())
        if source.is_symlink():
            digest.update(b"\0symlink\0")
            digest.update(os.readlink(source).encode())
            continue
        if not source.is_file():
            raise StagingDeployError(f"prepared source file is missing: {relative}")
        digest.update(b"\0file\0")
        digest.update(f"{source.stat().st_mode & 0o777:04o}".encode())
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(
    repository_root: Path,
    *,
    head: str,
) -> tuple[str, bytes, tuple[Path, ...], tuple[Path, ...]]:
    diff = _git_bytes(repository_root, "diff", "--binary", "HEAD")
    tracked = _tracked_files(repository_root)
    untracked = _untracked_files(repository_root)
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(b"\0diff\0")
    digest.update(diff)
    for relative in sorted(tracked, key=lambda value: value.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise StagingDeployError(f"tracked source leaves repository: {relative}")
        source = repository_root.resolve() / relative
        digest.update(b"\0tracked-mode\0")
        digest.update(relative.as_posix().encode())
        if source.is_symlink():
            digest.update(b"\0symlink\0")
        elif source.is_file():
            digest.update(f"{source.stat().st_mode & 0o777:04o}".encode())
        elif not source.exists():
            digest.update(b"\0missing\0")
        else:
            raise StagingDeployError(f"unsupported tracked source type: {relative}")
    for relative in sorted(untracked, key=lambda value: value.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise StagingDeployError(f"untracked source leaves repository: {relative}")
        source = repository_root.resolve() / relative
        digest.update(b"\0untracked\0")
        digest.update(relative.as_posix().encode())
        if source.is_symlink():
            raise StagingDeployError(
                f"untracked symlinks are not accepted for staging: {relative}"
            )
        elif source.is_file():
            digest.update(b"\0file\0")
            digest.update(f"{source.stat().st_mode & 0o777:04o}".encode())
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise StagingDeployError(f"unsupported untracked source type: {relative}")
    return digest.hexdigest(), diff, tracked, untracked


def _copy_untracked_files(
    repository_root: Path,
    snapshot_root: Path,
    paths: Sequence[Path],
) -> None:
    for relative in paths:
        source = repository_root / relative
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            raise StagingDeployError(
                f"untracked symlinks are not accepted for staging: {relative}"
            )
        elif source.is_file():
            shutil.copy2(source, target)
        else:
            raise StagingDeployError(f"unsupported untracked source type: {relative}")


def _copy_tracked_file_modes(
    repository_root: Path,
    snapshot_root: Path,
    paths: Sequence[Path],
) -> None:
    for relative in paths:
        source = repository_root / relative
        target = snapshot_root / relative
        if source.is_symlink() or not source.exists():
            continue
        if not source.is_file() or not target.is_file():
            raise StagingDeployError(
                f"tracked source mode cannot be preserved: {relative}"
            )
        target.chmod(source.stat().st_mode & 0o777)


def _apply_snapshot_diff(snapshot_root: Path, diff: bytes) -> None:
    if not diff:
        return
    print("+ git apply --binary -", file=sys.stderr, flush=True)
    completed = subprocess.run(
        ["git", "apply", "--binary", "-"],
        cwd=snapshot_root,
        input=diff,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise StagingDeployError(
            "cannot apply staging source diff: "
            + completed.stderr.decode(errors="replace").strip()
        )


def _load_source_snapshot(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_base_commit: str,
    expected_staging_only: bool,
    snapshot_dir: Path,
) -> SourceCheckout | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        schema_version = int(value["schema_version"])
        repository_root = Path(str(value["repository_root"])).resolve()
        git_commit = str(value["git_commit"])
        fingerprint = str(value["fingerprint"])
        base_commit = str(value["base_commit"])
        raw_staging_only = value.get("staging_only", git_commit != base_commit)
        if not isinstance(raw_staging_only, bool):
            raise ValueError("staging_only must be a boolean")
        staging_only = raw_staging_only
        prepared_tree_sha256 = value.get("prepared_tree_sha256")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise StagingDeployError("staging source snapshot metadata is invalid") from exc
    if schema_version not in {1, 2}:
        raise StagingDeployError("staging source snapshot schema is invalid")
    if fingerprint != expected_fingerprint or base_commit != expected_base_commit:
        raise StagingDeployError("staging source snapshot identity does not match")
    if staging_only is not expected_staging_only:
        raise StagingDeployError("staging source snapshot tier does not match")
    try:
        repository_root.relative_to(snapshot_dir.resolve())
    except ValueError as exc:
        raise StagingDeployError(
            "staging source snapshot repository leaves its state directory"
        ) from exc
    if not repository_root.is_dir():
        raise StagingDeployError("staging source snapshot repository is missing")
    if _worktree_status(repository_root):
        raise StagingDeployError("staging source snapshot is not clean")
    if _git_output(repository_root, "rev-parse", "HEAD") != git_commit:
        raise StagingDeployError("staging source snapshot commit does not match")
    if staging_only:
        if _git_output(repository_root, "rev-parse", "HEAD^") != base_commit:
            raise StagingDeployError(
                "staging source snapshot base commit does not match"
            )
    elif git_commit != base_commit:
        raise StagingDeployError("clean source snapshot base commit does not match")
    if prepared_tree_sha256 is not None:
        if (
            not isinstance(prepared_tree_sha256, str)
            or len(prepared_tree_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in prepared_tree_sha256
            )
            or _prepared_tree_sha256(repository_root) != prepared_tree_sha256
        ):
            raise StagingDeployError(
                "staging source snapshot prepared tree does not match"
            )
    return SourceCheckout(
        repository_root=repository_root,
        git_commit=git_commit,
        fingerprint=fingerprint,
        snapshot=staging_only,
        isolated=True,
    )


def prepare_source_checkout(
    repository_root: Path,
    *,
    state_dir: Path,
) -> SourceCheckout:
    head = _repository_head(repository_root)
    staging_only = bool(_worktree_status(repository_root))
    fingerprint, diff, tracked, untracked = _source_fingerprint(
        repository_root,
        head=head,
    )
    snapshot_dir = state_dir / "source-snapshots" / fingerprint
    metadata_path = snapshot_dir / "snapshot.json"
    existing = _load_source_snapshot(
        metadata_path,
        expected_fingerprint=fingerprint,
        expected_base_commit=head,
        expected_staging_only=staging_only,
        snapshot_dir=snapshot_dir,
    )
    if existing is not None:
        return existing
    snapshot_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot_dir.chmod(0o700)
    candidate = snapshot_dir / (f"repository-{os.getpid()}-{secrets.token_hex(4)}")
    _run(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(candidate),
            head,
        ],
        cwd=repository_root,
    )
    _apply_snapshot_diff(candidate, diff)
    _copy_untracked_files(repository_root, candidate, untracked)
    _copy_tracked_file_modes(repository_root, candidate, tracked)
    if staging_only:
        _run(["git", "add", "-A"], cwd=candidate)
        commit_environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "GPU Fault Staging Snapshot",
            "GIT_AUTHOR_EMAIL": "staging@localhost",
            "GIT_COMMITTER_NAME": "GPU Fault Staging Snapshot",
            "GIT_COMMITTER_EMAIL": "staging@localhost",
        }
        _run(
            [
                "git",
                "commit",
                "--no-gpg-sign",
                "--no-verify",
                "-m",
                f"staging snapshot {fingerprint[:12]}",
            ],
            cwd=candidate,
            env=commit_environment,
        )
    git_commit = _git_output(candidate, "rev-parse", "HEAD")
    prepared_tree_sha256 = _prepared_tree_sha256(candidate)
    metadata = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "base_commit": head,
        "git_commit": git_commit,
        "repository_root": str(candidate),
        "staging_only": staging_only,
        "prepared_tree_sha256": prepared_tree_sha256,
    }
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, metadata_path)
    return SourceCheckout(
        repository_root=candidate,
        git_commit=git_commit,
        fingerprint=fingerprint,
        snapshot=staging_only,
        isolated=True,
    )


def ensure_signing_material(
    state_dir: Path,
    *,
    repository_root: Path,
) -> SigningMaterial:
    signing_dir = state_dir / "release-signing"
    signing_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    signing_dir.chmod(0o700)
    private_key = signing_dir / "cosign.key"
    public_key = signing_dir / "cosign.pub"
    password_file = signing_dir / "cosign.password"
    key_present = private_key.is_file()
    public_present = public_key.is_file()
    if key_present != public_present:
        missing = "public key" if key_present else "private key"
        raise StagingDeployError(
            "release signing material is incomplete; missing " + missing
        )
    stored_password = (
        password_file.read_text(encoding="utf-8").strip()
        if password_file.is_file()
        else ""
    )
    password = stored_password or os.getenv("COSIGN_PASSWORD", "")
    if not password:
        if key_present:
            raise StagingDeployError(
                "existing Cosign key requires release-signing/cosign.password "
                "or COSIGN_PASSWORD"
            )
        password = secrets.token_urlsafe(48)
    if not stored_password:
        password_file.write_text(password, encoding="utf-8")
        password_file.chmod(0o600)
    if not key_present:
        environment = {**os.environ, "COSIGN_PASSWORD": password}
        previous_umask = os.umask(0o077)
        try:
            _run(
                [
                    "cosign",
                    "generate-key-pair",
                    "--output-key-prefix",
                    str(signing_dir / "cosign"),
                ],
                cwd=repository_root,
                env=environment,
            )
        finally:
            os.umask(previous_umask)
        private_key.chmod(0o600)
        public_key.chmod(0o644)
    return SigningMaterial(
        private_key=private_state_file(private_key, "Cosign signing key"),
        public_key=public_state_file(public_key, "Cosign public key"),
        password_file=private_state_file(password_file, "Cosign password file"),
        password=password,
    )


def source_deploy_identity(repository_root: Path) -> dict[str, object]:
    output = _run(
        [
            sys.executable,
            str(repository_root / "scripts/deploy_source_identity.py"),
            "--root",
            str(repository_root),
        ],
        cwd=repository_root,
        capture=True,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StagingDeployError("source deploy identity output is invalid") from exc
    if not isinstance(value, dict):
        raise StagingDeployError("source deploy identity must be an object")
    for name in ("application", "deploy_host"):
        item = value.get(name)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or len(str(item["sha256"])) != 64
        ):
            raise StagingDeployError(f"source deploy {name} identity is invalid")
    deploy_host = value["deploy_host"]
    assert isinstance(deploy_host, dict)
    bundle = deploy_host.get("bundle")
    if (
        not isinstance(bundle, dict)
        or not isinstance(bundle.get("sha256"), str)
        or len(str(bundle["sha256"])) != 64
    ):
        raise StagingDeployError("deploy-host bundle identity is invalid")
    return value


def restore_trusted_ci_candidate(
    repository_root: Path,
    *,
    staging_only: bool,
) -> dict[str, object] | None:
    if staging_only:
        return None
    output = _run(
        [
            sys.executable,
            str(repository_root / "scripts/restore_ci_candidate.py"),
            "--root",
            str(repository_root),
            "--destination",
            str(repository_root / "dist"),
        ],
        cwd=repository_root,
        capture=True,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StagingDeployError("trusted CI candidate result is invalid") from exc
    if not isinstance(value, dict) or not isinstance(value.get("available"), bool):
        raise StagingDeployError("trusted CI candidate result is incomplete")
    return value if value["available"] is True else None


def record_trusted_ci_candidate(
    repository_root: Path,
    *,
    state_dir: Path,
    signing: SigningMaterial,
    candidate: Mapping[str, object],
) -> None:
    output = _run(
        [
            sys.executable,
            str(repository_root / "scripts/ci_candidate_receipt.py"),
            "write",
            "--root",
            str(repository_root),
            "--state-dir",
            str(state_dir),
            "--gate",
            str(candidate["ci_gate"]),
            "--repository",
            str(candidate["repository"]),
            "--run-id",
            str(candidate["run_id"]),
            "--signing-key",
            str(signing.private_key),
        ],
        cwd=repository_root,
        env={**os.environ, "COSIGN_PASSWORD": signing.password},
        capture=True,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StagingDeployError(
            "trusted CI candidate receipt output is invalid"
        ) from exc
    if not isinstance(value, dict) or value.get("available") is not True:
        raise StagingDeployError("trusted CI candidate receipt was not written")


def ensure_deploy_host_bundle(
    artifacts: DeployHostArtifacts,
    *,
    repository_root: Path,
    signing: SigningMaterial,
    wheelhouse_cache: Path | None = None,
) -> bool:
    present = (
        artifacts.archive.is_file(),
        artifacts.checksum.is_file(),
        artifacts.signature_bundle.is_file(),
    )
    if any(present) and not all(present):
        for path in (
            artifacts.archive,
            artifacts.checksum,
            artifacts.signature_bundle,
        ):
            path.unlink(missing_ok=True)
    if all(present):
        return True
    artifacts.archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = {**os.environ, "COSIGN_PASSWORD": signing.password}
    command = [
        "make",
        "deploy-host-bundle",
        f"PYTHON={sys.executable}",
        f"COSIGN_SIGNING_KEY={signing.private_key}",
        f"DEPLOY_HOST_ARCHIVE={artifacts.archive}",
        f"DEPLOY_HOST_SIGNATURE_BUNDLE={artifacts.signature_bundle}",
    ]
    if wheelhouse_cache is not None:
        command.append(f"DEPLOY_HOST_WHEELHOUSE={wheelhouse_cache}")
    _run(
        command,
        cwd=repository_root,
        env=environment,
    )
    if not all(
        path.is_file()
        for path in (
            artifacts.archive,
            artifacts.checksum,
            artifacts.signature_bundle,
        )
    ):
        raise StagingDeployError("deploy-host bundle build did not publish all files")
    return False


def ensure_deploy_host_venv(
    state_dir: Path,
    *,
    repository_root: Path,
    signing: SigningMaterial,
    artifacts: DeployHostArtifacts,
    lock_fd: int,
) -> Path:
    """Install the signed bundle into ``deployer-venv`` and bind it to the state.

    ``lock_fd`` is the site operation lock the caller holds. Superseded versions
    are deleted here and only here: the setup script this shells out to is also
    hand-run with no lock, and a tree a concurrent install is writing into looks
    exactly like one it abandoned.
    """

    venv = state_dir / "deployer-venv"
    environment = {**os.environ, "PYTHON": sys.executable}
    _run(
        [
            str(repository_root / "scripts/setup-deploy-host.sh"),
            "--venv",
            str(venv),
            "--bundle",
            str(artifacts.archive),
            "--signature-bundle",
            str(artifacts.signature_bundle),
            "--cosign-key",
            str(signing.public_key),
        ],
        cwd=repository_root,
        env=environment,
    )
    admin = venv / "bin/gpu-fault-admin"
    if not admin.is_file():
        raise StagingDeployError(f"deploy-host venv has no admin CLI: {admin}")
    binding = venv.resolve() / "gpu-fault-managed-state-dir.json"
    temporary = binding.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_dir": str(state_dir.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, binding)
    try:
        prune_venv_versions(venv, lock_fd=lock_fd)
    except OSError as exc:
        # The new venv is installed, active and bound. Failing the deploy over
        # the disk the superseded versions occupy would report a broken host for
        # a host that is ready, so this is reported and left for the next run.
        print(f"staging-deploy: venv versions not pruned: {exc}", file=sys.stderr)
    return venv


def load_successful_source_deploy(
    state_dir: Path,
    *,
    signing: SigningMaterial,
) -> dict[str, object] | None:
    state_path, signature_path = source_success_paths(state_dir)
    present = (state_path.is_file(), signature_path.is_file())
    if not any(present):
        return None
    if not all(present):
        raise StagingDeployError("successful source deploy authorization is incomplete")
    _run(
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(signature_path),
            "--key",
            str(signing.public_key),
            str(state_path),
        ],
        cwd=state_dir,
        capture=True,
    )
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StagingDeployError(
            "successful source deploy authorization is invalid"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StagingDeployError(
            "successful source deploy authorization schema is invalid"
        )
    if value.get("status") != "PASSED":
        raise StagingDeployError("successful source deploy authorization is not passed")
    identities = value.get("identities")
    source = value.get("source")
    if not isinstance(identities, dict) or not isinstance(source, dict):
        raise StagingDeployError("successful source deploy authorization is incomplete")
    for name in ("application", "deploy_host"):
        item = identities.get(name)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or len(str(item["sha256"])) != 64
        ):
            raise StagingDeployError(
                f"successful source deploy {name} identity is invalid"
            )
    return value


def record_successful_source_deploy(
    state_dir: Path,
    *,
    source_repository_root: Path,
    source: SourceCheckout,
    identities: Mapping[str, object],
    signing: SigningMaterial,
    mode: str,
    live_evidence: Mapping[str, object],
) -> Path:
    state_path, signature_path = source_success_paths(state_dir)
    application = identities.get("application")
    deploy_host = identities.get("deploy_host")
    if not isinstance(application, Mapping) or not isinstance(deploy_host, Mapping):
        raise StagingDeployError("source deploy identities are incomplete")
    bundle = deploy_host.get("bundle")
    if not isinstance(bundle, Mapping):
        raise StagingDeployError("source deploy bundle identity is incomplete")
    value = {
        "schema_version": 1,
        "status": "PASSED",
        "mode": mode,
        "source_repository_root": str(source_repository_root),
        "prepared_repository_root": str(source.repository_root),
        "source": {
            "git_commit": source.git_commit,
            "fingerprint": source.fingerprint,
            "snapshot": source.snapshot,
            "isolated": source.isolated,
        },
        "identities": {
            "application": {"sha256": application.get("sha256")},
            "deploy_host": {
                "sha256": deploy_host.get("sha256"),
                "bundle": {"sha256": bundle.get("sha256")},
            },
        },
        "site_file": str(state_dir / "site.yaml"),
        "live": dict(live_evidence),
    }
    temporary_state = state_path.with_suffix(".tmp")
    temporary_signature = signature_path.with_suffix(".tmp")
    temporary_state.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_state.chmod(0o600)
    temporary_signature.unlink(missing_ok=True)
    _run(
        [
            "cosign",
            "sign-blob",
            "--yes",
            "--key",
            str(signing.private_key),
            "--bundle",
            str(temporary_signature),
            str(temporary_state),
        ],
        cwd=state_dir,
        env={**os.environ, "COSIGN_PASSWORD": signing.password},
        capture=True,
    )
    temporary_signature.chmod(0o600)
    os.replace(temporary_state, state_path)
    os.replace(temporary_signature, signature_path)
    return state_path


def classify_source_deploy(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
    *,
    source: SourceCheckout,
    site_exists: bool,
    live_matches: bool = False,
) -> str:
    if previous is None or not site_exists:
        return "APPLICATION_RELEASE"
    identities = previous.get("identities")
    previous_source = previous.get("source")
    if not isinstance(identities, dict) or not isinstance(previous_source, dict):
        return "APPLICATION_RELEASE"
    application = current.get("application")
    deploy_host = current.get("deploy_host")
    previous_application = identities.get("application")
    previous_deploy_host = identities.get("deploy_host")
    if not all(
        isinstance(item, dict)
        for item in (
            application,
            deploy_host,
            previous_application,
            previous_deploy_host,
        )
    ):
        return "APPLICATION_RELEASE"
    assert isinstance(application, dict)
    assert isinstance(deploy_host, dict)
    assert isinstance(previous_application, dict)
    assert isinstance(previous_deploy_host, dict)
    if application.get("sha256") != previous_application.get("sha256"):
        return "APPLICATION_RELEASE"
    if deploy_host.get("sha256") != previous_deploy_host.get("sha256"):
        return "DEPLOY_HOST_ONLY"
    if previous_source.get("fingerprint") == source.fingerprint and live_matches:
        return "UNCHANGED"
    if previous_source.get("fingerprint") == source.fingerprint:
        return "APPLICATION_RELEASE"
    return "QUALITY_ONLY"


def _impact_base(
    repository_root: Path,
    previous: Mapping[str, object],
    fallback: str,
) -> str:
    source = previous.get("source")
    candidate = str(source.get("git_commit") or "") if isinstance(source, dict) else ""
    if candidate:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", f"{candidate}^{{commit}}"],
            cwd=repository_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return candidate
    return fallback


def run_source_impact_gate(
    *,
    repository_root: Path,
    state_dir: Path,
    source: SourceCheckout,
    previous: Mapping[str, object],
    fallback_base: str,
) -> dict[str, object]:
    output = state_dir / "source-gates" / source.fingerprint / "impact-plan.json"
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.parent.chmod(0o700)
    base = _impact_base(repository_root, previous, fallback_base)
    selector = repository_root / "scripts/select-affected-tests.py"
    environment = tool_cache_environment(state_dir)
    raw = _run(
        [
            sys.executable,
            str(selector),
            "--base",
            base,
            "--format",
            "json",
            "--write-plan",
            str(output),
        ],
        cwd=repository_root,
        capture=True,
        env=environment,
    )
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StagingDeployError("source impact gate plan is invalid") from exc
    if not isinstance(plan, dict):
        raise StagingDeployError("source impact gate plan must be an object")
    _run(
        [
            sys.executable,
            str(selector),
            "--base",
            base,
            "--read-plan",
            str(output),
            "--execute",
        ],
        cwd=repository_root,
        env=environment,
    )
    return plan


def run_admin_preflight(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    lock_fd: int | None = None,
) -> None:
    environment = (
        {**os.environ, SITE_OPERATION_LOCK_FD_ENV: str(lock_fd)}
        if lock_fd is not None
        else None
    )
    _run(
        [
            str(venv / "bin/gpu-fault-admin"),
            "preflight",
            "--state-dir",
            str(state_dir),
        ],
        cwd=repository_root,
        env=environment,
        pass_fds=(lock_fd,) if lock_fd is not None else (),
    )


def _link_or_copy(source: str, destination: str) -> str:
    path = Path(source)
    if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
        try:
            os.link(source, destination)
            return destination
        except OSError:
            pass
    shutil.copy2(source, destination)
    return destination


def prepare_deploy_host_only_checkout(
    *,
    repository_root: Path,
    state_dir: Path,
    signing: SigningMaterial,
) -> bytes:
    site_file = state_dir / "site.yaml"
    original = site_file.read_bytes()
    try:
        document = yaml.safe_load(original)
        configured = document["spec"]["repositoryRoot"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise StagingDeployError(
            "managed site has no valid repository root for host-only update"
        ) from exc
    previous_root = Path(str(configured)).expanduser()
    if not previous_root.is_absolute():
        previous_root = site_file.parent / previous_root
    previous_root = previous_root.resolve()
    previous_dist = previous_root / "dist"
    if not previous_dist.is_dir():
        raise StagingDeployError("managed site release artifacts are missing")
    current_dist = repository_root / "dist"
    if previous_root != repository_root:
        if current_dist.exists():
            shutil.rmtree(current_dist)
        shutil.copytree(
            previous_dist,
            current_dist,
            copy_function=_link_or_copy,
        )
    _run(
        [
            sys.executable,
            str(repository_root / "scripts/verify-release-attestation.py"),
            "--attestation",
            str(current_dist / "current-attestation.json"),
            "--bundle",
            str(current_dist / "current-attestation.bundle.json"),
            "--cosign-key",
            str(signing.public_key),
            "--allow-staging-release",
        ],
        cwd=repository_root,
    )
    document["spec"]["repositoryRoot"] = str(repository_root)
    temporary = site_file.with_suffix(".tmp")
    temporary.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, site_file)
    return original


def run_admin_deploy(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    cpu_cluster_arn: str,
    gpu_cluster_arns: Sequence[str],
    admin_email: str,
    staging_only_release: bool,
    impact_base: str,
    lock_fd: int,
) -> None:
    command = [
        str(venv / "bin/gpu-fault-admin"),
        "deploy",
        "--cpu-cluster-arn",
        cpu_cluster_arn,
    ]
    for arn in gpu_cluster_arns:
        command.extend(("--gpu-cluster-arn", arn))
    command.extend(
        (
            "--state-dir",
            str(state_dir),
            "--admin-email",
            admin_email,
            "--repo-root",
            str(repository_root),
            "--impact-base",
            impact_base,
            "--prepared-source-release",
        )
    )
    if staging_only_release:
        command.append("--staging-only-release")
    # The admin deploy re-runs the static gates inside the prepared snapshot,
    # which by construction carries no tool cache; pointing them at the state
    # directory is what makes the second run of an identical analysis cheap.
    _run(
        command,
        cwd=repository_root,
        env={
            **tool_cache_environment(state_dir),
            SITE_OPERATION_LOCK_FD_ENV: str(lock_fd),
        },
        pass_fds=(lock_fd,),
    )


def _ensure_deploy_host(
    state_dir: Path,
    *,
    source: SourceCheckout,
    signing: SigningMaterial,
    artifacts: DeployHostArtifacts,
    lock_fd: int,
) -> tuple[bool, Path]:
    """The signed bundle and the venv installed from it, built once if missing.

    Returns whether the bundle was reused, and the venv the admin CLI runs from.
    """

    wheelhouse_cache = deploy_host_wheelhouse_cache(
        state_dir,
        repository_root=source.repository_root,
    )
    bundle_reused = ensure_deploy_host_bundle(
        artifacts,
        repository_root=source.repository_root,
        signing=signing,
        wheelhouse_cache=wheelhouse_cache,
    )
    venv = ensure_deploy_host_venv(
        state_dir,
        repository_root=source.repository_root,
        signing=signing,
        artifacts=artifacts,
        lock_fd=lock_fd,
    )
    return bundle_reused, venv


def apply_source_deploy(
    *,
    arguments: argparse.Namespace,
    repository_root: Path,
    state_dir: Path,
    source: SourceCheckout,
    identities: Mapping[str, object],
    signing: SigningMaterial,
    venv: Path,
    prepared_mode: str,
    trusted_ci_candidate: Mapping[str, object] | None,
    lock_fd: int,
    live_evidence: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Apply ``prepared_mode`` and record the success that authorizes the next run.

    ``lock_fd`` is the site operation lock the caller holds across its own
    classification, and it is required. That is what removes a
    ``gpu-fault-admin status`` -- about 45 seconds -- from every deploy: nothing
    can change the site between the reading that produced ``prepared_mode`` and
    this application, so there is no re-classification to make and no second
    decision path that could disagree with the caller's.

    ``live_evidence`` is that same reading. UNCHANGED applies nothing, so it is
    still true afterwards and is recorded as the success; every other mode
    changes the site and reads it again.
    """

    mode = prepared_mode
    evidence: dict[str, object] | None = (
        dict(live_evidence) if live_evidence is not None else None
    )
    record_source_deploy_state(
        state_dir, source_repository_root=repository_root, source=source
    )
    if trusted_ci_candidate is not None and mode == "APPLICATION_RELEASE":
        record_trusted_ci_candidate(
            source.repository_root,
            state_dir=state_dir,
            signing=signing,
            candidate=trusted_ci_candidate,
        )
    if mode == "DEPLOY_HOST_ONLY":
        original_site = prepare_deploy_host_only_checkout(
            repository_root=source.repository_root,
            state_dir=state_dir,
            signing=signing,
        )
        try:
            run_admin_preflight(
                repository_root=source.repository_root,
                state_dir=state_dir,
                venv=venv,
                lock_fd=lock_fd,
            )
        except Exception:
            restore_site_file(state_dir, original_site)
            raise
    elif mode == "APPLICATION_RELEASE":
        run_admin_deploy(
            repository_root=source.repository_root,
            state_dir=state_dir,
            venv=venv,
            cpu_cluster_arn=arguments.cpu_cluster_arn,
            gpu_cluster_arns=tuple(arguments.gpu_cluster_arn),
            admin_email=arguments.admin_email,
            staging_only_release=source.snapshot,
            impact_base=arguments.base.strip(),
            lock_fd=lock_fd,
        )
    # UNCHANGED deployed nothing, so the evidence read before the apply still
    # describes the site. Every other mode changed it, so it is read again --
    # that reading is the success record, not a probe.
    if mode != "UNCHANGED" or evidence is None:
        evidence = collect_live_deploy_evidence(
            repository_root=source.repository_root,
            state_dir=state_dir,
            venv=venv,
            lock_fd=lock_fd,
        )
    record_successful_source_deploy(
        state_dir,
        source_repository_root=repository_root,
        source=source,
        identities=identities,
        signing=signing,
        mode=mode,
        live_evidence=evidence,
    )
    return mode, evidence


def deploy(arguments: argparse.Namespace) -> dict[str, object]:
    repository_root = arguments.repo_root.expanduser().resolve()
    state_dir = arguments.state_dir.expanduser().resolve()
    if not arguments.base.strip():
        raise StagingDeployError("impact test base must not be empty")
    try:
        state_dir.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise StagingDeployError(
            "staging state directory must be outside the Git repository"
        )
    # Created before the gate runs so the gate has somewhere to record its
    # verdict; the directory is empty and 0700 either way, and nothing is
    # deployed from it until the gate has passed.
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    verdict_cache = public_release_verdict_cache(state_dir)
    validate_source_checkout(repository_root, verdict_cache=verdict_cache)
    source = prepare_source_checkout(
        repository_root,
        state_dir=state_dir,
    )
    if source.repository_root != repository_root:
        validate_source_checkout(source.repository_root, verdict_cache=verdict_cache)
    identities = source_deploy_identity(source.repository_root)
    signing = ensure_signing_material(
        state_dir,
        repository_root=source.repository_root,
    )
    # The site lock is taken before the first ``gpu-fault-admin status`` and held
    # through the apply: the classification below is read from the live site, and
    # a classification that another operation can invalidate has to be re-read,
    # which is the 45-second call this ordering deletes.
    with site_operation_lock(state_dir, wait=True) as lock_fd:
        previous = load_successful_source_deploy(state_dir, signing=signing)
        venv = state_dir / "deployer-venv"
        live_evidence: dict[str, object] | None = None
        if (
            previous is not None
            and (state_dir / "site.yaml").is_file()
            and (venv / "bin/gpu-fault-admin").is_file()
        ):
            try:
                live_evidence = collect_live_deploy_evidence(
                    repository_root=source.repository_root,
                    state_dir=state_dir,
                    venv=venv,
                    lock_fd=lock_fd,
                )
            except (LiveEvidenceError, StagingDeployError):
                live_evidence = None
        mode = classify_source_deploy(
            previous,
            identities,
            source=source,
            site_exists=(state_dir / "site.yaml").is_file(),
            live_matches=successful_source_live_matches(previous, live_evidence),
        )
        prepared_mode = mode
        trusted_ci_candidate = (
            restore_trusted_ci_candidate(
                source.repository_root,
                staging_only=source.snapshot,
            )
            if mode == "APPLICATION_RELEASE"
            else None
        )
        deploy_host = identities["deploy_host"]
        assert isinstance(deploy_host, dict)
        bundle_identity = deploy_host["bundle"]
        assert isinstance(bundle_identity, dict)
        artifacts = deploy_host_artifacts(
            state_dir,
            payload_identity_sha256=str(bundle_identity["sha256"]),
        )
        bundle_reused = True
        if mode in {"APPLICATION_RELEASE", "DEPLOY_HOST_ONLY"}:
            bundle_reused, venv = _ensure_deploy_host(
                state_dir,
                source=source,
                signing=signing,
                artifacts=artifacts,
                lock_fd=lock_fd,
            )
        elif not (venv / "bin/gpu-fault-admin").is_file():
            # Rebuilding the deploy host means there is no admin CLI to apply a
            # quality-only pass with, so this is an application release -- and
            # ``prepared_mode`` has to say so too, or the apply would classify
            # QUALITY_ONLY again and deploy nothing while the release path here
            # has already skipped the impact gate.
            mode = "APPLICATION_RELEASE"
            prepared_mode = mode
            bundle_reused, venv = _ensure_deploy_host(
                state_dir,
                source=source,
                signing=signing,
                artifacts=artifacts,
                lock_fd=lock_fd,
            )

        gated = mode in {"DEPLOY_HOST_ONLY", "QUALITY_ONLY"}
        impact_plan: dict[str, object] | None = None
        if gated and trusted_ci_candidate is None:
            assert previous is not None
            impact_plan = run_source_impact_gate(
                repository_root=source.repository_root,
                state_dir=state_dir,
                source=source,
                previous=previous,
                fallback_base=arguments.base.strip(),
            )
        elif gated:
            impact_plan = {
                "source": "signed_main_ci_candidate",
                "run_id": trusted_ci_candidate["run_id"],
            }
        mode, live_evidence = apply_source_deploy(
            arguments=arguments,
            repository_root=repository_root,
            state_dir=state_dir,
            source=source,
            identities=identities,
            signing=signing,
            venv=venv,
            prepared_mode=prepared_mode,
            trusted_ci_candidate=trusted_ci_candidate,
            lock_fd=lock_fd,
            live_evidence=live_evidence,
        )
        try:
            prune_source_snapshots(
                state_dir,
                source_repository_root=repository_root,
                current=source.repository_root,
            )
        except (OSError, StagingDeployError) as exc:
            # The deploy is applied and its success is already signed. Failing it
            # now over disk hygiene would report a failure for a site that is
            # fully deployed, so this is reported and left for the next run.
            print(
                f"staging-deploy: source snapshots not pruned: {exc}", file=sys.stderr
            )
        return {
            "schema_version": 1,
            "git_commit": source.git_commit,
            "source_fingerprint": source.fingerprint,
            "source_checkout": str(source.repository_root),
            "source_snapshot": source.snapshot,
            "source_isolated": source.isolated,
            "deploy_mode": mode,
            "application_identity_sha256": str(
                cast(Mapping[str, object], identities["application"])["sha256"]
            ),
            "deploy_host_identity_sha256": str(deploy_host["sha256"]),
            "impact_plan": impact_plan,
            "trusted_ci_candidate": trusted_ci_candidate,
            "state_dir": str(state_dir),
            "deploy_host_bundle": str(artifacts.archive),
            "deploy_host_bundle_reused": bundle_reused,
            "deploy_host_venv": str(venv),
            "site_file": str(state_dir / "site.yaml"),
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Build or reuse a signed deploy-host environment, then bootstrap "
            "or upgrade one staging site with the same command"
        )
    )
    value.add_argument("--cpu-cluster-arn", required=True)
    value.add_argument(
        "--gpu-cluster-arn",
        action="append",
        required=True,
    )
    value.add_argument("--state-dir", required=True, type=Path)
    value.add_argument("--admin-email", required=True)
    value.add_argument("--base", default="origin/main")
    value.add_argument("--repo-root", type=Path, default=ROOT)
    value.add_argument("--quiet", action="store_true")
    return value


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parser().parse_args(arguments)
    try:
        result = deploy(parsed)
    except (
        LiveEvidenceError,
        OSError,
        StagingDeployError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"staging-deploy: {exc}", file=sys.stderr)
        return 2
    if not parsed.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
