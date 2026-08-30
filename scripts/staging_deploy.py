from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

if __package__:
    from scripts.deploy_host_bundle import bundle_platform_id
else:
    from deploy_host_bundle import bundle_platform_id


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEPLOY_STATE = "source-deploy.json"


class StagingDeployError(RuntimeError):
    pass


@dataclass(frozen=True)
class SigningMaterial:
    private_key: Path
    public_key: Path
    password_file: Path
    password: str


@dataclass(frozen=True)
class DeployHostArtifacts:
    archive: Path
    checksum: Path
    signature_bundle: Path


@dataclass(frozen=True)
class SourceCheckout:
    repository_root: Path
    git_commit: str
    fingerprint: str
    snapshot: bool


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> str:
    print("+ " + " ".join(arguments), file=sys.stderr, flush=True)
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


def validate_source_checkout(repository_root: Path) -> None:
    _run(
        [
            sys.executable,
            str(repository_root / "scripts/check-public-release.py"),
            "--root",
            str(repository_root),
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


def _source_fingerprint(
    repository_root: Path,
    *,
    head: str,
) -> tuple[str, bytes, tuple[Path, ...]]:
    diff = _git_bytes(repository_root, "diff", "--binary", "HEAD")
    untracked = _untracked_files(repository_root)
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(b"\0diff\0")
    digest.update(diff)
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
    return digest.hexdigest(), diff, untracked


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
    snapshot_dir: Path,
) -> SourceCheckout | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        repository_root = Path(str(value["repository_root"])).resolve()
        git_commit = str(value["git_commit"])
        fingerprint = str(value["fingerprint"])
        base_commit = str(value["base_commit"])
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise StagingDeployError("staging source snapshot metadata is invalid") from exc
    if fingerprint != expected_fingerprint or base_commit != expected_base_commit:
        raise StagingDeployError("staging source snapshot identity does not match")
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
    if _git_output(repository_root, "rev-parse", "HEAD^") != base_commit:
        raise StagingDeployError("staging source snapshot base commit does not match")
    return SourceCheckout(
        repository_root=repository_root,
        git_commit=git_commit,
        fingerprint=fingerprint,
        snapshot=True,
    )


def prepare_source_checkout(
    repository_root: Path,
    *,
    state_dir: Path,
) -> SourceCheckout:
    head = _repository_head(repository_root)
    if not _worktree_status(repository_root):
        return SourceCheckout(
            repository_root=repository_root,
            git_commit=head,
            fingerprint=head,
            snapshot=False,
        )
    fingerprint, diff, untracked = _source_fingerprint(
        repository_root,
        head=head,
    )
    snapshot_dir = state_dir / "source-snapshots" / fingerprint
    metadata_path = snapshot_dir / "snapshot.json"
    existing = _load_source_snapshot(
        metadata_path,
        expected_fingerprint=fingerprint,
        expected_base_commit=head,
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
    metadata = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "base_commit": head,
        "git_commit": git_commit,
        "repository_root": str(candidate),
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
        snapshot=True,
    )


def _private_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise StagingDeployError(f"{description} is missing: {path}")
    if path.stat().st_mode & 0o077:
        raise StagingDeployError(
            f"{description} must not be group/other accessible: {path}"
        )
    return path


def _public_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise StagingDeployError(f"{description} is missing: {path}")
    if path.stat().st_mode & 0o022:
        raise StagingDeployError(
            f"{description} must not be group/other writable: {path}"
        )
    return path


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
        private_key=_private_file(private_key, "Cosign signing key"),
        public_key=_public_file(public_key, "Cosign public key"),
        password_file=_private_file(password_file, "Cosign password file"),
        password=password,
    )


def deploy_host_artifacts(
    state_dir: Path,
    *,
    git_commit: str,
) -> DeployHostArtifacts:
    platform = bundle_platform_id()
    output = state_dir / "deploy-host" / git_commit
    archive = output / f"gpu-fault-deploy-host-{platform}.tar.gz"
    return DeployHostArtifacts(
        archive=archive,
        checksum=archive.with_suffix(archive.suffix + ".sha256"),
        signature_bundle=(output / f"gpu-fault-deploy-host-{platform}.sigstore.json"),
    )


def ensure_deploy_host_bundle(
    artifacts: DeployHostArtifacts,
    *,
    repository_root: Path,
    signing: SigningMaterial,
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
    _run(
        [
            "make",
            "deploy-host-bundle",
            f"PYTHON={sys.executable}",
            f"COSIGN_SIGNING_KEY={signing.private_key}",
            f"DEPLOY_HOST_ARCHIVE={artifacts.archive}",
            f"DEPLOY_HOST_SIGNATURE_BUNDLE={artifacts.signature_bundle}",
        ],
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
) -> Path:
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
    return venv


def record_source_deploy_state(
    state_dir: Path,
    *,
    source_repository_root: Path,
    source: SourceCheckout,
) -> Path:
    target = state_dir / SOURCE_DEPLOY_STATE
    value = {
        "schema_version": 1,
        "source_repository_root": str(source_repository_root),
        "prepared_repository_root": str(source.repository_root),
        "release_ref": source.git_commit,
        "source_fingerprint": source.fingerprint,
        "source_snapshot": source.snapshot,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, target)
    return target


def run_admin_deploy(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    cpu_cluster_arn: str,
    gpu_cluster_arns: Sequence[str],
    admin_email: str,
    profile_approval: str | None,
    staging_only_release: bool,
    impact_base: str,
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
    if profile_approval:
        command.extend(("--profile-approval", profile_approval))
    if staging_only_release:
        command.append("--staging-only-release")
    _run(command, cwd=repository_root)


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
    validate_source_checkout(repository_root)
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    source = prepare_source_checkout(
        repository_root,
        state_dir=state_dir,
    )
    if source.repository_root != repository_root:
        validate_source_checkout(source.repository_root)
    record_source_deploy_state(
        state_dir,
        source_repository_root=repository_root,
        source=source,
    )
    signing = ensure_signing_material(
        state_dir,
        repository_root=source.repository_root,
    )
    artifacts = deploy_host_artifacts(
        state_dir,
        git_commit=source.git_commit,
    )
    bundle_reused = ensure_deploy_host_bundle(
        artifacts,
        repository_root=source.repository_root,
        signing=signing,
    )
    venv = ensure_deploy_host_venv(
        state_dir,
        repository_root=source.repository_root,
        signing=signing,
        artifacts=artifacts,
    )
    run_admin_deploy(
        repository_root=source.repository_root,
        state_dir=state_dir,
        venv=venv,
        cpu_cluster_arn=arguments.cpu_cluster_arn,
        gpu_cluster_arns=tuple(arguments.gpu_cluster_arn),
        admin_email=arguments.admin_email,
        profile_approval=arguments.profile_approval,
        staging_only_release=source.snapshot,
        impact_base=arguments.base.strip(),
    )
    return {
        "schema_version": 1,
        "git_commit": source.git_commit,
        "source_fingerprint": source.fingerprint,
        "source_checkout": str(source.repository_root),
        "source_snapshot": source.snapshot,
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
    value.add_argument("--profile-approval")
    value.add_argument("--base", default="origin/main")
    value.add_argument("--repo-root", type=Path, default=ROOT)
    value.add_argument("--quiet", action="store_true")
    return value


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parser().parse_args(arguments)
    try:
        result = deploy(parsed)
    except (OSError, StagingDeployError, subprocess.SubprocessError) as exc:
        print(f"staging-deploy: {exc}", file=sys.stderr)
        return 2
    if not parsed.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
