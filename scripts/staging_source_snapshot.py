"""The source snapshot one staging deploy is built from.

Split from ``staging_deploy.py``: the detached git worktree copy of the
developer's checkout -- clean or carrying uncommitted work -- that the release
is built from, the fingerprint that names it, the metadata that lets a rerun
reuse it unchanged, and the subprocess helpers the snapshot and the rest of
the deploy share.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from gpu_fault.admin.command_log import child_failure, last_output_line

if __package__:
    from scripts.staging_state_hygiene import SourceCheckout, StagingDeployError
else:
    from staging_state_hygiene import SourceCheckout, StagingDeployError


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
        # The nested ``gpu-fault-admin`` this drives has already reported its
        # failure on the shared descriptors; a foreign command gets one line.
        raise child_failure(
            StagingDeployError,
            arguments,
            completed.returncode,
            detail=last_output_line(completed.stderr) if capture else "",
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
