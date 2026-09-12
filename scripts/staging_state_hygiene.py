"""What a site's deploy state directory holds, and what it must stop holding.

The state directory outlives every deploy that writes into it. It holds the
Cosign signing material, the recorded source-deploy state and its signed
success, the deploy-host bundle addressed by payload content, the wheelhouse
cache, the live ``site.yaml``, and one prepared source tree per source
fingerprint. Only the last of those grows without bound, and nothing removed it:
production reached 59 snapshots and 3.5 GB. So the pruning rules live here,
beside the records that decide which prepared trees a deploy can still be asked
to reproduce.

``StagingDeployError`` is defined here rather than in ``staging_deploy`` because
this is the lower of the two modules and both halves raise it. ``staging_deploy``
re-exports it, so the single error type a caller catches is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]

if __package__:
    from scripts.deploy_host_bundle import bundle_platform_id
else:
    from deploy_host_bundle import bundle_platform_id


SOURCE_DEPLOY_STATE = "source-deploy.json"
SOURCE_DEPLOY_SUCCESS_STATE = "source-deploy-success.json"
SOURCE_DEPLOY_SUCCESS_SIGNATURE = "source-deploy-success.sigstore.json"


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
    isolated: bool = False


def private_state_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise StagingDeployError(f"{description} is missing: {path}")
    if path.stat().st_mode & 0o077:
        raise StagingDeployError(
            f"{description} must not be group/other accessible: {path}"
        )
    return path


def public_state_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise StagingDeployError(f"{description} is missing: {path}")
    if path.stat().st_mode & 0o022:
        raise StagingDeployError(
            f"{description} must not be group/other writable: {path}"
        )
    return path


def deploy_host_artifacts(
    state_dir: Path,
    *,
    payload_identity_sha256: str,
) -> DeployHostArtifacts:
    if len(payload_identity_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in payload_identity_sha256
    ):
        raise StagingDeployError("deploy-host payload identity is invalid")
    platform = bundle_platform_id()
    output = state_dir / "deploy-host" / "by-content" / payload_identity_sha256
    archive = output / f"gpu-fault-deploy-host-{platform}.tar.gz"
    return DeployHostArtifacts(
        archive=archive,
        checksum=archive.with_suffix(archive.suffix + ".sha256"),
        signature_bundle=(output / f"gpu-fault-deploy-host-{platform}.sigstore.json"),
    )


def deploy_host_wheelhouse_cache(
    state_dir: Path,
    *,
    repository_root: Path,
) -> Path:
    digest = hashlib.sha256()
    for name in ("build.lock", "deploy-host.lock"):
        digest.update((repository_root / "requirements" / name).read_bytes())
    cache = (
        state_dir
        / "deploy-host-wheelhouse"
        / f"{bundle_platform_id()}-{digest.hexdigest()[:16]}"
    )
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache.chmod(0o700)
    return cache


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
        "source_isolated": source.isolated,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, target)
    return target


def source_success_paths(state_dir: Path) -> tuple[Path, Path]:
    return (
        state_dir / SOURCE_DEPLOY_SUCCESS_STATE,
        state_dir / SOURCE_DEPLOY_SUCCESS_SIGNATURE,
    )


def restore_site_file(state_dir: Path, content: bytes) -> None:
    site_file = state_dir / "site.yaml"
    temporary = site_file.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.chmod(0o600)
    os.replace(temporary, site_file)


def _referenced_snapshot_roots(state_dir: Path) -> tuple[Path, ...]:
    """The prepared trees the recorded deploy state still points at.

    Read unverified on purpose: the signature on the success record authorizes a
    deploy, and this only ever adds to what is kept. A record that cannot be read
    at all is different -- then nothing here knows what is in use, and the caller
    stops rather than guessing.
    """

    referenced: list[Path] = []
    for name in (SOURCE_DEPLOY_STATE, SOURCE_DEPLOY_SUCCESS_STATE):
        path = state_dir / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StagingDeployError(
                f"cannot read {name} to protect the snapshot it names"
            ) from exc
        prepared = (
            value.get("prepared_repository_root") if isinstance(value, dict) else None
        )
        if isinstance(prepared, str) and prepared.strip():
            referenced.append(Path(prepared))
    referenced.extend(_site_referenced_paths(state_dir))
    return tuple(referenced)


def _site_referenced_paths(state_dir: Path) -> list[Path]:
    """The trees the site document itself still names.

    ``spec.repositoryRoot`` and ``spec.runtimeProfile.templateSource`` are read by
    every later command (`status`, `verify`, the release engine); a snapshot they
    point into is in use whatever the deploy records say (live 2026-09-12: the
    template path still named the first run's snapshot when the pruning removed
    it, and `status` refused). An unreadable site is not protected: the caller
    only ever adds to what is kept, and a missing site names nothing.
    """

    site = state_dir / "site.yaml"
    if not site.is_file():
        return []
    try:
        document = yaml.safe_load(site.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    spec = document.get("spec") if isinstance(document, dict) else None
    if not isinstance(spec, dict):
        return []
    paths: list[Path] = []
    root = spec.get("repositoryRoot")
    if isinstance(root, str) and root.strip():
        paths.append(Path(root))
    profile = spec.get("runtimeProfile")
    template = profile.get("templateSource") if isinstance(profile, dict) else None
    if isinstance(template, str) and template.strip():
        paths.append(Path(template))
    return paths


def _remove_snapshot_tree(directory: Path, *, source_repository_root: Path) -> None:
    """Unregister the worktrees under ``directory``, then delete it.

    Removing the directory alone leaves the registration in the source
    repository, and the next ``git worktree add`` for a path Git still believes
    in fails.
    """

    for child in sorted(directory.iterdir()):
        if child.is_symlink() or not child.is_dir() or not (child / ".git").exists():
            continue
        arguments = ["git", "worktree", "remove", "--force", str(child)]
        print("+ " + " ".join(arguments), file=sys.stderr, flush=True)
        try:
            subprocess.run(
                arguments,
                cwd=source_repository_root,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            shutil.rmtree(child, ignore_errors=True)
    shutil.rmtree(directory, ignore_errors=True)


def _snapshot_retention() -> int:
    """How many prepared trees to keep beyond the ones a deploy still names."""

    raw = os.getenv("GPU_FAULT_SOURCE_SNAPSHOT_RETAINED", "").strip()
    if not raw:
        return 5
    try:
        retained = int(raw)
    except ValueError as exc:
        raise StagingDeployError(
            "GPU_FAULT_SOURCE_SNAPSHOT_RETAINED must be a positive integer"
        ) from exc
    if retained < 1:
        raise StagingDeployError(
            f"GPU_FAULT_SOURCE_SNAPSHOT_RETAINED must be a positive integer, not {raw}"
        )
    return retained


def prune_source_snapshots(
    state_dir: Path,
    *,
    source_repository_root: Path,
    current: Path | None = None,
) -> tuple[Path, ...]:
    """Drop prepared source trees no deploy can still be asked to reproduce.

    Every fingerprint gets its own worktree here and nothing removed them, so a
    long-lived state directory reached 59 snapshots and 3.5 GB. Kept: the tree
    named by the pending deploy state, the tree named by the last success (a
    rollback reads it), the tree being deployed right now, and the newest
    ``GPU_FAULT_SOURCE_SNAPSHOT_RETAINED`` (5) by modification time.

    Call this with the site lock held: it deletes trees a concurrent deploy could
    otherwise be running from.
    """

    root = state_dir / "source-snapshots"
    if not root.is_dir():
        return ()
    retained = _snapshot_retention()
    protected = list(_referenced_snapshot_roots(state_dir))
    if current is not None:
        protected.append(current)
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    removed: list[Path] = []
    for position, directory in enumerate(candidates):
        if position < retained:
            continue
        if any(
            reference == directory or directory in reference.parents
            for reference in protected
        ):
            continue
        _remove_snapshot_tree(
            directory,
            source_repository_root=source_repository_root,
        )
        removed.append(directory)
    return tuple(removed)
