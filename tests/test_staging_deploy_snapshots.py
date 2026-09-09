from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import staging_deploy, staging_state_hygiene


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def test_clean_source_is_isolated_without_staging_tier(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / ".gitignore").write_text("dist/\n", encoding="utf-8")
    tracked = repository / "tracked.txt"
    tracked.write_text("release\n", encoding="utf-8")
    tracked.chmod(0o664)
    _git(repository, "add", ".gitignore", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    state = tmp_path / "state"

    first = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    first.repository_root.joinpath("dist").mkdir()
    first.repository_root.joinpath("dist/current-release.json").write_text(
        "signed-release\n", encoding="utf-8"
    )
    repository.joinpath("dist").mkdir()
    repository.joinpath("dist/current-release.json").write_text(
        "later-local-build\n", encoding="utf-8"
    )
    second = staging_deploy.prepare_source_checkout(repository, state_dir=state)

    assert first == second
    assert first.repository_root != repository
    assert first.snapshot is False
    assert first.isolated is True
    assert first.git_commit == _git(repository, "rev-parse", "HEAD")
    assert first.repository_root.joinpath("tracked.txt").stat().st_mode & 0o777 == 0o664
    assert (
        first.repository_root.joinpath("dist/current-release.json").read_text(
            encoding="utf-8"
        )
        == "signed-release\n"
    )


def test_dirty_source_is_snapshotted_without_changing_original(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    (repository / "new.txt").write_text("new\n", encoding="utf-8")

    first = staging_deploy.prepare_source_checkout(
        repository, state_dir=tmp_path / "state"
    )
    second = staging_deploy.prepare_source_checkout(
        repository, state_dir=tmp_path / "state"
    )

    assert first.snapshot is True
    assert first.isolated is True
    assert first == second
    assert (first.repository_root / "tracked.txt").read_text() == "after\n"
    assert (first.repository_root / "new.txt").read_text() == "new\n"
    assert _git(first.repository_root, "status", "--porcelain") == ""
    assert "tracked.txt" in _git(repository, "status", "--short")
    assert "new.txt" in _git(repository, "status", "--short")


def test_dirty_snapshot_preserves_tracked_modes_across_umasks(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    tracked.chmod(0o664)
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    state = tmp_path / "state"

    previous_umask = os.umask(0o077)
    try:
        first = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    finally:
        os.umask(previous_umask)

    assert first.repository_root.joinpath("tracked.txt").stat().st_mode & 0o777 == 0o664

    tracked.chmod(0o644)
    second = staging_deploy.prepare_source_checkout(repository, state_dir=state)

    assert second.fingerprint != first.fingerprint
    assert (
        second.repository_root.joinpath("tracked.txt").stat().st_mode & 0o777 == 0o644
    )


def test_snapshot_metadata_must_match_current_fingerprint(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    state = tmp_path / "state"
    checkout = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    snapshot_dir = state / "source-snapshots" / checkout.fingerprint
    metadata = snapshot_dir / "snapshot.json"
    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["fingerprint"] = "c" * 64
    metadata.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        staging_deploy.StagingDeployError, match="snapshot identity does not match"
    ):
        staging_deploy.prepare_source_checkout(repository, state_dir=state)


def test_snapshot_prepared_tree_must_remain_unchanged(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("release\n", encoding="utf-8")
    tracked.chmod(0o644)
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    state = tmp_path / "state"
    checkout = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    checkout.repository_root.joinpath("tracked.txt").chmod(0o600)

    with pytest.raises(
        staging_deploy.StagingDeployError, match="prepared tree does not match"
    ):
        staging_deploy.prepare_source_checkout(repository, state_dir=state)


def _snapshot_tree(root: Path, index: int) -> Path:
    """One ``source-snapshots/<fingerprint>/repository-*`` pair, mtime by index."""

    fingerprint = f"{index:064d}"
    worktree = root / fingerprint / f"repository-{index}"
    worktree.mkdir(parents=True)
    (worktree / "tracked.txt").write_text("release\n", encoding="utf-8")
    (root / fingerprint / "snapshot.json").write_text("{}", encoding="utf-8")
    stamp = 1_700_000_000 + index
    os.utime(root / fingerprint, (stamp, stamp))
    return worktree


def test_prune_source_snapshots_keeps_referenced_and_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshots a deploy can still be asked to reproduce always survive.

    Production had 59 snapshots and 3.5 GB in this directory because nothing
    removed them. What may not be removed is what the recorded state points at:
    the pending deploy, the last success, and the tree running right now.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.setenv("GPU_FAULT_SOURCE_SNAPSHOT_RETAINED", "3")
    repository = tmp_path / "repo"
    repository.mkdir()
    state = tmp_path / "state"
    snapshots = state / "source-snapshots"
    trees = [_snapshot_tree(snapshots, index) for index in range(8)]
    (state / staging_state_hygiene.SOURCE_DEPLOY_STATE).write_text(
        json.dumps({"prepared_repository_root": str(trees[0])}), encoding="utf-8"
    )
    (state / staging_state_hygiene.SOURCE_DEPLOY_SUCCESS_STATE).write_text(
        json.dumps({"prepared_repository_root": str(trees[1])}), encoding="utf-8"
    )

    removed = staging_deploy.prune_source_snapshots(
        state, source_repository_root=repository, current=trees[2]
    )

    assert set(removed) == {trees[3].parent, trees[4].parent}
    for index in (0, 1, 2, 5, 6, 7):
        assert trees[index].is_dir(), index
    for index in (3, 4):
        assert not trees[index].parent.exists(), index

    monkeypatch.delenv("GPU_FAULT_SOURCE_SNAPSHOT_RETAINED")

    assert (
        staging_deploy.prune_source_snapshots(
            state, source_repository_root=repository, current=trees[2]
        )
        == ()
    ), "the default retention removed a snapshot it should have kept"

    monkeypatch.setenv("GPU_FAULT_SOURCE_SNAPSHOT_RETAINED", "0")

    with pytest.raises(staging_deploy.StagingDeployError, match="positive integer"):
        staging_deploy.prune_source_snapshots(
            state, source_repository_root=repository, current=trees[2]
        )


def test_prune_source_snapshots_unregisters_git_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot is a worktree of the source repository, so Git has to be told.

    ``shutil.rmtree`` alone leaves the registration behind in the source
    repository, and the next ``git worktree add`` for the same path fails.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.setenv("GPU_FAULT_SOURCE_SNAPSHOT_RETAINED", "1")
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "tracked.txt").write_text("release\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    state = tmp_path / "state"
    stale = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "second")
    current = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    assert current.repository_root != stale.repository_root

    removed = staging_deploy.prune_source_snapshots(
        state, source_repository_root=repository, current=current.repository_root
    )

    assert removed == (stale.repository_root.parent,)
    assert not stale.repository_root.exists(), "the stale snapshot must be pruned"
    assert current.repository_root.is_dir(), "the current snapshot must survive pruning"
    assert str(stale.repository_root) not in _git(repository, "worktree", "list")
