from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.e2e.regional.boot020_release_candidates import copy_candidate_repository


def git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *arguments],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_candidate_from_git_worktree_cannot_modify_source_head_or_index(
    tmp_path, monkeypatch
):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init")
    (origin / "module.py").write_text("value = 1\n")
    git(origin, "add", "module.py")
    git(
        origin,
        "-c",
        "user.name=Unit",
        "-c",
        "user.email=unit@example.invalid",
        "commit",
        "--no-gpg-sign",
        "-m",
        "fixture",
    )
    snapshot = tmp_path / "snapshot"
    git(origin, "worktree", "add", "--detach", str(snapshot))
    assert (snapshot / ".git").is_file(), (
        "the fixture must exercise a linked worktree, not a standalone checkout"
    )
    before = git(snapshot, "rev-parse", "HEAD")
    source_index = Path(
        git(snapshot, "rev-parse", "--path-format=absolute", "--git-path", "index")
    ).read_bytes()
    # Ambient overrides must not redirect a candidate command to another index.
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "foreign-index"))
    candidate = tmp_path / "candidate"
    copy_candidate_repository(snapshot, candidate)
    monkeypatch.delenv("GIT_INDEX_FILE")
    assert (candidate / ".git").is_dir() and not (candidate / ".git").is_symlink()
    (candidate / "module.py").write_text("value = 2\n")
    git(candidate, "add", "module.py")
    git(
        candidate,
        "-c",
        "user.name=Unit",
        "-c",
        "user.email=unit@example.invalid",
        "commit",
        "--no-gpg-sign",
        "-m",
        "candidate",
    )
    assert git(snapshot, "rev-parse", "HEAD") == before
    assert git(origin, "rev-parse", "HEAD") == before
    assert git(snapshot, "status", "--porcelain") == ""
    assert (
        Path(
            git(snapshot, "rev-parse", "--path-format=absolute", "--git-path", "index")
        ).read_bytes()
        == source_index
    )
    assert git(candidate, "rev-parse", "HEAD") != before
    assert not (tmp_path / "foreign-index").exists(), (
        "candidate commands must not create an ambient override index"
    )
