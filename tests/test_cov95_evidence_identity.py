from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import pytest_result_identity as identity


@pytest.fixture
def git_transport(monkeypatch):
    state = {"head": b"unit-head\n", "diff": b"", "untracked": b"", "fail": False}

    def run(command, **options):
        assert command[0] == "git", "the identity probe may only inspect Git inputs"
        arguments = command[1:]
        if state["fail"]:
            return subprocess.CompletedProcess(
                command, 1, stdout=b"", stderr=b"synthetic Git read failure"
            )
        if arguments == ["rev-parse", "HEAD"]:
            output = state["head"]
        elif arguments == ["diff", "--binary", "HEAD"]:
            output = state["diff"]
        elif arguments == ["ls-files", "--others", "--exclude-standard", "-z"]:
            output = state["untracked"]
        else:
            pytest.fail(f"unexpected identity command: {arguments}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(identity.subprocess, "run", run)
    return state


def test_source_identity_binds_commit_diff_bytes_and_file_permissions(
    tmp_path: Path, git_transport
) -> None:
    path = tmp_path / "new source.py"
    path.write_bytes(b"original\n")
    path.chmod(0o600)
    git_transport["untracked"] = b"new source.py\0"
    previous = identity.source_identity(tmp_path)
    assert len(previous) == 64, "the source identity is one SHA-256 digest"
    assert identity.source_identity(tmp_path) == previous, (
        "unchanged inputs must produce the same source identity"
    )
    for field, value in (("head", b"next-head"), ("diff", b"modified tracked input")):
        git_transport[field] = value
        current = identity.source_identity(tmp_path)
        assert current != previous, (
            "commit and tracked diff changes must invalidate reuse"
        )
        previous = current
    path.write_bytes(b"changed\n")
    current = identity.source_identity(tmp_path)
    assert current != previous, "untracked source bytes must invalidate reuse"
    path.chmod(0o700)
    assert identity.source_identity(tmp_path) != current, (
        "executable-permission changes must invalidate reuse"
    )


def test_source_identity_handles_large_files_and_stable_inventory_order(
    tmp_path: Path, git_transport
) -> None:
    (tmp_path / "first.py").write_bytes(b"a" * (1024 * 1024 + 17))
    (tmp_path / "second.py").write_bytes(b"second")
    git_transport["untracked"] = b"second.py\0first.py\0"
    digest = identity.source_identity(tmp_path)
    git_transport["untracked"] = b"first.py\0second.py\0"
    assert identity.source_identity(tmp_path) == digest, (
        "Git listing order cannot alter the identity of identical source inputs"
    )
    git_transport["untracked"] = b"first.py\0"
    assert identity.source_identity(tmp_path) != digest, (
        "removing an untracked input must invalidate its previous proof"
    )


@pytest.mark.parametrize(
    "kind", ["absolute", "parent", "missing", "directory", "symlink"]
)
def test_untracked_identity_inputs_must_be_local_regular_files(
    tmp_path: Path, git_transport, kind
) -> None:
    name = "candidate.py"
    path = tmp_path / name
    if kind == "absolute":
        name = str(path)
    elif kind == "parent":
        name = "../candidate.py"
    elif kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        path.symlink_to(target)
    git_transport["untracked"] = name.encode() + b"\0"
    with pytest.raises(RuntimeError, match="leaves repository|unsupported untracked"):
        identity.source_identity(tmp_path)


def test_git_read_failure_cannot_produce_an_empty_source_identity(
    tmp_path: Path, git_transport
) -> None:
    git_transport["fail"] = True
    with pytest.raises(RuntimeError, match="git command failed.*synthetic Git"):
        identity.source_identity(tmp_path)
