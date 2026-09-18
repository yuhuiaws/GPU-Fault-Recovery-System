"""Real local lock and private-file checks, without cluster or host mutations."""

from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.context import ForkContext, SpawnContext
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr008_controller_lock as locks
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def try_lock(path: str, sender: Any) -> None:
    try:
        with locks.controller_ownership(Path(path)):
            sender.send("acquired")
    except RegionalFixtureError:
        sender.send("busy")
    finally:
        sender.close()


def test_same_thread_reentry_and_other_thread_exclusion(tmp_path: Path) -> None:
    path = tmp_path / "private" / "journal.json"
    assert not locks.ownership_held(path), "an unopened scope has no ownership"
    with locks.controller_ownership(path):
        assert locks.ownership_held(path), "the caller must own the guarded file"
        with locks.controller_ownership(path):
            assert locks.ownership_held(path), "nested operations share one owner"

        def contender() -> None:
            with pytest.raises(RegionalFixtureError, match="another controller"):
                with locks.controller_ownership(path):
                    pytest.fail("a second thread acquired live controller authority")

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(contender).result(timeout=5)
    assert not locks.ownership_held(path), "scope exit must release local authority"
    assert path.with_suffix(".lock").stat().st_mode & 0o777 == 0o600, (
        "the lock file must not expose its private scope"
    )


@pytest.mark.parametrize("method", ["fork", "spawn"])
@pytest.mark.parametrize("runtime_only_path", [False, True])
def test_independent_process_cannot_acquire_live_owner_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    runtime_only_path: bool,
) -> None:
    if runtime_only_path:
        isolated = tmp_path / "runtime-only"
        package = isolated / "gpu_fault"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("__all__ = ()\n", encoding="ascii")
        monkeypatch.syspath_prepend(str(isolated))
        monkeypatch.setenv("PYTHONPATH", str(isolated))
    path = tmp_path / "private" / "journal.json"
    context = ForkContext() if method == "fork" else SpawnContext()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=try_lock, args=(str(path), sender))
    root = Path(__file__).resolve().parents[2]
    parent_paths = list(sys.path)
    parent_pythonpath = os.environ.get("PYTHONPATH")
    try:
        with locks.controller_ownership(path):
            with monkeypatch.context() as child_imports:
                child_imports.syspath_prepend(str(root))
                child_imports.syspath_prepend(str(root / "src"))
                child_imports.setenv(
                    "PYTHONPATH", os.pathsep.join((str(root / "src"), str(root)))
                )
                process.start()
            assert sys.path == parent_paths, "child bootstrap must restore parent paths"
            assert os.environ.get("PYTHONPATH") == parent_pythonpath, (
                "child bootstrap must restore the caller's component environment"
            )
            sender.close()
            assert receiver.poll(10), (
                "the owned child must return a bounded lock result"
            )
            assert receiver.recv() == "busy", (
                "forked state cannot authorize a second owner"
            )
            process.join(10)
            assert process.exitcode == 0, "the contender must terminate normally"
            assert locks.ownership_held(path), (
                "a child must not unlock its parent's descriptor"
            )
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        receiver.close()
        sender.close()
        process.close()
    with locks.controller_ownership(path):
        assert locks.ownership_held(path), "a finished owner cannot leave its lock held"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "public"])
def test_nonprivate_or_substituted_lock_is_refused(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    journal = directory / "state.json"
    path = journal.with_suffix(".lock")
    original = directory / "original"
    original.touch(mode=0o600)
    if kind == "symlink":
        path.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    else:
        path.touch(mode=0o644)
        path.chmod(0o644)
    with pytest.raises((OSError, RegionalFixtureError)):
        with locks.controller_ownership(journal):
            pytest.fail("an unproven file cannot grant controller ownership")


def test_shared_directory_is_not_silently_made_authoritative(tmp_path: Path) -> None:
    directory = tmp_path / "shared"
    directory.mkdir()
    directory.chmod(0o755)
    with pytest.raises(RegionalFixtureError, match="directory must be private"):
        with locks.controller_ownership(directory / "state.json"):
            pytest.fail("shared directory cannot hold authoritative run state")
    assert directory.stat().st_mode & 0o777 == 0o755, (
        "unowned directory modes must remain unchanged"
    )


def test_replaced_lock_inode_invalidates_ownership(tmp_path: Path) -> None:
    journal = tmp_path / "private" / "state.json"
    path = journal.with_suffix(".lock")
    with locks.controller_ownership(journal):
        path.rename(path.with_suffix(".previous"))
        path.touch(mode=0o600)
        with pytest.raises(RegionalFixtureError, match="ownership"):
            locks.ownership_held(journal)
    assert not locks.ownership_held(journal), (
        "replacement cannot leave stale in-memory authority"
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"phase":"ARMED","phase":"CLOSED"}', "duplicate"),
        ("[]", "must be an object"),
        (" " * 262145, "size limit"),
    ],
)
def test_private_journal_requires_bounded_unambiguous_json(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "state.json"
    path.touch(mode=0o600)
    path.write_text(content)
    with pytest.raises(RegionalFixtureError, match=message):
        locks.read_private_document(path)


def test_private_journal_preserves_values(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.touch(mode=0o600)
    path.write_text('{"phase":"ARMED","action_started":true}')
    assert locks.read_private_document(path) == {
        "phase": "ARMED",
        "action_started": True,
    }, "private state parsing must not reinterpret its values"


@pytest.mark.parametrize("valid", [True, False])
def test_host_identity_requires_a_stable_machine_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid: bool
) -> None:
    path = tmp_path / "machine-id"
    path.write_bytes(b"a" * 32 + b"\n" if valid else b"unknown")
    monkeypatch.setattr(locks, "Path", lambda value: path)
    if valid:
        expected = hashlib.sha256(
            b"a" * 32 + b":" + str(os.getuid()).encode()
        ).hexdigest()
        assert locks.host_identity() == expected, (
            "only a digest may identify the deployment host"
        )
    else:
        with pytest.raises(RegionalFixtureError, match="host identity"):
            locks.host_identity()
