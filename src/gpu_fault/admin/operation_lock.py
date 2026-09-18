from __future__ import annotations

import fcntl
import json
import os
import stat
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, Iterator

SITE_OPERATION_LOCK = ".gpu-fault-site-operation.lock"
SITE_OPERATION_LOCK_FD_ENV = "GPU_FAULT_SITE_OPERATION_LOCK_FD"
_HOLDER_COMMAND_MAX_CHARS = 200
_THREAD_LOCKS_GUARD = Lock()
_THREAD_LOCKS: dict[Path, Lock] = {}
_THREAD_OWNERS: dict[Path, int] = {}


class SiteOperationBusy(RuntimeError):
    pass


def _holds_exclusive_lock(descriptor: int) -> bool:
    if descriptor < 3:
        return False
    try:
        information = Path(f"/proc/self/fdinfo/{descriptor}").read_text(
            encoding="ascii"
        )
    except (OSError, ValueError):
        return False
    # fdinfo reports locks on this open file description, not another open of
    # the same inode. Checking the inode alone accepts unrelated, unlocked FDs.
    return any(
        fields[:1] == ["lock:"] and fields[2:5] == ["FLOCK", "ADVISORY", "WRITE"]
        for line in information.splitlines()
        if (fields := line.split())
    )


def inherited_site_operation_lock_fd(state_dir: Path) -> int | None:
    raw = os.getenv(SITE_OPERATION_LOCK_FD_ENV, "").strip()
    if not raw:
        return None
    try:
        descriptor = int(raw)
        inherited = os.fstat(descriptor)
        expected = os.stat(state_dir.expanduser().resolve() / SITE_OPERATION_LOCK)
    except (OSError, ValueError):
        return None
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        return None
    return descriptor if _holds_exclusive_lock(descriptor) else None


def inherited_lock_pass_fds() -> tuple[int, ...]:
    raw = os.getenv(SITE_OPERATION_LOCK_FD_ENV, "").strip()
    if not raw:
        return ()
    try:
        descriptor = int(raw)
        os.fstat(descriptor)
    except (OSError, ValueError):
        return ()
    return (descriptor,) if _holds_exclusive_lock(descriptor) else ()


def _holder_record() -> dict[str, Any]:
    command = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else ""
    return {
        "pid": os.getpid(),
        "command": command[:_HOLDER_COMMAND_MAX_CHARS],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _record_holder(descriptor: int) -> None:
    payload = json.dumps(_holder_record(), sort_keys=True).encode()
    try:
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, payload, 0)
    except OSError:
        # The record is diagnostic only; the flock is what serializes.
        return


def _describe_holder(lock_file: Path) -> str:
    try:
        holder = json.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(holder, dict) or holder.get("pid") is None:
        return ""
    command = str(holder.get("command") or "").strip()
    return f" (held by pid {holder['pid']}" + (f": {command})" if command else ")")


def _busy(lock_file: Path) -> SiteOperationBusy:
    return SiteOperationBusy(
        "another administrator mutation is in progress" + _describe_holder(lock_file)
    )


@contextmanager
def site_operation_lock(state_dir: Path, *, wait: bool) -> Iterator[int]:
    """Hold the one site-wide operation lock.

    Every administrator mutation on a site -- deploy, join, remove, uninstall,
    config, workflow-reconcile -- takes this lock. The holder's pid and command
    are recorded in the lock file so a refused caller can name who is running.
    ``wait`` is for the deploy host script that deliberately queues behind a
    running operation; the CLI entry points fail fast.
    """

    root = state_dir.expanduser().resolve()
    inherited = inherited_site_operation_lock_fd(root)
    lock_file = root / SITE_OPERATION_LOCK
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(root, Lock())
        reentrant = _THREAD_OWNERS.get(root) == get_ident()
    if inherited is not None and reentrant:
        yield inherited
        return
    acquired = thread_lock.acquire(blocking=wait)
    if not acquired:
        raise _busy(lock_file)
    descriptor: int | None = None
    try:
        with _THREAD_LOCKS_GUARD:
            _THREAD_OWNERS[root] = get_ident()
        inherited = inherited_site_operation_lock_fd(root)
        if inherited is not None:
            yield inherited
            return
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        descriptor = os.open(
            lock_file, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SiteOperationBusy("site operation lock is not a regular file")
        os.fchmod(descriptor, 0o600)
        flags = fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError as exc:
            raise _busy(lock_file) from exc
        _record_holder(descriptor)
        try:
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with _THREAD_LOCKS_GUARD:
            _THREAD_OWNERS.pop(root, None)
        thread_lock.release()
