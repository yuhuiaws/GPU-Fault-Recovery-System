from __future__ import annotations

import fcntl
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

SITE_OPERATION_LOCK = ".gpu-fault-site-operation.lock"
SITE_OPERATION_LOCK_FD_ENV = "GPU_FAULT_SITE_OPERATION_LOCK_FD"
_HOLDER_COMMAND_MAX_CHARS = 200
_THREAD_LOCKS_GUARD = Lock()
_THREAD_LOCKS: dict[Path, Lock] = {}


class SiteOperationBusy(RuntimeError):
    pass


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
    return descriptor


def inherited_lock_pass_fds() -> tuple[int, ...]:
    raw = os.getenv(SITE_OPERATION_LOCK_FD_ENV, "").strip()
    if not raw:
        return ()
    try:
        descriptor = int(raw)
        os.fstat(descriptor)
    except (OSError, ValueError):
        return ()
    return (descriptor,)


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
    if inherited is not None:
        yield inherited
        return
    lock_file = root / SITE_OPERATION_LOCK
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(root, Lock())
    acquired = thread_lock.acquire(blocking=wait)
    if not acquired:
        raise _busy(lock_file)
    descriptor: int | None = None
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        descriptor = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
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
        thread_lock.release()
