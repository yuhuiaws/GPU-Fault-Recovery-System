from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

SITE_OPERATION_LOCK = ".gpu-fault-site-operation.lock"
SITE_OPERATION_LOCK_FD_ENV = "GPU_FAULT_SITE_OPERATION_LOCK_FD"
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


@contextmanager
def site_operation_lock(state_dir: Path, *, wait: bool) -> Iterator[int]:
    root = state_dir.expanduser().resolve()
    inherited = inherited_site_operation_lock_fd(root)
    if inherited is not None:
        yield inherited
        return
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(root, Lock())
    acquired = thread_lock.acquire(blocking=wait)
    if not acquired:
        raise SiteOperationBusy("another site mutation is in progress")
    descriptor: int | None = None
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        descriptor = os.open(root / SITE_OPERATION_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        flags = fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError as exc:
            raise SiteOperationBusy("another site mutation is in progress") from exc
        try:
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        thread_lock.release()
