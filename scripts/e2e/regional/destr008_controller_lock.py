"""Private local controller ownership; the live runner also holds its site lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock, get_ident
from typing import Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError


@dataclass
class Owner:
    mutex: RLock = field(default_factory=RLock)
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    thread: int | None = None
    depth: int = 0


_owners: dict[Path, Owner] = {}
_registry_lock = Lock()


def _after_fork() -> None:
    global _owners, _registry_lock
    for owner in _owners.values():
        if owner.descriptor is not None and owner.identity is not None:
            try:
                current = os.fstat(owner.descriptor)
                if (current.st_dev, current.st_ino) == owner.identity:
                    os.close(owner.descriptor)
            except OSError:
                pass
    _owners = {}
    _registry_lock = Lock()


os.register_at_fork(after_in_child=_after_fork)


def host_identity() -> str:
    value = Path("/etc/machine-id").read_bytes().strip()
    if not re.fullmatch(rb"[0-9a-fA-F]{32}", value):
        raise RegionalFixtureError("controller host identity is unavailable")
    return hashlib.sha256(value + b":" + str(os.getuid()).encode()).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RegionalFixtureError("controller journal has duplicate fields")
        value[key] = item
    return value


def read_private_document(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _private_regular(descriptor, path)
        if os.fstat(descriptor).st_size > 262144:
            raise RegionalFixtureError("controller journal exceeds its size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            value = json.loads(source.read(262145), object_pairs_hook=_unique)
        if not isinstance(value, dict):
            raise RegionalFixtureError("controller journal must be an object")
        return value
    finally:
        os.close(descriptor)


def _private_regular(descriptor: int, path: Path) -> None:
    opened = os.fstat(descriptor)
    current = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.getuid()
        or opened.st_mode & 0o077
        or opened.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise RegionalFixtureError("controller lock ownership is unproven")


def _owner(path: Path) -> Owner:
    with _registry_lock:
        return _owners.setdefault(path, Owner())


def ownership_held(journal: Path) -> bool:
    path = journal.with_suffix(".lock").absolute()
    owner = _owner(path)
    if owner.thread != get_ident() or owner.depth < 1 or owner.descriptor is None:
        return False
    _private_regular(owner.descriptor, path)
    return True


@contextmanager
def controller_ownership(journal: Path) -> Iterator[None]:
    path = journal.with_suffix(".lock").absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = path.parent.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != os.getuid()
        or directory.st_mode & 0o077
    ):
        raise RegionalFixtureError("controller journal directory must be private")
    owner = _owner(path)
    if not owner.mutex.acquire(blocking=False):
        raise RegionalFixtureError("another controller owns this run")
    try:
        if owner.depth == 0:
            descriptor = os.open(
                path,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            try:
                _private_regular(descriptor, path)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(descriptor)
                raise
            owner.descriptor = descriptor
            current = os.fstat(descriptor)
            owner.identity = current.st_dev, current.st_ino
            owner.thread = get_ident()
        elif owner.descriptor is None or owner.thread != get_ident():
            raise RegionalFixtureError("controller lock is internally inconsistent")
        _private_regular(owner.descriptor, path)
        owner.depth += 1
        try:
            yield
        finally:
            owner.depth -= 1
            if owner.depth == 0 and owner.descriptor is not None:
                try:
                    fcntl.flock(owner.descriptor, fcntl.LOCK_UN)
                finally:
                    try:
                        os.close(owner.descriptor)
                    finally:
                        owner.descriptor = None
                        owner.identity = None
                        owner.thread = None
    except BlockingIOError:
        raise RegionalFixtureError("another controller owns this run") from None
    finally:
        try:
            if owner.depth == 0 and owner.descriptor is not None:
                try:
                    os.close(owner.descriptor)
                finally:
                    owner.descriptor = None
                    owner.identity = None
                    owner.thread = None
        finally:
            owner.mutex.release()
