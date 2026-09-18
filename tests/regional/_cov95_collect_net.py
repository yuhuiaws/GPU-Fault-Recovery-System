"""Strictly local test boundaries for the COLLECT/NET coverage batch."""

from __future__ import annotations

import os
import socket
import subprocess
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class StopLoop(BaseException):
    """End an intentionally unbounded worker at a controlled test boundary."""


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("unmocked external transport or host operation")


@pytest.fixture(autouse=True)
def no_external_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)
    for name in ("system", "kill", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, forbidden)


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []
        self.on_sleep: Any = None

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(seconds)


def isolate_paths(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, root: Path
) -> Any:
    def local_path(value: Any = ".", *parts: Any) -> Path:
        path = Path(value, *parts)
        if path.is_absolute() and not path.is_relative_to(root):
            return root / path.relative_to("/")
        return path

    for name, value in vars(module).copy().items():
        if isinstance(value, Path) and value.is_absolute():
            monkeypatch.setattr(module, name, local_path(value))
    monkeypatch.setattr(module, "Path", local_path)
    return local_path


class FakeSocket:
    def __init__(self, reads: tuple[bytes, ...] = ()) -> None:
        self.reads = deque(reads)
        self.sent: list[bytes] = []
        self.options: list[tuple[Any, ...]] = []
        self.closed = False
        self.bound: Any = None
        self.backlog: int | None = None
        self.accepted: deque[FakeSocket] = deque()

    def __enter__(self) -> FakeSocket:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def recv(self, size: int) -> bytes:
        assert size == 65536, "relay must use its bounded receive buffer"
        return self.reads.popleft()

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True

    def setsockopt(self, *args: Any) -> None:
        self.options.append(args)

    def bind(self, address: Any) -> None:
        self.bound = address

    def listen(self, backlog: int) -> None:
        self.backlog = backlog

    def accept(self) -> tuple[FakeSocket, tuple[str, int]]:
        if not self.accepted:
            raise StopLoop()
        return self.accepted.popleft(), ("127.0.0.1", 1)


def local_socket_module(**overrides: Any) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            **{name: value for name, value in vars(socket).items()},
            "socket": forbidden,
            "create_connection": forbidden,
            "getaddrinfo": forbidden,
            **overrides,
        }
    )


def stop_sleep(seconds: float) -> None:
    raise StopLoop()
