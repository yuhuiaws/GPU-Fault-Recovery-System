"""Test-only boundaries: no subprocess, network, host files or SQL connections."""

from __future__ import annotations

import builtins
import io
import os
import signal
import socket
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import request

import pytest

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


def forbidden(*args, **kwargs):
    raise AssertionError("unfaked residual runner external or host operation")


@pytest.fixture(autouse=True)
def residual_isolation(monkeypatch):
    try:
        import psycopg
    except ImportError:
        pass
    else:
        monkeypatch.setattr(psycopg, "connect", forbidden)
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "kill", "killpg", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(os, "umask", lambda value: 0o077)
    monkeypatch.setattr(os, "access", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(
        "scripts.e2e.regional.live_driver_guard.source_digest",
        lambda: "private-residual-test-source",
    )
    handlers = {}

    def install_signal(number, handler):
        previous = handlers.get(number, signal.SIG_DFL)
        handlers[number] = handler
        return previous

    monkeypatch.setattr(
        signal, "getsignal", lambda number: handlers.get(number, signal.SIG_DFL)
    )
    monkeypatch.setattr(signal, "signal", install_signal)
    for module, name in ((builtins, "open"), (io, "open"), (os, "open")):
        original = getattr(module, name)

        def guarded(path, *args, _open=original, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                value = os.fsdecode(path)
                if any(
                    value == root or value.startswith(root + "/")
                    for root in (
                        "/proc",
                        "/sys",
                        "/dev",
                        "/tokens",
                        "/tls",
                        "/etc/gpu-fault",
                        "/var/lib/gpu-fault",
                    )
                ):
                    raise AssertionError("unfaked residual runner host file")
            return _open(path, *args, **kwargs)

        monkeypatch.setattr(module, name, guarded)


class StopLoop(BaseException):
    pass


class Clock:
    def __init__(self):
        self.seconds = 0.0
        self.waits = []

    def monotonic(self):
        return self.seconds

    def now(self):
        return NOW + timedelta(seconds=self.seconds)

    def time(self):
        return self.now().timestamp()

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.seconds += seconds


def redirect_path(monkeypatch, module, mapping):
    def local_path(value):
        return mapping.get(str(value), Path(value))

    monkeypatch.setattr(module, "Path", local_path)
