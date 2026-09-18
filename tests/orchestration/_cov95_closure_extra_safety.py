"""Offline boundaries with SQLite restricted to this test's temporary directory."""

from __future__ import annotations

import builtins
import io
import os
import socket
import sqlite3
import subprocess
from pathlib import Path
from urllib import request
from urllib.parse import unquote, urlsplit

import boto3
import pytest


def forbidden(*args, **kwargs):
    raise AssertionError("unfaked closure-extra external or host operation")


@pytest.fixture(autouse=True)
def closure_extra_isolation(monkeypatch, tmp_path):
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
    for name in ("Session", "client", "resource"):
        monkeypatch.setattr(boto3, name, forbidden)
    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    original_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        value = os.fsdecode(database)
        if value != ":memory:":
            parsed = urlsplit(value) if value.startswith("file:") else None
            path = Path(unquote(parsed.path) if parsed is not None else value)
            if not path.resolve().is_relative_to(tmp_path):
                raise AssertionError(
                    "SQLite escaped the closure test's temporary directory"
                )
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
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
                    raise AssertionError("unfaked closure-extra protected host file")
            return _open(path, *args, **kwargs)

        monkeypatch.setattr(module, name, guarded)
