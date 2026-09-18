"""Isolated transport and host boundaries for runtime ingest coverage."""

from __future__ import annotations

import builtins
import errno
import faulthandler
import io
import json
import os
import signal
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib import request

import pytest

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


def forbidden(*args, **kwargs):
    raise AssertionError("unfaked external transport or host operation")


def host_path(path: object) -> bool:
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    value = os.fsdecode(path)
    return value != "/dev/null" and any(
        value == root or value.startswith(root + "/")
        for root in ("/dev", "/proc", "/sys", "/var/lib/gpu-fault", "/etc/gpu-fault")
    )


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    handlers = {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL}

    def install_signal(number, handler):
        previous = handlers.get(number, signal.SIG_DFL)
        handlers[number] = handler
        return previous

    monkeypatch.setattr(signal, "signal", install_signal)
    monkeypatch.setattr(
        signal, "getsignal", lambda number: handlers.get(number, signal.SIG_DFL)
    )
    dump_signals = set()
    monkeypatch.setattr(
        faulthandler, "register", lambda number, **kwargs: dump_signals.add(number)
    )
    monkeypatch.setattr(
        faulthandler, "unregister", lambda number: dump_signals.discard(number)
    )
    fake_files = {}
    for index, (path, contents) in enumerate(
        {
            "/proc/sys/kernel/random/boot_id": "cov95-private-boot\n",
            "/proc/self/mountinfo": "",
            "/proc/self/status": "VmRSS: 1024 kB\nVmSize: 4096 kB\nThreads: 2\n",
            "/proc/self/smaps_rollup": "Rss: 1024 kB\nPss: 512 kB\n",
        }.items()
    ):
        target = tmp_path / f"host-file-{index}"
        target.write_text(contents)
        fake_files[path] = target
    monkeypatch.setenv(
        "GPU_FAULT_BOOT_ID_PATH", str(fake_files["/proc/sys/kernel/random/boot_id"])
    )
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "kill", "killpg", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(
        "gpu_fault.collectors.cloud.kubernetes.NODE_RESOURCE_HEARTBEAT_PATH",
        str(tmp_path / "node-resource-heartbeat"),
    )

    def probe_pid(pid, signal):
        if signal != 0:
            forbidden()
        if pid != os.getpid():
            raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", probe_pid)
    for module in (
        "gpu_fault.collectors.sinks",
        "gpu_fault.collectors.gpu.dcgm",
        "gpu_fault.processor.coordinator",
        "gpu_fault.processor.telemetry_spool",
        "gpu_fault.transport.http_client",
    ):
        monkeypatch.setattr(f"{module}.urlopen", forbidden)
    for module, name in ((builtins, "open"), (io, "open"), (os, "open")):
        original = getattr(module, name)

        def guarded_open(path, *args, _open=original, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                path = fake_files.get(os.fsdecode(path), path)
            if host_path(path):
                raise PermissionError(f"unfaked host path: {path}")
            return _open(path, *args, **kwargs)

        monkeypatch.setattr(module, name, guarded_open)
    for name in ("listdir", "scandir", "statvfs", "stat", "lstat"):
        original = getattr(os, name)

        def guarded_host(path, *args, _call=original, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                path = fake_files.get(os.fsdecode(path), path)
            if host_path(path) or _call.__name__ == "statvfs":
                raise FileNotFoundError(errno.ENOENT, "unfaked host probe", path)
            return _call(path, *args, **kwargs)

        monkeypatch.setattr(os, name, guarded_host)


class Clock:
    def __init__(self, seconds=0.0):
        self.seconds = seconds
        self.sleeps = []

    def monotonic(self):
        return self.seconds

    def now(self):
        return NOW + timedelta(seconds=self.seconds)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds


class StopLoop(BaseException):
    pass


class Response:
    def __init__(self, payload=None, *, body=None, status=200, headers=None):
        self.body = json.dumps(payload if payload is not None else {}).encode()
        if body is not None:
            self.body = body
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.closed = False

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class RecordingSink:
    def __init__(self, error=None):
        self.requests = []
        self.error = error

    def post(self, path, payload):
        self.requests.append((path, payload))
        if self.error is not None:
            raise self.error
        return {"accepted": True}


def collector_context(**updates):
    from gpu_fault.collectors.models import CollectorContext

    return CollectorContext(
        **{
            "cluster_id": "cluster-a",
            "runtime_profile_version": "simulated-v1",
            "product": "H100",
            "driver_branch": 575,
            "cuda_version": "12.9",
            **updates,
        }
    )


def host_roots(tmp_path: Path):
    roots = SimpleNamespace()
    for name in ("proc", "net", "rdma", "pci", "mount"):
        path = tmp_path / name
        path.mkdir()
        setattr(roots, name, path)
    return roots
