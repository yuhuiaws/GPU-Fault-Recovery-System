from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import boto3
import pytest

from gpu_fault.transport.http_client import CONNECTION_POOL
from scripts.e2e.regional.auth015_http import worker_source


def forbidden(*args, **kwargs):
    raise AssertionError(
        "scenario-peer test reached an unfaked external or host operation"
    )


@pytest.fixture(name="scenario_peer_transport_guard", autouse=True)
def scenario_peer_transport_guard_fixture(monkeypatch):
    try:
        import psycopg
    except ImportError:
        pass
    else:
        monkeypatch.setattr(psycopg, "connect", forbidden)
    ports = set()
    children = []
    authorized_command = None
    source_sha256 = hashlib.sha256(worker_source().encode()).hexdigest()
    original_run = subprocess.run
    original_popen = subprocess.Popen
    original_kill = os.kill
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "kill", "killpg", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("Session", "client", "resource"):
        monkeypatch.setattr(boto3, name, forbidden)
    original_connect = socket.socket.connect
    original_bind = socket.socket.bind
    original_lookup = socket.getaddrinfo

    def connect(connection, address):
        if (
            not isinstance(address, tuple)
            or address[0] != "127.0.0.1"
            or address[1] not in ports
        ):
            raise AssertionError(
                "scenario-peer transport escaped its own TLS loopback listener"
            )
        return original_connect(connection, address)

    def bind(connection, address):
        if address != ("127.0.0.1", 0):
            raise AssertionError("scenario-peer listener is not a fresh loopback port")
        return original_bind(connection, address)

    def lookup(host, port, *args, **kwargs):
        if host != "127.0.0.1" or port not in ports:
            raise AssertionError(
                "scenario-peer attempted an unowned name or port lookup"
            )
        return original_lookup(host, port, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket.socket, "bind", bind)
    monkeypatch.setattr(socket.socket, "sendto", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", lookup)

    def start_owned(command, *args, **kwargs):
        if command is not authorized_command:
            forbidden()
        child = original_popen(command, *args, **kwargs)
        children.append(child)
        return child

    def kill_owned(pid, selected_signal):
        if selected_signal not in {signal.SIGTERM, signal.SIGKILL} or not any(
            child.pid == pid and child.poll() is None for child in children
        ):
            forbidden()
        return original_kill(pid, selected_signal)

    def run_owned(command, **kwargs):
        nonlocal authorized_command
        if (
            not isinstance(command, list)
            or len(command) != 6
            or command[:5] != [sys.executable, "-I", "-S", "-B", "-c"]
            or hashlib.sha256(command[5].encode()).hexdigest() != source_sha256
            or set(kwargs) != {"input", "stdout", "stderr", "env", "timeout", "check"}
            or kwargs.get("check") is not False
            or not isinstance(kwargs.get("input"), bytes)
            or not 0 < kwargs.get("timeout", 0) <= 5
            or kwargs.get("stdout") != subprocess.PIPE
            or kwargs.get("stderr") != subprocess.PIPE
        ):
            forbidden()
        payload = json.loads(kwargs["input"])
        target = urlsplit(payload["url"])
        if (
            target.scheme != "https"
            or target.hostname != "127.0.0.1"
            or target.port not in ports
            or payload.get("method") != "GET"
            or payload.get("limit") != 8192
            or kwargs.get("env")
            != {
                "HOME": "/tmp",
                "PATH": os.defpath,
                "AWS_CONFIG_FILE": os.devnull,
                "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
                "AWS_EC2_METADATA_DISABLED": "true",
                "KUBECONFIG": os.devnull,
            }
        ):
            forbidden()
        authorized_command = command
        try:
            return original_run(command, **kwargs)
        finally:
            authorized_command = None

    monkeypatch.setattr(subprocess, "run", run_owned)
    monkeypatch.setattr(subprocess, "Popen", start_owned)
    monkeypatch.setattr(os, "kill", kill_owned)
    for module, name in ((builtins, "open"), (io, "open"), (os, "open")):
        original = getattr(module, name)

        def guarded(path, *args, _open=original, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                value = Path(os.fsdecode(path))
                if any(
                    value == root or root in value.parents
                    for root in map(
                        Path,
                        (
                            "/proc",
                            "/sys",
                            "/dev",
                            "/tokens",
                            "/tls",
                            "/etc/gpu-fault",
                            "/var/lib/gpu-fault",
                        ),
                    )
                ):
                    raise AssertionError("scenario-peer accessed a protected host path")
            return _open(path, *args, **kwargs)

        monkeypatch.setattr(module, name, guarded)
    CONNECTION_POOL.close()
    try:
        yield ports
    finally:
        CONNECTION_POOL.close()
        leaked = [child for child in children if child.poll() is None]
        for child in leaked:
            child.kill()
            child.wait(timeout=3)
        for child in children:
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()
        assert leaked == [], (
            "the HTTP proof returned while its owned worker was still alive"
        )
