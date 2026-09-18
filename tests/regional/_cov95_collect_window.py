"""A temporary host filesystem and command protocol for the window probe."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_window_probe as probe
from tests.regional._cov95_collect_net import (
    FakeSocket,
    isolate_paths,
    local_socket_module,
)


@pytest.fixture
def window_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    paths = isolate_paths(monkeypatch, probe, tmp_path)
    for name in ("/run", "/tmp", "/proc/sys/kernel/random", "/etc/gpu-fault"):
        paths(name).mkdir(parents=True, exist_ok=True)
    paths("/proc/sys/kernel/random/boot_id").write_text("boot-a\n")
    probe.COLLECTOR_ENV.write_text(
        "# comment\n\nignored\nGPU_FAULT_CLUSTER_ID='cluster-a'\n"
        "GPU_FAULT_EXPECTED_GPU_COUNT=2\n"
        "GPU_FAULT_CONTROL_PLANE_TOKEN=example-only\n"
        "GPU_FAULT_CONTROL_PLANE_URL=https://control.invalid\n"
    )
    for path in (probe.COLLECTOR_CLI, probe.VENV_PYTHON):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    source = tmp_path / "copied-probe.py"
    source.write_text("# inert watchdog file fixture\n")
    monkeypatch.setattr(probe, "__file__", str(source))
    monkeypatch.setattr(probe.shutil, "which", lambda name, **kw: f"/fake/{name}")
    monkeypatch.setattr(probe.secrets, "token_hex", lambda size: "example-invalid")
    host = SimpleNamespace(
        paths=paths,
        calls=[],
        states={},
        failures={},
        rules={},
        emitted=[],
        stdout=None,
        cli_status=0,
        writes=[],
        closed=[],
        timer_pid="123",
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        host.calls.append((command, kwargs))
        for prefix, failure in host.failures.items():
            if tuple(command[: len(prefix)]) == prefix:
                if isinstance(failure, list):
                    if failure:
                        raise failure.pop(0)
                else:
                    raise failure
        stdout, status = "", 0
        if command[0] == "systemd-run":
            unit = next(
                value.split("=", 1)[1]
                for value in command
                if value.startswith("--unit=")
            )
            host.states[f"{unit}.timer"] = "active"
        elif command[:2] == ["systemctl", "show"]:
            unit = command[2]
            active = host.states.get(unit, "active")
            stdout = (
                f"ActiveState={active}\nLoadState=loaded\nMainPID={host.timer_pid}\n"
                "InvocationID=invocation-a\nSubState=running\nNRestarts=0\nignored\n"
            )
        elif command[0] == "systemctl" and command[1] in {"stop", "start", "restart"}:
            host.states[command[2]] = "inactive" if command[1] == "stop" else "active"
        elif command[0] == "nvidia-smi":
            stdout = "invalid\n0, GPU-aaaaaaaa, 00000000:AF:00.0, NVIDIA H100\n"
        elif command[0] == str(probe.COLLECTOR_CLI):
            stdout, status = host.stdout or '{"pending": 1}\n', host.cli_status
            if command[-1] == "requeue-dead":
                status = 2
        elif command[0] == str(probe.VENV_PYTHON):
            stdout = (
                host.stdout
                if host.stdout is not None
                else '{"sink_outcome":"accepted"}\n'
            )
        elif command[0] == "/fake/iptables":
            operation = command[1]
            if operation == "-S":
                stdout = "\n".join(
                    f"-A OUTPUT -d {ip} --comment test-net"
                    for ip, count in host.rules.items()
                    for _ in range(count)
                )
            else:
                ip = command[command.index("-d") + 1]
                count = host.rules.get(ip, 0)
                if operation == "-C":
                    status = 0 if count else 1
                elif operation == "-I":
                    host.rules[ip] = count + 1
                elif operation == "-D":
                    host.rules[ip] = count - 1
                else:
                    raise AssertionError(f"unexpected firewall operation {operation}")
        return subprocess.CompletedProcess(
            command, status, stdout, "fixture-error" if status else ""
        )

    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(probe, "emit", host.emitted.append)
    monkeypatch.setattr(
        probe,
        "os",
        SimpleNamespace(
            **{
                **vars(os),
                "open": lambda path, flags: 42,
                "write": lambda fd, data: host.writes.append((fd, data)) or len(data),
                "close": host.closed.append,
                "getpid": lambda: 555,
                "uname": lambda: SimpleNamespace(nodename="node-a"),
            }
        ),
    )
    monkeypatch.setattr(
        probe,
        "socket",
        local_socket_module(
            getaddrinfo=lambda *a, **k: [
                (None, None, None, None, ("192.0.2.1", 443)),
                (None, None, None, None, ("192.0.2.1", 443)),
            ],
            create_connection=lambda address, timeout: FakeSocket(),
        ),
    )
    return host


def window_arguments(**changes: Any) -> Any:
    return SimpleNamespace(
        **{
            "run_id": "window-a",
            "unit": probe.ALLOWED_UNITS[0],
            "restore_seconds": 600,
            "env": [],
            "unset": ["GPU_FAULT_EXPECTED_GPU_COUNT"],
            "shadow_nvidia_smi": "",
            **changes,
        }
    )


def write_records(collector: str, values: list[Any]) -> Path:
    probe.OUTBOX_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = probe.OUTBOX_DIRECTORY / f"{collector}.ndjson"
    path.write_text(
        "".join(
            (value if isinstance(value, str) else json.dumps(value)) + "\n"
            for value in values
        )
    )
    return path
