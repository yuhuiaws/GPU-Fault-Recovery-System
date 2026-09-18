"""Local node-probe filesystem and command fixtures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional._cov95_collect_net import Clock, isolate_paths


@pytest.fixture
def node_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    paths = isolate_paths(monkeypatch, probe, tmp_path)
    monkeypatch.setattr(
        probe,
        "FM_STATE_CANDIDATES",
        tuple(paths(value) for value in probe.FM_STATE_CANDIDATES),
    )
    for name in (
        "/proc/sys/kernel/random",
        "/etc/gpu-fault",
        "/etc/systemd/system",
        "/usr/bin",
        "/sys/bus/pci/drivers/efa",
        "/var/lib/gpu-fault",
    ):
        paths(name).mkdir(parents=True, exist_ok=True)
    probe.BOOT_ID_FILE.write_text("boot-a")
    probe.COLLECTOR_ENV.write_text(
        "# ignored\n\nmalformed\nGPU_FAULT_EXPECTED_GPU_COUNT='2'\n"
        "GPU_FAULT_HOST_INTERVAL_SECONDS=15\nIGNORED_KEY=value\n"
    )
    probe.COLLECTOR_ENV.chmod(0o600)
    source = tmp_path / "probe-copy-input.py"
    source.write_text("# inert recovery fixture\n")
    monkeypatch.setattr(probe, "__file__", str(source))
    clock = Clock()
    monkeypatch.setattr(probe, "time", clock)
    host = SimpleNamespace(
        paths=paths,
        clock=clock,
        calls=[],
        emitted=[],
        failures={},
        unit_status={},
        power_limit=300,
        power_stdout=None,
        persistence="Enabled\n",
        gpu_stdout="bad-row\n0, GPU-aaaaaaaa, 00000000:AF:00.0, NVIDIA H100\n",
        writes=[],
        closed=[],
        started="",
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
        status, stdout = 0, ""
        if command[:2] == ["systemctl", "show"]:
            unit = command[2]
            if "-p" in command:
                stdout = f"ExecMainStartTimestamp={host.started}"
            else:
                status, load, active = host.unit_status.get(
                    unit, (0, "loaded", "active")
                )
                stdout = (
                    f"LoadState={load}\nActiveState={active}\nSubState=running\n"
                    "MainPID=123\nInvocationID=invocation-a\nignored\n"
                )
        elif command[0] == "nvidia-smi":
            query = command[1]
            if query == "-pl":
                host.power_limit = int(command[2])
            elif "power.draw" in query:
                stdout = (
                    host.power_stdout
                    if host.power_stdout is not None
                    else (f"0, GPU-aaaaaaaa, 150, {host.power_limit}, 100, 300, 90\n")
                )
            elif "persistence_mode" in query or query == "-q":
                stdout = host.persistence
            else:
                stdout = host.gpu_stdout
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
            }
        ),
    )
    return host


def process(
    root: Path, pid: int, *, cgroup: str = "", comm: str | None = "python"
) -> Path:
    path = root / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    (path / "cgroup").write_text(cgroup)
    if comm is not None:
        (path / "comm").write_text(comm)
    return path
