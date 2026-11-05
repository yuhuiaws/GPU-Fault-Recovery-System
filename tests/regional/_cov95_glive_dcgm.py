"""More of the node for the DCGM field-replay host controller's tests.

``tests/regional/_dcgm_probe_fakes.sandbox`` fakes systemd, the collector's
process environment, the replay tool's loopback and the clock. The survey, the
leftover checks and the override file also read the journal's boot list,
``nvidia-smi``, the real exporter's scrape and ``/proc``. ``node`` adds those
to the same sandbox: ``run`` answers ``journalctl`` and ``nvidia-smi`` from
scripted text, ``loopback`` answers scripted URLs (the real exporter, a tool
that misbehaves) before the fake tool, and ``/proc`` resolves under
``tmp_path``. Nothing touches the host.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_dcgm_probe as probe
from tests.regional._dcgm_probe_fakes import RUN, FakeSystemd, sandbox

#: The node's current boot (``/proc/sys/kernel/random/boot_id``).
BOOT_ID = "6f1c0e3e-1a2b-4c3d-8e4f-0123456789ab"
UUID0 = "GPU-11111111-2222-3333-4444-555555555555"
BDF0 = "00000000:53:00.0"
LABELS0 = f'gpu="0",UUID="{UUID0}",pci_bus_id="{BDF0}",modelName="NVIDIA H200"'
#: A real exporter scrape: two fields of GPU 0 and one sample without a UUID.
SCRAPE = (
    "# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature (in C).\n"
    "# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n"
    f"DCGM_FI_DEV_GPU_TEMP{{{LABELS0}}} 41\n"
    'DCGM_FI_DEV_GPU_TEMP{gpu="9"} 40\n'
    "# HELP DCGM_FI_DEV_POWER_USAGE Power draw (in W).\n"
    "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\n"
    f"DCGM_FI_DEV_POWER_USAGE{{{LABELS0}}} 312.5\n"
)
INVENTORY = f"0, {UUID0}, {BDF0}, NVIDIA H200\n"


class Node:
    """The sandbox's systemd and tool, plus the node's other read sources."""

    def __init__(self, fake: FakeSystemd, root: Path) -> None:
        self.fake = fake
        self.proc = root / "proc"
        #: ``journalctl --list-boots`` output.
        self.boots = ""
        #: ``nvidia-smi --query-gpu`` CSV output.
        self.inventory = INVENTORY
        #: URL -> ``(status, body)`` or an exception the loopback raises.
        self.answers: dict[str, Any] = {}
        self.base_run = fake.run
        self.base_loopback = fake.loopback

    def path(self, value: Any, *parts: Any) -> Path:
        path = Path(value, *parts)
        if path.is_relative_to("/proc"):
            return self.proc / path.relative_to("/proc")
        return path

    def write(self, absolute: str, content: str | bytes) -> Path:
        target = self.path(absolute)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        return target

    def run(self, command: list[str], *, check: bool = True, timeout: int = 120) -> Any:
        if command[0] == "nvidia-smi":
            output = self.inventory
        elif command[:2] == ["journalctl", "--list-boots"]:
            output = self.boots
        else:
            return self.base_run(command, check=check, timeout=timeout)
        self.fake.commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, output, "")

    def loopback(
        self, url: str, *, method: str = "GET", timeout: int = 10
    ) -> tuple[int, bytes]:
        answer = self.answers.get(f"{method} {url}", self.answers.get(url))
        if answer is None:
            return self.base_loopback(url, method=method, timeout=timeout)
        if isinstance(answer, Exception):
            raise answer
        return answer


def node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Node:
    """The sandbox (baseline collector on the real exporter) plus ``Node``."""

    fake = sandbox(monkeypatch, tmp_path)
    host = Node(fake, tmp_path / "host")
    monkeypatch.setattr(probe, "run", host.run)
    monkeypatch.setattr(probe, "loopback", host.loopback)
    monkeypatch.setattr(probe, "Path", host.path)
    host.write("/proc/sys/kernel/random/boot_id", BOOT_ID + "\n")
    return host


def installed_sources(*names: str) -> Path:
    """The run directory as ``install-source`` leaves it with ``names``."""

    root = probe.run_dir(RUN)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text(f"# reviewed {name}\n", encoding="utf-8")
    return root


def deploy_arguments(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "run_id": RUN,
        "tool": probe.FIELD_TOOL,
        "profile": "thermal",
        "gpu_uuid": "",
        "max_seconds": 1800,
        "port": 9412,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def repoint_arguments(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "run_id": RUN,
        "port": 9412,
        "restore_seconds": 1800,
        "inventory_wrapper": False,
        "boot_id": "",
        "unknown_expected_count": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)
