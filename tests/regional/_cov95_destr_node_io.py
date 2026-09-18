from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import destructive_node_probe as probe
from tests.regional._cov95_destr_warm import NOW, Clock


class NodeIO:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = tmp_path
        self.clock = Clock()
        self.boot = tmp_path / "boot-id"
        self.boot.write_text("boot-io\n", encoding="utf-8")
        self.kmsg = tmp_path / "fake-kmsg"
        self.kmsg.touch()
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.sampler = tmp_path / "sampler"
        self.quiesce = tmp_path / "quiesce"
        self.ledger = tmp_path / "ledger.db"
        self.inventory = "0, GPU-A, 00000000:0A:00.0, NVIDIA A100\n1, GPU-B, 00000000:0B:00.0, NVIDIA A100"
        self.compute = ""
        self.sample_output = "GPU-A\nGPU-B\n"
        self.sample_returncode = 0
        self.sample_timeout = False
        self.journal = ""
        self.service_outputs: dict[str, tuple[int, str]] = {}
        self.timers = ""
        self.active = False
        self.start_samples = True
        self.command_failures: dict[tuple[str, ...], int] = {}
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.records: list[dict[str, Any]] = []
        self.opened: list[tuple[str, int]] = []
        self.writes: list[bytes] = []
        self.closed: list[int] = []
        self.write_failure = False
        self.kmsg_writable = True
        monkeypatch.setattr(probe, "Path", self.path)
        monkeypatch.setattr(probe, "LEDGER", self.ledger)
        monkeypatch.setattr(probe, "QUIESCE_STATE_DIR", self.quiesce)
        monkeypatch.setattr(probe, "SAMPLER_DIR", self.sampler)
        monkeypatch.setattr(probe, "time", self.clock)
        monkeypatch.setattr(probe, "datetime", self.clock)
        monkeypatch.setattr(
            probe,
            "subprocess",
            SimpleNamespace(
                run=self.run,
                PIPE=subprocess.PIPE,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
        )
        monkeypatch.setattr(
            probe,
            "os",
            SimpleNamespace(
                access=lambda path, mode: self.kmsg_writable,
                W_OK=os.W_OK,
                O_WRONLY=os.O_WRONLY,
                O_CLOEXEC=os.O_CLOEXEC,
                open=self.open,
                write=self.write,
                close=self.closed.append,
                fsync=os.fsync,
            ),
        )
        client_pod_uid = probe.client_pod_uid
        monkeypatch.setattr(
            probe,
            "client_pod_uid",
            lambda pid: client_pod_uid(pid, proc_root=self.proc),
        )
        monkeypatch.setattr(probe, "emit", lambda value: self.records.append(value))

    def path(self, value: Any, *parts: str) -> Path:
        if str(value) == "/proc/sys/kernel/random/boot_id":
            return self.boot
        if str(value) == "/dev/kmsg":
            return self.kmsg
        return Path(value, *parts)

    def open(self, path: str, flags: int) -> int:
        assert path == "/dev/kmsg", path
        self.opened.append((path, flags))
        return 99999

    def write(self, descriptor: int, value: bytes) -> int:
        assert descriptor == 99999, descriptor
        if self.write_failure:
            raise OSError("fake write failure")
        self.writes.append(value)
        return len(value)

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), kwargs))
        for prefix, returncode in self.command_failures.items():
            if tuple(command[: len(prefix)]) == prefix:
                return subprocess.CompletedProcess(
                    command, returncode, "", "fake command error"
                )
        code = 0
        if command[0] == "nvidia-smi":
            query = command[1]
            if query.startswith("--query-compute-apps"):
                output = self.compute
            elif query == "--query-gpu=uuid":
                if self.sample_timeout:
                    raise subprocess.TimeoutExpired(command, 10)
                output, code = self.sample_output, self.sample_returncode
            else:
                output = self.inventory
        elif command[0] == "journalctl":
            output = self.journal
        elif command[:2] == ["systemctl", "show"]:
            code, output = self.service_outputs.get(
                command[2], (0, "LoadState=loaded\nActiveState=active\nignored")
            )
        elif command[:2] == ["systemctl", "list-timers"]:
            output = self.timers
        elif command[:2] == ["systemctl", "is-active"]:
            output = "active" if self.active else "inactive"
        elif command[:2] in (["systemctl", "stop"], ["systemctl", "reset-failed"]):
            self.active = False
            output = ""
        elif command[0] == "systemd-run":
            self.active = True
            if self.start_samples:
                path = Path(command[command.index("--output") + 1])
                samples = [
                    {
                        "observed_at": NOW.isoformat(),
                        "returncode": 0,
                        "gpu_count": 2,
                        "gpu_uuids": ["GPU-A", "GPU-B"],
                    },
                    {
                        "observed_at": NOW.isoformat(),
                        "returncode": 0,
                        "gpu_count": 1,
                        "gpu_uuids": ["GPU-B"],
                    },
                ]
                path.write_text(
                    "\n".join(json.dumps(sample) for sample in samples),
                    encoding="utf-8",
                )
            output = ""
        else:
            raise AssertionError(f"unexpected fake probe command: {command[0]}")
        return subprocess.CompletedProcess(command, code, output, "")

    def main(
        self, monkeypatch: pytest.MonkeyPatch, *args: str
    ) -> tuple[int, dict[str, Any]]:
        monkeypatch.setattr(sys, "argv", ["unit-node-probe", *args])
        result = probe.main()
        return result, self.records[-1] if self.records else {}
