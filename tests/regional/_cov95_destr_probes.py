from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.regional._cov95_destr_warm import NOW, Clock

RUN_ID = "unit-destr-run"
GUARD = "/run/gpu-fault-host-probe-abcdef123456.py"


class ProbeHarness:
    """Fake process boundary and private temporary paths for node probe APIs."""

    def __init__(
        self, module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.module = module
        self.tmp_path = tmp_path
        self.clock = Clock()
        self.boot_path = tmp_path / "boot-id"
        self.boot_path.write_text("boot-before\n", encoding="utf-8")
        self.guard_path = tmp_path / "guard.py"
        self.guard_path.write_text("# fake local guard transport\n", encoding="utf-8")
        self.state = tmp_path / "state.json"
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.rows: list[dict[str, Any]] = []
        self.arm_operation = (
            "VERIFY_NO_GPU_CLIENTS"
            if module.__name__.endswith("destr014_node_probe")
            else "QUIESCE_GPU_SERVICES"
        )
        self.row_delays = 0
        self.units: dict[str, dict[str, str]] = {}
        self.failures: dict[tuple[str, ...], BaseException | int] = {}
        self.agent_active = True
        self.agent_start_stuck = False
        self.pid = 100
        self.restart_changes_pid = True
        self.clients: list[dict[str, str]] = []
        self.records: list[dict[str, Any]] = []
        monkeypatch.setattr(
            module,
            "subprocess",
            SimpleNamespace(
                run=self.run,
                PIPE=subprocess.PIPE,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
        )
        if hasattr(module, "Path"):
            monkeypatch.setattr(module, "Path", self.path)
        if hasattr(module, "BOOT_ID_PATH"):
            monkeypatch.setattr(module, "BOOT_ID_PATH", self.boot_path)
        if hasattr(module, "LEDGER"):
            monkeypatch.setattr(module, "LEDGER", tmp_path / "node-actions.db")
        if hasattr(module, "state_path"):
            monkeypatch.setattr(
                module, "state_path", lambda _run, **_kwargs: self.state
            )
        if hasattr(module, "ledger_rows"):
            monkeypatch.setattr(module, "ledger_rows", self.ledger_rows)
        if hasattr(module, "device_clients"):
            monkeypatch.setattr(
                module, "device_clients", lambda _device: deepcopy(self.clients)
            )
        if hasattr(module, "agent_env"):
            monkeypatch.setattr(module, "agent_env", lambda: {})
        monkeypatch.setattr(module, "time", self.clock)
        monkeypatch.setattr(module, "datetime", self.clock)
        monkeypatch.setattr(
            module, "emit", lambda record: self.records.append(deepcopy(record))
        )

    def path(self, value: Any, *parts: str) -> Path:
        if str(value) == "/proc/sys/kernel/random/boot_id":
            return self.boot_path
        if str(value) == GUARD:
            return self.guard_path
        return Path(value, *parts)

    def ledger_rows(self) -> list[dict[str, Any]]:
        if self.row_delays:
            self.row_delays -= 1
            return []
        return deepcopy(self.rows)

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), dict(kwargs)))
        for prefix, failure in self.failures.items():
            if tuple(command[: len(prefix)]) == prefix:
                if isinstance(failure, BaseException):
                    raise failure
                return subprocess.CompletedProcess(
                    command, failure, "", "fake command failure"
                )
        output = ""
        if command[0] == "systemctl":
            verb = command[1]
            unit = command[2] if len(command) > 2 else ""
            agent = unit == "gpu-fault-node-agent.service"
            if verb == "show":
                values = self.units.get(
                    unit,
                    {
                        "LoadState": "loaded" if agent else "not-found",
                        "UnitFileState": "enabled",
                        "ActiveState": "active"
                        if agent and self.agent_active
                        else "inactive",
                        "SubState": "running"
                        if agent and self.agent_active
                        else "dead",
                        "MainPID": str(self.pid) if agent else "0",
                        "InvocationID": f"invocation-{self.pid}",
                    },
                )
                output = (
                    "\n".join(f"{key}={value}" for key, value in values.items())
                    + "\nignored-line"
                )
            elif verb == "restart" and agent:
                self.pid += int(self.restart_changes_pid)
                self.agent_active = not self.agent_start_stuck
                self.units.pop(unit, None)
            elif verb == "start" and agent:
                self.agent_active = not self.agent_start_stuck
                self.units.pop(unit, None)
            elif verb == "start":
                self.units[unit] = {
                    "LoadState": "loaded",
                    "ActiveState": "inactive" if self.agent_start_stuck else "active",
                    "MainPID": "222",
                }
            elif verb in {"stop", "reset-failed"}:
                self.units[unit] = {
                    "LoadState": "loaded",
                    "ActiveState": "inactive",
                    "MainPID": "0",
                }
                if agent and verb == "stop":
                    self.agent_active = False
            elif verb == "list-timers":
                output = "\n".join(
                    unit
                    for unit, values in self.units.items()
                    if unit.endswith(".timer") and values["ActiveState"] == "active"
                )
            elif verb == "is-active":
                output = self.units.get(unit, {}).get("ActiveState", "inactive")
        elif command[0] == "systemd-run":
            if "--unit" in command:
                unit = command[command.index("--unit") + 1]
            else:
                unit = next(
                    arg.split("=", 1)[1] for arg in command if arg.startswith("--unit=")
                )
            suffix = (
                ".timer"
                if any(arg.startswith("--on-active=") for arg in command)
                else ".service"
            )
            self.units[unit + suffix] = {
                "LoadState": "loaded",
                "ActiveState": "active",
                "MainPID": "222",
            }
        return subprocess.CompletedProcess(command, 0, output, "")

    def arguments(self, command: str, *extra: str) -> argparse.Namespace:
        return self.module.parser().parse_args([command, *extra])

    def main(
        self, monkeypatch: pytest.MonkeyPatch, *args: str
    ) -> tuple[int, dict[str, Any]]:
        monkeypatch.setattr(sys, "argv", ["unit-fake-probe", *args])
        code = self.module.main()
        return code, self.records[-1] if self.records else {}

    def arm_arguments(self, *, ledger: bool = True) -> argparse.Namespace:
        args = [
            "--run-id",
            RUN_ID,
            "--drill-id",
            "drill-owned",
            "--device",
            "/dev/nvidia0",
            "--max-hold-seconds",
            "60",
            "--probe-script",
            str(self.module.__file__),
        ]
        if ledger:
            args += ["--after-ledger-op", self.arm_operation]
        return self.arguments("arm-holder", *args)

    def append_arm_row(self) -> None:
        self.rows.append(
            {
                "command_id": "arm-command-owned",
                "operation": self.arm_operation,
                "state": "SUCCEEDED",
                "attempt": 1,
                "started_at": NOW.isoformat(),
                "completed_at": (NOW + timedelta(seconds=1)).isoformat(),
            }
        )

    def proof(self) -> dict[str, Any]:
        return {
            "run_id": RUN_ID,
            "drill_id": "drill-owned",
            "device": "/dev/nvidia0",
            "boot_id": "boot-before",
            "waiting": True,
            "window_expires_at": (NOW + timedelta(minutes=5)).isoformat(),
            "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
        }
