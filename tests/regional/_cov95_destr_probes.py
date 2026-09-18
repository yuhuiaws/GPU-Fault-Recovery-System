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
NODE = "node-a"
WORKFLOW = "workflow-owned"
INCIDENT = "incident-owned"


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

    def pre_authorization(
        self,
        not_before_seconds: dict[str, int],
        *,
        valid_for_seconds: int = 60,
        **overrides: Any,
    ) -> dict[str, Any]:
        """The conditional proof a runner delivers before the injection.

        Bounded to the 60 s device hold ``arm_arguments`` asks for: the probe
        refuses a validity that outlives the holder.
        """

        proof = {
            "kind": "conditional-barrier-pre-authorization",
            "conditional": True,
            "run_id": RUN_ID,
            "node_id": NODE,
            "boot_id": "boot-before",
            "device": "/dev/nvidia0",
            "drill_id": "drill-owned",
            "marker": "marker-owned",
            "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
            "maintenance_window_seconds": 600,
            "authorized_at": self.clock.now().isoformat(),
            "expires_at": (
                self.clock.now() + timedelta(seconds=valid_for_seconds)
            ).isoformat(),
            "not_before_seconds": dict(not_before_seconds),
            "ledger": {
                "quiesce": "QUIESCE_GPU_SERVICES",
                "verify": "VERIFY_NO_GPU_CLIENTS",
                "verify_refusal": "clients are still active",
                "forbidden": [
                    "RESET_GPU",
                    "RESET_ALL_GPUS_NVSWITCHES",
                    "RESTORE_GPU_SERVICES",
                ],
            },
        }
        proof.update(overrides)
        return proof

    def barrier_rows(
        self,
        *,
        workflow: str = WORKFLOW,
        verify_state: str = "FAILED",
        verify_error: str | None = "GPU device clients are still active: GPU-a:4242",
        extra_operations: tuple[str, ...] = (),
        offset_seconds: int = 1,
    ) -> list[dict[str, Any]]:
        """Ledger rows of one workflow parked at the client-verification barrier."""

        stamp = (NOW + timedelta(seconds=offset_seconds)).isoformat()
        rows = [
            {
                "command_id": f"{workflow}/2/QUIESCE_GPU_SERVICES/{NODE}/agent-4",
                "operation": "QUIESCE_GPU_SERVICES",
                "state": "SUCCEEDED",
                "attempt": 1,
                "started_at": stamp,
                "completed_at": stamp,
                "workflow_request_id": workflow,
                "incident_id": INCIDENT,
                "agent_generation": 4,
                "error": None,
            },
            {
                "command_id": f"{workflow}/3/VERIFY_NO_GPU_CLIENTS/{NODE}/agent-4",
                "operation": "VERIFY_NO_GPU_CLIENTS",
                "state": verify_state,
                "attempt": 1,
                "started_at": stamp,
                "completed_at": stamp,
                "workflow_request_id": workflow,
                "incident_id": INCIDENT,
                "agent_generation": 4,
                "error": verify_error,
            },
        ]
        for index, operation in enumerate(extra_operations, start=4):
            rows.append(
                {
                    "command_id": f"{workflow}/{index}/{operation}/{NODE}/agent-4",
                    "operation": operation,
                    "state": "SUCCEEDED",
                    "attempt": 1,
                    "started_at": stamp,
                    "completed_at": stamp,
                    "workflow_request_id": workflow,
                    "incident_id": INCIDENT,
                    "agent_generation": 4,
                    "error": None,
                }
            )
        return rows

    def holder_alive(self, alive: bool = True) -> None:
        """Make the device holder unit report active (or not) to systemctl."""

        self.units[self.module.holder_unit(RUN_ID) + ".service"] = {
            "LoadState": "loaded",
            "ActiveState": "active" if alive else "inactive",
            "MainPID": "222" if alive else "0",
        }
