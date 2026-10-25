"""Recorded-systemd fakes for the DCGM field-replay host controller.

``probes/general_live_dcgm_probe.py`` runs on the node; these tests run it
in-process over a fake ``systemctl`` / ``systemd-run`` (unit states answer
``show`` and ``list-units``, transient units appear and vanish, every command
is recorded), a fake ``/proc`` (the allow-listed environment per PID and
whether that PID has exec'd yet), a fake loopback to the replay tool and a
fake clock, with the state root, drop-in directory and collector env file
under ``tmp_path``. Nothing touches the host.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_dcgm_probe as probe

METRICS = probe.METRICS_UNIT
ENDPOINT = probe.ENDPOINT_ENV
REAL = "http://127.0.0.1:9400/metrics"
REPLAY = "http://127.0.0.1:9412/metrics"
#: The run the live 2026-10-25 residue came from (digest ``d96819b19bd0``).
RUN = "gl019-corrected-ecc-a2-4116e7cc"
DIGEST = probe.digest(RUN)
FIRES = "Sun 2026-10-25 10:23:10 UTC"
FOREIGN_DROPIN = "gpu-fault-acceptance-0123456789abcdef.conf"
MISSING = {
    "LoadState": "not-found",
    "ActiveState": "inactive",
    "SubState": "dead",
    "MainPID": "0",
}


def _done(command: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout, "")


class FakeClock:
    """``time`` for the probe: ``sleep`` advances ``monotonic``."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        step = max(float(seconds), 0.01)
        self.now += step
        self.slept += step


class FakeSystemd:
    """``probe.run`` and ``probe.loopback`` stand-ins with recorded calls."""

    def __init__(self) -> None:
        self.units: dict[str, dict[str, str]] = {}
        self.commands: list[list[str]] = []
        #: PID -> the allow-listed environment ``/proc/<pid>/environ`` shows.
        self.environ: dict[str, dict[str, str]] = {}
        #: PIDs that have exec'd (``has_execed``).
        self.execed: set[str] = set()
        self.next_pid = 500
        #: Called with the new MainPID after every restart of the metrics unit.
        self.on_restart: Any = None
        #: Whether a replay tool answers on the loopback port.
        self.tool_answers = False
        self.posts: list[str] = []
        self.emitted: list[dict[str, Any]] = []

    def unit(self, name: str, **state: str) -> None:
        self.units[name] = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "MainPID": "0",
            **state,
        }

    def collector(
        self, pid: str, environment: dict[str, str], *, execed: bool = True
    ) -> None:
        """The metrics collector runs as ``pid`` with ``environment``."""

        self.unit(METRICS, MainPID=pid)
        self.environ[pid] = dict(environment)
        if execed:
            self.execed.add(pid)
        else:
            self.execed.discard(pid)

    def run(
        self, command: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        tool, args = command[0], command[1:]
        if tool == "systemctl":
            return self._systemctl(command, args)
        if tool == "systemd-run":
            unit = next(a.split("=", 1)[1] for a in args if a.startswith("--unit="))
            if any(a.startswith("--on-active=") for a in args):
                self.unit(
                    f"{unit}.timer", SubState="waiting", NextElapseUSecRealtime=FIRES
                )
            else:
                self.next_pid += 1
                self.unit(f"{unit}.service", MainPID=str(self.next_pid))
                self.tool_answers = True
            return _done(command)
        return _done(command)

    def _systemctl(
        self, command: list[str], args: list[str]
    ) -> subprocess.CompletedProcess[str]:
        verb = args[0]
        if verb == "show":
            state = self.units.get(args[1], MISSING)
            return _done(command, "".join(f"{k}={v}\n" for k, v in state.items()))
        if verb == "list-units":
            names = sorted(n for n in self.units if n.startswith(probe.UNIT_PREFIX))
            return _done(
                command,
                "".join(
                    f"{n} loaded {self.units[n]['ActiveState']} "
                    f"{self.units[n]['SubState']} transient\n"
                    for n in names
                ),
            )
        if verb == "stop":
            for name in args[1:]:
                if name.startswith(probe.UNIT_PREFIX):
                    # A stopped transient unit is unloaded.
                    if name.endswith("-proxy.service") and name in self.units:
                        self.tool_answers = False
                    self.units.pop(name, None)
                elif name in self.units:
                    self.units[name].update(
                        ActiveState="inactive", SubState="dead", MainPID="0"
                    )
            return _done(command)
        if verb == "restart":
            self.next_pid += 1
            pid = str(self.next_pid)
            self.unit(args[1], MainPID=pid)
            if args[1] == METRICS and self.on_restart is not None:
                self.on_restart(pid)
            return _done(command)
        return _done(command)

    def loopback(
        self, url: str, *, method: str = "GET", timeout: int = 10
    ) -> tuple[int, bytes]:
        path = url.rsplit("/", 1)[-1]
        if not self.tool_answers:
            raise probe.ProbeError(f"loopback {method} failed: connection refused")
        if method == "POST":
            self.posts.append(path)
            return 200, b"ok\n"
        if path == "status":
            return 200, json.dumps({"armed": False, "reads": 0}).encode()
        return 404, b"not found\n"

    def index(self, *prefix: str) -> int:
        """Position of the first recorded command starting with ``prefix``."""

        for position, command in enumerate(self.commands):
            if command[: len(prefix)] == list(prefix):
                return position
        raise AssertionError(f"no recorded command starts with {prefix}")


def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSystemd:
    """Point the probe at ``tmp_path`` and the fakes; a baseline collector runs."""

    fake = FakeSystemd()
    env_file = tmp_path / "collector.env"
    env_file.write_text(f"{ENDPOINT}={REAL}\nGPU_FAULT_METRICS_MODE=dcgm\n")
    venv = tmp_path / "venv-python"
    venv.write_text("")
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    monkeypatch.setattr(probe, "DROPIN_DIR", tmp_path / "dropins")
    monkeypatch.setattr(probe, "COLLECTOR_ENV", env_file)
    monkeypatch.setattr(probe, "VENV_PYTHON", venv)
    monkeypatch.setattr(probe, "run", fake.run)
    monkeypatch.setattr(probe, "loopback", fake.loopback)
    monkeypatch.setattr(
        probe, "process_environment", lambda pid: dict(fake.environ.get(pid, {}))
    )
    monkeypatch.setattr(probe, "has_execed", lambda pid: pid in fake.execed)
    monkeypatch.setattr(probe, "time", FakeClock())
    monkeypatch.setattr(probe, "emit", fake.emitted.append)
    fake.collector("100", {ENDPOINT: REAL})
    return fake


def leftover_run(fake: FakeSystemd, *, deadman: bool = True) -> None:
    """The node as the live run left it: repointed, proxy up, deadman armed."""

    root = probe.run_dir(RUN)
    root.mkdir(parents=True)
    (root / probe.RUN_MARKER).write_text(json.dumps({"run_id": RUN, "digest": DIGEST}))
    (root / "plan.json").write_text("{}")
    (root / "replay.jsonl").write_text("")
    (root / "repoint.json").write_text(
        json.dumps({"run_id": RUN, "original_endpoint": REAL, "port": 9412})
    )
    probe.override_env_path(RUN).write_text(f"{ENDPOINT}={REPLAY}\n")
    probe.DROPIN_DIR.mkdir(parents=True, exist_ok=True)
    probe.dropin_path(RUN).write_text(
        f"[Service]\nEnvironmentFile=-{probe.override_env_path(RUN)}\n"
    )
    fake.unit(f"{probe.proxy_unit(RUN)}.service", MainPID="300")
    fake.tool_answers = True
    if deadman:
        fake.unit(
            f"{probe.deadman_unit(RUN)}.timer",
            SubState="waiting",
            NextElapseUSecRealtime=FIRES,
        )
    fake.collector("200", {ENDPOINT: REPLAY})


def restored_on_restart(fake: FakeSystemd) -> None:
    """After a restart the collector execs with the real endpoint again."""

    fake.on_restart = lambda pid: fake.collector(pid, {ENDPOINT: REAL})


def stuck_on_restart(fake: FakeSystemd) -> None:
    """After a restart the collector still reads the replay (restore cannot
    prove the endpoint)."""

    fake.on_restart = lambda pid: fake.collector(pid, {ENDPOINT: REPLAY})
