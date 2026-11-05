"""A fake GPU node for the GF-ENV-LIVE host controller (``env_replay_proxy``).

The controller runs on the node: it drives ``systemctl`` / ``systemd-run``,
reads ``/proc/<pid>/{environ,comm,exe,stat}`` and talks HTTP to the replay
proxy and the real exporter on loopback. ``EnvNode`` answers all of it:

* a recorded systemd whose transient units appear on ``systemd-run`` and
  vanish on ``stop``; ``restart`` of the metrics unit starts a new main
  process whose environment is what systemd would hand it -- the endpoint of
  ``collector.env``, overridden by the run's environment file while the
  repoint drop-in lists it (``EnvironmentFile=-``); ``kill`` with
  ``SIGSTOP``/``SIGCONT`` moves the process in and out of state ``T``;
* ``/proc`` under ``tmp_path`` (``PROC_ROOT`` and the probe's ``Path``);
* an HTTP opener (the probe's own ``build_opener``) for the proxy's
  ``/status``, ``/arm`` and the exporter's ``/metrics``, so the probe's
  loopback client runs for real;
* a clock whose ``sleep`` advances ``monotonic``.

Nothing touches the host.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import env_replay_proxy as probe

RUN = "env009-gap-a1-cafe"
METRICS = probe.METRICS_UNIT
ENDPOINT = probe.ENDPOINT_ENV
REAL = "http://127.0.0.1:9400/metrics"
PROXY = "http://127.0.0.1:9411/metrics"
MISSING = {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"}


def done(command: list[str], stdout: str = "", code: int = 0) -> Any:
    return subprocess.CompletedProcess(command, code, stdout, "fake stderr\n")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(float(seconds), 0.01)


class Response(io.BytesIO):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(body.encode())
        self.status = status


class EnvNode:
    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path
        self.proc = tmp_path / "proc"
        self.units: dict[str, dict[str, str]] = {}
        self.commands: list[list[str]] = []
        #: Command prefix -> return code.
        self.failures: dict[tuple[str, ...], int] = {}
        #: ``systemd-run --on-active`` timers come up active unless set.
        self.timers_fail = False
        #: Transient services come up in this state.
        self.service_state = "active"
        self.next_pid = 700
        #: The proxy tool's ``/status``; ``None`` while nothing answers.
        self.status: dict[str, Any] | None = None
        #: URL -> (status, body), an exception, or a callable answering one of
        #: them per request; ahead of the tool and the exporter.
        self.http: dict[str, Any] = {}
        self.requests: list[tuple[str, str]] = []
        self.exporter = ""
        #: ``nvidia-smi --query-gpu`` CSV output.
        self.inventory = ""
        #: One entry per coming restart of the metrics unit, each overriding
        #: what that restart yields: ``state`` (ActiveState), ``environment``
        #: (instead of what systemd would hand it) and ``execed``.
        self.restarts: list[dict[str, Any]] = []
        self.emitted: list[dict[str, Any]] = []
        self.clock = FakeClock()

    # -- /proc -------------------------------------------------------------
    def path(self, value: Any, *parts: Any) -> Path:
        path = Path(value, *parts)
        if path.is_relative_to("/proc"):
            return self.proc / path.relative_to("/proc")
        if path.is_relative_to("/run"):
            return self.base / "run-root" / path.relative_to("/run")
        return path

    def process(
        self, pid: str, environment: dict[str, str], *, execed: bool = True
    ) -> None:
        root = self.proc / pid
        root.mkdir(parents=True, exist_ok=True)
        (root / "environ").write_bytes(
            b"".join(f"{k}={v}".encode() + b"\0" for k, v in environment.items())
        )
        (root / "comm").write_text("python3\n" if execed else "(python3)\n")
        exe = root / "exe"
        exe.unlink(missing_ok=True)
        exe.symlink_to("/opt/gpu-fault/current/venv/bin/python3")
        self.signal(pid, "S")

    def signal(self, pid: str, state: str) -> None:
        (self.proc / pid / "stat").write_text(f"{pid} (python3) {state} 1 2 3\n")

    def collector(self, pid: str, environment: dict[str, str], **kw: Any) -> None:
        self.unit(METRICS, MainPID=pid)
        self.process(pid, environment, **kw)

    # -- systemd -----------------------------------------------------------
    def unit(self, name: str, **state: str) -> None:
        self.units[name] = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "MainPID": "0",
            **state,
        }

    def run(self, command: list[str], *, check: bool = True, timeout: int = 120) -> Any:
        self.commands.append(list(command))
        for prefix, code in self.failures.items():
            if tuple(command[: len(prefix)]) == prefix:
                if check:
                    raise probe.ProbeError(f"command failed ({code}): {command[0]}")
                return done(command, code=code)
        if command[0] == "systemctl":
            return self.systemctl(command[1], command[2:], command)
        if command[0] == "systemd-run":
            name = next(a.split("=", 1)[1] for a in command if a.startswith("--unit="))
            if any(a.startswith("--on-active=") for a in command):
                if not self.timers_fail:
                    self.unit(f"{name}.timer", SubState="waiting")
            else:
                self.next_pid += 1
                self.unit(
                    f"{name}.service",
                    MainPID=str(self.next_pid),
                    ActiveState=self.service_state,
                )
            return done(command)
        if command[0] == "nvidia-smi":
            return done(command, self.inventory)
        return done(command)

    def systemctl(self, verb: str, arguments: list[str], command: list[str]) -> Any:
        if verb == "show":
            state = self.units.get(arguments[0], MISSING)
            return done(command, "".join(f"{k}={v}\n" for k, v in state.items()))
        if verb == "list-units":
            prefix = arguments[-1].rstrip("*")
            rows = [
                f"{name} {s['LoadState']} {s['ActiveState']} {s['SubState']} fake\n"
                for name, s in sorted(self.units.items())
                if name.startswith(prefix)
            ]
            return done(command, "".join(rows))
        if verb == "stop":
            for name in arguments:
                if name.startswith(probe.UNIT_PREFIX):
                    self.units.pop(name, None)
            return done(command)
        if verb == "restart" and arguments[0] == METRICS:
            self.next_pid += 1
            pid = str(self.next_pid)
            plan = self.restarts.pop(0) if self.restarts else {}
            environment = plan.get("environment", self.environment())
            self.collector(pid, environment, execed=plan.get("execed", True))
            self.units[METRICS]["ActiveState"] = plan.get("state", "active")
            return done(command)
        if verb == "kill" and arguments[-1] == METRICS:
            state = "T" if "--signal=SIGSTOP" in arguments else "S"
            self.signal(self.units[METRICS]["MainPID"], state)
        return done(command)

    def environment(self) -> dict[str, str]:
        """What systemd hands a (re)started collector: ``collector.env``, then
        the run's environment file while the drop-in lists it."""

        values = {ENDPOINT: probe.baseline_endpoint()}
        for dropin in probe.dropin_dir().glob("*.conf"):
            for line in dropin.read_text().splitlines():
                if line.startswith("EnvironmentFile=-"):
                    override = Path(line.removeprefix("EnvironmentFile=-"))
                    if override.is_file():
                        key, _, value = override.read_text().strip().partition("=")
                        values[key] = value
        return values

    # -- loopback HTTP -----------------------------------------------------
    def build_opener(self, *handlers: Any) -> EnvNode:
        assert len(handlers) == 1 and handlers[0].proxies == {}, "never a proxy"
        return self

    def open(self, request: Any, timeout: int) -> Response:
        url, method = request.full_url, request.get_method()
        self.requests.append((method, url))
        answer = self.http.get(url)
        if callable(answer):
            answer = answer()
        if isinstance(answer, Exception):
            raise answer
        if answer is not None:
            return Response(*answer)
        if url == REAL:
            return Response(200, self.exporter)
        if self.status is None:
            raise ConnectionRefusedError(111, "Connection refused")
        if url.endswith("/status"):
            return Response(200, json.dumps(self.status))
        if url.endswith("/arm") and method == "POST":
            self.status["armed"] = True
            return Response(200, "armed\n")
        return Response(404, "not found\n")

    # -- wiring ------------------------------------------------------------
    def install(self, monkeypatch: pytest.MonkeyPatch) -> EnvNode:
        env_file = self.base / "collector.env"
        env_file.write_text(f"{ENDPOINT}={REAL}\nGPU_FAULT_METRICS_MODE=dcgm\n")
        venv = self.base / "venv-python"
        venv.write_text("")
        monkeypatch.setattr(probe, "COLLECTOR_ENV", env_file)
        monkeypatch.setattr(probe, "VENV_PYTHON", venv)
        monkeypatch.setattr(probe, "ACCEPTANCE_ROOT", self.base / "acceptance")
        monkeypatch.setattr(probe, "PROC_ROOT", self.proc)
        monkeypatch.setattr(probe, "Path", self.path)
        monkeypatch.setattr(probe, "run", self.run)
        monkeypatch.setattr(probe, "build_opener", self.build_opener)
        monkeypatch.setattr(probe, "emit", self.emitted.append)
        monkeypatch.setattr(probe, "time", self.clock)
        self.collector("100", {ENDPOINT: REAL, "GPU_FAULT_METRICS_MODE": "dcgm"})
        return self


def env_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> EnvNode:
    return EnvNode(tmp_path).install(monkeypatch)


def proxy_up(node: EnvNode, *, reads: int = 0, armed: bool = False) -> None:
    """This run's proxy unit runs and answers ``/status``."""

    node.unit(f"{probe.proxy_unit(RUN)}.service", MainPID="300")
    node.status = {"armed": armed, "reads": reads}


def installed(*names: str) -> Path:
    root = probe.run_dir(RUN)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text(f"# reviewed {name}\n", encoding="utf-8")
    return root
