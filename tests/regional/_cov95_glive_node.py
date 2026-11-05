"""A fake GPU node for the GF-GENERAL-LIVE host probe (``general_live_node_probe``).

The probe runs on the node: it reads ``/proc``, ``/sys``, ``/etc`` and ``/run``
and drives ``systemctl`` / ``systemd-run``. ``NodeHost`` answers all of that
from a tree under ``tmp_path`` and a recorded fake ``run``: transient units
appear when ``systemd-run`` starts them and vanish when they are stopped,
``show`` answers per unit, ``list-units`` filters by the probe's glob,
``modinfo`` knows a configured set of modules and a configured command prefix
can fail. ``install`` points the probe's roots, ``Path``, ``run``, ``emit`` and
clock at the fakes; nothing touches the host.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_node_probe as nodep

RUN = "gl-cov95-node-a1-deadbeef"
#: The checkout the probe file itself lives in (``probe_copy`` copies it).
CHECKOUT = Path(nodep.__file__).resolve().parents[4]
#: Host trees the fake node owns; every other path (the checkout, ``tmp_path``
#: itself) stays what it is.
REROOTED = ("/proc", "/sys", "/etc", "/run", "/var", "/opt")
NOT_FOUND = {
    "LoadState": "not-found",
    "ActiveState": "inactive",
    "SubState": "dead",
    "MainPID": "0",
}
PAUSE_HOOK = (
    "ExecStartPost={ path=/usr/bin/kill ; argv[]=/usr/bin/kill -STOP $MAINPID ; }"
)


def done(command: list[str], stdout: str = "", code: int = 0) -> Any:
    return subprocess.CompletedProcess(command, code, stdout, "fake stderr\n")


class FakeClock:
    """``time`` for the probe: ``sleep`` advances both clocks."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class NodeHost:
    """The node the probe sees: a file tree plus recorded systemd."""

    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path
        self.root = tmp_path / "host"
        self.root.mkdir()
        self.units: dict[str, dict[str, str]] = {}
        self.commands: list[list[str]] = []
        self.options: list[dict[str, Any]] = []
        #: Command prefix -> return code the fake answers with.
        self.failures: dict[tuple[str, ...], int] = {}
        self.modules: set[str] = set()
        #: Units whose loaded ``ExecStartPost`` still stops the main process.
        self.hooks: set[str] = set()
        #: ``systemd-run --on-active`` timers come up active unless this is set.
        self.timers_fail = False
        self.next_pid = 4000
        self.emitted: list[dict[str, Any]] = []
        self.clock = FakeClock()

    # -- files -------------------------------------------------------------
    def path(self, value: Any = ".", *parts: Any) -> Path:
        path = Path(value, *parts)
        text = path.as_posix()
        if path.is_relative_to(self.base) or path.is_relative_to(CHECKOUT):
            return path
        if any(text == item or text.startswith(item + "/") for item in REROOTED):
            return self.root / path.relative_to("/")
        return path

    def write(self, absolute: str, content: str | bytes) -> Path:
        target = self.path(absolute)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        return target

    def process(self, pid: int, argv: list[str], state: str = "S") -> None:
        """``/proc/<pid>`` with a cmdline and a stat whose start ticks are pid*10."""

        self.write(f"/proc/{pid}/cmdline", b"\0".join(a.encode() for a in argv) + b"\0")
        rest = [state, *[str(n) for n in range(1, 19)], str(pid * 10), "0", "0"]
        self.write(
            f"/proc/{pid}/stat", f"{pid} ({Path(argv[0]).name}) {' '.join(rest)}\n"
        )

    # -- systemd -----------------------------------------------------------
    def unit(self, name: str, **state: str) -> None:
        self.units[name] = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "MainPID": "0",
            **state,
        }

    def run(
        self,
        command: list[str],
        *,
        check: bool = True,
        timeout: int = 180,
        input_text: str | None = None,
    ) -> Any:
        self.commands.append(list(command))
        self.options.append({"check": check, "timeout": timeout, "input": input_text})
        for prefix, code in self.failures.items():
            if tuple(command[: len(prefix)]) == prefix:
                return done(command, code=code)
        tool = command[0]
        if tool == "systemctl":
            return self.systemctl(command)
        if tool == "systemd-run":
            name = next(a.split("=", 1)[1] for a in command if a.startswith("--unit="))
            if any(a.startswith("--on-active=") for a in command):
                if not self.timers_fail:
                    self.unit(f"{name}.timer", SubState="waiting")
            else:
                self.next_pid += 1
                self.unit(f"{name}.service", MainPID=str(self.next_pid))
            return done(command)
        if tool == "modinfo":
            return done(command, code=0 if command[1] in self.modules else 1)
        return done(command)

    def systemctl(self, command: list[str]) -> Any:
        verb, arguments = command[1], command[2:]
        if verb == "show" and arguments[0] == "--property=ExecStartPost":
            return done(command, PAUSE_HOOK if arguments[1] in self.hooks else "")
        if verb == "show":
            state = self.units.get(arguments[0], NOT_FOUND)
            return done(command, "".join(f"{k}={v}\n" for k, v in state.items()))
        if verb == "list-units":
            pattern = arguments[-1]
            rows = [
                f"{name} loaded {state['ActiveState']} {state['SubState']} fake\n"
                for name, state in sorted(self.units.items())
                if fnmatch.fnmatch(name, pattern)
            ]
            return done(command, "".join(rows))
        if verb == "stop":
            for name in arguments:
                if name.startswith(nodep.UNIT_PREFIX):
                    self.units.pop(name, None)
                elif name in self.units:
                    self.units[name].update(
                        ActiveState="inactive", SubState="dead", MainPID="0"
                    )
        return done(command)

    def calls(self, *prefix: str) -> list[list[str]]:
        """Recorded commands starting with ``prefix``."""

        return [c for c in self.commands if c[: len(prefix)] == list(prefix)]

    # -- wiring ------------------------------------------------------------
    def install(self, monkeypatch: pytest.MonkeyPatch) -> NodeHost:
        monkeypatch.setattr(nodep, "Path", self.path)
        monkeypatch.setattr(nodep, "STATE_ROOT", self.path(nodep.STATE_ROOT))
        monkeypatch.setattr(nodep, "COLLECTOR_ENV", self.path(nodep.COLLECTOR_ENV))
        monkeypatch.setattr(
            nodep, "PAUSE_DROPIN_ROOT", self.path(nodep.PAUSE_DROPIN_ROOT)
        )
        monkeypatch.setattr(
            nodep, "BACKING_ROOTS", tuple(self.path(r) for r in nodep.BACKING_ROOTS)
        )
        monkeypatch.setattr(nodep, "run", self.run)
        monkeypatch.setattr(nodep, "emit", self.emitted.append)
        monkeypatch.setattr(nodep, "time", self.clock)
        return self


def node_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NodeHost:
    return NodeHost(tmp_path).install(monkeypatch)


def command(*argv: str) -> None:
    """Run one probe subcommand through its parser, as the fixture does."""

    arguments = nodep.parser().parse_args(list(argv))
    arguments.handler(arguments)
