"""A fake host for the GF-GENERAL-LIVE kernel-log source replay probe.

``FakeSystemd`` answers the ``systemctl``/``systemd-run``/``journalctl`` calls the
probe makes through its own ``run`` wrapper from a small unit table, the way the
host's systemd would: ``restart`` brings a unit up under a new invocation,
``stop`` takes it down, ``enable --runtime`` marks it ``enabled-runtime`` and
``systemd-run`` creates the deadman's transient timer. ``install_host`` points
every host root the probe writes (``/run`` state, the ``/run/systemd/system``
drop-ins, ``/proc``) at a ``tmp_path`` tree and swaps the probe's
``subprocess``/``shutil`` for the fake, so nothing reaches the real host.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_kernel_source_probe as probe

RUN_ID = "gl016-cov-window-a1-deadbeef"
LOG = probe.LOG_UNIT
KERNEL = probe.KERNEL_UNIT
#: Where ``which`` must look for the real journalctl the wrapper forwards to.
SYSTEM_PATH = "/usr/bin:/usr/local/bin:/bin:/sbin"
REAL_LINE = {"MESSAGE": "real kernel line", "__CURSOR": "s=real-1"}


def unit_properties(
    *,
    load: str = "loaded",
    active: str = "inactive",
    file_state: str = "disabled",
    pid: str = "0",
    invocation: str = "",
) -> dict[str, str]:
    return {
        "LoadState": load,
        "ActiveState": active,
        "SubState": "running" if active == "active" else "dead",
        "UnitFileState": file_state,
        "MainPID": pid,
        "NRestarts": "0",
        "InvocationID": invocation,
    }


class FakeSystemd:
    """The host's systemd as far as the probe can see it."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.units: dict[str, dict[str, str]] = {
            LOG: unit_properties(),
            KERNEL: unit_properties(
                active="active", file_state="enabled", pid="4242", invocation="a" * 32
            ),
        }
        #: ``systemctl <verb>`` / ``systemd-run`` return codes to fail with.
        self.failing: dict[str, int] = {}
        #: The ActiveState a ``systemctl restart`` leaves its unit in.
        self.restart_leaves = "active"
        self.timer_starts = True
        self.journal: list[str] = []
        self.journal_returncode = 0
        #: Side effects of ``systemctl stop <unit>``, by unit.
        self.on_stop: dict[str, Callable[[], None]] = {}
        self.invocations = 0

    def unit(self, name: str) -> dict[str, str]:
        return self.units.setdefault(
            name, unit_properties(load="not-found", file_state="")
        )

    def mutations(self) -> list[list[str]]:
        """Every call except the read-only ``systemctl show``."""

        return [call for call in self.calls if call[:2] != ["systemctl", "show"]]

    def run(
        self,
        command: list[str],
        *,
        text: bool,
        stdout: int,
        stderr: int,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert (text, stdout, stderr, check) == (
            True,
            subprocess.PIPE,
            subprocess.PIPE,
            False,
        ), "the probe captures text output and judges the return code itself"
        assert 0 < timeout <= 120, f"every host call is bounded: {timeout}"
        command = list(command)
        self.calls.append(command)
        if command[0] == "journalctl":
            return self.answer(
                command, self.journal_returncode, "\n".join(self.journal)
            )
        if command[0] == "systemd-run":
            name = next(
                item.split("=", 1)[1] for item in command if item.startswith("--unit=")
            )
            code = self.failing.get("systemd-run", 0)
            if not code and self.timer_starts:
                self.unit(f"{name}.timer").update(
                    LoadState="loaded", ActiveState="active"
                )
                self.unit(f"{name}.service").update(LoadState="loaded")
            return self.answer(command, code)
        assert command[0] == "systemctl", f"unexpected host command: {command}"
        verb = command[1]
        code = self.failing.get(verb, 0)
        if verb == "show":
            properties = self.unit(command[2])
            lines = [f"{key}={value}" for key, value in properties.items()]
            # A line without a separator says nothing and is skipped.
            return self.answer(command, code, "\n".join([*lines, "garbage"]) + "\n")
        if code:
            return self.answer(command, code)
        if verb == "restart":
            self.invocations += 1
            self.unit(command[2]).update(
                ActiveState=self.restart_leaves,
                MainPID=str(5000 + self.invocations),
                InvocationID=f"{self.invocations:032x}",
            )
        elif verb == "stop":
            for name in command[2:]:
                self.unit(name).update(ActiveState="inactive", MainPID="0")
                self.on_stop.get(name, lambda: None)()
        elif verb == "enable":
            assert "--runtime" in command, "the probe only ever enables for this boot"
            self.unit(command[-1])["UnitFileState"] = "enabled-runtime"
        elif verb == "disable":
            self.unit(command[-1])["UnitFileState"] = "disabled"
        else:
            assert verb in {"daemon-reload", "reset-failed"}, command
        return self.answer(command, 0)

    @staticmethod
    def answer(
        command: list[str], code: int, stdout: str = ""
    ) -> subprocess.CompletedProcess[str]:
        stderr = f"Failed to run {command[0]}: unit busy\n" if code else ""
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)


def install_host(monkeypatch: pytest.MonkeyPatch, root: Path) -> SimpleNamespace:
    """Point the probe at ``root`` and ``FakeSystemd``; return the handles."""

    systemd = FakeSystemd()
    journalctl = root / "usr" / "bin" / "journalctl"
    journalctl.parent.mkdir(parents=True)
    journalctl.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(REAL_LINE)}'\n", encoding="utf-8"
    )
    journalctl.chmod(0o755)
    found: dict[str, str] = {"journalctl": str(journalctl)}
    lookups: list[tuple[str, str | None]] = []

    def which(name: str, path: str | None = None) -> str | None:
        lookups.append((name, path))
        return found.get(name)

    monkeypatch.setattr(
        probe, "subprocess", SimpleNamespace(run=systemd.run, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        probe,
        "shutil",
        SimpleNamespace(which=which, copyfile=shutil.copyfile, rmtree=shutil.rmtree),
    )
    monkeypatch.setattr(probe, "STATE_ROOT", root / "run" / "gpu-fault-kernel-replay")
    monkeypatch.setattr(probe, "DROPIN_ROOT", root / "run" / "systemd" / "system")
    monkeypatch.setattr(probe, "PROC_ROOT", root / "proc")
    return SimpleNamespace(
        systemd=systemd, found=found, lookups=lookups, journalctl=journalctl, root=root
    )


def records_flags(records: list[dict[str, Any]]) -> tuple[str, str]:
    """The ``--sha256``/``--b64`` pair the runner sends for ``records``."""

    content = json.dumps(records).encode("utf-8")
    return hashlib.sha256(content).hexdigest(), base64.b64encode(content).decode()
