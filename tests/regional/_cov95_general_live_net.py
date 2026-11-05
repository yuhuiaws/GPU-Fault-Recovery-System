"""The net probe's own process boundary, answered by the TCP fixture's ``FakeHost``.

``test_general_live_tcp_fixture.FakeHost`` plays ``ip``, ``tc``, ``nsenter`` and
systemd over a fake ``/sys/class/net`` but is wired in by replacing the probe's
``run``, ``tool_path`` and ``emit``. ``NetBoundary`` puts the same host behind
the probe's ``subprocess`` and ``shutil.which`` instead, so the probe's command
runner, tool lookup, JSON output and CLI run for real, and adds what a live node
can answer: ``RTNETLINK answers: File exists`` for a name that is taken, the
peer of a host-side veth pair, a deadman timer that does not arm or that runs
this very cleanup, and scripted answers for a command prefix.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import general_live_net_probe as netp
from tests.regional.test_general_live_tcp_fixture import NSENTER, SNMP, FakeHost


class NetBoundary:
    def __init__(self, host: FakeHost) -> None:
        self.host = host
        self.hidden: set[str] = set()
        self.timer_active = True
        #: ``MainPID`` of the deadman services.
        self.deadman_pid = "0"
        #: inner command prefix -> (return code, stdout).
        self.answers: dict[tuple[str, ...], tuple[int, str]] = {}
        #: Run just before ``ip link add`` reaches the host.
        self.on_add: Callable[[], None] | None = None

    @staticmethod
    def done(
        command: list[str], code: int = 0, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)

    def run(
        self,
        command: list[str],
        *,
        input: str | None,
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
        inner = (
            command[len(NSENTER) :] if command[: len(NSENTER)] == NSENTER else command
        )
        for prefix, (code, out) in self.answers.items():
            if tuple(inner[: len(prefix)]) == prefix:
                self.host.calls.append(command)
                error = f"{inner[0]}: refused\n" if code else ""
                return self.done(command, code, out, error)
        if inner[:3] == ["ip", "link", "add"]:
            if (self.host.net_class / inner[3]).exists():
                self.host.calls.append(command)
                return self.done(command, 2, stderr="RTNETLINK answers: File exists\n")
            if self.on_add is not None:
                self.on_add()
        if inner[:2] == ["systemctl", "show"]:
            self.host.calls.append(command)
            unit = inner[2]
            deadman = netp.DEADMAN_PREFIX in unit
            timer = unit.endswith(".timer")
            active = (
                "inactive" if deadman and timer and not self.timer_active else "active"
            )
            pid = self.deadman_pid if deadman and not timer else "0"
            text = f"ActiveState={active}\nMainPID={pid}\nwithout a separator\n"
            return self.done(command, 0, text)
        completed = self.host.run(command)
        if (
            inner[:3] == ["ip", "link", "add"]
            and "peer" in inner
            and "netns" not in inner
        ):
            # A host-side veth pair: the peer appears in the host netns too.
            peer = inner[inner.index("name") + 1]
            if not (self.host.net_class / peer).exists():
                self.host.add_link(peer)
        return completed

    def which(self, name: str) -> str | None:
        return None if name in self.hidden else f"/usr/sbin/{name}"


def install_boundary(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[FakeHost, NetBoundary]:
    host = FakeHost(root)
    boundary = NetBoundary(host)
    sys_module = root / "sys-module"
    (sys_module / netp.NETEM_MODULE).mkdir(parents=True)
    snmp = root / "snmp"
    snmp.write_text(SNMP.format(retrans=100), encoding="utf-8")
    monkeypatch.setattr(netp, "NET_CLASS", host.net_class)
    monkeypatch.setattr(netp, "STATE_ROOT", root / "state")
    monkeypatch.setattr(netp, "SYS_MODULE", sys_module)
    monkeypatch.setattr(netp, "MODULES_ROOT", root / "lib-modules")
    monkeypatch.setattr(netp, "PROC_NET_SNMP", snmp)
    monkeypatch.setattr(
        netp, "subprocess", SimpleNamespace(run=boundary.run, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        netp, "shutil", SimpleNamespace(which=boundary.which, copyfile=shutil.copyfile)
    )
    return host, boundary
