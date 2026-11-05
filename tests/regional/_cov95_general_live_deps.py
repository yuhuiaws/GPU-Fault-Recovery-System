"""The deps probe's own process boundary, answered by the deps tests' ``FakeNode``.

``test_general_live_deps_probe.FakeNode`` plays dnf, rpm, modprobe, semodule and
systemd of one AL2023 node but is wired in by replacing the probe's ``run``,
``tool_present``, ``kernel_release`` and ``emit``. ``Boundary`` puts the same
node behind the probe's ``subprocess``, ``shutil.which`` and ``os.uname``
instead, so the probe's own command runner, tool lookup, kernel release, JSON
output and CLI run for real, and adds the failures a live node can show: an rpm
database error, a deadman timer that does not arm or that is running this very
cleanup, a dnf install or removal that fails, a removal that takes another
package with it and a module ``modprobe`` cannot load.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import general_live_deps_probe as deps
from tests.regional.test_general_live_deps_probe import KERNEL, FakeNode


class Boundary:
    def __init__(self, node: FakeNode) -> None:
        self.node = node
        #: ``env`` of every command that set one, with the command.
        self.environments: list[tuple[list[str], dict[str, str]]] = []
        #: Tools ``which`` does not find although the node would have them.
        self.hidden: set[str] = set()
        self.timer_active = True
        #: ``MainPID`` of the deadman service.
        self.deadman_pid = "0"
        self.rpm_error = False
        self.install_error = False
        self.remove_error = False
        self.modprobe_error = False
        #: Extra raw ``rpm -qa`` rows, and a package a removal takes along.
        self.rpm_rows: list[str] = []
        self.cascade: str | None = None

    @staticmethod
    def done(
        command: list[str], code: int = 0, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)

    def run(
        self,
        command: list[str],
        *,
        text: bool,
        stdout: int,
        stderr: int,
        timeout: int,
        check: bool,
        env: dict[str, str] | None,
    ) -> subprocess.CompletedProcess[str]:
        assert (text, stdout, stderr, check) == (
            True,
            subprocess.PIPE,
            subprocess.PIPE,
            False,
        ), "the probe captures text output and judges the return code itself"
        assert 0 < timeout <= deps.DNF_TIMEOUT, f"every call is bounded: {timeout}"
        command = list(command)
        if env is not None:
            self.environments.append((command, env))
        answer = self.intercept(command)
        if answer is not None:
            self.node.calls.append(command)
            return answer
        completed = self.node.run(command)
        if command[0] == "rpm" and self.rpm_rows:
            completed.stdout += "".join(row + "\n" for row in self.rpm_rows)
        if command[:2] == ["dnf", "-y"] and "remove" in command and self.cascade:
            self.node.packages.pop(self.cascade, None)
        return completed

    def intercept(self, command: list[str]) -> subprocess.CompletedProcess[str] | None:
        head = command[0]
        if head == "rpm" and self.rpm_error:
            return self.done(command, 1, stderr="error: rpmdb: BDB0113 lock held\n")
        if command[:2] == ["systemctl", "show"] and "pdeadman" in command[2]:
            if command[2].endswith(".timer"):
                state = "active" if self.timer_active else "inactive"
                text = f"LoadState=loaded\nActiveState={state}\nnot a property\n"
            else:
                text = f"LoadState=loaded\nActiveState=active\nMainPID={self.deadman_pid}\n"
            return self.done(command, 0, text)
        if command[:2] == ["dnf", "-y"]:
            if "install" in command and self.install_error:
                error = (
                    "Error: Transaction test error:\n  file /usr/sbin/tc conflicts\n"
                )
                return self.done(command, 1, stderr=error)
            if "remove" in command and self.remove_error:
                return self.done(command, 1, stderr="Error: package is protected\n")
        if head == "modprobe" and len(command) == 2 and self.modprobe_error:
            error = f"modprobe: FATAL: Module {command[1]} not found in /lib/modules\n"
            return self.done(command, 1, stderr=error)
        if command[:2] == ["modprobe", "-r"]:
            module = self.node.sys_module / command[2]
            if module.is_dir() and not (module / "refcnt").exists():
                shutil.rmtree(module)  # a module without a refcnt: nothing holds it
                return self.done(command)
        return None

    def which(self, name: str) -> str | None:
        if name in self.hidden or not self.node.tool(name):
            return None
        return f"/usr/bin/{name}"


def install_boundary(
    monkeypatch: pytest.MonkeyPatch, root: Path, node: FakeNode
) -> Boundary:
    boundary = Boundary(node)
    monkeypatch.setattr(deps, "STATE_ROOT", root / "state")
    monkeypatch.setattr(deps, "SYS_MODULE", node.sys_module)
    monkeypatch.setattr(deps, "MODULES_ROOT", root / "lib-modules")
    monkeypatch.setattr(
        deps, "subprocess", SimpleNamespace(run=boundary.run, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        deps, "shutil", SimpleNamespace(which=boundary.which, copyfile=shutil.copyfile)
    )
    uname = SimpleNamespace(release=KERNEL)
    monkeypatch.setattr(
        deps, "os", SimpleNamespace(**{**vars(os), "uname": lambda: uname})
    )
    return boundary
