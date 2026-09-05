"""The warm-spare host probe: a delayed stop, and a failsafe that outlives it.

`kubernetes-not-ready` stops the very service that carries the probe's reply.
An inline `systemctl stop kubelet.service` therefore cannot answer, and the
caller only regains the channel once the failsafe restores kubelet -- after the
NotReady window it wanted to observe has closed. These tests pin the delayed
stop, the failsafe arming that has to allow for it, and the disarm on restore.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
PROBE = lazy_script_module(
    ROOT / "scripts/e2e/regional/probes/warm_spare_node_probe.py"
)

ACTIVE = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running"}
INACTIVE = {"LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead"}


class _Systemctl:
    """Records the commands the probe runs, and answers `systemctl show`."""

    def __init__(self, states: list[dict[str, str]]) -> None:
        self.commands: list[list[str]] = []
        self.states = states

    def snapshot(self, service: str) -> dict[str, str]:
        self.commands.append(["systemctl", "show", service])
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def run(self, command: list[str], **_: Any) -> None:
        self.commands.append(list(command))

    def timers(self) -> list[list[str]]:
        return [item for item in self.commands if item[0] == "systemd-run"]


def _install(
    monkeypatch: pytest.MonkeyPatch, states: list[dict[str, str]]
) -> _Systemctl:
    systemctl = _Systemctl(states)
    monkeypatch.setattr(PROBE, "service_snapshot", systemctl.snapshot)
    monkeypatch.setattr(PROBE, "run", systemctl.run)
    return systemctl


def _stop_arguments(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "service": "kubelet.service",
        "run_id": "destr008-kubernetes-not-ready-2",
        "restore_seconds": 300,
        "stop_delay_seconds": 15,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_delayed_stop_answers_before_the_service_goes_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    systemctl = _install(monkeypatch, [ACTIVE])

    PROBE.stop_with_failsafe(_stop_arguments())

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["scheduled"] is True, emitted
    assert emitted["stop_delay_seconds"] == 15, emitted
    assert emitted["stop_unit"].endswith(".timer"), emitted
    # No inline stop: it would have to answer over the channel it just killed.
    assert ["systemctl", "stop", "kubelet.service"] not in systemctl.commands
    stop_timer, failsafe = (
        next(item for item in systemctl.timers() if "stop" in item),
        next(item for item in systemctl.timers() if "start" in item),
    )
    assert "--on-active=15s" in stop_timer, stop_timer
    # The failsafe counts from now, so it has to cover the delay as well or the
    # service would be restored before it is ever stopped.
    assert "--on-active=315s" in failsafe, failsafe


def test_an_inline_stop_still_verifies_the_service_actually_stopped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    systemctl = _install(monkeypatch, [ACTIVE, INACTIVE])

    PROBE.stop_with_failsafe(
        _stop_arguments(service="gpu-fault-node-agent.service", stop_delay_seconds=0)
    )

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["scheduled"] is False, emitted
    assert emitted["after"] == INACTIVE, emitted
    assert ["systemctl", "stop", "gpu-fault-node-agent.service"] in systemctl.commands
    assert "--on-active=300s" in systemctl.timers()[0], systemctl.timers()


def test_an_inline_stop_that_left_the_service_active_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, [ACTIVE, ACTIVE])

    with pytest.raises(PROBE.ProbeError, match="did not stop"):
        PROBE.stop_with_failsafe(_stop_arguments(stop_delay_seconds=0))


def test_restore_disarms_the_pending_stop_before_starting_the_service(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A stop timer that has not fired yet would take the service back down
    # after the restore reported success.
    systemctl = _install(monkeypatch, [INACTIVE, ACTIVE])

    PROBE.restore_service(
        argparse.Namespace(
            service="kubelet.service", run_id="destr008-kubernetes-not-ready-2"
        )
    )

    emitted = json.loads(capsys.readouterr().out)
    stop_timer = emitted["stop_unit"]
    disarm = ["systemctl", "stop", stop_timer]
    start = ["systemctl", "start", "kubelet.service"]
    assert disarm in systemctl.commands, systemctl.commands
    assert systemctl.commands.index(disarm) < systemctl.commands.index(start)
    assert emitted["restore_unit"] != stop_timer, emitted


def test_the_delay_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [ACTIVE])
    monkeypatch.setattr(
        "sys.argv",
        [
            "warm_spare_node_probe.py",
            "stop-with-failsafe",
            "--service",
            "kubelet.service",
            "--run-id",
            "run-a",
            "--restore-seconds",
            "300",
            "--stop-delay-seconds",
            "600",
        ],
    )

    assert PROBE.main() == 1
