"""GF-GENERAL-LIVE node pause commands: refusals, the failed-pause undo, log collector.

The pause module runs inside the node probe; every host effect (systemctl,
signals, the deadman timer, the run's state directory) is the probe's own
function, replaced here so nothing on this host is touched.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_node_pause as pausep
from scripts.e2e.regional.probes import general_live_node_probe as nodep

RUN = "gl030-stale-a1-deadbeef"
METRICS_UNIT = "gpu-fault-metrics-collector.service"


class Node:
    """The probe's host functions: unit states, commands, signals, deadmen."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        assert pausep.probe is nodep, "the pause module runs on the node probe"
        self.units: dict[str, dict[str, str]] = {}
        self.commands: list[list[str]] = []
        self.emitted: list[dict[str, Any]] = []
        self.resumed: list[tuple[str, str]] = []
        self.signals: list[tuple[int, int]] = []
        self.fail_on: str | None = None
        self.processes: list[dict[str, Any]] = []
        monkeypatch.setattr(nodep, "STATE_ROOT", tmp_path / "state")
        monkeypatch.setattr(nodep, "PAUSE_DROPIN_ROOT", tmp_path / "run-systemd")
        monkeypatch.setattr(nodep, "run", self.run)
        monkeypatch.setattr(nodep, "emit", self.emitted.append)
        monkeypatch.setattr(nodep, "unit_state", lambda unit: dict(self.units[unit]))
        monkeypatch.setattr(nodep, "unit_is_stopped", lambda unit: True)
        monkeypatch.setattr(
            nodep, "arm_deadman", lambda run_id, kind, seconds, *rest: f"deadman-{kind}"
        )
        monkeypatch.setattr(nodep, "resume_unit", self.resume_unit)
        monkeypatch.setattr(nodep, "exporter_processes", lambda: list(self.processes))
        monkeypatch.setattr(pausep.shutil, "which", lambda name: "/usr/bin/kill")
        monkeypatch.setattr(pausep.os, "kill", self.kill)

    def run(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        if self.fail_on is not None and self.fail_on in command:
            raise nodep.ProbeError(f"{command[0]} {command[1]} failed")
        return subprocess.CompletedProcess(command, 0, "", "")

    def resume_unit(self, run_id: str, unit: str) -> dict[str, Any]:
        self.resumed.append((run_id, unit))
        return {"unit": unit, "resumed": True}

    def kill(self, pid: int, signal_number: int) -> None:
        self.signals.append((pid, signal_number))


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Node:
    return Node(monkeypatch, tmp_path)


def arguments(**values: Any) -> argparse.Namespace:
    return argparse.Namespace(run_id=RUN, **values)


def test_a_pause_drop_in_needs_a_kill_binary(
    node: Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pausep.shutil, "which", lambda name: "/nonexistent/bin/kill")

    with pytest.raises(nodep.ProbeError, match="no kill binary for the pause drop-in"):
        pausep.install_pause_dropin(RUN, METRICS_UNIT)

    assert node.commands == []


def test_a_unit_that_is_not_active_is_never_paused(node: Node) -> None:
    node.units[METRICS_UNIT] = {"ActiveState": "inactive"}

    with pytest.raises(nodep.ProbeError, match="is not active at baseline"):
        pausep.unit_pause(arguments(unit=METRICS_UNIT, seconds=300))

    assert node.commands == [], "nothing was loaded or stopped"


def test_a_pause_that_fails_after_the_deadman_is_resumed_at_once(node: Node) -> None:
    node.units[METRICS_UNIT] = {"ActiveState": "active", "MainPID": "7"}
    node.fail_on = "--signal=SIGSTOP"

    with pytest.raises(nodep.ProbeError, match="systemctl kill failed"):
        pausep.unit_pause(arguments(unit=METRICS_UNIT, seconds=300))

    assert node.resumed == [(RUN, METRICS_UNIT)]
    assert ["systemctl", "daemon-reload"] in node.commands
    assert node.emitted == [], "a failed pause reports nothing as stopped"


def test_the_exporter_pause_needs_exactly_one_exporter_process(node: Node) -> None:
    node.processes = []

    with pytest.raises(nodep.ProbeError, match="found 0"):
        pausep.exporter_pause(arguments(seconds=300))

    assert node.signals == []


def test_the_exporter_pause_stops_the_one_process_under_a_deadman(node: Node) -> None:
    node.processes = [{"pid": 4242, "start_ticks": 1000}]

    pausep.exporter_pause(arguments(seconds=300))

    assert node.signals == [(4242, pausep.signal.SIGSTOP)]
    state = nodep.load_state(RUN)
    assert state["paused_processes"]["4242"]["start_ticks"] == 1000
    assert node.emitted[-1]["pid"] == 4242


@pytest.mark.parametrize(
    ("approved", "state", "message"),
    [
        (False, {}, "starting the node log collector requires the runner's approval"),
        (
            True,
            {"LoadState": "not-found"},
            "the node log collector unit is not installed",
        ),
        (
            True,
            {"LoadState": "loaded", "ActiveState": "active"},
            "the node log collector is already running",
        ),
    ],
)
def test_the_log_collector_start_refuses_without_a_clean_installed_unit(
    node: Node, approved: bool, state: dict[str, str], message: str
) -> None:
    node.units[nodep.LOG_COLLECTOR_UNIT] = state

    with pytest.raises(nodep.ProbeError, match=message):
        pausep.log_collector_enable(arguments(approved=approved, seconds=300))

    assert node.commands == []


def test_the_log_collector_starts_under_its_deadman(node: Node) -> None:
    unit = nodep.LOG_COLLECTOR_UNIT
    node.units[unit] = {"LoadState": "loaded", "ActiveState": "inactive"}

    pausep.log_collector_enable(arguments(approved=True, seconds=300))

    assert node.commands == [["systemctl", "enable", "--runtime", "--now", unit]]
    assert node.emitted[-1]["deadman"] == "deadman-logcollector"
    assert nodep.load_state(RUN)["log_collector"]["before"]["ActiveState"] == "inactive"


def test_unit_resume_reports_the_probes_resume(node: Node) -> None:
    pausep.unit_resume(arguments(unit=METRICS_UNIT))

    assert node.resumed == [(RUN, METRICS_UNIT)]
    assert node.emitted == [{"unit": METRICS_UNIT, "resumed": True}]
