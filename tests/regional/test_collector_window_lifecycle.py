from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.collector_window_fixture import open_window_or_rollback
from scripts.e2e.regional.probes import collector_window_probe as probe


class Abort(BaseException):
    pass


@pytest.fixture
def window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(probe, "ACCEPTANCE_ROOT", tmp_path / "windows")
    monkeypatch.setattr(probe.shutil, "which", lambda *a, **k: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        probe,
        "dropin_path",
        lambda unit, run_id: (
            tmp_path / "systemd" / f"{unit}.d" / probe.dropin_name(run_id)
        ),
    )
    arguments = argparse.Namespace(
        run_id="collector-review",
        unit="gpu-fault-host-collector.service",
        restore_seconds=600,
        env=[],
        unset=[],
        shadow_nvidia_smi="hang:30",
    )
    state: dict[str, Any] = {
        "timer_active": False,
        "service_active": True,
        "fail_restart": None,
        "fail_arm": False,
        "inactive_timer": False,
        "bad_readback": False,
        "calls": [],
        "emitted": [],
    }

    def unit_state(unit: str) -> dict[str, str]:
        active = state["timer_active"] if unit.endswith(".timer") else True
        if unit == arguments.unit:
            active = state["service_active"]
        return {
            "ActiveState": "active" if active else "inactive",
            "LoadState": "loaded",
            "MainPID": "1234" if active else "0",
            "NRestarts": "0",
        }

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        state["calls"].append(list(command))
        if command[0] == "systemd-run":
            assert not probe.dropin_path(arguments.unit, arguments.run_id).exists(), (
                f"{arguments.run_id}: drop-in was installed before the watchdog armed"
            )
            record = json.loads(
                probe.window_paths(arguments.run_id)["state"].read_text()
            )
            assert record["restore_needed"] is False
            if state["fail_arm"]:
                raise probe.ProbeError("timer could not arm")
            state["timer_active"] = not state["inactive_timer"]
        elif command[:2] == ["systemctl", "restart"]:
            failure = state["fail_restart"]
            state["fail_restart"] = None
            if failure is not None:
                raise failure
            state["service_active"] = not state["bad_readback"]
        elif command[:2] == ["systemctl", "stop"] and command[2].endswith(".timer"):
            state["timer_active"] = False
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(probe, "unit_state", unit_state)
    monkeypatch.setattr(probe, "run", run)
    monkeypatch.setattr(probe, "emit", state["emitted"].append)
    return {**state, "state": state, "arguments": arguments}


def test_window_arms_before_mutation_and_closes_idempotently(
    window: dict[str, Any],
) -> None:
    arguments, state = window["arguments"], window["state"]
    vendor = probe.dropin_path(arguments.unit, arguments.run_id).parent / "vendor.conf"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("[Service]\nEnvironment=KEEP=1\n")
    probe.open_window(arguments)
    assert state["timer_active"]
    assert probe.window_paths(arguments.run_id)["state"].is_file(), (
        f"{arguments.run_id}: an open window has no recovery state"
    )
    assert probe.dropin_path(arguments.unit, arguments.run_id).is_file(), (
        f"{arguments.unit}: opening the window did not install its drop-in"
    )
    probe.close_window(arguments)
    assert not state["timer_active"]
    assert not probe.window_paths(arguments.run_id)["root"].exists(), (
        f"{arguments.run_id}: successful close left the private window directory"
    )
    assert not probe.dropin_path(arguments.unit, arguments.run_id).exists(), (
        f"{arguments.unit}: successful close left the test drop-in"
    )
    assert vendor.read_text() == "[Service]\nEnvironment=KEEP=1\n"
    restarts = sum(call[:2] == ["systemctl", "restart"] for call in state["calls"])
    probe.close_window(arguments)
    assert state["emitted"][-1]["already_closed"] is True
    assert restarts == 2
    assert sum(call[:2] == ["systemctl", "restart"] for call in state["calls"]) == 2


@pytest.mark.parametrize("failure", [probe.ProbeError("restart failed"), Abort()])
def test_half_open_window_restores_on_exception_or_abort(
    window: dict[str, Any], failure: BaseException
) -> None:
    arguments, state = window["arguments"], window["state"]
    state["fail_restart"] = failure
    with pytest.raises(type(failure)) as caught:
        probe.open_window(arguments)
    assert caught.value is failure
    assert not state["timer_active"]
    assert state["service_active"]
    assert not probe.window_paths(arguments.run_id)["root"].exists(), (
        f"{type(failure).__name__}: partial-open rollback left recovery files"
    )
    assert not probe.dropin_path(arguments.unit, arguments.run_id).exists(), (
        f"{type(failure).__name__}: partial-open rollback left the collector override"
    )


@pytest.mark.parametrize("field", ["fail_arm", "inactive_timer"])
def test_unarmed_window_never_installs_dropin_or_restarts_collector(
    window: dict[str, Any], field: str
) -> None:
    arguments, state = window["arguments"], window["state"]
    state[field] = True
    with pytest.raises(probe.ProbeError):
        probe.open_window(arguments)
    assert not any(call[:2] == ["systemctl", "restart"] for call in state["calls"]), (
        f"{field}: collector restarted without an armed watchdog: {state['calls']!r}"
    )
    assert not probe.dropin_path(arguments.unit, arguments.run_id).exists(), (
        f"{field}: an unarmed window installed a collector override"
    )
    assert not probe.window_paths(arguments.run_id)["root"].exists(), (
        f"{field}: failed watchdog admission left a private window directory"
    )


@pytest.mark.parametrize("failure_mode", ["command_failure", "bad_readback"])
def test_failed_close_keeps_watchdog_and_recovery_state_for_retry(
    window: dict[str, Any], failure_mode: str
) -> None:
    arguments, state = window["arguments"], window["state"]
    probe.open_window(arguments)
    if failure_mode == "command_failure":
        state["fail_restart"] = probe.ProbeError("restore restart failed")
    else:
        state["bad_readback"] = True
    with pytest.raises(probe.ProbeError):
        probe.close_window(arguments)
    paths = probe.window_paths(arguments.run_id)
    assert paths["state"].is_file(), (
        f"{failure_mode}: failed close discarded recovery state {paths['state']}"
    )
    assert paths["probe_copy"].is_file(), (
        f"{failure_mode}: failed close discarded the watchdog probe"
    )
    assert state["timer_active"]
    state["bad_readback"] = False
    probe.close_window(arguments)
    assert not paths["root"].exists(), (
        f"{failure_mode}: successful retry left recovery directory {paths['root']}"
    )
    assert not state["timer_active"]
    assert state["service_active"]


@pytest.mark.parametrize("field", ["run_id", "unit", "dropin"])
def test_close_refuses_tampered_window_identity(
    window: dict[str, Any], tmp_path: Path, field: str
) -> None:
    arguments, state = window["arguments"], window["state"]
    probe.open_window(arguments)
    path = probe.window_paths(arguments.run_id)["state"]
    record = json.loads(path.read_text())
    foreign = tmp_path / "foreign-file"
    foreign.write_text("preserve")
    record[field] = (
        str(foreign)
        if field == "dropin"
        else "gpu-fault-metrics-collector.service"
        if field == "unit"
        else "another-run"
    )
    path.write_text(json.dumps(record))
    before = len(state["calls"])
    with pytest.raises(probe.ProbeError, match="identity mismatch"):
        probe.close_window(arguments)
    assert foreign.read_text() == "preserve"
    assert len(state["calls"]) == before
    assert state["timer_active"]


def test_existing_window_is_not_replaced_or_rolled_back(window: dict[str, Any]) -> None:
    arguments, state = window["arguments"], window["state"]
    probe.open_window(arguments)
    calls = len(state["calls"])
    with pytest.raises(probe.ProbeError, match="already exists"):
        probe.open_window(arguments)
    assert len(state["calls"]) == calls
    assert state["timer_active"]


def test_remote_open_abort_attempts_unit_bound_rollback() -> None:
    calls: list[tuple[str, ...]] = []
    abort = Abort()

    class Fixture:
        def execute(self, *arguments: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(arguments)
            if arguments[0] == "open-window":
                raise abort
            return {}

    unit = "gpu-fault-host-collector.service"
    with pytest.raises(Abort) as caught:
        open_window_or_rollback(Fixture(), "review", "--unit", unit)
    assert caught.value is abort
    assert calls[-1] == ("close-window", "--run-id", "review", "--unit", unit)
