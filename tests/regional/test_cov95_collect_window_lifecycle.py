"""Collector overrides retain recovery ownership through failures and aborts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_window_probe as probe
from tests.regional import _cov95_collect_window as window_support
from tests.regional._cov95_collect_net import (
    StopLoop,
    no_external_effects,  # noqa: F401
)
from tests.regional._cov95_collect_window import window_arguments

window_host = window_support.window_host


@pytest.mark.parametrize("mode", ["override", "unset", "hang", "drop"])
@pytest.mark.parametrize("self_recovery", [False, True])
def test_window_restores_original_state_and_does_not_stop_its_own_watchdog(
    window_host: Any, mode: str, self_recovery: bool
) -> None:
    host = window_host
    args = window_arguments(unset=[])
    if mode == "override":
        args.env = ["GPU_FAULT_CONTROL_PLANE_TOKEN=@invalid"]
    elif mode == "unset":
        args.unset = ["GPU_FAULT_EXPECTED_GPU_COUNT"]
    elif mode == "hang":
        args.shadow_nvidia_smi = "hang:30"
    else:
        args.shadow_nvidia_smi = "drop-uuid:GPU-aaaaaaaa:1"
    original = probe.COLLECTOR_ENV.read_bytes()
    probe.open_window(args)
    opened = host.emitted[-1]
    paths = probe.window_paths(args.run_id)
    dropin = probe.dropin_path(args.unit, args.run_id)
    assert paths["state"].stat().st_mode & 0o777 == 0o600
    assert dropin.is_file(), "window must install its owned drop-in"
    assert opened["restore_needed"] is True
    assert opened["after"]["ActiveState"] == "active"
    if mode == "override":
        assert paths["override_env"].stat().st_mode & 0o777 == 0o600
        assert opened["overrides"] == ["GPU_FAULT_CONTROL_PLANE_TOKEN"]
    if mode in {"hang", "drop"}:
        assert (paths["shadow_dir"] / "nvidia-smi").stat().st_mode & 0o777 == 0o755
    commands = [call[0] for call in host.calls]
    armed = next(i for i, call in enumerate(commands) if call[0] == "systemd-run")
    restarted = commands.index(["systemctl", "restart", args.unit])
    assert armed < restarted
    host.calls.clear()
    host.timer_pid = "555" if self_recovery else "123"
    probe.close_window(args)
    commands = [call[0] for call in host.calls]
    assert (["systemctl", "stop", opened["deadman_unit"] + ".service"] in commands) is (
        not self_recovery
    )
    assert not paths["root"].exists(), "successful restore must delete owned state"
    assert not dropin.exists(), "successful restore must remove only its drop-in"
    assert probe.COLLECTOR_ENV.read_bytes() == original
    assert host.emitted[-1]["window_root_removed"] is True
    probe.close_window(args)
    assert host.emitted[-1]["already_closed"] is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"env": ["OTHER=value"]}, "not an overridable"),
        ({"env": ["GPU_FAULT_CONTROL_PLANE_TOKEN=not-approved"]}, "accepts only"),
        ({"env": ["GPU_FAULT_CONTROL_PLANE_TOKEN"]}, "not an overridable"),
        ({"unset": ["PATH"]}, "may not be unset"),
        ({"unset": []}, "override, unset or shadow"),
        ({"restore_seconds": 1}, "restore seconds"),
        ({"restore_seconds": 1801}, "restore seconds"),
        ({"unit": "gpu-fault-node-agent.service"}, "allow-list"),
        ({"run_id": "../unsafe"}, "unsafe run ID"),
        ({"shadow_nvidia_smi": "bad"}, "shadow mode"),
        ({"shadow_nvidia_smi": "hang:0"}, "1..120"),
        ({"shadow_nvidia_smi": "drop-uuid:GPU-aaaaaaaa:2"}, "at most"),
        ({"shadow_nvidia_smi": "drop-uuid:GPU-bbbbbbbb:1"}, "does not have"),
    ],
)
def test_invalid_windows_never_install_override(
    window_host: Any, changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(probe.ProbeError, match=message):
        probe.open_window(window_arguments(**changes))
    assert not any(call[0][0] == "systemd-run" for call in window_host.calls), (
        "invalid window must fail before watchdog or collector changes"
    )


@pytest.mark.parametrize("existing", ["root", "dropin", "inactive"])
def test_open_preserves_preexisting_window_or_inactive_baseline(
    window_host: Any, existing: str
) -> None:
    args = window_arguments()
    paths = probe.window_paths(args.run_id)
    dropin = probe.dropin_path(args.unit, args.run_id)
    if existing == "root":
        paths["root"].mkdir(parents=True)
        (paths["root"] / "foreign").write_text("preserve")
    elif existing == "dropin":
        dropin.parent.mkdir(parents=True)
        dropin.write_text("preserve")
    else:
        window_host.states[args.unit] = "inactive"
    with pytest.raises(probe.ProbeError):
        probe.open_window(args)
    if existing == "root":
        assert (paths["root"] / "foreign").read_text() == "preserve"
    elif existing == "dropin":
        assert dropin.read_text() == "preserve"
    assert not any(call[0][0] == "systemd-run" for call in window_host.calls), (
        "baseline refusal cannot arm or mutate"
    )


@pytest.mark.parametrize(
    "failure_at", ["copy", "timer", "inactive-timer", "restart", "abort"]
)
def test_partial_open_rolls_back_on_failure_or_abort(
    window_host: Any, monkeypatch: Any, failure_at: str
) -> None:
    host = window_host
    args = window_arguments()
    failure: BaseException = (
        StopLoop() if failure_at == "abort" else RuntimeError("fixture failure")
    )
    if failure_at == "copy":

        def copy(*args: Any, **kwargs: Any) -> None:
            raise failure

        monkeypatch.setattr(probe.shutil, "copyfile", copy)
    elif failure_at == "inactive-timer":
        original = probe.unit_state
        monkeypatch.setattr(
            probe,
            "unit_state",
            lambda unit: {"ActiveState": "inactive"}
            if unit.endswith(".timer")
            else original(unit),
        )
        failure = probe.ProbeError("timer inactive")
    else:
        key = (
            ("systemd-run",)
            if failure_at == "timer"
            else ("systemctl", "restart", args.unit)
        )
        host.failures[key] = [failure]
    with pytest.raises(type(failure)):
        probe.open_window(args)
    paths = probe.window_paths(args.run_id)
    assert not paths["root"].exists(), (
        "failed open must remove successfully restored state"
    )
    assert not probe.dropin_path(args.unit, args.run_id).exists(), (
        "rollback must remove override"
    )
    if failure_at in {"copy", "timer", "inactive-timer"}:
        assert not any(
            call[0][:2] == ["systemctl", "restart"] for call in host.calls
        ), "collector must not restart before a proven active watchdog"


def test_failed_rollback_preserves_state_and_error_note_for_retry(
    window_host: Any,
) -> None:
    host = window_host
    args = window_arguments()
    initial = RuntimeError("opening restart failed")
    host.failures[("systemctl", "restart", args.unit)] = [
        initial,
        RuntimeError("rollback restart failed"),
    ]
    with pytest.raises(RuntimeError) as caught:
        probe.open_window(args)
    assert caught.value is initial
    assert caught.value.__notes__ == ["collector window rollback failed: RuntimeError"]
    paths = probe.window_paths(args.run_id)
    assert paths["state"].is_file(), "failed cleanup must keep its recovery receipt"
    assert paths["probe_copy"].is_file(), (
        "failed cleanup must keep independent recovery"
    )
    assert host.states[probe.deadman_unit(args.run_id) + ".timer"] == "active"
    probe.close_window(args)
    assert not paths["root"].exists(), "retry must complete the same owned restore"


@pytest.mark.parametrize("field", ["run_id", "unit", "dropin", "not-object"])
def test_close_refuses_identity_drift_without_touching_owned_files(
    window_host: Any, field: str
) -> None:
    host = window_host
    args = window_arguments()
    probe.open_window(args)
    path = probe.window_paths(args.run_id)["state"]
    record = json.loads(path.read_text())
    if field == "not-object":
        record = []
    else:
        record[field] = (
            probe.ALLOWED_UNITS[1] if field == "unit" else "another-identity"
        )
    path.write_text(json.dumps(record))
    host.calls.clear()
    with pytest.raises(probe.ProbeError, match="identity mismatch"):
        probe.close_window(args)
    assert host.calls == [], "mismatched recovery identity must not run host commands"
    assert probe.dropin_path(args.unit, args.run_id).exists(), (
        "foreign state cannot authorize deletion"
    )


@pytest.mark.parametrize("problem", ["no-unit", "root-only", "dropin-only", "inactive"])
def test_close_without_state_requires_absence_and_healthy_unit(
    window_host: Any, problem: str
) -> None:
    args = window_arguments()
    if problem == "no-unit":
        args.unit = ""
    elif problem == "root-only":
        probe.window_paths(args.run_id)["root"].mkdir(parents=True)
    elif problem == "dropin-only":
        path = probe.dropin_path(probe.ALLOWED_UNITS[1], args.run_id)
        path.parent.mkdir(parents=True)
        path.touch()
    else:
        window_host.states[args.unit] = "inactive"
    with pytest.raises(probe.ProbeError):
        probe.close_window(args)


@pytest.mark.parametrize("phase", ["open", "close"])
def test_unhealthy_restart_does_not_disarm_watchdog(
    window_host: Any, monkeypatch: Any, phase: str
) -> None:
    host = window_host
    args = window_arguments()
    if phase == "close":
        probe.open_window(args)
    original = probe.unit_state

    def state(unit: str) -> dict[str, str]:
        value = original(unit)
        restarts = [
            call
            for call in host.calls
            if call[0] == ["systemctl", "restart", args.unit]
        ]
        if unit == args.unit and len(restarts) >= (1 if phase == "open" else 2):
            value["ActiveState"] = "inactive"
        return value

    monkeypatch.setattr(probe, "unit_state", state)
    with pytest.raises(probe.ProbeError, match="not active"):
        (probe.open_window if phase == "open" else probe.close_window)(args)
    assert probe.window_paths(args.run_id)["state"].is_file(), (
        "keep failed restore evidence"
    )
    assert host.states[probe.deadman_unit(args.run_id) + ".timer"] == "active"


def test_missing_shadow_binary_aborts_open_and_restores(
    window_host: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(probe.shutil, "which", lambda *a, **k: None)
    with pytest.raises(probe.ProbeError, match="not installed"):
        probe.open_window(window_arguments(shadow_nvidia_smi="hang:30"))
    assert not probe.window_paths("window-a")["root"].exists(), (
        "failed shadow setup must restore"
    )
