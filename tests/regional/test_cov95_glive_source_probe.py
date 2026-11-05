"""Host-collector source replay probe: its refusals, guards and read-only reads.

These run the probe's ``main`` over the ``FakeHost`` of
``test_general_live_source_probe`` (systemctl, systemd-run and nsenter, with
the unit's private view modelled from the drop-in the probe writes): every
step a window may refuse before it touches the node, the closes that undo a
window whose unit or verification failed, the deadman's own guards, and the
real ``/proc`` reader, clock helpers, command helper and script entry point.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional.test_general_live_source_probe import (
    KERNEL_UNIT,
    MAIN_PID,
    RUN,
    UNIT,
    FakeHost,
    make_host,
)
from tools import host_command_replay as wrapper

MEMINFO = b"MemTotal: 1000000 kB\nMemFree: 10 kB\nMemAvailable: 20 kB\n"
#: An older wrapper: it validates tools and units but knows no per-unit plans.
OLD_WRAPPER = (
    b'TOOL_ENV = "GFRS_TOOL"\nDIR_ENV = "GFRS_DIR"\n\n\n'
    b"def checked_tool(tool):\n    return tool\n\n\n"
    b"def checked_unit(unit):\n    return unit\n"
)
OPEN = ["open-window", "--run-id", RUN]
BIND = ["--mode", "bind", "--bind", "meminfo:/proc/meminfo"]


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    return make_host(tmp_path, monkeypatch)


def call(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def install(capsys: pytest.CaptureFixture[str], name: str, content: bytes) -> None:
    digest = hashlib.sha256(content).hexdigest()
    encoded = base64.b64encode(content).decode("ascii")
    argv = ["install-source", "--run-id", RUN, "--name", name, "--sha256", digest]
    code, receipt = call(capsys, *argv, "--b64", encoded)
    assert code == 0, receipt


def plan(tool: str) -> bytes:
    rule = {"query": "scan", "action": "exit", "exit_code": 2}
    return json.dumps(
        {"tool": tool, "not_after": time.time() + 600, "rules": [rule]}
    ).encode()


def after_run(
    monkeypatch: pytest.MonkeyPatch, host: FakeHost, hook: Any
) -> list[list[str]]:
    """``hook(command)`` runs after each command the fake host answered."""

    def run(command: list[str], **options: Any) -> Any:
        result = host.run(command, **options)
        hook(command)
        return result

    monkeypatch.setattr(probe, "run", run)
    return host.calls


def untouched(host: FakeHost) -> bool:
    """Nothing was armed, written or restarted."""

    return (
        not host.timers
        and host.restarts == 0
        and not probe.dropin_path(RUN).exists()
        and not probe.load_state(RUN)
    )


# --------------------------------------------------------------------------- #
# helpers that run for real: command, /proc, clocks, unit state
# --------------------------------------------------------------------------- #
def test_run_names_the_command_and_a_bounded_stderr_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_run(command: list[str], **options: Any) -> Any:
        seen.append(options)
        return subprocess.CompletedProcess(command, 2, "", "x" * 400 + "\n")

    fake = SimpleNamespace(run=fake_run, PIPE=subprocess.PIPE)
    monkeypatch.setattr(probe, "subprocess", fake)
    with pytest.raises(probe.ProbeError) as caught:
        probe.run(["systemctl", "restart", UNIT])
    assert str(caught.value) == f"command failed (2): systemctl restart {UNIT}: " + (
        "x" * 300
    )
    assert caught.value.detail is None
    assert (
        probe.run(["systemctl", "show", UNIT], check=False, timeout=9).returncode == 2
    )
    common = {"text": True, "errors": "replace", "check": False}
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    assert seen == [
        {**common, **pipes, "timeout": 120},
        {**common, **pipes, "timeout": 9},
    ]


def test_unit_state_keeps_only_assignments(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "ActiveState=active\nWarning: unit file changed\n\nMainPID=4242\n"
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, output, ""),
    )
    assert probe.unit_state(UNIT) == {"ActiveState": "active", "MainPID": "4242"}


def test_monotonic_us_is_the_monotonic_clock_and_pause_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = time.monotonic_ns() // 1000
    value = probe.monotonic_us()
    assert before <= value <= time.monotonic_ns() // 1000
    slept: list[float] = []
    monkeypatch.setattr(probe, "time", SimpleNamespace(sleep=slept.append))
    probe.pause(0.5)
    assert slept == [0.5]


def test_process_environment_reads_only_the_asked_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc = tmp_path / "proc"

    def path(value: Any, *parts: Any) -> Path:
        item = Path(value, *parts)
        return (
            proc / item.relative_to("/proc") if item.is_relative_to("/proc") else item
        )

    monkeypatch.setattr(probe, "Path", path)
    (proc / MAIN_PID).mkdir(parents=True)
    (proc / MAIN_PID / "environ").write_bytes(
        b"PATH=/usr/bin\0GPU_FAULT_PCI_DEVICES_ROOT=/run/x\0SECRET=y\0NOEQUALS\0\0"
    )
    keys = ("PATH", probe.PCI_ENV)
    assert probe.process_environment(MAIN_PID, keys) == {
        "PATH": "/usr/bin",
        probe.PCI_ENV: "/run/x",
    }
    for pid in ("", "0", "x1"):
        assert probe.process_environment(pid, keys) == {}, f"no process {pid!r}"
    assert probe.process_environment("999", keys) == {}, "a process that exited"


def test_load_state_refuses_another_runs_window(host: FakeHost) -> None:
    probe.save_state(RUN, {"run_id": "gl007-another-run-a1-cafe"})
    with pytest.raises(probe.ProbeError, match="window identity mismatch"):
        probe.load_state(RUN)
    (probe.state_dir(RUN) / "window.json").write_text("[]")
    with pytest.raises(probe.ProbeError, match="window identity mismatch"):
        probe.load_state(RUN)


def test_the_unreadable_node_is_made_once_per_run(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = probe.unreadable_node(RUN)
    assert node == probe.state_dir(RUN) / probe.UNREADABLE_NODE and node.exists()
    made: list[Path] = []
    monkeypatch.setattr(probe, "make_unopenable_node", made.append)
    assert probe.unreadable_node(RUN) == node
    assert made == [], "an existing node is reused"


# --------------------------------------------------------------------------- #
# install-source, builders, set-value, wrappers
# --------------------------------------------------------------------------- #
TOO_BIG = b"#" * (probe.MAX_SOURCE_BYTES + 1)


@pytest.mark.parametrize(
    ("sha256", "b64", "message"),
    [
        ("abc", "AA==", "a sha256 is required for every installed source"),
        ("0" * 64, "not base64!", "source is not valid base64: "),
        (
            hashlib.sha256(TOO_BIG).hexdigest(),
            base64.b64encode(TOO_BIG).decode("ascii"),
            "installed source exceeds its size bound",
        ),
    ],
    ids=["no-sha256", "not-base64", "too-big"],
)
def test_install_source_refuses_what_it_cannot_prove(
    host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    sha256: str,
    b64: str,
    message: str,
) -> None:
    argv = ["install-source", "--run-id", RUN, "--name", "meminfo"]
    code, failure = call(capsys, *argv, "--sha256", sha256, "--b64", b64)
    assert code == 1 and failure["error_message"].startswith(message), failure
    assert not probe.state_dir(RUN).exists(), "nothing is written"


def test_install_source_never_replaces_a_built_fixture_tree(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = probe.fixtures_dir(RUN) / "infiniband"
    tree.mkdir(parents=True)
    argv = ["install-source", "--run-id", RUN, "--name", "infiniband"]
    digest = hashlib.sha256(b"x").hexdigest()
    code, failure = call(capsys, *argv, "--sha256", digest, "--b64", "eA==")
    assert (
        code == 1 and failure["error_message"] == "a fixture tree of that name exists"
    )
    assert tree.is_dir() and not any(tree.iterdir()), "the tree is left as it was"


def test_installed_builders_that_cannot_be_loaded_are_refused(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, probe.PROBE_ALIAS, probe)
    installed = probe.fixtures_dir(RUN) / probe.BUILDERS_NAME
    installed.parent.mkdir(parents=True)
    installed.write_text("raise SystemExit('never executed')\n")
    with monkeypatch.context() as patched:
        patched.setattr(
            probe.importlib.util, "spec_from_file_location", lambda *a, **k: None
        )
        with pytest.raises(probe.ProbeError, match="fixture builders cannot be loaded"):
            probe.load_builders_module(RUN)


def test_read_source_needs_the_units_main_process(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    host.units[UNIT] = "0"
    argv = ["read-source", "--run-id", RUN, "--path", "/proc/meminfo"]
    code, failure = call(capsys, *argv)
    assert code == 1
    assert failure["error_message"] == f"{UNIT} has no main process to read through"


def test_set_value_needs_an_open_window_and_an_installed_fixture(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["set-value", "--run-id", RUN, "--name", "meminfo", "--value", "MemFree: 1"]
    code, failure = call(capsys, *argv)
    assert code == 1
    assert failure["error_message"] == "no source replay window is open for this run"
    probe.save_state(RUN, {"run_id": RUN, "unit": UNIT})
    code, failure = call(capsys, *argv, "--path", "ports/1/state")
    assert (code, failure["error_message"]) == (1, "fixture tree is not built")
    code, failure = call(capsys, *argv)
    assert (code, failure["error_message"]) == (1, "fixture is not installed")
    fixture = probe.fixtures_dir(RUN) / "meminfo"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(MEMINFO.decode())
    code, receipt = call(capsys, *argv)
    assert code == 0 and receipt["value"] == "MemFree: 1"
    assert receipt["unit"]["ActiveState"] == "active"
    assert fixture.read_text() == "MemFree: 1", "rewritten in place"


def test_install_wrapper_needs_a_loadable_wrapper_that_knows_unit_plans(
    host: FakeHost, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = ["install-wrapper", "--run-id", RUN, "--tool", "smartctl"]
    code, failure = call(capsys, *argv)
    assert code == 1
    assert failure["error_message"] == (
        f"install {probe.WRAPPER_NAME} before wrapping a tool"
    )
    install(capsys, probe.WRAPPER_NAME, OLD_WRAPPER)
    install(capsys, "plan-smartctl.json", plan("smartctl"))
    with monkeypatch.context() as patched:
        patched.setattr(
            probe.importlib.util, "spec_from_file_location", lambda *a, **k: None
        )
        with pytest.raises(probe.ProbeError, match="wrapper cannot be loaded"):
            probe.load_wrapper_module(RUN)
    code, failure = call(capsys, *argv)
    assert code == 1
    assert failure["error_message"] == (
        "the installed wrapper does not know per-unit plans"
    )
    assert not probe.bin_dir(RUN).exists(), "no shim for a wrapper it cannot check"


def test_install_wrapper_refuses_a_tool_the_host_lacks_unless_allowed(
    host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(probe, "STANDARD_PATH", str(tmp_path / "empty-bin"))
    install(capsys, probe.WRAPPER_NAME, Path(wrapper.__file__).read_bytes())
    install(capsys, "plan-smartctl.json", plan("smartctl"))
    argv = ["install-wrapper", "--run-id", RUN, "--tool", "smartctl"]
    code, failure = call(capsys, *argv)
    assert code == 1 and failure["error_message"].startswith(
        "smartctl is not installed on the host; pass --allow-missing-real"
    ), failure
    code, receipt = call(capsys, *argv, "--allow-missing-real")
    assert code == 0 and receipt["real"] is None and receipt["rules"] == 1, receipt
    assert Path(receipt["shim"]).is_file(), "the shim replays without a real binary"


# --------------------------------------------------------------------------- #
# open-window refusals and the closes of a failed open
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--mode", "command"], "command mode needs at least one --wrapper TOOL"),
        (["--mode", "tree"], "tree mode needs at least one --tree SRC:DEST"),
        (["--mode", "mixed"], "a window must bind, wrap, hide or override something"),
        (BIND, "bind source "),
        (["--mode", "tree", "--tree", "ib:/sys/class/infiniband"], "tree source "),
        (["--mode", "pci-root"], "build the PCI fixture before opening a pci-root"),
    ],
    ids=["no-wrapper", "no-tree", "nothing", "no-bind-file", "no-tree-dir", "no-pci"],
)
def test_open_window_refuses_an_empty_or_unbuilt_window(
    host: FakeHost, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    code, failure = call(capsys, *OPEN, *argv)
    assert code == 1 and failure["error_message"].startswith(message), failure
    assert untouched(host), host.calls


def test_open_window_refuses_an_open_window_a_stray_drop_in_and_a_down_unit(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    install(capsys, "meminfo", MEMINFO)
    probe.save_state(RUN, {"run_id": RUN, "unit": UNIT})
    code, failure = call(capsys, *OPEN, *BIND)
    assert (
        failure["error_message"]
        == "a source replay window for this run is already open"
    )
    (probe.state_dir(RUN) / "window.json").unlink()
    dropin = probe.dropin_path(RUN)
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\n")
    code, failure = call(capsys, *OPEN, *BIND)
    assert (
        failure["error_message"] == "an untracked source replay drop-in already exists"
    )
    dropin.unlink()
    host.unit_active = False
    code, failure = call(capsys, *OPEN, *BIND)
    assert (code, failure["error_message"]) == (1, f"{UNIT} is not active at baseline")
    assert host.restarts == 0 and not host.timers, "nothing was armed or restarted"


def test_open_window_closes_again_when_the_unit_does_not_come_up(
    host: FakeHost, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(capsys, "meminfo", MEMINFO)

    def flap(command: list[str]) -> None:
        if command[:2] == ["systemctl", "restart"]:
            host.unit_active = host.restarts > 1

    after_run(monkeypatch, host, flap)
    code, failure = call(capsys, *OPEN, *BIND)
    assert code == 1
    assert failure["error_message"] == (
        f"{UNIT} is not active after the window opened; closed"
    )
    assert host.restarts == 2, "the open's restart and the close's"
    assert not probe.dropin_path(RUN).exists() and not probe.state_dir(RUN).exists()
    assert not host.timers, "the close disarmed the deadman"


def test_open_window_closes_again_when_verification_cannot_run(
    host: FakeHost, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(capsys, "meminfo", MEMINFO)

    def lose_pid(command: list[str]) -> None:
        if command[:2] == ["systemctl", "restart"]:
            host.units[UNIT] = "0" if host.restarts == 1 else MAIN_PID

    after_run(monkeypatch, host, lose_pid)
    code, failure = call(capsys, *OPEN, *BIND)
    assert code == 1
    assert failure["error_message"] == (
        f"window verification failed: {UNIT} has no main process to verify "
        "against; closed"
    )
    assert host.restarts == 2 and not probe.dropin_path(RUN).exists()


def test_open_window_refuses_a_deadman_that_is_not_armed(
    host: FakeHost, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(capsys, "meminfo", MEMINFO)
    timer = f"{probe.deadman_unit(RUN)}.timer"
    after_run(monkeypatch, host, lambda command: host.timers.discard(timer))
    code, failure = call(capsys, *OPEN, *BIND)
    assert code == 1
    assert failure["error_message"] == "source replay deadman timer is not active"
    assert not probe.dropin_path(RUN).exists(), "no drop-in without a deadman"
    assert host.restarts == 0, "the unit is never restarted"


def test_disarm_deadman_leaves_the_deadman_running_this_close(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = probe.deadman_unit(RUN)

    def run(command: list[str], **options: Any) -> Any:
        if command[:3] == ["systemctl", "show", f"{name}.service"]:
            host.calls.append(list(command))
            stdout = f"ActiveState=active\nMainPID={os.getpid()}\n"
            return subprocess.CompletedProcess(command, 0, stdout, "")
        return host.run(command, **options)

    monkeypatch.setattr(probe, "run", run)
    probe.disarm_deadman(RUN)
    stops = [c for c in host.calls if c[:2] == ["systemctl", "stop"]]
    assert stops == [["systemctl", "stop", f"{name}.timer"]], stops
    assert host.calls[-1] == [
        "systemctl",
        "reset-failed",
        f"{name}.service",
        f"{name}.timer",
    ]


# --------------------------------------------------------------------------- #
# verify, close-window, await-ticks
# --------------------------------------------------------------------------- #
def test_verify_needs_an_open_window_and_the_units_main_process(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    code, failure = call(capsys, "verify", "--run-id", RUN)
    assert failure["error_message"] == "no source replay window is open for this run"
    probe.save_state(RUN, {"run_id": RUN, "unit": KERNEL_UNIT})
    host.units[KERNEL_UNIT] = "0"
    code, failure = call(capsys, "verify", "--run-id", RUN)
    assert (code, failure["error_message"]) == (
        1,
        f"{KERNEL_UNIT} has no main process to verify against",
    )


def test_close_window_keeps_the_deadman_when_the_unit_stays_down(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    install(capsys, "meminfo", MEMINFO)
    code, opened = call(capsys, *OPEN, *BIND)
    assert code == 0, opened
    host.unit_active = False
    code, failure = call(capsys, "close-window", "--run-id", RUN)
    assert code == 1
    assert failure["error_message"] == f"{UNIT} is not active after the window closed"
    assert f"{probe.deadman_unit(RUN)}.timer" in host.timers, "the deadman retries"


def test_close_window_removes_a_state_directory_left_without_a_window(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    (probe.fixtures_dir(RUN)).mkdir(parents=True)
    code, closed = call(capsys, "close-window", "--run-id", RUN)
    assert code == 0 and closed == {
        "run_id": RUN,
        "already_closed": True,
        "clean": True,
    }
    assert not probe.state_dir(RUN).exists(), "the fixtures went with it"
    assert host.restarts == 0, "nothing to restore, nothing restarted"


def test_await_ticks_without_close_leaves_the_window_open(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    install(capsys, "meminfo", MEMINFO)
    code, opened = call(capsys, *OPEN, *BIND)
    assert code == 0, opened
    host.pings = [20.0]
    argv = ["await-ticks", "--run-id", RUN, "--count", "1", "--interval", "15"]
    code, result = call(capsys, *argv)
    assert code == 0 and result["completed"] is True, result
    assert len(result["ticks"]) == 1 and "close" not in result, result
    assert probe.load_state(RUN)["unit"] == UNIT, "the window is still open"
    assert probe.dropin_path(RUN).exists(), "and its drop-in in place"


# --------------------------------------------------------------------------- #
# preflight and the script entry point
# --------------------------------------------------------------------------- #
def test_preflight_lists_tools_units_and_infiniband_devices(
    host: FakeHost,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sbin = tmp_path / "usr-sbin"
    sbin.mkdir()
    for name, mode in (("smartctl", 0o755), ("nsenter", 0o755), ("ipmitool", 0o644)):
        (sbin / name).write_text("#!/bin/sh\n")
        (sbin / name).chmod(mode)
    monkeypatch.setattr(probe, "STANDARD_PATH", f"{sbin}:{tmp_path / 'absent'}")
    sysfs = tmp_path / "sys"
    for device in ("rdmap1", "rdmap0"):
        (sysfs / "class" / "infiniband" / device).mkdir(parents=True)

    def path(value: Any, *parts: Any) -> Path:
        item = Path(value, *parts)
        return sysfs / item.relative_to("/sys") if item.is_relative_to("/sys") else item

    monkeypatch.setattr(probe, "Path", path)
    code, report = call(capsys, "preflight")
    assert code == 0, report
    assert report["tools"] == {
        "smartctl": [str(sbin / "smartctl")],
        "ipmitool": [],
        "ethtool": [],
        "nvidia-smi": [],
    }, "a file without the execute bit is no tool"
    assert report["nsenter"] is True
    assert report["infiniband_devices"] == ["rdmap0", "rdmap1"]
    assert set(report["units"]) == {UNIT, KERNEL_UNIT}
    assert report["unit"]["MainPID"] == MAIN_PID


def test_the_probe_file_runs_as_a_script_and_reports_a_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["general_live_source_probe.py", "snapshot", "--run-id", "Bad Run"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert caught.value.code == 1
    out, err = capsys.readouterr()
    failure = json.loads(out.splitlines()[-1])
    assert failure["error_message"] == "unsafe run ID"
    assert failure["command"] == "snapshot" and failure["error_detail"] is None
    assert failure["error_site"].startswith("safe_id:"), failure
    assert err == "ProbeError: unsafe run ID\n"
