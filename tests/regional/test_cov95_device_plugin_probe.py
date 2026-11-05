"""The device-plugin freeze probe over a fake ``/proc`` and a scripted systemd.

``test_general_live_device_plugin_probe.py`` replaces the process discovery
itself; here ``PROC_ROOT`` points at a tree holding the processes a live
plugin Pod runs, so the probe's own discovery, state letters and cgroup match
decide -- with a process whose cmdline, stat or cgroup cannot be read, a pid
reused by another process, a task that never reaches the stopped state inside
a first-timeout window, the deadman disarmed from inside its own unit, a
deadman that never armed, the node's product generation and the collector's
journal in their odd shapes, and the CLI's survey/residue/error paths, the real
``run`` wrapper and the script's ``__main__``. ``os.kill`` is always a recorder.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_device_plugin_probe as probe

RUN = "gl14-gpu-plugin-a1-deadbeef"
POD = "00000000-0000-0000-0000-00000000aaaa"
PLUGIN = "/usr/bin/nvidia-device-plugin"


class Systemd:
    """``systemctl`` / ``systemd-run`` / ``journalctl`` as the probe calls them."""

    def __init__(
        self,
        *,
        armed_state: str | None = "active",
        main_pid: str = "1",
        properties: str = "",
        journal: str = "",
    ) -> None:
        self.calls: list[list[str]] = []
        self.armed_state = armed_state
        self.main_pid = main_pid
        self.properties = properties
        self.journal = journal

    def __call__(
        self, command: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))
        stdout = ""
        if command[0] == "journalctl":
            stdout = self.journal
        elif command[:2] == ["systemctl", "show"]:
            prop = command[-1]
            if prop == "--property=ActiveState":
                stdout = (
                    f"ActiveState={self.armed_state}\n"
                    if self.armed_state is not None
                    else "LoadState=loaded\nSubState=dead\n"
                )
            elif prop == "--property=MainPID":
                stdout = f"ControlPID=0\nMainPID={self.main_pid}\n"
            else:
                stdout = self.properties
        return subprocess.CompletedProcess(command, 0, stdout, "")


def _stat(pid: int, name: str, state: str, ticks: str) -> str:
    fields = [state, *["0"] * 18, ticks, *["0"] * 5]
    return f"{pid} ({name}) " + " ".join(fields) + "\n"


class Proc:
    """A fake ``/proc``; SIGSTOP/SIGCONT flip the state letter of a listed pid."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.kills: list[tuple[int, int]] = []

    def add(
        self,
        pid: int,
        argv: list[str],
        *,
        pod: str = POD,
        state: str = "S",
        ticks: str = "100",
        cgroup: bool = True,
    ) -> Path:
        directory = self.root / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(
            b"\0".join(a.encode() for a in argv) + b"\0"
        )
        (directory / "stat").write_text(_stat(pid, Path(argv[0]).name, state, ticks))
        if cgroup:
            (directory / "cgroup").write_text(f"0::/kubepods/besteffort/pod{pod}/c1\n")
        return directory

    def kill(self, pid: int, sig: int) -> None:
        self.kills.append((pid, sig))
        stat = self.root / str(pid) / "stat"
        head, _, tail = stat.read_text().rpartition(") ")
        if tail.startswith("D"):
            return  # an uninterruptible sleep defers the signal
        letter = "T" if sig == probe.signal.SIGSTOP else "S"
        stat.write_text(f"{head}) {letter}{tail[1:]}")


@pytest.fixture
def proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Proc:
    root = tmp_path / "proc"
    root.mkdir()
    fake = Proc(root)
    monkeypatch.setattr(probe, "PROC_ROOT", root)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(probe.os, "kill", fake.kill)
    return fake


def _main(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# The real wrappers and the script
# --------------------------------------------------------------------------- #
def test_run_names_a_failed_command_by_its_code_and_stderr() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('denied'); sys.exit(2)",
    ]
    with pytest.raises(probe.ProbeError, match=r"command failed \(2\): .*: denied"):
        probe.run(command)
    assert probe.run(command, check=False).returncode == 2


def test_the_script_answers_help_as_a_standalone_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [probe.__file__, "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert "freeze" in out and "probe-log" in out, out


# --------------------------------------------------------------------------- #
# /proc reads
# --------------------------------------------------------------------------- #
def test_proc_reads_tolerate_a_vanished_or_reused_pid(proc: Proc) -> None:
    proc.add(4242, [PLUGIN], state="T", ticks="100")
    (proc.root / "4243").mkdir()  # exited: no cmdline, no stat
    blank = proc.add(4244, [PLUGIN])
    (blank / "stat").write_text("4244 (nvidia-device-plugin)\n")

    assert probe.process_cmdline(4243) == []
    assert probe.process_start_ticks(4243) is None
    assert probe.process_state(4243, None) is None
    assert probe.process_state(4244, None) is None, "a stat without fields"
    assert probe.process_state(4242, "100") == "T"
    assert probe.process_state(4242, "999") is None, "the pid now names another task"
    assert probe.process_is_stopped(4242, "100") is True
    assert probe.process_is_stopped(4242, "999") is False


def test_discovery_needs_the_pods_cgroup_and_an_allow_listed_binary(proc: Proc) -> None:
    proc.add(4242, [PLUGIN, "--fail-on-init-error=false"])
    proc.add(4250, [PLUGIN], cgroup=False)  # cgroup unreadable: not in the Pod
    proc.add(4260, ["/pause"])
    proc.add(4270, [PLUGIN], pod="11111111-1111-1111-1111-111111111111")
    (proc.root / "self").mkdir()
    found = probe.plugin_processes(POD, ["nvidia-device-plugin"])
    assert found == [
        {
            "pid": 4242,
            "start_ticks": "100",
            "binary": "nvidia-device-plugin",
            "argv": [PLUGIN, "--fail-on-init-error=false"],
        }
    ]
    assert probe.pod_cgroup_processes(POD) == [
        {"pid": 4242, "binary": "nvidia-device-plugin"},
        {"pid": 4260, "binary": "pause"},
    ]


# --------------------------------------------------------------------------- #
# freeze / resume / cleanup / residue
# --------------------------------------------------------------------------- #
def test_a_task_that_never_stops_in_a_first_timeout_window_is_resumed_at_once(
    proc: Proc, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Systemd()
    monkeypatch.setattr(probe, "run", fake)
    # The bounded wait for ``T`` ends unstopped (the task sits in ``D``).
    monkeypatch.setattr(probe, "wait_until_stopped", lambda pid, ticks: False)
    proc.add(4242, [PLUGIN], state="D")
    probe.probe_copy(RUN).write_text("# the copy the deadman was armed with\n")
    probe.freeze(
        argparse.Namespace(
            run_id=RUN,
            pod_uid=POD,
            binary=["nvidia-device-plugin"],
            pod_command_binary=["nvidia-device-plugin", "bad name"],
            seconds=300,
            resume_after_first_timeout=True,
            window_seconds=30,
        )
    )
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["frozen"] == [
        {"pid": 4242, "binary": "nvidia-device-plugin", "stopped": False, "state": "D"}
    ]
    assert receipt["resumed"] == [{"pid": 4242, "resumed": True, "stopped": False}]
    assert "first_timeout" not in receipt, "no journal watch for an unstopped task"
    assert proc.kills == [(4242, probe.signal.SIGSTOP), (4242, probe.signal.SIGCONT)]
    assert probe.load_state(RUN)["paused_processes"] == {}
    assert (
        probe.probe_copy(RUN).read_text() == "# the copy the deadman was armed with\n"
    )


def test_a_deadman_that_never_arms_refuses_before_any_signal(
    proc: Proc, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe, "run", Systemd(armed_state=None))
    proc.add(4242, [PLUGIN])
    code, error = _main(
        capsys,
        "freeze",
        "--run-id",
        RUN,
        "--pod-uid",
        POD,
        "--binary",
        "nvidia-device-plugin",
        "--seconds",
        "300",
    )
    assert code == 1, error
    assert error["error_message"] == (
        f"deadman {probe.unit_name(RUN, 'deadman-' + probe.RESUME_KIND)} is not armed"
    )
    assert error["error_site"].startswith("arm_deadman:"), error["error_site"]
    assert proc.kills == [], "nothing is signalled without the deadman"


def test_the_deadmans_cleanup_spares_its_own_unit_and_the_residue_reads_the_pids(
    proc: Proc, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Systemd(main_pid=str(os.getpid()))
    monkeypatch.setattr(probe, "run", fake)
    proc.add(4242, [PLUGIN], state="T", ticks="100")
    proc.add(4243, [PLUGIN], state="S", ticks="777")  # reused pid
    paused = {
        "4242": {"start_ticks": "100", "binary": "nvidia-device-plugin"},
        "4243": {"start_ticks": "100", "binary": "nvidia-device-plugin"},
    }
    probe.save_state(RUN, {"run_id": RUN, "paused_processes": dict(paused)})
    code, surveyed = _main(capsys, "survey", "--run-id", RUN)
    assert code == 0, surveyed
    assert surveyed["residue"] == {"paused_processes": {"4242": paused["4242"]}}
    assert surveyed["clean"] is False, "a still-stopped plugin is residue"
    code, cleaned = _main(capsys, "cleanup", "--run-id", RUN, "--kind", "all")
    assert code == 0 and cleaned["clean"] is True, cleaned
    assert cleaned["processes"] == [
        {"pid": 4242, "resumed": True, "stopped": False},
        {"pid": 4243, "resumed": False, "gone": True},
    ]
    service = f"{probe.unit_name(RUN, 'deadman-' + probe.RESUME_KIND)}.service"
    assert ["systemctl", "stop", service] not in fake.calls, (
        "the deadman never stops the unit it runs in"
    )
    code, shown = _main(capsys, "residue", "--run-id", RUN)
    assert code == 0 and shown == {"residue": {"paused_processes": {}}, "clean": True}


def test_the_cli_refuses_an_unsafe_run_and_a_state_of_another_run(
    proc: Proc, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe, "run", Systemd())
    code, error = _main(capsys, "survey", "--run-id", "Not A Run")
    assert code == 1 and error["error_message"] == "unsafe run ID", error
    assert error["error_site"].startswith("survey:"), (
        "checked_run_id is a helper frame: the site is the command"
    )
    probe.save_state(RUN, {"run_id": "gl14-other-a1-cafe", "paused_processes": {}})
    code, error = _main(capsys, "residue", "--run-id", RUN)
    assert code == 1, error
    assert error["error_message"] == "device-plugin probe state identity mismatch"


# --------------------------------------------------------------------------- #
# The product generation and the collector's journal (read-only)
# --------------------------------------------------------------------------- #
def test_feature_reads_the_units_venv_release_link_and_plugin_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    venv = tmp_path / "releases" / "r20" / "venv"
    for relative in probe.FEATURE_MODULES.values():
        module = venv / "lib" / "python3.12" / "site-packages" / relative
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("")
    current = tmp_path / "current"
    current.symlink_to(tmp_path / "releases" / "r20")
    env = tmp_path / "collector.env"
    env.write_text(
        "# installer\nGPU_FAULT_DEVICE_PLUGIN_TIMEOUT_SECONDS='3'\nno assignment\n"
        "GPU_FAULT_OTHER=1\n"
    )
    monkeypatch.setattr(probe, "AGENT_CURRENT", current)
    monkeypatch.setattr(probe, "COLLECTOR_ENV", env)
    executable = venv / "bin" / "gpu-fault-collector"
    monkeypatch.setattr(
        probe,
        "run",
        Systemd(
            properties=(
                f"ExecStart={{ path={executable} ; argv[]={executable} host ; }}\n"
                "Environment=GPU_FAULT_DEVICE_PLUGIN_REQUIRED=2 LANG=C.UTF-8\n"
                "ActiveState=active\nInvocationID=0f1e\n"
            )
        ),
    )
    code, value = _main(capsys, "feature", "--run-id", RUN)
    assert code == 0, value
    assert value["venv"] == str(venv) and value["venv_present"] is True
    assert value["current_release"] == str(tmp_path / "releases" / "r20")
    assert value["liveness_probe"] is True and value["active_state"] == "active"
    assert value["device_plugin_env"] == {
        "GPU_FAULT_DEVICE_PLUGIN_TIMEOUT_SECONDS": "3",
        "GPU_FAULT_DEVICE_PLUGIN_REQUIRED": "2",
    }, "the unit's own Environment= wins over the file and only the prefix counts"


TIMEOUT_LINE = (
    "device plugin nvidia-gpu.sock did not answer GetDevicePluginOptions within "
    "2.0s (1 consecutive; finding after 2)"
)


def test_the_journal_decodes_byte_messages_and_stops_at_its_bound(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = "\n".join(
        [
            json.dumps(
                {
                    "MESSAGE": list(TIMEOUT_LINE.encode()),
                    "__REALTIME_TIMESTAMP": "not-a-number",
                }
            ),
            json.dumps(
                {
                    "MESSAGE": "device plugin socket directory "
                    "/var/lib/kubelet/device-plugins is absent",
                    "__REALTIME_TIMESTAMP": "1793084182000000",
                }
            ),
        ]
    )
    monkeypatch.setattr(probe, "run", Systemd(journal=journal))
    code, log = _main(capsys, "probe-log", "--run-id", RUN, "--since-epoch", "1.5")
    assert code == 0, log
    first, second = log["entries"]
    assert first["kind"] == "timeout" and first["realtime_epoch"] == 0.0, first
    assert first["sockets"] == ["nvidia-gpu.sock"] and first["consecutive"] == 1
    assert second == {
        "realtime_epoch": 1793084182.0,
        "message": "device plugin socket directory /var/lib/kubelet/device-plugins "
        "is absent",
        "kind": "directory_absent",
        "directory": "/var/lib/kubelet/device-plugins",
    }
    monkeypatch.setattr(probe, "MAX_JOURNAL_ENTRIES", 1)
    assert [item["kind"] for item in probe.collector_journal(1.5)] == ["timeout"], (
        "the read stops at MAX_JOURNAL_ENTRIES"
    )
