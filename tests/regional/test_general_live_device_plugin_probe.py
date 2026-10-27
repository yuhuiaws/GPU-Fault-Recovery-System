"""Contract tests for the device-plugin freeze probe (no systemd, no /proc).

``run`` (systemctl / systemd-run) is a fake that models the deadman timer, and
``plugin_processes`` / ``os.kill`` / ``process_is_stopped`` are replaced so the
test can prove: only an allow-listed, pod-scoped process is signalled, the
resume deadman is armed *before* the first SIGSTOP, the state records exactly
what was frozen, and resume/cleanup clear it and disarm the deadman.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_device_plugin_probe as probe

RUN = "gl14-gpu-plugin-a1-deadbeef"
POD = "00000000-0000-0000-0000-00000000aaaa"


class FakeRun:
    """systemctl / systemd-run, and an ordered log shared with ``os.kill``."""

    def __init__(self) -> None:
        self.log: list[tuple[str, Any]] = []
        self.timer_active = False

    def __call__(self, command: list[str], *, check: bool = True, timeout: int = 120):
        import subprocess

        self.log.append(("run", list(command)))
        stdout = ""
        if command[0] == "systemd-run":
            self.timer_active = True
        elif command[:2] == ["systemctl", "show"]:
            if "--property=ActiveState" in command:
                stdout = f"ActiveState={'active' if self.timer_active else 'inactive'}"
            elif "--property=MainPID" in command:
                stdout = "MainPID=1"
        elif command[:2] == ["systemctl", "stop"] and command[-1].endswith(".timer"):
            self.timer_active = False
        return subprocess.CompletedProcess(command, 0, stdout, "")


@pytest.fixture()
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    fake = FakeRun()
    monkeypatch.setattr(probe, "run", fake)
    kills: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        fake.log.append(("kill", (pid, sig)))
        kills.append((pid, sig))

    monkeypatch.setattr(probe.os, "kill", fake_kill)
    monkeypatch.setattr(probe, "process_is_stopped", lambda pid, ticks: True)
    monkeypatch.setattr(probe, "process_start_ticks", lambda pid: "100")
    fake.kills = kills  # type: ignore[attr-defined]
    return fake


def _args(**values: Any) -> Any:
    import argparse

    return argparse.Namespace(**values)


# --------------------------------------------------------------------------- #
# validators
# --------------------------------------------------------------------------- #
def test_checked_binaries_narrows_to_allow_list() -> None:
    assert probe.checked_binaries(["nvidia-device-plugin", "kubelet", "sshd"]) == [
        "nvidia-device-plugin"
    ]
    with pytest.raises(probe.ProbeError):
        probe.checked_binaries(["kubelet", "node-agent"])


def test_checked_pod_uid() -> None:
    assert probe.checked_pod_uid(POD) == POD
    with pytest.raises(probe.ProbeError):
        probe.checked_pod_uid("not-a-uid")


def test_cgroup_matches_across_drivers() -> None:
    systemd = (
        f"0::/kubepods.slice/kubepods-besteffort-pod{POD.replace('-', '_')}.slice/x"
    )
    cgroupfs = f"0::/kubepods/besteffort/pod{POD}/abc"
    assert probe.cgroup_names_pod(systemd, POD), "systemd cgroup driver must match"
    assert probe.cgroup_names_pod(cgroupfs, POD), "cgroupfs driver must match"
    assert not probe.cgroup_names_pod(
        "0::/kubepods/besteffort/podffffffff-0000-0000-0000-000000000000/abc", POD
    ), "a different pod UID must not match"


# --------------------------------------------------------------------------- #
# freeze / resume / cleanup
# --------------------------------------------------------------------------- #
def test_freeze_arms_deadman_before_kill_and_records_state(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        probe,
        "plugin_processes",
        lambda pod_uid, binaries: [
            {
                "pid": 4242,
                "start_ticks": "100",
                "binary": "nvidia-device-plugin",
                "argv": ["nvidia-device-plugin"],
            }
        ],
    )
    probe.freeze(
        _args(run_id=RUN, pod_uid=POD, binary=["nvidia-device-plugin"], seconds=300)
    )
    # The resume deadman (systemd-run) is armed before the first SIGSTOP.
    systemd_run = next(
        i
        for i, item in enumerate(host.log)
        if item[0] == "run" and item[1][0] == "systemd-run"
    )
    first_kill = next(i for i, item in enumerate(host.log) if item[0] == "kill")
    assert systemd_run < first_kill
    assert host.kills == [(4242, probe.signal.SIGSTOP)]
    state = probe.load_state(RUN)
    assert "4242" in state["paused_processes"]
    assert state["paused_processes"]["4242"]["pod_uid"] == POD


def test_freeze_refuses_when_no_allow_listed_process(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "plugin_processes", lambda pod_uid, binaries: [])
    with pytest.raises(probe.ProbeError, match="no allow-listed plugin process"):
        probe.freeze(
            _args(run_id=RUN, pod_uid=POD, binary=["nvidia-device-plugin"], seconds=300)
        )
    # Nothing was signalled.
    assert host.kills == []


def test_resume_sigconts_and_clears_state(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probe,
        "plugin_processes",
        lambda pod_uid, binaries: [
            {
                "pid": 4242,
                "start_ticks": "100",
                "binary": "nvidia-device-plugin",
                "argv": ["x"],
            }
        ],
    )
    probe.freeze(
        _args(run_id=RUN, pod_uid=POD, binary=["nvidia-device-plugin"], seconds=300)
    )
    probe.resume(_args(run_id=RUN))
    assert (4242, probe.signal.SIGCONT) in host.kills
    assert probe.load_state(RUN)["paused_processes"] == {}
    assert probe.residue_is_clean(probe.residue(RUN)), (
        "state must be clean after resume"
    )


def test_resume_skips_a_replaced_pid(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probe,
        "plugin_processes",
        lambda pod_uid, binaries: [
            {
                "pid": 4242,
                "start_ticks": "100",
                "binary": "nvidia-device-plugin",
                "argv": ["x"],
            }
        ],
    )
    probe.freeze(
        _args(run_id=RUN, pod_uid=POD, binary=["nvidia-device-plugin"], seconds=300)
    )
    # The product deleted the Pod: the PID now has different start ticks.
    monkeypatch.setattr(probe, "process_start_ticks", lambda pid: "999")
    host.kills.clear()
    probe.cleanup(_args(run_id=RUN, kind="all"))
    assert host.kills == []  # nothing resumed; the DaemonSet's replacement is healthy
    assert probe.residue_is_clean(probe.residue(RUN)), (
        "a replaced PID leaves no residue"
    )


def test_cleanup_rejects_unknown_kind(host: FakeRun) -> None:
    with pytest.raises(probe.ProbeError, match="unsafe cleanup kind"):
        probe.cleanup(_args(run_id=RUN, kind="reboot-node"))


# --------------------------------------------------------------------------- #
# stopped-state confirmation (the live stopped:false race)
# --------------------------------------------------------------------------- #
def test_freeze_confirms_stopped_state_after_a_scheduling_race(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    monkeypatch.setattr(
        probe,
        "plugin_processes",
        lambda pod_uid, binaries: [
            {
                "pid": 4242,
                "start_ticks": "100",
                "binary": "nvidia-device-plugin",
                "argv": ["x"],
            }
        ],
    )
    # os.kill only queues SIGSTOP: the first /proc read still sees the running
    # task, the next sees it stopped. freeze must poll, not record the race.
    reads = iter([False, True])
    polled = {"n": 0}

    def racing_is_stopped(pid: int, ticks: str | None) -> bool:
        polled["n"] += 1
        return next(reads, True)

    monkeypatch.setattr(probe, "process_is_stopped", racing_is_stopped)
    monkeypatch.setattr(probe, "process_state", lambda pid, ticks: "T")
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    probe.freeze(
        _args(
            run_id=RUN,
            pod_uid=POD,
            binary=["nvidia-device-plugin"],
            pod_command_binary=[],
            seconds=300,
        )
    )
    frozen = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["frozen"]
    assert frozen[0]["stopped"] is True
    assert polled["n"] >= 2, "freeze must poll past the first non-stopped read"


def test_freeze_records_stopped_false_when_the_task_never_stops(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    monkeypatch.setattr(
        probe,
        "plugin_processes",
        lambda pod_uid, binaries: [
            {
                "pid": 4242,
                "start_ticks": "100",
                "binary": "nvidia-device-plugin",
                "argv": ["x"],
            }
        ],
    )
    monkeypatch.setattr(probe, "process_is_stopped", lambda pid, ticks: False)
    monkeypatch.setattr(probe, "process_state", lambda pid, ticks: "S")
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    monkeypatch.setattr(probe, "STOP_CONFIRM_SECONDS", 0.0)
    probe.freeze(
        _args(
            run_id=RUN,
            pod_uid=POD,
            binary=["nvidia-device-plugin"],
            pod_command_binary=[],
            seconds=300,
        )
    )
    frozen = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["frozen"]
    assert frozen[0]["stopped"] is False
    assert frozen[0]["state"] == "S"


# --------------------------------------------------------------------------- #
# refusal recording (the live EFA "no allow-listed process" with no record)
# --------------------------------------------------------------------------- #
def test_freeze_refusal_names_the_pod_processes_it_found(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "plugin_processes", lambda pod_uid, binaries: [])
    monkeypatch.setattr(
        probe,
        "pod_cgroup_processes",
        lambda pod_uid: [{"pid": 71, "binary": "efa-k8s-device-plugin-real"}],
    )
    with pytest.raises(probe.ProbeError, match="no allow-listed plugin process") as exc:
        probe.freeze(
            _args(
                run_id=RUN,
                pod_uid=POD,
                binary=["aws-efa-k8s-device-plugin"],
                pod_command_binary=[],
                seconds=300,
            )
        )
    assert exc.value.reason == "no_plugin_process"
    assert exc.value.detail is not None
    assert exc.value.detail["pod_processes"] == [
        {"pid": 71, "binary": "efa-k8s-device-plugin-real"}
    ]
    assert host.kills == []  # nothing was signalled


def test_main_emits_refusal_message_site_and_detail(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    monkeypatch.setattr(probe, "plugin_processes", lambda pod_uid, binaries: [])
    monkeypatch.setattr(
        probe,
        "pod_cgroup_processes",
        lambda pod_uid: [{"pid": 71, "binary": "efa-real"}],
    )
    rc = probe.main(
        [
            "freeze",
            "--run-id",
            RUN,
            "--pod-uid",
            POD,
            "--binary",
            "aws-efa-k8s-device-plugin",
            "--seconds",
            "300",
        ]
    )
    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["error_kind"] == "probe"
    assert payload["error_reason"] == "no_plugin_process"
    assert payload["error_site"].startswith("freeze:"), payload["error_site"]
    assert "refusing to signal" in payload["error_message"]
    assert payload["error_detail"]["pod_processes"][0]["binary"] == "efa-real"


def test_pod_command_binary_reaches_process_discovery(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_plugin_processes(pod_uid: str, binaries: list[str]):
        captured["binaries"] = list(binaries)
        return [
            {"pid": 4242, "start_ticks": "100", "binary": "efa-real", "argv": ["x"]}
        ]

    monkeypatch.setattr(probe, "plugin_processes", fake_plugin_processes)
    monkeypatch.setattr(probe, "process_is_stopped", lambda pid, ticks: True)
    monkeypatch.setattr(probe, "process_state", lambda pid, ticks: "T")
    probe.freeze(
        _args(
            run_id=RUN,
            pod_uid=POD,
            binary=["aws-efa-k8s-device-plugin"],
            pod_command_binary=["efa-real"],
            seconds=300,
        )
    )
    # The Pod-spec-derived basename is handed to discovery alongside the static
    # allow-list, so the provider-specific plugin process can be matched.
    assert "efa-real" in captured["binaries"]
    assert "aws-efa-k8s-device-plugin" in captured["binaries"]


def test_checked_pod_binaries_shape() -> None:
    assert probe.checked_pod_binaries(
        ["efa-real", "has space", "x" * 65, "ok_1.2"]
    ) == ["efa-real", "ok_1.2"]
