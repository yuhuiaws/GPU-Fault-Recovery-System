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


# --------------------------------------------------------------------------- #
# product generation (``feature``) and the host collector's probe journal
# --------------------------------------------------------------------------- #
EXEC_START = (
    "{ path=/opt/gpu-fault/current/venv/bin/gpu-fault-collector ; "
    "argv[]=/opt/gpu-fault/current/venv/bin/gpu-fault-collector host ; "
    "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; "
    "code=(null) ; status=0/0 }"
)
TIMEOUT_LINE = (
    "2026-10-27 10:00:05,123 WARNING gpu_fault.collectors.host.device_plugin device "
    "plugin nvidia-gpu.sock did not answer GetDevicePluginOptions within 5s "
    "(1 consecutive; finding after 2): nvidia-gpu.sock: no bytes from the plugin "
    "within 5s"
)


def _journal(*messages: tuple[int, str]) -> str:
    import json

    return "\n".join(
        json.dumps({"__REALTIME_TIMESTAMP": str(stamp), "MESSAGE": message})
        for stamp, message in messages
    )


def _venv(tmp_path: Path, *, modules: bool) -> Path:
    venv = tmp_path / "venv"
    site = venv / "lib" / "python3.12" / "site-packages" / "gpu_fault"
    (site / "collectors" / "host").mkdir(parents=True)
    (site / "collectors" / "host" / "collector.py").write_text("", encoding="utf-8")
    if modules:
        for relative in probe.FEATURE_MODULES.values():
            path = venv / "lib" / "python3.12" / "site-packages" / relative
            path.write_text("", encoding="utf-8")
    return venv


def test_exec_start_path_reads_the_unit_executable() -> None:
    assert (
        probe.exec_start_path(EXEC_START)
        == "/opt/gpu-fault/current/venv/bin/gpu-fault-collector"
    )
    assert probe.exec_start_path("") is None


def test_installed_modules_reports_each_probe_module(tmp_path: Path) -> None:
    assert probe.installed_modules(_venv(tmp_path, modules=True)) == {
        name: True for name in probe.FEATURE_MODULES
    }
    assert probe.installed_modules(_venv(tmp_path / "old", modules=False)) == {
        name: False for name in probe.FEATURE_MODULES
    }


def _feature_run(venv: Path, environment: str = "") -> Any:
    def fake_run(command: list[str], *, check: bool = True, timeout: int = 120):
        import subprocess

        assert command[:2] == ["systemctl", "show"], command
        exec_start = EXEC_START.replace("/opt/gpu-fault/current/venv", str(venv))
        stdout = (
            f"ExecStart={exec_start}\nEnvironment={environment}\nActiveState=active\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake_run


def test_feature_names_the_liveness_generation_from_the_unit_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    venv = _venv(tmp_path, modules=True)
    monkeypatch.setattr(
        probe, "run", _feature_run(venv, "GPU_FAULT_HOST_INTERVAL_SECONDS=15")
    )
    monkeypatch.setattr(probe, "AGENT_CURRENT", tmp_path / "no-such-link")
    env_file = tmp_path / "collector.env"
    env_file.write_text(
        "GPU_FAULT_FILESYSTEMS=/\nGPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR=''\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "COLLECTOR_ENV", env_file)
    probe.feature(_args(run_id=RUN))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["liveness_probe"] is True
    assert payload["venv"] == str(venv)
    assert payload["exec_start_path"] == f"{venv}/bin/gpu-fault-collector"
    assert payload["device_plugin_env"] == {"GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR": ""}
    assert payload["active_state"] == "active"


def test_feature_reports_the_earlier_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    venv = _venv(tmp_path, modules=False)
    monkeypatch.setattr(probe, "run", _feature_run(venv))
    monkeypatch.setattr(probe, "AGENT_CURRENT", tmp_path / "no-such-link")
    monkeypatch.setattr(probe, "COLLECTOR_ENV", tmp_path / "absent.env")
    probe.feature(_args(run_id=RUN))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["liveness_probe"] is False
    assert payload["modules"] == {name: False for name in probe.FEATURE_MODULES}
    assert payload["device_plugin_env"] == {}


def test_collector_journal_parses_timeout_and_absent_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def fake_run(command: list[str], *, check: bool = True, timeout: int = 120):
        assert command[:3] == ["journalctl", "-u", probe.HOST_COLLECTOR_UNIT], command
        assert command[3:5] == ["--since", "@1700000000.000000"], command
        stdout = _journal(
            (1700000001000000, "host collector started"),
            (1700000005123000, TIMEOUT_LINE),
            (
                1700000006000000,
                "INFO gpu_fault.collectors.host.device_plugin device plugin socket "
                "directory /var/lib/kubelet/device-plugins is absent; the device "
                "plugin liveness probe is not running on this node",
            ),
        )
        return subprocess.CompletedProcess(command, 0, stdout + "\nnot json", "")

    monkeypatch.setattr(probe, "run", fake_run)
    entries = probe.collector_journal(1700000000.0)
    assert [item["kind"] for item in entries] == ["timeout", "directory_absent"]
    first = entries[0]
    assert first["sockets"] == ["nvidia-gpu.sock"]
    assert first["consecutive"] == 1 and first["required"] == 2
    assert first["timeout_seconds"] == 5.0
    assert first["realtime_epoch"] == pytest.approx(1700000005.123)
    assert entries[1]["directory"] == "/var/lib/kubelet/device-plugins"


def test_await_first_timeout_returns_on_the_first_logged_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = iter(
        [
            [],
            [
                {
                    "kind": "timeout",
                    "consecutive": 1,
                    "required": 2,
                    "realtime_epoch": 1700000005.1,
                    "sockets": ["nvidia-gpu.sock"],
                }
            ],
        ]
    )
    monkeypatch.setattr(probe, "collector_journal", lambda since: next(reads, []))
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    watch = probe.await_first_timeout(1700000000.0, 30)
    assert watch["first_timeout"]["consecutive"] == 1
    assert watch["second_timeout_seen"] is False


def test_await_first_timeout_gives_up_at_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "collector_journal", lambda since: [])
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    clock = iter(float(index) for index in range(0, 1000))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    watch = probe.await_first_timeout(1700000000.0, 5)
    assert watch["first_timeout"] is None
    assert watch["second_timeout_seen"] is False


def _frozen_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(probe, "process_state", lambda pid, ticks: "T")
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)


def test_freeze_resumes_the_instant_the_first_timeout_is_logged(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    _frozen_plugin(monkeypatch)
    entry = {
        "kind": "timeout",
        "consecutive": 1,
        "required": 2,
        "realtime_epoch": 1700000005.1,
        "sockets": ["nvidia-gpu.sock"],
        "message": TIMEOUT_LINE,
    }
    reads = iter([[], [entry]])
    monkeypatch.setattr(probe, "collector_journal", lambda since: next(reads, [entry]))
    probe.freeze(
        _args(
            run_id=RUN,
            pod_uid=POD,
            binary=["nvidia-device-plugin"],
            pod_command_binary=[],
            seconds=300,
            resume_after_first_timeout=True,
            window_seconds=40,
        )
    )
    assert host.kills == [(4242, probe.signal.SIGSTOP), (4242, probe.signal.SIGCONT)]
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["first_timeout"]["consecutive"] == 1
    assert payload["second_timeout_seen"] is False
    assert payload["window_seconds"] == 40
    assert payload["resumed"][0]["resumed"] is True
    assert payload["frozen_since_epoch"] > 0
    assert probe.residue_is_clean(probe.residue(RUN)), "nothing stays paused"
    assert not host.timer_active, "the resume deadman is disarmed with the resume"


def test_freeze_resumes_at_the_window_without_a_logged_tick(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    _frozen_plugin(monkeypatch)
    monkeypatch.setattr(probe, "collector_journal", lambda since: [])
    clock = iter(float(index) for index in range(0, 100000))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    probe.freeze(
        _args(
            run_id=RUN,
            pod_uid=POD,
            binary=["nvidia-device-plugin"],
            pod_command_binary=[],
            seconds=300,
            resume_after_first_timeout=True,
            window_seconds=5,
        )
    )
    assert (4242, probe.signal.SIGCONT) in host.kills
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["first_timeout"] is None
    assert payload["resume_latency_seconds"] is None
    assert probe.residue_is_clean(probe.residue(RUN)), "the window always resumes"


def test_freeze_window_is_bounded(
    host: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _frozen_plugin(monkeypatch)
    with pytest.raises(probe.ProbeError, match="window seconds"):
        probe.freeze(
            _args(
                run_id=RUN,
                pod_uid=POD,
                binary=["nvidia-device-plugin"],
                pod_command_binary=[],
                seconds=300,
                resume_after_first_timeout=True,
                window_seconds=0,
            )
        )
    assert host.kills == [], "a refused window signals nothing"


def test_probe_log_requires_a_positive_since(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "collector_journal", lambda since: [])
    with pytest.raises(probe.ProbeError, match="since-epoch"):
        probe.probe_log(_args(run_id=RUN, since_epoch=0.0))


def test_probe_log_emits_timeouts_and_absence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    monkeypatch.setattr(
        probe,
        "collector_journal",
        lambda since: [
            {"kind": "timeout", "consecutive": 1, "required": 2, "realtime_epoch": 1.0},
            {"kind": "directory_absent", "directory": "/x", "realtime_epoch": 2.0},
        ],
    )
    probe.probe_log(_args(run_id=RUN, since_epoch=1700000000.0))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(payload["timeouts"]) == 1
    assert payload["directory_absent"] is True
    assert payload["since_epoch"] == 1700000000.0


def test_parser_knows_the_read_only_companions() -> None:
    parser = probe.parser()
    feature = parser.parse_args(["feature", "--run-id", RUN])
    assert feature.handler is probe.feature
    log = parser.parse_args(["probe-log", "--run-id", RUN, "--since-epoch", "17.5"])
    assert log.handler is probe.probe_log and log.since_epoch == 17.5
    frozen = parser.parse_args(
        [
            "freeze",
            "--run-id",
            RUN,
            "--pod-uid",
            POD,
            "--binary",
            "nvidia-device-plugin",
            "--seconds",
            "300",
            "--resume-after-first-timeout",
            "--window-seconds",
            "40",
        ]
    )
    assert frozen.resume_after_first_timeout is True and frozen.window_seconds == 40


# --------------------------------------------------------------------------- #
# the live HyperPod EFA plugin Pod (hp lane, 2026-10-27, run
# gl014-efa-plugin-a6-471be317-r2): what probe-error.json recorded
# --------------------------------------------------------------------------- #
LIVE_EFA_POD = "a9d8615c-bc52-4379-b74e-d10a64738556"
LIVE_EFA_POD_PROCESSES = [
    {"pid": 51880, "binary": "pause"},
    {"pid": 53504, "binary": "efa-k8s-device-plugin"},
]
#: ``requested_binaries`` of that refusal: the fixture's tuple intersected with
#: the then-static allow-list, plus the Pod's command/image basename.
LIVE_EFA_REQUESTED = ["aws-efa-k8s-device-plugin", "device-plugin", "k8s-device-plugin"]


def _fake_proc(root: Path, pid: int, argv: list[str], cgroup: str) -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (directory / "cgroup").write_text(cgroup + "\n")
    comm = Path(argv[0]).name[:15]
    fields = " ".join(str(i) for i in range(50))
    (directory / "stat").write_text(f"{pid} ({comm}) S {fields}\n")


def _pod_scope(pod_uid: str, container: str) -> str:
    slice_uid = pod_uid.replace("-", "_")
    return (
        "0::/kubepods.slice/kubepods-besteffort.slice/"
        f"kubepods-besteffort-pod{slice_uid}.slice/cri-containerd-{container}.scope"
    )


def test_live_hyperpod_efa_pod_processes_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    _fake_proc(proc, 51880, ["/pause"], _pod_scope(LIVE_EFA_POD, "aa" * 32))
    _fake_proc(
        proc, 53504, ["/efa-k8s-device-plugin"], _pod_scope(LIVE_EFA_POD, "bb" * 32)
    )
    # Noise the gates must exclude: the identical binary in ANOTHER Pod, and a
    # host process -- neither cgroup names the discovered Pod UID.
    other_pod = "ffffffff-0000-0000-0000-000000000000"
    _fake_proc(proc, 60000, ["/efa-k8s-device-plugin"], _pod_scope(other_pod, "cc"))
    _fake_proc(proc, 1, ["/usr/lib/systemd/systemd"], "0::/init.scope")
    (proc / "self").mkdir()  # a non-numeric entry every /proc has
    monkeypatch.setattr(probe, "PROC_ROOT", proc)

    # What the refusal recorded live: both Pod processes, basenames only.
    assert probe.pod_cgroup_processes(LIVE_EFA_POD) == LIVE_EFA_POD_PROCESSES
    # The request made live matched nothing -- the basename the node runs was
    # off the list -- which is exactly why the probe refused.
    assert probe.plugin_processes(LIVE_EFA_POD, LIVE_EFA_REQUESTED) == []
    # With the basename allow-listed, exactly the plugin process is matched:
    # pause is not allow-listed, and the other Pod's identical binary and the
    # host process fail the pod-UID cgroup gate.
    wanted = probe.checked_binaries([*LIVE_EFA_REQUESTED, "efa-k8s-device-plugin"])
    assert "efa-k8s-device-plugin" in wanted, wanted
    matched = probe.plugin_processes(LIVE_EFA_POD, wanted)
    assert [(item["pid"], item["binary"]) for item in matched] == [
        (53504, "efa-k8s-device-plugin")
    ]
    assert matched[0]["start_ticks"] == "18"  # field 22 of /proc/<pid>/stat


def test_allow_list_carries_the_hyperpod_efa_plugin_basename() -> None:
    assert "efa-k8s-device-plugin" in probe.PLUGIN_BINARY_ALLOW_LIST
    assert "pause" not in probe.PLUGIN_BINARY_ALLOW_LIST
    assert probe.checked_binaries(["pause", "efa-k8s-device-plugin"]) == [
        "efa-k8s-device-plugin"
    ]
