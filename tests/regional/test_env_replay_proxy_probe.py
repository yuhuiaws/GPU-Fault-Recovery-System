"""Unit tests for the env-live host controller probe (no node, no systemd)."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import env_replay_proxy as probe


@pytest.fixture(autouse=True)
def _on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "ACCEPTANCE_ROOT", tmp_path / "run")
    monkeypatch.setattr(probe, "COLLECTOR_ENV", tmp_path / "collector.env")
    monkeypatch.setattr(probe, "VENV_PYTHON", tmp_path / "venv-python")
    (tmp_path / "venv-python").write_text("#!/bin/sh\n")
    (tmp_path / "collector.env").write_text(
        "GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9400/metrics\n"
        "GPU_FAULT_METRICS_MODE=dcgm\n"
    )


def _run(argv: list[str]) -> dict[str, Any]:
    """Capture the single JSON line the probe emits."""

    captured: list[str] = []
    import builtins

    original = builtins.print

    def spy(*args: Any, **kwargs: Any) -> None:
        captured.append(" ".join(str(a) for a in args))

    builtins.print = spy
    try:
        code = probe.main(argv)
    finally:
        builtins.print = original
    return {"code": code, "json": json.loads(captured[-1])}


def test_safe_id_and_digest_are_stable_and_checked() -> None:
    assert probe.safe_id("c019-a1", "x") == "c019-a1"
    with pytest.raises(probe.ProbeError):
        probe.safe_id("bad id with spaces", "x")
    assert probe.digest("run") == probe.digest("run")
    assert len(probe.digest("run")) == 16


def test_port_and_restore_bounds() -> None:
    assert probe.checked_port(9411) == 9411
    for bad in (0, 80, 70000):
        with pytest.raises(probe.ProbeError):
            probe.checked_port(bad)
    assert probe.checked_restore_seconds(600) == 600
    for bad in (10, 5000):
        with pytest.raises(probe.ProbeError):
            probe.checked_restore_seconds(bad)


def test_parse_env_and_baseline_endpoint() -> None:
    assert probe.baseline_endpoint() == "http://127.0.0.1:9400/metrics"


def test_install_source_verifies_sha_and_refuses_foreign_names() -> None:
    content = b"print('hi')\n"
    sha = hashlib.sha256(content).hexdigest()
    b64 = base64.b64encode(content).decode()
    out = _run(
        [
            "install-source",
            "--run-id",
            "r1",
            "--name",
            probe.REPLAY_TOOL_NAME,
            "--sha256",
            sha,
            "--b64",
            b64,
        ]
    )
    assert out["code"] == 0 and out["json"]["sha256"] == sha
    # A second write of the same file is refused.
    again = _run(
        [
            "install-source",
            "--run-id",
            "r1",
            "--name",
            probe.REPLAY_TOOL_NAME,
            "--sha256",
            sha,
            "--b64",
            b64,
        ]
    )
    assert again["code"] == 1 and "already installed" in again["json"]["error"]
    # A wrong sha is refused.
    wrong = _run(
        [
            "install-source",
            "--run-id",
            "r2",
            "--name",
            probe.PLAN_NAME,
            "--sha256",
            "0" * 64,
            "--b64",
            b64,
        ]
    )
    assert wrong["code"] == 1 and "sha256 does not match" in wrong["json"]["error"]
    # A foreign file name is refused.
    bad = _run(
        [
            "install-source",
            "--run-id",
            "r3",
            "--name",
            "evil.sh",
            "--sha256",
            sha,
            "--b64",
            b64,
        ]
    )
    assert bad["code"] == 1 and "only the replay tool and plan" in bad["json"]["error"]


def test_deploy_proxy_validates_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: calls.append(cmd) or _ok())
    # Bad GPU UUID.
    out = _run(["deploy-proxy", "--run-id", "r1", "--gpu-uuid", "notauuid"])
    assert out["code"] == 1 and "physical GPU UUID" in out["json"]["error"]
    # Missing installed sources.
    out = _run(["deploy-proxy", "--run-id", "r1", "--gpu-uuid", "GPU-" + "a" * 12])
    assert out["code"] == 1 and "install the replay tool" in out["json"]["error"]


def _ok() -> Any:
    return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def test_deploy_proxy_launches_and_waits_for_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = probe.run_dir("r1")
    root.mkdir(parents=True)
    (root / probe.REPLAY_TOOL_NAME).write_text("print('tool')\n")
    (root / probe.PLAN_NAME).write_text('{"steps": [{"reads": 1, "values": {}}]}')
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: _ok())
    monkeypatch.setattr(
        probe, "unit_state", lambda unit: {"ActiveState": "active", "MainPID": "42"}
    )
    monkeypatch.setattr(
        probe, "proxy_status", lambda port, **kw: {"armed": False, "reads": 0}
    )
    out = _run(["deploy-proxy", "--run-id", "r1", "--gpu-uuid", "GPU-" + "a" * 12])
    assert out["code"] == 0
    assert out["json"]["endpoint"] == "http://127.0.0.1:9411/metrics"
    assert out["json"]["status"]["armed"] is False


def test_deploy_proxy_refuses_an_armed_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    root = probe.run_dir("r1")
    root.mkdir(parents=True)
    (root / probe.REPLAY_TOOL_NAME).write_text("x")
    (root / probe.PLAN_NAME).write_text("{}")
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: _ok())
    monkeypatch.setattr(probe, "unit_state", lambda unit: {"ActiveState": "active"})
    monkeypatch.setattr(probe, "proxy_status", lambda port, **kw: {"armed": True})
    out = _run(["deploy-proxy", "--run-id", "r1", "--gpu-uuid", "GPU-" + "a" * 12])
    assert out["code"] == 1 and "came up armed" in out["json"]["error"]


def test_repoint_arms_the_deadman_before_the_dropin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = probe.run_dir("r1")
    root.mkdir(parents=True)
    dropin_root = tmp_path / "dropins"
    monkeypatch.setattr(
        probe, "dropin_dir", lambda: dropin_root / f"{probe.METRICS_UNIT}.d"
    )
    monkeypatch.setattr(
        probe,
        "dropin_path",
        lambda run_id: dropin_root / f"{probe.METRICS_UNIT}.d" / f"env-{run_id}.conf",
    )
    order: list[str] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        order.append(" ".join(cmd[:3]))
        return _ok()

    monkeypatch.setattr(probe, "run", fake_run)
    monkeypatch.setattr(probe, "unit_state", lambda unit: {"ActiveState": "active"})
    monkeypatch.setattr(
        probe, "effective_endpoint", lambda unit: "http://127.0.0.1:9411/metrics"
    )
    out = _run(
        [
            "repoint-collector",
            "--run-id",
            "r1",
            "--port",
            "9411",
            "--restore-seconds",
            "600",
        ]
    )
    assert out["code"] == 0
    assert out["json"]["proxy_endpoint"] == "http://127.0.0.1:9411/metrics"
    # systemd-run (the deadman) ran before the daemon-reload / restart.
    assert any("systemd-run" in step for step in order), order
    assert order.index("systemctl daemon-reload") > next(
        index for index, step in enumerate(order) if "systemd-run" in step
    )


def test_stop_proxy_refuses_while_the_collector_still_points_at_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe, "effective_endpoint", lambda unit: "http://127.0.0.1:9411/metrics"
    )
    out = _run(["stop-proxy", "--run-id", "r1", "--port", "9411"])
    assert out["code"] == 1 and "still points at it" in out["json"]["error"]


def test_restore_collector_puts_the_endpoint_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = probe.run_dir("r1")
    root.mkdir(parents=True)
    (root / "repoint.json").write_text(
        json.dumps({"original_endpoint": "http://127.0.0.1:9400/metrics"})
    )
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: _ok())
    monkeypatch.setattr(
        probe, "unit_state", lambda unit: {"ActiveState": "active", "MainPID": "1"}
    )
    monkeypatch.setattr(
        probe, "effective_endpoint", lambda unit: "http://127.0.0.1:9400/metrics"
    )
    monkeypatch.setattr(probe, "dropin_path", lambda run_id: root / "missing.conf")
    out = _run(["restore-collector", "--run-id", "r1"])
    assert out["code"] == 0
    assert out["json"]["restored_endpoint"] == "http://127.0.0.1:9400/metrics"


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-009: pause-collector / pause-wait / pause-status / resume / restart
# --------------------------------------------------------------------------- #
def _repointed_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "probe.py").write_text("# probe copy\n")
    (root / "repoint.json").write_text(
        json.dumps(
            {
                "original_endpoint": "http://127.0.0.1:9400/metrics",
                "proxy_endpoint": "http://127.0.0.1:9411/metrics",
            }
        )
    )


def _recording_run(order: list[list[str]]) -> Any:
    def fake_run(cmd: list[str], **kw: Any) -> Any:
        order.append(list(cmd))
        return _ok()

    return fake_run


def test_pause_collector_schedules_a_waiter_unit_before_the_pause_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = probe.run_dir("r1")
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(probe, "unit_state", lambda unit: {"ActiveState": "active"})
    monkeypatch.setattr(probe, "proxy_status", lambda port, **kw: {"reads": 2})
    # No repoint yet: no probe copy to run the waiter from.
    out = _run(
        ["pause-collector", "--run-id", "r1", "--after-reads", "7", "--seconds", "195"]
    )
    assert out["code"] == 1 and "repoint the collector" in out["json"]["error"]
    _repointed_run(root)
    out = _run(
        ["pause-collector", "--run-id", "r1", "--after-reads", "7", "--seconds", "195"]
    )
    assert out["code"] == 0, out
    assert out["json"]["target_reads"] == 7 and out["json"]["reads_at_schedule"] == 2
    assert out["json"]["waiter"]["ActiveState"] == "active"
    waiter = next(cmd for cmd in order if cmd[0] == "systemd-run")
    assert "pause-wait" in waiter and waiter[waiter.index("--after-reads") + 1] == "7"
    assert any(item.startswith("--property=RuntimeMaxSec=") for item in waiter), waiter
    record = json.loads((root / probe.PAUSE_FILE).read_text())
    assert record["target_reads"] == 7 and record["pause_seconds"] == 195
    # A second schedule for the same run is refused.
    again = _run(
        ["pause-collector", "--run-id", "r1", "--after-reads", "7", "--seconds", "195"]
    )
    assert again["code"] == 1 and "already scheduled" in again["json"]["error"]


def test_pause_collector_refuses_a_pause_read_already_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repointed_run(probe.run_dir("r1"))
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: _ok())
    monkeypatch.setattr(probe, "unit_state", lambda unit: {"ActiveState": "active"})
    monkeypatch.setattr(probe, "proxy_status", lambda port, **kw: {"reads": 7})
    out = _run(
        ["pause-collector", "--run-id", "r1", "--after-reads", "7", "--seconds", "195"]
    )
    assert out["code"] == 1 and "already consumed" in out["json"]["error"]
    for bad in ("30", "5000"):
        out = _run(
            [
                "pause-collector",
                "--run-id",
                "r1",
                "--after-reads",
                "7",
                "--seconds",
                bad,
            ]
        )
        assert out["code"] == 1 and "pause seconds" in out["json"]["error"]


def test_pause_wait_arms_the_resume_deadman_then_stops_the_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = probe.run_dir("r1")
    _repointed_run(root)
    probe.write_pause("r1", {"run_id": "r1", "target_reads": 4})
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(
        probe, "unit_state", lambda unit: {"ActiveState": "active", "MainPID": "0"}
    )
    reads = iter([2, 3, 4])
    monkeypatch.setattr(
        probe, "proxy_status", lambda port, **kw: {"reads": next(reads)}
    )
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    out = _run(
        [
            "pause-wait",
            "--run-id",
            "r1",
            "--after-reads",
            "4",
            "--seconds",
            "195",
            "--wait-seconds",
            "60",
        ]
    )
    assert out["code"] == 0, out
    assert out["json"]["reads_at_pause"] == 4 and out["json"]["missed"] is False
    deadman = next(i for i, cmd in enumerate(order) if cmd[0] == "systemd-run")
    stop = next(i for i, cmd in enumerate(order) if "--signal=SIGSTOP" in cmd)
    assert deadman < stop, "the resume deadman is armed before the SIGSTOP"
    assert "resume-collector" in order[deadman]
    assert any(item.startswith("--on-active=195s") for item in order[deadman]), order[
        deadman
    ]
    assert probe.METRICS_UNIT in order[stop]
    record = json.loads((root / probe.PAUSE_FILE).read_text())
    assert record["paused_at"] and record["target_reads"] == 4


def test_pause_wait_times_out_without_pausing(monkeypatch: pytest.MonkeyPatch) -> None:
    root = probe.run_dir("r1")
    _repointed_run(root)
    probe.write_pause("r1", {"run_id": "r1", "target_reads": 4})
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(probe, "proxy_status", lambda port, **kw: {"reads": 1})
    clock = iter([0.0, 0.0, 0.5, 2.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    out = _run(
        [
            "pause-wait",
            "--run-id",
            "r1",
            "--after-reads",
            "4",
            "--seconds",
            "195",
            "--wait-seconds",
            "1",
        ]
    )
    assert out["code"] == 1 and "did not reach the pause read" in out["json"]["error"]
    assert not any("--signal=SIGSTOP" in cmd for cmd in order), order
    record = json.loads((root / probe.PAUSE_FILE).read_text())
    assert record["timed_out_at"] and record["reads_at_timeout"] == 1


def test_pause_status_resume_and_restore_resume_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = probe.run_dir("r1")
    _repointed_run(root)
    probe.write_pause(
        "r1", {"run_id": "r1", "target_reads": 4, "paused_at": "2031-01-01T00:00:00Z"}
    )
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(
        probe, "unit_state", lambda unit: {"ActiveState": "active", "MainPID": "1"}
    )
    monkeypatch.setattr(probe, "unit_is_stopped", lambda unit: True)
    monkeypatch.setattr(
        probe, "effective_endpoint", lambda unit: "http://127.0.0.1:9400/metrics"
    )
    status = _run(["pause-status", "--run-id", "r1"])
    assert status["code"] == 0 and status["json"]["stopped"] is True
    assert status["json"]["pause"]["target_reads"] == 4
    resumed = _run(["resume-collector", "--run-id", "r1"])
    assert resumed["code"] == 0 and resumed["json"]["pause"]["resumed_at"]
    assert any("--signal=SIGCONT" in cmd for cmd in order), order
    assert any(
        cmd[:2] == ["systemctl", "stop"] and cmd[2].endswith(".timer") for cmd in order
    ), order
    # The restore of a still-paused run resumes before it restarts the unit.
    order.clear()
    probe.write_pause(
        "r1", {"run_id": "r1", "target_reads": 4, "paused_at": "2031-01-01T00:00:00Z"}
    )
    monkeypatch.setattr(probe, "dropin_path", lambda run_id: root / "missing.conf")
    out = _run(["restore-collector", "--run-id", "r1"])
    assert out["code"] == 0 and out["json"]["resumed_pause"]["pause"]["resumed_at"]
    cont = next(i for i, cmd in enumerate(order) if "--signal=SIGCONT" in cmd)
    restart = next(
        i for i, cmd in enumerate(order) if cmd[:2] == ["systemctl", "restart"]
    )
    assert cont < restart


def test_restart_collector_needs_the_dropin_and_a_running_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = probe.run_dir("r1")
    _repointed_run(root)
    dropin = tmp_path / "env-r1.conf"
    monkeypatch.setattr(probe, "dropin_path", lambda run_id: dropin)
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: _ok())
    monkeypatch.setattr(
        probe, "unit_state", lambda unit: {"ActiveState": "active", "MainPID": "1"}
    )
    monkeypatch.setattr(
        probe, "effective_endpoint", lambda unit: "http://127.0.0.1:9411/metrics"
    )
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "drop-in" in out["json"]["error"]
    dropin.write_text("[Service]\n")
    monkeypatch.setattr(probe, "unit_is_stopped", lambda unit: True)
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "resume the paused collector" in out["json"]["error"]
    monkeypatch.setattr(probe, "unit_is_stopped", lambda unit: False)
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 0
    assert out["json"]["effective_endpoint"] == "http://127.0.0.1:9411/metrics"
    assert out["json"]["restarted_at"]
    # An endpoint that changed across the restart is refused.
    endpoints = iter(["http://127.0.0.1:9411/metrics", "http://127.0.0.1:9400/metrics"])
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit: next(endpoints))
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "changed across the restart" in out["json"]["error"]
