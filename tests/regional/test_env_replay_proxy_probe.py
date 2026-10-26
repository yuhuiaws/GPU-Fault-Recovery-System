"""Unit tests for the env-live host controller probe (no node, no systemd)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import env_replay_proxy as probe

REAL = "http://127.0.0.1:9400/metrics"
PROXY = "http://127.0.0.1:9411/metrics"
ENDPOINT = probe.ENDPOINT_ENV


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
    """Capture the single JSON line the probe emits (``error``: its message)."""

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
    payload = json.loads(captured[-1])
    return {
        "code": code,
        "json": payload,
        "error": payload.get("error_message") or payload.get("error"),
    }


def _ok() -> Any:
    return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def _recording_run(order: list[list[str]]) -> Any:
    def fake_run(cmd: list[str], **kw: Any) -> Any:
        order.append(list(cmd))
        return _ok()

    return fake_run


def _active(pid: str = "7", dropins: str = "") -> dict[str, str]:
    return {"ActiveState": "active", "MainPID": pid, "DropInPaths": dropins}


def _settled(endpoint: str | None, pid: str = "7") -> Any:
    """A ``settled_environment`` double: the exec'd process reads ``endpoint``."""

    environment = {} if endpoint is None else {ENDPOINT: endpoint}
    return lambda unit, **kw: (_active(pid), dict(environment))


def _dropins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the drop-in under ``tmp_path``; returns the drop-in path of run r1."""

    directory = tmp_path / "dropins" / f"{probe.METRICS_UNIT}.d"
    monkeypatch.setattr(probe, "dropin_dir", lambda: directory)
    monkeypatch.setattr(
        probe, "dropin_path", lambda run_id: directory / f"env-{run_id}.conf"
    )
    return directory / "env-r1.conf"


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


def test_failures_carry_the_probe_error_fields_for_the_host_fixture() -> None:
    out = _run(
        [
            "install-source",
            "--run-id",
            "bad id",
            "--name",
            probe.PLAN_NAME,
            "--sha256",
            "0" * 64,
            "--b64",
            "AA==",
        ]
    )
    assert out["code"] == 1
    payload = out["json"]
    assert payload["error_kind"] == "probe"
    assert payload["error"] == payload["error_class"] == "ProbeError"
    assert re.fullmatch(r"[0-9a-f]{16}", payload["error_code"]), payload["error_code"]
    assert payload["error_site"].startswith("safe_id:"), payload["error_site"]
    assert payload["error_message"] == "unsafe run ID"
    assert payload["command"] == "install-source"


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
    assert again["code"] == 1 and "already installed" in again["error"]
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
    assert wrong["code"] == 1 and "sha256 does not match" in wrong["error"]
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
    assert bad["code"] == 1 and "only the replay tool and plan" in bad["error"]


def test_preflight_reports_cadence_keys_leftover_units_and_the_baseline_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Live 2026-10-26: GF-ENV-LIVE-006 ran on the node 004 had just used; the
    # preflight now carries what an earlier run may have left -- its window,
    # its transient units, its drop-in, the endpoint the collector reads --
    # and the cadence keys the runner sizes its windows from.
    (tmp_path / "collector.env").write_text(
        f"{ENDPOINT}={REAL}\n"
        "GPU_FAULT_METRICS_MODE=dcgm\n"
        "GPU_FAULT_METRICS_INTERVAL_SECONDS=15\n"
        "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS=15\n"
        "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS=20\n"
        "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO=0.1\n"
        "GPU_FAULT_NODE_ACTION_KEY=never-reported\n"
    )
    listed: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        if cmd[:2] == ["systemctl", "list-units"]:
            listed.append(cmd)
            stdout = (
                "gpu-fault-env-collector-restore-239299cded976dc8.timer loaded "
                "active waiting gpu-fault-env-collector-restore-239299cded976dc8.timer\n"
                "gpu-fault-env-replay-239299cded976dc8.service loaded failed failed "
                "/opt/gpu-fault/current/venv/bin/python -I /run/.../tool.py\n"
                "other.service loaded active running Other\n"
            )
            return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()
        if cmd[:2] == ["systemctl", "show"]:
            stdout = (
                "ActiveState=active\nMainPID=7\nInvocationID=abc\n"
                "DropInPaths=/run/systemd/system/gpu-fault-metrics-collector.service.d/"
                "gpu-fault-env-replay-239299cded976dc8.conf\n"
            )
            return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()
        return _ok()

    monkeypatch.setattr(probe, "run", fake_run)
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: PROXY)
    monkeypatch.setattr(probe, "device_limits", lambda: [])
    monkeypatch.setattr(probe, "gpu_inventory", lambda: [])
    monkeypatch.setattr(probe, "current_tlimit_readings", lambda: {})
    (probe.ACCEPTANCE_ROOT / "239299cded976dc8").mkdir(parents=True)
    out = _run(["preflight"])
    assert out["code"] == 0, out
    receipt = out["json"]
    assert receipt["collector_env"] == {
        ENDPOINT: REAL,
        "GPU_FAULT_METRICS_MODE": "dcgm",
        "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
        "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS": "15",
        "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS": "20",
        "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO": "0.1",
    }, "only the allow-listed keys leave the node"
    assert receipt["baseline_endpoint"] == REAL
    assert receipt["effective_endpoint"] == PROXY
    assert receipt["open_windows"] == ["239299cded976dc8"]
    assert receipt["leftover_units"] == [
        {
            "unit": "gpu-fault-env-collector-restore-239299cded976dc8.timer",
            "load": "loaded",
            "active": "active",
            "sub": "waiting",
        },
        {
            "unit": "gpu-fault-env-replay-239299cded976dc8.service",
            "load": "loaded",
            "active": "failed",
            "sub": "failed",
        },
    ]
    assert (
        "gpu-fault-env-replay-239299cded976dc8.conf"
        in (receipt["metrics_unit"]["DropInPaths"])
    )
    assert listed and "--all" in listed[0] and listed[0][-1] == "gpu-fault-env-*"
    # ``snapshot`` lists the same leftovers (the zero-read re-verification).
    out = _run(["snapshot"])
    assert out["code"] == 0, out
    assert out["json"]["leftover_units"] == receipt["leftover_units"]
    assert out["json"]["open_windows"] == ["239299cded976dc8"]
    assert out["json"]["effective_endpoint"] == PROXY


def test_deploy_proxy_validates_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: calls.append(cmd) or _ok())
    # Bad GPU UUID.
    out = _run(["deploy-proxy", "--run-id", "r1", "--gpu-uuid", "notauuid"])
    assert out["code"] == 1 and "physical GPU UUID" in out["error"]
    # Missing installed sources.
    out = _run(["deploy-proxy", "--run-id", "r1", "--gpu-uuid", "GPU-" + "a" * 12])
    assert out["code"] == 1 and "install the replay tool" in out["error"]


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
    assert out["code"] == 1 and "came up armed" in out["error"]


# --------------------------------------------------------------------------- #
# The effective endpoint is the process's, never ``systemctl show``'s
# --------------------------------------------------------------------------- #
def test_effective_endpoint_reads_the_process_environment_not_systemctl_show(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proc = tmp_path / "proc"
    (proc / "42").mkdir(parents=True)
    (proc / "42" / "environ").write_bytes(
        b"PATH=/usr/bin\0"
        b"GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9400/metrics\0"
        b"GPU_FAULT_METRICS_MODE=dcgm\0"
        b"GPU_FAULT_CONTROL_PLANE_TOKEN=secret\0"
    )
    (proc / "42" / "comm").write_text("gpu-fault-colle\n")
    (proc / "42" / "exe").symlink_to("/opt/gpu-fault/current/venv/bin/python3.12")
    monkeypatch.setattr(probe, "PROC_ROOT", proc)
    # The live 2026-10-26 state: the drop-in ``Environment=`` shows in
    # ``systemctl show`` while the process runs with collector.env's endpoint.
    monkeypatch.setattr(
        probe,
        "unit_state",
        lambda unit: {
            "ActiveState": "active",
            "MainPID": "42",
            "Environment": f"{ENDPOINT}={PROXY}",
        },
    )
    assert probe.effective_endpoint(probe.METRICS_UNIT, seconds=0) == REAL
    assert probe.process_environment("42") == {
        ENDPOINT: REAL,
        "GPU_FAULT_METRICS_MODE": "dcgm",
    }, "allow-listed keys only"
    assert probe.has_execed("42") is True
    # systemd's pre-exec child: ``(<basename>)`` as comm, systemd's exe.
    (proc / "43").mkdir()
    (proc / "43" / "comm").write_text("(gpu-fault-coll)\n")
    (proc / "43" / "exe").symlink_to("/usr/lib/systemd/systemd")
    assert probe.has_execed("43") is False
    assert probe.process_environment("0") == {}
    assert probe.process_environment("nope") == {}
    assert probe.process_environment("44") == {}, "a vanished process reads empty"


def test_settled_environment_waits_for_the_restarted_collector_to_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active("9"))
    environments = iter([{}, {}, {ENDPOINT: PROXY}])
    monkeypatch.setattr(
        probe,
        "process_environment",
        lambda pid, keys=probe.PROCESS_KEYS: dict(
            next(environments, {ENDPOINT: PROXY})
        ),
    )
    monkeypatch.setattr(probe, "has_execed", lambda pid: False)
    slept: list[float] = []
    monkeypatch.setattr(probe.time, "sleep", slept.append)
    state, environment = probe.settled_environment(probe.METRICS_UNIT, seconds=30)
    assert state["MainPID"] == "9" and environment == {ENDPOINT: PROXY}
    assert len(slept) == 2, "the two pre-exec reads were waited out"


def test_first_read_budget_follows_the_collector_cadence_and_is_capped() -> None:
    assert probe.first_read_seconds({}) == 65, "15 spread + 20 jitter + 15 + 15"
    assert (
        probe.first_read_seconds(
            {
                probe.INTERVAL_ENV: "30",
                probe.STARTUP_SPREAD_ENV: "10",
                probe.STARTUP_JITTER_ENV: "5",
            }
        )
        == 60
    )
    assert (
        probe.first_read_seconds({probe.INTERVAL_ENV: "120"})
        == probe.MAX_FIRST_READ_SECONDS
    )
    assert (
        probe.first_read_seconds(
            {probe.INTERVAL_ENV: "bad", probe.STARTUP_JITTER_ENV: "-3"}
        )
        == 65
    ), "malformed or negative values fall back to the product defaults"
    assert probe.checked_first_read_seconds(0, {}) == 65
    assert probe.checked_first_read_seconds(30, {}) == 30
    for bad in (5, 500):
        with pytest.raises(probe.ProbeError, match="first-read seconds"):
            probe.checked_first_read_seconds(bad, {})


# --------------------------------------------------------------------------- #
# repoint-collector: EnvironmentFile= drop-in, process proof, first read
# --------------------------------------------------------------------------- #
REPOINT = [
    "repoint-collector",
    "--run-id",
    "r1",
    "--port",
    "9411",
    "--restore-seconds",
    "600",
]


def test_repoint_hands_the_override_as_an_environment_file_and_proves_the_first_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = probe.run_dir("r1")
    root.mkdir(parents=True)
    dropin = _dropins(monkeypatch, tmp_path)
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active())
    # The restarted unit is systemd's pre-exec child for one read, then the
    # collector running with the override.
    environments = iter([{}, {ENDPOINT: PROXY, probe.INTERVAL_ENV: "15"}])
    monkeypatch.setattr(
        probe,
        "process_environment",
        lambda pid, keys=probe.PROCESS_KEYS: dict(
            next(environments, {ENDPOINT: PROXY})
        ),
    )
    monkeypatch.setattr(probe, "has_execed", lambda pid: False)
    # The proxy's reads: the pass-through baseline at the repoint, then the
    # first-read polls (none yet, then the restarted collector's scrape).
    reads = iter([0, 0, 1])
    monkeypatch.setattr(
        probe,
        "proxy_status",
        lambda port, **kw: {"armed": False, "reads": next(reads, 1)},
    )
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    out = _run(REPOINT)
    assert out["code"] == 0, out
    receipt = out["json"]
    assert receipt["repointed"] is True
    assert receipt["proxy_endpoint"] == PROXY and receipt["original_endpoint"] == REAL
    assert receipt["process_environment"] == {ENDPOINT: PROXY, probe.INTERVAL_ENV: "15"}
    first = receipt["first_read"]
    assert first["observed"] is True
    assert first["reads_before"] == 0 and first["reads_after"] == 1
    assert first["budget_seconds"] == 65
    # The override travels as the drop-in's EnvironmentFile=, never Environment=.
    override = probe.override_env_path("r1")
    text = dropin.read_text()
    assert f"EnvironmentFile=-{override}" in text, text
    assert "\nEnvironment=" not in text, text
    assert override.read_text() == f"{ENDPOINT}={PROXY}\n"
    assert receipt["dropin"] == str(dropin) and receipt["override"] == str(override)
    repoint = json.loads((root / "repoint.json").read_text())
    assert repoint["original_endpoint"] == REAL and repoint["override"] == str(override)
    # The deadman is armed before the drop-in is loaded and the unit restarted.
    deadman = next(i for i, cmd in enumerate(order) if cmd[0] == "systemd-run")
    assert "restore-collector" in order[deadman]
    reload = order.index(["systemctl", "daemon-reload"])
    restart = order.index(["systemctl", "restart", probe.METRICS_UNIT])
    assert deadman < reload < restart
    assert order.count(["systemctl", "restart", probe.METRICS_UNIT]) == 1


def test_repoint_rolls_back_an_override_the_process_did_not_pick_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    probe.run_dir("r1").mkdir(parents=True)
    dropin = _dropins(monkeypatch, tmp_path)
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active())
    # What the live node did on 2026-10-26: the unit restarted, the process
    # kept collector.env's endpoint.
    monkeypatch.setattr(
        probe,
        "process_environment",
        lambda pid, keys=probe.PROCESS_KEYS: {ENDPOINT: REAL},
    )
    monkeypatch.setattr(
        probe, "proxy_status", lambda port, **kw: {"armed": False, "reads": 0}
    )
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    out = _run(REPOINT)
    assert out["code"] == 0, "a rolled-back repoint is an answer, not a probe failure"
    receipt = out["json"]
    assert receipt["repointed"] is False
    assert receipt["reason"] == probe.REASON_NOT_PICKED_UP
    assert receipt["process_environment"] == {ENDPOINT: REAL}
    assert "first_read" not in receipt, "no first read is awaited on a dead override"
    rollback = receipt["rollback"]
    assert rollback["dropin_removed"] is True and rollback["override_removed"] is True
    assert rollback["restarted"] is True and rollback["restored_endpoint"] == REAL
    assert not dropin.exists() and not probe.override_env_path("r1").exists()
    restarts = [i for i, cmd in enumerate(order) if cmd[:2] == ["systemctl", "restart"]]
    assert len(restarts) == 2, "the repoint's restart and the rollback's"
    deadman = probe.deadman_unit("r1")
    assert ["systemctl", "stop", f"{deadman}.timer"] in order[restarts[-1] :]


def test_repoint_rolls_back_when_the_collector_never_reads_the_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    probe.run_dir("r1").mkdir(parents=True)
    dropin = _dropins(monkeypatch, tmp_path)
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active())
    # The process reads the proxy after the repoint's restart and the real
    # exporter again after the rollback's.
    environments = iter([{ENDPOINT: PROXY}, {ENDPOINT: REAL}])
    monkeypatch.setattr(
        probe,
        "process_environment",
        lambda pid, keys=probe.PROCESS_KEYS: dict(next(environments, {ENDPOINT: REAL})),
    )
    monkeypatch.setattr(
        probe, "proxy_status", lambda port, **kw: {"armed": False, "reads": 4}
    )
    clock = iter(float(30 * step) for step in range(100))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    out = _run([*REPOINT, "--first-read-seconds", "45"])
    assert out["code"] == 0, out
    receipt = out["json"]
    assert receipt["repointed"] is False
    assert receipt["reason"] == probe.REASON_NO_FIRST_READ
    first = receipt["first_read"]
    assert first["observed"] is False and first["budget_seconds"] == 45
    assert first["reads_before"] == 4 and first["reads_after"] == 4
    assert receipt["rollback"]["restored_endpoint"] == REAL
    assert not dropin.exists(), "the rollback removed the drop-in"


def test_repoint_refuses_an_armed_proxy_and_a_duplicate_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    probe.run_dir("r1").mkdir(parents=True)
    dropin = _dropins(monkeypatch, tmp_path)
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active())
    monkeypatch.setattr(
        probe, "proxy_status", lambda port, **kw: {"armed": True, "reads": 0}
    )
    out = _run(REPOINT)
    assert out["code"] == 1 and "armed proxy" in out["error"]
    assert order == [] and not dropin.exists(), "nothing changed before the refusal"
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\n")
    out = _run(REPOINT)
    assert out["code"] == 1 and "already exists" in out["error"]


def test_arm_refuses_while_the_collector_does_not_read_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: list[str] = []
    monkeypatch.setattr(
        probe, "_loopback", lambda url, **kw: posted.append(url) or (200, "armed\n")
    )
    monkeypatch.setattr(probe, "proxy_status", lambda port, **kw: {"armed": True})
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: REAL)
    out = _run(["arm", "--run-id", "r1", "--port", "9411"])
    assert out["code"] == 1 and "does not read the proxy" in out["error"]
    assert posted == []
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: PROXY)
    out = _run(["arm", "--run-id", "r1", "--port", "9411"])
    assert out["code"] == 0 and out["json"]["armed"] is True
    assert posted == ["http://127.0.0.1:9411/arm"]


def test_stop_proxy_refuses_while_the_collector_still_points_at_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: PROXY)
    out = _run(["stop-proxy", "--run-id", "r1", "--port", "9411"])
    assert out["code"] == 1 and "still points at it" in out["error"]


# --------------------------------------------------------------------------- #
# restore-collector: symmetric and idempotent
# --------------------------------------------------------------------------- #
def test_restore_collector_puts_the_endpoint_back_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = probe.run_dir("r1")
    root.mkdir(parents=True)
    (root / "repoint.json").write_text(json.dumps({"original_endpoint": REAL}))
    dropin = tmp_path / "env-r1.conf"
    dropin.write_text("[Service]\n")
    override = probe.override_env_path("r1")
    override.write_text(f"{ENDPOINT}={PROXY}\n")
    monkeypatch.setattr(probe, "dropin_path", lambda run_id: dropin)
    order: list[list[str]] = []
    monkeypatch.setattr(probe, "run", _recording_run(order))
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active("1"))
    monkeypatch.setattr(probe, "settled_environment", _settled(REAL, "1"))
    out = _run(["restore-collector", "--run-id", "r1"])
    assert out["code"] == 0, out
    receipt = out["json"]
    assert receipt["restored_endpoint"] == REAL
    assert receipt["dropin_removed"] is True and receipt["override_removed"] is True
    assert receipt["restarted"] is True
    assert receipt["process_environment"] == {ENDPOINT: REAL}
    assert not dropin.exists() and not override.exists()
    restart = ["systemctl", "restart", probe.METRICS_UNIT]
    assert order.index(["systemctl", "daemon-reload"]) < order.index(restart)
    # A second restore finds nothing to undo and restarts nothing.
    order.clear()
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: REAL)
    again = _run(["restore-collector", "--run-id", "r1"])
    assert again["code"] == 0, again
    assert again["json"]["restarted"] is False
    assert again["json"]["dropin_removed"] is False
    assert again["json"]["restored_endpoint"] == REAL
    assert restart not in order
    # A process still reading the proxy with no drop-in left is restarted.
    order.clear()
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: PROXY)
    third = _run(["restore-collector", "--run-id", "r1"])
    assert third["code"] == 0 and third["json"]["restarted"] is True
    assert restart in order


def test_restore_collector_refuses_a_foreign_endpoint_or_a_loaded_dropin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = probe.run_dir("r1")
    root.mkdir(parents=True)
    (root / "repoint.json").write_text(json.dumps({"original_endpoint": REAL}))
    dropin = tmp_path / "env-r1.conf"
    dropin.write_text("[Service]\n")
    monkeypatch.setattr(probe, "dropin_path", lambda run_id: dropin)
    monkeypatch.setattr(probe, "run", lambda cmd, **kw: _ok())
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active("1"))
    monkeypatch.setattr(probe, "settled_environment", _settled(PROXY, "1"))
    out = _run(["restore-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "is not the original" in out["error"]
    # systemd still listing the removed drop-in means the reload did not land.
    dropin.write_text("[Service]\n")
    monkeypatch.setattr(probe, "settled_environment", _settled(REAL, "1"))
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active("1", str(dropin)))
    out = _run(["restore-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "still lists the repoint drop-in" in out["error"]


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
    assert out["code"] == 1 and "repoint the collector" in out["error"]
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
    assert again["code"] == 1 and "already scheduled" in again["error"]


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
    assert out["code"] == 1 and "already consumed" in out["error"]
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
        assert out["code"] == 1 and "pause seconds" in out["error"]


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
    assert out["code"] == 1 and "did not reach the pause read" in out["error"]
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
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active("1"))
    monkeypatch.setattr(probe, "unit_is_stopped", lambda unit: True)
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: REAL)
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
    dropin = root / "present.conf"
    dropin.write_text("[Service]\n")
    monkeypatch.setattr(probe, "dropin_path", lambda run_id: dropin)
    monkeypatch.setattr(probe, "settled_environment", _settled(REAL, "1"))
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
    monkeypatch.setattr(probe, "unit_state", lambda unit: _active("1"))
    monkeypatch.setattr(probe, "effective_endpoint", lambda unit, **kw: PROXY)
    monkeypatch.setattr(probe, "settled_environment", _settled(PROXY, "1"))
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "drop-in" in out["error"]
    dropin.write_text("[Service]\n")
    monkeypatch.setattr(probe, "unit_is_stopped", lambda unit: True)
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "resume the paused collector" in out["error"]
    monkeypatch.setattr(probe, "unit_is_stopped", lambda unit: False)
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 0
    assert out["json"]["effective_endpoint"] == PROXY
    assert out["json"]["process_environment"] == {ENDPOINT: PROXY}
    assert out["json"]["restarted_at"]
    # An endpoint that changed across the restart is refused.
    monkeypatch.setattr(probe, "settled_environment", _settled(REAL, "1"))
    out = _run(["restart-collector", "--run-id", "r1"])
    assert out["code"] == 1 and "changed across the restart" in out["error"]
