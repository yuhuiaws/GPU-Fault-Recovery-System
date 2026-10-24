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
