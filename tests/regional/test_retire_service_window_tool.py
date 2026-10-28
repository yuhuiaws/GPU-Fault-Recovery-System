"""``retire_service_window.py``: plan-only by default, confirmed execute.

The standalone form of the 030 pre-step, for journals no pending 030 run will
reach (the HyperPod lanes' 2026-10-24/25 journals for one spare).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import retire_service_window as tool
from scripts.e2e.regional import service_window_retire as sw
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_service_window import abandoned_window_journal

NODE = "hyperpod-i-00000000000000001"
NODE_UID = "11111111-2222-3333-4444-555555555555"


class Regional:
    def __init__(self, uid: str = NODE_UID) -> None:
        self.uid = uid

    def node_snapshot(self, node: str) -> dict[str, Any]:
        assert node == NODE
        return {"name": node, "uid": self.uid}


@pytest.fixture
def journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    kube = tmp_path / "gpu.kubeconfig"
    kube.write_text("apiVersion: v1\n", encoding="utf-8")
    monkeypatch.setattr(tool, "install_site_profile", lambda *a, **k: None)
    monkeypatch.setattr(tool, "install_abort_signals", lambda *a, **k: None)
    monkeypatch.setattr(
        tool,
        "settings_from_arguments",
        lambda _a: SimpleNamespace(
            gpu_kubeconfig=kube, gpu_context="ctx", namespace="gpu-fault-system"
        ),
    )
    monkeypatch.setattr(tool, "RegionalLiveFixture", lambda _settings: Regional())
    reads: list[list[str]] = []

    def kubectl(command: list[str], **_kwargs: Any) -> Any:
        reads.append(command)
        assert command[7] == "get", "plan mode only reads"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sw, "run_fixture_command", kubectl)
    path, _window = abandoned_window_journal(
        tmp_path / "run" / "cases" / tool.DEFAULT_CASE / tool.DEFAULT_STATE,
        node=NODE,
        node_uid=NODE_UID,
        run_id="gl030-not-ready-a1-471be317",
        release_id="bf1bb0037847",
    )
    return path


def _argv(tmp_path: Path, *mode: str) -> list[str]:
    return [
        "--run-dir",
        str(tmp_path / "run"),
        "--node",
        NODE,
        "--host-probe-image",
        "registry.example/probe@sha256:" + "1" * 64,
        *mode,
    ]


def test_plan_mode_names_the_journal_and_changes_nothing(
    tmp_path: Path, journal: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = journal.read_bytes()
    assert tool.main(_argv(tmp_path, "--plan")) == 0
    plan = json.loads(capsys.readouterr().out)
    (entry,) = plan["journals"]
    assert entry["journal"] == journal.name and entry["refusals"] == [], entry
    assert (
        entry["phase"] == "CLEANUP_REQUIRED" and entry["release_id"] == "bf1bb0037847"
    )
    assert entry["window"]["service"] == "kubelet.service"
    assert entry["host_probe"]["objects"] == []
    assert plan["confirmation"] == tool.CONFIRMATION
    assert "leaves everything as it was" in plan["hard_stop"]
    assert journal.read_bytes() == before
    assert not (journal.parent / sw.RETIRE_STATE).exists(), "no probe, no state"


def test_execute_needs_the_exact_confirmation(tmp_path: Path, journal: Path) -> None:
    with pytest.raises(RegionalFixtureError, match="confirmation must be exactly"):
        tool.main(_argv(tmp_path, "--execute", "--confirm", "yes"))


@pytest.mark.parametrize("retired", [True, False])
def test_execute_reports_and_exits_by_the_outcome(
    tmp_path: Path,
    journal: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    retired: bool,
) -> None:
    from gpu_fault.admin import operator_identity

    monkeypatch.setattr(operator_identity, "caller_identity_arn", lambda **_: None)
    calls: list[dict[str, Any]] = []

    def fake(directory: Path, node: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"directory": directory, "node": node, **kwargs})
        return [
            {
                "journal": journal.name,
                "retired": retired,
                "refusals": [] if retired else ["named"],
            }
        ]

    monkeypatch.setattr(tool, "retire_stale_windows", fake)
    code = tool.main(
        _argv(
            tmp_path,
            "--execute",
            "--confirm",
            tool.CONFIRMATION,
            "--maintenance-window-end",
            "2099-01-01T00:00:00+00:00",
        )
    )
    assert code == (0 if retired else 1)
    result = json.loads(capsys.readouterr().out)
    assert (
        result["retired"] is retired and result["reports"][0]["journal"] == journal.name
    )
    (call,) = calls
    assert call["node"] == NODE and call["via"].endswith("retire_service_window.py")
    assert call["settings"].image.startswith("registry.example/probe@sha256:"), call[
        "settings"
    ]
    (report,) = sorted((journal.parent / sw.RETIRE_STATE).glob("retire-report-*.json"))
    assert json.loads(report.read_text(encoding="utf-8"))["retired"] is retired
