"""Script entrypoints are exercised only behind preinstalled fake boundaries."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import (
    collector_window_fixture,
    live_driver_guard,
    net_command_fixture,
    regional_live_fixture,
)
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/e2e/regional"


@pytest.mark.parametrize(
    "script,kind",
    [
        ("run_collect016_training_recovery.py", "standard"),
        ("run_collect017_efa_plugin.py", "standard"),
        ("run_collect018_rejected_event.py", "window"),
        ("run_collect019_nvidia_smi_hang.py", "window"),
        ("run_collect020_gpu_identity.py", "window"),
        ("run_collect021_late_xid_after_pod_death.py", "standard"),
        ("run_collect022_fm_cursor_recovery.py", "standard"),
        ("run_collector_acceptance.py", "selected"),
        ("run_collector_destructive.py", "selected"),
        ("run_net002_command_recovery.py", "command"),
        ("run_net003_result_retry.py", "command"),
        ("run_net006_lease_loss_withheld_result.py", "plain"),
        ("run_net007_transient_api_outage.py", "standard"),
        ("run_net008_outbox_dead_letter.py", "window"),
    ],
)
def test_script_dispatches_to_guarded_owner_without_running_case(
    monkeypatch: Any, script: str, kind: str
) -> None:
    calls = []

    def record(label: str, *args: Any, **kwargs: Any) -> int:
        calls.append((label, args, kwargs))
        return 13

    monkeypatch.setattr(
        live_driver_guard,
        "run_standard_case",
        lambda *a, **k: record("standard", *a, **k),
    )
    monkeypatch.setattr(
        live_driver_guard,
        "run_selected_case",
        lambda *a, **k: record("selected", *a, **k),
    )
    monkeypatch.setattr(
        live_driver_guard, "run_plain_case", lambda *a, **k: record("plain", *a, **k)
    )
    monkeypatch.setattr(
        collector_window_fixture, "run_window_case", lambda **k: record("window", **k)
    )
    monkeypatch.setattr(
        net_command_fixture, "run_main", lambda **k: record("command", **k)
    )
    monkeypatch.setattr(
        regional_live_fixture, "run_case_main", lambda callback: callback()
    )
    monkeypatch.setattr(sys, "argv", ["isolated-entrypoint"])
    monkeypatch.setattr(
        sys,
        "path",
        [value for value in sys.path if value not in {str(ROOT), str(ROOT / "src")}],
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    assert caught.value.code == 13
    assert len(calls) == 1
    assert calls[0][0] == kind
    if kind in {"window", "command"}:
        assert calls[0][2]["case_id"].startswith("GF-REGIONAL-"), (
            "entrypoint must delegate its regional case identity"
        )
    else:
        surface = calls[0][1][0]
        assert callable(surface.parser), "entrypoint must preserve its parser callback"


def test_net001_default_entrypoint_cannot_cross_fake_plan_boundary(
    tmp_path: Path, monkeypatch: Any
) -> None:
    kubeconfig = tmp_path / "fixture-kubeconfig"
    kubeconfig.touch()
    plans = []
    monkeypatch.setattr(live_driver_guard, "install_site_profile", lambda: None)
    monkeypatch.setattr(
        live_driver_guard,
        "build_plan",
        lambda **kwargs: plans.append(kwargs) or {"fixture": True},
    )
    monkeypatch.setattr(
        regional_live_fixture, "predecessor_evidence", lambda *a, **k: {"valid": True}
    )
    monkeypatch.setattr(os, "umask", lambda mode: 0o077)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isolated-entrypoint",
            "--run-dir",
            str(tmp_path),
            "--cpu-kubeconfig",
            str(kubeconfig),
            "--gpu-kubeconfig",
            str(kubeconfig),
            "--gpu-context",
            "gpu-a",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-test-1",
            "--node",
            "node-a",
            "--endpoint-host",
            "control.invalid",
            "--host-probe-image",
            "example@sha256:" + "a" * 64,
        ],
    )
    monkeypatch.setattr(
        sys, "path", [value for value in sys.path if value != str(ROOT)]
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(
            str(SCRIPTS / "run_net001_collector_replay.py"), run_name="__main__"
        )
    assert caught.value.code == 0
    assert len(plans) == 1
    assert plans[0]["arguments"].execute is False
    assert plans[0]["environment"]["GPU_FAULT_CLUSTER_ID"] == "cluster-a"


@pytest.mark.parametrize(
    "script,args,error",
    [
        ("audit_collector_outbox.py", ["--release-id", "fixture-release"], SystemExit),
        ("audit_net004_dependency_boundary.py", [], SystemExit),
        ("audit_raw_evidence_periodic_cleanup.py", ["invalid"], ValueError),
    ],
)
def test_audit_entrypoint_refuses_invalid_arguments_before_external_work(
    monkeypatch: Any, script: str, args: list[str], error: type[Exception]
) -> None:
    monkeypatch.setattr(os, "umask", lambda mode: 0o077)
    monkeypatch.setattr(sys, "argv", ["isolated-entrypoint", *args])
    monkeypatch.setattr(
        sys, "path", [value for value in sys.path if value != str(ROOT)]
    )
    with pytest.raises(error) as caught:
        runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == 1
