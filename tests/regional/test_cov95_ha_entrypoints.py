from __future__ import annotations

import importlib
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import (
    live_driver_guard,
    regional_commands,
    regional_live_fixture,
)

ROOT = Path(__file__).resolve().parents[2] / "scripts/e2e/regional"
RUNNERS = [
    "run_ha001_control_plane_failover.py",
    "run_ha002_pdb_topology.py",
    "run_ha003_aurora_failover_reset.py",
    "run_ha004_waiting_reclaim_reset.py",
    "run_ha005_rollout_continuity.py",
    "run_ha006_executor_takeover.py",
    "run_ha007_control_worker_shutdown.py",
    "run_ha008_processor_exit_acceptance.py",
    "run_ha008_processor_exit_probe.py",
    "run_ha009_aurora_credential_rotation.py",
    "run_ha010_aurora_blackout_liveness.py",
    "run_notification_acceptance.py",
    "run_notify007_delivery_states.py",
]


@pytest.mark.parametrize("filename", RUNNERS)
def test_cli_help_contract_under_fake_profile_and_transport_boundaries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], filename: str
) -> None:
    monkeypatch.syspath_prepend(str(ROOT))
    modules = [
        live_driver_guard,
        regional_live_fixture,
        regional_commands,
        *(
            importlib.import_module(name)
            for name in (
                "live_driver_guard",
                "regional_live_fixture",
                "regional_commands",
            )
        ),
    ]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("help must never start a child or use a remote transport")

    for module in modules:
        for name in ("install_site_profile", "install_abort_signals"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, lambda: None)
        for name in ("run_fixture_command", "run_command"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(sys, "argv", [filename, "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / filename), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
