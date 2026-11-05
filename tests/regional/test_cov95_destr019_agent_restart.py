"""DESTR-019 Agent restart: preflight refusals and probe residue at the cleanup."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr019_agent_restart_ledger as case
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_agent import AgentHarness
from tests.regional._cov95_destr_warm import NOW


def test_preflight_names_an_unpassed_predecessor_and_probe_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.predecessor_valid = False
    h.residuals = {"pod": True, "configmap": False}

    preflight = h.plan(tmp_path)

    assert preflight["errors"] == [
        f"{case.PREDECESSOR_CASE_ID} predecessor evidence is not PASS",
        "probe residue remains after the preflight: {'pod': True, 'configmap': False}",
    ]


def test_execute_refuses_to_start_on_a_failed_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == []
    h.predecessor_valid = False
    h.calls.clear()

    with pytest.raises(RegionalFixtureError, match="^preflight failed: .*predecessor"):
        case.execute_case(h.settings, tmp_path, 1, NOW + timedelta(hours=1))

    names = [name for name, _ in h.calls]
    assert "agent.restart-agent" not in names, names
    evidence = tmp_path / "cases" / case.CASE_ID / f"{case.CASE_ID}.json"
    assert not evidence.exists(), "a refused run publishes no evidence"


def test_probe_residue_left_by_the_run_fails_its_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == []
    record = h.call

    def call(name: str, detail: Any = None) -> None:
        record(name, detail)
        if name == "agent.restart-agent":
            # From here on every probe cleanup reports its Pod left behind.
            h.residuals = {"pod": True}

    monkeypatch.setattr(h, "call", call)

    code, report = h.execute(tmp_path)

    assert code == 1, report
    assert any(
        "residual resources remain: {'pod': True}" in str(error)
        for error in report["cleanup"]["errors"]
    ), report["cleanup"]
