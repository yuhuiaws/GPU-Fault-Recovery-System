"""GF-REGIONAL-DESTR-014 runner: the verdict and supervision paths the happy
harness never takes -- a HyperPod node count or control-plane blast radius
that moved during the case, a follow-up workflow that would execute a
recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.regional._cov95_destr_exhaustion import ExhaustionHarness


def test_a_moved_provider_count_or_blast_radius_fails_the_data_plane_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], "the drift must happen inside the case"
    provider_inventory = h.warm.provider_inventory
    cpu_blast_snapshot = h.regional.cpu_blast_snapshot

    def grown_inventory() -> dict[str, Any]:
        inventory = provider_inventory()
        if h.rebooted:
            inventory["count"] += 1
        return inventory

    def moved_blast_radius() -> dict[str, Any]:
        snapshot = cpu_blast_snapshot()
        if h.rebooted:
            snapshot["cpu"] = [*snapshot.get("cpu", []), "api-c"]
        return snapshot

    monkeypatch.setattr(h.warm, "provider_inventory", grown_inventory)
    monkeypatch.setattr(h.regional, "cpu_blast_snapshot", moved_blast_radius)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert report["verdict"] == "FAIL", report
    assert "HyperPod node count changed" in report["errors"], report["errors"]
    assert "control-plane EKS state differs from baseline" in report["errors"], report[
        "errors"
    ]


def test_a_follow_up_that_would_execute_a_recovery_fails_the_control_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == []
    cpu_python = h.regional.cpu_python

    def executable_follow_up(script: str, request_id: str) -> dict[str, Any]:
        value = cpu_python(script, request_id)
        value["workflow"]["official_steps"] = [
            {"operation": "ESCALATE_SUPPORT"},
            {"operation": "RESTART_NODE"},
        ]
        return value

    monkeypatch.setattr(h.regional, "cpu_python", executable_follow_up)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert (
        "unknown reboot generated an executable recovery successor"
        in (report["errors"])
    ), report["errors"]
    assert report["verdict"] == "FAIL", report
