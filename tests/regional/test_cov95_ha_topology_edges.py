from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha001_control_plane_failover as common
from scripts.e2e.regional import run_ha002_pdb_topology as ha002
from tests.regional._cov95_ha001_harness import DEADLINE
from tests.regional._cov95_ha002_harness import HA002Harness


def test_missing_ingress_peer_refuses_before_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    del harness.pods[f"{common.INGRESS_APP}-2"]
    with pytest.raises(RuntimeError, match="second ingress"):
        harness.execute()
    assert harness.events == []


@pytest.mark.parametrize("phase", ["cordon", "eviction"])
def test_window_expiry_during_the_baseline_or_cordon_stops_the_next_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)

    def now(tz: Any = None) -> datetime:
        expired = (
            harness.clock.now >= 10
            if phase == "cordon"
            else harness.nodes["node-0"]["spec"]["unschedulable"]
        )
        return (
            DEADLINE + timedelta(seconds=1) if expired else datetime.now(timezone.utc)
        )

    monkeypatch.setattr(ha002, "datetime", SimpleNamespace(now=now))
    code, report = harness.execute()
    assert code == 1
    assert "maintenance window ended" in report["error"]
    assert harness.evictions == []
    assert harness.resources == {}
    assert harness.nodes["node-0"]["spec"]["unschedulable"] is False


def test_cordon_ack_without_observed_state_never_authorizes_eviction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    run = common.run

    def unobserved(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "patch" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return run(argv, **kwargs)

    monkeypatch.setattr(common, "run", unobserved)
    code, report = harness.execute()
    assert code == 1
    assert "did not become unschedulable" in report["error"]
    assert harness.evictions == []
    assert harness.watchdog.returncode == 0
    assert report["cleanup_errors"] == []


def test_spool_peer_loss_stops_the_spool_disruption_and_fails_postflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path, spool=3)
    observe = common.observe_phase

    def lose_peers(label: str, *args: Any, **kwargs: Any) -> dict:
        result = observe(label, *args, **kwargs)
        if label == "ha002-worker":
            del harness.pods[f"{common.SPOOL_APP}-1"]
            del harness.pods[f"{common.SPOOL_APP}-2"]
        return result

    monkeypatch.setattr(common, "observe_phase", lose_peers)
    code, report = harness.execute()
    assert code == 1
    assert "no second spool-worker" in report["error"]
    assert len(harness.evictions) == 4
    assert any(
        "spool-worker replicas" in error for error in report["cleanup_errors"]
    ), report


def test_taint_drift_is_reported_without_removing_the_foreign_taint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    recover = ha002.wait_recovery

    def changed(**kwargs: Any) -> list[dict]:
        value = recover(**kwargs)
        harness.nodes["node-0"]["spec"]["taints"] = [{"key": "foreign"}]
        return value

    monkeypatch.setattr(ha002, "wait_recovery", changed)
    code, report = harness.execute()
    assert code == 1
    assert "node scheduling state was not restored" in report["errors"]
    assert harness.nodes["node-0"]["spec"]["taints"] == [{"key": "foreign"}]
    assert any("taints changed" in error for error in report["cleanup_errors"]), report


def test_remaining_probe_resources_invalidate_an_otherwise_successful_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    harness.delete_failure = OSError("unit delete unavailable")
    code, report = harness.execute()
    assert code == 1
    assert any(
        "probe resources remain" in error for error in report["cleanup_errors"]
    ), report
    assert harness.nodes["node-0"]["spec"]["unschedulable"] is False


def test_final_probe_window_is_rechecked_after_the_last_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    roles = common.role_snapshot
    read = common.read_probe
    role_reads = 0

    def role_snapshot() -> dict:
        nonlocal role_reads
        role_reads += 1
        return roles()

    def read_probe(*args: Any) -> dict:
        value = read(*args)
        if role_reads == 2:
            value["max_failure_window_seconds"] = 999
        return value

    monkeypatch.setattr(common, "role_snapshot", role_snapshot)
    monkeypatch.setattr(common, "read_probe", read_probe)
    code, report = harness.execute()
    assert code == 1
    assert any("failure window exceeded" in error for error in report["errors"]), report
    assert report["cleanup_errors"] == []
