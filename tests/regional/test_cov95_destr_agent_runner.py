from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_agent import COMMAND_ID, AgentHarness


def test_agent_restart_audits_one_command_and_migrates_only_scratch_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.heartbeat_delays = 1
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 0 and report["errors"] == report["cleanup"]["errors"] == [], report
    assert report["agent_generation"] == 4, report
    names = [name for name, _ in h.calls]
    order = [
        "agent.migration-drill",
        "agent.restart-agent",
        "injector.write-xid45",
        "workflow.wait",
        "agent.journal",
        "agent.ledger-audit",
        "agent.ensure-agent-active",
        "agent.disarm-restore",
        "injector.ensure-fabric-manager-active",
    ]
    assert [names.index(name) for name in order] == sorted(
        names.index(name) for name in order
    ), names
    for name in ("agent.journal", "agent.ledger-audit"):
        args = next(detail["args"] for entry, detail in h.calls if entry == name)
        assert args[args.index("--command-id") + 1] == COMMAND_ID, args
    assert not list(tmp_path.glob("destr019-ledger-*")), (
        "scratch migration directory must be removed"
    )
    assert not (tmp_path / "scratch.db").exists(), (
        "migration must never create the named real ledger"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "fabric-kmsg",
        "fabric-clients",
        "fabric-inactive",
        "health",
        "migration",
        "restart",
    ],
)
def test_agent_prerequisite_failure_stops_before_injection(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "fabric-kmsg":
        h.fabric["kmsg_writable"] = False
    elif defect == "fabric-clients":
        h.fabric["compute_clients"] = ["foreign"]
    elif defect == "fabric-inactive":
        h.fabric["fabric_manager"]["ActiveState"] = "inactive"
    elif defect == "health":
        original = h.regional.store_snapshot

        def store(**kwargs: Any) -> dict[str, Any]:
            value = original(**kwargs)
            if "observed_after" in kwargs:
                return value
            h.health_before["http_status"] = 503
            return value

        monkeypatch.setattr(h.regional, "store_snapshot", store)
    elif defect == "migration":
        h.migration_override = {"before": {"user_version": 99}}
    else:
        h.restart_report["after"]["MainPID"] = "100"
        h.restart_report["after"]["InvocationID"] = "i1"
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert h.injected is False, h.calls
    assert any(
        name == "injector.ensure-fabric-manager-active" for name, _ in h.calls
    ), h.calls


@pytest.mark.parametrize("busy", ["queue", "commands", "unknown"])
def test_agent_rechecks_command_quiescence_immediately_before_restart(
    busy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if busy == "queue":
        h.quiet_queue = {"depth": 1}
    elif busy == "commands":
        h.quiet_commands = {"open_by_cluster": {"cluster-a": 1}}
    else:
        h.quiet_commands = None
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert not h.restarted and not h.injected, h.calls


@pytest.mark.parametrize(
    "phase",
    [
        "injector.create",
        "agent.migration-drill",
        "agent.restart-agent",
        "injector.write-xid45",
        "workflow.wait",
        "agent.journal",
        "agent.ledger-audit",
        "provider.events",
        "agent.ensure-agent-active",
        "agent.disarm-restore",
        "injector.ensure-fabric-manager-active",
        "injector.cleanup",
        "runtime.verify",
    ],
)
def test_agent_phase_failure_downgrades_verdict_and_attempts_all_cleanup(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "agent.cleanup" in names and "injector.cleanup" in names, names
    if "agent.restart-agent" in names:
        assert "agent.disarm-restore" in names, names


@pytest.mark.parametrize("defect", ["generation", "incarnation", "boot", "heartbeat"])
def test_agent_restart_must_keep_incarnation_and_refresh_heartbeat(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "generation":
        h.changed_agent["generation"] = 5
    elif defect == "incarnation":
        h.changed_agent["agent_incarnation_id"] = "other"
    elif defect == "boot":
        h.changed_agent["boot_id"] = "new-boot"
    else:
        h.heartbeat_delays = 10000
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert not h.injected, h.calls


@pytest.mark.parametrize(
    "defect", ["journal", "ledger", "health-counters", "provider", "disarm"]
)
def test_agent_audit_or_cleanup_failure_never_certifies_success(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "journal":
        h.journal["lines"] = []
    elif defect == "ledger":
        h.audit["rows"][0]["fencing_token"] = 999
    elif defect == "health-counters":
        h.health_after["payload"]["counters"]["completed"] = 2
    elif defect == "provider":
        h.events = [{"event_name": "BatchRebootClusterNodes"}]
    else:
        h.disarm["restore_timer"]["ActiveState"] = "active"
    code, report = h.execute(tmp_path)
    assert code == 1 and (report["errors"] or report["cleanup"]["errors"]), report


@pytest.mark.parametrize(
    "phase", ["before", "agent.migration-drill", "agent.restart-agent"]
)
def test_agent_window_expiry_stops_the_next_mutation(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "window ended" in report["error"], report
    assert not h.injected, h.calls


def test_agent_abort_during_restart_keeps_failsafe_cleanup_owed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["agent.restart-agent"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert names.index("agent.ensure-agent-active") < names.index(
        "agent.disarm-restore"
    ), names
    assert "injector.cleanup" in names, names


def test_agent_plan_drift_rejects_generation_change_before_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.agent["generation"] += 1
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        h.execute(tmp_path)
    assert not h.restarted, h.calls
