from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e.regional import run_destr018_lifetime_deadline as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_lifetime import LifetimeHarness, data, window


def test_lifetime_drill_proves_cancellation_absorb_and_ordered_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.absorb_delays = 1
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    assert report["errors"] == report["cleanup"]["errors"] == [], report
    assert report["cancelled_at"] == data.T_CANCEL.isoformat(), report
    assert report["cleanup"]["close_reset_incident"]["closed"] is True, report
    assert report["metrics"]["lifetime_exceeded_total"]["delta"] == 1, report
    names = [name for name, _ in h.calls]
    order = [
        "window.open",
        "holder.arm-holder",
        "holder.holder-status",
        "injector.write-xid46",
        "workflow.wait",
        "injector.write-xid79",
        "holder.disarm-holder",
        "restore.create",
        "window.close",
        "injector.cleanup",
    ]
    assert [names.index(name) for name in order] == sorted(
        names.index(name) for name in order
    ), names
    assert (
        next(detail for name, detail in h.calls if name == "window.open")
        == h.settings.assignments()
    ), h.calls
    worker = h.runtime["deployments"]["cpu"][window.DEPLOYMENT]
    assert worker["generation"] == 3 and worker["template_sha256"] == "original", worker


@pytest.mark.parametrize(
    "defect", ["inventory", "clients", "quiesce", "kmsg", "holder"]
)
def test_lifetime_host_premise_must_hold_before_injection(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "inventory":
        h.baseline["gpu_inventory"] = []
    elif defect == "clients":
        h.baseline["compute_clients"] = ["foreign"]
    elif defect == "quiesce":
        h.baseline["quiesce_states"] = ["foreign"]
    elif defect == "kmsg":
        h.baseline["kmsg_writable"] = False
    else:
        h.holder_clients = []
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert h.injected is False, h.calls
    if defect == "holder":
        assert "would reset a live GPU" in report["error"], report
        assert report["cleanup"]["disarm_holder"]["ok"] is True, report


@pytest.mark.parametrize(
    "phase",
    [
        "injector.create",
        "holder.create",
        "window.open",
        "holder.arm-holder",
        "injector.write-xid46",
        "workflow.wait",
        "escalation.read",
        "injector.write-xid79",
        "provider.events",
        "holder.disarm-holder",
        "restore.create",
        "restore.wait",
        "window.close",
        "holder.cleanup",
        "injector.cleanup",
    ],
)
def test_lifetime_phase_failure_preserves_failure_and_finishes_remaining_cleanup(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "holder.cleanup" in names and "injector.cleanup" in names, names
    if "window.open" in names:
        assert "window.close" in names, names


@pytest.mark.parametrize(
    "defect",
    ["xid", "steps", "cancellation", "cadence", "escalation", "provider", "metric"],
)
def test_lifetime_evidence_defects_cannot_certify_deadline_behavior(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "xid":
        h.state["event"]["xid"] = 79
    elif defect == "steps":
        h.state["workflow"]["official_steps"] = []
    elif defect == "cancellation":
        h.state["commands"] = []
    elif defect == "cadence":
        h.after["ledger"] = []
    elif defect == "escalation":
        h.escalation = {}
    elif defect == "provider":
        h.events = [{"event_name": "BatchRebootClusterNodes"}]
    else:
        h.metric_after = 1
    code, report = h.execute(tmp_path)
    assert code == 1 and report["errors"], report
    if defect in {"xid", "steps", "cancellation", "cadence", "escalation"}:
        assert "injector.write-xid79" not in [name for name, _ in h.calls], h.calls


@pytest.mark.parametrize(
    "phase", ["before", "injector.snapshot", "holder.holder-status", "escalation.read"]
)
def test_lifetime_window_expiry_blocks_later_injections(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "window ended" in report["error"], report
    if phase == "escalation.read":
        assert "injector.write-xid79" not in [name for name, _ in h.calls], h.calls
    else:
        assert not h.injected, h.calls


def test_lifetime_abort_disarms_holder_and_closes_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["workflow.wait"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert names.index("holder.disarm-holder") < names.index("window.close"), names
    assert "holder.cleanup" in names and "injector.cleanup" in names, names


@pytest.mark.parametrize("all_refused", [False, True])
def test_lifetime_restore_tries_only_the_known_incident_owners(
    all_refused: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[f"incident.idle/{data.SUPPORT_INCIDENT_ID}"] = RuntimeError(
        "owner retired"
    )
    if all_refused:
        h.failures["incident.idle/inc-1"] = RuntimeError("reset owner unproven")
    code, report = h.execute(tmp_path)
    assert code == (1 if all_refused else 0), report
    if all_refused:
        assert "no incident could carry" in str(report["cleanup"]["errors"]), report
    else:
        restored = report["cleanup"]["restore_isolated_node"]
        assert restored["incident_id"] == "inc-1" and len(restored["refused"]) == 1, (
            restored
        )
        assert report["cleanup"]["close_reset_incident"]["closed"] is False, report


@pytest.mark.parametrize(
    "variation", ["restore-failed", "incident-stuck", "probe-residual"]
)
def test_lifetime_cleanup_must_close_incident_and_remove_resources(
    variation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if variation == "restore-failed":
        h.restore_status = "FAILED"
    elif variation == "incident-stuck":
        h.incident_close_stuck = True
    else:
        h.residuals["pod"] = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["errors"], report


def test_lifetime_unrecorded_env_window_or_open_incident_prevents_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.open_incidents = [{"incident_id": "earlier-case", "state": "ESCALATED"}]
    h.survey_data["deployment"]["variables"] = {"foreign-window": {"present": True}}
    preflight = h.plan(tmp_path)
    assert any("earlier-case" in error for error in preflight["errors"]), preflight
    assert any("already set" in error for error in preflight["errors"]), preflight
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert not any(name == "window.open" for name, _ in h.calls), h.calls


def test_lifetime_plan_uid_drift_refuses_before_opening_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "replacement"
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        h.execute(tmp_path)
    assert not any(name == "window.open" for name, _ in h.calls), h.calls


@pytest.mark.parametrize("explicit", [False, True])
def test_lifetime_configure_keeps_timing_values_and_predecessor_path(
    explicit: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    args = case.parser().parse_args(["--run-dir", str(tmp_path)])
    args.cpu_kubeconfig = str(h.settings.regional.cpu_kubeconfig)
    args.gpu_kubeconfig = str(h.settings.regional.gpu_kubeconfig)
    args.gpu_context = "fake-gpu"
    args.cluster_id = "cluster-a"
    args.region = "us-west-2"
    args.node = data.NODE
    args.host_probe_image = "example.test/probe"
    args.predecessor_evidence = str(tmp_path / "previous.json") if explicit else ""
    configured = case.configure(args)
    assert configured.node == data.NODE and configured.hold_seconds == 1200, configured
    assert configured.assignments() == h.settings.assignments(), configured
    assert configured.predecessor_path.name == (
        "previous.json" if explicit else f"{case.PREDECESSOR_CASE_ID}.json"
    ), configured
    assert configured.environment()["GPU_FAULT_DESTR018_LIFETIME_SECONDS"] == str(
        configured.lifetime_seconds
    ), configured
