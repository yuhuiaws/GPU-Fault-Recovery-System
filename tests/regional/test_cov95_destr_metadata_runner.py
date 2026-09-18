from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e.regional import run_destr021_adversarial_node_metadata as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_metadata import MetadataHarness, verdicts


def test_metadata_case_preserves_conflict_waiting_evidence_and_cleans_all_owned_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.workflow_delays = 1
    h.incident_delays = 3
    code, report = h.execute(tmp_path)
    assert code == 0 and report["errors"] == report["cleanup"]["errors"] == [], report
    assert report["conflict_retries_observed"] == 1, report
    assert report["foreign_incident"] == h.foreign, report
    assert h.bound and not h.writer.running, h.calls
    assert all(
        h.node["annotations"].get(key) is None for key in case.MUTATED_ANNOTATIONS
    ), h.node
    names = [name for name, _ in h.calls]
    assert (
        names.index("node.patch")
        < names.index("writer.start")
        < names.index("collector.unbind-efa")
    ), names
    assert names.index("collector.restore-efa") < names.index("writer.clear"), names


@pytest.mark.parametrize(
    "defect", ["predecessor", "tests", "no-device", "efa", "residual"]
)
def test_metadata_preflight_refuses_unproven_prerequisites(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    if defect == "predecessor":
        h.predecessor_valid = False
    elif defect == "tests":
        h.tests_pass = False
    elif defect == "no-device":
        h.no_bound_device = True
    elif defect == "efa":
        h.efa_count = "unknown"
    else:
        h.residuals["pod"] = True
    assert h.plan(tmp_path)["errors"], h.calls
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert not h.injection_attempted, h.calls


@pytest.mark.parametrize("defect", ["drop-preseed", "queue", "commands"])
def test_metadata_case_refuses_injection_if_preseed_or_quiescence_is_unproven(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "drop-preseed":
        h.drop_patch = True
    elif defect == "queue":
        h.quiet_queue = {"depth": 1}
    else:
        h.quiet_commands = None
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert not h.injection_attempted, h.calls
    assert report["cleanup"]["writer"] == {"not_started": True}, report


@pytest.mark.parametrize(
    "phase",
    [
        "node.patch",
        "writer.start",
        "collector.unbind-efa",
        "workflow.latest",
        "commands.read",
        "provider.events",
        "collector.restore-efa",
        "writer.clear",
        "runtime.verify",
    ],
)
def test_metadata_phase_failure_keeps_efa_restore_and_writer_cleanup_owed(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "writer.stop" in names and "collector.cleanup" in names, names
    if "collector.unbind-efa" in names:
        assert "collector.restore-efa" in names, names


def test_metadata_writer_still_running_defers_annotation_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.writer.stuck = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["annotations_restore_deferred"] is True, (
        report
    )
    assert h.node["annotations"][verdicts.TICK_ANNOTATION] == "tick", h.node


@pytest.mark.parametrize(
    "defect", ["missing-workflow", "missing-recovery", "provider", "writer-report"]
)
def test_metadata_unsettled_workflow_or_incomplete_race_is_not_pass(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "missing-workflow":
        h.workflow_missing = True
    elif defect == "missing-recovery":
        h.recovery_missing = True
    elif defect == "provider":
        h.events = [{"event_name": "BatchRebootClusterNodes"}]
    else:
        h.writer.report_changes["patches"] = 0
    code, report = h.execute(tmp_path)
    assert code == 1 and (
        report.get("error") or report.get("errors") or report["cleanup"]["errors"]
    ), report
    assert h.bound, h.calls


@pytest.mark.parametrize("phase", ["before", "store.quiet"])
def test_metadata_expiry_stops_before_efa_unbind(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "maintenance window" in report["error"], report
    assert not h.injection_attempted, h.calls


def test_metadata_abort_on_unbind_ack_still_restores_efa_and_stops_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["collector.unbind-efa"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert "collector.restore-efa" in names and "writer.stop" in names, names
    assert "collector.cleanup" in names, names


def test_metadata_plan_rejects_node_uid_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MetadataHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "recreated"
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        h.execute(tmp_path)
    assert not h.injection_attempted, h.calls
