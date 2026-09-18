from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_ha_reset_harness import ResetHarness


@pytest.fixture(params=[ha003, ha004], ids=["aurora", "executor"])
def harness(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> ResetHarness:
    return ResetHarness(monkeypatch, tmp_path, request.param)


def test_full_reset_window_lifecycle_keeps_identity_and_cleans(
    harness: ResetHarness,
) -> None:
    baseline = copy.deepcopy(harness.api.document["spec"])
    code, report = harness.execute()
    assert code == 0, report.get("error", report.get("errors"))
    assert report["verdict"] == "PASS"
    assert report["cleanup"]["errors"] == []
    assert report["incident_id"] == "incident-unit"
    assert report["workflow_request_id"] == "workflow-unit"
    assert report["release_id"] == "unit-release"
    assert report["cluster_id"] == "unit-cluster"
    assert (
        "restore-quiesce",
        ("--incident-id", "incident-unit"),
    ) in harness.host.arguments
    assert harness.events.index("start-reset-sampler") < harness.events.index(
        "write-xid46"
    )
    assert harness.events.index("restore-quiesce") < harness.events.index(
        "host-cleanup"
    )
    assert harness.host.settings.state_directory == harness.directory / "host-probes"
    if harness.module is ha003:
        assert report["failover_overlap_observed"] is True
        assert report["rds_after"]["writer"] != report["rds_before"]["writer"]
        # Two full binding reads remain: the execute-phase preflight and the
        # one immediately before the reset is injected. The barrier before the
        # failover request no longer re-reads the binding (~40 s live, most of
        # the ~65 s a batched reset stays WAITING); it is one describe call.
        assert harness.binding_reads == 2
        injected = harness.events.index("write-xid46")
        requested = harness.events.index("failover-request")
        assert injected < requested
        assert harness.events[injected:requested].count("rds-describe") == 1, (
            "the barrier between the injection and the failover request is one "
            "describe-db-clusters call"
        )
        timing = json.loads((harness.directory / "failover-timing.json").read_text())
        stamps = [
            "claim_observed_at",
            "recheck_started_at",
            "aurora_checked_at",
            "store_snapshot_at",
            "requested_at",
        ]
        assert sorted(timing) == sorted(stamps), (
            "the failover request must record where the reset window went"
        )
        assert [timing[key] for key in stamps] == sorted(
            timing[key] for key in stamps
        ), "the stamps must follow the barrier's order: claim, describe, store, request"
        assert report["provider_events_provisional"] is True
    else:
        assert report["remote_command_ids"] == ["command-4"]
        assert report["lease_owners"] == ["unit/pod-0", "unit/pod-1"]
        assert report["killed_owner_pod"]["uid"] == "uid-0"
        assert harness.api.document["spec"] == baseline
        assert harness.watchdog.returncode == 0
        assert harness.events.index("watchdog-start") < harness.events.index(
            "write-xid46"
        )


@pytest.mark.parametrize("phase", ["host-create", "start-reset-sampler", "write-xid46"])
def test_abort_at_each_setup_stage_limits_cleanup_to_started_work(
    harness: ResetHarness, phase: str
) -> None:
    harness.host.failure_at[phase] = RegionalFixtureError("unit abort")
    code, report = harness.execute()
    assert code == 1
    assert "unit abort" in report["error"]
    assert report["cleanup"]["errors"] == []
    assert "host-cleanup" in harness.events
    assert "failover-request" not in harness.events
    assert "delete-owner" not in harness.events
    assert ("stop-reset-sampler" in harness.events) is (phase == "write-xid46")


@pytest.mark.parametrize("phase", ["host-create", "write-xid46", "host-cleanup"])
def test_supervision_loss_prohibits_any_further_transport(
    harness: ResetHarness, phase: str
) -> None:
    harness.host.failure_at[phase] = ProcessSupervisionLost("unit supervision lost")
    code, report = harness.execute()
    assert code == 1
    assert report["supervision_lost"] is True
    assert report["cleanup"] is None
    assert harness.events[-1] == phase


@pytest.mark.parametrize(
    "phase", ["restore-quiesce", "stop-reset-sampler", "host-cleanup"]
)
def test_cleanup_failures_are_recorded_and_other_cleanup_still_runs(
    harness: ResetHarness, phase: str
) -> None:
    harness.host.failure_at[phase] = RuntimeError("unit cleanup failure")
    code, report = harness.execute()
    assert code == 1
    assert any(
        "unit cleanup failure" in value for value in report["cleanup"]["errors"]
    ), "cleanup transport errors must affect the verdict"
    assert "restore-quiesce" in harness.events
    assert "stop-reset-sampler" in harness.events
    assert "host-cleanup" in harness.events
    if harness.module is ha004:
        assert "watchdog-stop" in harness.events


@pytest.mark.parametrize("stage", ["capture", "boundary", "terminal"])
def test_incident_drift_is_rejected_without_losing_original_cleanup_identity(
    harness: ResetHarness, stage: str
) -> None:
    if stage == "terminal":
        harness.api.final["incident"]["incident_id"] = "foreign-incident"
    else:
        index = 1 if stage == "capture" else 2
        harness.api.states[index]["incident"]["incident_id"] = "foreign-incident"
    code, report = harness.execute()
    assert code == 1
    assert "incident identity changed" in report["error"]
    assert (
        "restore-quiesce",
        ("--incident-id", "incident-unit"),
    ) in harness.host.arguments


@pytest.mark.parametrize(
    "defect,error",
    [
        ("node", "Ready and pre-cordoned"),
        ("ownership", "pre-existing gpu-fault ownership"),
        ("workload", "non-system running Pods"),
        ("agent", "not ACTIVE"),
        ("capability", "gpuReset is not OWN"),
        ("queue", "processor queue is not empty"),
        ("commands", "remote command queue is not empty"),
        ("tests", "focused regression tests failed"),
    ],
)
def test_preflight_rejects_unsafe_or_unknown_baselines(
    harness: ResetHarness, defect: str, error: str
) -> None:
    api = harness.api
    if defect == "node":
        api.node["ready"] = "Unknown"
    elif defect == "ownership":
        api.node["ownership_annotations"] = {"unit-owner": "foreign"}
    elif defect == "workload":
        api.workloads = [{"name": "foreign-workload"}]
    elif defect == "agent":
        api.baseline["agent"] = {}
    elif defect == "capability":
        api.baseline["profile"]["capabilities"] = {}
    elif defect == "queue":
        api.baseline["queue"]["depth"] = 1
    elif defect == "commands":
        api.baseline["remote_commands"]["open_by_cluster"] = {"unit": 1}
    else:
        api.focused_returncode = 1
    with pytest.raises(RegionalFixtureError, match=error):
        harness.execute()
    assert "host-create" not in harness.events
    assert "write-xid46" not in harness.events


@pytest.mark.parametrize("module", [ha003, ha004])
def test_source_valid_focused_result_is_reused_without_nested_pytest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: Any
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, module)
    harness.reuse_tests = True
    preflight = module.read_only_preflight(harness.settings, harness.directory)
    assert preflight["errors"] == []
    assert preflight["focused_tests"]["reused_from_plan"] is True
    assert "focused-tests" not in harness.events


@pytest.mark.parametrize(
    "boundary",
    [1, 2, "writer", "status"],
    ids=["preflight", "injection", "failover-writer", "failover-status"],
)
def test_aurora_binding_drift_stops_before_the_next_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, boundary: int | str
) -> None:
    """Drift at the preflight and injection boundaries is the full binding
    read refusing. The barrier before the failover request no longer re-reads
    the binding: drift there is its one describe call answering a moved
    writer or a cluster that is not available, and it must stop before the
    store re-read and the provider request."""
    harness = ResetHarness(monkeypatch, tmp_path, ha003)
    if isinstance(boundary, int):
        harness.binding_failure_at = boundary
        error = "Aurora binding drift"
    else:
        harness.api.rds_drift_before_failover = boundary
        error = "Aurora moved before the failover request"
    if boundary == 1:
        with pytest.raises(RegionalFixtureError, match=error):
            harness.execute()
        assert "host-create" not in harness.events
    else:
        code, report = harness.execute()
        assert code == 1
        assert error in report["error"]
        assert "host-cleanup" in harness.events
    assert "failover-request" not in harness.events
    assert ("write-xid46" in harness.events) is (boundary not in (1, 2))
    if boundary not in (1, 2):
        # The reset capture took two store reads (PENDING, then WAITING);
        # a moved writer must refuse before the barrier's third one.
        assert harness.api.state_reads == 2, harness.events


def test_failover_after_reset_is_inconclusive_not_a_false_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha003)
    for state in harness.api.states[3:]:
        state["commands"][4]["status"] = "SUCCEEDED"
    code, report = harness.execute()
    assert code == 1
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["errors"] == []
    assert report["cleanup"]["errors"] == []


def test_executor_pod_uid_race_is_rejected_by_the_conditional_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    harness.api.replace_on_delete = True
    code, report = harness.execute()
    assert code == 1
    assert "replacement Pod UID" in report["error"]
    assert "delete-owner" not in harness.events
    assert report["cleanup"]["errors"] == []
    assert harness.watchdog.returncode == 0
