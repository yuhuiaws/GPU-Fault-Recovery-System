from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr001_gpu_reset as reset_case
from scripts.e2e.regional import run_destr023_idle_cluster_reset as idle_case
from scripts.e2e.regional import run_destr024_watcher_down_fail_closed as watcher_case
from scripts.e2e.regional.host_probe_fixture import HostProbeTransportError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_idle import IdleHarness
from tests.regional._destructive_acceptance_builders import escalated_reset_state

POD_GONE = "owned host probe Pod has terminated"


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
def test_idle_runner_success_uses_observed_evidence_and_finishes_cleanup(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    assert report["errors"] == [], report
    assert report["probe_residuals"] == {"pod": False, "configmap": False}, report
    assert report["provider_events_provisional"] is True, report
    names = [name for name, _ in h.calls]
    assert names.index("host.create") < names.index("host.write-xid46"), names
    assert names.index("workflow.wait") < names.index("host.cleanup"), names
    if module is watcher_case:
        assert report["coverage_stale"]["workload_state"] == "UNKNOWN", report
        assert report["coverage_recovered"]["workload_state"] == "IDLE", report
        order = [
            "watchdog.arm",
            "watcher.down",
            "host.write-xid46",
            "watcher.restore",
            "watcher.rollout",
            "watchdog.kill",
            "restore.create",
            "host.cleanup",
        ]
        assert [names.index(name) for name in order] == sorted(
            names.index(name) for name in order
        ), names
        assert report["workflow_status"] == "BLOCKED", report
        restore = next(detail for name, detail in h.calls if name == "restore.create")
        assert restore["incident_id"] == report["incident_id"], restore
        assert "host.restore-quiesce" not in names, names
    else:
        order = [
            "host.start-reset-sampler",
            "host.write-xid46",
            "workflow.wait",
            "host.stop-reset-sampler",
            *(["host.restore-quiesce"] if module is idle_case else []),
            "host.cleanup",
        ]
        assert [names.index(name) for name in order] == sorted(
            names.index(name) for name in order
        ), names
        if module is reset_case:
            assert report["quiesce_recovery"]["product_restoration_observed"] is True
            assert "host.restore-quiesce" not in names
        else:
            assert report["quiesce_recovery"]["ok"] is True, report


@pytest.mark.parametrize("module", [idle_case, watcher_case])
@pytest.mark.parametrize(
    "fault", ["predecessor", "managed", "canary", "coverage", "watcher", "tests"]
)
def test_idle_preflight_refuses_invalid_premises_before_probes(
    module: Any, fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    if fault == "predecessor":
        h.predecessor_valid = False
    elif fault == "managed":
        h.pods = [
            {
                "metadata": {
                    "name": "managed",
                    "namespace": "test-system",
                    "labels": {idle_case.MANAGED_LABEL: "true"},
                },
                "status": {"phase": "Succeeded"},
                "spec": {"nodeName": "node-a"},
            }
        ]
    elif fault == "canary":
        h.jobs = [{"metadata": {"name": f"{idle_case.COVERAGE_CANARY_JOB}-owned"}}]
    elif fault == "coverage":
        h.coverage_changes["heartbeat_supported"] = False
    elif fault == "watcher":
        h.watcher_ready = False
    else:
        h.tests_pass = False
    result = h.plan(tmp_path)
    assert result["errors"], result
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert "host.create" not in [name for name, _ in h.calls], h.calls


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gpu_inventory", [], "GPU inventory"),
        ("compute_clients", [{"pid": 111}], "active NVIDIA"),
        ("quiesce_states", ["existing"], "pre-existing quiesce"),
        ("kmsg_writable", False, "not writable"),
    ],
)
def test_host_baseline_refusal_never_injects_and_still_removes_probe(
    module: Any,
    field: str,
    value: Any,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.baseline[field] = value
    code, report = h.execute(tmp_path)
    assert code == 1 and message in report["error"], report
    names = [name for name, _ in h.calls]
    assert "host.write-xid46" not in names and "host.cleanup" in names, names


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
@pytest.mark.parametrize(
    "phase",
    [
        "host.create",
        "host.write-xid46",
        "workflow.wait",
        "provider.events",
        "host.cleanup",
    ],
)
def test_idle_runner_phase_failure_is_fail_and_cleanup_is_attempted(
    module: Any, phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert phase in str(report), report
    assert any(name == "host.cleanup" for name, _ in h.calls), h.calls
    if module is watcher_case and phase != "host.create":
        assert h.watcher_replicas == 1, report


@pytest.mark.parametrize("module", [reset_case, idle_case])
@pytest.mark.parametrize(
    "phase",
    ["host.start-reset-sampler", "host.stop-reset-sampler", "host.restore-quiesce"],
)
def test_reset_sampler_ack_loss_and_restore_failure_cannot_leave_a_pass(
    module: Any, phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = TimeoutError(f"fake timeout at {phase}")
    if module is reset_case and phase == "host.restore-quiesce":
        h.after["quiesce_states"] = [{"reset_issued": "unresolved"}]
    code, report = h.execute(tmp_path)
    assert code == 1, report
    names = [name for name, _ in h.calls]
    assert "host.stop-reset-sampler" in names and "host.cleanup" in names, names
    if phase == "host.start-reset-sampler":
        assert "host.write-xid46" not in names, names
    elif phase == "host.stop-reset-sampler":
        assert "sampler_cleanup_error" in report, report
    else:
        if module is reset_case:
            assert "host.restore-quiesce" not in names
            assert report["quiesce_recovery"]["operator_review_required"] is True
        else:
            assert "error" in report["quiesce_recovery"], report


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
def test_probe_residuals_make_an_otherwise_successful_run_fail(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.residuals["pod"] = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["probe_residuals"]["pod"] is True, report
    assert report["errors"] == [], report


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
@pytest.mark.parametrize("variation", ["provider", "cpu", "final-ready"])
def test_idle_runner_requires_unchanged_blast_radius_and_healthy_postflight(
    module: Any, variation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if variation == "provider":
        h.events = [{"event_name": "BatchRebootClusterNodes"}]
    elif variation == "cpu":
        h.cpu_after = {"nodes": ["unexpected"]}
    else:
        h.final_node_changes = {"ready": "False"}
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    if variation == "provider":
        assert any("provider mutation" in error for error in report["errors"]), report
    elif variation == "cpu":
        assert "control-plane EKS state differs from baseline" in report["errors"], (
            report
        )
    else:
        assert report["final_node"]["ready"] == "False", report


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
def test_idle_runner_honors_abort_after_injection_and_runs_finally(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["workflow.wait"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert "host.cleanup" in names, names
    if module is watcher_case:
        assert h.watcher_replicas == 1 and "watchdog.kill" in names, names
    else:
        assert "host.stop-reset-sampler" in names, names


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
@pytest.mark.parametrize("phase", ["before-probe", "host.snapshot"])
def test_idle_runner_expired_window_blocks_the_next_mutation(
    module: Any, phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if phase == "before-probe":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "window ended" in report["error"], report
    assert "host.write-xid46" not in [name for name, _ in h.calls], h.calls


def test_watcher_cleanup_keeps_watchdog_armed_when_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher_case, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["watcher.restore"] = RuntimeError("restore unavailable")
    code, report = h.execute(tmp_path)
    assert code == 1 and report["watchdog_left_armed"] is True, report
    assert report["node_left_isolated"] is True, report
    assert "restore unavailable" in report["watcher_restore_error"], report
    assert "watchdog.kill" not in [name for name, _ in h.calls], h.calls


@pytest.mark.parametrize("fails", [False, True])
def test_unexpected_quiesce_on_blocked_node_is_recovered_but_never_passes(
    fails: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher_case, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.after["quiesce_states"] = ["unexpected-quiesce"]
    if fails:
        h.failures["host.restore-quiesce"] = RuntimeError("restore failed")
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "a GPU quiesce state exists on the host" in report["errors"], report
    assert ("error" in report["quiesce_recovery"]) is fails, report
    assert any(name == "host.restore-quiesce" for name, _ in h.calls), h.calls


@pytest.mark.parametrize("module", [reset_case, idle_case, watcher_case])
def test_idle_postflight_rejects_node_recreation(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(module, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.final_node_changes = {"uid": "same-name-new-node"}
    code, report = h.execute(tmp_path)
    assert code == 1 and report["node_identity_changed"] is True, report


def test_watcher_cannot_scale_without_a_live_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher_case, tmp_path, monkeypatch)
    fixture = watcher_case.WatcherScaleFixture(
        h.regional, case_dir=tmp_path, baseline_replicas=1
    )
    with pytest.raises(RegionalFixtureError, match="before the watchdog"):
        fixture.scale_down()
    fixture.arm(120)
    h.process.returncode = 1
    with pytest.raises(RegionalFixtureError, match="exited"):
        fixture.scale_down()
    assert "watcher.down" not in [name for name, _ in h.calls], h.calls


def test_watcher_wait_timeout_preserves_need_for_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher_case, tmp_path, monkeypatch)
    h.watcher_stuck = True
    fixture = watcher_case.WatcherScaleFixture(
        h.regional, case_dir=tmp_path, baseline_replicas=1
    )
    fixture.arm(500)
    with pytest.raises(RegionalFixtureError, match="did not leave"):
        fixture.scale_down()
    assert fixture.scaled_down and not fixture.restored, h.calls
    h.watcher_stuck = False
    record = fixture.restore()
    assert fixture.restored and record["watchdog"]["disarmed"], record
    assert h.clock.elapsed == watcher_case.WATCHER_GONE_BUDGET_SECONDS, h.clock.elapsed


@pytest.mark.parametrize("fault", ["rollout", "readiness", "disarm"])
def test_watcher_does_not_mark_restore_complete_without_all_proofs(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher_case, tmp_path, monkeypatch)
    fixture = watcher_case.WatcherScaleFixture(
        h.regional, case_dir=tmp_path, baseline_replicas=1
    )
    fixture.arm(500)
    fixture.scale_down()
    if fault == "rollout":
        h.failures["watcher.rollout"] = RuntimeError("rollout failed")
    elif fault == "readiness":
        h.watcher_ready = False
    else:
        h.process.returncode = 1
    with pytest.raises((RuntimeError, RegionalFixtureError)):
        fixture.restore()
    assert fixture.restored is False, h.calls
    assert "watchdog.kill" not in [name for name, _ in h.calls], h.calls


def test_watchdog_stop_handles_absence_exit_timeout_and_kill_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(watcher_case, tmp_path, monkeypatch)
    assert watcher_case.stop_process_group(None) == {"armed": False}, (
        "an absent watchdog must be a no-op"
    )
    h.process.timeout_once = True
    result = watcher_case.stop_process_group(h.process)
    assert result == {"armed": True, "fired": False, "disarmed": True}, result
    kills = [detail for name, detail in h.calls if name == "watchdog.kill"]
    assert [entry["signal"] for entry in kills] == [signal.SIGTERM, signal.SIGKILL], (
        kills
    )
    result = watcher_case.stop_process_group(h.process)
    assert (
        result["fired"] is True and "descendants are unproved" in result["stop_error"]
    ), result
    h.process.returncode = None
    h.failures["watchdog.kill"] = PermissionError("fake access denied")
    result = watcher_case.stop_process_group(h.process)
    assert "PermissionError" in result["stop_error"], result


def test_coverage_poll_records_each_failed_sample_and_stops_at_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(idle_case, tmp_path, monkeypatch)
    with pytest.raises(RegionalFixtureError, match="coverage did not settle"):
        idle_case.wait_for_coverage(
            h.regional,
            "node-a",
            case_dir=tmp_path,
            judge=lambda sample: ["coverage unproven"],
            budget_seconds=30,
            label="unit",
        )
    timeline = json.loads((tmp_path / "coverage-unit-timeline.json").read_text())[
        "entries"
    ]
    assert len(timeline) == 3 and h.clock.elapsed == 30, timeline
    assert all(item["errors"] == ["coverage unproven"] for item in timeline), timeline


def test_cross_cluster_heartbeat_is_not_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = IdleHarness(idle_case, tmp_path, monkeypatch)
    h.coverage_changes["heartbeat"] = {"cluster_id": "foreign"}
    with pytest.raises(RegionalFixtureError, match="another cluster"):
        idle_case.coverage_probe(h.regional, "node-a")


def _dead_probe(
    h: IdleHarness, monkeypatch: pytest.MonkeyPatch, *, reboot_on_failure: bool
) -> list[str]:
    """Post-reset probe calls fail on a terminated Pod; return what was attempted.

    ``reboot_on_failure`` gives the node a new boot id at the moment the probe
    is found dead: the reboot the node snapshot before the call did not show.
    """

    attempted: list[str] = []
    original = h.host.execute

    def execute(action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if action == "stop-reset-sampler" or (action == "snapshot" and args):
            attempted.append(action)
            if reboot_on_failure:
                h.node["boot_id"] = "boot-after"
            raise HostProbeTransportError(POD_GONE)
        return original(action, *args, **kwargs)

    monkeypatch.setattr(h.host, "execute", execute)
    return attempted


def test_reset_escalated_to_a_reboot_reports_the_product_story_not_the_dead_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DESTR-001 on plain EKS: RESET_GPU failed busy and the product rebooted.

    The reboot ended the probe Pod and the sampler; the result used to say only
    ``HostProbeTransportError`` and lose the failed reset and its escalation.
    """

    h = IdleHarness(reset_case, tmp_path, monkeypatch)
    h.node["boot_id"] = "boot-before"
    h.plan(tmp_path)
    h.workflow = escalated_reset_state()
    original = h.regional.wait_for_workflow

    def rebooted(**kwargs: Any) -> dict[str, Any]:
        state = original(**kwargs)
        h.node["boot_id"] = "boot-after"
        return state

    monkeypatch.setattr(h.regional, "wait_for_workflow", rebooted)
    attempted = _dead_probe(h, monkeypatch, reboot_on_failure=False)
    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    first = report["errors"][0]
    assert first.startswith("RESET_GPU failed on node-a: ResetProgressError"), first
    assert "In use by another client" in first, first
    assert first.endswith("the product escalated the branch to RESTART_NODE"), first
    escalation = report["product_escalation"]
    assert escalation["escalations"][0]["from_operation"] == "RESET_GPU", escalation
    assert escalation["escalations"][0]["to_operation"] == "RESTART_NODE", escalation
    assert escalation["node_lifecycle_completed"] == ["RESTART_NODE"], escalation
    assert "error" not in report, "the dead probe must not be the case's story"
    assert "sampler_cleanup_error" not in report, report
    assert "host_after_error" not in report, report
    assert report["host_after_skipped"] == (
        "RESTART_NODE completed; boot id changed from boot-before to boot-after"
    ), report
    assert report["sampler_stopped_by_reboot"] is True, report
    assert any(
        error.startswith(
            "the physical single-GPU reset proof is unavailable because the "
            "product rebooted the node"
        )
        for error in report["errors"]
    ), report["errors"]
    assert report["incident_id"] == "inc-destr001-escalated", report
    assert report["workflow_request_id"] == "wf-destr001-escalated", report
    assert attempted == [], "neither the snapshot nor the stop may use the dead Pod"
    names = [name for name, _ in h.calls]
    assert names.count("host.snapshot") == 1, names
    for name in ("provider.events", "cpu.snapshot", "host.cleanup"):
        assert name in names, f"{name} still runs after the product reboot: {names}"
    assert report["reset_notification"]["notification_count"] == 0, report
    assert report["post_case_isolation"]["unschedulable"] is False, report


@pytest.mark.parametrize("reboot", [True, False])
def test_probe_transport_loss_is_secondary_only_after_a_reboot(
    reboot: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reboot rung never completed, so the probe is asked and found dead.

    A node that has a new boot by then turns the failure into the secondary
    ``host_after_error``; one that does not keeps it as ``error``. Either way the
    failed reset and its escalation were in ``errors`` before the probe call.
    """

    h = IdleHarness(reset_case, tmp_path, monkeypatch)
    h.node["boot_id"] = "boot-before"
    h.plan(tmp_path)
    h.workflow = escalated_reset_state(restart_status="FAILED")
    attempted = _dead_probe(h, monkeypatch, reboot_on_failure=reboot)
    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    assert report["errors"][0].startswith("RESET_GPU failed on node-a"), report
    assert report["product_escalation"]["node_lifecycle_completed"] == [], report
    assert attempted[0] == "snapshot", attempted
    failure = f"HostProbeTransportError: {POD_GONE}"
    if reboot:
        assert "error" not in report, report
        assert report["host_after_error"] == failure, report
        assert report["host_after_skipped"] == (
            "boot id changed from boot-before to boot-after"
        ), report
        assert report["sampler_stopped_by_reboot"] is True, report
        assert any(
            "because the node rebooted during the case" in error
            for error in report["errors"]
        ), report["errors"]
        assert attempted == ["snapshot"], attempted
    else:
        assert report["error"] == failure, report
        assert "host_after_error" not in report, report
        assert "host_after_skipped" not in report, report
        assert report["sampler_cleanup_error"] == failure, report
    assert "host.cleanup" in [name for name, _ in h.calls], h.calls


@pytest.mark.parametrize("owner", ["other", "case"])
def test_post_case_isolation_names_the_incident_still_holding_the_node(
    owner: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another incident re-cordoned and quarantined the node after the workflow."""

    h = IdleHarness(reset_case, tmp_path, monkeypatch)
    h.plan(tmp_path)
    incident = (
        "inc-drain-after-workflow-8de85ab8"
        if owner == "other"
        else h.workflow["incident"]["incident_id"]
    )
    quarantine = {
        "key": "gpu-fault.io/quarantined",
        "value": "incident-c57db5a2",
        "effect": "NoSchedule",
    }
    h.final_node_changes = {
        "unschedulable": True,
        "taints": [
            quarantine,
            {"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule"},
        ],
        "ownership_annotations": {
            "gpu-fault.io/fencing-token": "1",
            "gpu-fault.io/incident-id": incident,
        },
    }
    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    isolation = report["post_case_isolation"]
    assert isolation == {
        "unschedulable": True,
        "taints": [quarantine],
        "ownership_incident": incident,
        "owned_by_case_incident": owner == "case",
        "other_incident_owner": incident if owner == "other" else None,
    }, isolation
    holder = "another incident" if owner == "other" else "this case's incident"
    assert report["errors"] == [
        "target node is still isolated at the end of the case: owned by "
        f"{holder} {incident}; taint "
        "gpu-fault.io/quarantined=incident-c57db5a2:NoSchedule; unschedulable"
    ], report["errors"]
    assert report["final_node"]["ownership_annotations"][
        "gpu-fault.io/incident-id"
    ] == (incident), report
