"""Failure and retry boundaries of the inventory recovery state machine."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.e2e.regional import collector_inventory_reboot as runner
from scripts.e2e.regional.collector_acceptance_fixture import CollectorAcceptanceFixture
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._collector_inventory_reboot_support import (
    APPLIED,
    BASELINE,
    INTENT,
    InventoryHarness,
)
from tests.regional.test_collector_inventory_reboot_runner import recovery


def arm(subject: runner.InventoryRecovery) -> None:
    subject.accept_arming(
        {
            "run_id": "run-a",
            "cluster_id": "cluster-a",
            "node_id": "node-a",
            "boot_id": "boot-a",
            "mutation_started": True,
            "timer_armed": True,
            "boot_restore_armed": True,
            "baseline_sha256": BASELINE,
            "applied_sha256": APPLIED,
            "intent_sha256": INTENT,
        }
    )


@pytest.mark.parametrize(
    "patch,expected",
    [
        ({}, True),
        ({"ready": "False"}, True),
        ({"ready": "Unknown"}, True),
        ({"ready": None}, None),
        ({"boot_id": "boot-b"}, True),
        ({"boot_id": ""}, None),
    ],
)
def test_reboot_transition_requires_known_node_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    patch: dict[str, Any],
    expected: bool | None,
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    arm(subject)
    harness.regional.node_snapshot = lambda *args: {
        "uid": "node-uid-a",
        "name": "node-a",
        "boot_id": "boot-a",
        "ready": "True",
        **patch,
    }
    if expected is None:
        with pytest.raises(RegionalFixtureError, match="incomplete"):
            subject.reboot_wait_authorized()
    else:
        assert subject.reboot_wait_authorized() is expected


@pytest.mark.parametrize(
    "problem", ["unarmed", "submissions", "workflows", "submission-proof"]
)
def test_unproven_provider_reboot_does_not_authorize_transport_deferral(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    subject.bind_finding(harness.records())
    if problem != "unarmed":
        arm(subject)
    state = harness.state()
    if problem == "submissions":
        state["submissions"] = []
    elif problem == "workflows":
        state["workflows"].append({**state["workflows"][0], "request_id": "workflow-b"})
        state["node_workflow_ids"].append("workflow-b")
    elif problem == "submission-proof":
        monkeypatch.setattr(
            runner, "submitted_reboot_errors", lambda **kwargs: ["unproven"]
        )
    original = harness.regional.cpu_python
    harness.regional.cpu_python = lambda script, *args: (
        state if script == runner.INVENTORY_RECOVERY_STATE else original(script, *args)
    )
    assert subject.reboot_wait_authorized() is False


def test_node_replacement_during_transport_failure_is_a_hard_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    arm(subject)
    harness.regional.node_snapshot = lambda *args: {"uid": "replacement"}
    with pytest.raises(RegionalFixtureError, match="UID changed"):
        subject.reboot_wait_authorized()


def test_different_threshold_event_cannot_take_over_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    subject.bind_finding(harness.records())
    records = harness.records()
    records[-1]["batch_id"] = "host-node-a-later"
    records[-1]["record_id"] = "host-telemetry/host-node-a-later"
    with pytest.raises(RegionalFixtureError, match="finding identity changed"):
        subject.bind_finding(records)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", "other"),
        ("workflows", []),
        ("workflows", [{"request_id": ""}]),
        ("node_workflow_ids", None),
        ("node_workflow_ids", ["workflow-a", "workflow-a"]),
        ("node_workflow_ids", ["other"]),
        ("node_workflow_ids", [None]),
    ],
)
def test_incomplete_recovery_snapshots_stay_unresolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: Any
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    subject.bind_finding(harness.records())
    state = deepcopy(harness.state())
    state[field] = value
    harness.regional.cpu_python = lambda *args: state
    with pytest.raises(RegionalFixtureError, match="incomplete"):
        subject.read()


def test_running_workflow_retains_a_fixed_wait_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    subject.bind_finding(harness.records())
    state = harness.state()
    state["workflows"][0]["status"] = "RUNNING"
    harness.regional.cpu_python = lambda *args: state
    assert subject.wait(terminal=False, timeout_seconds=10)["status"] == "RUNNING"
    with pytest.raises(RegionalFixtureError, match="did not settle"):
        subject.wait(terminal=True, timeout_seconds=10)
    assert harness.clock.sleeps == [5, 5]


@pytest.mark.parametrize("changed", [False, True])
def test_rejected_setup_can_only_release_its_hold_with_verified_untouched_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: bool
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    receipt = {
        "run_id": "run-a",
        "state": "NOT_STARTED",
        "mutation_started": False,
        "no_mutation": True,
        "restored": False,
        "cleanup_verified": True,
        "timer_disarmed": True,
        "recovery_stopped": True,
    }
    monkeypatch.setattr(
        runner, "restore_collector_env", lambda *args, **kwargs: receipt
    )
    if changed:
        harness.collector.snapshot = lambda: {
            "collector_env_file": {"sha256": APPLIED},
            "boot_id": "boot-a",
        }
        with pytest.raises(RegionalFixtureError, match="changed its baseline"):
            subject.restore(cast(CollectorAcceptanceFixture, harness.collector))
        with pytest.raises(RegionalFixtureError, match="operator hold"):
            subject.read_for_cleanup()
    else:
        assert (
            subject.restore(cast(CollectorAcceptanceFixture, harness.collector))
            == receipt
        )
        assert subject.read_for_cleanup() == {
            "incidents": [],
            "workflows": [],
            "commands": [],
        }
        assert "recovery" not in harness.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "foreign"),
        ("restored", False),
        ("cleanup_verified", False),
        ("timer_disarmed", False),
        ("state", "RESTORED"),
        ("baseline_sha256", APPLIED),
        ("intent_sha256", "d" * 64),
    ],
)
def test_partial_or_foreign_restore_receipt_never_releases_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: Any
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    arm(subject)
    receipt = {
        "run_id": "run-a",
        "restored": True,
        "cleanup_verified": True,
        "timer_disarmed": True,
        "state": "CLEANED",
        "baseline_sha256": BASELINE,
        "intent_sha256": INTENT,
        field: value,
    }
    monkeypatch.setattr(
        runner, "restore_collector_env", lambda *args, **kwargs: receipt
    )
    with pytest.raises(RegionalFixtureError, match="restoration is not verified"):
        subject.restore(cast(CollectorAcceptanceFixture, harness.collector))
    with pytest.raises(RegionalFixtureError, match="operator hold"):
        subject.read_for_cleanup()


@pytest.mark.parametrize(
    "field,value",
    [
        ("uid", "replacement"),
        ("boot_id", "other-boot"),
        ("ready", "False"),
        ("unschedulable", True),
        ("taints", [{"key": "foreign"}]),
        ("ownership_annotations", {"owner": "other"}),
    ],
)
def test_target_drift_before_injection_does_not_authorize_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: Any
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    baseline = harness.regional.node_snapshot("node-a")
    harness.regional.node_snapshot = lambda *args: {**baseline, field: value}
    with pytest.raises(RegionalFixtureError, match="not idle and unchanged"):
        harness.run()
    assert harness.run_id == "" and harness.restores == 0


def test_scope_drift_after_preflight_cannot_begin_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "capture_reboot_scope",
        lambda *args, **kwargs: {**harness.scope, "node_uid": "replacement"},
    )
    with pytest.raises(RegionalFixtureError, match="changed after preflight"):
        harness.run()
    assert harness.run_id == "" and harness.restores == 0


def test_missing_final_submission_is_not_a_completed_reboot_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    original = harness.regional.cpu_python

    def cpu(script: str, *args: str) -> dict[str, Any]:
        state: dict[str, Any] = original(script, *args)
        if script == runner.INVENTORY_RECOVERY_STATE:
            state["submissions"] = []
        return state

    harness.regional.cpu_python = cpu
    result = harness.run()
    assert result["verdict"] == "FAIL", result
    assert "submission is not unique" in result["errors"][0], result["errors"]
    # Isolated sampling never touches the production collector configuration,
    # so a failed proof leaves nothing to restore; the hold is released by the
    # normal guarded cleanup, not by a collector.env undo.
    assert harness.restores == 0, harness.calls
    assert "restore-collector-env" not in harness.calls, harness.calls
    assert result["production_configuration_modified"] is False, result
