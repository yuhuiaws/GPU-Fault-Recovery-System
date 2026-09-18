from __future__ import annotations

import hashlib
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from scripts.e2e.regional.collector_acceptance_fixture import CollectorAcceptanceFixture
from scripts.e2e.regional.collector_case_cleanup import CaseCleanup
from scripts.e2e.regional.collector_recovery_safety import require_settled_recovery
from scripts.e2e.regional.probes import destructive_node_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def state():
    return {
        "seed_marker": "owned-marker",
        "incidents": [{"incident_id": "incident"}],
        "workflows": [{"status": "SUCCEEDED", "step_executions": []}],
        "commands": [{"status": "SUCCEEDED", "result_details": {}}],
        "events": [{"event_id": "owned-event"}],
    }


@pytest.mark.parametrize(
    "defect",
    [
        "leased",
        "cancelled",
        "expired",
        "cancel-source",
        "unknown",
        "legacy-reset",
        "nested",
        "nested-outer",
        "compound",
        "compound-waiting",
        "operator",
        "legacy-blocked",
        "missing",
        "bad-details",
    ],
)
def test_every_shared_cleanup_entry_preserves_unknown_state_and_never_restores_host(
    defect,
):
    snapshot = state()
    command = snapshot["commands"][0]
    if defect in {"leased", "cancelled", "expired"}:
        command["status"] = defect.upper()
    elif defect == "cancel-source":
        command.update(
            status="FAILED", status_source="workflow-timeout", last_lease_owner="old"
        )
    elif defect in {"unknown", "legacy-reset"}:
        command["status"] = "FAILED"
        command["result_details"] = (
            {"outcome_unknown": True}
            if defect == "unknown"
            else {"reset_outcome_unknown": ["GPU-a"]}
        )
    elif defect in {"nested", "compound"}:
        key = "node_results" if defect == "nested" else BATCHED_RESULTS_KEY
        command["result_details"] = {
            key: {"node": {"details": {"outcome_unknown": True}}}
        }
    elif defect == "nested-outer":
        command["result_details"] = {
            "node_results": {"node": {"outcome_unknown": True, "details": {}}}
        }
    elif defect == "compound-waiting":
        command["result_details"] = {
            BATCHED_RESULTS_KEY: {"0": {"status": "WAITING", "details": {}}}
        }
    elif defect in {"operator", "legacy-blocked"}:
        snapshot["workflows"][0].update(status="BLOCKED")
        if defect == "operator":
            snapshot["workflows"][0]["blocked_kind"] = "NEEDS_OPERATOR"
    elif defect == "missing":
        snapshot.pop("commands")
    else:
        command["result_details"] = None
    with pytest.raises(RegionalFixtureError):
        require_settled_recovery(snapshot)
    host = Mock()
    fixture = SimpleNamespace(
        node="node-a",
        store_snapshot=lambda marker: deepcopy(snapshot),
        restore_incidents=Mock(return_value=[]),
    )
    tracker = CaseCleanup()
    tracker.register_seed(fixture, "owned-marker", quiesce_host=host)
    tracker.register_state(fixture, snapshot)
    result = tracker.finish(profile_version="profile", reason="unit")
    assert result["errors"], "unsettled evidence must prevent cleanup success"
    fixture.restore_incidents.assert_not_called()
    host.execute.assert_not_called()
    assert tracker.seed_markers and tracker.incident_states


def test_shared_cleanup_only_reads_host_state_before_product_validated_restoration():
    snapshot = state()
    host = Mock()
    host.execute.return_value = {"quiesce_states": []}
    fixture = SimpleNamespace(
        node="node-a",
        store_snapshot=lambda marker: deepcopy(snapshot),
        restore_incidents=Mock(return_value=[{"status": "SUCCEEDED"}]),
    )
    tracker = CaseCleanup()
    tracker.register_seed(fixture, "owned-marker", quiesce_host=host)
    tracker.register_state(fixture, snapshot)
    result = tracker.finish(profile_version="profile", reason="unit")
    assert result["errors"] == []
    host.execute.assert_called_once_with("snapshot", timeout=180)
    fixture.restore_incidents.assert_called_once()
    assert tracker.incident_states == [] and tracker.seed_markers == []


def test_product_quiesce_residual_never_gets_cleared_by_shared_cleanup():
    snapshot = state()
    host = Mock()
    host.execute.return_value = {"quiesce_states": [{"reset_issued": "unknown"}]}
    fixture = SimpleNamespace(
        store_snapshot=lambda marker: deepcopy(snapshot), restore_incidents=Mock()
    )
    tracker = CaseCleanup()
    tracker.register_seed(fixture, "owned-marker", quiesce_host=host)
    with pytest.raises(RegionalFixtureError, match="quiesce restoration is unproven"):
        tracker.restore(fixture, snapshot, profile_version="profile", reason="unit")
    host.execute.assert_called_once_with("snapshot", timeout=180)
    fixture.restore_incidents.assert_not_called()


def test_direct_collector_fixture_rechecks_current_unknown_evidence(monkeypatch):
    initial = state()
    current = state()
    current["commands"][0].update(
        status="FAILED", result_details={"outcome_unknown": True}
    )
    fixture = CollectorAcceptanceFixture.__new__(CollectorAcceptanceFixture)
    fixture.store_snapshot = Mock(return_value=current)
    with pytest.raises(RegionalFixtureError, match="outcome is unresolved"):
        fixture.restore_incidents(initial, profile_version="profile", reason="unit")
    fixture.store_snapshot.assert_called_once_with("owned-marker")


def test_shared_refresh_cannot_erase_a_previously_observed_incident():
    previous = state()
    missing = state()
    missing["incidents"] = []
    fixture = SimpleNamespace(
        store_snapshot=lambda marker: missing, restore_incidents=Mock()
    )
    tracker = CaseCleanup()
    with pytest.raises(RegionalFixtureError, match="lost a tracked identity"):
        tracker.restore(fixture, previous, profile_version="profile", reason="unit")
    fixture.restore_incidents.assert_not_called()


@pytest.mark.parametrize(
    "condition", ["known", "operator", "uncertain-failure", "unrestored-quiesce"]
)
def test_cpu_restore_creation_rechecks_product_state_before_creating_a_successor(
    monkeypatch, condition
):
    import sys

    from gpu_fault.app import ApplicationContext
    from gpu_fault.models import (
        BlockedKind,
        WorkflowOperation,
        WorkflowStatus,
        WorkflowStepStatus,
    )
    from scripts.e2e.regional.warm_spare_fixture import CREATE_RESTORE_WORKFLOW
    from tests._builders import (
        build_store,
        fault_incident,
        workflow_request,
        workflow_step,
        workflow_step_execution,
    )

    store = build_store()
    operation = (
        WorkflowOperation.QUIESCE_GPU_SERVICES
        if condition == "unrestored-quiesce"
        else WorkflowOperation.RESET_GPU
    )
    incident = fault_incident(
        "incident", "event", workflow_request_id="workflow", fencing_token=3
    )
    workflow = workflow_request(
        "workflow",
        "incident",
        status=WorkflowStatus.BLOCKED
        if condition == "operator"
        else WorkflowStatus.FAILED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR if condition == "operator" else None,
        official_steps=[workflow_step(operation)],
        completed_step_indexes=[0] if condition == "unrestored-quiesce" else [],
        completed_operations=[operation] if condition == "unrestored-quiesce" else [],
        step_executions=[
            workflow_step_execution(
                0,
                operation,
                details={"outcome_unknown": True}
                if condition == "uncertain-failure"
                else {},
            ).model_copy(
                update={
                    "status": WorkflowStepStatus.FAILED
                    if condition == "uncertain-failure"
                    else WorkflowStepStatus.SUCCEEDED
                }
            )
        ],
    )
    store.save_incident_and_workflow(incident, workflow)
    wake = Mock()
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        lambda: SimpleNamespace(store=store, dispatcher=SimpleNamespace(wake=wake)),
    )
    monkeypatch.setattr(sys, "argv", ["probe", "incident", "node-a", "profile", "unit"])
    if condition == "known":
        exec(compile(CREATE_RESTORE_WORKFLOW, "<owned-restore-probe>", "exec"), {})
        assert len(store.list_workflows()) == 2
        wake.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="operator|quiesce|outcome"):
            exec(compile(CREATE_RESTORE_WORKFLOW, "<owned-restore-probe>", "exec"), {})
        assert len(store.list_workflows()) == 1
        wake.assert_not_called()


@pytest.mark.parametrize("present", [False, True])
def test_legacy_host_restore_command_is_read_only_and_preserves_failsafe(
    tmp_path, monkeypatch, present
):
    monkeypatch.setattr(probe, "QUIESCE_STATE_DIR", tmp_path)
    monkeypatch.setattr(
        probe, "run", Mock(side_effect=AssertionError("no service commands"))
    )
    emitted = []
    monkeypatch.setattr(probe, "emit", emitted.append)
    digest = hashlib.sha256(b"owned-incident").hexdigest()[:20]
    path = tmp_path / f"quiesce-{digest}.json"
    original = '{"reset_issued":"unknown","failsafe":"armed"}'
    if present:
        path.write_text(original)
        with pytest.raises(probe.ProbeError, match="product-owned"):
            probe.restore_quiesce(SimpleNamespace(incident_id="owned-incident"))
        assert path.read_text() == original
        assert emitted == []
    else:
        probe.restore_quiesce(SimpleNamespace(incident_id="owned-incident"))
        assert emitted == [
            {
                "restore_attempted": False,
                "state_absent": True,
                "physical_outcome_proven": False,
            }
        ]
    probe.run.assert_not_called()
