from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.dcgm_diagnostic_analysis import extract_dcgm_diagnostic_findings
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration import IncidentOrchestrator, placement_hold
from gpu_fault.store import NotFoundError
from gpu_fault.training_health import TrainingHealthService, training_health_signal_key
from tests.orchestration._cov95_orch_extra_support import (
    NOW,
    finding,
    memory_store,
    observation,
    stored_workflow,
)
from tests.orchestration.test_cov95_orch_extra_training import heartbeat

LEGACY_HOLD = "hold-cluster-a-unit-attempt"


def seed_legacy_hold(store, orchestrator, monkeypatch):
    stored_workflow(store, [WorkflowOperation.RESTART_NODE], identity="repair-a")
    with monkeypatch.context() as old_writer:
        old_writer.setattr(
            placement_hold, "hold_incident_id", lambda *_args: LEGACY_HOLD
        )
        pair = orchestrator.hold_attempt_on_repairing_nodes(observation())
    assert pair is not None, (
        "legacy fixture must be created through the public hold API"
    )
    incident = store.get_incident(pair[0].incident_id)
    workflow = store.get_workflow(pair[1].request_id)
    store.save_workflow(
        workflow.model_copy(
            update={"status": WorkflowStatus.SUPERSEDED, "superseded_step_indexes": [0]}
        ),
        expected=workflow,
    )
    store.save_incident(
        incident.model_copy(update={"state": IncidentState.RECOVERED}),
        expected=incident,
    )
    return store.get_incident(incident.incident_id), store.get_workflow(
        workflow.request_id
    )


def test_verified_terminal_legacy_hold_still_suppresses_a_duplicate_after_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = memory_store()
    orchestrator = IncidentOrchestrator(store)
    incident, workflow = seed_legacy_hold(store, orchestrator, monkeypatch)
    before = store.list_workflows()
    assert orchestrator.hold_attempt_on_repairing_nodes(observation()) is None, (
        "a completed hold for this exact attempt must not restart its busy window"
    )
    assert store.list_workflows() == before, "upgrade must not create a second hold"
    assert store.get_incident(incident.incident_id) == incident, incident
    assert store.get_workflow(workflow.request_id) == workflow, workflow
    canonical = placement_hold.hold_incident_id("cluster-a", "unit-attempt")
    assert (
        canonical != LEGACY_HOLD and store.get_incident_by_event(canonical) is None
    ), "legacy recognition must not rewrite or alias the historical event"


def test_foreign_legacy_collision_cannot_consume_the_new_canonical_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = memory_store()
    orchestrator = IncidentOrchestrator(store)
    incident, workflow = seed_legacy_hold(store, orchestrator, monkeypatch)
    stored_workflow(
        store,
        [WorkflowOperation.RESTART_NODE],
        identity="repair-b",
        incident_updates={"cluster_id": "cluster"},
    )
    held = orchestrator.hold_attempt_on_repairing_nodes(
        observation(cluster_id="cluster", attempt_id="a-unit-attempt")
    )
    assert held is not None, "an unrelated legacy key cannot suppress protection"
    assert held[0].cluster_id == "cluster" and held[0].attempt_id == "a-unit-attempt", (
        held
    )
    assert held[0].incident_id != LEGACY_HOLD, held
    assert store.get_incident(incident.incident_id) == incident, (
        "foreign incident changed"
    )
    assert store.get_workflow(workflow.request_id) == workflow, (
        "foreign workflow changed"
    )


@pytest.mark.parametrize(
    "defect", ["no-pointer", "missing-row", "foreign-pointer", "wrong-kind"]
)
def test_incomplete_or_contradictory_legacy_hold_requires_reconciliation(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = memory_store()
    orchestrator = IncidentOrchestrator(store)
    incident, workflow = seed_legacy_hold(store, orchestrator, monkeypatch)
    if defect == "wrong-kind":
        store.save_workflow(
            workflow.model_copy(update={"placement_hold": False}), expected=workflow
        )
    else:
        pointer = None if defect == "no-pointer" else "missing-workflow"
        if defect == "foreign-pointer":
            _, foreign = stored_workflow(
                store,
                [WorkflowOperation.FREEZE_EVIDENCE],
                identity="foreign",
                incident_updates={"cluster_id": "cluster-b"},
                workflow_updates={"status": WorkflowStatus.SUCCEEDED},
            )
            pointer = foreign.request_id
        store.save_incident(
            incident.model_copy(update={"workflow_request_id": pointer}),
            expected=incident,
        )
    before = store.list_workflows()
    with pytest.raises((ValueError, NotFoundError)):
        orchestrator.hold_attempt_on_repairing_nodes(observation())
    assert store.list_workflows() == before, (
        "an unverifiable legacy hold must not authorize or suppress work as if it were valid"
    )


def legacy_training_finding():
    return finding(
        f"training-nonfinite-loss-unit-attempt-rank-0-{int(NOW.timestamp())}",
        cluster_id="cluster-a",
        node_id="node-0",
        category=NodeHealthCategory.TRAINING,
        metric_name="training_nonfinite-loss",
        evidence_ref="training-progress://unit-attempt/rank-0",
    )


def test_training_upgrade_does_not_reuse_a_foreign_legacy_incident() -> None:
    store = memory_store()
    orchestrator = IncidentOrchestrator(store)
    old_finding = legacy_training_finding()
    old_incident, old_workflow = orchestrator.ingest_node_health(old_finding)
    assert old_workflow is not None, "legacy fixture needs a real persisted workflow"
    before_incident = store.get_incident(old_incident.incident_id)
    before_workflow = store.get_workflow(old_workflow.request_id)
    store.save_attempt_observation(observation(cluster_id="cluster-b"))
    [new_finding] = (
        TrainingHealthService(store)
        .ingest(heartbeat(cluster_id="cluster-b", numerical_error=True))
        .findings
    )
    assert new_finding.event_id != old_finding.event_id, new_finding
    new_incident, _ = orchestrator.ingest_node_health(new_finding)
    assert new_incident.cluster_id == "cluster-b", new_incident
    assert new_incident.incident_id != old_incident.incident_id, new_incident
    assert store.get_incident(old_incident.incident_id) == before_incident, old_incident
    assert store.get_workflow(old_workflow.request_id) == before_workflow, old_workflow


def test_legacy_training_notification_latch_remains_effective_for_new_event_ids() -> (
    None
):
    store = memory_store()
    service = TrainingHealthService(store)
    old_finding = legacy_training_finding()
    key = training_health_signal_key(old_finding)
    assert key is not None, "legacy evidence reference must still identify its latch"
    assert store.claim_health_signal_transition(key, True, NOW), (
        "the simulated old writer must create its signal before notification"
    )
    service.mark_notified([old_finding])
    result = service.ingest(
        heartbeat(numerical_error=True, observed_at=NOW + timedelta(seconds=1))
    )
    assert result.findings == [], (
        "ID migration must not re-notify a latched active signal"
    )
    assert store.get_health_signal_state(key).notified is True, key


def test_nested_dcgm_message_arrays_preserve_text_without_promoting_metadata() -> None:
    [result] = extract_dcgm_diagnostic_findings(
        {
            "name": "unit-check",
            "status": "WARN",
            "warnings": [
                "temperature high",
                {"warning": "ECC evidence", "error_code": 17, "gpu_id": 0},
                ["nested message", None],
            ],
        }
    )
    assert result["messages"] == [
        "temperature high",
        "ECC evidence",
        "nested message",
    ], result
    assert result["error_codes"] == ["17"] and result["entities"] == ["0"], result
