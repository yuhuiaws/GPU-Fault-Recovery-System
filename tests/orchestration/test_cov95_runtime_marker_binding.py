from __future__ import annotations

import pytest

from gpu_fault.models import (
    RecoveryAction,
    TerminalStatus,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.service import CompletionService
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident, workflow_request, workflow_step
from tests._cov95_recovery_services import marker_for, profile_without


@pytest.mark.parametrize("incident_attempt", [None, "attempt-foreign"])
def test_local_marker_cannot_borrow_or_retire_another_clusters_incident(
    failed_event, incident_attempt, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    incident = fault_incident(
        "foreign-repair",
        "foreign-event",
        cluster_id="other-cluster",
        attempt_id=incident_attempt,
        workflow_request_id="foreign-workflow",
    )
    workflow = workflow_request(
        "foreign-workflow",
        incident.incident_id,
        WorkflowStatus.SUCCEEDED,
        official_steps=[workflow_step(WorkflowOperation.RESTART_WORKLOAD)],
    )
    store.save_incident_and_workflow(incident, workflow)
    marker = marker_for(
        failed_event,
        RecoveryAction.REBOOT_NODE,
        cluster_id=failed_event.cluster_id,
        incident_id=incident.incident_id,
    )
    service = CompletionService(store)
    service.add_marker(marker)
    before = store.get_workflow(workflow.request_id)
    before_incident = store.get_incident(incident.incident_id)
    plans = []
    original_save_plan = store.save_plan

    def save_plan(plan):
        plans.append(plan)
        return original_save_plan(plan)

    monkeypatch.setattr(store, "save_plan", save_plan)
    with pytest.raises(ValueError, match="marker incident.*cluster"):
        service.handle_terminal(failed_event)
    assert store.get_decision_by_event(failed_event.event_key) is None, (
        "contradictory marker identity must not cache an automatic decision"
    )
    assert plans == [], (
        "contradictory evidence must not fall back to an automatic restart"
    )
    assert store.list_markers() == [marker], (
        "foreign incident markers must not be retired"
    )
    assert store.get_workflow(workflow.request_id) == before, (
        "foreign workflow must remain unchanged"
    )
    assert store.get_incident(incident.incident_id) == before_incident, (
        "foreign incident must remain unchanged"
    )


def test_marker_incident_cluster_is_rechecked_before_planning(
    failed_event, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    incident = fault_incident(
        "local-repair",
        "local-event",
        cluster_id=failed_event.cluster_id,
        attempt_id=failed_event.attempt_id,
    )
    store.save_incident(incident)
    service = CompletionService(store)
    service.add_marker(
        marker_for(
            failed_event,
            RecoveryAction.REBOOT_NODE,
            cluster_id=failed_event.cluster_id,
            incident_id=incident.incident_id,
        )
    )
    reads = 0
    plans = []
    original = store.get_incident
    original_save_plan = store.save_plan

    def read(incident_id: str):
        nonlocal reads
        value = original(incident_id)
        if incident_id == incident.incident_id:
            reads += 1
            if reads > 1:
                return value.model_copy(update={"cluster_id": "other-cluster"})
        return value

    def save_plan(plan):
        plans.append(plan)
        return original_save_plan(plan)

    monkeypatch.setattr(store, "get_incident", read)
    monkeypatch.setattr(store, "save_plan", save_plan)
    with pytest.raises(ValueError, match="marker incident.*cluster"):
        service.handle_terminal(failed_event)
    assert reads == 2, (
        "both marker liveness and plan construction must check the incident"
    )
    assert store.get_decision_by_event(failed_event.event_key) is None, (
        "a changed incident binding must not leave a cached completion decision"
    )
    assert plans == [], "no recovery plan may rely on foreign incident state"


@pytest.mark.parametrize("lookup", ["explicit", "event-index"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cluster_id", "other-cluster"),
        ("job_id", "other-job"),
        ("job_id", None),
        ("attempt_id", "other-attempt"),
        ("attempt_id", None),
    ],
)
def test_stopped_terminal_cannot_borrow_unbound_or_foreign_passive_containment(
    failed_event, lookup: str, field: str, value: str | None
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    identity = {
        "cluster_id": failed_event.cluster_id,
        "job_id": failed_event.job_id,
        "attempt_id": failed_event.attempt_id,
        field: value,
    }
    event_key = (
        f"{failed_event.cluster_id}/{failed_event.attempt_id}/TrainingAttemptFailureDetected"
        if lookup == "event-index"
        else "other-containment-event"
    )
    incident = fault_incident(
        "passive-incident",
        event_key,
        "TRAINING_ATTEMPT_FAILURE_DETECTED",
        workflow_request_id="passive-workflow",
        **identity,
    )
    workflow = workflow_request(
        "passive-workflow",
        incident.incident_id,
        WorkflowStatus.SUCCEEDED,
        official_steps=[workflow_step(WorkflowOperation.STOP_WORKLOADS)],
    )
    store.save_incident_and_workflow(incident, workflow)
    stopped = failed_event.model_copy(
        update={
            "terminal_status": TerminalStatus.STOPPED,
            "termination_initiator_incident_id": (
                incident.incident_id if lookup == "explicit" else None
            ),
        }
    )
    service = CompletionService(store)
    before = store.get_workflow(workflow.request_id)
    with pytest.raises(ValueError, match="passive containment.*identity"):
        service.handle_terminal(stopped)
    assert store.get_decision_by_event(stopped.event_key) is None, (
        "mismatched containment must not cache a decision or authorize a restart"
    )
    assert store.get_workflow(workflow.request_id) == before, (
        "another job's containment workflow must not be modified"
    )
