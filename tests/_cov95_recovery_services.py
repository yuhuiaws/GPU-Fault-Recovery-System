from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    TerminalStatus,
    WorkflowOperation,
)
from gpu_fault.service import CompletionService
from gpu_fault.watcher import AllocationCompleteness, FailureDetectedEvent
from tests._builders import fault_incident, workflow_request, workflow_step


def profile_without(*capabilities):
    profile = default_simulated_profile()
    return profile.model_copy(
        update={
            "capabilities": [
                claim
                for claim in profile.capabilities
                if claim.capability not in capabilities
            ]
        }
    )


def marker_for(event, action, **changes):
    return NodeMarker(
        **{
            "marker_id": "unit-marker",
            "source": "unit",
            "trusted": True,
            "incident_id": "unit-incident",
            "cluster_id": event.cluster_id,
            "observed_at": event.ended_at,
            "expires_at": event.ended_at + timedelta(hours=1),
            "scope": MarkerScope(node_ids=["node-a"], gpu_uuids=["GPU-a", "GPU-a"]),
            "severity": Severity.CRITICAL,
            "recommended_action": action,
            "mapping_version": "unit",
            **changes,
        }
    )


def failure_for(event, **changes):
    return FailureDetectedEvent(
        **{
            "cluster_id": event.cluster_id,
            "job_id": event.job_id,
            "attempt_id": event.attempt_id,
            "detected_at": event.ended_at,
            "runtime_profile_version": event.runtime_profile_version,
            "workload_ids": ["training/job/unit"],
            "node_ids": ["node-a"],
            "first_failed_rank": 0,
            "node_id": "node-a",
            "exit_code": 1,
            "reason": "unit failed rank",
            "allocation_completeness": AllocationCompleteness.COMPLETE,
            **changes,
        }
    )


def active_recovery(store, event, name="unit", *, incident_values=None, **changes):
    incident = fault_incident(
        f"{name}-incident",
        f"{name}-event",
        event_type="TRAINING_ATTEMPT_TERMINAL",
        cluster_id=event.cluster_id,
        job_id=event.job_id,
        attempt_id=event.attempt_id,
        state=IncidentState.ACTION_PENDING,
        effective_action=RecoveryAction.RESTART_WORKLOAD,
        workflow_request_id=f"{name}-workflow",
    ).model_copy(update=incident_values or {})
    workflow = workflow_request(
        f"{name}-workflow",
        incident.incident_id,
        official_steps=[workflow_step(WorkflowOperation.RESTART_WORKLOAD)],
    ).model_copy(update=changes)
    store.save_incident_and_workflow(incident, workflow)
    return incident, store.get_workflow(workflow.request_id)


def exercise_failed_withdrawal(store, event, monkeypatch, boundary):
    store.save_profile(profile_without())
    _, workflow = active_recovery(store, event)
    original = getattr(store, boundary)

    def unavailable(*args, **kwargs):
        raise OSError("synthetic withdrawal storage failure")

    stopped = event.model_copy(update={"terminal_status": TerminalStatus.STOPPED})
    service = CompletionService(store)
    with monkeypatch.context() as local:
        local.setattr(store, boundary, unavailable)
        with pytest.raises(OSError, match="withdrawal storage failure"):
            service.handle_terminal(stopped)
        assert store.get_decision_by_event(stopped.event_key) is None, (
            "a failed withdrawal must not become a cached successful terminal decision"
        )
        assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None, (
            "the failed write must remain pending for retry"
        )
        local.setattr(store, boundary, original)
        result = service.handle_terminal(stopped)
    assert result.status is DecisionStatus.NO_ACTION, (
        "the completed retry may finally record the terminal decision"
    )
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is not None, (
        "retry must finish withdrawal before suppressing subsequent terminal deliveries"
    )
