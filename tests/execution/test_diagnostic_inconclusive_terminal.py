"""A failed workflow that never touched the node ends RECOVERED, not ESCALATED.

The node-health ``RUN_DIAGNOSTICS`` workflow is ``FREEZE_EVIDENCE`` then
``VALIDATE_HOST``: nothing before the validation changes the node. When the
validation failed, the generic failure derivation made the incident ESCALATED
with no follow-up step, and the marker that opened it stayed live until its
TTL -- a node nobody repaired read as "under remediation" for an hour.

Such a failure is a diagnostic that did not conclude. The incident ends
RECOVERED with the reason spelled out, its markers are retired, and one
advisory notification tells the operator the diagnostic was inconclusive.
Any workflow that completed, failed or even planned a node-changing or
isolating operation keeps the ESCALATED / QUARANTINED verdict.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.notifications import DiagnosticInconclusiveEmailBuilder
from tests._builders import active_workflow_executor, build_store, execute_workflow
from tests.execution._support import FakeAdapter, workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
VALIDATE_HOST = WorkflowOperation.VALIDATE_HOST
VALIDATE_GPU = WorkflowOperation.VALIDATE_GPU
RESET = WorkflowOperation.RESET_GPU
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


def _marker(incident_id: str) -> NodeMarker:
    return NodeMarker(
        marker_id="marker-cpu",
        source="host-collector",
        trusted=True,
        incident_id=incident_id,
        observed_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity=Severity.WARNING,
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        mapping_version="test-v1",
    )


def _run(store, operations, outcomes, sent):
    incident, workflow = workflow_state(store, operations)
    store.add_marker(_marker(incident.incident_id))
    executor = active_workflow_executor(
        store, [FakeAdapter(outcomes)], operations, notification_sender=sent.append
    )
    result = execute_workflow(executor, workflow.request_id)
    return incident, store.get_workflow(workflow.request_id), result


def test_a_failed_pure_diagnostic_workflow_closes_the_incident_as_inconclusive():
    store = build_store()
    sent: list[str] = []

    incident, ended, result = _run(
        store,
        [FREEZE, VALIDATE_HOST],
        {
            FREEZE: WorkflowStepOutcome.succeeded(),
            VALIDATE_HOST: WorkflowStepOutcome.failed("host validation refused"),
        },
        sent,
    )

    assert result.status is WorkflowStatus.FAILED, result
    assert ended.status is WorkflowStatus.FAILED
    closed = store.get_incident(incident.incident_id)
    assert closed.state is IncidentState.RECOVERED, closed.state
    assert any("diagnostic inconclusive" in reason for reason in closed.reasons), (
        closed.reasons
    )
    (terminal,) = [e for e in ended.events if e.kind is WorkflowEventKind.TERMINAL]
    assert terminal.details["incident_state"] == IncidentState.RECOVERED.value
    assert terminal.details["diagnostic_inconclusive"] is True
    (marker,) = store.list_markers_for_incident(incident.incident_id)
    assert marker.active is False, "the marker must not outlive the verdict"
    assert "diagnostic inconclusive" in (marker.retired_reason or "")
    (notification,) = store.list_notifications()
    assert notification.incident_id == incident.incident_id
    assert notification.deduplication_key == (
        f"cluster-a/{incident.incident_id}/diagnostic-inconclusive/{ended.request_id}"
    )
    assert "VALIDATE_HOST" in notification.body_text
    assert "host validation refused" in notification.body_text
    assert sent == [notification.notification_id]


def test_the_inconclusive_notification_is_idempotent_per_workflow():
    build = DiagnosticInconclusiveEmailBuilder().build
    kwargs = dict(
        cluster_id="cluster-a",
        incident_id="incident-a",
        workflow_id="workflow-a",
        event_id="event-a",
        node_ids=["node-a"],
        operations=["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        failed_operation="VALIDATE_HOST",
        error="host validation refused",
        policy_source="SITE_NODE_HEALTH",
        official_action="RUN_DIAGNOSTICS",
        reasons=["CPU saturation threshold exceeded"],
    )

    first, second = build(**kwargs), build(**kwargs)

    assert first.deduplication_key == second.deduplication_key
    assert first.deduplication_key.endswith("/diagnostic-inconclusive/workflow-a")
    assert "诊断未定论" in first.subject
    assert "node-a" in first.body_text
    assert "CPU saturation threshold exceeded" in first.body_text
    assert first.support_case_draft == ""


def test_a_failed_validation_after_a_node_change_still_escalates():
    """Regression guard for the reboot/replace/reset chains."""
    store = build_store()
    sent: list[str] = []

    incident, ended, result = _run(
        store,
        [FREEZE, RESET, VALIDATE_GPU],
        {
            FREEZE: WorkflowStepOutcome.succeeded(),
            RESET: WorkflowStepOutcome.succeeded(),
            VALIDATE_GPU: WorkflowStepOutcome.failed("gpu validation refused"),
        },
        sent,
    )

    assert result.status is WorkflowStatus.FAILED
    assert ended.status is WorkflowStatus.FAILED
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED
    (marker,) = store.list_markers_for_incident(incident.incident_id)
    assert marker.active is True
    assert store.list_notifications() == []
    assert sent == []


def test_an_unattempted_node_change_after_the_failed_validation_still_escalates():
    """The verdict is about the plan, not only about what ran: a validation
    that gates a later reboot is a gate, not a diagnostic."""
    store = build_store()
    sent: list[str] = []

    incident, ended, _ = _run(
        store,
        [FREEZE, VALIDATE_HOST, WorkflowOperation.RESTART_NODE],
        {
            FREEZE: WorkflowStepOutcome.succeeded(),
            VALIDATE_HOST: WorkflowStepOutcome.failed("host validation refused"),
        },
        sent,
    )

    assert ended.status is WorkflowStatus.FAILED
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED
    assert store.list_notifications() == []


def test_a_failed_isolating_workflow_keeps_the_quarantine_verdict():
    store = build_store()
    sent: list[str] = []

    incident, ended, _ = _run(
        store,
        [MARK, VALIDATE_HOST],
        {
            MARK: WorkflowStepOutcome.succeeded(),
            VALIDATE_HOST: WorkflowStepOutcome.failed("host validation refused"),
        },
        sent,
    )

    assert ended.status is WorkflowStatus.FAILED
    assert store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED
    assert store.list_notifications() == []
