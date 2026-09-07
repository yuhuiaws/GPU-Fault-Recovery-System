"""Marker lifecycle (FINAL-建议汇总 F-G6).

A marker is written before ingestion picks its incident, so it must not guess
the pointer; a repaired incident retires its markers (``active=False``) on the
two production paths that learn about the repair; a marker whose incident was
repaired for a *different* attempt does not plan a blind restart for this one;
and the correlation window can never outlive the marker TTL.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.markers import marker_blocks_spare
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    TerminalEvent,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.planner import PlanBuilder
from gpu_fault.service import CompletionService
from gpu_fault.spare_health import HyperPodSpareHealthController, SpareHealthState
from tests._builders import (
    build_context,
    copy_model,
    fault_incident,
    node_health_finding,
    workflow_request,
)
from tests.hyperpod.test_hyperpod_spares import (
    coordinator,
    hyperpod_node,
    kubernetes_node,
)

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


def _finding():
    return node_health_finding(
        "finding-marker-lifecycle",
        "event-marker-lifecycle",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason="node level GPU fault",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def _marker(
    ended_at: datetime, *, incident_id: str, node_id: str = "node-a"
) -> NodeMarker:
    return NodeMarker(
        marker_id="marker-1",
        source="test-agent",
        trusted=True,
        incident_id=incident_id,
        observed_at=ended_at - timedelta(seconds=10),
        expires_at=ended_at + timedelta(minutes=30),
        scope=MarkerScope(node_ids=[node_id]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.REBOOT_NODE,
        action_owner="simulated-runtime",
        mapping_version="test-v1",
    )


def test_failed_ingestion_leaves_an_observational_marker_not_a_dangling_pointer(
    monkeypatch,
) -> None:
    """The marker written before ingestion must not name an incident that
    ingestion may never persist. Without a pointer it still blocks the node
    (fail closed), and it is re-pointed once an incident exists."""
    context = build_context()

    def explode(_finding):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", explode)

    with pytest.raises(RuntimeError, match="store unavailable"):
        NodeHealthIngestionService(context).ingest("batch-1", [_finding()])

    (marker,) = context.store.list_markers()
    assert marker.incident_id == ""
    assert marker.active and marker.trusted
    assert marker_blocks_spare(context.store, marker, now=NOW) is True


def test_successful_ingestion_points_the_marker_at_the_persisted_incident() -> None:
    context = build_context()

    result = NodeHealthIngestionService(context).ingest("batch-1", [_finding()])

    (marker,) = context.store.list_markers()
    assert marker.incident_id == result.incident_ids[0]
    assert (
        context.store.get_incident(marker.incident_id).incident_id == marker.incident_id
    )


def test_a_plan_from_an_observational_marker_mints_its_own_incident_id(
    failed_event: TerminalEvent, ended_at: datetime
) -> None:
    plan = PlanBuilder().from_marker(
        failed_event, _marker(ended_at, incident_id=""), default_simulated_profile()
    )

    assert plan.incident_id.startswith("inc-"), (
        'expected plan.incident_id.startswith("inc-") to be true'
    )
    assert len(plan.incident_id) > len("inc-")


def test_spare_health_retires_the_incident_markers_once_the_spare_is_healthy() -> None:
    """``active`` had no writer on any success path; the spare-health
    controller is the one place that knows the node is good again."""
    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(ready=False)}
    coordinator_service, store = coordinator(nodes, core_nodes)
    store.save_profile(default_simulated_profile())
    controller = HyperPodSpareHealthController(
        coordinator_service, IncidentOrchestrator(store), store, failure_threshold=2
    )
    controller.scan()
    rebooting = controller.scan()[0]
    incident = store.get_incident(rebooting["incident_id"])
    observed_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    store.add_marker(
        NodeMarker(
            marker_id="marker-spare",
            source="test",
            trusted=True,
            incident_id=incident.incident_id,
            observed_at=observed_at,
            expires_at=observed_at + timedelta(hours=1),
            scope=MarkerScope(node_ids=["hyperpod-i-spare1"]),
            severity=Severity.CRITICAL,
            recommended_action=RecoveryAction.REBOOT_NODE,
            mapping_version="test",
        )
    )
    workflow = store.get_workflow(incident.workflow_request_id)
    core_nodes["hyperpod-i-spare1"]["status"]["conditions"][0]["status"] = "True"
    store.save_workflow(copy_model(workflow, status=WorkflowStatus.SUCCEEDED))

    controller.scan()
    recovered = controller.scan()[0]

    assert recovered["state"] == SpareHealthState.HEALTHY.value
    (marker,) = store.list_markers_for_incident(incident.incident_id)
    assert marker.active is False


def _repaired_incident(context: ApplicationContext, *, attempt_id: str | None) -> None:
    context.store.save_incident(
        fault_incident(
            "inc-existing",
            "event-existing",
            node_ids=["node-a"],
            attempt_id=attempt_id,
            workflow_request_id="wf-existing",
            state=IncidentState.RECOVERED,
        )
    )
    context.store.save_workflow(
        workflow_request("wf-existing", "inc-existing", WorkflowStatus.SUCCEEDED)
    )


def test_terminal_skips_a_marker_whose_incident_was_repaired_for_another_attempt(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    """A repaired incident about a different attempt says nothing about why
    *this* attempt died; matching it planned a blind same-allocation restart
    inside the TTL. The marker is retired so nothing matches it again."""
    _repaired_incident(context, attempt_id="train-other")
    context.completion.add_marker(_marker(ended_at, incident_id="inc-existing"))

    decision = context.completion.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PENDING_TRIAGE
    assert decision.matched_marker_ids == []
    (marker,) = context.store.list_markers()
    assert marker.active is False


def test_terminal_still_matches_a_repaired_incident_about_this_attempt(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    """The pinned regional flow: the attempt that died during the repair is
    restarted on its repaired allocation once the incident is RECOVERED."""
    _repaired_incident(context, attempt_id=failed_event.attempt_id)
    context.completion.add_marker(_marker(ended_at, incident_id="inc-existing"))

    decision = context.completion.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PLAN_CREATED
    assert decision.matched_marker_ids == ["marker-1"]
    plan = context.store.get_plan(decision.recovery_plan_id)
    assert plan.steps[0].parameters["requires_incident_state"] == "RECOVERED"


def test_marker_window_must_not_exceed_the_marker_ttl() -> None:
    context = build_context()

    with pytest.raises(ValueError, match="marker_window"):
        CompletionService(
            context.store,
            context.diagnostics,
            marker_window=timedelta(hours=2),
            marker_ttl_seconds=3600,
        )


def test_application_context_validates_the_window_against_the_policy_ttl() -> None:
    context = build_context()

    assert context.completion.marker_ttl_seconds == (
        context.policy.policy.marker_ttl_seconds
    )
    assert context.completion.marker_window.total_seconds() <= (
        context.completion.marker_ttl_seconds
    )
