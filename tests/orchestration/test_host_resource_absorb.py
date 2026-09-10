"""A sustained host-resource finding for a node whose same-signal incident is
still with an operator (or still running diagnostics) is recorded on that
incident and mints nothing.

Sustained host-resource signals (``SITE_HOST_RESOURCE_HEALTH``) are tracked
per GPU device on purpose -- several jobs may share a node -- so an 8-GPU
node emits eight LOW_GPU_UTILIZATION findings per activation and re-arms
every time the workload goes idle. Each finding used to mint its own
``inc-<event_id>`` plus a RUN_DIAGNOSTICS workflow; when the diagnostic
failed the incident parked ESCALATED and the next finding minted another.
One node accumulated 42 ESCALATED incidents in a day, each running
``dcgmi diag`` on the same node. The per-device *finding* stays; the
incident side now absorbs: same cluster + node + metric + signal, incident
ESCALATED or ACTION_PENDING -> link the event, append one bounded reason,
count it, return the existing pair.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    INCIDENT_REASONS_LIMIT,
    REASONS_TRUNCATED_PREFIX,
    IncidentState,
    NotificationResult,
    NotificationStatus,
    RecoveryAction,
    Severity,
    WorkflowStatus,
    WorkloadState,
)
from tests._builders import build_context, copy_model, node_health_finding

NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
DEVICES = [f"GPU-{index}" for index in range(8)]
GPU_METRIC = "host_gpu_utilization_percent"
GPU_RULE = "LOW_GPU_UTILIZATION"
GPU_REASON = "GPU utilization remained near zero while a training workload was active"


class RecordingNotifier:
    def __init__(self) -> None:
        self.notifications = []

    def send(self, notification):
        self.notifications.append(notification)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="ses-not-applicable-1",
        )


def _finding(
    device: str | None,
    *,
    node_id: str = "node-a",
    batch_id: str = "batch-1",
    metric_name: str = GPU_METRIC,
    rule_id: str = GPU_RULE,
    category: NodeHealthCategory = NodeHealthCategory.GPU,
    reason: str = GPU_REASON,
    observed_at: datetime = NOW,
) -> NodeHealthFinding:
    # Mirrors ``NodeHealthPolicy._sustained_rule``: the event id is
    # ``<batch>-<metric>-<device|node>-<rule lower>`` and the rule id travels
    # in ``diagnostic_parameters["signal"]``. A telemetry batch is one node's,
    # so its id is unique per node; the node goes into the batch id here.
    event_id = (
        f"{batch_id}-{node_id}-{metric_name}-{device or 'node'}-{rule_id.lower()}"
    )
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        node_id=node_id,
        observed_at=observed_at,
        category=category,
        severity=Severity.WARNING,
        reason=reason,
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name=metric_name,
        value=0.0,
        device=device,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/train-a"],
        diagnostic_parameters={
            "signal": rule_id,
            "comparison": "lte",
            "threshold_percent": 5,
            "minimum_active_seconds": 300,
            "related_metrics": [],
        },
        policy_source="SITE_HOST_RESOURCE_HEALTH",
        policy_reference="site-configurable sustained host resource policy",
    )


def _absorb_counter(context: ApplicationContext) -> int:
    return (
        context.orchestrator._workflow_merger.unsettled_host_resource_record_only_total
    )


def _escalate(context: ApplicationContext, incident) -> None:
    """Park the incident the way a failed RUN_DIAGNOSTICS leaves it."""
    stored = context.store.get_incident(incident.incident_id)
    context.store.save_incident(
        copy_model(stored, state=IncidentState.ESCALATED), expected=stored
    )


def test_eight_device_findings_on_one_node_share_one_incident_and_workflow(
    context: ApplicationContext,
) -> None:
    first_incident, first_workflow = context.orchestrator.ingest_node_health(
        _finding(DEVICES[0])
    )
    assert first_workflow is not None
    _escalate(context, first_incident)

    results = [
        context.orchestrator.ingest_node_health(_finding(device))
        for device in DEVICES[1:]
    ]

    assert {incident.incident_id for incident, _ in results} == {
        first_incident.incident_id
    }
    assert {workflow.request_id for _, workflow in results} == {
        first_workflow.request_id
    }
    assert len(context.store.list_workflows()) == 1
    assert (
        len(
            context.store.list_incidents_by_state(
                "cluster-a", set(IncidentState), node_ids={"node-a"}
            )
        )
        == 1
    )
    for device in DEVICES[1:]:
        linked = context.store.get_incident_by_event(_finding(device).event_id)
        assert linked is not None
        assert linked.incident_id == first_incident.incident_id
    incident = context.store.get_incident(first_incident.incident_id)
    absorbed = [reason for reason in incident.reasons if reason.startswith("absorbed")]
    assert len(absorbed) == 7
    assert (
        f"absorbed {GPU_RULE} finding {_finding(DEVICES[1]).event_id} on device "
        f"{DEVICES[1]}: recorded only, incident awaits operator"
    ) in absorbed
    assert incident.state is IncidentState.ESCALATED
    assert incident.updated_at > first_incident.updated_at
    assert _absorb_counter(context) == 7


def test_a_same_signal_finding_absorbs_into_a_running_diagnostic_too(
    context: ApplicationContext,
) -> None:
    incident, workflow = context.orchestrator.ingest_node_health(_finding(DEVICES[0]))
    assert incident.state is IncidentState.ACTION_PENDING
    context.store.save_workflow(copy_model(workflow, status=WorkflowStatus.RUNNING))

    again_incident, again_workflow = context.orchestrator.ingest_node_health(
        _finding(DEVICES[1])
    )

    assert again_incident.incident_id == incident.incident_id
    assert again_workflow.request_id == workflow.request_id
    assert again_workflow.status is WorkflowStatus.RUNNING
    assert len(context.store.list_workflows()) == 1
    assert _absorb_counter(context) == 1


def test_a_different_metric_on_the_same_node_opens_its_own_incident(
    context: ApplicationContext,
) -> None:
    gpu_incident, _ = context.orchestrator.ingest_node_health(_finding(DEVICES[0]))
    _escalate(context, gpu_incident)

    cpu_incident, cpu_workflow = context.orchestrator.ingest_node_health(
        _finding(
            None,
            metric_name="cpu_usage_percent",
            rule_id="LOW_CPU_UTILIZATION",
            category=NodeHealthCategory.CPU,
            reason="CPU utilization remained near zero while a workload was active",
        )
    )

    assert cpu_incident.incident_id != gpu_incident.incident_id
    assert cpu_workflow is not None
    assert len(context.store.list_workflows()) == 2
    assert _absorb_counter(context) == 0


def test_the_same_signal_on_another_node_opens_its_own_incident(
    context: ApplicationContext,
) -> None:
    incident_a, _ = context.orchestrator.ingest_node_health(_finding(DEVICES[0]))
    _escalate(context, incident_a)

    incident_b, workflow_b = context.orchestrator.ingest_node_health(
        _finding(DEVICES[0], node_id="node-b")
    )

    assert incident_b.incident_id != incident_a.incident_id
    assert workflow_b is not None
    assert incident_b.node_ids == ["node-b"]
    assert len(context.store.list_workflows()) == 2
    assert _absorb_counter(context) == 0


def test_a_recovered_or_quarantined_incident_does_not_absorb(
    context: ApplicationContext,
) -> None:
    for state in (IncidentState.RECOVERED, IncidentState.QUARANTINED):
        node_id = f"node-{state.value.lower()}"
        incident, _ = context.orchestrator.ingest_node_health(
            _finding(DEVICES[0], node_id=node_id)
        )
        stored = context.store.get_incident(incident.incident_id)
        context.store.save_incident(copy_model(stored, state=state), expected=stored)

        again, again_workflow = context.orchestrator.ingest_node_health(
            _finding(DEVICES[1], node_id=node_id)
        )

        assert again.incident_id != incident.incident_id, state
        assert again_workflow is not None
        assert again_workflow.request_id != stored.workflow_request_id
    assert len(context.store.list_workflows()) == 4
    assert _absorb_counter(context) == 0


def test_a_re_posted_absorbed_event_stays_on_the_incident_without_a_second_reason(
    context: ApplicationContext,
) -> None:
    incident, _ = context.orchestrator.ingest_node_health(_finding(DEVICES[0]))
    _escalate(context, incident)
    context.orchestrator.ingest_node_health(_finding(DEVICES[1]))

    again, _ = context.orchestrator.ingest_node_health(_finding(DEVICES[1]))

    assert again.incident_id == incident.incident_id
    reasons = context.store.get_incident(incident.incident_id).reasons
    assert len([reason for reason in reasons if reason.startswith("absorbed")]) == 1
    assert _absorb_counter(context) == 1


def test_absorbed_reasons_stay_within_the_incident_reason_budget(
    context: ApplicationContext,
) -> None:
    incident, _ = context.orchestrator.ingest_node_health(_finding(DEVICES[0]))
    stored = context.store.get_incident(incident.incident_id)
    filler = [f"earlier reason {index}" for index in range(INCIDENT_REASONS_LIMIT)]
    context.store.save_incident(
        copy_model(stored, state=IncidentState.ESCALATED, reasons=filler),
        expected=stored,
    )

    for device in DEVICES[1:]:
        context.orchestrator.ingest_node_health(_finding(device))

    reasons = context.store.get_incident(incident.incident_id).reasons
    assert len(reasons) <= INCIDENT_REASONS_LIMIT
    assert reasons[-1].startswith(f"absorbed {GPU_RULE} finding"), reasons[-1]
    assert DEVICES[-1] in reasons[-1]
    assert any(reason.startswith(REASONS_TRUNCATED_PREFIX) for reason in reasons), (
        "the bounded reasons carry the truncation marker"
    )
    assert _absorb_counter(context) == 7


def test_one_node_gets_one_notification_for_eight_device_findings_and_none_on_re_arm() -> (
    None
):
    notifier = RecordingNotifier()
    context = build_context(notification_notifier=notifier)
    service = NodeHealthIngestionService(context)

    first = service.ingest("batch-1", [_finding(device) for device in DEVICES])

    assert len(set(first.incident_ids)) == 1
    assert len(set(first.workflow_request_ids)) == 1
    assert len(first.notification_ids) == 1
    assert len(notifier.notifications) == 1
    body = notifier.notifications[0].body_text
    assert all(device in body for device in DEVICES), body
    assert _absorb_counter(context) == 7

    # The diagnostic failed; the signal re-arms two hours later (outside the
    # notification cooldown bucket) and every device reports again.
    _escalate(context, context.store.get_incident(first.incident_ids[0]))
    later = NOW + timedelta(hours=2)
    second = service.ingest(
        "batch-2",
        [_finding(device, batch_id="batch-2", observed_at=later) for device in DEVICES],
    )

    assert set(second.incident_ids) == set(first.incident_ids)
    assert set(second.workflow_request_ids) == set(first.workflow_request_ids)
    assert second.notification_ids == []
    assert len(notifier.notifications) == 1
    assert len(context.store.list_notifications()) == 1
    assert len(context.store.list_workflows()) == 1
    assert _absorb_counter(context) == 15
