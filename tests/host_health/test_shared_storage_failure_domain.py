"""GF-GENERAL-LIVE-005 ``shared_failure_domain``: one export failing on several
nodes is one shared-storage outage, not N node faults.

Every node still mints its own diagnostic-only chain (FREEZE_EVIDENCE ->
VALIDATE_HOST, nothing isolated); on top of that the control plane records one
cluster-scoped ``SHARED_STORAGE_FAILURE_DOMAIN`` incident naming the export and
the nodes, references it from every per-node incident, and mails once.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
)
from gpu_fault.shared_storage_failure_domain import (
    SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
    SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY,
    SHARED_STORAGE_FAILURE_DOMAIN_POLICY_SOURCE,
    shared_filesystem_identity,
)
from tests._builders import asgi_client, host_telemetry_batch
from tests.host_health._support import NOW, RecordingNotifier

EXPORT = "10.0.0.5:/export"
OTHER_EXPORT = "10.0.0.6:/other"
MOUNT = "/mnt/shared"
LABELS = {"source": EXPORT, "fstype": "nfs4"}
DIAGNOSTIC_ONLY = [WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.VALIDATE_HOST]


def shared_batch(
    batch_id: str,
    node_id: str,
    value: float,
    observed_at=NOW,
    *,
    labels: dict[str, str] | None = LABELS,
    mount: str = MOUNT,
) -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        observed_at,
        [
            HostMetricSample(
                name="shared_filesystem_unavailable",
                value=value,
                device=mount,
                labels=dict(labels or {}),
            )
        ],
        node_id=node_id,
        runtime_profile_version="simulated-v1",
    )


def ingest(context: ApplicationContext, *batches: HostTelemetryBatch) -> list[dict]:
    async def scenario() -> list[dict]:
        results = []
        async with asgi_client(context) as client:
            for batch in batches:
                response = await client.post(
                    "/v1/collector-events/host-telemetry",
                    json=batch.model_dump(mode="json"),
                )
                assert response.status_code < 300, response.text
                results.append(response.json())
        return results

    return asyncio.run(scenario())


def domain_records(context: ApplicationContext) -> list[FaultIncident]:
    return [
        incident
        for incident in context.store.list_incidents_by_state(
            "cluster-a", set(IncidentState)
        )
        if incident.event_type == SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE
    ]


def node_incidents(context: ApplicationContext, node_id: str) -> list[FaultIncident]:
    return [
        incident
        for incident in context.store.list_incidents_by_state(
            "cluster-a", set(IncidentState), node_ids={node_id}
        )
        if incident.event_type == "NODE_HEALTH"
    ]


def test_two_nodes_same_export_share_one_failure_domain_record() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)

    first, second = ingest(
        context,
        shared_batch("a-1", "node-a", 1),
        shared_batch("b-1", "node-b", 1, NOW + timedelta(seconds=15)),
    )

    # Both nodes keep their own diagnostic-only chain: nothing is isolated or
    # stopped because of a shared-storage outage.
    chains = {}
    for node_id in ("node-a", "node-b"):
        incidents = node_incidents(context, node_id)
        assert len(incidents) == 1, f"{node_id} should own exactly one node incident"
        (incident,) = incidents
        assert incident.effective_action is RecoveryAction.RUN_DIAGNOSTICS
        assert incident.workflow_request_id is not None
        workflow = context.store.get_workflow(incident.workflow_request_id)
        assert [step.operation for step in workflow.official_steps] == DIAGNOSTIC_ONLY
        chains[node_id] = incident

    domains = domain_records(context)
    assert len(domains) == 1, "one export on two nodes is one failure domain"
    (domain,) = domains
    assert domain.node_ids == ["node-a", "node-b"]
    assert domain.state is IncidentState.DETECTED
    assert domain.policy_source == SHARED_STORAGE_FAILURE_DOMAIN_POLICY_SOURCE
    assert domain.policy_reference == f"shared-filesystem:{EXPORT}"
    assert domain.event_source == "shared_filesystem_unavailable"
    assert domain.effective_action is None, "the domain record drives no action"
    assert domain.workflow_request_id is None, "the domain record has no workflow"
    assert EXPORT in domain.reasons[0] and "node-a" in domain.reasons[0]
    assert "node-b" in domain.reasons[0]
    for node_id, incident in chains.items():
        assert any(
            incident.incident_id in reason and node_id in reason
            for reason in domain.reasons
        ), f"the domain record must reference {node_id}'s incident"
        stored = context.store.get_incident(incident.incident_id)
        assert any(
            domain.incident_id in reason and EXPORT in reason
            for reason in stored.reasons
        ), f"{node_id}'s incident must reference the failure domain"
        # The reference is prose on the record only: the chain is untouched.
        assert stored.effective_action is RecoveryAction.RUN_DIAGNOSTICS
        assert stored.state is incident.state

    assert first["notification_ids"] == []
    assert len(second["notification_ids"]) == 1
    assert len(notifier.notifications) == 1, "one notification for the domain"
    notification = notifier.notifications[0]
    assert notification.category == SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY
    assert notification.incident_id == domain.incident_id
    assert EXPORT in notification.subject
    for node_id, incident in chains.items():
        assert node_id in notification.body_text
        assert incident.incident_id in notification.body_text
        assert incident.incident_id in notification.evidence_refs
    assert "not" in notification.body_text and "isolat" in notification.body_text


def test_two_nodes_different_exports_are_not_correlated() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)

    ingest(
        context,
        shared_batch("a-1", "node-a", 1),
        shared_batch(
            "b-1",
            "node-b",
            1,
            NOW + timedelta(seconds=15),
            labels={"source": OTHER_EXPORT, "fstype": "nfs4"},
        ),
    )

    assert domain_records(context) == []
    assert notifier.notifications == []
    assert len(node_incidents(context, "node-a")) == 1
    assert len(node_incidents(context, "node-b")) == 1


def test_one_node_alone_is_not_a_failure_domain() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)

    ingest(
        context,
        shared_batch("a-1", "node-a", 1),
        shared_batch("a-2", "node-a", 1, NOW + timedelta(seconds=15)),
    )

    assert domain_records(context) == []
    assert notifier.notifications == []
    assert len(node_incidents(context, "node-a")) == 1


def test_missing_identity_holds_no_correlation() -> None:
    # A collector that ships no export identity (an older node image) leaves the
    # mount point as the only key, and a mount point is per node: two nodes may
    # mount different exports at /mnt/shared. Missing identity is UNKNOWN, and
    # UNKNOWN is never correlated; the per-node chains are unchanged.
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)

    ingest(
        context,
        shared_batch("a-1", "node-a", 1, labels={}),
        shared_batch("b-1", "node-b", 1, NOW + timedelta(seconds=15), labels={}),
    )

    assert domain_records(context) == []
    assert notifier.notifications == []
    for node_id in ("node-a", "node-b"):
        (incident,) = node_incidents(context, node_id)
        assert incident.effective_action is RecoveryAction.RUN_DIAGNOSTICS


def test_domain_clears_when_every_member_reports_the_export_available() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    ingest(
        context,
        shared_batch("a-1", "node-a", 1),
        shared_batch("b-1", "node-b", 1, NOW + timedelta(seconds=15)),
    )
    (domain,) = domain_records(context)

    ingest(context, shared_batch("a-2", "node-a", 0, NOW + timedelta(seconds=60)))
    still_open = context.store.get_incident(domain.incident_id)
    assert still_open.state is IncidentState.DETECTED, (
        "node-b still reports the export unavailable"
    )

    ingest(context, shared_batch("b-2", "node-b", 0, NOW + timedelta(seconds=75)))
    cleared = context.store.get_incident(domain.incident_id)
    assert cleared.state is IncidentState.RECOVERED
    assert "available again" in cleared.reasons[-1]
    assert EXPORT in cleared.reasons[-1]
    assert cleared.node_ids == ["node-a", "node-b"], "membership is history"
    assert len(notifier.notifications) == 1, "clearing mails nothing new"
    assert domain_records(context) == [cleared]


def test_silent_member_escalates_the_domain_to_the_operator() -> None:
    # node-b stops reporting (collector wedged, node gone): silence is not
    # health. The record cannot close on positive evidence, so it is handed to
    # the operator (ESCALATED is the one state an operator may close), and a
    # later reading of 0 from node-b still closes it.
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    ingest(
        context,
        shared_batch("a-1", "node-a", 1),
        shared_batch("b-1", "node-b", 1, NOW + timedelta(seconds=15)),
    )
    (domain,) = domain_records(context)

    ingest(context, shared_batch("a-2", "node-a", 0, NOW + timedelta(minutes=15)))
    escalated = context.store.get_incident(domain.incident_id)
    assert escalated.state is IncidentState.ESCALATED
    assert "node-b" in escalated.reasons[-1]
    assert "operator" in escalated.reasons[-1]

    ingest(context, shared_batch("b-2", "node-b", 0, NOW + timedelta(minutes=16)))
    recovered = context.store.get_incident(domain.incident_id)
    assert recovered.state is IncidentState.RECOVERED
    assert len(notifier.notifications) == 1


def test_third_node_joins_the_open_domain_without_a_second_notification() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)

    ingest(
        context,
        shared_batch("a-1", "node-a", 1),
        shared_batch("b-1", "node-b", 1, NOW + timedelta(seconds=15)),
        shared_batch("c-1", "node-c", 1, NOW + timedelta(seconds=30)),
    )

    (domain,) = domain_records(context)
    assert domain.node_ids == ["node-a", "node-b", "node-c"]
    (incident,) = node_incidents(context, "node-c")
    assert any(incident.incident_id in reason for reason in domain.reasons), (
        "the domain record must reference node-c's incident"
    )
    assert any(domain.incident_id in reason for reason in incident.reasons), (
        "node-c's incident must reference the domain record"
    )
    assert len(notifier.notifications) == 1


def test_failing_sample_after_a_missed_activation_joins_via_the_safety_net() -> None:
    # node-b's activation carried no identity (nothing to correlate on); the
    # next failing sample does. No new finding is minted for an already-active
    # signal, so the per-batch safety net is what correlates it.
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)

    ingest(
        context,
        shared_batch("a-1", "node-a", 1),
        shared_batch("b-1", "node-b", 1, NOW + timedelta(seconds=15), labels={}),
    )
    assert domain_records(context) == []

    ingest(context, shared_batch("b-2", "node-b", 1, NOW + timedelta(seconds=30)))

    (domain,) = domain_records(context)
    assert domain.node_ids == ["node-a", "node-b"]
    (incident,) = node_incidents(context, "node-b")
    assert any(incident.incident_id in reason for reason in domain.reasons), (
        "the domain record must reference node-b's incident"
    )
    assert len(notifier.notifications) == 1


def test_identity_resolves_only_from_an_export_identity() -> None:
    assert shared_filesystem_identity({"source": " 10.0.0.5:/export/ "}) == EXPORT
    assert shared_filesystem_identity({"mount_source": EXPORT}) == EXPORT
    assert shared_filesystem_identity({"server": "10.0.0.5", "export": "/export"}) == (
        EXPORT
    )
    assert shared_filesystem_identity({"source": "fs-0abc@tcp:/"}) == "fs-0abc@tcp:/"
    assert shared_filesystem_identity({}) is None
    assert shared_filesystem_identity({"source": "   "}) is None
    assert shared_filesystem_identity({"server": "10.0.0.5"}) is None
    assert shared_filesystem_identity({"fstype": "nfs4", "mount": MOUNT}) is None
