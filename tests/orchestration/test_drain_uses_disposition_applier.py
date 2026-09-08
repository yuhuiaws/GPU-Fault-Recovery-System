"""The drain routes apply the disposition they asked for (C-07, F-B5 tail).

``DrainOperationService`` asked ``disposition()`` and then folded its nine
verdicts into two of its own: route one (critical site quarantine joining a
node incumbent) treated everything but ABSORB as "append behind the node's
branch", route two (device-resource findings) treated WIDEN_IN_PLACE as
ABSORB and every branch verdict as QUEUE_SUCCESSOR. A PENDING, not-yet-started
RESET_GPU incumbent plus a critical QUARANTINE finding is the review's
example: the verdict is REPLACE_IN_PLACE (stronger and mutable) and the route
queued the quarantine *after* the reset. Both routes now go through
``DispositionApplier`` -- the one implementation the three fault families
already share -- with one documented exception: route one keeps the terminal
branch join for QUEUE_SUCCESSOR, because a critical quarantine must land
inside the RUNNING incumbent's DAG so its readmission and restart are
superseded atomically (the route's whole purpose).
"""

from __future__ import annotations

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    RecoveryAction,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import node_health_finding

from ._support import (
    NOW,
    ApplicationContext,
    WorkloadState,
    _device_resource_finding,
    _node_event,
    ingest,
)


def _critical_quarantine(event_id: str):
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        observed_at=NOW,
        category=NodeHealthCategory.MCE,
        severity="critical",
        reason="machine check on idle recovery node",
        recommended_action=RecoveryAction.QUARANTINE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def _operations(workflow) -> list[WorkflowOperation]:
    return [step.operation for step in workflow.official_steps]


def test_a_critical_quarantine_replaces_an_unstarted_reset_in_place(
    context: ApplicationContext,
) -> None:
    _, reset_incident, reset = ingest(
        context, _node_event(48, event_id="idle-reset-pending", gpu_uuid="GPU-a")
    )
    assert reset.status is WorkflowStatus.PENDING

    incident, workflow = context.orchestrator.ingest_node_health(
        _critical_quarantine("idle-mce-over-pending-reset")
    )

    assert workflow.request_id == reset.request_id
    assert incident.incident_id == reset_incident.incident_id
    assert workflow.fencing_token == reset.fencing_token + 1
    assert incident.fencing_token == workflow.fencing_token
    assert WorkflowOperation.QUARANTINE in _operations(workflow)
    assert WorkflowOperation.RESET_GPU not in _operations(workflow), (
        "the stronger isolation replaced the reset instead of queueing behind it"
    )
    assert WorkflowEventCode.PLAN_REPLACED in {event.code for event in workflow.events}
    assert len(context.store.list_workflows(limit=10)) == 1


def test_a_device_resource_upgrade_records_the_plan_replacement(
    context: ApplicationContext,
) -> None:
    """Route two already replaced in place; now it does so through the shared
    applier, so the swap leaves the PLAN_REPLACED audit event the other
    families leave."""

    _, plugin = context.orchestrator.ingest_node_health(
        _device_resource_finding(
            event_id="efa-plugin-first",
            metric_name="efa_kubernetes_allocatable_mismatch",
            action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
        )
    )

    _, upgraded = context.orchestrator.ingest_node_health(
        _device_resource_finding(
            event_id="efa-driver-second",
            metric_name="efa_inventory_mismatch",
            action=RecoveryAction.REMEDIATE_EFA_DRIVER,
        )
    )

    assert upgraded.request_id == plugin.request_id
    assert upgraded.fencing_token == plugin.fencing_token + 1
    assert upgraded.predecessor_workflow_id == plugin.predecessor_workflow_id
    assert WorkflowEventCode.PLAN_REPLACED in {event.code for event in upgraded.events}
