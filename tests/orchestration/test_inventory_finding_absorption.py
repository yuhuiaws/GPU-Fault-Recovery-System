"""Allocatable dips a RUNNING node-exclusive chain causes are absorbed by it.

Live (gpu-eks-cluster, XID 46 reset): the eks-managed quiesce taint evicted the
EFA device plugin as well as the GPU one. ``efa_kubernetes_allocatable_mismatch``
was not a transient finding, so the drain route queued a
RESTART_EFA_DEVICE_PLUGIN successor behind the RUNNING reset; the GPU finding
that followed asked for the node's active node-exclusive workflow, got that
PENDING successor first, judged "not RUNNING, not covering" and merged into the
successor instead -- replacing its plan and moving the incident to generation 2
while the reset chain held 1. Every later command of the chain was refused.
Both findings now belong to the running chain, and the covering check looks at
the RUNNING incumbent even when a successor is queued ahead of it in the listing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.app import default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import RecoveryAction, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import (
    DEVICE_PLUGIN_UNAVAILABLE_OPERATIONS,
    INVENTORY_FINDING_VALIDATIONS,
    TRANSIENT_INVENTORY_FINDINGS,
    NodeConflictService,
)
from tests._builders import (
    build_store,
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
)

CLUSTER = "cluster-a"
NOW = datetime(2026, 10, 5, 20, 11, tzinfo=timezone.utc)
RESET_CHAIN = [
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESTORE_GPU_SERVICES,
    WorkflowOperation.VALIDATE_GPU,
]


def _finding(
    metric: str, *, event_id: str, action: RecoveryAction
) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        cluster_id=CLUSTER,
        node_id="node-a",
        observed_at=NOW,
        category=NodeHealthCategory.RDMA if "efa" in metric else NodeHealthCategory.GPU,
        severity="critical",
        metric_name=metric,
        reason=f"{metric} on node-a",
        recommended_action=action,
        runtime_profile_version="simulated-v1",
    )


def _efa_allocatable() -> NodeHealthFinding:
    return _finding(
        "efa_kubernetes_allocatable_mismatch",
        event_id="event-efa-allocatable",
        action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
    )


def _gpu_allocatable() -> NodeHealthFinding:
    return _finding(
        "gpu_kubernetes_allocatable_mismatch",
        event_id="event-gpu-allocatable",
        action=RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
    )


def _running_reset(store, *, with_pending_successor: bool = False) -> None:
    incident = fault_incident(
        "inc-reset",
        "event-reset",
        cluster_id=CLUSTER,
        node_ids=["node-a"],
        workflow_request_id="wf-reset",
        fencing_token=1,
    )
    reset = workflow_request(
        "wf-reset",
        "inc-reset",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[workflow_step(operation) for operation in RESET_CHAIN],
        completed_step_indexes=[0, 1],
        completed_operations=RESET_CHAIN[:2],
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident_and_workflow(incident, reset)
    if with_pending_successor:
        # Listed ahead of the reset by its earlier timestamps: the shape that
        # hid the RUNNING incumbent from the first-match lookup.
        store.save_workflow(
            workflow_request(
                "wf-efa-plugin",
                "inc-reset",
                status=WorkflowStatus.PENDING,
                fencing_token=1,
                predecessor_workflow_id="wf-reset",
                official_steps=[
                    workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE),
                    workflow_step(WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN),
                    workflow_step(WorkflowOperation.VALIDATE_FABRIC),
                ],
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )


def test_the_efa_findings_are_transient_during_a_plugin_evicting_chain() -> None:
    assert WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN in (
        DEVICE_PLUGIN_UNAVAILABLE_OPERATIONS
    )
    for metric in ("efa_kubernetes_allocatable_mismatch", "efa_inventory_mismatch"):
        assert metric in TRANSIENT_INVENTORY_FINDINGS
        assert (
            WorkflowOperation.VALIDATE_FABRIC in INVENTORY_FINDING_VALIDATIONS[metric]
        )
        assert (
            WorkflowOperation.VALIDATE_GPU in INVENTORY_FINDING_VALIDATIONS[metric]
        ), (
            "a reset chain ends in VALIDATE_GPU; the EFA plugin returns with the "
            "taint's removal and a lasting mismatch is raised again by the next scan"
        )
    assert (
        WorkflowOperation.QUIESCE_GPU_SERVICES
        in TRANSIENT_INVENTORY_FINDINGS["efa_kubernetes_allocatable_mismatch"]
    )
    assert INVENTORY_FINDING_VALIDATIONS["gpu_kubernetes_allocatable_mismatch"] == {
        WorkflowOperation.VALIDATE_GPU
    }


def test_the_running_reset_covers_the_efa_allocatable_dip_its_taint_causes() -> None:
    store = build_store()
    _running_reset(store)
    service = NodeConflictService(store, RecoveryArbiter())

    covered = service.active_workflow_covers_inventory_finding(_efa_allocatable())

    assert covered is not None, "the quiesce taint evicted the EFA plugin too"
    assert covered[0].incident_id == "inc-reset"
    assert covered[1].request_id == "wf-reset"


def test_a_pending_successor_ahead_in_the_listing_does_not_hide_the_incumbent() -> None:
    store = build_store()
    _running_reset(store, with_pending_successor=True)
    service = NodeConflictService(store, RecoveryArbiter())
    listed = [
        workflow.request_id
        for workflow in service.active_node_exclusive_workflows(CLUSTER, {"node-a"})
    ]
    assert set(listed) == {"wf-reset", "wf-efa-plugin"}, listed

    covered = service.active_workflow_covers_inventory_finding(_gpu_allocatable())

    assert covered is not None, "the RUNNING chain is the incumbent, not the queue"
    assert covered[1].request_id == "wf-reset"
    assert covered[1].status is WorkflowStatus.RUNNING


def test_the_efa_dip_joins_the_running_reset_without_a_successor() -> None:
    store = build_store()
    store.save_profile(default_simulated_profile())
    _running_reset(store)
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    incident, workflow = orchestrator.ingest_node_health(_efa_allocatable())
    incident_again, workflow_again = orchestrator.ingest_node_health(_gpu_allocatable())

    assert incident.incident_id == incident_again.incident_id == "inc-reset"
    assert workflow is not None and workflow.request_id == "wf-reset"
    assert workflow_again is not None and workflow_again.request_id == "wf-reset"
    assert [item.request_id for item in store.list_workflows()] == ["wf-reset"], (
        "neither dip queues a device-plugin successor on the reset's incident"
    )
    current = store.get_incident("inc-reset")
    assert current.workflow_request_id == "wf-reset", "the incident pointer stays"
    assert current.fencing_token == 1, "and so does the generation the chain holds"
