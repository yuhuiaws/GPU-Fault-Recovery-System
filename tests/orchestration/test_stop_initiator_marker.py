"""Every STOP_WORKLOADS the control plane plans names the incident that
initiated it (F-C9 remainder).

The completion watcher tells a controller-initiated stop from a user stop by
the initiator annotation the stop step carries. The node-health family
compiled its STOP_WORKLOADS without it, so a job the system stopped to repair
a node was later judged a failed job and re-planned as a new fault.
"""

from __future__ import annotations

from gpu_fault.models import WorkflowOperation, WorkloadState
from tests.orchestration._support import _inventory_mismatch_finding


def test_a_node_health_stop_names_the_incident_that_initiated_it(context):
    finding = _inventory_mismatch_finding(
        event_id="inventory-with-workload",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/train"],
    )

    incident, workflow = context.orchestrator.ingest_node_health(finding)

    assert workflow is not None
    stops = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    ]
    assert stops, "an active workload on a rebooting node must be stopped first"
    for step in stops:
        assert (
            step.parameters["termination_initiator_incident_id"] == incident.incident_id
        )
