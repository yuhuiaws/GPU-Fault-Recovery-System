"""A running attempt observed on a node under repair opens a placement hold.

Rule A (F-N1 §8) bounded the wait of a job workflow that already existed. A
new attempt that lands on a node whose GPU is mid-repair (the race before the
cordon, or a pod pinned with ``nodeName``) is only *observed* through
``POST /v1/workload-observations``; nothing looked the node's workflow up, so
the job ran into the repair and the node workflow's VERIFY_NO_GPU_CLIENTS
yielded to the job's GPU processes. The orchestrator now opens a job workflow
for the attempt -- a placement hold -- that waits under rule A: it dissolves
when the nodes are freed inside the window and otherwise stops the job with
the controller-initiated marker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.app.context import default_simulated_profile
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestrator import IncidentOrchestrator
from tests._builders import (
    attempt_observation,
    build_store,
    container_observation,
    fault_incident,
    workflow_request,
    workflow_step,
)

NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
REBOOT = WorkflowOperation.RESTART_NODE
RESET = WorkflowOperation.RESET_GPU
STOP = WorkflowOperation.STOP_WORKLOADS
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
CORDON = WorkflowOperation.MARK_UNSCHEDULABLE


def _store():
    store = build_store()
    store.save_profile(default_simulated_profile())
    return store


def _node_workflow(store, *, operations, request_id="wf-node", incident_id="inc-node"):
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        node_ids=["node-a"],
        fencing_token=1,
    )
    workflow = workflow_request(
        request_id,
        incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[workflow_step(op, node_ids=["node-a"]) for op in operations],
        execution_owner_id="executor-elsewhere",
        execution_lease_expires_at=NOW + timedelta(minutes=10),
    )
    store.save_incident_and_workflow(incident, workflow)


def _observation(observed_at: datetime = NOW, **values):
    return attempt_observation(
        "train-1",
        "train-1-a1",
        observed_at,
        expected_critical_ranks=2,
        containers=[
            container_observation(
                "pod-0", "train-1-0", 0, "node-a", gpu_uuids=["GPU-a"]
            ),
            container_observation(
                "pod-1", "train-1-1", 1, "node-b", gpu_uuids=["GPU-b"]
            ),
        ],
        workload_ids=["training/job/train-1"],
        **values,
    )


def test_an_attempt_observed_on_a_node_under_repair_opens_a_hold():
    store = _store()
    _node_workflow(store, operations=[CORDON, REBOOT])
    orchestrator = IncidentOrchestrator(store)

    held = orchestrator.hold_attempt_on_repairing_nodes(_observation())

    assert held is not None, "a mutating repair on node-a must open a hold"
    incident, workflow = held
    assert incident.incident_id == "hold-cluster-a-train-1-a1"
    assert incident.event_id == incident.incident_id
    assert incident.event_type == "WORKLOAD_PLACED_ON_REPAIRING_NODE"
    assert incident.policy_source == "SITE_PLACEMENT_HOLD"
    assert incident.official_action == "STOP_WORKLOADS"
    assert incident.node_ids == ["node-a", "node-b"]
    assert (incident.job_id, incident.attempt_id) == ("train-1", "train-1-a1")
    assert incident.state is IncidentState.ACTION_PENDING
    assert incident.workflow_request_id == workflow.request_id
    assert any("wf-node" in reason for reason in incident.reasons), incident.reasons

    assert workflow.placement_hold is True
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.fencing_token == 1
    assert workflow.created_at == NOW
    assert [step.operation for step in workflow.official_steps] == [STOP]
    (stop,) = workflow.official_steps
    assert stop.node_ids == ["node-a", "node-b"]
    assert stop.workload_ids == ["training/job/train-1"]
    assert stop.parameters["termination_initiator_incident_id"] == incident.incident_id
    # Compiled against the attempt's runtime profile like every other STOP.
    assert stop.execution_owner == "simulated-runtime"

    (opened,) = [e for e in workflow.events if e.kind is WorkflowEventKind.HOLD]
    assert opened.code == WorkflowEventCode.PLACEMENT_HOLD_OPENED.value
    assert opened.details["remediation_workflow_ids"] == ["wf-node"]
    assert opened.details["nodes"] == ["node-a"]

    # Durable, and the incident row points at the hold.
    assert store.get_workflow(workflow.request_id).placement_hold is True
    assert store.get_incident(incident.incident_id).workflow_request_id == (
        workflow.request_id
    )
    assert orchestrator.placement_holds_opened_total == 1


def test_a_read_only_evidence_workflow_on_the_node_does_not_open_a_hold():
    store = _store()
    _node_workflow(store, operations=[FREEZE, CORDON])
    orchestrator = IncidentOrchestrator(store)

    assert orchestrator.hold_attempt_on_repairing_nodes(_observation()) is None
    assert orchestrator.placement_holds_opened_total == 0
    assert store.list_active_workflow_incidents("cluster-a", job_id="train-1") == []


def test_a_repair_step_already_resolved_does_not_open_a_hold():
    store = _store()
    _node_workflow(store, operations=[RESET, CORDON])
    store.amend_workflow("wf-node", {"completed_step_indexes": [0]})
    orchestrator = IncidentOrchestrator(store)

    assert orchestrator.hold_attempt_on_repairing_nodes(_observation()) is None


def test_repeated_observations_open_exactly_one_hold():
    store = _store()
    _node_workflow(store, operations=[REBOOT])
    orchestrator = IncidentOrchestrator(store)

    first = orchestrator.hold_attempt_on_repairing_nodes(_observation())
    second = orchestrator.hold_attempt_on_repairing_nodes(
        _observation(observed_at=NOW + timedelta(seconds=30))
    )

    assert first is not None, "the first observation opens the hold"
    assert second is None
    holds = [
        workflow
        for _, workflow in store.list_active_workflow_incidents(
            "cluster-a", job_id="train-1"
        )
        if workflow.placement_hold
    ]
    assert [workflow.request_id for workflow in holds] == [first[1].request_id]
    assert orchestrator.placement_holds_opened_total == 1


def test_an_attempt_that_already_has_a_job_workflow_is_not_held():
    store = _store()
    _node_workflow(store, operations=[REBOOT])
    job_incident = fault_incident(
        "inc-job",
        "event-job",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-job",
        node_ids=["node-a", "node-b"],
        job_id="train-1",
        attempt_id="train-1-a1",
        fencing_token=1,
    )
    job_workflow = workflow_request(
        "wf-job",
        "inc-job",
        fencing_token=1,
        official_steps=[
            workflow_step(STOP, node_ids=["node-a", "node-b"]),
            workflow_step(WorkflowOperation.RESTART_WORKLOAD, node_ids=["node-a"]),
        ],
    )
    store.save_incident_and_workflow(job_incident, job_workflow)
    orchestrator = IncidentOrchestrator(store)

    assert orchestrator.hold_attempt_on_repairing_nodes(_observation()) is None
    assert orchestrator.placement_holds_opened_total == 0


def test_an_observation_carrying_the_initiator_marker_is_not_held():
    store = _store()
    _node_workflow(store, operations=[REBOOT])
    orchestrator = IncidentOrchestrator(store)

    held = orchestrator.hold_attempt_on_repairing_nodes(
        _observation(termination_initiator_incident_id="inc-node")
    )

    assert held is None


def test_a_terminal_attempt_is_not_held():
    store = _store()
    _node_workflow(store, operations=[REBOOT])
    orchestrator = IncidentOrchestrator(store)

    held = orchestrator.hold_attempt_on_repairing_nodes(
        _observation(workload_phase="FAILED")
    )

    assert held is None


def test_a_hold_stamps_created_at_no_later_than_now():
    # A data-plane clock ahead of the control plane must not shorten the
    # window the hold waits (created_at + wait is what the dispatcher reads).
    store = _store()
    _node_workflow(store, operations=[REBOOT])
    orchestrator = IncidentOrchestrator(store)
    ahead = datetime.now(timezone.utc) + timedelta(hours=1)

    held = orchestrator.hold_attempt_on_repairing_nodes(_observation(observed_at=ahead))

    assert held is not None, "a mutating repair on node-a must open a hold"
    assert held[1].created_at <= datetime.now(timezone.utc)
