"""A record-only absorption leaves the incumbent's scope alone.

Live (eks-first site, COLLECT-016 D on the two-node topology): a training
job's RESET_GPU workflow on node A had stopped the job (STOP_WORKLOADS
SUCCEEDED, RESTART_WORKLOAD pending). The peer rank on node B, left without
its partner, raised a read-only hung-traffic finding every collector cycle;
each was judged ABSORB_RECORD_ONLY -- the pending job restart on B already
covers a read-only look at the attempt -- yet the grouped-health merge then
widened the incident to both nodes and the reset's pending VALIDATE_GPU and
RESTORE_SCHEDULING onto B, eighteen times: the reset validated and uncordoned
a node it never isolated, and its incident no longer named the faulted node
alone. A record-only absorption now links the event and records the reason,
and nothing else moves.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.orchestration._cov95_closure_extra_grouped import (
    RESTART,
    STOP,
    WORKLOAD,
    finding,
    grouped_service,
    observed_attempt,
    seed_group,
)
from tests.orchestration._cov95_closure_extra_safety import (
    closure_extra_isolation as closure_extra_isolation,
)
from tests.orchestration._cov95_closure_extra_support import NOW
from tests.orchestration._cov95_closure_extra_support import (
    closure_store_fixture as closure_store_fixture,
)

FAULTED = "node-a"
PEER = "node-b"
HUNG_REASON = "EFA traffic remains at zero"


def _two_node_reset(identity: str = "reset"):
    """The incumbent as the live reset left it mid-flight: the job stopped on
    both nodes, the GPU work on the faulted node ahead, the restart pending."""

    node_steps = (
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    )
    steps = [
        workflow_step(
            WorkflowOperation.MARK_UNSCHEDULABLE,
            node_ids=[FAULTED],
            gpu_uuids=["GPU-a"],
        ),
        workflow_step(STOP, node_ids=[FAULTED, PEER], workload_ids=[WORKLOAD]),
        *(
            workflow_step(operation, node_ids=[FAULTED], gpu_uuids=["GPU-a"])
            for operation in node_steps
        ),
        workflow_step(WorkflowOperation.VALIDATE_GPU, node_ids=[FAULTED]),
        workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=[FAULTED]),
        workflow_step(RESTART, node_ids=[FAULTED, PEER], workload_ids=[WORKLOAD]),
    ]
    workflow = workflow_request(
        f"wf-{identity}",
        identity,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE, STOP],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowStepStatus.SUCCEEDED
            ),
            workflow_step_execution(1, STOP, WorkflowStepStatus.SUCCEEDED),
        ],
        aggregation_max_deadline=NOW - timedelta(minutes=1),
        created_at=NOW - timedelta(minutes=2),
        updated_at=NOW - timedelta(minutes=1),
    )
    incident = fault_incident(
        identity,
        f"event-{identity}",
        node_ids=[FAULTED],
        state=IncidentState.ACTION_PENDING,
        effective_action=RecoveryAction.RESET_GPU,
        official_action="RESET_GPU",
        workflow_request_id=workflow.request_id,
        fencing_token=workflow.fencing_token,
        gpu_uuids=["GPU-a"],
        reasons=["XID on the faulted node"],
        created_at=NOW - timedelta(minutes=2),
        updated_at=NOW - timedelta(minutes=1),
    )
    return incident, workflow


def _peer_evidence_candidate():
    """What the node-health family compiles for the peer's hung finding: a
    read-only look at node B, no GPU identity."""

    workflow = workflow_request(
        "wf-peer-evidence",
        "peer-evidence",
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=[PEER]),
            workflow_step(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE, node_ids=[PEER]),
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    incident = fault_incident(
        "peer-evidence",
        "event-peer-evidence",
        node_ids=[PEER],
        state=IncidentState.ACTION_PENDING,
        effective_action=RecoveryAction.RUN_DIAGNOSTICS,
        workflow_request_id=workflow.request_id,
        fencing_token=workflow.fencing_token,
        gpu_uuids=[],
        reasons=[HUNG_REASON],
        created_at=NOW,
        updated_at=NOW,
    )
    return incident, workflow


def _peer_hang(event_id: str):
    return finding(
        event_id,
        node_id=PEER,
        category=NodeHealthCategory.NCCL,
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        gpu_uuids=[],
        reason=HUNG_REASON,
        diagnostic_parameters={"diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED"},
    )


def test_a_record_only_peer_finding_keeps_the_incident_and_the_plan_on_the_faulted_node(
    closure_store,
) -> None:
    observation = observed_attempt()
    incident, workflow = seed_group(closure_store, observation, *_two_node_reset())
    before = [list(step.node_ids) for step in workflow.official_steps]
    harness = grouped_service(closure_store, observation, _peer_evidence_candidate())

    for event_id in ("peer-hang-1", "peer-hang-2", "peer-hang-3"):
        report = _peer_hang(event_id)
        merged_incident, merged_workflow = harness.service.ingest(report)

        assert merged_workflow.request_id == workflow.request_id, (
            "a record-only absorption neither queues nor replaces the reset"
        )
        assert merged_workflow.fencing_token == workflow.fencing_token
        assert [list(step.node_ids) for step in merged_workflow.official_steps] == (
            before
        ), (
            "the reset's pending VALIDATE_GPU/RESTORE_SCHEDULING stay on the faulted node"
        )
        assert merged_incident.node_ids == [FAULTED], (
            "the incident still names the faulted node alone"
        )
        assert merged_incident.gpu_uuids == ["GPU-a"], "no GPU joined the incident"
        assert merged_incident.reasons[-1] == (
            f"{PEER}: EFA_TRAFFIC_HUNG_SUSPECTED: {HUNG_REASON}"
        ), "the peer's finding is recorded on the incident"
        linked = closure_store.get_incident_by_event(report.event_id)
        assert linked is not None and linked.incident_id == incident.incident_id, (
            "the event is linked to the incident"
        )

    assert harness.widenings == [], "nothing was widened onto the peer"
    assert len(closure_store.list_workflows()) == 1, "no successor was queued"
    stored = closure_store.get_workflow(workflow.request_id)
    assert [list(step.node_ids) for step in stored.official_steps] == before
    assert closure_store.get_incident(incident.incident_id).node_ids == [FAULTED]
