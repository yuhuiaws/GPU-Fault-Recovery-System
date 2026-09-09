"""Control-plane review 2026-09-08: merge shapes that rewrote history.

C-06: the node-lifecycle family bypassed ``disposition()`` and merged a new
finding into a BLOCKED record inside the aggregation window, recompiling the
whole workflow under the same id -- the silent un-blocking F-B4 forbids.

C-08: ``GroupedFaultService._scope_workflow`` rewrote ``node_ids``,
``workload_ids`` and ``parameters`` of *every* official step after a merge,
including steps that completed, were superseded or have an execution record
(F-B6 fixed the widen paths, not this one). A completed STOP_WORKLOADS lost
the nodes it actually stopped; a WAITING RESTART_WORKLOAD had its budget and
source attempt swapped under the command in flight.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.families.grouped_faults import (
    GroupedFaultCallbacks,
    GroupedFaultContext,
    GroupedFaultService,
)
from gpu_fault.orchestration.families.node_lifecycle import (
    NodeLifecycleCallbacks,
    NodeLifecycleOperationService,
    ReplacementContext,
)
from tests._builders import (
    attempt_observation,
    build_store,
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

from ._support import NOW, WorkloadState, event

# ------------------------------------------------------------------ C-06


def _lifecycle_service() -> NodeLifecycleOperationService:
    unused = lambda *args, **kwargs: None  # noqa: E731
    return NodeLifecycleOperationService(
        build_store(),
        None,
        RecoveryArbiter(),
        None,
        NodeLifecycleCallbacks(
            active_job_recovery_workflow=unused,
            active_node_exclusive_workflow=unused,
            aggregation_deadlines=lambda now, *_: (
                now + timedelta(seconds=5),
                now + timedelta(seconds=30),
            ),
            can_append_parallel_job_branch=unused,
            claims_node_exclusively=unused,
            inventory_validation_parameters=unused,
            preemption_scope_matches=unused,
            prepare_preempting_successor=unused,
        ),
        aggregation_window_seconds=30,
    )


def _replacement_context() -> ReplacementContext:
    finding = node_health_finding(
        "finding-b",
        "event-node-b",
        node_id="node-b",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="unrecoverable GPU fault on node-b",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=["GPU-b"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )
    return ReplacementContext(
        finding=finding,
        observation=attempt_observation("job-a", "attempt-a", NOW),
        allocation_nodes=["node-a", "node-b"],
        profile_version="simulated-v1",
        profile=None,
        profile_errors=[],
        group_key='["cluster-a","job-a","attempt-a"]',
        workload_ids=["training/job/job-a"],
        source_gpu_count=2,
    )


def _existing_replacement(status: WorkflowStatus, **updates):
    now = datetime.now(timezone.utc)
    incident = fault_incident(
        "inc-a",
        "event-node-a",
        node_ids=["node-a"],
        state=IncidentState.ESCALATED
        if status is WorkflowStatus.BLOCKED
        else IncidentState.ACTION_PENDING,
        workflow_request_id="wf-a",
        created_at=now,
        updated_at=now,
    )
    workflow = workflow_request(
        "wf-a",
        "inc-a",
        status=status,
        not_before=now + timedelta(seconds=20),
        official_steps=[
            workflow_step(WorkflowOperation.REPLACE_NODE, node_ids=["node-a"])
        ],
        **updates,
    )
    return incident, workflow


def test_a_second_finding_does_not_merge_into_a_blocked_replacement():
    service = _lifecycle_service()
    incident, blocked = _existing_replacement(
        WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        blocked_reasons=["runtime profile does not exist: simulated-v1"],
    )

    state = service._state(
        _replacement_context(), incident, blocked, datetime.now(timezone.utc)
    )

    assert state.merge_existing is False
    assert state.workflow_id != blocked.request_id


def test_a_second_finding_still_merges_into_a_pending_replacement_in_window():
    service = _lifecycle_service()
    incident, pending = _existing_replacement(WorkflowStatus.PENDING)

    state = service._state(
        _replacement_context(), incident, pending, datetime.now(timezone.utc)
    )

    assert state.merge_existing is True
    assert state.workflow_id == pending.request_id
    assert state.fault_nodes == ["node-a", "node-b"]


# ------------------------------------------------------------------ C-08


def _grouped_service() -> GroupedFaultService:
    unused = lambda *args, **kwargs: None  # noqa: E731
    arbiter = RecoveryArbiter()
    return GroupedFaultService(
        build_store(),
        arbiter,
        DagBrancher(arbiter),
        GroupedFaultCallbacks(
            active_job_recovery_workflow=unused,
            active_node_exclusive_workflow=unused,
            aggregation_deadlines=unused,
            attempt_group_key=unused,
            attempt_observation=unused,
            build_workflow=unused,
            claims_node_exclusively=unused,
            generation_fence=unused,
            incident_state_for_workflow=lambda workflow: IncidentState.ACTION_PENDING,
            merge_disposition=unused,
            prepare_preempting_successor=unused,
            preempt_parallel_job_branch=unused,
            quiesce_parameters=lambda parameters, observation: parameters,
            reopen_if_terminal=unused,
            runtime_effective_action=unused,
            widen_node_action_scope=lambda workflow, scope: workflow,
        ),
        aggregation_window_seconds=30,
        workflow_preemption_enabled=True,
        attempt_group_actions=set(),
        groupable_operations=set(),
        official_operation={},
    )


def _grouped_context(*, restart_budget: int, allocation_nodes: list[str]):
    from gpu_fault.policy import ActionDisposition, FaultPolicyDecision
    from tests._builders import container_observation

    xid = event(
        48,
        event_id="xid48-second",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    ).model_copy(update={"gpu_uuid": "GPU-b", "job_id": "job-a", "attempt_id": "att"})
    observation = attempt_observation(
        "job-a",
        "att",
        NOW,
        containers=[
            container_observation(
                f"pod-{n}", f"worker-{n}", i, n, gpu_uuids=[f"GPU-{n[-1]}"]
            )
            for i, n in enumerate(allocation_nodes)
        ],
        workload_ids=["training/job/job-a"],
        restart_budget=restart_budget,
    )
    decision = FaultPolicyDecision.model_construct(
        event_id=xid.event_id,
        action=RecoveryAction.RESET_GPU,
        official_action="RESET_GPU",
        disposition=ActionDisposition.EXECUTABLE,
        reasons=[],
    )
    return GroupedFaultContext(
        event=xid,
        decision=decision,
        observation=observation,
        group_key='["cluster-a","job-a","att"]',
        allocation_nodes=allocation_nodes,
        workload_ids=["training/job/job-a"],
    )


def test_scoping_after_a_merge_leaves_finished_and_in_flight_steps_alone():
    service = _grouped_service()
    stop = workflow_step(
        WorkflowOperation.STOP_WORKLOADS,
        node_ids=["node-a", "node-b"],
        workload_ids=["training/job/job-a"],
        parameters={"termination_initiator_incident_id": "inc-a"},
    )
    reset = workflow_step(
        WorkflowOperation.RESET_GPU, node_ids=["node-a"], gpu_uuids=["GPU-a"]
    )
    restart = workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        node_ids=["node-a", "node-b"],
        workload_ids=["training/job/job-a"],
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "job-a",
            "source_attempt_id": "att",
            "source_gpu_count": 2,
            "restart_budget": 3,
        },
    )
    workflow = workflow_request(
        "wf-a",
        "inc-a",
        status=WorkflowStatus.RUNNING,
        official_steps=[stop, reset, restart],
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.STOP_WORKLOADS],
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.STOP_WORKLOADS),
            workflow_step_execution(
                2, WorkflowOperation.RESTART_WORKLOAD, WorkflowStepStatus.WAITING
            ),
        ],
    )
    incident = fault_incident("inc-a", "xid48-first", workflow_request_id="wf-a")
    # node-b's container terminated since; the restart budget moved to 1.
    context = _grouped_context(restart_budget=1, allocation_nodes=["node-a"])

    _, scoped = service._scope_workflow(context, incident, workflow, workflow)

    assert scoped.official_steps[0] == stop, "completed STOP_WORKLOADS is history"
    assert scoped.official_steps[2] == restart, "WAITING RESTART_WORKLOAD is in flight"
    assert scoped.official_steps[1].workload_ids == ["training/job/job-a"], (
        "the pending step still takes the current scope"
    )
