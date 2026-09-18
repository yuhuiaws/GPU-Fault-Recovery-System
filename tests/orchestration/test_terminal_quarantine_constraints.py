from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from gpu_fault.policy import XidEvent
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.workflow_quarantine import (
    TERMINAL_QUARANTINE_NODES,
    inherit_terminal_quarantine,
    suppress_readmission,
    terminal_quarantine_covered,
    terminal_quarantine_nodes,
)
from tests._builders import (
    attempt_observation,
    build_context,
    container_observation,
    copy_model,
    node_health_finding,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation


def health_finding(
    event_id: str, action: RecoveryAction, *, active: bool = False
) -> NodeHealthFinding:
    quarantine = action is RecoveryAction.QUARANTINE
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        observed_at=datetime.now(timezone.utc),
        category=NodeHealthCategory.NETWORK if quarantine else NodeHealthCategory.GPU,
        severity="critical",
        reason="critical node signal",
        metric_name=(
            "critical_network_link_down" if quarantine else "gpu_inventory_mismatch"
        ),
        recommended_action=action,
        gpu_uuids=["GPU-a"] if action is RecoveryAction.RESET_GPU else [],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE if active else WorkloadState.IDLE,
        affected_workload_ids=["training/job/job-a"] if active else [],
    )


@pytest.fixture(params=["memory", "sqlite"])
def health_context(request, tmp_path):
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "health.sqlite"))
    )
    try:
        yield build_context(store=store)
    finally:
        if isinstance(store, SqliteStore):
            store.close()


def live_operations(workflow: WorkflowRequest) -> set[WorkflowOperation]:
    return {
        step.operation
        for index, step in enumerate(workflow.official_steps)
        if index not in workflow.superseded_step_indexes
    }


def prove_quarantine_over_pending_reboot(context: ApplicationContext) -> None:
    _, reboot = context.orchestrator.ingest_node_health(
        health_finding("inventory-first", RecoveryAction.REBOOT_NODE)
    )
    assert reboot is not None
    assert {OP.QUARANTINE, OP.RESTART_NODE, OP.RESTORE_SCHEDULING} <= live_operations(
        reboot
    )

    incident, isolated = context.orchestrator.ingest_node_health(
        health_finding("network-second", RecoveryAction.QUARANTINE)
    )

    assert isolated is not None
    assert isolated.request_id == reboot.request_id
    assert isolated.fencing_token == reboot.fencing_token + 1
    assert incident.fencing_token == isolated.fencing_token
    assert OP.QUARANTINE in live_operations(isolated)
    assert not {
        OP.RESTART_NODE,
        OP.RESTORE_SCHEDULING,
        OP.RESTART_WORKLOAD,
    }.intersection(live_operations(isolated)), (
        "a persistent quarantine must replace the unstarted temporary recovery"
    )
    assert terminal_quarantine_nodes(isolated) == frozenset({"node-a"})
    assert context.store.get_incident_by_event("network-second") == incident


def test_persistent_quarantine_replaces_temporary_reboot_plan(health_context):
    prove_quarantine_over_pending_reboot(health_context)


def prove_quarantine_after_completed_stop(context: ApplicationContext) -> None:
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            datetime.now(timezone.utc),
            workload_ids=["training/job/job-a"],
            containers=[
                container_observation(
                    "pod-a", "trainer", 0, "node-a", gpu_uuids=["GPU-a"]
                )
            ],
        )
    )
    event = XidEvent(
        event_id="reset-first",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=datetime.now(timezone.utc),
        xid=48,
        gpu_uuid="GPU-a",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )
    _, reset = context.orchestrator.ingest(event, context.policy.evaluate_xid(event))
    assert reset is not None
    stop = next(
        index
        for index, step in enumerate(reset.official_steps)
        if step.operation is OP.STOP_WORKLOADS
    )
    reset_index = next(
        index
        for index, step in enumerate(reset.official_steps)
        if step.operation is OP.RESET_GPU
    )
    started = copy_model(
        reset,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-test",
        completed_step_indexes=list(range(stop + 1)),
        step_executions=[
            workflow_step_execution(stop, OP.STOP_WORKLOADS),
            workflow_step_execution(
                reset_index,
                OP.RESET_GPU,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/reset-test",
            ),
        ],
    )
    context.store.save_workflow(started)
    _, isolated = context.orchestrator.ingest_node_health(
        health_finding("network-after-stop", RecoveryAction.QUARANTINE, active=True)
    )

    assert isolated is not None
    assert isolated.request_id == started.request_id
    assert OP.QUARANTINE in live_operations(isolated)
    assert OP.RESTORE_SCHEDULING not in live_operations(isolated)
    assert OP.RESTART_WORKLOAD not in live_operations(isolated)
    assert isolated.step_executions == started.step_executions
    assert isolated.official_steps[reset_index].gpu_uuids == ["GPU-a"]
    assert reset_index not in isolated.superseded_step_indexes
    assert stop not in isolated.superseded_step_indexes
    assert context.store.get_incident_by_event("network-after-stop") is not None


def test_post_stop_quarantine_is_not_record_only(health_context):
    prove_quarantine_after_completed_stop(health_context)


def prove_reboot_preserves_quarantine(context: ApplicationContext) -> None:
    _, first = context.orchestrator.ingest_node_health(
        health_finding("network-first", RecoveryAction.QUARANTINE)
    )
    assert first is not None

    _, later = context.orchestrator.ingest_node_health(
        health_finding("inventory-second", RecoveryAction.REBOOT_NODE)
    )

    assert later is not None
    assert OP.QUARANTINE in live_operations(later)
    assert OP.RESTORE_SCHEDULING not in live_operations(later)
    assert OP.RESTART_WORKLOAD not in live_operations(later)
    assert terminal_quarantine_nodes(later) == frozenset({"node-a"})


def test_later_reboot_cannot_undo_persistent_quarantine(health_context):
    prove_reboot_preserves_quarantine(health_context)


def merger() -> WorkflowMergeService:
    arbiter = RecoveryArbiter()
    return WorkflowMergeService(
        arbiter,
        DagBrancher(arbiter),
        preemption_enabled=True,
        workload_scoped_operations=set(WORKLOAD_SCOPED_OPERATIONS),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=NodeConflictService.workflow_resource_claims_by_node,
    )


def quarantine_plan() -> WorkflowRequest:
    return workflow_request(
        "quarantine", "incident", official_steps=[workflow_step(OP.QUARANTINE)]
    )


@pytest.mark.parametrize("operation", [OP.MARK_UNSCHEDULABLE, OP.QUARANTINE])
def test_scheduler_mutation_is_not_treated_as_read_only_after_stop(operation):
    existing = workflow_request(
        "existing",
        "incident",
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        official_steps=[
            workflow_step(OP.STOP_WORKLOADS),
            workflow_step(OP.RESET_GPU, gpu_uuids=["GPU-a"]),
            workflow_step(OP.RESTORE_SCHEDULING),
            workflow_step(OP.RESTART_WORKLOAD),
        ],
    )
    candidate = workflow_request(
        "candidate", "incident", official_steps=[workflow_step(operation)]
    )

    assert merger().disposition(existing, candidate, "node-a", set()) != (
        "ABSORB_RECORD_ONLY"
    )
    assert OP.QUARANTINE not in NODE_MUTATING_OPERATIONS


@pytest.mark.parametrize("raw", [None, True, "node-a", [None], [[]], ["node-b"]])
def test_malformed_explicit_quarantine_scope_fails_closed(raw):
    plan = quarantine_plan()
    plan.official_steps[0].parameters[TERMINAL_QUARANTINE_NODES] = raw
    with pytest.raises(ValueError, match="quarantine scope"):
        terminal_quarantine_nodes(plan)


def test_temporary_and_persistent_quarantine_have_different_final_constraints():
    persistent = quarantine_plan()
    temporary = copy_model(
        persistent,
        official_steps=[
            workflow_step(OP.QUARANTINE),
            workflow_step(OP.RESTART_NODE),
            workflow_step(OP.RESTORE_SCHEDULING),
        ],
    )
    assert terminal_quarantine_nodes(temporary) == frozenset()
    assert terminal_quarantine_nodes(persistent) == frozenset({"node-a"})
    assert not terminal_quarantine_covered(temporary, persistent), (
        "the temporary taint does not cover a persistent hold"
    )
    assert terminal_quarantine_covered(persistent, temporary), (
        "a candidate without a final hold adds no final-state constraint"
    )


def test_existing_explicit_hold_with_pending_readmission_is_not_coverage():
    persistent = quarantine_plan()
    broken = copy_model(
        persistent,
        official_steps=[
            workflow_step(
                OP.QUARANTINE, parameters={TERMINAL_QUARANTINE_NODES: ["node-a"]}
            ),
            workflow_step(OP.RESTORE_SCHEDULING),
        ],
    )
    assert not terminal_quarantine_covered(broken, persistent), (
        "the planned readmission conflicts even with an explicit quarantine marker"
    )


def test_shared_restore_is_narrowed_without_releasing_the_isolated_node():
    existing = workflow_request(
        "existing",
        "incident",
        official_steps=[
            workflow_step(
                OP.RESTORE_SCHEDULING,
                node_ids=["node-a", "node-b"],
                parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": []}},
            ),
            workflow_step(OP.RESTART_WORKLOAD, node_ids=["node-a", "node-b"]),
        ],
    )

    steps, retired = suppress_readmission(
        existing, existing.official_steps, frozenset({"node-a"})
    )

    assert retired == set()
    assert steps[0].node_ids == ["node-b"]
    assert steps[0].parameters["gpu_uuids_by_node"] == {"node-b": []}
    assert steps[1] == existing.official_steps[1]
    assert existing.official_steps[0].node_ids == ["node-a", "node-b"]


def test_submitted_readmission_is_never_rewritten():
    existing = workflow_request(
        "existing",
        "incident",
        official_steps=[workflow_step(OP.RESTORE_SCHEDULING)],
        step_executions=[
            workflow_step_execution(
                0, OP.RESTORE_SCHEDULING, WorkflowStepStatus.WAITING
            )
        ],
    )
    steps, retired = suppress_readmission(
        existing, existing.official_steps, frozenset({"node-a"})
    )
    assert steps == existing.official_steps
    assert retired == set()


def test_reset_inherits_quarantine_without_reintroducing_retired_steps():
    existing = quarantine_plan()
    candidate = workflow_request(
        "reset",
        "incident",
        official_steps=[
            workflow_step(OP.RESET_GPU),
            workflow_step(OP.RESTORE_SCHEDULING),
            workflow_step(OP.RESTART_WORKLOAD),
        ],
    )
    inherited = inherit_terminal_quarantine(existing, candidate)
    assert terminal_quarantine_nodes(inherited) == frozenset({"node-a"})
    assert live_operations(inherited) == {OP.RESET_GPU, OP.QUARANTINE}
    assert inherit_terminal_quarantine(existing, inherited) == inherited

    joined = DagBrancher(RecoveryArbiter()).append_parallel_job_branch(
        existing, candidate
    )
    assert live_operations(joined) == {OP.RESET_GPU, OP.QUARANTINE}
    assert candidate.superseded_step_indexes == []
