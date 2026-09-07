"""An UNKNOWN workload state blocks any plan that mutates the node, in both
families (ARCH-B2).

The fault family (coordinator) already refused to compile a node-mutating plan
when the event's workload state was UNKNOWN; the node-health family only asked
"is it ACTIVE?" to decide whether to insert STOP_WORKLOADS, so an
``efa_inventory_mismatch`` REBOOT_NODE plan for a node whose workload state was
unknown restarted the node under a training job without stopping it first. The
gate now lives in the one ``compile_steps`` both families pass through.
"""

from __future__ import annotations

from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from tests._builders import node_health_finding

from ._support import (
    NOW,
    ApplicationContext,
    _inventory_mismatch_finding,
    _save_attempt,
    event,
    ingest,
    pytest,
)

UNKNOWN_REASON = "node workload state is UNKNOWN"


def _efa_inventory_reboot(
    *,
    event_id: str,
    workload_state: WorkloadState,
    affected_workload_ids: list[str] | None = None,
) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        observed_at=NOW,
        category=NodeHealthCategory.RDMA,
        severity="critical",
        metric_name="efa_inventory_mismatch",
        reason="EFA device inventory does not match the configured node invariant",
        recommended_action=RecoveryAction.REBOOT_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=workload_state,
        affected_workload_ids=affected_workload_ids or [],
        diagnostic_parameters={"expected_count": "16"},
    )


def _operations(workflow) -> list[WorkflowOperation]:
    return [step.operation for step in workflow.official_steps]


def test_health_reboot_plan_with_unknown_workload_state_is_blocked(
    context: ApplicationContext,
) -> None:
    incident, workflow = context.orchestrator.ingest_node_health(
        _efa_inventory_reboot(
            event_id="efa-unknown-workload", workload_state=WorkloadState.UNKNOWN
        )
    )

    assert workflow is not None, "the finding still produces a record"
    assert workflow.status is WorkflowStatus.BLOCKED, workflow.status
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR, workflow.blocked_kind
    assert UNKNOWN_REASON in workflow.blocked_reasons, workflow.blocked_reasons
    assert WorkflowOperation.RESTART_NODE in _operations(workflow), (
        "the plan is kept for the operator to read, it is just not executable"
    )
    assert incident.state is IncidentState.ESCALATED, incident.state
    assert context.store.get_workflow(workflow.request_id).status is (
        WorkflowStatus.BLOCKED
    ), "the BLOCKED verdict is what is persisted"


@pytest.mark.parametrize(
    "action",
    [
        RecoveryAction.REBOOT_NODE,
        RecoveryAction.REPLACE_NODE,
        RecoveryAction.REMEDIATE_EFA_DRIVER,
        RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
    ],
)
def test_every_node_mutating_health_action_is_gated_on_unknown_workload_state(
    context: ApplicationContext, action: RecoveryAction
) -> None:
    finding = _efa_inventory_reboot(
        event_id=f"unknown-{action.value.lower()}", workload_state=WorkloadState.UNKNOWN
    ).model_copy(update={"recommended_action": action})

    _, workflow = context.orchestrator.ingest_node_health(finding)

    assert workflow is not None, "the finding still produces a record"
    assert workflow.status is WorkflowStatus.BLOCKED, (action, workflow.status)
    assert UNKNOWN_REASON in workflow.blocked_reasons, workflow.blocked_reasons


def test_health_plan_without_node_mutation_is_not_gated_on_unknown_state(
    context: ApplicationContext,
) -> None:
    finding = _efa_inventory_reboot(
        event_id="unknown-quarantine", workload_state=WorkloadState.UNKNOWN
    ).model_copy(update={"recommended_action": RecoveryAction.QUARANTINE})

    _, workflow = context.orchestrator.ingest_node_health(finding)

    assert workflow is not None, "the finding still produces a record"
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert UNKNOWN_REASON not in workflow.blocked_reasons, (
        "containment must not be withheld because the workload state is unknown"
    )


def test_health_reboot_plan_with_active_workload_stops_the_job_first(
    context: ApplicationContext,
) -> None:
    _save_attempt(context, ("node-a",))
    _, workflow = context.orchestrator.ingest_node_health(
        _efa_inventory_reboot(
            event_id="efa-active-workload",
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=["training/job/job-a"],
        )
    )

    assert workflow is not None, "the finding still produces a record"
    operations = _operations(workflow)
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert WorkflowOperation.STOP_WORKLOADS in operations, operations
    assert operations.index(WorkflowOperation.STOP_WORKLOADS) < operations.index(
        WorkflowOperation.RESTART_NODE
    ), operations


def test_health_reboot_plan_with_idle_workload_compiles_executable(
    context: ApplicationContext,
) -> None:
    _, workflow = context.orchestrator.ingest_node_health(
        _efa_inventory_reboot(
            event_id="efa-idle-workload", workload_state=WorkloadState.IDLE
        )
    )

    assert workflow is not None, "the finding still produces a record"
    operations = _operations(workflow)
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert workflow.blocked_kind is None, workflow.blocked_kind
    assert WorkflowOperation.RESTART_NODE in operations, operations
    assert WorkflowOperation.STOP_WORKLOADS not in operations, operations


def test_gpu_inventory_reboot_with_unknown_workload_state_is_blocked_too(
    context: ApplicationContext,
) -> None:
    _, workflow = context.orchestrator.ingest_node_health(
        _inventory_mismatch_finding(
            event_id="gpu-inventory-unknown", workload_state=WorkloadState.UNKNOWN
        )
    )

    assert workflow is not None, "the finding still produces a record"
    assert workflow.status is WorkflowStatus.BLOCKED, workflow.status
    assert UNKNOWN_REASON in workflow.blocked_reasons, workflow.blocked_reasons


def test_fault_family_gate_on_unknown_workload_state_is_unchanged(
    context: ApplicationContext,
) -> None:
    _, _, workflow = ingest(
        context,
        event(48, event_id="xid48-unknown", workload_state=WorkloadState.UNKNOWN),
    )

    assert workflow.status is not WorkflowStatus.PENDING, workflow.status
    assert UNKNOWN_REASON in workflow.blocked_reasons, workflow.blocked_reasons
