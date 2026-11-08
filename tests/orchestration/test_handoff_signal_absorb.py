"""A signal that re-arms on a node its own operator hand-off holds is recorded
on that hand-off and mints nothing.

Live 2026-11-08 (HyperPod, GF-GENERAL-LIVE-013 post_repair_isolation_lifted):
an EFA driver chain's VALIDATE_FABRIC failed, its reboot rung was refused for
want of observed isolation and the refusal was handed over (support-after at
``containment_or_release``: the isolation put back, one ticket). The chain's
TRIGGER_HEALTH_SNAPSHOT had restarted the host collector, whose inventory
verdict counts consecutive samples in memory: its first tick read 0 -- a clear
to the health signal -- and its next one, 15 s later, 1 -- a new episode under
a new event id, after the whole chain had ended. That finding minted a second
incident, its MARK_UNSCHEDULABLE was refused because the hand-off's
quarantine holds the node, and its ``containment_refused`` successor opened a
second support ticket for the one fault. The batch timestamps below are the
live ones.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.orchestration.failure_outcome import (
    ISOLATION_NOT_ISOLATED,
    LIFECYCLE_ISOLATION_DETAIL,
)
from gpu_fault.orchestration.families.health import handoff_holds_node
from tests._builders import copy_model, node_health_finding, workflow_step_execution

OP = WorkflowOperation
NODE = "hyperpod-node-a"
#: The batch timestamps (microseconds) of the live finding and its re-arm.
FIRST_BATCH = 1794140939456058
REARM_BATCH = 1794140974984554
DRIVER_UNBOUND = "EFA PCI device is present but the efa driver is not bound"
PCI_MISSING = "EFA PCI device inventory is below the configured node invariant"


def _finding(
    batch: int,
    *,
    metric: str = "efa_inventory_mismatch",
    node_id: str = NODE,
    reason: str = DRIVER_UNBOUND,
    action: RecoveryAction = RecoveryAction.REMEDIATE_EFA_DRIVER,
    failure_mode: str = "DRIVER_UNBOUND",
) -> NodeHealthFinding:
    # ``NodeHealthPolicy.evaluate_metrics``: ``<batch>-<metric>-<device|node>``
    # with the collector's ``host-<node>-<observed microseconds>`` batch id.
    event_id = f"host-{node_id}-{batch}-{metric}-node"
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        node_id=node_id,
        observed_at=datetime(2026, 11, 8, 12, 29, tzinfo=timezone.utc),
        category=NodeHealthCategory.RDMA,
        severity=Severity.CRITICAL,
        reason=reason,
        recommended_action=action,
        metric_name=metric,
        value=1.0,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
        diagnostic_parameters={
            "failure_mode": failure_mode,
            "expected_count": "16",
            "observed_count": "15",
            "missing_count": "1",
        },
    )


def _ran(
    workflow: WorkflowRequest,
    failed_index: int | None = None,
    *,
    error: str | None = None,
    details: dict | None = None,
) -> list:
    executions = []
    for index, step in enumerate(workflow.official_steps):
        if index == failed_index:
            executions.append(
                workflow_step_execution(
                    index,
                    step.operation,
                    WorkflowStepStatus.FAILED,
                    error=error,
                    details=details or {},
                )
            )
            break
        executions.append(workflow_step_execution(index, step.operation))
    return executions


def _finish(
    context: ApplicationContext,
    workflow: WorkflowRequest,
    executions: list,
    status: WorkflowStatus,
    state: IncidentState,
) -> WorkflowRequest:
    """The record as the executor terminalizes it, its incident in ``state``."""

    stored = context.store.get_workflow(workflow.request_id)
    succeeded = [
        item for item in executions if item.status is WorkflowStepStatus.SUCCEEDED
    ]
    context.store.save_workflow(
        copy_model(
            stored,
            status=status,
            step_executions=executions,
            completed_step_indexes=[item.step_index for item in succeeded],
            completed_operations=[item.operation for item in succeeded],
        ),
        expected=stored,
    )
    incident = context.store.get_incident(stored.incident_id)
    context.store.save_incident(copy_model(incident, state=state), expected=incident)
    return context.store.get_workflow(workflow.request_id)


def _handed_over(context: ApplicationContext) -> WorkflowRequest:
    """The live chain up to its one hand-off: the EFA driver chain fails at
    VALIDATE_FABRIC, its reboot rung is refused for want of isolation, and the
    refusal's support-after puts the isolation back and opens the ticket."""

    incident, workflow = context.orchestrator.ingest_node_health(_finding(FIRST_BATCH))
    assert workflow is not None
    assert [step.operation for step in workflow.official_steps][2:6] == [
        OP.REMEDIATE_EFA_DRIVER,
        OP.RESTART_EFA_DEVICE_PLUGIN,
        OP.TRIGGER_HEALTH_SNAPSHOT,
        OP.VALIDATE_FABRIC,
    ]
    failed = _finish(
        context,
        workflow,
        _ran(
            workflow,
            5,
            error=f"validation failed on nodes: {NODE} "
            "(efa_inventory_active_count=15,expected=16)",
            details={
                "failed_nodes": [NODE],
                "node_failures": {NODE: ["efa_inventory_active_count=15,expected=16"]},
                "validation": "VALIDATE_FABRIC",
            },
        ),
        WorkflowStatus.FAILED,
        IncidentState.QUARANTINED,
    )
    rung = context.orchestrator.escalate_failed_hardware_remediation(failed)
    assert rung is not None
    _, reboot = rung
    assert OP.RESTART_NODE in {step.operation for step in reboot.official_steps}
    refused = _finish(
        context,
        reboot,
        _ran(
            reboot,
            3,
            error=f"node {NODE} is not isolated: spec.unschedulable is not true",
            details={
                "safety_rejection": True,
                LIFECYCLE_ISOLATION_DETAIL: ISOLATION_NOT_ISOLATED,
                "node_failures": {
                    NODE: [
                        "RESTART_NODE not submitted, isolation not observed: "
                        "spec.unschedulable is not true"
                    ]
                },
            },
        ),
        WorkflowStatus.FAILED,
        IncidentState.QUARANTINED,
    )
    hand_off = context.orchestrator.escalate_failed_hardware_remediation(refused)
    assert hand_off is not None
    _, support = hand_off
    assert support.request_id == (
        f"workflow-support-after-workflow-reboot-after-{workflow.request_id}"
    )
    assert [step.operation for step in support.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.MARK_UNSCHEDULABLE,
        OP.QUARANTINE,
        OP.ESCALATE_SUPPORT,
    ]
    return _finish(
        context,
        support,
        _ran(support),
        WorkflowStatus.SUCCEEDED,
        IncidentState.ESCALATED,
    )


def _support_hand_offs(context: ApplicationContext) -> list[str]:
    return sorted(
        workflow.request_id
        for workflow in context.store.list_workflows()
        if OP.ESCALATE_SUPPORT in {step.operation for step in workflow.official_steps}
    )


def test_the_re_armed_finding_joins_the_open_hand_off_and_mints_nothing(
    context: ApplicationContext,
) -> None:
    support = _handed_over(context)
    workflows = {workflow.request_id for workflow in context.store.list_workflows()}
    rearm = _finding(REARM_BATCH)

    incident, workflow = context.orchestrator.ingest_node_health(rearm)
    replay = context.orchestrator.ingest_node_health(rearm)

    assert incident.incident_id == support.incident_id
    assert workflow is not None and workflow.request_id == support.request_id
    assert {item.request_id for item in context.store.list_workflows()} == workflows, (
        "no second incident/workflow"
    )
    assert context.store.get_incident_by_event(f"inc-{rearm.event_id}") is None
    assert _support_hand_offs(context) == [support.request_id], (
        "one fault, one hand-off"
    )
    linked = context.store.get_incident_by_event(rearm.event_id)
    assert linked is not None and linked.incident_id == support.incident_id
    stored = context.store.get_incident(support.incident_id)
    assert stored.state is IncidentState.ESCALATED, "the operator still owns it"
    absorbed = [line for line in stored.reasons if rearm.event_id in line]
    assert absorbed == [
        f"absorbed efa_inventory_mismatch finding {rearm.event_id} on node: "
        "recorded only, the operator hand-off holds the node for this signal"
    ]
    assert replay[0].incident_id == support.incident_id
    assert handoff_holds_node(context.store.get_workflow(support.request_id), NODE), (
        "the hand-off keeps the node cordoned and quarantined"
    )


@pytest.mark.parametrize(
    "rearm",
    [
        pytest.param(
            _finding(REARM_BATCH, metric="efa_kubernetes_allocatable_mismatch"),
            id="another-signal",
        ),
        pytest.param(
            _finding(
                REARM_BATCH,
                reason=PCI_MISSING,
                action=RecoveryAction.REBOOT_NODE,
                failure_mode="PCI_DEVICE_MISSING",
            ),
            id="same-signal-another-failure-mode",
        ),
        pytest.param(
            _finding(REARM_BATCH, node_id="hyperpod-node-b"), id="another-node"
        ),
    ],
)
def test_a_genuinely_new_fault_still_opens_its_own_incident(
    context: ApplicationContext, rearm: NodeHealthFinding
) -> None:
    support = _handed_over(context)

    incident, workflow = context.orchestrator.ingest_node_health(rearm)

    assert incident.incident_id == f"inc-{rearm.event_id}"
    assert workflow is not None and workflow.request_id != support.request_id
    assert not any(
        rearm.event_id in line
        for line in context.store.get_incident(support.incident_id).reasons
    ), "the hand-off records nothing of another fault"


def test_a_hand_off_the_operator_closed_absorbs_nothing(
    context: ApplicationContext,
) -> None:
    support = _handed_over(context)
    handed = context.store.get_incident(support.incident_id)
    context.store.save_incident(
        copy_model(handed, state=IncidentState.RECOVERED), expected=handed
    )
    rearm = _finding(REARM_BATCH)

    incident, _ = context.orchestrator.ingest_node_health(rearm)

    assert incident.incident_id == f"inc-{rearm.event_id}"


def test_a_hand_off_that_released_the_node_absorbs_nothing(
    context: ApplicationContext,
) -> None:
    support = _handed_over(context)
    stored = context.store.get_workflow(support.request_id)
    released = [
        *stored.official_steps,
        copy_model(stored.official_steps[1], operation=OP.RESTORE_SCHEDULING),
    ]
    context.store.save_workflow(
        copy_model(
            stored,
            official_steps=released,
            step_executions=[
                *stored.step_executions,
                workflow_step_execution(4, OP.RESTORE_SCHEDULING),
            ],
        ),
        expected=stored,
    )
    assert not handoff_holds_node(
        context.store.get_workflow(support.request_id), NODE
    ), "a completed RESTORE_SCHEDULING released the node"
    rearm = _finding(REARM_BATCH)

    incident, _ = context.orchestrator.ingest_node_health(rearm)

    assert incident.incident_id == f"inc-{rearm.event_id}"
