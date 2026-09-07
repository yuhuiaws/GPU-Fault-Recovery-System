"""The hardware escalation is bounded: a refused containment is handed to an
operator without re-planning the isolation, and a support escalation that
fails is never answered with another support escalation.

ARCH-E2E-2A finding 1 (DESTR-020). Group A made a ``read_node`` 404 on a node
that must be isolated a FAILED ``MARK_UNSCHEDULABLE`` carrying
``safety_rejection`` + ``absent``. Classified ``containment_or_release``, that
failure opened a support workflow on the same node whose MARK_UNSCHEDULABLE
hit the same 404, failed, and was escalated again by the dispatcher's failed
workflow reconcile: one new FAILED incident/workflow pair per tick, forever.
``failure_handling_max_attempts`` bounds only a handler that raises.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
QUARANTINE = WorkflowOperation.QUARANTINE
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT
ALIAS = "acceptance-alias-0123456789ab"
ABSENT_REFUSAL = {"safety_rejection": True, "node_id": ALIAS, "absent": True}


class _StubBuilder:
    """Compiles one step per operation with the requested scope."""

    def compile_steps(
        self,
        operations: list[WorkflowOperation],
        profile: Any,
        node_ids: list[str],
        gpu_uuids: list[str],
        workload_ids: list[str] | None = None,
    ) -> tuple[list[Any], list[str]]:
        return (
            [
                workflow_step(
                    operation,
                    node_ids=list(node_ids),
                    gpu_uuids=list(gpu_uuids),
                    workload_ids=list(workload_ids or []),
                )
                for operation in operations
            ],
            [],
        )


def _operations(workflow: Any) -> list[WorkflowOperation]:
    return [step.operation for step in workflow.official_steps]


def _failed_isolation(
    store: Any,
    *,
    incident_id: str,
    event_id: str,
    request_id: str,
    steps: list[Any],
    executions: list[Any],
) -> tuple[Any, Any]:
    """A FAILED workflow whose incident the terminal funnel left ESCALATED
    (nothing was isolated before the failure)."""

    store.save_profile(default_simulated_profile())
    incident = fault_incident(
        incident_id,
        event_id,
        event_type="XID",
        state=IncidentState.ESCALATED,
        effective_action=RecoveryAction.ESCALATE_OPERATOR,
        workflow_request_id=request_id,
        node_ids=[ALIAS],
        policy_version="610",
        policy_source="SITE_SAFETY",
        fencing_token=1,
    )
    workflow = workflow_request(
        request_id,
        incident_id,
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_steps=steps,
        completed_step_indexes=[
            execution.step_index
            for execution in executions
            if execution.status is WorkflowStepStatus.SUCCEEDED
        ],
        completed_operations=[
            execution.operation
            for execution in executions
            if execution.status is WorkflowStepStatus.SUCCEEDED
        ],
        step_executions=executions,
    )
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow


def _safety_plan() -> list[Any]:
    return [
        workflow_step(FREEZE, node_ids=[ALIAS]),
        workflow_step(MARK, node_ids=[ALIAS]),
        workflow_step(QUARANTINE, node_ids=[ALIAS]),
    ]


def _refused_at_isolation() -> list[Any]:
    return [
        workflow_step_execution(0, FREEZE),
        workflow_step_execution(
            1,
            MARK,
            WorkflowStepStatus.FAILED,
            error=f"node {ALIAS} is absent; isolation cannot be observed",
            details=dict(ABSENT_REFUSAL),
        ),
    ]


# ------------------------------------------------------- (a) refused containment


def test_a_refused_isolation_is_escalated_once_without_planning_the_isolation_again() -> (
    None
):
    store = build_store()
    source, workflow = _failed_isolation(
        store,
        incident_id="inc-src",
        event_id="event-src",
        request_id="wf-src",
        steps=_safety_plan(),
        executions=_refused_at_isolation(),
    )
    service = HardwareEscalationService(store, _StubBuilder())

    first = service.escalate(workflow)
    second = service.escalate(workflow)

    assert first is not None, "a refused isolation must still reach an operator"
    assert second is not None, "the second pass must return the same escalation"
    escalated, support = first
    assert second[1].request_id == support.request_id, "one support workflow"
    assert escalated.effective_action is RecoveryAction.ESCALATE_OPERATOR, (
        escalated.effective_action
    )
    assert escalated.node_ids == [ALIAS], escalated.node_ids
    assert escalated.reasons[0].startswith("containment_refused"), escalated.reasons
    assert _operations(support) == [FREEZE, SUPPORT], _operations(support)
    assert support.status is WorkflowStatus.PENDING, support.status
    assert len(store.list_workflows()) == 2, [
        item.request_id for item in store.list_workflows()
    ]
    assert store.get_incident(source.incident_id).state is IncidentState.ESCALATED, (
        "the source incident stays ESCALATED"
    )
    assert service.containment_refused_escalations_total == 1, (
        service.containment_refused_escalations_total
    )
    assert service.escalation_chain_terminated_total == 0, (
        "a first-order escalation is not a terminated chain"
    )


def test_a_refused_release_is_also_handed_over_without_a_new_cordon() -> None:
    store = build_store()
    steps = [
        workflow_step(MARK, node_ids=[ALIAS]),
        workflow_step(RESTORE, node_ids=[ALIAS]),
    ]
    _, workflow = _failed_isolation(
        store,
        incident_id="inc-src",
        event_id="event-src",
        request_id="wf-src",
        steps=steps,
        executions=[
            workflow_step_execution(0, MARK),
            workflow_step_execution(
                1,
                RESTORE,
                WorkflowStepStatus.FAILED,
                error=f"node {ALIAS} isolation ownership does not match",
                details={"safety_rejection": True, "node_id": ALIAS},
            ),
        ],
    )

    result = HardwareEscalationService(store, _StubBuilder()).escalate(workflow)

    assert result is not None, "a refused release must still reach an operator"
    _, support = result
    assert MARK not in _operations(support), _operations(support)
    assert QUARANTINE not in _operations(support), _operations(support)
    assert SUPPORT in _operations(support), _operations(support)


# ----------------------------------------------------------- (b) chain bound


def _support_plan() -> list[Any]:
    return [*_safety_plan(), workflow_step(SUPPORT, node_ids=[ALIAS])]


@pytest.mark.parametrize(
    "executions",
    [
        pytest.param(_refused_at_isolation(), id="isolation-refused-again"),
        pytest.param(
            [
                workflow_step_execution(0, FREEZE),
                workflow_step_execution(
                    1,
                    MARK,
                    WorkflowStepStatus.FAILED,
                    error="workflow lifetime exceeded",
                    details={"workflow_lifetime_exceeded": True},
                ),
            ],
            id="lifetime-exceeded",
        ),
        pytest.param(
            [
                workflow_step_execution(0, FREEZE),
                workflow_step_execution(
                    1, MARK, WorkflowStepStatus.FAILED, error="cordon refused"
                ),
            ],
            id="plain-containment-failure",
        ),
    ],
)
def test_a_failed_support_escalation_terminates_the_chain(
    executions: list[Any],
) -> None:
    store = build_store()
    support_incident, support_workflow = _failed_isolation(
        store,
        incident_id="inc-support-after-wf-src",
        event_id="support-after-wf-src",
        request_id="workflow-support-after-wf-src",
        steps=_support_plan(),
        executions=executions,
    )
    service = HardwareEscalationService(store, _StubBuilder())

    first = service.escalate(support_workflow)
    second = service.escalate(support_workflow)

    assert first is None and second is None, (first, second)
    assert [item.request_id for item in store.list_workflows()] == [
        support_workflow.request_id
    ], "no second-order support workflow"
    assert (
        store.get_incident_by_event("support-after-workflow-support-after-wf-src")
        is None
    ), "no second-order support incident"
    current = store.get_incident(support_incident.incident_id)
    assert current.state is IncidentState.ESCALATED, current.state
    assert any("escalation chain terminated" in reason for reason in current.reasons), (
        current.reasons
    )
    notifications = store.list_notifications()
    assert len(notifications) == 1, [item.subject for item in notifications]
    assert notifications[0].incident_id == support_incident.incident_id, notifications[
        0
    ].incident_id
    assert service.escalation_chain_terminated_total == 1, (
        service.escalation_chain_terminated_total
    )
    assert service.escalation_chain_terminated_last_seen_timestamp_seconds > 0, (
        "the termination stamps its last-seen time"
    )


def test_the_dispatcher_stops_minting_pairs_once_the_chain_is_terminated() -> None:
    store = build_store()
    _, support_workflow = _failed_isolation(
        store,
        incident_id="inc-support-after-wf-src",
        event_id="support-after-wf-src",
        request_id="workflow-support-after-wf-src",
        steps=_support_plan(),
        executions=_refused_at_isolation(),
    )
    service = HardwareEscalationService(store, _StubBuilder())
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True),
        failure_handler=service.escalate,
    )

    for _ in range(3):
        dispatcher.run_once()

    assert [item.request_id for item in store.list_workflows()] == [
        support_workflow.request_id
    ], "reconcile ticks must not add incident/workflow pairs"
    handled = store.get_workflow(support_workflow.request_id)
    assert handled.failure_handled_at is not None, "the failure is handled once"
    assert len(store.list_notifications()) == 1, "one operator notification"
    assert service.escalation_chain_terminated_total == 1, (
        service.escalation_chain_terminated_total
    )
    assert dispatcher.failure_handling_abandoned_total == 0, (
        "termination is a handled outcome, not an abandoned handler"
    )


# ---------------------------------------------------------- (c) regression


def test_a_plain_failed_containment_still_produces_exactly_one_support_workflow() -> (
    None
):
    store = build_store()
    steps = [
        workflow_step(MARK, node_ids=["node-a"]),
        workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"]),
        workflow_step(RESTORE, node_ids=["node-a"]),
    ]
    _, workflow = _failed_isolation(
        store,
        incident_id="inc-src",
        event_id="event-src",
        request_id="wf-src",
        steps=steps,
        executions=[
            workflow_step_execution(0, MARK),
            workflow_step_execution(1, WorkflowOperation.RESET_GPU),
            workflow_step_execution(
                2, RESTORE, WorkflowStepStatus.FAILED, error="uncordon refused"
            ),
        ],
    )
    service = HardwareEscalationService(store, _StubBuilder())

    first = service.escalate(workflow)
    second = service.escalate(workflow)

    assert first is not None and second is not None, (first, second)
    escalated, support = first
    assert second[1].request_id == support.request_id, "one support workflow"
    assert escalated.reasons[0].startswith("containment_or_release"), escalated.reasons
    assert _operations(support) == [FREEZE, MARK, QUARANTINE, SUPPORT], _operations(
        support
    )
    assert len(store.list_workflows()) == 2, [
        item.request_id for item in store.list_workflows()
    ]
    assert store.list_notifications() == [], "no out-of-band notification"
    assert service.escalation_chain_terminated_total == 0, (
        service.escalation_chain_terminated_total
    )
    assert service.containment_refused_escalations_total == 0, (
        service.containment_refused_escalations_total
    )
