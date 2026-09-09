"""A diagnostic marker never makes its incident the owner of a failed attempt.

A WARNING-level ``RUN_DIAGNOSTICS`` marker (CPU at 98 % for two minutes, a
disk-await blip) points at an incident whose workflow only observes the node
(``FREEZE_EVIDENCE`` + ``VALIDATE_HOST``). When the training attempt on that
node then failed, the completion watcher matched the marker, decided that
"node remediation is owned by the existing incident" and planned a restart
gated on ``requires_incident_state: RECOVERED``. Nothing was repairing the
node, the incident closed as RECOVERED or ESCALATED and the marker outlived
both until its TTL, so the job could not restart automatically for an hour.

Only a marker whose recommended action changes the node (reboot, replace, GPU
reset, quarantine, ...) may claim the attempt's recovery. Advisory markers are
skipped and the terminal is decided as if they were absent. The
classification is derived from the operation registry, not a second list.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.markers import (
    DIAGNOSTIC_ACTIONS,
    SPARE_BLOCKING_ACTIONS,
    marker_is_diagnostic,
)
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    TerminalEvent,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import fault_incident, workflow_request, workflow_step

DIAGNOSTIC = (
    RecoveryAction.RUN_DIAGNOSTICS,
    RecoveryAction.COLLECT_EVIDENCE,
    RecoveryAction.VALIDATE_NODE,
)


def _marker(
    ended_at: datetime,
    *,
    action: RecoveryAction,
    incident_id: str = "inc-diagnostic",
    severity: Severity = Severity.WARNING,
) -> NodeMarker:
    return NodeMarker(
        marker_id="marker-diagnostic",
        source="host-collector",
        trusted=True,
        incident_id=incident_id,
        observed_at=ended_at - timedelta(seconds=10),
        expires_at=ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity=severity,
        recommended_action=action,
        action_owner="simulated-runtime",
        mapping_version="test-v1",
    )


def _diagnostic_incident(
    context: ApplicationContext, *, operations: list[WorkflowOperation]
) -> None:
    context.store.save_incident(
        fault_incident(
            "inc-diagnostic",
            "event-diagnostic",
            event_type="HOST_TELEMETRY",
            node_ids=["node-a"],
            state=IncidentState.ACTION_PENDING,
            workflow_request_id="wf-diagnostic",
        )
    )
    context.store.save_workflow(
        workflow_request(
            "wf-diagnostic",
            "inc-diagnostic",
            WorkflowStatus.RUNNING,
            official_steps=[workflow_step(operation) for operation in operations],
        )
    )


def test_diagnostic_actions_are_derived_from_the_registry() -> None:
    assert DIAGNOSTIC_ACTIONS == frozenset(DIAGNOSTIC)
    assert not (DIAGNOSTIC_ACTIONS & SPARE_BLOCKING_ACTIONS), (
        "an action cannot both block a spare and be advisory"
    )


@pytest.mark.parametrize("action", DIAGNOSTIC, ids=[item.value for item in DIAGNOSTIC])
def test_a_diagnostic_marker_does_not_claim_the_terminal(
    context: ApplicationContext,
    failed_event: TerminalEvent,
    ended_at: datetime,
    action: RecoveryAction,
) -> None:
    _diagnostic_incident(
        context,
        operations=[WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.VALIDATE_HOST],
    )
    marker = _marker(ended_at, action=action)
    context.completion.add_marker(marker)
    assert marker_is_diagnostic(marker) is True

    decision = context.completion.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PLAN_CREATED, decision
    assert decision.matched_marker_ids == []
    plan = context.store.get_plan(decision.recovery_plan_id)
    assert plan.trigger == "no-hardware-evidence:RESTART", plan
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD]
    (stored,) = context.store.list_markers()
    assert stored.active is True, "an observation is not retired by a terminal"


def test_a_repair_marker_still_pins_the_restart_to_its_incident(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    """Regression guard: the reboot/replace chain keeps the RECOVERED gate."""
    _diagnostic_incident(
        context,
        operations=[
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
    )
    marker = _marker(
        ended_at, action=RecoveryAction.REBOOT_NODE, severity=Severity.CRITICAL
    )
    context.completion.add_marker(marker)
    assert marker_is_diagnostic(marker) is False

    decision = context.completion.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PLAN_CREATED, decision
    assert decision.matched_marker_ids == ["marker-diagnostic"]
    plan = context.store.get_plan(decision.recovery_plan_id)
    assert plan.steps[0].action is RecoveryAction.RESTART_WORKLOAD
    assert plan.restart_after_incident_id == "inc-diagnostic"


def test_the_repair_marker_wins_when_both_kinds_match(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    _diagnostic_incident(
        context,
        operations=[WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.RESET_GPU],
    )
    context.completion.add_marker(
        _marker(ended_at, action=RecoveryAction.RUN_DIAGNOSTICS)
    )
    repair = _marker(
        ended_at, action=RecoveryAction.RESET_GPU, severity=Severity.CRITICAL
    )
    context.completion.add_marker(
        repair.model_copy(update={"marker_id": "marker-reset"})
    )

    decision = context.completion.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PLAN_CREATED
    assert decision.matched_marker_ids == ["marker-reset"]
