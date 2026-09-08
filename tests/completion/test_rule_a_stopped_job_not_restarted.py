"""Rule A's last sentence: nothing restarts a job the system itself stopped.

When a job workflow gives up waiting on a busy node, the dispatcher rewrites
it to a STOP_WORKLOADS carrying ``termination_initiator_incident_id`` -- the
job's own incident -- and the workflow ends FAILED / ESCALATED. The terminal
event the data plane then reports names that initiator, and the completion
service must answer NO_ACTION without planning a new restart, whether the
job's containers went out with a clean stop or a failing exit code.
"""

from __future__ import annotations

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    RecoveryAction,
    TerminalEvent,
    TerminalStatus,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step

JOB_INCIDENT = "inc-job"
STOPPED_BY_RULE_A = (
    "NODE_REMEDIATION_TIMEOUT: nodes still under remediation after 240s: "
    "node-a (wf-node)"
)


def _job_stopped_by_rule_a(context: ApplicationContext, event: TerminalEvent) -> None:
    incident = fault_incident(
        JOB_INCIDENT,
        "event-job",
        cluster_id=event.cluster_id,
        node_ids=["node-a", "node-b"],
        job_id=event.job_id,
        attempt_id=event.attempt_id,
        state=IncidentState.ESCALATED,
        workflow_request_id="wf-job",
    )
    workflow = workflow_request(
        "wf-job",
        JOB_INCIDENT,
        status=WorkflowStatus.FAILED,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                node_ids=["node-a", "node-b"],
                workload_ids=["training/pytorchjob/distributed-training"],
                parameters={"termination_initiator_incident_id": JOB_INCIDENT},
            )
        ],
        completed_step_indexes=[0],
        terminal_failure_reason=STOPPED_BY_RULE_A,
    )
    context.store.save_incident_and_workflow(incident, workflow)


def _counts(context: ApplicationContext) -> tuple[int, int]:
    return len(context.store._incidents), len(context.store.list_workflows())


@pytest.mark.parametrize(
    "terminal_status",
    [TerminalStatus.STOPPED, TerminalStatus.FAILED],
    ids=["clean stop", "failing exit under the stop"],
)
def test_a_job_stopped_by_rule_a_is_not_restarted(
    context: ApplicationContext,
    failed_event: TerminalEvent,
    terminal_status: TerminalStatus,
) -> None:
    _job_stopped_by_rule_a(context, failed_event)
    before = _counts(context)
    terminal = copy_model(
        failed_event,
        terminal_status=terminal_status,
        termination_initiator_incident_id=JOB_INCIDENT,
    )

    decision = context.completion.handle_terminal(terminal)

    assert decision.status is DecisionStatus.NO_ACTION, decision
    assert decision.recovery_plan_id is None, decision
    assert JOB_INCIDENT in decision.reason, decision.reason
    assert _counts(context) == before, "no new incident or workflow may be planned"
    assert context.store.get_workflow("wf-job").status is WorkflowStatus.FAILED
    assert context.store.get_incident(JOB_INCIDENT).state is IncidentState.ESCALATED


def test_the_same_exit_without_the_initiator_marker_is_still_recovered(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    """The control: it is the initiator marker, not the FAILED workflow's
    presence, that keeps the job from being restarted."""

    _job_stopped_by_rule_a(context, failed_event)

    decision = context.completion.handle_terminal(failed_event)
    plan = context.store.get_plan(decision.recovery_plan_id)

    assert decision.status is DecisionStatus.PLAN_CREATED, decision
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD]
