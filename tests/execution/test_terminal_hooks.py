"""Every terminal transition of a workflow reaches the ``on_terminal`` hooks
(ARCH-B3).

Kubernetes-side leases -- a warm-spare reservation, a node lock -- outlive the
workflow record unless something releases them when the workflow ends. The
executor owns the one funnel through which a claimed workflow ends (F-C9), so
that funnel is where the release hooks fire: for SUCCEEDED, FAILED, a BLOCKED
written by the dispatcher, and a preempted (SUPERSEDED) predecessor alike. One
failing hook is logged and does not stop the others or the terminalization.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from tests._builders import active_workflow_executor, build_store, execute_workflow
from tests.execution._support import FakeAdapter, _preempting_successor, workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE


class RecordingHook:
    def __init__(self) -> None:
        self.calls: list[
            tuple[WorkflowRequest, FaultIncident | None, list[WorkflowStepSpec]]
        ] = []

    def __call__(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident | None,
        steps: list[WorkflowStepSpec],
    ) -> None:
        self.calls.append((workflow, incident, steps))


def _executor(store, outcome: WorkflowStepOutcome, hook: RecordingHook):
    adapter = FakeAdapter({FREEZE: outcome})
    executor = active_workflow_executor(store, [adapter], {FREEZE})
    executor.on_terminal.append(hook)
    return executor


def _only_call(hook: RecordingHook):
    assert len(hook.calls) == 1, f"expected exactly one hook call, got {hook.calls}"
    return hook.calls[0]


def test_hook_fires_once_when_the_workflow_succeeds():
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE])
    hook = RecordingHook()
    executor = _executor(store, WorkflowStepOutcome.succeeded(), hook)

    result = execute_workflow(executor, workflow.request_id)
    ended, ended_incident, steps = _only_call(hook)

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert ended.request_id == workflow.request_id, ended.request_id
    assert ended.status is WorkflowStatus.SUCCEEDED, ended.status
    assert ended_incident is not None, "the incident is handed to the hook"
    assert ended_incident.incident_id == incident.incident_id, ended_incident
    assert ended_incident.state is IncidentState.RECOVERED, ended_incident.state
    assert [step.operation for step in steps] == [FREEZE], steps


def test_hook_fires_once_when_the_workflow_fails():
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    hook = RecordingHook()
    executor = _executor(store, WorkflowStepOutcome.failed("freeze refused"), hook)

    result = execute_workflow(executor, workflow.request_id)
    ended, ended_incident, _ = _only_call(hook)

    assert result.status is WorkflowStatus.FAILED, result
    assert ended.status is WorkflowStatus.FAILED, ended.status
    assert ended_incident is not None, "the incident is handed to the hook"
    assert ended_incident.state is IncidentState.ESCALATED, ended_incident.state


def test_hook_fires_when_a_predecessor_is_preempted_at_the_step_boundary():
    store = build_store()
    incident, predecessor = workflow_state(store, [FREEZE])
    successor = _preempting_successor(store, incident, predecessor)
    hook = RecordingHook()
    executor = _executor(store, WorkflowStepOutcome.succeeded(), hook)

    result = execute_workflow(executor, predecessor.request_id)
    ended, _, steps = _only_call(hook)

    assert result.status is WorkflowStatus.SUPERSEDED, result
    assert ended.request_id == predecessor.request_id, ended.request_id
    assert ended.status is WorkflowStatus.SUPERSEDED, ended.status
    assert ended.preempted_by_workflow_id == successor.request_id, ended
    assert [step.operation for step in steps] == [FREEZE], steps
    assert (
        store.get_workflow(predecessor.request_id).status is WorkflowStatus.SUPERSEDED
    ), "the preemption itself still lands"


def test_hook_fires_for_a_terminalization_claimed_outside_execute():
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE])
    hook = RecordingHook()
    executor = _executor(store, WorkflowStepOutcome.succeeded(), hook)
    claimed = store.claim_workflow(
        workflow.request_id,
        executor.config.executor_id,
        workflow.fencing_token,
        lease_duration=timedelta(seconds=30),
    )

    executor.terminalize_claimed(
        claimed,
        incident,
        WorkflowStatus.BLOCKED,
        claimed.execution_epoch,
        reason="internal error",
        actor="dispatcher-internal-error",
        incident_state=IncidentState.ESCALATED,
        updates={"blocked_kind": BlockedKind.INTERNAL_ERROR},
    )
    ended, ended_incident, _ = _only_call(hook)

    assert ended.status is WorkflowStatus.BLOCKED, ended.status
    assert ended.blocked_kind is BlockedKind.INTERNAL_ERROR, ended.blocked_kind
    assert ended_incident is not None, "the incident is handed to the hook"
    assert ended_incident.state is IncidentState.ESCALATED, ended_incident.state


def test_a_failing_hook_neither_blocks_the_others_nor_the_terminalization():
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE])
    second = RecordingHook()
    executor = _executor(store, WorkflowStepOutcome.succeeded(), second)

    def broken(*_args: object) -> None:
        raise RuntimeError("lease release exploded")

    executor.on_terminal.insert(0, broken)

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert len(second.calls) == 1, second.calls
    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.SUCCEEDED, (
        "the terminal write must land despite the failing hook"
    )
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED, (
        "the incident must still be closed despite the failing hook"
    )


def test_hooks_do_not_fire_for_a_non_terminal_waiting_result():
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE])
    hook = RecordingHook()
    executor = _executor(
        store, WorkflowStepOutcome.waiting(operation_id="freeze-pending"), hook
    )

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING, result
    assert hook.calls == [], "a WAITING step is not a terminal transition"
