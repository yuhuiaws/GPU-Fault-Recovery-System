"""A step boundary cancels its remote commands all or nothing, judged live.

Control-plane review 2026-09-08, D-8 (F-C1 remainder). ``_supersede_if_safe``
judged each WAITING record by the ``remote_status`` it recorded on the *previous*
poll and cancelled one command at a time, returning on the first refusal. A
command recorded PENDING that the data plane had leased in between made the
store refuse its cancel -- after a sibling's command had already been
cancelled. The next tick read that sibling's FAILED (our own cancellation) as
a step failure and escalated its branch a rung, spending cluster budget on a
node nothing was wrong with.

Now every command is re-read from the store first and the boundary stays
closed unless all of them are still cancellable; and a step that failed only
because its command was cancelled by the workflow (``status_source``
``workflow-preempted`` / ``workflow-timeout``) never escalates.
"""

from __future__ import annotations

from gpu_fault.execution.models import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import build_store, copy_model, workflow_step_execution
from tests.execution import test_branch_escalation as branches
from tests.execution._support import (
    FakeAdapter,
    _preempting_successor,
    active_workflow_executor,
    workflow_state,
)

QUARANTINE = WorkflowOperation.QUARANTINE
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
RESET = WorkflowOperation.RESET_GPU


def _remote_record(index: int, operation: WorkflowOperation, remote_status: str):
    return workflow_step_execution(
        index,
        operation,
        WorkflowStepStatus.WAITING,
        adapter_operation_id=f"remote/cmd-{index}",
        details={"remote_status": remote_status, "remote_command_id": f"cmd-{index}"},
    )


def _command(store, workflow, index: int, status: RemoteCommandStatus):
    incident = store.get_incident(workflow.incident_id)
    step = workflow.official_steps[index]
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id=f"cmd-{index}",
            cluster_id=incident.cluster_id,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=index,
            fencing_token=workflow.fencing_token,
            idempotency_key=f"{workflow.request_id}/{index}/{step.operation.value}",
            step=step,
            workflow=workflow,
            incident=incident,
            status=status,
            lease_owner="executor-elsewhere"
            if status is RemoteCommandStatus.LEASED
            else None,
        )
    )


def test_a_command_leased_since_the_last_poll_keeps_every_sibling_uncancelled():
    store = build_store()
    incident, workflow = workflow_state(store, [QUARANTINE, BUNDLE, RESET])
    dag = copy_model(
        workflow,
        dag_enabled=True,
        status=WorkflowStatus.RUNNING,
        official_steps=[
            copy_model(workflow.official_steps[0], node_ids=["node-a", "node-b"]),
            copy_model(
                workflow.official_steps[1],
                node_ids=["node-a"],
                depends_on_step_indexes=[0],
                branch_id="branch:node-a",
            ),
            copy_model(
                workflow.official_steps[2],
                node_ids=["node-b"],
                depends_on_step_indexes=[0],
                branch_id="branch:node-b",
            ),
        ],
        completed_step_indexes=[0],
        completed_operations=[QUARANTINE],
        step_executions=[
            workflow_step_execution(0, QUARANTINE),
            # A collection the node is running: cancellable while WAITING.
            _remote_record(1, BUNDLE, "WAITING"),
            # Recorded PENDING on the last poll ...
            _remote_record(2, RESET, "PENDING"),
        ],
    )
    store.save_workflow(dag)
    _command(store, dag, 1, RemoteCommandStatus.WAITING)
    # ... but the data plane leased it since.
    _command(store, dag, 2, RemoteCommandStatus.LEASED)
    _preempting_successor(store, incident, dag)
    adapter = FakeAdapter(
        {
            BUNDLE: WorkflowStepOutcome.waiting(
                operation_id="remote/cmd-1",
                details={"remote_status": "WAITING", "remote_command_id": "cmd-1"},
            ),
            RESET: WorkflowStepOutcome.waiting(
                operation_id="remote/cmd-2",
                details={"remote_status": "LEASED", "remote_command_id": "cmd-2"},
            ),
        }
    )
    executor = active_workflow_executor(store, [adapter], {QUARANTINE, BUNDLE, RESET})

    result = executor.execute(
        dag.request_id,
        WorkflowExecutionRequest(expected_fencing_token=dag.fencing_token),
    )

    assert result.status is WorkflowStatus.RUNNING, result
    assert store.get_remote_command("cmd-1").status is RemoteCommandStatus.WAITING, (
        "the sibling's collection must not be cancelled when the boundary cannot close"
    )
    assert store.get_remote_command("cmd-2").status is RemoteCommandStatus.LEASED
    saved = store.get_workflow(dag.request_id)
    assert saved.status is WorkflowStatus.RUNNING
    assert not [
        event
        for event in saved.events
        if event.code == WorkflowEventCode.BRANCH_ESCALATED.value
    ]


def test_a_step_failed_by_the_workflows_own_cancellation_does_not_escalate():
    store = build_store()
    incident, workflow = branches._job_dag(store)
    outcomes = {
        RESET: WorkflowStepOutcome.failed(
            "remote command cancelled by stronger workflow workflow-successor",
            details={"remote_status_source": "workflow-preempted"},
        ),
        **branches._ok(branches.REBOOT, branches.RESTART_JOB, *branches.VALIDATIONS),
    }

    result, adapter, saved = branches._run(
        store, workflow, outcomes, escalator=branches._escalator()
    )

    assert result.status is WorkflowStatus.FAILED
    assert saved.branch_escalation_counts == {}, "our own cancellation is not a rung"
    assert not [
        event
        for event in saved.events
        if event.code == WorkflowEventCode.BRANCH_ESCALATED.value
    ]
    assert all("RESTART_WORKLOAD" not in call for call in adapter.calls)
