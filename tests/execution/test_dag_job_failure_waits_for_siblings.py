"""A job-level DAG failure waits for the sibling commands still on the nodes.

Control-plane review 2026-09-08, D-3. When a DAG step failed and the branch
escalator had nothing to say (no escalator, a job-level step, a profile that
would not compile), ``_execute_dag`` ended the workflow FAILED at the end of
the batch while a sibling branch's remote command was still LEASED on its node.
Nobody cancelled it and nobody waited for it: its result came back to a row no
workflow read, and the hardware escalation that followed could plan a second
action on the same node.

Now the batch failure first looks at the preemption boundary: cancellable
commands (PENDING, or a WAITING collection) are cancelled; anything in flight
that cannot be stopped defers the verdict -- the untouched steps are retired,
the failure is parked in ``pending_failure_*`` and the workflow stays RUNNING
until the in-flight steps settle, then ends FAILED through the compensation
funnel exactly as it would have.
"""

from __future__ import annotations

from gpu_fault.execution.models import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.models import IncidentState, WorkflowStatus, WorkflowStepStatus
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import active_workflow_executor, build_store
from tests.execution import test_branch_escalation as branches
from tests.execution._support import FakeAdapter

STOP = branches.STOP
RESET = branches.RESET
REBOOT = branches.REBOOT
BUNDLE = branches.BUNDLE
RESTART_JOB = branches.RESTART_JOB


def _in_flight(store, workflow, index: int, status: RemoteCommandStatus) -> str:
    incident = store.get_incident(workflow.incident_id)
    step = workflow.official_steps[index]
    command_id = f"cmd-{index}"
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id=command_id,
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
            lease_owner=(
                "executor-elsewhere" if status is RemoteCommandStatus.LEASED else None
            ),
        )
    )
    return command_id


def _waiting(index: int, remote_status: str) -> WorkflowStepOutcome:
    return WorkflowStepOutcome.waiting(
        operation_id=f"remote/cmd-{index}",
        details={"remote_status": remote_status, "remote_command_id": f"cmd-{index}"},
    )


def _executor(store, adapter):
    return active_workflow_executor(store, [adapter], branches.ALL_OPERATIONS)


def _execute(executor, workflow):
    return executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )


def test_a_leased_sibling_command_defers_the_failure_until_it_settles():
    store = build_store()
    incident, workflow = branches._job_dag(store)  # node-b RESET, node-c REBOOT
    _in_flight(store, workflow, 2, RemoteCommandStatus.LEASED)
    adapter = FakeAdapter(
        {
            RESET: WorkflowStepOutcome.failed("reset refused"),
            REBOOT: _waiting(2, "RUNNING"),
            RESTART_JOB: WorkflowStepOutcome.succeeded(),
        }
    )
    executor = _executor(store, adapter)  # no branch escalator: job-level failure

    first = _execute(executor, workflow)

    assert first.status is WorkflowStatus.RUNNING, first
    assert first.waiting_step_index == 2
    parked = store.get_workflow(workflow.request_id)
    assert parked.status is WorkflowStatus.RUNNING
    assert parked.pending_failure_step_index == 1
    assert parked.pending_failure_error == "reset refused"
    # The untouched join is retired; the failed step will not run again; the
    # in-flight reboot is left to finish.
    assert 3 in parked.superseded_step_indexes
    assert 1 in parked.superseded_step_indexes
    assert 2 not in parked.superseded_step_indexes
    assert store.get_remote_command("cmd-2").status is RemoteCommandStatus.LEASED
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.ACTION_PENDING
    )

    # The node's reboot finishes on a later tick.
    adapter.outcomes[REBOOT] = WorkflowStepOutcome.succeeded(
        operation_id="remote/cmd-2"
    )
    second = _execute(executor, workflow)

    assert second.status is WorkflowStatus.FAILED, second
    assert second.error == "reset refused"
    ended = store.get_workflow(workflow.request_id)
    assert ended.status is WorkflowStatus.FAILED
    assert ended.pending_failure_step_index is None
    assert 2 in ended.completed_step_indexes
    assert all("RESTART_WORKLOAD" not in call for call in adapter.calls), adapter.calls
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


def test_a_sibling_that_fails_while_the_verdict_is_parked_does_not_escalate():
    store = build_store()
    _, workflow = branches._job_dag(store)
    _in_flight(store, workflow, 2, RemoteCommandStatus.LEASED)
    adapter = FakeAdapter(
        {
            RESET: WorkflowStepOutcome.failed("reset refused"),
            REBOOT: _waiting(2, "RUNNING"),
            **branches._ok(RESTART_JOB, branches.REPLACE, *branches.VALIDATIONS),
        }
    )
    executor = _executor(store, adapter)
    executor.branch_escalator = branches._escalator()
    # With an escalator node-b's RESET would escalate in place; make the
    # failure job-level by having the escalator decline to compile.
    executor.branch_escalator.compile_steps = lambda *_args: []

    assert _execute(executor, workflow).status is WorkflowStatus.RUNNING
    adapter.outcomes[REBOOT] = WorkflowStepOutcome.failed("node did not come back")
    result = _execute(executor, workflow)

    assert result.status is WorkflowStatus.FAILED
    saved = store.get_workflow(workflow.request_id)
    assert saved.branch_escalation_counts == {}
    assert saved.exhausted_branch_ids == []
    assert all("REPLACE_NODE" not in call for call in adapter.calls), adapter.calls


def test_a_cancellable_sibling_command_is_cancelled_and_the_failure_lands_now():
    store = build_store()
    incident, workflow = branches._job_dag(store, node_c_operation=BUNDLE)
    _in_flight(store, workflow, 2, RemoteCommandStatus.WAITING)
    adapter = FakeAdapter(
        {
            RESET: WorkflowStepOutcome.failed("reset refused"),
            BUNDLE: _waiting(2, "WAITING"),
            RESTART_JOB: WorkflowStepOutcome.succeeded(),
        }
    )

    result = _execute(_executor(store, adapter), workflow)

    assert result.status is WorkflowStatus.FAILED
    command = store.get_remote_command("cmd-2")
    assert command.status is RemoteCommandStatus.FAILED
    assert command.status_source == "workflow-preempted"
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED
    saved = store.get_workflow(workflow.request_id)
    assert saved.pending_failure_step_index is None
    assert [item.status for item in saved.step_executions if item.step_index == 1] == [
        WorkflowStepStatus.FAILED
    ]
