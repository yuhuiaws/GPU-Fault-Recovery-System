"""Data-plane half of the compound remote command (性能 C).

The executor runs the covered steps in order through its one local adapter,
hands each step the context it would have had as a command of its own, posts
progress after each, stops at the first non-success, and resumes from the
progress a previous claim left behind.

Every command here is driven the way production drives it: ``run_once``
claims it from the (fake) regional client, executes it under a lease
heartbeat and posts the terminal result back through ``complete``. What the
tests assert on is therefore what crossed that boundary -- the completed
result, the progress posts, the renewals -- never a return value of the
executor's internals.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from threading import Event
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import (
    BatchedStep,
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from tests._builders import build_store
from tests.regional._batching_support import (
    NODE_OWNER,
    NODE_SIDE,
    chain_state,
    quiesce_details,
    reset_chain,
    step_context,
)

CLUSTER = "cluster-a"
EXECUTOR = "executor-a"
WAIT = 5.0


@pytest.fixture(autouse=True)
def claim_state_outside_shared_tmp(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "claim.json")
    )


class FakeClient:
    """In-memory regional client: hands ``command`` out on the first claim and
    records everything the executor posts back under that lease."""

    cluster_id = CLUSTER

    def __init__(
        self,
        command: RemoteActionCommand | None = None,
        *,
        progress_error: BaseException | None = None,
        cancel_on_renew: str | None = None,
        before_renew: Callable[[], None] | None = None,
    ) -> None:
        self.pending = [command] if command is not None else []
        self.progress_posts: list[dict[str, Any]] = []
        self.progress_error = progress_error
        self.completed: list[RemoteCommandResult] = []
        self.renewals = 0
        # When set, the heartbeat answers with a cancellation request carrying
        # this reason -- the control plane's way of telling the executor that
        # nothing new should start. ``before_renew`` runs first, so a test can
        # hold the answer until the moment it wants the request to land.
        self.cancel_on_renew = cancel_on_renew
        self.before_renew = before_renew
        self.renewed = Event()

    def claim(self, *_args, **_kwargs):
        commands, self.pending = self.pending, []
        return commands

    def complete(self, command, result):
        self.completed.append(result)
        return command

    def renew(self, command, *_args, **_kwargs):
        if self.before_renew is not None:
            self.before_renew()
        self.renewals += 1
        if self.cancel_on_renew is not None:
            command = command.model_copy(
                update={
                    "cancellation_requested_at": datetime.now(timezone.utc),
                    "cancellation_reason": self.cancel_on_renew,
                }
            )
        self.renewed.set()
        return command

    def progress(self, command, executor_id, batched_results):
        self.progress_posts.append(
            {
                "command_id": command.command_id,
                "executor_id": executor_id,
                **batched_results,
            }
        )
        if self.progress_error is not None:
            raise self.progress_error
        return command

    def reported(self) -> RemoteCommandResult:
        """The one terminal result the executor posted under the lease."""
        (result,) = self.completed
        return result


class FakeNodeAdapter:
    """Answers each operation from a table and records every context."""

    def __init__(
        self,
        outcomes: dict[WorkflowOperation, Any],
        *,
        before_answer: Callable[[Any], None] | None = None,
    ) -> None:
        self.owner = NODE_OWNER
        self.outcomes = outcomes
        self.contexts: list[Any] = []
        self.before_answer = before_answer

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner

    def execute(self, context) -> WorkflowStepOutcome:
        self.contexts.append(context)
        if self.before_answer is not None:
            self.before_answer(context)
        outcome = self.outcomes[context.step.operation]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class OneRenewalStopEvent(Event):
    """Stop event for the lease heartbeat that runs exactly one renewal.

    ``_execute_and_report`` hands the renewer a fresh ``Event`` and the
    renewer waits on it for the heartbeat interval (>= 3.3 s). Returning
    False once and True afterwards renews immediately and then ends the
    loop, with no real sleeping; same stand-in as
    ``test_cluster_executor_lease_and_report``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.waits = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.waits += 1
        return self.waits > 1


def all_succeed() -> dict[WorkflowOperation, Any]:
    return {
        WorkflowOperation.QUIESCE_GPU_SERVICES: WorkflowStepOutcome.succeeded(
            details=quiesce_details()
        ),
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS: WorkflowStepOutcome.succeeded(
            details={"gpu_client_quiesce_attempt": 1}
        ),
        WorkflowOperation.RESET_GPU: WorkflowStepOutcome.succeeded(
            details={"node_results": {"node-a": {"reset_gpu_uuids": ["GPU-a"]}}}
        ),
        WorkflowOperation.RESTORE_GPU_SERVICES: WorkflowStepOutcome.succeeded(
            details={"node_results": {"node-a": {"status": "SUCCEEDED"}}}
        ),
    }


def compound_command(
    result_details: dict[str, Any] | None = None,
) -> RemoteActionCommand:
    incident, workflow = chain_state(build_store(), reset_chain())
    contexts = [step_context(workflow, incident, index) for index in range(1, 5)]
    head = contexts[0]
    return RemoteActionCommand(
        command_id="remote-" + "c" * 24,
        cluster_id=CLUSTER,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=head.step_index,
        fencing_token=3,
        idempotency_key=head.idempotency_key,
        step=head.step,
        workflow=workflow,
        incident=incident,
        lease_token="lease-c",
        status=RemoteCommandStatus.LEASED,
        lease_owner=EXECUTOR,
        batched_steps=[
            BatchedStep(
                step_index=item.step_index,
                step=item.step,
                idempotency_key=item.idempotency_key,
            )
            for item in contexts[1:]
        ],
        result_details=result_details or {},
    )


def executor(client: FakeClient, adapter: FakeNodeAdapter) -> ClusterActionExecutor:
    return ClusterActionExecutor(
        client, [adapter], executor_id=EXECUTOR, allowed_namespaces={"training"}
    )


def run_claimed_command(
    client: FakeClient, adapter: FakeNodeAdapter
) -> tuple[ClusterActionExecutor, RemoteCommandResult]:
    """One ``run_once``: claim the fake's command, execute it, post its result."""
    run = executor(client, adapter)
    assert run.run_once() == 1, "the executor must claim exactly the one command"
    return run, client.reported()


def test_the_steps_run_in_order_with_their_own_contexts_and_progress_after_each() -> (
    None
):
    command = compound_command()
    client = FakeClient(command)
    adapter = FakeNodeAdapter(all_succeed())

    run, result = run_claimed_command(client, adapter)

    assert result.status is RemoteCommandStatus.SUCCEEDED
    assert [context.step.operation for context in adapter.contexts] == list(NODE_SIDE)
    assert [context.step_index for context in adapter.contexts] == [1, 2, 3, 4]
    assert [context.idempotency_key for context in adapter.contexts] == [
        "workflow-a/1/QUIESCE_GPU_SERVICES",
        "workflow-a/2/VERIFY_NO_GPU_CLIENTS",
        "workflow-a/3/RESET_GPU",
        "workflow-a/4/RESTORE_GPU_SERVICES",
    ]
    reset = adapter.contexts[2]
    quiesce_record = next(
        item
        for item in reversed(reset.workflow.step_executions)
        if item.step_index < reset.step_index
        and item.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
        and item.status is WorkflowStepStatus.SUCCEEDED
    )
    assert quiesce_record.details == quiesce_details(), (
        "the RESET step's workflow copy carries the QUIESCE evidence exactly as "
        "_maintenance_generations reads it today"
    )
    assert quiesce_record.adapter_operation_id == f"remote/{command.command_id}"
    assert reset.workflow.completed_step_indexes == [1, 2]
    assert adapter.contexts[0].workflow.step_executions == [], (
        "the head sees the workflow as the control plane minted it"
    )
    assert [
        sorted(k for k in post if k.isdigit()) for post in client.progress_posts
    ] == [["1"], ["2"], ["3"], ["4"]]
    assert all(post["executor_id"] == EXECUTOR for post in client.progress_posts), (
        "every progress post must carry the executor that holds the lease"
    )
    results = result.details[BATCHED_RESULTS_KEY]
    assert [results[str(index)]["status"] for index in (1, 2, 3, 4)] == [
        "SUCCEEDED"
    ] * 4
    assert results["1"]["details"] == quiesce_details()
    assert result.details["batched_step_indexes"] == [1, 2, 3, 4]
    snapshot = run.metrics_snapshot()
    assert {"batched_commands_total", "batched_steps_total"} <= set(snapshot)


def test_a_waiting_step_stops_the_run_and_a_reclaim_resumes_at_it() -> None:
    outcomes = all_succeed()
    outcomes[WorkflowOperation.VERIFY_NO_GPU_CLIENTS] = WorkflowStepOutcome.waiting(
        details={"gpu_client_quiesce_attempt": 1, "waiting_nodes": ["node-a"]}
    )
    client = FakeClient(compound_command())
    adapter = FakeNodeAdapter(outcomes)

    _, first = run_claimed_command(client, adapter)

    assert first.status is RemoteCommandStatus.WAITING
    assert first.details["batched_step_index"] == 2
    assert first.details["waiting_nodes"] == ["node-a"]
    assert [context.step_index for context in adapter.contexts] == [1, 2]
    results = first.details[BATCHED_RESULTS_KEY]
    assert results["1"]["status"] == "SUCCEEDED" and results["2"]["status"] == "WAITING"

    outcomes[WorkflowOperation.VERIFY_NO_GPU_CLIENTS] = WorkflowStepOutcome.succeeded(
        details={"gpu_client_quiesce_attempt": 2}
    )
    # The control plane re-issues the command with the posted result as its
    # ``result_details``; a fresh claim picks it up there.
    resumed_client = FakeClient(compound_command(result_details=first.details))
    resumed_adapter = FakeNodeAdapter(outcomes)

    _, second = run_claimed_command(resumed_client, resumed_adapter)

    assert second.status is RemoteCommandStatus.SUCCEEDED
    assert [context.step_index for context in resumed_adapter.contexts] == [2, 3, 4], (
        "QUIESCE is not run again"
    )
    verify = resumed_adapter.contexts[0]
    prior = [item for item in verify.workflow.step_executions if item.step_index == 2]
    assert [item.status for item in prior] == [WorkflowStepStatus.WAITING]
    assert prior[0].details["gpu_client_quiesce_attempt"] == 1, (
        "the step sees its own last WAITING record, so its attempt counter advances"
    )
    assert second.details[BATCHED_RESULTS_KEY]["2"]["details"] == {
        "gpu_client_quiesce_attempt": 2
    }


def test_a_failed_step_fails_the_command_and_the_rest_never_run() -> None:
    outcomes = all_succeed()
    outcomes[WorkflowOperation.RESET_GPU] = WorkflowStepOutcome.failed(
        "reset refused", details={"node_results": {"node-a": {"status": "FAILED"}}}
    )
    client = FakeClient(compound_command())
    adapter = FakeNodeAdapter(outcomes)

    _, result = run_claimed_command(client, adapter)

    assert result.status is RemoteCommandStatus.FAILED
    assert result.error == "reset refused"
    assert result.details["batched_step_index"] == 3
    assert result.details["batched_operation"] == "RESET_GPU"
    assert [context.step_index for context in adapter.contexts] == [1, 2, 3]
    results = result.details[BATCHED_RESULTS_KEY]
    assert sorted(results) == ["1", "2", "3"], "RESTORE was never run nor recorded"
    assert results["3"] == {
        "status": "FAILED",
        "status_source": None,
        "details": {"node_results": {"node-a": {"status": "FAILED"}}},
        "error": "reset refused",
    }


def test_a_cancellation_between_steps_stops_before_the_next_one(monkeypatch) -> None:
    """A cancellation arrives the way it does in production: the lease
    heartbeat's ``renew`` answer carries the control plane's request, and the
    executor reads it through the node-action lease guard before it starts
    the next covered step."""

    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", OneRenewalStopEvent)
    # The heartbeat runs on its own thread the moment the command starts, so
    # the request is held back until QUIESCE is under way and QUIESCE does not
    # finish until the request has landed: it must be seen between 1 and 2,
    # not before 1.
    quiesce_started = Event()
    client = FakeClient(
        compound_command(),
        cancel_on_renew="cancelled by a stronger workflow",
        before_renew=lambda: quiesce_started.wait(WAIT),
    )

    def let_the_heartbeat_land_during_quiesce(context) -> None:
        if context.step_index == 1:
            quiesce_started.set()
            assert client.renewed.wait(WAIT), "the lease heartbeat never renewed"

    adapter = FakeNodeAdapter(
        all_succeed(), before_answer=let_the_heartbeat_land_during_quiesce
    )

    run, result = run_claimed_command(client, adapter)

    assert client.renewals == 1
    assert run.cancellations_observed_total == 1
    assert result.status is RemoteCommandStatus.WAITING
    assert result.status_source == "executor-stopped-between-batched-steps"
    assert result.details["batched_stopped_before_step_index"] == 2
    assert result.details["reason"] == "cancelled by a stronger workflow"
    assert result.details["node_action_not_started"] is True
    assert [context.step_index for context in adapter.contexts] == [1]
    assert sorted(result.details[BATCHED_RESULTS_KEY]) == ["1"]


def test_an_exception_inside_a_step_is_classified_for_that_step() -> None:
    outcomes = all_succeed()
    outcomes[WorkflowOperation.VERIFY_NO_GPU_CLIENTS] = AttributeError("no such helper")
    client = FakeClient(compound_command())
    adapter = FakeNodeAdapter(outcomes)

    run, result = run_claimed_command(client, adapter)

    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source == "executor-internal-error"
    assert result.details["batched_step_index"] == 2
    assert result.details[BATCHED_RESULTS_KEY]["2"]["status_source"] == (
        "executor-internal-error"
    )
    assert run.unexpected_failures == 1
    assert [context.step_index for context in adapter.contexts] == [1, 2]


def test_a_rejected_progress_post_does_not_stop_the_run() -> None:
    client = FakeClient(
        compound_command(),
        progress_error=ClusterExecutorError("not found", status_code=404),
    )
    adapter = FakeNodeAdapter(all_succeed())

    run, result = run_claimed_command(client, adapter)

    assert result.status is RemoteCommandStatus.SUCCEEDED
    assert len(client.progress_posts) == 4
    assert run.batched_progress_failures_total == 4
    assert sorted(result.details[BATCHED_RESULTS_KEY]) == ["1", "2", "3", "4"], (
        "the terminal result carries every step's verdict regardless"
    )


def test_run_once_reports_the_compound_result_under_one_lease() -> None:
    client = FakeClient(compound_command())
    adapter = FakeNodeAdapter(all_succeed())

    assert executor(client, adapter).run_once() == 1

    (reported,) = client.completed
    assert reported.status is RemoteCommandStatus.SUCCEEDED
    assert sorted(reported.details[BATCHED_RESULTS_KEY]) == ["1", "2", "3", "4"]
    assert len(client.progress_posts) == 4
    assert client.claim() == [], "the command was handed out exactly once"
