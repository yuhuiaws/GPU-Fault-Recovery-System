"""What a stopping regional executor does with commands it will not run.

SIGTERM sets ``stop_requested``, but the long-poll claim already on the wire
keeps going and can still lease a command to the exiting process. Until now
that lease was parked: admission refused it, nothing was posted, and a sibling
could take the command only after the whole lease window (10 s under HA-004's
env window, 120 s in production). And a replica that was admitted before the
signal kept starting node actions and answering Agent challenges as authority
for as long as its adapter call ran, because the lease guard knew nothing about
the shutdown.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.adapters.node_action.lease_guard import (
    active_lease_guard,
    lease_hold_reason,
)
from gpu_fault.cluster_executor import ClusterExecutorError
from gpu_fault.cluster_executor.lease import SHUTDOWN_RELEASE_STATUS_SOURCE
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
    RecordingAdapter,
    build_executor,
    remote_command,
)

# A command mid-flight: the agent accepted the reset, the executor was polling.
IN_FLIGHT = {
    "node_action_command_id": "idem-command-a/node-a/agent-1",
    "node_action_accepted_nodes": ["node-a"],
    "node_action_state": "PENDING",
}


def test_a_command_claimed_after_stop_is_released_at_once() -> None:
    client = FakeExecutorClient(
        [remote_command("command-a", result_details=dict(IN_FLIGHT))]
    )
    adapter = RecordingAdapter()
    executor = build_executor(client, [adapter])
    executor.request_stop("SIGTERM received")

    executor.run_once()

    assert adapter.contexts == [], "a stopping executor runs nothing new"
    assert client.renewals == [], "and renews nothing"
    released = client.reported("command-a")
    assert released.status is RemoteCommandStatus.WAITING
    assert released.status_source == SHUTDOWN_RELEASE_STATUS_SOURCE
    assert released.details == IN_FLIGHT, (
        "complete_remote_command replaces result_details, so the release must "
        "echo the command's own record and change nothing but the lease"
    )
    assert released.lease_token == "lease-command-a"
    assert executor.results_withheld_total == 1


def test_a_release_is_not_attempted_under_a_lease_the_claim_did_not_prove() -> None:
    class ForeignLeaseClient(FakeExecutorClient):
        def claim(self, executor_id: str, **kwargs: Any) -> list[RemoteActionCommand]:
            return [
                command.model_copy(update={"lease_owner": "someone-else"})
                for command in super().claim(executor_id, **kwargs)
            ]

    client = ForeignLeaseClient([remote_command("command-a")])
    executor = build_executor(client, [RecordingAdapter()])
    executor.request_stop("SIGTERM received")

    executor.run_once()

    assert client.completed == [], "a lease this executor cannot vouch for is left"
    assert executor.lease_lost_total == 1


def test_a_release_that_cannot_be_posted_leaves_the_lease_to_lapse() -> None:
    client = FakeExecutorClient(
        [remote_command("command-a")],
        complete_errors={
            "command-a": ClusterExecutorError("control plane down", status_code=503)
        },
    )
    executor = build_executor(client, [RecordingAdapter()])
    executor.request_stop("SIGTERM received")

    executor.run_once()

    assert [command_id for command_id, _ in client.completed] == ["command-a"], (
        "one release attempt, no retry loop on the way out"
    )
    assert executor.results_withheld_total == 1


def test_a_stop_during_an_admitted_command_holds_new_actions_and_posts_waiting(
    monkeypatch: Any,
) -> None:
    client = FakeExecutorClient(
        [remote_command("command-a", result_details=dict(IN_FLIGHT))]
    )
    observed: dict[str, Any] = {}

    class StoppingAdapter(RecordingAdapter):
        def execute(self, context: Any) -> WorkflowStepOutcome:
            observed["before"] = lease_hold_reason()
            executor.request_stop("SIGTERM received")
            observed["after"] = lease_hold_reason()
            # What the node-action adapter does under a hold: nothing new
            # starts, the pointer to the accepted action is kept.
            return WorkflowStepOutcome.waiting(
                details={**IN_FLIGHT, "node_action_state": "LEASE_LOST"}
            )

    executor = build_executor(client, [StoppingAdapter()])

    executor.run_once()

    assert observed["before"] is None, "a live lease holds nothing back"
    assert observed["after"] is not None and "shutdown" in observed["after"], (
        "the guard must report the shutdown to the adapter on the same thread"
    )
    assert client.renewals == [("command-a", EXECUTOR, 120)], (
        "the admission renewal happened; the heartbeat then lets the lease lapse"
    )
    posted = client.reported("command-a")
    assert posted.status is RemoteCommandStatus.WAITING
    assert (
        posted.details["node_action_command_id"] == IN_FLIGHT["node_action_command_id"]
    )
    assert executor.results_withheld_total == 0, (
        "the lease is still ours, so the WAITING hand-back is posted, not withheld"
    )


def test_the_pre_adapter_lease_hold_keeps_the_commands_continuation_state() -> None:
    """A hold answered before the adapter ran is posted as the command's whole
    record, so it must carry the previous cycle's state forward like every
    other executor-manufactured WAITING (``_hold_details``)."""

    continuation = {
        "agent_baselines": {"node-a": {"boot_id": "boot-1"}},
        "spare_failover_pending": True,
    }
    command = remote_command("command-a", result_details=dict(continuation))
    executor = build_executor(FakeExecutorClient(), [RecordingAdapter()])
    token = active_lease_guard.set(lambda: "executor shutdown requested: SIGTERM")
    try:
        result = executor.dispatch.execute(command)
    finally:
        active_lease_guard.reset(token)

    assert result.status is RemoteCommandStatus.WAITING
    assert result.details["lease_guard_blocked"] is True
    assert result.details["adapter_started"] is False
    assert "shutdown" in result.details["reason"]
    for key, value in continuation.items():
        assert result.details[key] == value, key
