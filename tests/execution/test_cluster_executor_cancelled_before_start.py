"""A cancellation observed at lease admission is answered, not swallowed.

Live 2026-09-28 (DESTR-018): the workflow lifetime expired at the instant the
executor re-claimed the compound reset carrier. The renewal carried the
cancellation, admission was refused and the executor withheld any result, so
the command stayed LEASED with a cancellation nobody answered; the engine read
the never-started RESET_GPU as an unknown outcome, withheld the compensation
restore and parked the workflow NEEDS_OPERATOR without the support hand-off.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.cluster_executor.lease import CANCELLED_BEFORE_START_STATUS_SOURCE
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from tests.execution.test_cluster_executor_lease_and_report import (
    FakeExecutorClient,
    RecordingAdapter,
    build_executor,
    remote_command,
)

RECORD = {
    "batched_results": {"2": {"status": "SUCCEEDED"}},
    "node_action_state": "PENDING",
}


def _cancelled(command: RemoteActionCommand) -> RemoteActionCommand:
    return command.model_copy(
        update={
            "cancellation_requested_at": datetime.now(timezone.utc),
            "cancellation_reason": "workflow lifetime exceeded",
        }
    )


def test_a_claim_that_already_carries_the_cancellation_reports_a_no_start() -> None:
    client = FakeExecutorClient(
        [_cancelled(remote_command("command-a", result_details=dict(RECORD)))]
    )
    adapter = RecordingAdapter()
    executor = build_executor(client, [adapter])

    executor.run_once()

    assert adapter.contexts == [], "a cancelled command runs nothing"
    assert client.renewals == [], "and is not renewed"
    reported = client.reported("command-a")
    assert reported.status is RemoteCommandStatus.WAITING
    assert reported.status_source == CANCELLED_BEFORE_START_STATUS_SOURCE
    assert reported.details == {
        **RECORD,
        "node_action_not_started": True,
        "cancelled_before_start": True,
        "cancellation_reason": "workflow lifetime exceeded",
    }, "the command's own record is echoed with the no-start markers"
    assert reported.lease_token == "lease-command-a"
    assert executor.results_withheld_total == 1, (
        "nothing ran, so nothing was reported as run"
    )


class CancellingOnRenewalClient(FakeExecutorClient):
    """The control plane cancelled between the claim and the admission renewal."""

    def renew(self, command, executor_id, lease_seconds):
        renewed = super().renew(command, executor_id, lease_seconds)
        return _cancelled(renewed)


def test_a_cancellation_seen_on_the_admission_renewal_reports_a_no_start() -> None:
    client = CancellingOnRenewalClient(
        [remote_command("command-a", result_details=dict(RECORD))]
    )
    adapter = RecordingAdapter()
    executor = build_executor(client, [adapter])

    executor.run_once()

    assert adapter.contexts == [], "the adapter must not start after a cancellation"
    assert [identity for identity, _, _ in client.renewals] == ["command-a"]
    reported = client.reported("command-a")
    assert reported.status_source == CANCELLED_BEFORE_START_STATUS_SOURCE
    assert reported.details["node_action_not_started"] is True, reported.details


def test_a_no_start_report_that_cannot_be_posted_is_only_logged() -> None:
    client = FakeExecutorClient(
        [_cancelled(remote_command("command-a"))],
        complete_errors={"command-a": ConnectionError("control plane away")},
    )
    executor = build_executor(client, [RecordingAdapter()])

    executor.run_once()

    assert [identity for identity, _ in client.completed] == ["command-a"], (
        "one post is attempted; the lease lapses on the control plane's schedule"
    )
    assert executor.results_withheld_total == 1, "the failed post is not a run"
    assert datetime.now(timezone.utc) - timedelta(minutes=1) < datetime.now(
        timezone.utc
    ), "no retry loop delayed the executor"
