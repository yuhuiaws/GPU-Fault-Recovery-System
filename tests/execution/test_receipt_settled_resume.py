"""A NEEDS_OPERATOR record settled by remote receipts is resumed, not stranded.

Live 2026-09-28 (GF-REGIONAL-DESTR-018): the lifetime deadline landed while the
executor held the reset carrier's lease; the record was parked NEEDS_OPERATOR in
the same tick, and the executor's no-start receipt that arrived right after
resolved every unresolved action -- but nothing read it. The dispatcher's sweep
now re-opens such a record so the compensation restore it owes runs and the
record ends FAILED, the shape it would have had one tick later.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.node_action_uncertainty import has_unresolved_node_action
from gpu_fault.execution.receipt_settled_resume import (
    RESUME_MARKER,
    receipt_settled_resume_reasons,
    resume_receipt_settled_workflows,
)
from gpu_fault.models import (
    BlockedKind,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests.execution.test_regional_action_uncertainty import claim, remote_flow, report

NO_START = {"node_action_not_started": True, "cancelled_before_start": True}
OP = WorkflowOperation


def _park_at_lifetime(flow):
    """Claim the reset carrier, expire the lifetime, park the record NEEDS_OPERATOR."""

    command = claim(flow)
    flow.execute()
    flow.amend(lifetime_deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    result = flow.execute()
    assert result.status is WorkflowStatus.BLOCKED, "the leased reset parks the record"
    parked = flow.store.get_workflow(flow.workflow.request_id)
    assert parked.blocked_kind is BlockedKind.NEEDS_OPERATOR
    cancelled = flow.store.get_remote_command(command.command_id)
    assert cancelled.cancellation_requested_at is not None, (
        "the deadline requested the cancellation of the leased command"
    )
    return command


def test_the_no_start_receipt_resumes_the_parked_record_and_the_restore_runs() -> None:
    flow = remote_flow(OP.RESET_GPU, batched=True)
    command = _park_at_lifetime(flow)
    now = datetime.now(timezone.utc)
    assert resume_receipt_settled_workflows(flow.store, now=now) == [], (
        "without the receipt the record stays parked: the reset may still be running"
    )

    # The executor answers the cancellation it saw at admission: nothing ran.
    report(flow, command, dict(NO_START))
    resumed_ids = resume_receipt_settled_workflows(flow.store, now=now)

    assert resumed_ids == [flow.workflow.request_id], "the receipt settles the record"
    resumed = flow.store.get_workflow(flow.workflow.request_id)
    assert resumed.status is WorkflowStatus.PENDING and resumed.blocked_kind is None
    assert resumed.pending_failure_step_index == 1, (
        "the reset's deadline failure is parked"
    )
    assert "lifetime exceeded" in str(resumed.pending_failure_error)
    assert not has_unresolved_node_action(resumed), "the no-start is a resolved outcome"
    event = resumed.events[-1]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.details["resumption"] == RESUME_MARKER
    assert event.actor == "dispatcher"

    # The ordinary executor path now runs the compensation the record owes.
    flow.execute()
    restores = [
        item
        for item in flow.store.list_remote_commands()
        if item.step.operation is OP.RESTORE_GPU_SERVICES
    ]
    assert len(restores) == 1, "the owed RESTORE_GPU_SERVICES is issued once"
    restore = claim(flow)
    assert restore.command_id == restores[0].command_id, "the restore is the next claim"
    report(flow, restore, {}, status=RemoteCommandStatus.SUCCEEDED)
    final = flow.execute()

    assert final.status is WorkflowStatus.FAILED, "the deadline verdict lands FAILED"
    saved = flow.store.get_workflow(flow.workflow.request_id)
    assert saved.status is WorkflowStatus.FAILED and saved.blocked_kind is None
    assert saved.pending_failure_step_index is None
    assert OP.RESTORE_GPU_SERVICES in saved.completed_operations


def test_a_record_the_incident_moved_on_from_is_left_to_the_settled_close() -> None:
    flow = remote_flow(OP.RESET_GPU, batched=True)
    command = _park_at_lifetime(flow)
    report(flow, command, dict(NO_START))
    incident = flow.store.get_incident(flow.incident.incident_id)
    flow.store.save_incident(
        incident.model_copy(update={"workflow_request_id": "workflow-successor"})
    )

    now = datetime.now(timezone.utc)
    reasons, _ = receipt_settled_resume_reasons(
        flow.store, flow.store.get_workflow(flow.workflow.request_id), now=now
    )
    assert reasons == ["the incident has moved on to another workflow"], reasons
    assert resume_receipt_settled_workflows(flow.store, now=now) == []
    assert (
        flow.store.get_workflow(flow.workflow.request_id).status
        is WorkflowStatus.BLOCKED
    )


def test_a_second_sweep_after_a_resume_changes_nothing() -> None:
    flow = remote_flow(OP.RESET_GPU, batched=True)
    command = _park_at_lifetime(flow)
    report(flow, command, dict(NO_START))
    now = datetime.now(timezone.utc)
    assert resume_receipt_settled_workflows(flow.store, now=now) == [
        flow.workflow.request_id
    ]

    assert resume_receipt_settled_workflows(flow.store, now=now) == [], (
        "a PENDING record is not a NEEDS_OPERATOR record"
    )


def test_a_batched_step_the_carrier_never_reached_is_a_known_no_start() -> None:
    """The carrier's LEASED-time snapshot marked the later RESTORE step unknown;
    once the carrier ends without an entry for it, the step never started."""

    flow = remote_flow(OP.RESET_GPU, batched=True)
    command = _park_at_lifetime(flow)
    parked = flow.store.get_workflow(flow.workflow.request_id)
    restore = [item for item in parked.step_executions if item.step_index == 2]
    assert restore and restore[-1].details.get("outcome_unknown") is True, (
        "while the carrier is leased the unreached restore is rightly unknown"
    )

    report(flow, command, dict(NO_START))
    reasons, refreshed = receipt_settled_resume_reasons(
        flow.store, parked, now=datetime.now(timezone.utc)
    )

    assert reasons == [], reasons
    latest = [item for item in refreshed.step_executions if item.step_index == 2][-1]
    assert latest.details.get("node_action_not_started") is True, latest.details
    assert latest.details.get("outcome_unknown") is not True, latest.details
    # A step the carrier never ran has nothing in flight: its row must end,
    # or the FAILED record keeps an "unknown provider action" that refuses
    # the incident close (live 2026-09-28, DESTR-018 cleanup).
    unreached = [
        item
        for item in refreshed.step_executions
        if item.details.get("batched_step_not_reached") is True
    ]
    assert unreached, "the refreshed record names no never-reached step"
    assert all(item.status is WorkflowStepStatus.FAILED for item in unreached), [
        (item.step_index, item.status.value) for item in unreached
    ]
    assert not any(
        item.status is WorkflowStepStatus.WAITING for item in refreshed.step_executions
    ), "a never-reached step stayed WAITING on the refreshed record"


def test_a_reset_behind_a_waiting_verify_does_not_stay_waiting_after_the_receipt() -> (
    None
):
    """Live 2026-09-28 (DESTR-018 rerun): the carrier held QUIESCE+VERIFY+RESET;
    VERIFY was still waiting on GPU clients when the lifetime ran out. The
    post-cancellation receipt named VERIFY only, so RESET -- dispatched WAITING
    with the compound and never reached -- kept its WAITING row on the FAILED
    record, and the incident close refused it as an unknown provider action."""

    flow = remote_flow(
        OP.RESET_GPU,
        batched=True,
        operations=[
            OP.QUIESCE_GPU_SERVICES,
            OP.VERIFY_NO_GPU_CLIENTS,
            OP.RESET_GPU,
            OP.RESTORE_GPU_SERVICES,
        ],
    )
    command = claim(flow)
    assert 2 in command.covered_step_indexes, (
        f"the carrier must cover RESET: {command.covered_step_indexes}"
    )
    report(flow, command, {"reason": "GPU device clients are still active"})
    flow.execute()
    # The executor re-claims the carrier for the next VERIFY attempt; the
    # deadline lands while it is LEASED (a WAITING carrier would be cancelled
    # straight to FAILED and the record would not park).
    reclaimed = claim(flow)
    assert reclaimed.command_id == command.command_id, "a different carrier was claimed"
    command = reclaimed
    flow.amend(lifetime_deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    result = flow.execute()
    assert result.status is WorkflowStatus.BLOCKED, (
        "the leased carrier parks the record"
    )
    parked = flow.store.get_workflow(flow.workflow.request_id)
    reset_rows = [item for item in parked.step_executions if item.step_index == 2]
    assert reset_rows and reset_rows[-1].status is WorkflowStepStatus.WAITING, (
        "parking under the leased carrier records the covered RESET as WAITING/unknown"
    )
    # The executor's next VERIFY report lands after the cancellation: the store
    # ends the carrier FAILED/completed-after-cancellation with no RESET entry.
    report(flow, command, {"reason": "GPU device clients are still active"})
    ended = flow.store.get_remote_command(command.command_id)
    assert ended.status is RemoteCommandStatus.FAILED, ended.status

    now = datetime.now(timezone.utc)
    resumed = resume_receipt_settled_workflows(flow.store, now=now)
    assert resumed == [flow.workflow.request_id], resumed
    flow.execute()
    restore = claim(flow)
    assert restore.step.operation is OP.RESTORE_GPU_SERVICES, restore.step.operation
    report(flow, restore, {}, status=RemoteCommandStatus.SUCCEEDED)
    final = flow.execute()
    assert final.status is WorkflowStatus.FAILED, final.status
    record = flow.store.get_workflow(flow.workflow.request_id)
    waiting = [
        (item.step_index, item.operation.value)
        for item in record.step_executions
        if item.status is WorkflowStepStatus.WAITING
    ]
    assert waiting == [], f"the ended record still names in-flight steps: {waiting}"
    reset = [item for item in record.step_executions if item.step_index == 2][-1]
    assert reset.status is WorkflowStepStatus.FAILED, reset.status
    assert reset.details.get("batched_step_not_reached") is True, reset.details
    assert not has_unresolved_node_action(record), (
        "the never-run RESET reads unresolved"
    )
