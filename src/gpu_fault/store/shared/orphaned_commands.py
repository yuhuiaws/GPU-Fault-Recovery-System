"""Cancellation records and audit payload shared by the Store backends."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Mapping

from gpu_fault.models import (
    WorkflowEvent,
    WorkflowEventKind,
    WorkflowRequest,
    WorkflowStatus,
    build_operator_event,
)
from gpu_fault.remote_command_models import RemoteCommandStatus

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand

AUDIT_ACTION = "cancelled orphaned remote commands"
TERMINAL_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.FAILED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    }
)


@dataclass(frozen=True)
class OrphanedCommandCancellation:
    cancelled: int = 0
    cancellation_requested: int = 0
    command_ids: tuple[str, ...] = ()

    @property
    def counters(self) -> dict[str, int]:
        return {
            "cancelled": self.cancelled,
            "cancellation_requested": self.cancellation_requested,
        }


def cancellation_reason(workflow: WorkflowRequest, actor: str) -> str:
    return (
        f"{actor} reconciliation: cancelled remote commands orphaned by "
        f"{workflow.status.value} workflow {workflow.request_id}"
    )


def cancellation_event(
    workflow: WorkflowRequest,
    cancelled: Mapping[str, int],
    command_ids: Iterable[str],
    *,
    now: datetime,
    actor: str,
) -> WorkflowEvent:
    return build_operator_event(
        workflow,
        WorkflowEventKind.OPERATOR_RECONCILED,
        actor=actor,
        reference=None,
        previous_status=workflow.status,
        at=now,
        details={
            "action": AUDIT_ACTION,
            "cancelled_remote_commands": dict(cancelled),
            "command_ids": list(command_ids),
        },
    )


def orphaned_cancellation_records(
    workflow: WorkflowRequest,
    commands: Iterable[RemoteActionCommand | None],
    *,
    now: datetime,
    actor: str,
) -> tuple[list[RemoteActionCommand], OrphanedCommandCancellation]:
    """Plan changes to freshly locked commands; the caller commits the audit."""

    reason = cancellation_reason(workflow, actor)
    records = []
    cancelled = requested = 0
    for command in commands:
        if command is None or command.workflow_request_id != workflow.request_id:
            continue
        if command.status in {RemoteCommandStatus.PENDING, RemoteCommandStatus.WAITING}:
            updated = command.model_copy(
                update={
                    "status": RemoteCommandStatus.FAILED,
                    "error": reason,
                    "status_source": "workflow-timeout",
                    "lease_owner": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            cancelled += 1
        elif (
            command.status is RemoteCommandStatus.LEASED
            and command.cancellation_requested_at is None
        ):
            updated = command.model_copy(
                update={
                    "cancellation_requested_at": now,
                    "cancellation_reason": reason,
                    "updated_at": now,
                }
            )
            requested += 1
        else:
            continue
        records.append(updated)
    records.sort(key=lambda command: command.command_id)
    return records, OrphanedCommandCancellation(
        cancelled, requested, tuple(command.command_id for command in records)
    )
