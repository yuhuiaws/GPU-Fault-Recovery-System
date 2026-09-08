"""Cancel remote commands a terminal workflow left open.

A remote command is the GPU-side executor's unit of work for one workflow
step. When the step's workflow reaches a terminal state the command should be
settled too, and the current executor does that: a step that hits its waiting
cap cancels its remote commands before the step fails. An older release did
not, so a step failed by the per-step cap could leave its command ``WAITING``
forever -- observed live: ``CHECK_MECHANICALS`` on a workflow already
``FAILED``, thirteen hours old.

Nothing settled such a command afterwards: the executor never revisits a
terminal workflow, and the retired-generation sweep cancels only for
generations an incident re-planned away from. Yet the release upgrade refuses
to start while any command is ``PENDING``, ``LEASED`` or ``WAITING``, so the
orphan blocked exactly the release that stops orphans from forming. Until
2026-09-08 the cancel was an operator's ``workflow-reconcile --mode
orphaned-commands``; the predicate is a pure Store read, so the dispatcher now
runs it on its periodic sweep (``WorkflowDispatcher.sweep_stuck_records``).

The shape: every open command whose workflow is terminal (``FAILED``,
``SUCCEEDED``, ``BLOCKED``, ``SUPERSEDED``). A command whose workflow is still
``PENDING``/``RUNNING``/``SAFETY_PENDING`` is never touched, however old -- an
open command there is live work, and an executor holding its lease is the only
party allowed to settle it. Cancelling goes through the Store's own
``cancel_remote_commands_for_workflow`` (what the workflow-timeout and
retired-generation paths use): ``PENDING``/``WAITING`` go ``FAILED`` at once, a
``LEASED`` one gets a cancellation request the agent side honours. The
workflow row is not moved; the cancel is recorded on it as an
``OPERATOR_RECONCILED`` event with actor ``dispatcher`` through
``amend_workflow``, in the same transaction as the amend. Nothing is deleted.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from gpu_fault.models import (
    WorkflowEventKind,
    WorkflowRequest,
    WorkflowStatus,
    build_operator_event,
)
from gpu_fault.remote_command_models import RemoteCommandStatus

LOGGER = logging.getLogger(__name__)

DISPATCHER_ACTOR = "dispatcher"
AUDIT_ACTION = "cancelled orphaned remote commands"
TERMINAL_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.FAILED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    }
)
OPEN_REMOTE_STATUSES = frozenset(
    {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.LEASED,
        RemoteCommandStatus.WAITING,
    }
)
# Newest terminal workflows first: an orphan is made at the moment its workflow
# goes terminal, so it is among the most recently updated terminal rows when the
# next tick looks, and the terminal history behind that only grows.
SWEEP_LIMIT = 1000


def orphaned_commands(
    workflow: WorkflowRequest,
    commands: Iterable[Any],
) -> list[Any]:
    """The open commands ``workflow`` left behind; none if it is not terminal."""

    if workflow.status not in TERMINAL_WORKFLOW_STATUSES:
        return []
    return sorted(
        (
            command
            for command in commands
            if command.workflow_request_id == workflow.request_id
            and command.status in OPEN_REMOTE_STATUSES
        ),
        key=lambda command: str(command.command_id),
    )


def cancellation_reason(workflow: WorkflowRequest, actor: str) -> str:
    return (
        f"{actor} reconciliation: cancelled remote commands orphaned by "
        f"{workflow.status.value} workflow {workflow.request_id}"
    )


def record_cancellation(
    store: Any,
    workflow: WorkflowRequest,
    cancelled: dict[str, int],
    command_ids: list[str],
    *,
    now: datetime,
    actor: str,
) -> None:
    """Append the audit event for a cancel to the (unchanged) workflow.

    The workflow is terminal and stays as it was; the event is the one place its
    history says who cancelled the commands it left open. ``amend_workflow`` is
    the out-of-lease write and lands the event in the same transaction.
    """

    store.amend_workflow(
        workflow.request_id,
        {},
        event=build_operator_event(
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
        ),
    )


def cancel_orphaned_commands(
    store: Any,
    *,
    now: datetime,
    limit: int = SWEEP_LIMIT,
) -> dict[str, dict[str, int]]:
    """The dispatcher's sweep: cancel every open command of a terminal workflow.

    One Store cancel per workflow, isolated: a failure on one is logged and
    leaves it for the next tick without stopping the others. The audit event is
    written only when the cancel changed something, so a ``LEASED`` command
    whose cancellation is already requested does not grow the workflow's
    history every tick. Returns the Store's counters per workflow cancelled.
    """

    workflows = store.list_workflows(
        set(TERMINAL_WORKFLOW_STATUSES), limit=limit, newest_first=True
    )
    if not workflows:
        return {}
    open_by_workflow: dict[str, list[Any]] = defaultdict(list)
    for command in store.list_remote_commands(
        workflow_request_ids=[workflow.request_id for workflow in workflows]
    ):
        if command.status in OPEN_REMOTE_STATUSES:
            open_by_workflow[str(command.workflow_request_id)].append(command)
    cancelled: dict[str, dict[str, int]] = {}
    for workflow in workflows:
        orphans = orphaned_commands(
            workflow, open_by_workflow.get(workflow.request_id, [])
        )
        if not orphans:
            continue
        command_ids = [str(command.command_id) for command in orphans]
        try:
            result = dict(
                store.cancel_remote_commands_for_workflow(
                    workflow.request_id,
                    reason=cancellation_reason(workflow, DISPATCHER_ACTOR),
                )
            )
            if sum(int(value) for value in result.values()) == 0:
                continue
            record_cancellation(
                store,
                workflow,
                result,
                command_ids,
                now=now,
                actor=DISPATCHER_ACTOR,
            )
        except Exception:  # noqa: BLE001 - keep sweeping, retry next tick
            LOGGER.exception(
                "orphaned remote command cancel failed, left for the next tick: %s",
                workflow.request_id,
            )
            continue
        cancelled[workflow.request_id] = result
        LOGGER.warning(
            "orphaned remote commands cancelled by the dispatcher: workflow=%s "
            "workflow_status=%s commands=%s cancelled=%s cancellation_requested=%s",
            workflow.request_id,
            workflow.status.value,
            ",".join(command_ids),
            result.get("cancelled", 0),
            result.get("cancellation_requested", 0),
        )
    return cancelled
