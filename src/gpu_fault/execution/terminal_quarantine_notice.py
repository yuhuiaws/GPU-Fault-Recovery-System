"""The notice a remediation owes when it ends holding its node.

A record that ends SUCCEEDED with its own quarantine still on the node leaves
the incident QUARANTINED: cordoned and tainted, with no step left that would
release it. The node-health QUARANTINE plan of a BMC sensor or a host signal
(FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> [STOP_WORKLOADS] -> QUARANTINE) is
that shape, and so is any finding graded QUARANTINE and a critical quarantine
folded into another chain as its terminal branch. None of them mails: the
isolation step saves no notification and there is no ESCALATE_SUPPORT, so the
record used to end with nobody told. At the terminal write the executor sends
one ``quarantined-node-notice-zh-v1`` notice per incident
(``notifications/quarantined_node_notice.py``); the incident state and the
operator's exits (the validated restore, the departure close) do not change.

Not this end: a hold the incident's *predecessor* left (the record itself
holds nothing; ``terminal_state.predecessor_holds_isolation``), a support
hand-off (ESCALATED, its own mail), a settled safety plan (BLOCKED; the
contained hand-off speaks for the missing-owner shape) and every FAILED record
(the escalation ladder's: a failed step after the cordon of such a plan hands
the held node to support, ``failure_outcome.held_chain_hand_off``). This
module reads the record as written; the executor only calls it
(``workflow_notifications.after_terminal_write``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from gpu_fault.execution.terminal_state import TerminalDecision
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.notifications.quarantined_node_notice import (
    QuarantinedNodeNoticeEmailBuilder,
)
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.suspended_workloads import suspended_workload_line, suspended_workloads
from gpu_fault.workflow_quarantine import (
    has_unrecovered_quarantine,
    terminal_quarantine_nodes,
)

LOGGER = logging.getLogger("gpu_fault.execution.executor")
NotificationSender = Callable[[str], object]
BUILDER = QuarantinedNodeNoticeEmailBuilder()


@dataclass(frozen=True)
class TerminalQuarantine:
    node_ids: list[str]
    steps_run: list[str]
    suspended_workloads: list[str]


def terminal_quarantine(decision: TerminalDecision) -> TerminalQuarantine | None:
    """What the notice names for a record that ended holding its own node,
    or ``None`` for every other end (see the module docstring)."""

    workflow, incident = decision.workflow, decision.incident
    if (
        incident is None
        or incident.state is not IncidentState.QUARANTINED
        or workflow.status is not WorkflowStatus.SUCCEEDED
        or not has_unrecovered_quarantine(workflow)
    ):
        return None
    try:
        held = sorted(terminal_quarantine_nodes(workflow))
    except ValueError:
        held = []
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    completed = set(workflow.completed_step_indexes)
    return TerminalQuarantine(
        node_ids=held or sorted(incident.node_ids),
        steps_run=[
            step.operation.value
            for index, step in enumerate(steps)
            if index in completed
        ],
        suspended_workloads=[
            suspended_workload_line(item) for item in suspended_workloads([workflow])
        ],
    )


def notify_terminal_quarantine(
    store: ControlPlaneStore,
    decision: TerminalDecision,
    held: TerminalQuarantine,
    *,
    sender: NotificationSender | None,
) -> None:
    """Run after the terminal write; a failure here must not undo it. The
    deduplication key is the incident's, so a second terminal write of the
    same hold finds the saved notice and the send is the idempotent one."""

    workflow, incident = decision.workflow, decision.incident
    if incident is None:
        return
    try:
        notification = BUILDER.build(
            cluster_id=incident.cluster_id,
            incident_id=incident.incident_id,
            workflow_id=workflow.request_id,
            event_id=incident.event_id,
            node_ids=held.node_ids,
            steps_run=held.steps_run,
            suspended_workloads=held.suspended_workloads,
            policy_source=incident.policy_source,
            recommended_action=(
                incident.effective_action.value if incident.effective_action else None
            ),
            runtime_profile_version=workflow.runtime_profile_version,
            reasons=list(incident.reasons),
        )
        notification = store.save_notification_if_absent(notification)
        if sender is not None:
            sender(notification.notification_id)
    except Exception:  # noqa: BLE001 - the terminal write already landed
        LOGGER.exception(
            "notifying the quarantined node of incident %s workflow %s",
            incident.incident_id,
            workflow.request_id,
        )
