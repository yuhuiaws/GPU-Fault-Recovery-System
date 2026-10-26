"""The settled containment prefix of a plan an operator now owns.

A plan the compiler refused only for a missing executable owner runs its
containment-and-evidence prefix as safety steps and settles
BLOCKED/SAFETY_SETTLED (``WorkflowBuilder.containment_before_block``). At that
terminal write the executor says so in two places -- a reason on the incident,
written in the same transaction as the terminal state, and one hand-off notice
(``notifications/contained_operator_handoff.py``) -- so the incident reads
"contained, this capability needs an operator" rather than "escalated" alone.
This module derives both from the terminal record; the executor only calls it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.notifications.contained_operator_handoff import (
    ContainedOperatorHandoffEmailBuilder,
)
from gpu_fault.orchestration.workflow_builder import (
    missing_owner_capabilities,
    operations_without_owner,
)
from gpu_fault.store.contracts import ControlPlaneStore

LOGGER = logging.getLogger("gpu_fault.execution.executor")
NotificationSender = Callable[[str], object]


@dataclass(frozen=True)
class ContainedHandoff:
    node_ids: list[str]
    containment_operations: list[str]
    waiting_operations: list[str]
    missing_owner_capabilities: list[str]

    @property
    def incident_reasons(self) -> list[str]:
        """What ``terminal_decision`` appends to the incident in the terminal write."""

        return [self.incident_reason]

    @property
    def incident_reason(self) -> str:
        return (
            "node "
            + ", ".join(self.node_ids)
            + " contained (cordoned and quarantined) by "
            + " -> ".join(self.containment_operations)
            + "; operator action required: no executable owner for "
            + ", ".join(self.missing_owner_capabilities)
            + " ("
            + (", ".join(self.waiting_operations) or "no further step")
            + " did not run); exit: submit-remediation --disposition restore once "
            "the node is cleared, or close the incident on node evidence once "
            "the node is replaced"
        )


def settled_handoff(
    workflow: WorkflowRequest, status: WorkflowStatus
) -> ContainedHandoff | None:
    """The hand-off a record ending in ``status`` carries, or ``None`` for
    every other end. ``workflow`` is the record as it is about to be written
    (the path's ``updates``, ``blocked_kind`` included, already applied).

    Only a settled safety plan (BLOCKED/SAFETY_SETTLED) whose
    ``blocked_reasons`` name a missing executable owner is this shape; a safety
    plan settled for a policy block or missing evidence keeps the fault
    family's quiet QUARANTINED hand-off it always had.
    """

    if (
        status is not WorkflowStatus.BLOCKED
        or workflow.blocked_kind is not BlockedKind.SAFETY_SETTLED
        or not workflow.executes_safety_steps
    ):
        return None
    capabilities = missing_owner_capabilities(workflow.blocked_reasons)
    if not capabilities:
        return None
    ran = [step.operation for step in workflow.safety_steps]
    unowned = operations_without_owner(capabilities)
    waiting = [
        step.operation for step in workflow.official_steps if step.operation not in ran
    ]
    for operation in unowned:
        if operation not in waiting:
            waiting.append(operation)
    return ContainedHandoff(
        node_ids=sorted(
            {node for step in workflow.safety_steps for node in step.node_ids}
        ),
        containment_operations=[operation.value for operation in ran],
        waiting_operations=[operation.value for operation in waiting],
        missing_owner_capabilities=capabilities,
    )


def notify_contained_operator_handoff(
    store: ControlPlaneStore,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    handoff: ContainedHandoff,
    *,
    builder: ContainedOperatorHandoffEmailBuilder,
    sender: NotificationSender | None,
) -> None:
    """Run after the terminal write; a failure here must not undo it."""

    try:
        notification = builder.build(
            cluster_id=incident.cluster_id,
            incident_id=incident.incident_id,
            workflow_id=workflow.request_id,
            event_id=incident.event_id,
            node_ids=handoff.node_ids or list(incident.node_ids),
            containment_operations=handoff.containment_operations,
            waiting_operations=handoff.waiting_operations,
            missing_owner_capabilities=handoff.missing_owner_capabilities,
            blocked_reasons=list(workflow.blocked_reasons),
            policy_source=incident.policy_source,
            official_action=incident.official_action,
            runtime_profile_version=workflow.runtime_profile_version,
            reasons=list(incident.reasons),
        )
        notification = store.save_notification_if_absent(notification)
        if sender is not None:
            sender(notification.notification_id)
    except Exception:  # noqa: BLE001 - the terminal write already landed
        LOGGER.exception(
            "notifying the contained operator hand-off for incident %s workflow %s",
            incident.incident_id,
            workflow.request_id,
        )
