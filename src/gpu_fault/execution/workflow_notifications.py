"""Notification persistence and post-terminal effects with explicit dependencies."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace

from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.execution.terminal_state import inconclusive_reason
from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.notifications import (
    DiagnosticInconclusiveEmailBuilder,
    WarmSpareReplacementEmailBuilder,
)
from gpu_fault.store.contracts import ControlPlaneStore

LOGGER = logging.getLogger("gpu_fault.execution.executor")
NotificationSender = Callable[[str], object]
TerminalHook = Callable[
    [WorkflowRequest, FaultIncident | None, list[WorkflowStepSpec]], None
]


def persist_spare_completion(
    store: ControlPlaneStore,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    step: WorkflowStepSpec,
    step_index: int,
    outcome: WorkflowStepOutcome,
    *,
    builder: WarmSpareReplacementEmailBuilder,
    sender: NotificationSender | None,
) -> WorkflowStepOutcome:
    details = outcome.details or {}
    if (
        outcome.status is not WorkflowStepStatus.SUCCEEDED
        or step.operation is not WorkflowOperation.REPLACE_NODE
        or details.get("action") != "SPARE_FAILOVER"
    ):
        return outcome
    notification = builder.build(
        cluster_id=incident.cluster_id,
        incident_id=incident.incident_id,
        workflow_id=workflow.request_id,
        event_id=incident.event_id,
        policy_source=incident.policy_source,
        official_action=incident.official_action,
        effective_action=(
            incident.effective_action.value if incident.effective_action else None
        ),
        reasons=incident.reasons,
        operation_id=f"{workflow.request_id}/{step_index}/REPLACE_NODE",
        fault_node_ids=step.node_ids,
        spare_node_ids=list(details.get("activated_spare_nodes") or []),
        node_rebindings={
            str(key): str(value)
            for key, value in (details.get("node_rebindings") or {}).items()
        },
        confirmation_source=str(
            details.get("confirmation_source", "healthy-running-warm-spare")
        ),
        provider_mutation_submitted=bool(
            details.get("provider_mutation_submitted", False)
        ),
    )
    notification = store.save_notification_if_absent(notification)
    if sender is not None:
        sender(notification.notification_id)
    return replace(
        outcome,
        details={
            **details,
            "notification_id": details.get("notification_id")
            or notification.notification_id,
        },
    )


def close_inconclusive_diagnostic(
    store: ControlPlaneStore,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    error: str | None,
    *,
    actor: str,
    builder: DiagnosticInconclusiveEmailBuilder,
    sender: NotificationSender | None,
) -> None:
    """Run after the terminal write; either effect may fail without undoing it."""

    try:
        retire_markers_for_incident(
            store,
            incident.incident_id,
            reason=inconclusive_reason(error),
            retired_by=actor,
        )
    except Exception:  # noqa: BLE001 - the terminal write already landed
        LOGGER.exception(
            "retiring markers of incident %s after inconclusive workflow %s",
            incident.incident_id,
            workflow.request_id,
        )
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    failed = [
        item
        for item in workflow.step_executions
        if item.status is WorkflowStepStatus.FAILED
    ]
    try:
        notification = builder.build(
            cluster_id=incident.cluster_id,
            incident_id=incident.incident_id,
            workflow_id=workflow.request_id,
            event_id=incident.event_id,
            node_ids=sorted(
                {node for step in steps for node in step.node_ids}
                or set(incident.node_ids)
            ),
            operations=[step.operation.value for step in steps],
            failed_operation=failed[-1].operation.value if failed else None,
            error=error,
            policy_source=incident.policy_source,
            official_action=incident.official_action,
            reasons=incident.reasons,
        )
        notification = store.save_notification_if_absent(notification)
        if sender is not None:
            sender(notification.notification_id)
    except Exception:  # noqa: BLE001 - the terminal write already landed
        LOGGER.exception(
            "notifying inconclusive diagnostic for incident %s workflow %s",
            incident.incident_id,
            workflow.request_id,
        )


def notify_terminal_hooks(
    hooks: Sequence[TerminalHook],
    workflow: WorkflowRequest,
    incident: FaultIncident | None,
) -> None:
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    for hook in list(hooks):
        try:
            hook(workflow, incident, list(steps))
        except Exception:  # noqa: BLE001 - one hook must not stop the rest
            LOGGER.exception(
                "on_terminal hook failed: workflow=%s status=%s hook=%s",
                workflow.request_id,
                workflow.status.value,
                getattr(hook, "__qualname__", repr(hook)),
            )
