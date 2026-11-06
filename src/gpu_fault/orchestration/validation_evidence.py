"""What a FAILED validation step's record says about the node it judged.

The validation adapter (``adapters/gpu_validation.py``) writes two kinds of
record. A verdict -- ``failed_nodes`` / ``node_failures`` -- names the nodes
it looked at and found failing on fresh readings. A wait -- ``node_pending``
with ``pending_nodes``, ``failed_nodes_observed`` and
``node_failures_observed`` -- names what each node is still missing
(``missing_or_stale_collectors``, ``missing_fresh_inventory_metric``,
``missing_recent_metrics``, the quiesce taint, ...) and which nodes it has
already seen failing while it waits for the others. The per-step cap
(``execution/step_bounds.bounded_waiting_outcome``) fails a step that waited
too long by stamping ``step_waiting_timeout_seconds`` onto that last wait, so a
cap-failed validation's FAILED record is the wait itself: no verdict, and
whatever the wait had observed. The workflow deadline
(``workflow_deadline_failure``) fails the step with its own record and no
verdict at all.

Both escalation ladders read these to tell "looked and found a fault" from
"could not look". The quiesce-taint case (``validation_unobserved_failure``)
came first: live 2026-10-07 (gpu-eks-cluster DESTR-025). The general case
followed live 2026-10-25 (plain EKS, GF-GENERAL-LIVE-013 ``link_inactive``):
the EFA diagnostic chain's VALIDATE_FABRIC, whose plan carries
``failure_escalation_action=REBOOT_NODE``, failed at the 600 s cap with the
node's collectors dark and no node observed failing, and the requested reboot
fired on a node nothing had been learned about
(``validation_without_observed_failure``, ``requested_escalations``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepStatus,
)

VALIDATION_OPERATIONS = frozenset(
    {
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
    }
)
#: The plan parameter a step asks for an escalation with when it fails
#: (``orchestration/families/health.py``: the EFA driver, device-plugin and
#: diagnostic chains ask for ``REBOOT_NODE``).
FAILURE_ESCALATION_ACTION = "failure_escalation_action"
#: The detail keys the bounds in ``execution/step_bounds.py`` stamp on a step
#: they failed: the per-step cap and the workflow execution deadline.
_BOUND_FAILURE_KEYS = ("step_waiting_timeout_seconds", "workflow_execution_deadline")
_VERDICT_KEYS = ("failed_nodes", "node_failures")
_OBSERVED_FAILURE_KEYS = ("failed_nodes_observed", "node_failures_observed")


def validation_without_observed_failure(details: Mapping[str, Any] | None) -> bool:
    """Whether a FAILED validation ended without observing a node failure.

    True when the record carries no verdict (no ``failed_nodes`` /
    ``node_failures``), no node observed failing (``failed_nodes_observed`` /
    ``node_failures_observed`` empty) and either the wait the step was in
    (``node_pending``) or the mark of the bound that failed it (the per-step
    cap's ``step_waiting_timeout_seconds``, the deadline's
    ``workflow_execution_deadline``). Such a step's verdict is "could not
    look", not "looked and found a fault": a reboot or a replacement planned
    on it acts on a node nothing was learned about. A record with none of
    these -- an adapter failure with its own details, a legacy row -- is not
    judged here and keeps its classification.
    """

    if details is None:
        return False
    if any(details.get(key) for key in _VERDICT_KEYS + _OBSERVED_FAILURE_KEYS):
        return False
    pending = details.get("node_pending")
    if isinstance(pending, Mapping) and pending:
        return True
    return any(key in details for key in _BOUND_FAILURE_KEYS)


def validation_unobserved_failure(details: Mapping[str, Any] | None) -> bool:
    """Whether a FAILED validation never observed its node because the
    product's own quiesce taint kept the exporter off it.

    True only when every node the step was still waiting for is pending with
    ``quiesce_taint_present`` and no node was observed failing: the step's
    verdict is "could not look", not "looked and found a fault". Both ladders
    read this and climb on neither -- a REPLACE_NODE after such a validation
    replaces a node nothing was learned about.
    """

    if details is None:
        return False
    pending = details.get("node_pending")
    if not isinstance(pending, Mapping) or not pending:
        return False
    if not validation_without_observed_failure(details):
        return False
    return all(
        isinstance(record, Mapping) and record.get("quiesce_taint_present") is True
        for record in pending.values()
    )


def withheld_escalation(
    workflow: WorkflowRequest, execution: WorkflowStepExecution
) -> str | None:
    """The ``failure_escalation_action`` a FAILED validation asked for and is
    not granted, or ``None``.

    The plan's request is for a validation that looked and found the fault
    still there -- the adapter's verdict, or a node observed failing while the
    step waited for its siblings. A validation the cap or the deadline failed
    before it observed any failure asks for nothing: the record goes to an
    operator with the missing evidence named.
    """

    if (
        execution.status is not WorkflowStepStatus.FAILED
        or execution.operation not in VALIDATION_OPERATIONS
    ):
        return None
    steps = workflow.official_steps
    if not 0 <= execution.step_index < len(steps):
        return None
    action = steps[execution.step_index].parameters.get(FAILURE_ESCALATION_ACTION)
    if not action or not validation_without_observed_failure(execution.details):
        return None
    return str(action)


def held_for_unsent_hand_off(workflow: WorkflowRequest) -> bool:
    """Whether ``workflow`` holds its node for an operator hand-off that never ran.

    The DRAIN chain -- the node-health one and the ladder's ``drain-after-``
    successor -- cordons and quarantines its node, leaves it held (RMA class,
    no RESTORE_SCHEDULING) and hands it over with the ESCALATE_SUPPORT behind
    its VALIDATE_GPU, which an RMA-class GPU often still fails. No recovery
    action ran before that validation, so neither ladder has a rung for it; the
    hand-off is still owed, or the drain parks QUARANTINED with no support
    record and no mail. True when an isolation of the record completed and its
    planned ESCALATE_SUPPORT has no execution record at all: a hand-off that ran
    or failed keeps its own ending, and a settled containment prefix (no
    ESCALATE_SUPPORT among its safety steps) keeps its own notice.
    """

    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    return (
        bool(
            {WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.QUARANTINE}
            & set(workflow.completed_operations)
        )
        and any(step.operation is WorkflowOperation.ESCALATE_SUPPORT for step in steps)
        and not any(
            execution.operation is WorkflowOperation.ESCALATE_SUPPORT
            for execution in workflow.step_executions
        )
    )


def requested_escalations(workflow: WorkflowRequest) -> set[str]:
    """The ``failure_escalation_action`` values the record's FAILED steps ask
    for and are owed: a failed node action's as designed, a failed
    validation's only when it observed the failure (``withheld_escalation``)."""

    steps = workflow.official_steps
    requested: set[str] = set()
    for execution in workflow.step_executions:
        if (
            execution.status is not WorkflowStepStatus.FAILED
            or not 0 <= execution.step_index < len(steps)
        ):
            continue
        action = steps[execution.step_index].parameters.get(FAILURE_ESCALATION_ACTION)
        if action and withheld_escalation(workflow, execution) is None:
            requested.add(str(action))
    return requested


def withheld_escalation_reasons(workflow: WorkflowRequest) -> list[str]:
    """One incident reason per validation whose requested escalation was
    withheld: which request, which step, which bound failed it."""

    reasons: list[str] = []
    for execution in workflow.step_executions:
        action = withheld_escalation(workflow, execution)
        if action is None:
            continue
        details = execution.details
        if "step_waiting_timeout_seconds" in details:
            bound = (
                f"the {details['step_waiting_timeout_seconds']}s per-step cap failed it"
            )
        elif "workflow_execution_deadline" in details:
            bound = "the workflow execution deadline failed it"
        else:
            bound = "it failed"
        nodes = ", ".join(workflow.official_steps[execution.step_index].node_ids)
        reasons.append(
            f"requested escalation {action} withheld: {execution.operation.value} "
            f"on {nodes} observed no node failure before {bound} -- missing "
            "evidence, not a verdict; an operator reviews the node"
        )
    return list(dict.fromkeys(reasons))


def missing_evidence_reason(record: Mapping[str, Any]) -> str | None:
    """The wait a validation's ``node_pending`` record names, as one reason.

    The adapter's own ``reason`` when it wrote one (the quiesce taint names
    its holders and the operator's exit); else every key of the record --
    lists and mapping keys joined, flags by name, scalars as ``key: value``
    -- so a hand-off says ``missing_or_stale_collectors: GPU_METRICS,
    HOST_TELEMETRY`` instead of "validation failed".
    """

    reason = record.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    parts: list[str] = []
    for key in sorted(record):
        value = record[key]
        if key == "reason" or value is None or value is False:
            continue
        if value is True:
            parts.append(str(key))
        elif isinstance(value, Mapping):
            parts.append(f"{key}: {', '.join(sorted(str(item) for item in value))}")
        elif isinstance(value, (list, tuple, set, frozenset)):
            parts.append(f"{key}: {', '.join(str(item) for item in value)}")
        else:
            parts.append(f"{key}: {value}")
    return "; ".join(parts) or None
