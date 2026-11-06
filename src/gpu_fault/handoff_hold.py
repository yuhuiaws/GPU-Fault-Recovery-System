"""The nodes a remediation's hand-off chain leaves cordoned and quarantined.

The support hand-off mail (``adapters/support_escalation.py``, and the chain
terminator's copy in ``orchestration/escalation.py``) says what the remediation
did to each node it names. The product's own records are the source: the
hand-off workflow and the workflows it escalated from. MARK_UNSCHEDULABLE and
QUARANTINE are one Kubernetes patch -- the cordon, the
``gpu-fault.io/quarantined`` taint and the incident annotations
(``adapters/kubernetes/adapter.py`` routes both to ``_isolate``) -- so either
one that SUCCEEDED holds the nodes of its step, and a FAILED one holds the
nodes its ``isolated_nodes`` detail names (a partial isolation still holds
them: ``execution/terminal_state.failure_incident_state``). A SUCCEEDED
RESTORE_SCHEDULING releases the nodes of its step. A node another incident
holds, or one cordoned by hand, is not this remediation's hold and is not
reported as one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepStatus,
    execution_phase,
)

#: The step details' ``hardware_disposition``: every named node held, none,
#: or some of them.
OFFLINE_QUARANTINED = "OFFLINE_QUARANTINED"
NOT_ISOLATED = "NOT_ISOLATED"
PARTIALLY_QUARANTINED = "PARTIALLY_QUARANTINED"
_ISOLATION = frozenset(
    {WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.QUARANTINE}
)


def _step_nodes(
    workflow: WorkflowRequest, execution: WorkflowStepExecution
) -> set[str]:
    phase = execution.phase or execution_phase(workflow)
    steps = workflow.safety_steps if phase == "safety" else workflow.official_steps
    if not 0 <= execution.step_index < len(steps):
        return set()
    return set(steps[execution.step_index].node_ids)


def held_nodes(
    workflows: Iterable[WorkflowRequest], node_ids: Iterable[str]
) -> list[str]:
    """The nodes of ``node_ids`` the records of ``workflows`` -- newest first:
    the hand-off and the workflows it escalated from -- leave held.

    The oldest record is read first and each one in execution order, so the
    last word on a node wins: a later RESTORE_SCHEDULING releases an earlier
    cordon, a later cordon holds a node an earlier restore released.
    """

    held: set[str] = set()
    for workflow in reversed(list(workflows)):
        for execution in workflow.step_executions:
            if execution.operation is WorkflowOperation.RESTORE_SCHEDULING:
                if execution.status is WorkflowStepStatus.SUCCEEDED:
                    held -= _step_nodes(workflow, execution)
            elif execution.operation in _ISOLATION:
                if execution.status is WorkflowStepStatus.SUCCEEDED:
                    held |= _step_nodes(workflow, execution)
                elif execution.status is WorkflowStepStatus.FAILED:
                    isolated = execution.details.get("isolated_nodes")
                    if isinstance(isolated, list):
                        held |= {str(node) for node in isolated}
    return [node for node in dict.fromkeys(node_ids) if node in held]


def hold_disposition(node_ids: Sequence[str], held: Sequence[str]) -> str:
    """The ``hardware_disposition`` of a hand-off naming ``node_ids``."""

    if not held:
        return NOT_ISOLATED
    if set(node_ids) <= set(held):
        return OFFLINE_QUARANTINED
    return PARTIALLY_QUARANTINED
