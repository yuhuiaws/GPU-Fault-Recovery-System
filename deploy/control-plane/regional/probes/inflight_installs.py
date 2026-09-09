"""Report driver / firmware / EFA-driver install steps that are still in flight.

Request: nothing on stdin.
Response: one JSON object on stdout with ``inflight`` (at most 100 entries,
each ``workflow_id`` / ``incident_id`` / ``workflow_status`` / ``operation`` /
``step_index`` / ``node_ids`` / ``step_status``) and ``inflight_count``.

Why the release engine asks: R4 (2026-09-09) moved the node-action command id
of REMEDIATE_DRIVER / UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER from
``<key>/<node>/agent-N`` to ``<key>/<node>``. A control plane of the *other*
shape that takes over such a step mid-install finds no agent-ledger row under
the id it derives and submits the install a second time, on a node that is
still running the first one (install timeout 1800 s). The release engine
therefore refuses to open an upgrade or rollback transaction while one of these
steps is PENDING or WAITING, and this probe is the one store read behind that
refusal.

A step is in flight when its workflow is still non-terminal, its index is
neither completed nor superseded, and its latest execution record is absent
(PENDING: the executor has not handed it to an adapter yet) or WAITING (the
node agent has it). A latest execution that FAILED is not in flight: failure
handling, not a retry, is what happens next. BLOCKED workflows never execute
again on their own, so only a step already WAITING there counts.

The operation set is spelled out here rather than imported from
``gpu_fault.operation_registry.GENERATION_STABLE_COMMAND_OPERATIONS``: the
engine ships this file to the Pod as source (see ``probes/README.md``) and it
runs against whatever ``gpu_fault`` is *already deployed* -- for the release
that first carries R4, and for every rollback target, a module without that
name. ``tests/regional/test_release_inflight_install_gate.py`` pins the two
sets equal.

The ``except Exception: continue`` arm fails closed in the only direction that
is safe here: a workflow the probe cannot classify is counted, never skipped.
"""

import json
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus

INSTALL_OPERATIONS = {
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
}
STATUSES = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.BLOCKED,
    WorkflowStatus.RUNNING,
}
LIMIT = 100


def step_state(workflow: Any, index: int) -> str | None:
    """``PENDING`` / ``WAITING`` (or another live execution status) when the
    step is in flight, ``None`` when it is not."""

    if index in set(workflow.completed_step_indexes) or index in set(
        workflow.superseded_step_indexes
    ):
        return None
    latest = None
    for execution in workflow.step_executions:
        if execution.step_index == index:
            latest = execution
    if latest is None:
        return None if workflow.status is WorkflowStatus.BLOCKED else "PENDING"
    if latest.status is WorkflowStepStatus.FAILED:
        return None
    if workflow.status is WorkflowStatus.BLOCKED and (
        latest.status is not WorkflowStepStatus.WAITING
    ):
        return None
    return str(latest.status.value)


def main() -> None:
    store = ApplicationContext.from_environment().store
    inflight: list[dict[str, Any]] = []
    for workflow in store.list_workflows(statuses=STATUSES, limit=1001):
        for index, step in enumerate(workflow.official_steps):
            if step.operation not in INSTALL_OPERATIONS:
                continue
            try:
                state = step_state(workflow, index)
            except Exception:
                state = "UNKNOWN"
            if state is None:
                continue
            inflight.append(
                {
                    "workflow_id": workflow.request_id,
                    "incident_id": workflow.incident_id,
                    "workflow_status": workflow.status.value,
                    "operation": step.operation.value,
                    "step_index": index,
                    "node_ids": list(step.node_ids),
                    "step_status": state,
                }
            )
    print(
        json.dumps(
            {"inflight": inflight[:LIMIT], "inflight_count": len(inflight)},
            sort_keys=True,
        )
    )


main()
