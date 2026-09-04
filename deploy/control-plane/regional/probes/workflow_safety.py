"""Report non-terminal destructive workflows, split by whether they can still act.

Request: nothing on stdin.
Response: one JSON object on stdout with ``blockers`` / ``blocker_count``,
``resolved_blocked`` / ``resolved_blocked_count`` and ``abandoned_generation`` /
``abandoned_generation_count``.

A destructive workflow that is still non-terminal is a reason not to touch the
fleet. Two kinds of record are not:

``resolved_by_restore`` -- BLOCKED is also where a workflow ends up when a
*later* workflow already recovered the incident and restored scheduling: a
record waiting to be closed, not a node still cordoned. Reporting both as
blockers made every release wait on paperwork. The successor test is
deliberately three-part -- the incident is RECOVERED, it names a *different*
workflow, and that workflow SUCCEEDED with ``RESTORE_SCHEDULING`` among its
completed operations. Any weaker test would clear a workflow whose node is still
unschedulable.

``abandoned_generation`` -- a workflow whose incident re-planned away from it
without linking it as a preempted predecessor. It is PENDING with no step ever
handed to an adapter, holds no lease and no budget claim, and its incident names
a different workflow at a strictly higher generation. It is not "a remediation
in progress" under any reading: every step it could attempt would be rejected,
and nothing it holds needs releasing before a rollout. Counting it as a blocker
wedged releases permanently -- on 2026-09-04 one such record blocked every
release for four and a half hours, including the release carrying its own fix.

That last case is why this test is inlined here rather than imported from
``gpu_fault.workflow_resolution``, where the product-side copy lives. The engine
ships this file to the Pod as source (see ``probes/README.md``) and it runs
against whatever ``gpu_fault`` is *already deployed* -- which, for the release
that first carries the fix, is a module without it.
``tests/regional/test_release_workflow_safety.py`` pins the two copies together.

The ``except Exception: return False`` arms fail closed: a Store read that
cannot prove a workflow harmless leaves it counted as a blocker.
"""

import json
from datetime import datetime, timezone
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.operation_registry import DESTRUCTIVE_OPERATIONS

STATUSES = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.BLOCKED,
    WorkflowStatus.RUNNING,
}


def main() -> None:
    context = ApplicationContext.from_environment()
    store = context.store

    def resolved_by_restore(workflow: Any) -> bool:
        if workflow.status is not WorkflowStatus.BLOCKED:
            return False
        try:
            incident = store.get_incident(workflow.incident_id)
        except Exception:
            return False
        successor_id = incident.workflow_request_id
        if (
            incident.state is not IncidentState.RECOVERED
            or not successor_id
            or successor_id == workflow.request_id
        ):
            return False
        try:
            successor = store.get_workflow(successor_id)
        except Exception:
            return False
        return bool(
            successor.status is WorkflowStatus.SUCCEEDED
            and WorkflowOperation.RESTORE_SCHEDULING in successor.completed_operations
        )

    now = datetime.now(timezone.utc)

    def abandoned_generation(workflow: Any) -> bool:
        if (
            workflow.status is not WorkflowStatus.PENDING
            or workflow.step_executions
            or workflow.completed_step_indexes
            or workflow.completed_operations
            or workflow.remediation_budget_claims
            or workflow.execution_owner_id is not None
            or (
                workflow.execution_lease_expires_at is not None
                and workflow.execution_lease_expires_at > now
            )
        ):
            return False
        try:
            incident = store.get_incident(workflow.incident_id)
        except Exception:
            return False
        successor_id = incident.workflow_request_id
        if (
            not successor_id
            or successor_id == workflow.request_id
            or workflow.fencing_token >= incident.fencing_token
        ):
            return False
        try:
            successor = store.get_workflow(successor_id)
        except Exception:
            return False
        return bool(
            successor.incident_id == workflow.incident_id
            and successor.fencing_token == incident.fencing_token
        )

    blockers: list[str] = []
    resolved: list[str] = []
    abandoned: list[str] = []
    for workflow in store.list_workflows(statuses=STATUSES, limit=1001):
        if not any(
            step.operation in DESTRUCTIVE_OPERATIONS for step in workflow.official_steps
        ):
            continue
        if resolved_by_restore(workflow):
            resolved.append(workflow.request_id)
        elif abandoned_generation(workflow):
            abandoned.append(workflow.request_id)
        else:
            blockers.append(workflow.request_id)
    print(
        json.dumps(
            {
                "blockers": blockers[:100],
                "blocker_count": len(blockers),
                "resolved_blocked": resolved[:100],
                "resolved_blocked_count": len(resolved),
                "abandoned_generation": abandoned[:100],
                "abandoned_generation_count": len(abandoned),
            },
            sort_keys=True,
        )
    )


main()
