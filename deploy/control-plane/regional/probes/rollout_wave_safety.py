"""Report what would be unsafe about restarting the agents in one wave.

Request on stdin::

    {"cluster_id": "...", "nodes": [...], "wave": [...],
     "minimum_lease_remaining_seconds": 30}

Response: one JSON object on stdout. It reports; it does not decide. The engine
compares the counts against the wave policy, so a new safety rule is a change
here plus a change there, never a silent one.

Three independent things make a wave unsafe, and each is counted separately:

* open remote commands -- work already handed to an agent that a restart drops;
* non-terminal destructive workflows -- a wave that restarts an agent
  mid-``RESET_GPU`` leaves the GPU in whatever state the reset reached.
  ``verified_restore_successor`` splits these: one whose restore was already
  verified by a later workflow is reported separately, because it is a record
  waiting to be closed rather than live danger; so is a BLOCKED record whose
  node action an operator confirmed on evidence and that holds nothing live
  (``operator_confirmed_destructive_workflows``, the same predicate as the
  preflight's ``operator_confirmed_parked``) -- it cannot act, and the agent a
  wave restarts is not mid-anything on its behalf;
* nodes *outside* the wave whose agent is not healthy -- taking the wave down
  then leaves the cluster with no margin at all.

A destructive record that is PENDING and was *never dispatched* -- no step
execution, no completed or superseded step, no owner, no live lease, no budget
claim, no remote command, older than the product's ten-minute dispatch guard,
still named by its incident (``never_dispatched_reconciliation_reasons`` minus
the departed-node proof, inlined here because the Pod runs whatever
``gpu_fault`` is deployed) -- is counted like any other and *also* listed under
``never_dispatched_destructive_workflows`` with its incident's cluster and nodes.
Whether those nodes are gone is the engine's question: it reads Kubernetes and
the provider from the deploy host and sets the record aside only on that proof
(``regional_release_departed_nodes``). Live 2026-10-06: a HyperPod spot node
reclaimed between two waves left such a record, and it blocked wave 2 and the
automatic rollback alike.

The report also carries ``minimum_lease_remaining_seconds``: the smallest lease
remaining across those outside nodes, in the same units as the request's
threshold of the same name. It is what lets the engine re-decide the same
question against a *larger* margin -- the one that must still hold after the
wave lease is taken -- by subtracting the time since this read instead of
paying for a second identical exec.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus, WorkflowStepStatus
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.workflow_resolution import verified_restore_successor

NONTERMINAL = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.BLOCKED,
    WorkflowStatus.RUNNING,
}
# The product's ``NEVER_DISPATCHED_GUARD_AGE``: a younger PENDING record may
# still be about to dispatch.
NEVER_DISPATCHED_GUARD = timedelta(minutes=10)
# The product's ``operator_confirmation`` reader, inlined like the preflight's
# ``workflow_safety.py`` copy: the Pod runs whatever ``gpu_fault`` is deployed.
OPERATOR_CONFIRMED_KEY = "operator_confirmed"
OPERATOR_CONFIRMATION_REQUIRED_KEYS = (
    "actor",
    "reference",
    "confirmed_at",
    "node_id",
    "operation",
)


def operator_confirmed(details: Any) -> bool:
    value = details.get(OPERATOR_CONFIRMED_KEY) if isinstance(details, dict) else None
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(value.get(key), str) and value.get(key)
        for key in OPERATOR_CONFIRMATION_REQUIRED_KEYS
    )


def operator_confirmed_parked(store: Any, workflow: Any, now: datetime) -> bool:
    """BLOCKED / NEEDS_OPERATOR, a confirmed node action, nothing live on it."""

    if (
        workflow.status is not WorkflowStatus.BLOCKED
        or str(getattr(workflow, "blocked_kind", None) or "") != "NEEDS_OPERATOR"
        or workflow.execution_owner_id is not None
        or (
            workflow.execution_lease_expires_at is not None
            and workflow.execution_lease_expires_at > now
        )
        or any(
            execution.status is WorkflowStepStatus.WAITING
            for execution in workflow.step_executions
        )
        or not any(
            execution.operation in NODE_MUTATING_OPERATIONS
            and operator_confirmed(execution.details)
            for execution in workflow.step_executions
        )
    ):
        return False
    try:
        commands = store.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
    except Exception:
        return False
    return not any(
        command.status
        in (
            RemoteCommandStatus.PENDING,
            RemoteCommandStatus.LEASED,
            RemoteCommandStatus.WAITING,
        )
        for command in commands
    )


def never_dispatched_record(store: Any, workflow: Any, now: datetime) -> Any:
    """The departed-node shape minus its proof, or None: PENDING, never dispatched."""

    if (
        workflow.status is not WorkflowStatus.PENDING
        or workflow.step_executions
        or workflow.completed_step_indexes
        or workflow.superseded_step_indexes
        or workflow.completed_operations
        or workflow.execution_owner_id is not None
        or (
            workflow.execution_lease_expires_at is not None
            and workflow.execution_lease_expires_at > now
        )
        or workflow.remediation_budget_claims
        or workflow.created_at > now - NEVER_DISPATCHED_GUARD
    ):
        return None
    try:
        incident = store.get_incident(workflow.incident_id)
        commands = store.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
    except Exception:
        return None
    if (
        incident is None
        or incident.workflow_request_id != workflow.request_id
        or commands
    ):
        return None
    return {
        "request_id": workflow.request_id,
        "incident_id": incident.incident_id,
        "cluster_id": incident.cluster_id,
        "node_ids": sorted(str(node_id) for node_id in incident.node_ids),
        "created_at": workflow.created_at.isoformat(),
    }


def main() -> None:
    expected = json.load(sys.stdin)
    context = ApplicationContext.from_environment()
    remote = context.store.remote_command_stats()
    open_remote = {
        status: int((remote.get("by_status") or {}).get(status, 0))
        for status in ("PENDING", "LEASED", "WAITING")
    }
    now = datetime.now(timezone.utc)
    destructive: list[str] = []
    resolved_destructive: list[str] = []
    confirmed_destructive: list[str] = []
    never_dispatched: list[dict[str, Any]] = []
    for item in context.store.list_workflows(statuses=NONTERMINAL, limit=1001):
        if not any(
            step.operation in DESTRUCTIVE_OPERATIONS for step in item.official_steps
        ):
            continue
        if verified_restore_successor(context.store, item) is not None:
            resolved_destructive.append(item.request_id)
        elif operator_confirmed_parked(context.store, item, now):
            confirmed_destructive.append(item.request_id)
        else:
            destructive.append(item.request_id)
            record = never_dispatched_record(context.store, item, now)
            if record is not None:
                never_dispatched.append(record)
    excluded = set(expected["wave"])
    required = set(expected["nodes"]) - excluded
    agents = {
        agent.node_id: agent
        for agent in context.store.list_agents(expected["cluster_id"])
        if agent.node_id in required
    }
    agent_blockers: list[dict[str, Any]] = []
    agent_blocker_count = 0
    lease_remaining_seconds: list[float] = []
    for node_id in sorted(required):
        agent = agents.get(node_id)
        if agent is None:
            agent_blocker_count += 1
            if len(agent_blockers) < 100:
                agent_blockers.append({"node_id": node_id, "reason": "missing"})
            continue
        lifecycle = getattr(agent.lifecycle_state, "value", agent.lifecycle_state)
        lease_remaining = (
            None
            if agent.lease_expires_at is None
            else (agent.lease_expires_at - now).total_seconds()
        )
        if lease_remaining is not None:
            lease_remaining_seconds.append(lease_remaining)
        if lifecycle != "ACTIVE":
            reason = "lifecycle-" + str(lifecycle)
        elif lease_remaining is None:
            reason = "lease-missing"
        elif lease_remaining < expected["minimum_lease_remaining_seconds"]:
            reason = "lease-margin"
        else:
            continue
        agent_blocker_count += 1
        if len(agent_blockers) < 100:
            agent_blockers.append(
                {
                    "node_id": node_id,
                    "reason": reason,
                    "last_seen_at": agent.last_seen_at.isoformat(),
                    "lease_expires_at": (
                        agent.lease_expires_at.isoformat()
                        if agent.lease_expires_at is not None
                        else None
                    ),
                    "lease_remaining_seconds": lease_remaining,
                }
            )
    result = {
        "open_remote": open_remote,
        "destructive_workflow_count": len(destructive),
        "destructive_workflows": destructive[:20],
        "resolved_destructive_workflow_count": len(resolved_destructive),
        "resolved_destructive_workflows": resolved_destructive[:20],
        "operator_confirmed_destructive_workflow_count": len(confirmed_destructive),
        "operator_confirmed_destructive_workflows": confirmed_destructive[:20],
        "never_dispatched_destructive_workflow_count": len(never_dispatched),
        "never_dispatched_destructive_workflows": never_dispatched[:20],
        "agent_blocker_count": agent_blocker_count,
        "agent_blockers": agent_blockers,
        "minimum_lease_remaining_seconds": (
            min(lease_remaining_seconds) if lease_remaining_seconds else None
        ),
    }
    print(json.dumps(result, sort_keys=True))


main()
