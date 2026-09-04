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
  waiting to be closed rather than live danger;
* nodes *outside* the wave whose agent is not healthy -- taking the wave down
  then leaves the cluster with no margin at all.
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus
from gpu_fault.operation_registry import DESTRUCTIVE_OPERATIONS
from gpu_fault.workflow_resolution import verified_restore_successor

NONTERMINAL = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.BLOCKED,
    WorkflowStatus.RUNNING,
}


def main() -> None:
    expected = json.load(sys.stdin)
    context = ApplicationContext.from_environment()
    remote = context.store.remote_command_stats()
    open_remote = {
        status: int((remote.get("by_status") or {}).get(status, 0))
        for status in ("PENDING", "LEASED", "WAITING")
    }
    destructive: list[str] = []
    resolved_destructive: list[str] = []
    for item in context.store.list_workflows(statuses=NONTERMINAL, limit=1001):
        if not any(
            step.operation in DESTRUCTIVE_OPERATIONS for step in item.official_steps
        ):
            continue
        if verified_restore_successor(context.store, item) is not None:
            resolved_destructive.append(item.request_id)
        else:
            destructive.append(item.request_id)
    now = datetime.now(timezone.utc)
    excluded = set(expected["wave"])
    required = set(expected["nodes"]) - excluded
    agents = {
        agent.node_id: agent
        for agent in context.store.list_agents(expected["cluster_id"])
        if agent.node_id in required
    }
    agent_blockers: list[dict[str, Any]] = []
    agent_blocker_count = 0
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
        "agent_blocker_count": agent_blocker_count,
        "agent_blockers": agent_blockers,
    }
    print(json.dumps(result, sort_keys=True))


main()
