"""A node action whose per-node key has not reached the executor yet.

The installer writes a node's key into the node-action-keys Secret and the
node's Agent registers at the end of the same install; the executor sees the
Secret's new entry only once the kubelet refreshes the mounted volume. Live
2026-10-08: a RESTART_GPU_DEVICE_PLUGIN workflow for a replacement node failed
terminally with "node action secret is unavailable" at the exact second the
node's installer Job completed; a destructive action would have quarantined a
healthy fresh node the same way. Inside a short grace after the Agent's first
registration the step waits instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.node_action_keys import NodeActionKeyUnavailableError

#: How long after a node's Agent first registered a missing per-node action key
#: holds its step WAITING instead of failing it.
NODE_ACTION_KEY_PROPAGATION_GRACE = timedelta(minutes=10)


def key_propagation_hold(
    context: WorkflowStepContext,
    node_id: str,
    exc: Exception,
    record: Any,
    *,
    now: datetime,
) -> WorkflowStepOutcome | None:
    """WAITING while a freshly registered node's action key is still propagating.

    Only for the typed key refusal, only when the fleet record exists and its
    ``first_seen_at`` is within ``NODE_ACTION_KEY_PROPAGATION_GRACE`` of now:
    the node was onboarded moments ago and the key is on its way. Anything
    else -- no record, an old record, another refusal -- keeps the fail-closed
    verdict. Nothing was sent, so the hold says so for the dispatcher.
    """

    if not isinstance(exc, NodeActionKeyUnavailableError) or record is None:
        return None
    first_seen = getattr(record, "first_seen_at", None)
    if not isinstance(first_seen, datetime):
        return None
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)
    age = now - first_seen
    if age < timedelta(0) or age > NODE_ACTION_KEY_PROPAGATION_GRACE:
        return None
    grace = int(NODE_ACTION_KEY_PROPAGATION_GRACE.total_seconds())
    return WorkflowStepOutcome.waiting(
        operation_id=context.idempotency_key,
        details={
            "node_action_state": "KEY_PENDING",
            "waiting_node": node_id,
            "node_action_not_started": True,
            "node_action_key_pending": True,
            "agent_first_seen_at": first_seen.isoformat(),
            "reason": (
                f"node action key for {node_id} is not yet available on this "
                f"executor; the node's Agent registered {int(age.total_seconds())}s "
                f"ago (key propagation grace {grace}s)"
            ),
        },
    )
