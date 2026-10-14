"""A node action whose per-node key has not reached the executor yet.

A replacement node's key enters the node-action-keys Secret before its install
(deploy, or the executor's node-key sync between deploys) and the node's Agent
registers at the end of that install; the executor sees the Secret's new entry
only once the kubelet refreshes the mounted volume. Live 2026-10-08: a
RESTART_GPU_DEVICE_PLUGIN workflow for a replacement node failed terminally
with "node action secret is unavailable" at the exact second the node's
installer Job completed; a destructive action would have quarantined a healthy
fresh node the same way. Inside a short grace after the Agent's first
registration the step waits instead.

The wait only helps if the refreshed file is read again: the adapter loaded
its keys once, at start, so a node keyed after that stayed unknown to the
running executor -- every such node's actions failed once the grace ran out,
until a deploy restarted it. ``PropagatingNodeSecrets`` re-reads the source on
a miss.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.node_action_keys import NodeActionKeyUnavailableError

LOGGER = logging.getLogger(__name__)

#: How long after a node's Agent first registered a missing per-node action key
#: holds its step WAITING instead of failing it.
NODE_ACTION_KEY_PROPAGATION_GRACE = timedelta(minutes=10)
#: The shortest pause between two re-reads of the key source for a node this
#: executor holds no key for: one action polls every few seconds, and a node
#: that really has no key must not turn each poll into a directory scan.
NODE_ACTION_KEY_RELOAD_SECONDS = 15.0


class PropagatingNodeSecrets(dict[str, str]):
    """The executor's node keys, refreshed for a node it does not know yet.

    A ``dict`` so every reader (``resolve_node_action_secret``) keeps working.
    A miss in ``get`` re-reads ``load`` -- the mounted Secret the kubelet keeps
    current -- at most every ``interval_seconds`` and takes in the names this
    map lacks; a key it already holds is never replaced underneath a running
    action, because a key rotation restarts the executor. A source that cannot
    be read leaves the miss a miss. Logs name nodes, never values.
    """

    def __init__(
        self,
        initial: Mapping[str, str],
        *,
        load: Callable[[], Mapping[str, str]],
        clock: Callable[[], float] = time.monotonic,
        interval_seconds: float = NODE_ACTION_KEY_RELOAD_SECONDS,
    ) -> None:
        super().__init__(initial)
        self.load = load
        self.clock = clock
        self.interval_seconds = interval_seconds
        self._next_reload: float | None = None
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        value = super().get(key)
        if value is not None:
            return value
        self._reload()
        return super().get(key, default)

    def _reload(self) -> None:
        with self._lock:
            now = self.clock()
            if self._next_reload is not None and now < self._next_reload:
                return
            self._next_reload = now + self.interval_seconds
            try:
                fresh = dict(self.load())
            except (OSError, ValueError) as exc:
                LOGGER.warning(
                    "could not re-read the node action keys: %s", type(exc).__name__
                )
                return
            added = sorted(name for name in fresh if name not in self)
            for name in added:
                dict.__setitem__(self, name, fresh[name])
        if added:
            LOGGER.info(
                "node action keys for %d node(s) reached this executor: %s",
                len(added),
                ",".join(added),
            )


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
