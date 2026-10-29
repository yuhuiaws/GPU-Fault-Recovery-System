"""Which claims the remediation storm breaker judges, and with what (§9.5).

A breaker counts nodes that newly enter destructive remediation, so it judges
a workflow's first admission of budgeted work and nothing else: a hold
(budgeted work begun), a RUNNING row that already holds its claims, and a
preempting successor taking over the hold its predecessor reserved for it all
continue work in flight and are never held. The nodes are those the claim
takes (``node:<cluster>:<node>``), the workload domain the incident's job or,
without one, the incident (``storm_domain``).

The cluster's GPU nodes are its fleet agents that are ACTIVE or DRAINING (one
per GPU node; a departed node's agent stays ACTIVE until the departed-node
cleanup retires it, so the count errs high and the threshold with it), read at
most once a minute per store and cluster -- a full agent list is not something
to read on every admission.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections.abc import Mapping
from threading import Lock
from typing import Any

from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.models import FaultIncident, WorkflowRequest, WorkflowStatus
from gpu_fault.remediation_storm import StormCandidate, storm_domain, storm_scope
from gpu_fault.store.shared.remediation_budgets import budget_wait_started_at

LOGGER = logging.getLogger(__name__)

NODE_COUNT_TTL_SECONDS = 60.0
_COUNTED = frozenset({AgentLifecycleState.ACTIVE, AgentLifecycleState.DRAINING})
_lock = Lock()
_counts: weakref.WeakKeyDictionary[Any, dict[str, tuple[float, int | None]]] = (
    weakref.WeakKeyDictionary()
)


def cluster_gpu_nodes(store: Any, cluster_id: str) -> int | None:
    """The cluster's GPU nodes, cached for ``NODE_COUNT_TTL_SECONDS``;
    ``None`` when the fleet cannot be read (the breaker's floor applies)."""

    now = time.monotonic()
    with _lock:
        cached = _counts.get(store, {}).get(cluster_id)
    if cached is not None and now - cached[0] < NODE_COUNT_TTL_SECONDS:
        return cached[1]
    try:
        count: int | None = sum(
            1
            for agent in store.list_agents(cluster_id)
            if agent.lifecycle_state in _COUNTED
        )
    except Exception:  # noqa: BLE001 - an unknown count is the floor, not a failure
        LOGGER.warning(
            "remediation storm breaker cannot count the GPU nodes of %s; the "
            "floor applies",
            cluster_id,
            exc_info=True,
        )
        count = None
    try:
        with _lock:
            _counts.setdefault(store, {})[cluster_id] = (now, count)
    except TypeError:  # a store that cannot be weakly referenced is not cached
        pass
    return count


def storm_candidate(
    store: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    claims: Mapping[str, int],
    *,
    hold: bool,
) -> StormCandidate | None:
    """The breaker's view of this claim, or ``None`` when it does not judge it."""

    if hold or not claims or workflow.preempt_predecessor:
        return None
    if workflow.status is WorkflowStatus.RUNNING and workflow.remediation_budget_claims:
        return None
    cluster_id = incident.cluster_id
    prefix = f"node:{cluster_id}:"
    held_since = (
        budget_wait_started_at(workflow)
        if workflow.remediation_budget_last_blocked_scope == storm_scope(cluster_id)
        else None
    )
    return StormCandidate(
        cluster_id=cluster_id,
        workflow_id=workflow.request_id,
        node_ids=frozenset(
            scope[len(prefix) :] for scope in claims if scope.startswith(prefix)
        ),
        domain=storm_domain(incident),
        held_since=held_since,
        cluster_nodes=cluster_gpu_nodes(store, cluster_id),
    )
