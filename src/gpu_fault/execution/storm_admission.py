"""Which claims the remediation storm breaker judges, and with what (§9.5).

A breaker counts nodes that newly enter destructive remediation, so it judges
a workflow's first admission of budgeted work and nothing else: a hold
(budgeted work begun), a RUNNING row that already holds its claims, and a
preempting successor taking over the hold its predecessor reserved for it all
continue work in flight and are never held. The nodes are those the claim
brings into destructive NODE remediation (``storm_counted_node_ids`` -- a
node-hardware mutation or an isolation, never a workload stop/restart of a
job the node belongs to), the workload domain the incident's job or, without
one, the incident (``storm_domain``).

The cluster's GPU nodes are its fleet agents that are ACTIVE or DRAINING (one
per GPU node; a departed node's agent stays ACTIVE until the departed-node
cleanup retires it, so the count errs high and the threshold with it), read at
most once a minute per store and cluster -- a full agent list is not something
to read on every admission. A store that counts in place answers with the
number (``count_agents``: PostgreSQL, one count query, no Agent record
decoded); only any other store is listed and counted here. Listing a 256-node
cluster decodes 256 records, and a burst's first admissions of one cluster
start before any of them has kept its count, so on the 32 x 4 x 4 action model
with 256-node clusters (a local A/B on PostgreSQL) that decoding made the
first dispatch tick about 4 s longer, and every workflow's first command
waited for it (batch 246).

A failed read is never kept as the count (b227; it was kept as unknown for
the minute, so one transient store error made a 256-node cluster's breaker
judge at the floor of 3 for that minute). The last good count judges instead
while it is younger than ``NODE_COUNT_LAST_GOOD_SECONDS``. A count changes only
when a node joins, departs or is retired, and it already errs high by every
node that departed in the last ten minutes or more: a departed node's agent
counts until the departed-node cleanup retires it, never before the 10-minute
departure guard (``departed_nodes.DEPARTED_AGENT_GUARD_AGE``). A last good
count at most five minutes old (half the guard) misses at most the last five
minutes of retirements, an error of the same kind; a node that joined since
makes it err low, toward the floor. With no good count that young the floor
applies, as for a fleet never read. After a failed read the store is not read again for
``NODE_COUNT_RETRY_SECONDS`` -- the dispatcher's default tick -- so a failing
store is read about once a tick per cluster and process rather than on every
admission of a storm, and a store that answers again is counted within a tick,
not a minute. Every failed read is logged, its traceback once per streak, and
counted by what judged instead (``node_count_read_failures``; ``/metrics``
renders it).
"""

from __future__ import annotations

import logging
import math
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from gpu_fault.execution.remediation_budget import storm_counted_node_ids
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.models import (
    FaultIncident,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.remediation_storm import (
    STORM_MIN_NODES,
    StormCandidate,
    storm_domain,
    storm_scope,
)
from gpu_fault.store.shared.remediation_budgets import budget_wait_started_at

LOGGER = logging.getLogger(__name__)

NODE_COUNT_TTL_SECONDS = 60.0
#: How long a cluster's last good count still judges after failed reads: five
#: TTLs, half the departed-node guard (module docstring).
NODE_COUNT_LAST_GOOD_SECONDS = 5 * NODE_COUNT_TTL_SECONDS
#: How long the next read waits after a failed one: one default dispatcher
#: tick (``WorkflowDispatcherConfig.poll_interval_seconds``), well inside a TTL.
NODE_COUNT_RETRY_SECONDS = 5.0
#: What judged a claim whose count read failed (the ``fallback`` label).
FALLBACK_LAST_GOOD = "last_good"
FALLBACK_FLOOR = "floor"
_COUNTED = frozenset({AgentLifecycleState.ACTIVE, AgentLifecycleState.DRAINING})


@dataclass(frozen=True)
class _NodeCount:
    """One cluster's count as this process last read it (replaced, never
    changed, under ``_lock``)."""

    nodes: int | None = None  # the last good count; None before the first
    read_at: float = -math.inf  # when that read began (time.monotonic())
    retry_at: float = -math.inf  # after a failed read, no read before this
    failing: bool = False  # the last read failed (its traceback is logged)

    def younger_than(self, seconds: float, now: float) -> int | None:
        return self.nodes if now - self.read_at < seconds else None


_lock = Lock()
_counts: weakref.WeakKeyDictionary[Any, dict[str, _NodeCount]] = (
    weakref.WeakKeyDictionary()
)
_failed_reads = {FALLBACK_LAST_GOOD: 0, FALLBACK_FLOOR: 0}


def cluster_gpu_nodes(
    store: Any, cluster_id: str, *, now: float | None = None
) -> int | None:
    """The cluster's GPU nodes, kept for ``NODE_COUNT_TTL_SECONDS``; ``None``
    when they cannot be counted (the breaker's floor applies).

    A failed read answers the last good count while it is younger than
    ``NODE_COUNT_LAST_GOOD_SECONDS`` and is never kept itself; the store is
    not read again for ``NODE_COUNT_RETRY_SECONDS`` after it. ``now`` is a
    ``time.monotonic()`` reading (tests pass their own); ``_lock`` guards the
    kept counts, never the store read.
    """

    started = time.monotonic() if now is None else now
    with _lock:
        kept = _kept(store, cluster_id)
    if kept is not None:
        fresh = kept.younger_than(NODE_COUNT_TTL_SECONDS, started)
        if fresh is not None:
            return fresh
        if started < kept.retry_at:
            return kept.younger_than(NODE_COUNT_LAST_GOOD_SECONDS, started)
    try:
        nodes = _count_gpu_nodes(store, cluster_id)
    except Exception as exc:  # noqa: BLE001 - a failed count read has a fallback
        failed_at = time.monotonic() if now is None else now
        return _failed_read(store, cluster_id, exc, failed_at)
    _keep_good_read(store, cluster_id, nodes, started)
    return nodes


def _count_gpu_nodes(store: Any, cluster_id: str) -> int:
    """The cluster's ACTIVE and DRAINING fleet Agents, read from ``store``.

    A store that counts in place (``count_agents``: PostgreSQL, one count over
    the agent cluster index) is asked for the number. Any other is listed and
    counted here, which decodes every Agent record of the cluster: about 60 ms
    for 256 live-size records, paid by the dispatch threads of a burst's first
    admissions -- concurrently for one cluster, before any of them has kept a
    count -- and on the first-action path of every workflow of the burst
    (batch 246).
    """

    count_agents = getattr(store, "count_agents", None)
    if count_agents is not None:
        return int(count_agents(cluster_id, _COUNTED))
    return sum(
        1
        for agent in store.list_agents(cluster_id)
        if agent.lifecycle_state in _COUNTED
    )


def node_count_read_failures() -> dict[str, int]:
    """This process's failed count reads, by what judged instead."""

    with _lock:
        return dict(_failed_reads)


def _kept(store: Any, cluster_id: str) -> _NodeCount | None:
    try:
        counts = _counts.get(store)
    except TypeError:  # a store that cannot be weakly referenced keeps none
        return None
    return None if counts is None else counts.get(cluster_id)


def _counts_of(store: Any) -> dict[str, _NodeCount] | None:
    try:
        return _counts.setdefault(store, {})
    except TypeError:
        return None


def _keep_good_read(store: Any, cluster_id: str, nodes: int, read_at: float) -> None:
    """Keep the count, unless a read that began later is kept already."""

    with _lock:
        counts = _counts_of(store)
        if counts is None:
            return
        kept = counts.get(cluster_id)
        if kept is None or kept.read_at <= read_at:
            counts[cluster_id] = _NodeCount(nodes=nodes, read_at=read_at)


def _failed_read(store: Any, cluster_id: str, exc: Exception, now: float) -> int | None:
    """The last good count while younger than the bound, else ``None`` (the
    floor); the next read waits ``NODE_COUNT_RETRY_SECONDS``."""

    with _lock:
        counts = _counts_of(store)
        kept = _NodeCount() if counts is None else counts.get(cluster_id, _NodeCount())
        if counts is not None:
            counts[cluster_id] = replace(
                kept, retry_at=now + NODE_COUNT_RETRY_SECONDS, failing=True
            )
        nodes = kept.younger_than(NODE_COUNT_LAST_GOOD_SECONDS, now)
        _failed_reads[FALLBACK_FLOOR if nodes is None else FALLBACK_LAST_GOOD] += 1
    _log_failed_read(cluster_id, exc, nodes, now - kept.read_at, first=not kept.failing)
    return nodes


def _log_failed_read(
    cluster_id: str, exc: Exception, nodes: int | None, age: float, *, first: bool
) -> None:
    """One warning per failed read; the traceback once per streak."""

    if nodes is None:
        verdict = (
            f"no good count is younger than {NODE_COUNT_LAST_GOOD_SECONDS:.0f} s, "
            f"so the floor of {STORM_MIN_NODES} nodes applies"
        )
    else:
        verdict = (
            f"it judges with the last good count, {nodes} nodes from {age:.0f} s "
            f"ago (for up to {NODE_COUNT_LAST_GOOD_SECONDS:.0f} s)"
        )
    LOGGER.warning(
        "remediation storm breaker cannot count the GPU nodes of %s (%s); %s; "
        "next read in %.0f s",
        cluster_id,
        exc,
        verdict,
        NODE_COUNT_RETRY_SECONDS,
        exc_info=exc if first else None,
    )


def storm_candidate(
    store: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    claims: Mapping[str, int],
    steps: Sequence[WorkflowStepSpec],
    *,
    hold: bool,
) -> StormCandidate | None:
    """The breaker's view of this claim, or ``None`` when it does not judge it.

    The nodes it counts are those the claim brings into destructive NODE
    remediation (``storm_counted_node_ids``): a node whose only budgeted step is
    a workload stop/restart of a job it belongs to is not one of them, so a
    reboot or reset of one node of a multi-node job never counts the job's
    healthy peers toward the breaker (b223). A claim that mutates no node's
    hardware and isolates none is not judged at all.
    """

    if hold or not claims or workflow.preempt_predecessor:
        return None
    if workflow.status is WorkflowStatus.RUNNING and workflow.remediation_budget_claims:
        return None
    node_ids = storm_counted_node_ids(workflow, steps)
    if not node_ids:
        return None
    cluster_id = incident.cluster_id
    held_since = (
        budget_wait_started_at(workflow)
        if workflow.remediation_budget_last_blocked_scope == storm_scope(cluster_id)
        else None
    )
    return StormCandidate(
        cluster_id=cluster_id,
        workflow_id=workflow.request_id,
        node_ids=node_ids,
        domain=storm_domain(incident),
        held_since=held_since,
        cluster_nodes=cluster_gpu_nodes(store, cluster_id),
    )
