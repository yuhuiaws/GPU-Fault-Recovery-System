"""The remediation storm circuit breaker of one GPU cluster (design §9.5).

No fixed number caps how many remediations a cluster runs at once. A training
job, or a GPU node without one, is one workflow; workflows are kept apart per
node (and per job for the steps that stop or restart it); what bounds a
correlated fault is this breaker. It watches the nodes that newly enter
destructive remediation in a cluster -- a workflow's first admission of
budgeted work -- over a sliding ``STORM_WINDOW_SECONDS``, and trips when they
number at least ``storm_threshold`` -- max(``STORM_MIN_NODES``,
ceil(``STORM_NODE_FRACTION`` x the cluster's GPU nodes)), the floor when the
count is unknown -- and span at least ``STORM_MIN_DOMAINS`` workload domains.
The nodes of one job, or of one incident without a job, are one domain
(``storm_domain``): one big job failing on many nodes is one fault and never
trips it; many unrelated nodes failing together are what a bad driver, a bad
image or a bad policy look like.

Tripped, the cluster's new destructive work is held, not failed: a first
admission is refused with the scope ``storm:<cluster>`` and waits PENDING like
any budget waiter. Work already admitted is never held -- holds, escalation
rungs and preempting successors go on -- and nothing is released without an
operator: ``release`` closes the breaker and re-arms it (the window starts
empty, and a row held through the trip is admitted without counting again),
``abort`` keeps it open and hands the held workflows to the operator. The
three constants are product constants, not configuration.

The record is the store's ``remediation_storm_breaker`` kind, keyed by cluster
id, read and written in the claim's own transaction under the cluster's
``remediation_storm/<cluster>`` lock, so replicas and restarts see one
breaker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import Field

from gpu_fault.models import FaultIncident, StrictModel

STORM_WINDOW_SECONDS = 600
STORM_NODE_FRACTION = 0.2
STORM_MIN_NODES = 3
STORM_MIN_DOMAINS = 2
STORM_SCOPE_PREFIX = "storm:"
STORM_RECORD_KIND = "remediation_storm_breaker"


class StormBreakerState(StrEnum):
    CLOSED = "CLOSED"
    TRIPPED = "TRIPPED"
    ABORTED = "ABORTED"


class StormEntry(StrictModel):
    """One node that entered destructive remediation inside the window."""

    node_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    admitted_at: datetime


class RemediationStormBreaker(StrictModel):
    cluster_id: str = Field(min_length=1)
    state: StormBreakerState = StormBreakerState.CLOSED
    window: list[StormEntry] = Field(default_factory=list)
    # The threshold the last admission was judged against (for /metrics).
    threshold: int = Field(default=0, ge=0)
    trip_id: str | None = None
    tripped_at: datetime | None = None
    trip_node_ids: list[str] = Field(default_factory=list)
    trip_domains: int = Field(default=0, ge=0)
    trip_threshold: int = Field(default=0, ge=0)
    trip_cluster_nodes: int | None = Field(default=None, ge=0)
    trip_workflow_id: str | None = None
    # The trip whose operator notification is saved (one per trip).
    notified_trip_id: str | None = None
    released_at: datetime | None = None
    released_by: str | None = None
    release_reason: str | None = None
    release_reference: str | None = None
    aborted_at: datetime | None = None
    aborted_by: str | None = None
    abort_reason: str | None = None
    abort_reference: str | None = None
    updated_at: datetime

    @property
    def open(self) -> bool:
        """Whether new destructive work in the cluster is held."""

        return self.state is not StormBreakerState.CLOSED


@dataclass(frozen=True)
class StormCandidate:
    """A first admission of budgeted work, as the breaker judges it.

    ``held_since`` is when the row's current wait on this cluster's breaker
    began (``None`` when its last refusal was not the breaker's): a row held
    through a trip the operator then released is admitted without counting.
    """

    cluster_id: str
    workflow_id: str
    node_ids: frozenset[str]
    domain: str
    held_since: datetime | None = None
    # The cluster's GPU nodes as last counted (``None``: unknown, the floor).
    cluster_nodes: int | None = None


def storm_scope(cluster_id: str) -> str:
    return f"{STORM_SCOPE_PREFIX}{cluster_id}"


def storm_domain(incident: FaultIncident) -> str:
    """The workload domain the breaker counts: the job, else the incident."""

    if incident.job_id:
        return f"job:{incident.job_id}"
    return f"incident:{incident.incident_id}"


def storm_threshold(cluster_nodes: int | None) -> int:
    """max(3, ceil(20% of the cluster's GPU nodes)); 3 when the count is unknown."""

    if not cluster_nodes or cluster_nodes < 1:
        return STORM_MIN_NODES
    return max(STORM_MIN_NODES, math.ceil(STORM_NODE_FRACTION * cluster_nodes))


def closed_breaker(cluster_id: str, now: datetime) -> RemediationStormBreaker:
    return RemediationStormBreaker(cluster_id=cluster_id, updated_at=now)


def held_reason(record: RemediationStormBreaker) -> str:
    """The refusal a held admission records (``remediation_budget_last_blocked_reason``)."""

    since = record.tripped_at.isoformat() if record.tripped_at else "unknown"
    verdict = (
        "aborted by an operator"
        if record.state is StormBreakerState.ABORTED
        else "tripped"
    )
    return (
        f"remediation storm breaker of cluster {record.cluster_id} is {verdict} "
        f"(since {since}: {len(record.trip_node_ids)} nodes entered destructive "
        f"remediation within {STORM_WINDOW_SECONDS}s across {record.trip_domains} "
        f"workload domains, threshold {record.trip_threshold}): new destructive "
        f"work is held; scope={storm_scope(record.cluster_id)}; resume with "
        "gpu-fault-admin remediation-storm release"
    )


def _current(window: list[StormEntry], now: datetime) -> list[StormEntry]:
    horizon = now - timedelta(seconds=STORM_WINDOW_SECONDS)
    return [entry for entry in window if entry.admitted_at > horizon]


def storm_admission(
    record: RemediationStormBreaker | None,
    candidate: StormCandidate,
    *,
    now: datetime,
    cluster_nodes: int | None,
) -> tuple[RemediationStormBreaker, str | None]:
    """The record after judging ``candidate``, and the refusal (``None``: admitted).

    Open, the breaker holds every candidate. Closed, a candidate that brings
    no node the window does not already have is admitted (its nodes are
    re-stamped); one that does trips the breaker -- and is held -- when the
    window's nodes with its own reach the threshold across two domains.
    """

    current = record or closed_breaker(candidate.cluster_id, now)
    if current.open:
        return current, held_reason(current)
    window = _current(current.window, now)
    if (
        candidate.held_since is not None
        and current.released_at is not None
        and candidate.held_since < current.released_at
    ):
        released = {"window": window, "updated_at": now}
        return current.model_copy(update=released), None
    known = {entry.node_id for entry in window}
    nodes = known | candidate.node_ids
    domains = {entry.domain for entry in window} | {candidate.domain}
    threshold = storm_threshold(cluster_nodes)
    current = current.model_copy(update={"threshold": threshold})
    if (
        candidate.node_ids - known
        and len(nodes) >= threshold
        and len(domains) >= STORM_MIN_DOMAINS
    ):
        tripped = current.model_copy(
            update={
                "state": StormBreakerState.TRIPPED,
                "window": window,
                "trip_id": f"{candidate.cluster_id}@{now.isoformat()}",
                "tripped_at": now,
                "trip_node_ids": sorted(nodes),
                "trip_domains": len(domains),
                "trip_threshold": threshold,
                "trip_cluster_nodes": cluster_nodes,
                "trip_workflow_id": candidate.workflow_id,
                "updated_at": now,
            }
        )
        return tripped, held_reason(tripped)
    kept = [entry for entry in window if entry.node_id not in candidate.node_ids]
    entries = [
        StormEntry(
            node_id=node_id,
            domain=candidate.domain,
            workflow_id=candidate.workflow_id,
            admitted_at=now,
        )
        for node_id in sorted(candidate.node_ids)
    ]
    return current.model_copy(
        update={"window": kept + entries, "updated_at": now}
    ), None


def released_breaker(
    record: RemediationStormBreaker,
    *,
    operator: str,
    reason: str,
    reference: str,
    now: datetime,
) -> RemediationStormBreaker:
    """Closed and re-armed: the window starts empty; idempotent when closed."""

    if not record.open:
        return record
    return record.model_copy(
        update={
            "state": StormBreakerState.CLOSED,
            "window": [],
            "released_at": now,
            "released_by": operator,
            "release_reason": reason,
            "release_reference": reference,
            "updated_at": now,
        }
    )


def aborted_breaker(
    record: RemediationStormBreaker,
    *,
    operator: str,
    reason: str,
    reference: str,
    now: datetime,
) -> RemediationStormBreaker:
    """Kept open for the operator; refused on a closed breaker, idempotent."""

    if not record.open:
        raise ValueError(
            f"remediation storm breaker of cluster {record.cluster_id} is not "
            "tripped: nothing is held to abort"
        )
    if record.state is StormBreakerState.ABORTED:
        return record
    return record.model_copy(
        update={
            "state": StormBreakerState.ABORTED,
            "aborted_at": now,
            "aborted_by": operator,
            "abort_reason": reason,
            "abort_reference": reference,
            "updated_at": now,
        }
    )
