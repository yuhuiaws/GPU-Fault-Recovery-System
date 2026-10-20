"""Retention for health-signal state rows (batch 145).

``health_signal_state`` and ``efa_traffic_state`` rows were never removed: an
attempt that ever read low left up to sixteen LOW_CPU/LOW_GPU windows per node
(eight GPUs, two rules; batch 140 and 142), training-health signals stayed per
attempt and rank, EFA traffic states per attempt, and the rows of departed
nodes (spot replacements, re-images, node-group rolls) stayed for good -- in a
table the ingest path reads and writes under advisory locks.

One sweep, run by the periodic ``processor-cleanup`` job under its task lease,
removes a row only when all of this holds:

* No claim wrote it within the grace period (``clock_at``, the control-plane
  receive time, else ``observed_at``). An open window is written on every
  sample, so it is never a candidate, and neither is anything a replica is
  claiming right now (below).
* An attempt-scoped row (:func:`signal_key_attempt`: a LOW_* window, a
  training-health signal) belongs to an attempt that is no longer live: its
  observation is terminal, names another job, or is older than the grace while
  a watcher still vouches for the cluster. A live attempt's rows stay whatever
  their age; a cluster no watcher vouches for says nothing about its attempts,
  so their rows stay until the retention period passed without a write.
* Any other row belongs to a node the departed-node cleanup proved gone (a
  ledger line with a ``reference``; the row is not newer than the line's last
  confirmation) or was not written for the retention period. The unscoped
  gate of a live node is written by every sample and stays.
* A per-attempt EFA row of a release before batch 145 is superseded by the
  node's row (``efa_traffic_state.legacy_efa_traffic_state_key``): it goes once
  the grace period passed without a write.

The scan is a keyset walk over the ``(kind, key)`` primary key, one bounded page
per statement and a bounded number of pages per call, resumed from a cursor kept
on the store; no schema change, no index. Each deletion is conditional on the
row being exactly what was judged, and on PostgreSQL also on the row's advisory
lock being free at that moment (``pg_try_advisory_xact_lock``, never a wait):
the lock ``claim_health_signal_transitions``, ``mark_health_signal_notified``
and ``observe_efa_traffic`` hold for their whole read-decide-write. So a claim
either finished first (the row changed: kept) or starts after the delete and
sees no row (its sample opens a fresh window); it can neither resurrect the
deleted state nor lose a window it is writing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.attempt_observation_state import TERMINAL_WORKLOAD_PHASES
from gpu_fault.efa_traffic_state import efa_traffic_state_key
from gpu_fault.models import HealthSignalState
from gpu_fault.store.shared.cleanup_log import log_cleanup
from gpu_fault.store.shared.health_signals import previous_clock, signal_key_attempt

LOGGER = logging.getLogger("gpu_fault.store.cleanup")

#: What :meth:`RetentionEvidence.attempt_status` says about an attempt.
LIVE, ENDED, UNKNOWN = "live", "ended", "unknown"
#: The kinds the sweep walks, in this order.
SIGNAL_RETENTION_KINDS = ("health_signal_state", "efa_traffic_state")
#: Pages of at most ``limit`` keys one call scans per kind before it hands the
#: rest of the key space to the next call: ten statements of a primary-key
#: range scan bound one call, the cursor makes the next one continue.
SIGNAL_RETENTION_PAGES_PER_CALL = 10


@dataclass(frozen=True)
class SignalRow:
    """One row old enough to judge, as read."""

    key: str
    state: Any
    #: The stored row as read; a deletion applies only while it is unchanged.
    token: object


@dataclass(frozen=True)
class SignalPage:
    """One page of the key walk: its rows older than the bound, in key order."""

    rows: tuple[SignalRow, ...]
    #: Keys the page read; fewer than asked means the key space ended.
    scanned: int
    last_key: str | None


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def signal_state_clock(state: Any) -> datetime:
    """When a claim last wrote the row (UTC): a health signal's control-plane
    clock when it has one, else the sample time; an EFA state's sample time."""

    if isinstance(state, HealthSignalState):
        return _utc(previous_clock(state))
    return _utc(state.observed_at)


class RetentionEvidence:
    """What the store says about attempts and nodes, read once per cluster.

    Lives for one sweep call. A store without one of the readers (a narrow
    double) gives the answer that keeps rows.
    """

    def __init__(self, store: Any, *, now: datetime, grace: timedelta) -> None:
        self._store = store
        self._fresh_after = now - grace
        self._attempts: dict[str, dict[str, Any] | None] = {}
        self._watched: dict[str, bool] = {}
        self._departed: dict[str, dict[str, datetime]] = {}

    def attempt_status(
        self, cluster_id: str, job_id: str | None, attempt_id: str
    ) -> str:
        """``ended``, ``live`` or ``unknown`` for one attempt.

        Ended: its observation is terminal, or now names another job (a
        metadata takeover: nothing resolves the old pair again), or it has not
        been observed within the grace while a watcher vouches for the
        cluster. Live: a fresh observation in a non-terminal phase. Unknown:
        no watcher vouches for the cluster, so silence proves nothing.
        """

        observations = self._observations(cluster_id)
        if observations is None:
            return UNKNOWN
        observation = observations.get(attempt_id)
        if observation is not None:
            if job_id is not None and observation.job_id != job_id:
                return ENDED
            if observation.workload_phase in TERMINAL_WORKLOAD_PHASES:
                return ENDED
            if _utc(observation.observed_at) >= self._fresh_after:
                return LIVE
        return ENDED if self._watched_cluster(cluster_id) else UNKNOWN

    def departed_at(self, cluster_id: str, node_id: str) -> datetime | None:
        """The last confirmation of a proven departure of the node, if any."""

        departed = self._departed.get(cluster_id)
        if departed is None:
            lister = getattr(self._store, "list_departed_node_cleanups", None)
            records = lister(cluster_id) if lister is not None else []
            departed = {
                record.node_id: _utc(record.updated_at)
                for record in records
                if record.reference is not None
            }
            self._departed[cluster_id] = departed
        return departed.get(node_id)

    def _observations(self, cluster_id: str) -> dict[str, Any] | None:
        if cluster_id not in self._attempts:
            lister = getattr(self._store, "list_attempt_observations", None)
            self._attempts[cluster_id] = (
                None
                if lister is None
                else {item.attempt_id: item for item in lister(cluster_id)}
            )
        return self._attempts[cluster_id]

    def _watched_cluster(self, cluster_id: str) -> bool:
        if cluster_id not in self._watched:
            reader = getattr(self._store, "get_workload_coverage_heartbeat", None)
            heartbeat = reader(cluster_id) if reader is not None else None
            self._watched[cluster_id] = (
                heartbeat is not None
                and _utc(heartbeat.observed_at) >= self._fresh_after
            )
        return self._watched[cluster_id]


def retention_reason(
    kind: str,
    row: SignalRow,
    evidence: RetentionEvidence,
    *,
    now: datetime,
    grace: timedelta,
    retention: timedelta,
) -> str | None:
    """Why ``row`` may go (``attempt_ended``, ``node_departed``,
    ``not_updated``, ``superseded``), or None to keep it."""

    clock = signal_state_clock(row.state)
    if clock >= now - grace:
        return None
    if kind == "efa_traffic_state":
        state = row.state
        if row.key != efa_traffic_state_key(state.cluster_id, state.node_id):
            return "superseded"
        cluster_id, node_id = state.cluster_id, state.node_id
    else:
        cluster_id, _, rest = row.key.partition("/")
        attempt = signal_key_attempt(row.key)
        if attempt is not None:
            status = evidence.attempt_status(cluster_id, *attempt)
            if status == ENDED:
                return "attempt_ended"
            if status == LIVE or clock > now - retention:
                return None
            return "not_updated"
        node_id = rest.partition("/")[0]
    departed_at = evidence.departed_at(cluster_id, node_id) if node_id else None
    if departed_at is not None and clock <= departed_at:
        return "node_departed"
    if clock <= now - retention:
        return "not_updated"
    return None


class SharedSignalRetentionMixin:
    """The sweep; each store supplies the page read and the guarded delete."""

    # Attributes supplied by the composed concrete implementation.
    _signal_state_page: Callable[..., SignalPage]
    _delete_unchanged_signal_states: Callable[[str, Sequence[SignalRow]], list[str]]

    #: Where the next call resumes, per kind. Process-local: the task lease
    #: moving or a restart starts the walk again at the beginning.
    _signal_retention_cursors: dict[str, str]

    def cleanup_health_signal_states(
        self,
        *,
        now: datetime,
        grace: timedelta,
        retention: timedelta,
        limit: int,
    ) -> dict[str, int]:
        """Remove at most ``limit`` rows per kind; rows removed, by kind."""

        if limit < 1:
            raise ValueError("health-signal retention limit must be positive")
        if not timedelta(0) < grace <= retention:
            raise ValueError(
                "health-signal retention needs 0 < grace <= retention, got "
                f"grace={grace} retention={retention}"
            )
        now = _utc(now)
        evidence = RetentionEvidence(self, now=now, grace=grace)
        return {
            kind: self._sweep_signal_kind(
                kind,
                evidence,
                now=now,
                grace=grace,
                retention=retention,
                limit=limit,
            )
            for kind in SIGNAL_RETENTION_KINDS
        }

    def _sweep_signal_kind(
        self,
        kind: str,
        evidence: RetentionEvidence,
        *,
        now: datetime,
        grace: timedelta,
        retention: timedelta,
        limit: int,
    ) -> int:
        cursors = getattr(self, "_signal_retention_cursors", None)
        if cursors is None:
            cursors = {}
            self._signal_retention_cursors = cursors
        after = cursors.get(kind, "")
        removed: dict[str, str] = {}
        attempted = 0
        for _ in range(SIGNAL_RETENTION_PAGES_PER_CALL):
            page = self._signal_state_page(
                kind, after_key=after, limit=limit, stale_before=now - grace
            )
            victims: list[SignalRow] = []
            reasons: dict[str, str] = {}
            resume = page.last_key
            for row in page.rows:
                reason = retention_reason(
                    kind, row, evidence, now=now, grace=grace, retention=retention
                )
                if reason is None:
                    continue
                if attempted + len(victims) >= limit:
                    # Full: the next call resumes after the last row taken.
                    resume = victims[-1].key
                    break
                victims.append(row)
                reasons[row.key] = reason
            attempted += len(victims)
            for key in self._delete_unchanged_signal_states(kind, victims):
                removed[key] = reasons[key]
            if page.scanned < limit and resume == page.last_key:
                after = ""
                break
            after = resume or after
            if attempted >= limit:
                break
        cursors[kind] = after
        if removed:
            counts: dict[str, int] = {}
            for reason in removed.values():
                counts[reason] = counts.get(reason, 0) + 1
            LOGGER.info(
                "health-signal retention removed %d %s rows (%s)",
                len(removed),
                kind,
                ", ".join(f"{name}={count}" for name, count in sorted(counts.items())),
            )
        return log_cleanup(kind, sorted(removed))
