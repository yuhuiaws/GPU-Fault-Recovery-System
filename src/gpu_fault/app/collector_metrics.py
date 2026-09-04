from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from threading import Event, RLock
from typing import Any, Callable

from gpu_fault.collector_requirements import (
    agent_is_current,
    collector_silent_thresholds,
    required_collectors_for_agent,
)
from gpu_fault.telemetry import (
    CollectorMetricsSnapshotRecord,
    CollectorStatus,
    collector_producer,
)

LOGGER = logging.getLogger(__name__)


class CollectorMetricsSnapshot:
    def __init__(
        self,
        context,
        *,
        owner_id: str,
        enabled: bool,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.context = context
        self.owner_id = owner_id
        self.enabled = enabled
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.interval_seconds = float(
            os.getenv("GPU_FAULT_METRICS_COLLECTOR_SNAPSHOT_SECONDS", "30")
        )
        self.top_n = max(
            0,
            int(os.getenv("GPU_FAULT_METRICS_COLLECTOR_TOP_N", "20")),
        )
        self._lock = RLock()
        self._lines: list[str] = []
        self._details: list[dict] = []
        self._observed_at: datetime | None = None

    def run(self, stop: Event) -> None:
        if not self.enabled:
            return
        while not stop.is_set():
            try:
                self.refresh()
            except Exception:
                LOGGER.exception("collector metrics snapshot refresh failed")
            stop.wait(self.interval_seconds)

    def refresh(self) -> None:
        now = self.now()
        lease = self.context.store.acquire_periodic_task_lease(
            "collector-metrics-snapshot",
            self.owner_id,
            now=now,
            lease_duration=timedelta(seconds=max(60.0, self.interval_seconds * 2)),
        )
        if lease.owner_id == self.owner_id:
            rows = self._rows(now)
            record = CollectorMetricsSnapshotRecord(
                observed_at=now,
                lines=self._aggregate_lines(rows),
                details=self._top_rows(rows),
            )
            self.context.store.save_collector_metrics_snapshot(record)
        else:
            record = self.context.store.get_collector_metrics_snapshot()
            if record is None:
                return
        with self._lock:
            self._lines = list(record.lines)
            self._details = list(record.details)
            self._observed_at = record.observed_at

    def lines(self) -> list[str]:
        if not self.enabled:
            return []
        with self._lock:
            lines = list(self._lines)
            observed_at = self._observed_at
        age = (
            max(0, int((self.now() - observed_at).total_seconds()))
            if observed_at is not None
            else "+Inf"
        )
        return [
            *lines,
            "# TYPE gpu_fault_collector_metrics_snapshot_age_seconds gauge",
            f"gpu_fault_collector_metrics_snapshot_age_seconds {age}",
        ]

    def details(self) -> list[dict]:
        with self._lock:
            return list(self._details)

    def _rows(self, observed_at: datetime) -> list[dict]:
        ctx = self.context
        cluster_ids = (
            [
                item.cluster_id
                for item in ctx.store.list_regional_clusters()
                if item.enabled
            ]
            if ctx.regional_mode
            else sorted({item.cluster_id for item in ctx.store.list_agents()})
        )
        silent_after = collector_silent_thresholds()
        rows = []
        for cluster_id in cluster_ids:
            statuses = {
                (item.node_id, item.collector): item
                for item in ctx.store.list_collector_statuses(cluster_id)
            }
            for agent in ctx.store.list_agents(cluster_id):
                if not agent_is_current(agent, observed_at=observed_at):
                    continue
                for collector in required_collectors_for_agent(agent):
                    status = statuses.get((agent.node_id, collector))
                    last = status.last_success_at if status is not None else None
                    age = (
                        max(
                            0.0,
                            (observed_at - last).total_seconds(),
                        )
                        if last is not None
                        else float("inf")
                    )
                    rows.append(
                        {
                            "cluster_id": cluster_id,
                            "node_id": agent.node_id,
                            "collector": collector_producer(collector),
                            "channel": collector.value,
                            "last_success_age_seconds": age,
                            "silent": age > silent_after[collector],
                            "erroring": is_erroring(status),
                        }
                    )
        return rows

    def _aggregate_lines(self, rows: list[dict[str, Any]]) -> list[str]:
        return aggregate_lines(rows, top_n=self.top_n)

    def _top_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return top_rows(rows, top_n=self.top_n)


def is_erroring(status: CollectorStatus | None) -> bool:
    """Whether the newest thing known about this collector is a failure.

    Silence and failure are different states and only one of them was
    observable before: a collector that reports collection errors every cycle is
    still delivering batches, so nothing about it is late, and its errors were
    invisible until its last success aged past the silence threshold -- minutes
    for the metrics channel, but a quarter of an hour for node logs.

    ``last_error_at`` and ``last_success_at`` are both sticky (each ingest keeps
    the previous value it does not set), so the comparison is what makes this
    self-clearing: one successful batch moves ``last_success_at`` past the error
    and the node stops counting.
    """

    if status is None or status.last_error_at is None:
        return False
    if status.last_success_at is None:
        return True
    return status.last_error_at >= status.last_success_at


def aggregate_lines(rows: list[dict[str, Any]], *, top_n: int) -> list[str]:
    """Fold per-node collector rows into the published gauge lines.

    The aggregate families deliberately carry no ``node_id``: which node went
    silent is the ``..._top_node`` family's job, and folding the count into one
    series per cluster/collector/channel is what lets the alert stay off a label
    the aggregate does not publish.
    """

    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["cluster_id"],
            row["collector"],
            row["channel"],
        )
        item = aggregates.setdefault(key, {"silent": 0, "erroring": 0, "max_age": 0.0})
        item["silent"] += int(row["silent"])
        # Indexed rather than `.get`: a row builder that stops reporting this
        # would otherwise publish a permanent zero, which reads as "no node is
        # failing" and is the one wrong answer this gauge can give.
        item["erroring"] += int(row["erroring"])
        item["max_age"] = max(
            item["max_age"],
            row["last_success_age_seconds"],
        )
    lines = [
        "# TYPE gpu_fault_collector_silent_nodes gauge",
        "# TYPE gpu_fault_collector_erroring_nodes gauge",
        "# TYPE gpu_fault_collector_last_success_age_seconds_max gauge",
    ]
    for key, values in sorted(aggregates.items()):
        labels = _labels(*key)
        lines.append(f"gpu_fault_collector_silent_nodes{{{labels}}} {values['silent']}")
        lines.append(
            f"gpu_fault_collector_erroring_nodes{{{labels}}} {values['erroring']}"
        )
        lines.append(
            "gpu_fault_collector_last_success_age_seconds_max"
            f"{{{labels}}} {values['max_age']}"
        )
    for rank, row in enumerate(top_rows(rows, top_n=top_n), 1):
        labels = _labels(
            row["cluster_id"],
            row["collector"],
            row["channel"],
            node_id=row["node_id"],
            rank=rank,
        )
        lines.append(
            f"gpu_fault_collector_silent_top_node{{{labels}}} {int(row['silent'])}"
        )
    return lines


def top_rows(rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -row["last_success_age_seconds"],
            row["cluster_id"],
            row["node_id"],
            row["collector"],
            row["channel"],
        ),
    )[:top_n]


def _labels(
    cluster_id: str,
    collector: str,
    channel: str,
    *,
    node_id: str | None = None,
    rank: int | None = None,
) -> str:
    values = {
        "cluster_id": cluster_id,
        "collector": collector,
        "channel": channel,
    }
    if node_id is not None:
        values["node_id"] = node_id
    if rank is not None:
        values["rank"] = str(rank)
    return ",".join(
        f'{key}="{str(value).replace(chr(34), chr(92) + chr(34))}"'
        for key, value in values.items()
    )
