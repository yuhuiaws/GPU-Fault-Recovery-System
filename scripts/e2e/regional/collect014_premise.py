"""COLLECT-014's fail-closed premise: no stored GPU inventory may vouch for the SXID.

The fail-closed direction posts a back-dated SXID 10003 and needs ingest to find
no inventory evidence for it. Ingest accepts a stored sample as evidence while
``-30s <= event - sample <= max age`` (``src/gpu_fault/app/ingest/faults.py``:
the snapshot's ``gpu_inventory_max_age``, the latest metrics'
``legacy_gpu_metrics_inventory_max_age``), so the premise holds when every
sample lies outside that window by a margin ingestion lag cannot eat.

The case establishes the premise instead of betting on it
(``establish_fail_closed_premise``): it re-reads the store, dating the event
afresh each time, until the premise holds, and a bound derived from the
product's max age ends the wait INCONCLUSIVE -- never with a post. It waits
rather than dating the event further back on purpose: on an idle cluster the
event's IDLE rests on the watcher heartbeat being within 600 s of the event
time (``WorkloadTopologyService``, symmetric window), and an older event would
add the workload-UNKNOWN reason to the BLOCKED plan the case judges.

Live 2026-11-03 (HyperPod, round 15): a latest-metric row the collector had
stopped refreshing -- the derived ``pcie_link_downtrain`` of one GPU, last
delivered before a GPU reset restarted the collector -- sat 148 s older than
the back-dated event, inside the 600 s window, and the single read refused the
case outright. Waited out, it leaves the window 660 s after the event passes it.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

#: Ingest's tolerance for a sample a little newer than the event.
INGEST_FUTURE_TOLERANCE = timedelta(seconds=30)
#: The margin on either side of the window that ingestion lag cannot eat.
FAIL_CLOSED_SAFETY_MARGIN = timedelta(seconds=60)
#: How often the fail-closed direction re-reads the store while it waits.
PREMISE_REREAD_INTERVAL = timedelta(seconds=30)

# Read-only store probe: both inventory sources with the product's own max
# ages. ``legacy_rows`` names each latest-metric row (metric@GPU) beside its
# time in ``legacy_observed_at``, so a refusal can say which row it quotes.
GPU_INVENTORY_SNAPSHOT = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, node_id = sys.argv[1:]
context = ApplicationContext.from_environment()
store = context.store
snapshot = store.get_gpu_inventory_snapshot(cluster_id, node_id)
legacy = [
    item
    for item in store.list_gpu_metrics_latest(cluster_id, node_id)
    if item.sample.gpu_uuid
]
print(json.dumps({
    "snapshot_max_age_seconds": context.gpu_inventory_max_age.total_seconds(),
    "legacy_max_age_seconds": (
        context.legacy_gpu_metrics_inventory_max_age.total_seconds()
    ),
    "present": snapshot is not None,
    "observed_at": snapshot.observed_at.isoformat() if snapshot else None,
    "source_boot_id": snapshot.source_boot_id if snapshot else None,
    "device_count": len(snapshot.devices) if snapshot else 0,
    "legacy_observed_at": [item.observed_at.isoformat() for item in legacy],
    "legacy_rows": [
        f"{item.sample.canonical_name}@{item.sample.gpu_uuid}" for item in legacy
    ],
}, sort_keys=True, default=str))
"""


def _max_age(inventory: dict[str, Any], key: str) -> timedelta | None:
    value = inventory.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return timedelta(seconds=float(value))


def fail_closed_timing_errors(
    inventory: dict[str, Any],
    *,
    event_time: datetime,
) -> list[str]:
    """Both dedicated and legacy inventory must be outside the event window.

    Ingest takes a sample as evidence while ``-30s <= event - sample <= max
    age`` (``app/ingest/faults.py``: the snapshot's ``gpu_inventory_max_age``,
    the latest metrics' ``legacy_gpu_metrics_inventory_max_age``). A sample is
    therefore safe on either side: newer than the event by the future
    tolerance plus a margin, or older than the max age plus that margin. The
    second side matters for latest-metric rows a node no longer refreshes --
    a derived ``fabric_manager_status_held_seconds`` after the fabric trained,
    an XID field host-mode dcgm-exporter leaves blank -- which used to refuse
    the case on a healthy node. Without the product's max age (an older probe)
    only the newer side counts.
    """

    if type(inventory.get("present")) is not bool:
        return ["dedicated GPU inventory presence is unknown; SXID is not posted"]
    legacy = inventory.get("legacy_observed_at")
    if not isinstance(legacy, list):
        return ["legacy GPU inventory was not checked; SXID is not posted"]
    samples = [
        *(
            [
                (
                    inventory.get("observed_at"),
                    _max_age(inventory, "snapshot_max_age_seconds"),
                )
            ]
            if inventory["present"]
            else []
        ),
        *((item, _max_age(inventory, "legacy_max_age_seconds")) for item in legacy),
    ]
    needed = INGEST_FUTURE_TOLERANCE + FAIL_CLOSED_SAFETY_MARGIN
    for sample, max_age in samples:
        try:
            observed_at = datetime.fromisoformat(str(sample).replace("Z", "+00:00"))
            if observed_at.tzinfo is None:
                raise ValueError("inventory timestamp has no timezone")
            margin = observed_at - event_time
        except (TypeError, ValueError):
            return ["stored GPU inventory time is unknown; SXID is not posted"]
        if margin >= needed:
            continue
        if max_age is not None and -margin >= max_age + FAIL_CLOSED_SAFETY_MARGIN:
            continue
        older = (
            f" or {(max_age + FAIL_CLOSED_SAFETY_MARGIN).total_seconds():.0f}s older"
            if max_age is not None
            else ""
        )
        return [
            f"stored GPU inventory ({observed_at.isoformat()}) is "
            f"{margin.total_seconds():.0f}s from the back-dated event; it must be "
            f"{needed.total_seconds():.0f}s newer{older} for the fail-closed "
            "premise to hold, so the SXID is not posted"
        ]
    return []


def premise_wait_bound(inventory: dict[str, Any]) -> timedelta | None:
    """How long one stored sample can keep the premise from holding.

    The event is dated a fixed age before every read, so a sample drifts
    across ingest's window as the clock moves: it refuses the post from the
    moment it is less than the future tolerance plus the margin newer than the
    event until it is the product's max age plus the margin older. That whole
    band -- 90 s + 600 s + 60 s = 750 s at the product's legacy max age, 330 s
    for the 180 s snapshot -- outlasts every sample the first read saw; only a
    sample written during the wait can outlast the bound, and then the wait
    ends INCONCLUSIVE. Without a usable max age there is no bound to derive.
    """

    ages = [
        age
        for age in (
            _max_age(inventory, "snapshot_max_age_seconds"),
            _max_age(inventory, "legacy_max_age_seconds"),
        )
        if age is not None
    ]
    if not ages:
        return None
    return (
        INGEST_FUTURE_TOLERANCE
        + FAIL_CLOSED_SAFETY_MARGIN
        + max(ages)
        + FAIL_CLOSED_SAFETY_MARGIN
    )


def quoted_rows(inventory: dict[str, Any], errors: list[str]) -> list[str]:
    """The latest-metric rows (metric@GPU) whose time a refusal quotes.

    Diagnostic only: ``fail_closed_timing_errors`` decides, this names what it
    quoted. Empty when the probe sent no ``legacy_rows`` or the refusal quotes
    no legacy row (the dedicated snapshot, an unknown presence).
    """

    times = inventory.get("legacy_observed_at")
    names = inventory.get("legacy_rows")
    if (
        not errors
        or not isinstance(times, list)
        or not isinstance(names, list)
        or len(times) != len(names)
    ):
        return []
    return sorted(
        {
            str(name)
            for observed_at, name in zip(times, names, strict=True)
            if f"({observed_at})" in errors[0]
        }
    )


class PremiseClock(Protocol):
    def now(self) -> datetime: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The wall clock; ``time.sleep`` is looked up when it is called."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class FailClosedPremise:
    """The read the fail-closed post goes out on, or why there is none."""

    posted_at: datetime
    event_time: datetime
    inventory: dict[str, Any]
    reads: list[dict[str, Any]]
    bound: timedelta | None
    refusal: str | None

    def evidence(self) -> dict[str, Any]:
        """The case-result fields: the last read and the whole wait."""

        first = datetime.fromisoformat(str(self.reads[0]["read_at"]))
        return {
            "inventory_snapshot": self.inventory,
            "fail_closed_premise": {
                "held": self.refusal is None,
                "refusal": self.refusal,
                "posted_at": self.posted_at.isoformat(),
                "event_time": self.event_time.isoformat(),
                "bound_seconds": (
                    None if self.bound is None else self.bound.total_seconds()
                ),
                "waited_seconds": (self.posted_at - first).total_seconds(),
                "reads": self.reads,
            },
        }


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _say(line: str) -> None:
    print(f"COLLECT-014 fail-closed premise: {line}", file=sys.stderr, flush=True)


def establish_fail_closed_premise(
    read_inventory: Callable[[], dict[str, Any]],
    *,
    event_age: timedelta,
    clock: PremiseClock | None = None,
    log: Callable[[str], None] | None = None,
    reread_interval: timedelta = PREMISE_REREAD_INTERVAL,
) -> FailClosedPremise:
    """Re-read the stored inventory until the fail-closed premise holds.

    Every read dates the event afresh, ``event_age`` before the read, and is
    judged by the unchanged ``fail_closed_timing_errors``; the read that holds
    is the re-check the post goes out on, with that read's event time. A
    refusal starts a wait bounded by ``premise_wait_bound`` of the first read,
    re-read every ``reread_interval`` with the last read on the bound. The
    result carries ``refusal`` -- the case's INCONCLUSIVE reason -- when the
    bound ends, or at once when the read gives no bound. Each read is logged
    (start, every re-read, the outcome) and kept in ``reads``.
    """

    clock = clock or SystemClock()
    say = log or _say
    reads: list[dict[str, Any]] = []
    started: datetime | None = None
    bound: timedelta | None = None
    while True:
        posted_at = clock.now()
        event_time = posted_at - event_age
        inventory = read_inventory()
        errors = fail_closed_timing_errors(inventory, event_time=event_time)
        rows = quoted_rows(inventory, errors)
        reads.append(
            {
                "read_at": posted_at.isoformat(),
                "event_time": event_time.isoformat(),
                "errors": errors,
                "rows": rows,
            }
        )
        if started is None:
            started, bound = posted_at, premise_wait_bound(inventory)
        waited = posted_at - started
        head = f"read {len(reads)} at {_stamp(posted_at)}, event {_stamp(event_time)}"
        if not errors:
            after = f"after {waited.total_seconds():.0f} s" if len(reads) > 1 else ""
            say(f"{head}: holds {after or 'at the first read'}; posting the SXID")
            return FailClosedPremise(
                posted_at, event_time, inventory, reads, bound, None
            )
        why = errors[0] + (f" (rows: {', '.join(rows)})" if rows else "")
        if bound is None or waited >= bound:
            refusal = (
                f"{why}; no wait bound: the read has no usable product max age"
                if bound is None
                else f"the fail-closed premise did not hold within "
                f"{bound.total_seconds():.0f} s ({len(reads)} reads): {why}"
            )
            say(f"{head}: INCONCLUSIVE: {refusal}")
            return FailClosedPremise(
                posted_at, event_time, inventory, reads, bound, refusal
            )
        if len(reads) == 1:
            say(
                f"{head}: does not hold: {why}; waiting, re-reading every "
                f"{reread_interval.total_seconds():.0f} s for at most "
                f"{bound.total_seconds():.0f} s (until {_stamp(posted_at + bound)})"
            )
        else:
            say(
                f"{head}: still does not hold after {waited.total_seconds():.0f} "
                f"of {bound.total_seconds():.0f} s: {why}"
            )
        clock.sleep(min(reread_interval, bound - waited).total_seconds())
