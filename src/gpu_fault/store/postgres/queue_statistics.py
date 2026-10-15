"""The processor queue's planner statistics, refreshed when a claim proves them stale.

With the telemetry spool on, the queue is empty between bursts, and the
autoanalyze that follows a drained burst leaves ``pg_class.reltuples`` at 0
while the heap keeps its pages: the planner then believes the queue is empty,
whatever it holds. A burst's claims are planned on that belief until the next
autoanalyze reaches the queue, and on it every claim reads the whole backlog,
however its window is spelled -- 29,000 shared blocks a fault-stream claim on
a 3,200-row burst queue (processor_claims.py ``_CLAIM_ELIGIBLE_SQL`` keeps the
estimate up only while the planner believes the table holds rows). In a local
reproduction of the integrated 50-cluster fault phase whose statistics stayed
drained for the burst, 20 coordinators then spent 83 % of the burst's first
15 s inside claim statements, fault claims fell from 155 to 13-44 per half
second, taken by 7-14 of the 20 processes, and the synchronous action faults
completed at a p99 of 16.1 s -- the signature of the live stall.

A claim whose window came back full has just proved that the queue holds at
least a window of eligible rows. When the planner still counts fewer than two
windows, the store analyses the queue itself instead of waiting for
autovacuum: same reproduction, claim time 275 -> 46-56 s, action-fault p99
16.1 -> 3.4-4.3 s, no stall. Only the queue: lanes analysed early in a burst, when
nearly every lane row is one the burst has just leased, price the window's
lane anti-join as if nothing were eligible (the reproduction's p99 was
4.5-6.3 s with the lanes analysed alongside); their statistics
stay autovacuum's, taken over a table that keeps the earlier bursts' expired
lanes for an hour. The refresh is paced so it cannot become a load of its
own: the store reads ``pg_class`` at most once per ``CHECK_INTERVAL_SECONDS``
after a full window, analyses at most once per ``REFRESH_INTERVAL_SECONDS``,
one process of the site at a time (a transaction-scoped try-lock of its own
class), and gives the table up after ``ANALYZE_LOCK_TIMEOUT`` when autovacuum
holds it (that worker is analysing it anyway). ANALYZE takes a
ShareUpdateExclusive lock, which claims, admissions and completions never
wait for. A role that may not analyse the table gets PostgreSQL's warning and
the next autoanalyze, as before.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

LOGGER = logging.getLogger(__name__)

QUEUE_TABLE = "gpu_fault_processor_queue"
#: Advisory lock class of the refresh ("stat"), a key space of its own.
STATISTICS_LOCK_CLASS = 0x73746174
CHECK_INTERVAL_SECONDS = 1.0
REFRESH_INTERVAL_SECONDS = 10.0
ANALYZE_LOCK_TIMEOUT = "100ms"
_LOCK_NOT_AVAILABLE = "55P03"


@dataclass
class QueueStatisticsGuard:
    """Per-store pacing of the ``pg_class`` read and of the ANALYZE."""

    checked_at: float = float("-inf")
    refreshed_at: float = float("-inf")
    refreshes_total: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def check_due(self, now: float) -> bool:
        with self._lock:
            if (
                now - self.checked_at < CHECK_INTERVAL_SECONDS
                or now - self.refreshed_at < REFRESH_INTERVAL_SECONDS
            ):
                return False
            self.checked_at = now
            return True

    def refreshed(self, now: float) -> None:
        with self._lock:
            self.refreshed_at = now
            self.refreshes_total += 1


def refresh_if_stale(
    db: Any, guard: QueueStatisticsGuard, *, window_rows: int, now: float | None = None
) -> bool:
    """ANALYZE the queue when the planner counts fewer than two of the
    ``window_rows``-row windows a claim just found full; True when it did.

    Runs in its own transaction, after the claim's has committed. Never
    raises: a refresh that fails leaves the statistics to autovacuum.
    """

    monotonic = time.monotonic() if now is None else now
    if not guard.check_due(monotonic):
        return False
    try:
        with db.transaction():
            with db.cursor() as cursor:
                cursor.execute(
                    "SELECT reltuples FROM pg_class WHERE oid = to_regclass(%s)",
                    (QUEUE_TABLE,),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return False
                counted = float(row[0])
                if counted >= 2 * window_rows:
                    return False
                cursor.execute(
                    "SELECT pg_try_advisory_xact_lock(%s, 0)",
                    (STATISTICS_LOCK_CLASS,),
                )
                if not cursor.fetchone()[0]:
                    return False
                cursor.execute(f"SET LOCAL lock_timeout = '{ANALYZE_LOCK_TIMEOUT}'", ())
                cursor.execute(f"ANALYZE {QUEUE_TABLE}", ())
    except Exception as exc:
        if getattr(exc, "sqlstate", None) == _LOCK_NOT_AVAILABLE:
            LOGGER.info("processor queue statistics refresh skipped: table busy")
        else:
            LOGGER.warning("processor queue statistics refresh failed", exc_info=True)
        return False
    guard.refreshed(monotonic)
    LOGGER.info(
        "processor queue statistics refreshed: the planner counted %d rows "
        "where a claim found a full %d-row window",
        int(counted),
        window_rows,
    )
    return True
