"""The fleet-wide ``workflow-dispatch`` lease as one dispatcher process holds it.

Every control-worker process runs a dispatcher, but only the holder of this
periodic-task lease scans (F-A1): every process used to run a full scan and
race the same rows, and ``claim_workflow`` still guarantees one executor per
workflow whenever two holders overlap.

The lease used to be taken once, at the start of a tick, and nothing renewed it
while the tick ran. A tick longer than the lease (15 s as shipped) -- a
holder starved of CPU or of the GIL while it works a full batch -- let the next
process's ordinary poll take the lapsed lease and run a second full sweep and
scan while the first was still working: rows the first had queued were claimed
by both, every claim the loser tried was rolled back as "leased by another
executor", and every WAITING row the old holder kept was pinned to it for its
row lease after the handover.

The holder now renews at the tick's checkpoints -- every row the sweeps and the
scan walk, the end of the scan, and every row of its batch that finishes --
once a third of the lease has passed since the last renewal, so a tick that
keeps making progress keeps the lease however long it runs. Only progress
renews: a holder stuck in one Store call, or waiting on rows that never
finish, stops renewing, the lease lapses and the next process takes over, as
before. A holder that finds another owner at a checkpoint stops: no further
sweep row, scan row or batch row is started, rows it queued but had not
started stay PENDING for the new holder, rows already running finish.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Lock, local
from typing import Any, Callable, NamedTuple

from gpu_fault.store.contracts import ControlPlaneStore

LOGGER = logging.getLogger(__name__)

# Upper bounds of the tick histogram: ordinary ticks (well under a second idle,
# a few seconds with a full batch), 7.5 s -- the half lease the derived batch is
# sized to (``config.dispatch_batch_size``) -- the shipped 15-s lease, and the
# starved ticks of a CPU-bound control worker.
DISPATCH_TICK_SECONDS_BUCKETS: tuple[float, ...] = (
    0.5,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    15.0,
    30.0,
    60.0,
    120.0,
)
# The holder renews once this share of the lease has passed since its last
# renewal, the cadence its between-ticks renewals already have (the lease is at
# least three polls): two thirds of the lease stay as the margin a slow
# checkpoint gap may spend.
RENEW_AFTER_SHARE = 1 / 3
# When a holder found the lease taken: at a checkpoint inside its tick, or at
# the start of a tick (it was away, or starved, for longer than the lease).
LOSS_MOMENTS: tuple[str, ...] = ("mid_tick", "tick_start")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DispatchLeaseLost(Exception):
    """A tick checkpoint found the dispatch lease held by another process."""


class TickDurations:
    """Cumulative histogram of this process's dispatch tick durations."""

    def __init__(
        self, buckets: tuple[float, ...] = DISPATCH_TICK_SECONDS_BUCKETS
    ) -> None:
        self.buckets = buckets
        self._counts = [0] * len(buckets)
        self._count = 0
        self._total_seconds = 0.0
        self._lock = Lock()

    def observe(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        with self._lock:
            self._count += 1
            self._total_seconds += seconds
            for index, bound in enumerate(self.buckets):
                if seconds <= bound:
                    self._counts[index] += 1
                    break

    def snapshot(self) -> tuple[list[tuple[float, int]], int, float]:
        """Cumulative ``(le, count)`` pairs (``+Inf`` is the count), count, sum."""

        with self._lock:
            rows: list[tuple[float, int]] = []
            cumulative = 0
            for bound, observed in zip(self.buckets, self._counts, strict=True):
                cumulative += observed
                rows.append((bound, cumulative))
            return rows, self._count, self._total_seconds


class DispatchLease:
    """This process's hold on the ``workflow-dispatch`` periodic-task lease.

    ``acquire`` takes or renews it at the start of a tick; inside the tick
    (``holding_tick``) ``checkpoint`` and ``renew_if_due`` renew it once a third
    of the lease has passed, and report a lease another process took. Outside
    a held tick on the calling thread both are no-ops, so a sweep called
    directly (an admin tool, a test) never checks a lease it did not take.
    ``clock`` is the wall clock the lease row is written in; tests substitute
    a fake one.
    """

    def __init__(
        self,
        task_key: str,
        lease_seconds: float,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.task_key = task_key
        self.lease_seconds = float(lease_seconds)
        self.clock = clock
        self._duration = timedelta(seconds=self.lease_seconds)
        self._renew_after = self._duration * RENEW_AFTER_SHARE
        # Times this process took the lease under a new epoch, and times it
        # found the lease it held taken by another process, by LOSS_MOMENTS.
        self.acquisitions_total = 0
        self.losses_total: dict[str, int] = dict.fromkeys(LOSS_MOMENTS, 0)
        self.tick_seconds = TickDurations()
        self._lock = Lock()
        self._local = local()
        self._store: ControlPlaneStore | None = None
        self._owner_id = ""
        self._epoch: int | None = None
        # The last renewal this process tried, whether or not it landed.
        self._attempted_at: datetime | None = None
        self._expires_at: datetime | None = None

    @property
    def enabled(self) -> bool:
        return self.lease_seconds > 0

    def acquire(self, store: ControlPlaneStore, owner_id: str) -> bool:
        """Take or renew the lease at the start of a tick; True when held."""

        self._store, self._owner_id = store, owner_id
        return self._take(self.clock(), when="tick_start")

    def held(self) -> bool:
        """Whether this process holds the lease now, by its own clock."""

        expires_at = self._expires_at
        return expires_at is not None and self.clock() < expires_at

    @contextmanager
    def holding_tick(self, started: datetime) -> Iterator[None]:
        """Run one tick on this thread as the holder; its duration is observed."""

        previous = getattr(self._local, "active", False)
        self._local.active = True
        try:
            yield
        finally:
            self._local.active = previous
            self.tick_seconds.observe((self.clock() - started).total_seconds())

    def checkpoint(self) -> None:
        """Renew when due; raise ``DispatchLeaseLost`` once the lease is lost.
        A Store error from the renewal propagates like the tick's own."""

        if not self.renew_if_due(tolerate_errors=False):
            raise DispatchLeaseLost(
                f"this process no longer holds the {self.task_key} lease"
            )

    def renew_if_due(self, *, tolerate_errors: bool = True) -> bool:
        """Renew when a third of the lease has passed; False once it is lost.

        Lost is another owner, or a lease this process could not renew that
        has run out by its own clock; until then no other process can have
        taken it, so a failed renewal (``tolerate_errors``) keeps the batch
        going and is retried a third of the lease later -- not at every row,
        which in a Store outage would ask once per finished row.
        """

        if not self.enabled or not getattr(self._local, "active", False):
            return True
        attempted_at, expires_at = self._attempted_at, self._expires_at
        if attempted_at is None or expires_at is None:
            return False
        now = self.clock()
        if now - attempted_at < self._renew_after:
            return now < expires_at
        self._attempted_at = now
        try:
            return self._take(now, when="mid_tick")
        except Exception:
            if not tolerate_errors:
                raise
            LOGGER.warning(
                "workflow dispatch lease renewal failed mid-tick; the batch goes on "
                "while the lease granted until %s lasts",
                expires_at.isoformat(),
                exc_info=True,
            )
            return now < expires_at

    def _take(self, now: datetime, *, when: str) -> bool:
        if self._store is None:
            raise RuntimeError("the dispatch lease is used before acquire()")
        lease = self._store.acquire_periodic_task_lease(
            self.task_key,
            self._owner_id,
            now=now,
            lease_duration=self._duration,
        )
        with self._lock:
            held_epoch = self._epoch
            if lease.owner_id == self._owner_id:
                if lease.epoch != held_epoch:
                    self.acquisitions_total += 1
                    # Every new holder bumps the epoch once: a gap of more than
                    # one is another process that held it in between.
                    if held_epoch is not None and lease.epoch > held_epoch + 1:
                        self.losses_total[when] += 1
                self._epoch = lease.epoch
                self._attempted_at = now
                self._expires_at = lease.lease_expires_at
                return True
            if held_epoch is not None:
                self.losses_total[when] += 1
            self._epoch = self._attempted_at = self._expires_at = None
        if held_epoch is not None:
            LOGGER.warning(
                "workflow dispatch lease lost (%s): this process held epoch %s; "
                "%s holds epoch %s until %s",
                when,
                held_epoch,
                lease.owner_id,
                lease.epoch,
                lease.lease_expires_at.isoformat(),
            )
        return False


class BatchDrain(NamedTuple):
    completed: list[Future[Any]]
    # Queued rows cancelled at the cycle deadline (F-C7).
    deferred: int
    # Queued rows cancelled because the lease was found taken.
    left_queued: int
    lease_lost: bool


def _cancel_queued(pending: set[Future[Any]]) -> int:
    cancelled = [future for future in pending if future.cancel()]
    pending.difference_update(cancelled)
    return len(cancelled)


def drain_batch(
    pending: set[Future[Any]], *, deadline: float | None, lease: DispatchLease
) -> BatchDrain:
    """Wait for a tick's batch; every finished row is a lease checkpoint.

    Rows still queued when the cycle deadline passes (``deadline``, monotonic)
    or when a renewal finds the lease taken are cancelled and stay PENDING for
    the next scan, whoever runs it; rows already running always finish, so a
    process never has one row in two of its workers.
    """

    completed: list[Future[Any]] = []
    deferred = left_queued = 0
    lease_lost = False
    while pending:
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        completed.extend(done)
        if deadline is not None and pending and time.monotonic() >= deadline:
            deferred += _cancel_queued(pending)
            deadline = None
        if done and pending and not lease_lost and not lease.renew_if_due():
            lease_lost = True
            left_queued = _cancel_queued(pending)
            LOGGER.warning(
                "workflow dispatch lease lost mid-batch: %d queued rows left "
                "PENDING for the new holder, %d running rows finish here",
                left_queued,
                len(pending),
            )
    return BatchDrain(completed, deferred, left_queued, lease_lost)
