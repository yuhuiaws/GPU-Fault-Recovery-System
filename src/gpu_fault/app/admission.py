from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Callable

from gpu_fault.store.shared.errors import operation_should_retry
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _AdmissionEntry:
    """One telemetry request waiting for a batched enqueue.

    ``deadline`` is the submitter's own REQUEST_DEADLINE, carried
    explicitly because the flush loop runs in a different task - and
    therefore a different context - than the requests it serves.
    """

    item: object
    future: asyncio.Future
    deadline: float | None
    enqueued_at: float


def _admission_scope(item) -> str:
    """The counter row a batch for this item will contend for.

    One transaction per cluster is what the store does, so the cluster is
    also the unit of ordering and of head-of-line blocking.
    """

    return getattr(item, "cluster_id", None) or "__unscoped__"


class _StripedAdmissionScope:
    """Assign consecutive full batches to work-conserving stripes."""

    def __init__(self, *, batch_size: int, partitions: int) -> None:
        if batch_size <= 0 or partitions <= 0:
            raise ValueError(
                "striped admission batch size and partitions must be positive"
            )
        self.batch_size = batch_size
        self.partitions = partitions
        self._sequence = 0

    def __call__(self, _item) -> str:
        stripe = (self._sequence // self.batch_size) % self.partitions
        self._sequence += 1
        return f"telemetry-{stripe}"


class _ProcessorAdmissionBatcher:
    def __init__(
        self,
        store,
        store_io: AsyncStoreExecutor,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int,
        reserved_cluster_fault_depth: int,
        global_admission_guard: int,
        max_batch_size: int = 64,
        max_flush_groups: int = 16,
        flush_delay_seconds: float = 0.002,
        projection_margin: float = 1.0,
        admit_batch=None,
        label: str = "processor admission",
        scope_key: Callable[[object], str] = _admission_scope,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("processor admission batch size must be positive")
        if max_flush_groups <= 0:
            raise ValueError("processor admission flush groups must be positive")
        if flush_delay_seconds < 0:
            raise ValueError("processor admission batch delay cannot be negative")
        if projection_margin < 0:
            raise ValueError("processor admission projection margin cannot be negative")
        self.store = store
        self.store_io = store_io
        self.max_depth = max_depth
        self.max_cluster_depth = max_cluster_depth
        self.reserved_fault_depth = reserved_fault_depth
        self.reserved_cluster_fault_depth = reserved_cluster_fault_depth
        self.global_admission_guard = global_admission_guard
        self.max_batch_size = max_batch_size
        self.max_flush_groups = max_flush_groups
        self.flush_delay_seconds = flush_delay_seconds
        self.projection_margin = projection_margin
        # What one batch does to the store. The default admits into the
        # processor queue; the telemetry spool passes its own, because it
        # takes different caps and no fault reserve - a reserve only makes
        # sense while faults and telemetry share a queue, which is the
        # arrangement the spool exists to end. Everything else here -
        # per-scope pipelining, the deadline projection, the wait and
        # flush histograms - is about how work is grouped, not about where
        # it lands, and is shared verbatim.
        self.admit_batch = admit_batch or self._admit_queue_batch
        self.label = label
        self.scope_key = scope_key
        self._lock = asyncio.Lock()
        # Pending entries, keyed by cluster scope, FIFO inside a scope.
        #
        # This used to be one flat list drained in rounds: ``_flush``
        # picked up to ``max_flush_groups`` scopes, awaited all of them
        # with ``asyncio.gather``, and only looked at the queue again
        # once the slowest one returned. One cluster is enough to stop
        # the whole process that way, and it happened: two runs of
        # identical code and config differed only in that the second
        # had a flush sit 132s on its cluster's
        # ``gpu_fault_processor_queue_counts`` row, which dragged the
        # round time from 3.2s to 44.9s, pushed queue waits to 43s and
        # answered 2,216 requests with a deadline rejection they had the
        # budget to survive - 12,601 accepted against 27,296.
        #
        # Per-scope queues remove the barrier. The only ordering that
        # has to survive is inside a scope, so a scope is allowed one
        # in-flight flush at a time: two concurrent batches for the same
        # cluster would both take that cluster's counter row and
        # recreate the convoy a batch exists to avoid.
        self._pending_by_scope: dict[str, list[_AdmissionEntry]] = {}
        self._pending_count = 0
        self._flush_task: asyncio.Task | None = None
        # scope -> monotonic start of its in-flight flush. Only the
        # dispatcher mutates this, so "is this scope busy" needs no
        # extra synchronisation beyond ``_lock``.
        self._in_flight_scopes: dict[str, float] = {}
        self._flush_tasks: dict[asyncio.Task, str] = {}
        # A dispatcher parked on the in-flight flushes has to be woken by
        # an arrival too, or a new cluster waits for an unrelated one to
        # finish - the barrier this change removes, rebuilt by accident.
        self._arrival = asyncio.Event()
        # Waiting for a flush round is the only unbounded queue on the
        # telemetry ingress path (ingress backpressure caps at 2s,
        # decode at 5s, store I/O admission at 2s), so it is also the
        # only place a 15s budget can vanish without a counter moving.
        self.pending_depth = 0
        self.pending_depth_max = 0
        self.submitted_total = 0
        self.expired_total = 0
        self.rounds_total = 0
        self.groups_total = 0
        self.items_total = 0
        self.deferred_total = 0
        self.queue_wait_count = 0
        self.queue_wait_sum_seconds = 0.0
        self.queue_wait_max_seconds = 0.0
        self.flush_count = 0
        self.flush_sum_seconds = 0.0
        self.flush_max_seconds = 0.0
        self.round_seconds_ewma = 0.0
        self.shed_total = 0
        self.in_flight_max = 0
        self.cap_waits_total = 0
        self.scope_busy_defers_total = 0
        self.scope_flush_max_seconds = 0.0
        self.scope_flush_max_scope = ""

    def _admit_queue_batch(self, items):
        """Runs on a store I/O thread, like every other store call here."""

        return self.store.try_enqueue_processor_requests_batch(
            items,
            max_depth=self.max_depth,
            max_cluster_depth=self.max_cluster_depth,
            reserved_fault_depth=self.reserved_fault_depth,
            reserved_cluster_fault_depth=(self.reserved_cluster_fault_depth),
            global_admission_guard=self.global_admission_guard,
        )

    @property
    def max_in_flight_flushes(self) -> int:
        """How many per-scope flushes may run at once.

        Same knob as before the barrier came out, and deliberately so:
        under ``asyncio.gather`` the width of a round already was the
        concurrency against the store, so an A/B on this value still
        varies exactly one thing - how many admission transactions the
        database sees at once - and the earlier measurement that 64
        concurrent flushes cost more in counter-row contention than they
        win in batching still applies to it.
        """

        return self.max_flush_groups

    @property
    def in_flight(self) -> int:
        return len(self._in_flight_scopes)

    @property
    def scopes_pending(self) -> int:
        return len(self._pending_by_scope)

    @property
    def round_capacity(self) -> int:
        return self.max_flush_groups * self.max_batch_size

    def _projected_wait_seconds(self, ahead: int) -> float:
        """How long a request joining behind ``ahead`` others will wait.

        A burst offers far more than the batcher can admit, and a request
        that cannot be served inside its budget is going to get a 503
        either way. Doing it on arrival instead of fifteen seconds later
        is what stops the queue from being full of work nobody is waiting
        for any more - and it is measured, not assumed: the estimate uses
        the observed round time, so it disappears when the batcher is
        keeping up.

        With the flushes pipelined, ``round_seconds_ewma`` tracks how
        long one flush takes rather than how long a barrier round took.
        For this estimate that is the same quantity - a saturated
        batcher retires ``round_capacity`` items per flush time - and it
        no longer carries an unrelated cluster's stall, which used to
        make the projection shed requests for a queue that was moving.
        """
        if self.round_seconds_ewma <= 0:
            return 0.0
        rounds_ahead = 1 + ahead // self.round_capacity
        return rounds_ahead * self.round_seconds_ewma * self.projection_margin

    def _projected_scope_wait_seconds(self, ahead: int) -> float:
        """Rounds this scope still needs before a new entry is flushed."""
        if self.round_seconds_ewma <= 0:
            return 0.0
        rounds_ahead = 1 + ahead // self.max_batch_size
        return rounds_ahead * self.round_seconds_ewma * self.projection_margin

    async def submit(self, item):
        future = asyncio.get_running_loop().create_future()
        deadline = REQUEST_DEADLINE.get()
        scope = self.scope_key(item)
        if deadline is not None and self.projection_margin > 0:
            # Queues are per scope and one flush is in flight per scope, so
            # the wait this request faces is its own scope's backlog over
            # ``max_batch_size`` -- not the process-wide backlog over the
            # fleet-wide capacity, which let one stalled cluster shed a
            # healthy cluster's requests (F-E2).
            ahead = len(self._pending_by_scope.get(scope, ()))
            projected = self._projected_scope_wait_seconds(ahead)
            if projected > deadline - time.monotonic():
                self.shed_total += 1
                raise StoreIoCapacityExceeded(
                    "processor admission batch cannot reach this "
                    "request within its deadline"
                )
        async with self._lock:
            self._pending_by_scope.setdefault(scope, []).append(
                _AdmissionEntry(
                    item=item,
                    future=future,
                    deadline=deadline,
                    enqueued_at=time.monotonic(),
                )
            )
            self._pending_count += 1
            self._arrival.set()
            self.submitted_total += 1
            self.pending_depth = self._pending_count
            self.pending_depth_max = max(self.pending_depth_max, self.pending_depth)
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(
                    self._flush(),
                    name="gpu-fault-processor-admission-batch",
                )
        return await future

    def _expire_locked(self, now: float) -> float | None:
        """Reject entries whose own budget lapsed; return the next one.

        Entries whose budget already lapsed are answered where they
        waited rather than carried into a transaction: the client stopped
        listening, so the insert would be thrown away, and keeping them
        in the group used to drag the whole group's deadline down with
        them.

        The returned value is how long until the earliest still-valid
        deadline, so the dispatcher can wake for it instead of only when
        a flush completes. Without that, an entry behind a 132s flush was
        told its deadline had passed 132s after the fact, which is the
        same 503 the client had already given up waiting for.
        """
        earliest: float | None = None
        for scope in list(self._pending_by_scope):
            queue = self._pending_by_scope[scope]
            kept: list[_AdmissionEntry] = []
            dropped = 0
            for entry in queue:
                if entry.deadline is not None and entry.deadline <= now:
                    self.expired_total += 1
                    self._pending_count -= 1
                    dropped += 1
                    if not entry.future.done():
                        entry.future.set_exception(
                            StoreIoCapacityExceeded(
                                "request deadline exceeded while "
                                "waiting for a processor admission "
                                "batch"
                            )
                        )
                    continue
                if entry.deadline is not None and (
                    earliest is None or entry.deadline < earliest
                ):
                    earliest = entry.deadline
                kept.append(entry)
            if not dropped:
                # The common case: one pass per flush completion, so this
                # runs often enough that not rewriting the queue matters.
                continue
            if kept:
                self._pending_by_scope[scope] = kept
            else:
                del self._pending_by_scope[scope]
        self.pending_depth = self._pending_count
        if earliest is None:
            return None
        return max(0.0, earliest - now)

    def _dispatch_locked(self, now: float) -> list[tuple[str, list[_AdmissionEntry]]]:
        """Take one batch from every scope that can start one now.

        The store opens one transaction per cluster scope regardless of
        how the batch is assembled, so slicing the queue FIFO and then
        grouping degenerates into one two-row transaction per cluster
        once a burst spans dozens of clusters. Taking whole clusters
        instead keeps every transaction wide.

        A scope already flushing is skipped, not delayed behind anyone:
        its next batch starts when its own flush returns. A scope past
        the in-flight cap waits for the next free slot rather than for
        the slowest cluster in a round, which is the whole point of the
        change - the cap now bounds concurrency against the pool, it no
        longer couples unrelated clusters to each other.
        """
        dispatched: list[tuple[str, list[_AdmissionEntry]]] = []
        busy_defers = 0
        capacity = max(
            0,
            self.max_in_flight_flushes - len(self._in_flight_scopes),
        )
        for scope in list(self._pending_by_scope):
            queue = self._pending_by_scope[scope]
            if scope in self._in_flight_scopes:
                busy_defers += len(queue)
                continue
            if capacity <= 0:
                continue
            group = queue[: self.max_batch_size]
            del queue[: len(group)]
            for entry in group:
                waited = now - entry.enqueued_at
                self.queue_wait_count += 1
                self.queue_wait_sum_seconds += waited
                self.queue_wait_max_seconds = max(self.queue_wait_max_seconds, waited)
            self._pending_count -= len(group)
            # Re-inserting the scope at the back is what keeps a cluster
            # with a permanent backlog from monopolising the slots: dict
            # order is the dispatch order.
            del self._pending_by_scope[scope]
            if queue:
                self._pending_by_scope[scope] = queue
            self._in_flight_scopes[scope] = now
            dispatched.append((scope, group))
            capacity -= 1
        self.pending_depth = self._pending_count
        self.scope_busy_defers_total += busy_defers
        self.in_flight_max = max(self.in_flight_max, len(self._in_flight_scopes))
        if (
            self._pending_count
            and len(self._in_flight_scopes) >= self.max_in_flight_flushes
        ):
            self.cap_waits_total += 1
        if dispatched:
            self.rounds_total += 1
            self.groups_total += len(dispatched)
            self.items_total += sum(len(group) for _, group in dispatched)
            self.deferred_total += self._pending_count
        return dispatched

    async def _run_group(self, scope: str, group: list[_AdmissionEntry]) -> str:
        items = [entry.item for entry in group]
        # A batch may only be cut short by the request in it that gives
        # up last; anything earlier would reject requests that still had
        # budget left. Each group is its own task, so this deadline is
        # set in its own context copy - under the old ``gather`` the
        # concurrent groups shared one context and overwrote each
        # other's token.
        deadlines = [entry.deadline for entry in group]
        group_deadline = (
            None if any(deadline is None for deadline in deadlines) else max(deadlines)
        )
        token = REQUEST_DEADLINE.set(group_deadline)
        started = time.monotonic()
        results: list = []
        error: Exception | None = None
        try:
            results = await self.store_io.run(self.admit_batch, items)
            if len(results) != len(group):
                raise RuntimeError(
                    f"{self.label} batch returned the wrong number of results"
                )
        except Exception as exc:  # noqa: BLE001 - reported per entry
            # A retryable store failure (serialization, deadlock, lost
            # connection) is a capacity answer for every request in the
            # group -- a 503 with Retry-After -- not a 500 (F-E3). Anything
            # else is a bug and surfaces as itself.
            if not isinstance(exc, StoreIoCapacityExceeded) and operation_should_retry(
                exc
            ):
                error = StoreIoCapacityExceeded(
                    "PostgreSQL writer is temporarily unavailable"
                )
                error.__cause__ = exc
            else:
                error = exc
            results = []
        finally:
            REQUEST_DEADLINE.reset(token)
            elapsed = time.monotonic() - started
            self.flush_count += 1
            self.flush_sum_seconds += elapsed
            self.flush_max_seconds = max(self.flush_max_seconds, elapsed)
            if elapsed > self.scope_flush_max_seconds:
                self.scope_flush_max_seconds = elapsed
                self.scope_flush_max_scope = scope
            self.round_seconds_ewma = (
                elapsed
                if self.round_seconds_ewma <= 0
                else 0.7 * self.round_seconds_ewma + 0.3 * elapsed
            )
        if error is not None:
            LOGGER.error(
                "%s batch failed cluster_id=%s size=%s paths=%s",
                self.label,
                getattr(group[0].item, "cluster_id", None),
                len(group),
                sorted({getattr(entry.item, "path", "") for entry in group}),
                exc_info=(
                    type(error),
                    error,
                    error.__traceback__,
                ),
            )
            for entry in group:
                if not entry.future.done():
                    entry.future.set_exception(error)
            return scope
        for entry, result in zip(group, results, strict=True):
            if not entry.future.done():
                entry.future.set_result(result)
        return scope

    async def _flush(self) -> None:
        # The loop outlives the request that started it, so it must not
        # keep that request's deadline: every store call below runs under
        # the deadline of the group it is serving, set in _run_group.
        REQUEST_DEADLINE.set(None)
        try:
            await self._flush_loop()
        except Exception as exc:  # noqa: BLE001 - the loop itself broke
            # Nothing below _run_group answers the futures; if the loop's
            # own bookkeeping raised, every pending waiter would hang until
            # its client gave up (F-E3). Fail them, then let the next
            # submit start a fresh loop.
            LOGGER.exception(
                "%s flush loop failed; failing pending entries", self.label
            )
            async with self._lock:
                pending = [
                    entry
                    for entries in self._pending_by_scope.values()
                    for entry in entries
                ]
                self._pending_by_scope.clear()
                self._pending_count = 0
                self.pending_depth = 0
                self._flush_task = None
            for entry in pending:
                if not entry.future.done():
                    entry.future.set_exception(exc)

    async def _flush_loop(self) -> None:
        while True:
            # One coalescing window per pass rather than one per
            # dispatcher. Under the barrier a round lasted seconds, so
            # entries piled up and batches were wide for free; now an
            # arrival wakes the loop immediately, and without a window
            # a burst would be admitted roughly one item per
            # transaction - the opposite of what a batcher is for.
            await asyncio.sleep(self.flush_delay_seconds)
            async with self._lock:
                now = time.monotonic()
                # Cleared under the lock ``submit`` sets it under, so an
                # arrival can never be missed between the two.
                self._arrival.clear()
                next_deadline = self._expire_locked(now)
                dispatched = self._dispatch_locked(now)
                if not dispatched and not self._flush_tasks and not self._pending_count:
                    self._flush_task = None
                    return
            for scope, group in dispatched:
                task = asyncio.create_task(
                    self._run_group(scope, group),
                    name=(f"gpu-fault-processor-admission-flush-{scope}"),
                )
                self._flush_tasks[task] = scope
            arrival = asyncio.ensure_future(self._arrival.wait())
            try:
                done, _ = await asyncio.wait(
                    {*self._flush_tasks, arrival},
                    timeout=next_deadline,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not arrival.done():
                    arrival.cancel()
            done.discard(arrival)
            for task in done:
                scope = self._flush_tasks.pop(task, None)
                if scope is not None:
                    self._in_flight_scopes.pop(scope, None)
                if task.cancelled():
                    continue
                failure = task.exception()
                if failure is not None:
                    # _run_group answers its own entries, so reaching
                    # here means the task itself broke. Freeing the
                    # scope above is what keeps that from fencing the
                    # cluster for the life of the process.
                    LOGGER.exception(
                        "processor admission flush task failed cluster_id=%s",
                        scope,
                        exc_info=(
                            type(failure),
                            failure,
                            failure.__traceback__,
                        ),
                    )

    def scope_queue_snapshot(
        self, limit: int = 5
    ) -> tuple[list[tuple[str, float]], list[tuple[str, int, float]]]:
        """Per-scope queueing, worst first, for ``/metrics``.

        A scrape lands on a random uvicorn process, so this is only ever
        one process's view - but it is enough to name the cluster that is
        stalling, which the aggregate counters cannot do. Bounded to
        ``limit`` series so cluster count cannot become label
        cardinality.
        """
        now = time.monotonic()
        in_flight = sorted(
            (
                (scope, now - started)
                for scope, started in self._in_flight_scopes.items()
            ),
            key=lambda row: row[1],
            reverse=True,
        )[:limit]
        pending = sorted(
            (
                (scope, len(queue), now - queue[0].enqueued_at)
                for scope, queue in self._pending_by_scope.items()
                if queue
            ),
            key=lambda row: row[2],
            reverse=True,
        )[:limit]
        return in_flight, pending

    async def close(self) -> None:
        task = self._flush_task
        if task is not None and not task.done():
            await task
        # The dispatcher only returns with nothing in flight, so this is
        # belt and braces for the case where it died on its own.
        outstanding = list(self._flush_tasks)
        if outstanding:
            await asyncio.gather(*outstanding, return_exceptions=True)
