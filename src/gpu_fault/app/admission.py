from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

from gpu_fault.app.admission_concurrency import FlushConcurrencyController
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    AsyncStoreExecutor,
    RequestDeadlineExceeded,
    StoreIoCapacityExceeded,
)
from gpu_fault.store.shared.errors import operation_should_retry

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
        min_flush_groups: int | None = None,
        flush_open_below_seconds: float = 1.0,
        flush_close_above_seconds: float = 2.0,
        flush_delay_seconds: float = 0.002,
        flush_delay_max_seconds: float | None = None,
        contention_flush_seconds: float = 0.1,
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
        if flush_delay_max_seconds is None:
            flush_delay_max_seconds = flush_delay_seconds
        if flush_delay_max_seconds < flush_delay_seconds:
            raise ValueError(
                "processor admission batch delay cap cannot be below the base delay"
            )
        if contention_flush_seconds <= 0:
            raise ValueError(
                "processor admission contention threshold must be positive"
            )
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
        # ``max_flush_groups`` is the ceiling on concurrent flushes in this
        # process and ``min_flush_groups`` the floor the process starts at
        # (the ceiling when omitted, i.e. a fixed limit). The controller
        # moves the live limit between them on the flush-latency EWMA and
        # the dispatcher's own saturation signal; see admission_concurrency.
        self.max_flush_groups = max_flush_groups
        self.concurrency = FlushConcurrencyController(
            floor=max_flush_groups if min_flush_groups is None else min_flush_groups,
            ceiling=max_flush_groups,
            open_below_seconds=flush_open_below_seconds,
            close_above_seconds=flush_close_above_seconds,
        )
        self.flush_delay_seconds = flush_delay_seconds
        # The coalescing window is adaptive: ``flush_delay_seconds`` while
        # the store answers quickly, stretched toward
        # ``flush_delay_max_seconds`` once the flush EWMA passes
        # ``contention_flush_seconds`` (see ``effective_flush_delay_seconds``).
        # Equal values switch the adaptation off.
        self.flush_delay_max_seconds = flush_delay_max_seconds
        self.contention_flush_seconds = contention_flush_seconds
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
        # Whether the last dispatch pass left work queued with every slot
        # busy - the demand signal the concurrency controller opens on.
        self._saturated = False
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
        self.flush_delay_current_seconds = flush_delay_seconds
        self.delay_stretched_rounds_total = 0

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
        """How many per-scope flushes may run at once in this process, now.

        Under ``asyncio.gather`` the width of a round already was the
        concurrency against the store, so this still varies exactly one
        thing - how many admission transactions the database sees from
        this process at once. What the database sees is this times the
        number of ingress processes, which is why the regional runtime
        derives floor and ceiling from site-wide values
        (``postgres_capacity.flush_slots_per_process``) instead of carrying
        a per-process literal: 16 per process on 12 processes was 192
        concurrent admission INSERTs on one table, and 2 per process was a
        storm that expired 4,601 requests in the queue. The value moves
        between the floor and ``max_flush_groups`` (``self.concurrency``).
        """

        return self.concurrency.limit

    @property
    def in_flight_limit_raised_total(self) -> int:
        return self.concurrency.raised_total

    @property
    def in_flight_limit_lowered_total(self) -> int:
        return self.concurrency.lowered_total

    def effective_flush_delay_seconds(self) -> float:
        """The coalescing window the next dispatch pass waits for.

        The base delay while flushes are quick. Once the flush EWMA passes
        ``contention_flush_seconds`` the store is the bottleneck, and a
        longer window is pure gain: the same requests leave in fewer, wider
        transactions, and statements are exactly what a contended table
        needs fewer of (at ~1,000 req/s a 20-50 ms window cuts them 3-5x).
        The delay ramps linearly from the base at the threshold to
        ``flush_delay_max_seconds`` at twice the threshold and follows the
        EWMA back down, so an uncontended store never pays it.
        """

        base, cap = self.flush_delay_seconds, self.flush_delay_max_seconds
        threshold = self.contention_flush_seconds
        if cap <= base or self.round_seconds_ewma <= threshold:
            return base
        ratio = min(1.0, (self.round_seconds_ewma - threshold) / threshold)
        return base + (cap - base) * ratio

    @property
    def in_flight(self) -> int:
        return len(self._in_flight_scopes)

    @property
    def scopes_pending(self) -> int:
        return len(self._pending_by_scope)

    def projection_in_flight_limit(self) -> int:
        """The flush concurrency the deadline projection divides the queue by.

        The limit the controller is heading for, not the one it is at: while
        the store answers below the open threshold a saturated batcher keeps
        opening slots, so a backlog that arrives mid-ramp is served at the
        ceiling - counting it against the floor would shed requests for
        capacity that is about to open. Once the store is contended the
        ramp has stopped and the live limit is the honest figure.
        """

        if self.concurrency.may_open(self.round_seconds_ewma):
            return self.concurrency.ceiling
        return self.concurrency.limit

    def _projected_scope_wait_seconds(self, ahead: int, scopes_ahead: int) -> float:
        """How long a request joining its scope behind ``ahead`` others waits.

        A burst offers far more than the batcher can admit, and a request
        that cannot be served inside its budget is going to get a 503
        either way. Doing it on arrival instead of fifteen seconds later
        is what stops the queue from being full of work nobody is waiting
        for any more - and it is measured, not assumed: the estimate uses
        the observed flush time, so it disappears when the batcher is
        keeping up.

        Two waits add up. The scope's own: one flush per ``max_batch_size``
        entries already queued for it, since a scope runs one flush at a
        time. And the wait for a slot: the ``scopes_ahead`` other scopes
        with queued entries are dispatched before this one in rounds of
        ``projection_in_flight_limit`` flushes. The live storm had the
        second term and the projection did not: 32 scopes on 2 slots per
        process, every entry waited 10-15 s for a slot while the projection
        said one flush, and 4,601 requests expired in the queue instead of
        being answered on arrival. Dividing by the limit the controller is
        heading for keeps the term from shedding for capacity that opens.
        """
        if self.round_seconds_ewma <= 0:
            return 0.0
        rounds_ahead = 1 + ahead // self.max_batch_size
        rounds_ahead += scopes_ahead // self.projection_in_flight_limit()
        return rounds_ahead * self.round_seconds_ewma * self.projection_margin

    async def submit(self, item):
        future = asyncio.get_running_loop().create_future()
        deadline = REQUEST_DEADLINE.get()
        scope = self.scope_key(item)
        if deadline is not None and self.projection_margin > 0:
            # Queues are per scope and one flush is in flight per scope, so
            # the wait this request faces is its own scope's backlog over
            # ``max_batch_size`` plus the slot wait behind the other scopes
            # -- not the process-wide item backlog over the fleet-wide
            # capacity, which let one stalled cluster shed a healthy
            # cluster's requests (F-E2): a stalled scope holds one slot and
            # counts once here, the same as any other scope ahead.
            queue = self._pending_by_scope.get(scope)
            ahead = len(queue) if queue else 0
            scopes_ahead = len(self._pending_by_scope) - (1 if queue else 0)
            projected = self._projected_scope_wait_seconds(ahead, scopes_ahead)
            if projected > deadline - time.monotonic():
                self.shed_total += 1
                # The request's own budget is what cannot be met, so it is
                # reported as a deadline and not as store capacity (A-1).
                raise RequestDeadlineExceeded(
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
                            RequestDeadlineExceeded(
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

        The cap itself moves: the controller is asked first, with the
        previous pass's saturation and the flush EWMA, and this pass
        dispatches up to the limit it answers with - a slot opened on
        saturation is filled in the same pass, not on the next wake-up.
        """
        dispatched: list[tuple[str, list[_AdmissionEntry]]] = []
        busy_defers = 0
        limit = self.concurrency.observe(
            saturated=self._saturated, flush_ewma=self.round_seconds_ewma
        )
        capacity = max(0, limit - len(self._in_flight_scopes))
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
        self._saturated = bool(self._pending_count) and (
            len(self._in_flight_scopes) >= limit
        )
        if self._saturated:
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
                # The executor only counts the rejections it raises itself, so
                # a failure wrapped here is counted here, under the reason the
                # alert rules split on: the writer answered, briefly, with a
                # retryable error -- not a full lane.
                self.store_io.record_rejection("backend_unavailable")
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
            self.concurrency.record_flush()
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
            delay = self.effective_flush_delay_seconds()
            self.flush_delay_current_seconds = delay
            if delay > self.flush_delay_seconds:
                self.delay_stretched_rounds_total += 1
            await asyncio.sleep(delay)
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
