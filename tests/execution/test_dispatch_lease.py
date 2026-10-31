"""The workflow-dispatch lease under a slow tick (b226).

Every control-worker process runs a dispatcher and only the holder of the
``workflow-dispatch`` lease scans (F-A1). The lease used to be taken once per
tick and never renewed while the tick ran, so a tick longer than the 15-s lease
-- a holder starved of CPU while it works a full batch -- let the next
process's ordinary poll take the lapsed lease and run a second full scan while
the first was still working; rows the first had queued were claimed by both.

Here the lease runs on a fake clock (``DispatchLease.clock``) and a "slow" row
or sweep is one that moves it. A tick that keeps making progress keeps the
lease however long it runs; a holder that finds the lease taken stops starting
work and leaves its queued rows to the new holder; the counters say which.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatch_lease import (
    DispatchLease,
    DispatchLeaseLost,
    drain_batch,
)
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    WorkflowDispatchReport,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)

OP = WorkflowOperation.FREEZE_EVIDENCE
LEASE_SECONDS = 15.0
KEY = WorkflowDispatcher.DISPATCH_LEASE_KEY
NO_LOSSES = {"mid_tick": 0, "tick_start": 0}


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime.now(timezone.utc)
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += timedelta(seconds=seconds)


class StoreView:
    """One dispatcher's handle on the shared store.

    Counts its dispatch scan pages and its lease calls; ``before_renewal``
    runs before each lease call after the tick's first (every mid-tick
    renewal), ``on_first_sweep`` once, when its tick first lists PENDING rows.
    """

    def __init__(
        self,
        store: Any,
        *,
        before_renewal: Callable[[], None] | None = None,
        on_first_sweep: Callable[[], None] | None = None,
    ) -> None:
        self.inner = store
        self.before_renewal = before_renewal
        self.on_first_sweep = on_first_sweep
        self.scans = 0
        self.lease_calls = 0
        self.renewals = 0
        self.lease_owners: list[str] = []
        self.changed = threading.Condition()

    def acquire_periodic_task_lease(self, task_key, owner_id, *, now, lease_duration):
        self.lease_calls += 1
        renewal = self.lease_calls > 1
        if renewal and self.before_renewal is not None:
            self.before_renewal()
        lease = self.inner.acquire_periodic_task_lease(
            task_key, owner_id, now=now, lease_duration=lease_duration
        )
        with self.changed:
            self.renewals += int(renewal)
            self.lease_owners.append(lease.owner_id)
            self.changed.notify_all()
        return lease

    def wait_until(self, predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
        with self.changed:
            return self.changed.wait_for(predicate, timeout=timeout)

    def list_workflows(self, statuses=None, **kwargs):
        if kwargs.get("dispatchable_at") is not None:
            self.scans += 1
        elif self.on_first_sweep is not None and WorkflowStatus.PENDING in (
            statuses or ()
        ):
            hook, self.on_first_sweep = self.on_first_sweep, None
            hook()
        return self.inner.list_workflows(statuses, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class Rows:
    """The holder's adapter: ``before(index)`` runs as row ``index`` starts."""

    def __init__(self, before: Callable[[int], None] | None = None) -> None:
        self.before = before
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def supports(self, step) -> bool:
        return step.operation is OP

    def execute(self, context) -> WorkflowStepOutcome:
        with self._lock:
            index = len(self.calls)
            self.calls.append(context.workflow.request_id)
        if self.before is not None:
            self.before(index)
        return WorkflowStepOutcome.succeeded()


class RecordingExecutor:
    """A second process's executor: it records what its scan dispatched and
    executes nothing, so a row it reaches stays where the holder left it."""

    def __init__(self, executor_id: str) -> None:
        self.config = SimpleNamespace(executor_id=executor_id)
        self.requested: list[str] = []

    def execute(self, request_id, request):
        self.requested.append(request_id)
        raise WorkflowLeaseError("recorded only")


def _seed(store, *request_ids: str) -> None:
    for request_id in request_ids:
        incident = fault_incident(
            f"inc-{request_id}",
            f"event-{request_id}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=request_id,
            fencing_token=1,
        )
        store.save_incident_and_workflow(
            incident,
            workflow_request(
                request_id,
                incident.incident_id,
                fencing_token=1,
                official_steps=[workflow_step(OP)],
            ),
        )


def _dispatcher(store, executor, clock: FakeClock, *, workers: int = 1):
    dispatcher = WorkflowDispatcher(
        store,
        executor,
        WorkflowDispatcherConfig(
            enabled=True,
            batch_size=100,
            max_workers=workers,
            dispatch_lease_seconds=LEASE_SECONDS,
        ),
    )
    dispatcher.lease.clock = clock
    return dispatcher


def _holder(store, rows: Rows, clock: FakeClock, *, workers: int = 1):
    executor = active_workflow_executor(store, [rows], [OP], executor_id="executor-a")
    return _dispatcher(store, executor, clock, workers=workers)


def _current_lease(store, clock: FakeClock):
    """The lease row as it stands: asked as a caller whose clock reads a day
    earlier, the store sees an unexpired lease and only returns it."""

    return store.acquire_periodic_task_lease(
        KEY,
        "probe",
        now=clock() - timedelta(days=1),
        lease_duration=timedelta(seconds=LEASE_SECONDS),
    )


def _statuses(store, *request_ids: str) -> list[WorkflowStatus]:
    return [store.get_workflow(request_id).status for request_id in request_ids]


def test_a_tick_that_outlives_its_lease_while_making_progress_keeps_it() -> None:
    """Four rows, each a third of the lease long, on one worker: the tick runs
    20 s against a 15-s lease. Another process polls just before each of the
    holder's renewals -- at 15 s too, when the lease taken at the start of the
    tick has run out and the old code handed it over -- and is refused every
    time; the holder works its whole batch under one epoch."""

    clock = FakeClock()
    store = build_store()
    rows_ids = ("wf-1", "wf-2", "wf-3", "wf-4")
    _seed(store, *rows_ids)
    second = _dispatcher(store, RecordingExecutor("executor-b"), clock)
    polls: list[WorkflowDispatchReport] = []
    view = StoreView(store, before_renewal=lambda: polls.append(second.run_once()))
    out_of_order: list[int] = []

    def a_third_of_the_lease(index: int) -> None:
        # Row n starts once the holder handled row n-1's completion, so every
        # renewal lands between two rows and the order is deterministic.
        if index and not view.wait_until(lambda: view.renewals >= index):
            out_of_order.append(index)
        clock.advance(LEASE_SECONDS / 3)

    rows = Rows(before=a_third_of_the_lease)
    holder = _holder(view, rows, clock)

    report = holder.run_once()

    assert out_of_order == [], "a row started before the holder renewed"
    assert [poll.lease_held_by_other for poll in polls] == [True, True, True], (
        "the second process took the lease from a holder that kept working"
    )
    assert [poll.scanned for poll in polls] == [0, 0, 0], polls
    assert second.executor.requested == [], "the second process dispatched rows"
    assert (report.lease_held_by_other, report.executed) == (False, 4), report
    assert _statuses(store, *rows_ids) == [WorkflowStatus.SUCCEEDED] * 4
    lease = _current_lease(store, clock)
    assert (lease.owner_id, lease.epoch) == ("executor-a", 1), lease
    assert (holder.lease.acquisitions_total, holder.lease.losses_total) == (
        1,
        NO_LOSSES,
    )
    buckets, count, total = holder.lease.tick_seconds.snapshot()
    assert (count, total) == (1, 20.0), (count, total)
    assert (dict(buckets)[15.0], dict(buckets)[30.0]) == (0, 1), buckets
    assert holder.lease.held(), "the holder lost the lease it just renewed"
    assert second.lease.tick_seconds.snapshot()[1] == 0, (
        "a tick that found the lease held elsewhere is not a dispatch tick"
    )


def test_a_holder_overtaken_in_its_sweeps_stops_before_its_scan() -> None:
    """The holder stalls past its lease in the first sweep and another
    process's poll takes the lapsed lease; the holder's next checkpoint sees
    the new owner, so it never pages its dispatch scan nor starts a row."""

    clock = FakeClock()
    store = build_store()
    _seed(store, "wf-1", "wf-2", "wf-3")
    second = _dispatcher(store, RecordingExecutor("executor-b"), clock)
    polls: list[WorkflowDispatchReport] = []

    def stalled_past_the_lease() -> None:
        clock.advance(LEASE_SECONDS + 5)
        polls.append(second.run_once())

    view = StoreView(store, on_first_sweep=stalled_past_the_lease)
    rows = Rows()
    holder = _holder(view, rows, clock)

    report = holder.run_once()

    assert [poll.lease_held_by_other for poll in polls] == [False], polls
    assert sorted(second.executor.requested) == ["wf-1", "wf-2", "wf-3"], (
        "the new holder scans"
    )
    assert (report.lease_held_by_other, report.scanned) == (True, 0), report
    assert view.scans == 0, "the overtaken holder paged its dispatch scan"
    assert rows.calls == [], "the overtaken holder started a row"
    assert _statuses(store, "wf-1", "wf-2", "wf-3") == [WorkflowStatus.PENDING] * 3
    assert holder.lease.losses_total == {"mid_tick": 1, "tick_start": 0}
    assert not holder.lease.held(), "the gauge still claims the lost lease"
    assert second.lease.held(), "the new holder does not report its lease"
    assert holder.lease.tick_seconds.snapshot()[1] == 1, "the lost tick is a tick"

    later = holder.run_once()

    assert later.lease_held_by_other, "the overtaken holder dispatched again"
    assert holder.lease.losses_total == {"mid_tick": 1, "tick_start": 0}, (
        "one loss was counted twice"
    )


def test_a_holder_overtaken_mid_batch_notices_and_finishes_its_running_rows() -> None:
    """Two rows run on two workers; one stalls past the lease and another
    process takes it. The holder notices at the row that finishes -- the old
    code never noticed -- and its report says the lease is elsewhere; the row
    still running finishes. (Rows still queued are cancelled: the drain tests
    below.)"""

    clock = FakeClock()
    store = build_store()
    _seed(store, "wf-1", "wf-2")
    second = _dispatcher(store, RecordingExecutor("executor-b"), clock)
    polls: list[WorkflowDispatchReport] = []
    started = threading.Event()
    view = StoreView(store)
    timed_out: list[str] = []

    def row(index: int) -> None:
        if index == 0:
            # The other row is running before this one stalls past the lease.
            if not started.wait(10):
                timed_out.append("second row never started")
            clock.advance(LEASE_SECONDS + 5)
            polls.append(second.run_once())
            return
        started.set()
        # Finish only once the holder has seen the new owner.
        if not view.wait_until(lambda: "executor-b" in view.lease_owners):
            timed_out.append("the holder never looked at the lease")

    rows = Rows(before=row)
    holder = _holder(view, rows, clock, workers=2)

    report = holder.run_once()

    assert timed_out == [], timed_out
    assert [poll.lease_held_by_other for poll in polls] == [False], polls
    assert (report.lease_held_by_other, report.executed) == (True, 2), report
    assert holder.lease.losses_total == {"mid_tick": 1, "tick_start": 0}
    assert _statuses(store, "wf-1", "wf-2") == [WorkflowStatus.SUCCEEDED] * 2, (
        "a row the holder had running was abandoned"
    )
    lease = _current_lease(store, clock)
    assert (lease.owner_id, lease.epoch) == ("executor-b", 2), lease


# ---------------------------------------------------------------- the batch


def _lease_held_by_other_process(clock: FakeClock) -> tuple[DispatchLease, Any]:
    """A lease this process took, that another process took over after it
    lapsed; this process has not looked since."""

    store = build_store()
    lease = DispatchLease(KEY, LEASE_SECONDS, clock=clock)
    assert lease.acquire(store, "executor-a"), "the first take failed"
    clock.advance(LEASE_SECONDS + 5)
    taken = store.acquire_periodic_task_lease(
        KEY, "executor-b", now=clock(), lease_duration=timedelta(seconds=15)
    )
    assert taken.owner_id == "executor-b", taken
    return lease, store


def _done(value: object = None) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_result(value)
    return future


def test_a_lost_lease_cancels_the_queued_rows_and_keeps_the_finished_ones() -> None:
    clock = FakeClock()
    lease, _ = _lease_held_by_other_process(clock)
    finished = _done()
    queued: list[Future[Any]] = [Future(), Future()]

    with lease.holding_tick(clock()):
        drained = drain_batch({finished, *queued}, deadline=None, lease=lease)

    assert drained.completed == [finished], drained
    assert (drained.lease_lost, drained.left_queued, drained.deferred) == (True, 2, 0)
    assert all(future.cancelled() for future in queued), "a queued row was started"
    assert lease.losses_total == {"mid_tick": 1, "tick_start": 0}


def test_a_wake_without_a_finished_row_renews_nothing() -> None:
    """The cycle deadline wakes the drain with no row finished: queued rows are
    deferred as before (F-C7) and the lease is not renewed -- only progress
    renews, so a holder whose rows never finish lets the lease lapse."""

    clock = FakeClock()
    store = build_store()
    view = StoreView(store)
    lease = DispatchLease(KEY, LEASE_SECONDS, clock=clock)
    assert lease.acquire(view, "executor-a"), "the first take failed"
    clock.advance(LEASE_SECONDS)
    queued: Future[Any] = Future()

    with lease.holding_tick(clock()):
        drained = drain_batch({queued}, deadline=0.0, lease=lease)

    assert (drained.deferred, drained.left_queued, drained.lease_lost) == (1, 0, False)
    assert view.lease_calls == 1, "the lease was renewed without progress"


# ---------------------------------------------------------------- the lease


def _counting(store) -> StoreView:
    return StoreView(store)


def test_a_new_epoch_is_an_acquisition_and_a_renewal_is_not() -> None:
    clock = FakeClock()
    store = build_store()
    lease = DispatchLease(KEY, LEASE_SECONDS, clock=clock)

    assert lease.acquire(store, "executor-a"), "the first take failed"
    clock.advance(5)
    assert lease.acquire(store, "executor-a"), "the renewal failed"
    assert lease.acquisitions_total == 1
    clock.advance(LEASE_SECONDS + 5)
    assert lease.acquire(store, "executor-a"), "the lapsed lease was not re-taken"

    assert (lease.acquisitions_total, lease.losses_total) == (2, NO_LOSSES), (
        "re-taking its own lapsed lease is an acquisition, not a loss"
    )


def test_a_lease_taken_while_the_holder_was_away_is_a_tick_start_loss() -> None:
    clock = FakeClock()
    lease, store = _lease_held_by_other_process(clock)

    assert not lease.acquire(store, "executor-a"), "two holders"
    assert lease.losses_total == {"mid_tick": 0, "tick_start": 1}
    assert not lease.held(), "the gauge still claims the lost lease"
    clock.advance(LEASE_SECONDS + 5)
    assert lease.acquire(store, "executor-a"), "the lapsed lease was not taken back"

    assert (lease.acquisitions_total, lease.losses_total) == (
        2,
        {"mid_tick": 0, "tick_start": 1},
    ), "taking back a lease already counted lost is not a second loss"


def test_an_epoch_gap_is_a_loss_the_holder_never_saw() -> None:
    """Held, lapsed, taken and lapsed again by another process, then taken
    back: the holder never saw the other owner, the epoch says it was there."""

    clock = FakeClock()
    lease, store = _lease_held_by_other_process(clock)
    clock.advance(LEASE_SECONDS + 5)

    assert lease.acquire(store, "executor-a"), "the lapsed lease was not taken back"

    assert (lease.acquisitions_total, lease.losses_total) == (
        2,
        {"mid_tick": 0, "tick_start": 1},
    )


def test_a_checkpoint_renews_once_a_third_of_the_lease_has_passed() -> None:
    clock = FakeClock()
    view = _counting(build_store())
    lease = DispatchLease(KEY, LEASE_SECONDS, clock=clock)
    assert lease.acquire(view, "executor-a"), "the first take failed"

    with lease.holding_tick(clock()):
        clock.advance(LEASE_SECONDS / 3 - 0.5)
        lease.checkpoint()
        calls_before_due = view.lease_calls
        clock.advance(0.5)
        lease.checkpoint()

    assert (calls_before_due, view.lease_calls) == (1, 2), (
        "renewed before a third of the lease, or not at it"
    )
    assert _current_lease(view, clock).lease_expires_at == clock() + timedelta(
        seconds=LEASE_SECONDS
    ), "the renewal did not extend the lease from its own time"


def test_a_checkpoint_outside_a_held_tick_checks_nothing() -> None:
    """A sweep called directly (an admin tool, a test) runs its checkpoints
    outside a tick: they never call the store and never raise, even with a
    renewal long due -- here, or on another thread while a tick runs."""

    clock = FakeClock()
    view = _counting(build_store())
    lease = DispatchLease(KEY, LEASE_SECONDS, clock=clock)
    assert lease.acquire(view, "executor-a"), "the first take failed"
    clock.advance(LEASE_SECONDS * 2)
    answers: list[bool] = []

    def elsewhere() -> None:
        lease.checkpoint()
        answers.append(lease.renew_if_due())

    elsewhere()
    with lease.holding_tick(clock()):
        other = threading.Thread(target=elsewhere)
        other.start()
        other.join(10)

    assert answers == [True, True], answers
    assert view.lease_calls == 1, "a checkpoint outside a tick called the store"


def test_after_a_mid_tick_loss_every_checkpoint_raises_without_a_store_call() -> None:
    clock = FakeClock()
    view = _counting(build_store())
    lease = DispatchLease(KEY, LEASE_SECONDS, clock=clock)
    assert lease.acquire(view, "executor-a"), "the first take failed"
    raised = 0

    with lease.holding_tick(clock()):
        clock.advance(LEASE_SECONDS + 5)
        view.inner.acquire_periodic_task_lease(
            KEY, "executor-b", now=clock(), lease_duration=timedelta(seconds=15)
        )
        for _ in range(3):
            try:
                lease.checkpoint()
            except DispatchLeaseLost:
                raised += 1

    assert (raised, view.lease_calls) == (3, 2), (
        "a lost lease was asked for again, or a checkpoint went on after it"
    )
    assert lease.losses_total == {"mid_tick": 1, "tick_start": 0}


class _FailingRenewals(StoreView):
    def acquire_periodic_task_lease(self, task_key, owner_id, *, now, lease_duration):
        if self.lease_calls:
            self.lease_calls += 1
            raise ConnectionError("store unavailable")
        return super().acquire_periodic_task_lease(
            task_key, owner_id, now=now, lease_duration=lease_duration
        )


def test_a_failed_renewal_is_retried_later_and_lost_once_the_lease_runs_out(
    caplog,
) -> None:
    """A Store error in a renewal keeps the batch going while the lease granted
    last lasts, is retried a third of the lease later (not at every finished
    row), propagates from a checkpoint like the tick's own Store errors, and
    once the lease has run out by this process's clock the lease is lost."""

    clock = FakeClock()
    view = _FailingRenewals(build_store())
    lease = DispatchLease(KEY, LEASE_SECONDS, clock=clock)
    assert lease.acquire(view, "executor-a"), "the first take failed"
    answers: list[bool] = []

    with lease.holding_tick(clock()), caplog.at_level(logging.WARNING):
        clock.advance(LEASE_SECONDS / 3)
        answers.append(lease.renew_if_due())
        clock.advance(1)
        answers.append(lease.renew_if_due())
        calls_while_backing_off = view.lease_calls
        clock.advance(LEASE_SECONDS / 3)
        with pytest.raises(ConnectionError):
            lease.checkpoint()
        clock.advance(LEASE_SECONDS / 3)
        answers.append(lease.renew_if_due())
        with pytest.raises(DispatchLeaseLost):
            lease.checkpoint()

    assert answers == [True, True, False], (
        "a failed renewal is lost only once the granted lease ran out"
    )
    assert (calls_while_backing_off, view.lease_calls) == (2, 4), (
        "a failed renewal was retried before a third of the lease"
    )
    assert "renewal failed mid-tick" in caplog.text, caplog.text


def test_without_the_lease_every_tick_is_observed_and_nothing_is_held() -> None:
    clock = FakeClock()
    store = build_store()
    _seed(store, "wf-1")
    executor = active_workflow_executor(store, [Rows()], [OP], executor_id="solo")
    dispatcher = WorkflowDispatcher(
        store,
        executor,
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )
    dispatcher.lease.clock = clock

    report = dispatcher.run_once()

    assert (report.lease_held_by_other, report.executed) == (False, 1), report
    assert dispatcher.lease.tick_seconds.snapshot()[1] == 1, "the tick was not seen"
    assert not dispatcher.lease.held(), "a disabled lease is reported held"
    assert dispatcher.lease.acquisitions_total == 0, dispatcher.lease.losses_total
