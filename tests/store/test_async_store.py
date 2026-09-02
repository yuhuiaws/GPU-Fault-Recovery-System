from __future__ import annotations

import asyncio
import os
import time
from threading import Event

import httpx
import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
    remaining_budget,
)
from tests._builders import asgi_client


def test_store_io_executor_bounds_in_flight_work() -> None:
    release = Event()
    started = Event()
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=0.01
    )

    async def scenario() -> None:
        def blocking_transaction() -> bool:
            started.set()
            return release.wait(5)

        first = asyncio.create_task(executor.run(blocking_transaction))
        while not started.is_set():
            await asyncio.sleep(0.001)
        try:
            with pytest.raises(StoreIoCapacityExceeded):
                await executor.run(lambda: None)
            assert executor.in_flight == 1
            assert executor.rejected_total == 1
        finally:
            release.set()
        # The restricted test sandbox can suppress the executor's
        # cross-thread event-loop wakeup, so keep timers registered.
        for _ in range(2_000):
            if first.done():
                break
            await asyncio.sleep(0.001)
        assert first.done()
        assert first.result()
        assert executor.in_flight == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_cancelled_store_call_keeps_slot_until_transaction_finishes() -> None:
    release = Event()
    started = Event()
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=0.01
    )

    async def scenario() -> None:
        def blocking_transaction() -> None:
            started.set()
            release.wait(5)

        first = asyncio.create_task(executor.run(blocking_transaction))
        while not started.is_set():
            await asyncio.sleep(0.001)
        first.cancel()
        with pytest.raises(StoreIoCapacityExceeded):
            await executor.run(lambda: None)
        assert executor.in_flight == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        for _ in range(1_000):
            if executor.in_flight == 0:
                break
            await asyncio.sleep(0.001)
        assert executor.in_flight == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_store_io_executor_does_not_poll_with_asyncio_sleep(monkeypatch) -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )

    async def forbidden_sleep(_seconds):
        raise AssertionError("store executor must await the Future")

    monkeypatch.setattr("gpu_fault.async_store.asyncio.sleep", forbidden_sleep)
    try:
        assert asyncio.run(executor.run(lambda: 42)) == 42
    finally:
        executor.close()


@pytest.mark.parametrize(
    ("module", "name", "sqlstate"),
    [
        ("psycopg.errors", "ReadOnlySqlTransaction", "25006"),
        ("psycopg", "OperationalError", None),
        ("psycopg_pool", "PoolTimeout", None),
    ],
)
def test_store_io_executor_maps_writer_outage_to_capacity_error(
    module: str, name: str, sqlstate: str | None
) -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    error_type = type(name, (Exception,), {"__module__": module, "sqlstate": sqlstate})

    def unavailable_writer() -> None:
        raise error_type("writer unavailable")

    async def scenario() -> None:
        with pytest.raises(
            StoreIoCapacityExceeded, match="writer is temporarily unavailable"
        ):
            await executor.run(unavailable_writer)
        assert executor.rejected_total == 1
        assert executor.in_flight == 0

    try:
        asyncio.run(scenario())
    finally:
        executor.close()


def test_admission_wait_is_clamped_to_the_request_deadline() -> None:
    """The deadline wins over the admission timeout, not the other way.

    Without this, a request whose client had already given up still
    waited the full admission timeout for a slot, and the queues
    stacked: ingress wait + admission wait + pool checkout.
    """
    release = Event()
    started = Event()
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=30
    )

    async def scenario() -> None:
        def blocking_transaction() -> None:
            started.set()
            release.wait(5)

        first = asyncio.create_task(executor.run(blocking_transaction))
        while not started.is_set():
            await asyncio.sleep(0.001)
        token = REQUEST_DEADLINE.set(time.monotonic())
        # A deadline that already passed must not even wait for a slot.
        try:
            assert remaining_budget(30) <= 0
            with pytest.raises(StoreIoCapacityExceeded):
                await asyncio.wait_for(executor.run(lambda: None), timeout=1)
        finally:
            REQUEST_DEADLINE.reset(token)
            release.set()
        for _ in range(2_000):
            if first.done():
                break
            await asyncio.sleep(0.001)
        assert first.done()

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_request_deadline_is_unset_outside_a_request() -> None:
    # Background services - dispatcher, processor claims, cleanup - share
    # the executor and must keep their own timeout.
    assert REQUEST_DEADLINE.get() is None
    assert remaining_budget(7) == 7


def test_processor_owner_is_unique_per_process(monkeypatch) -> None:
    from gpu_fault.app import _pod_process_owner

    monkeypatch.setenv("POD_UID", "pod-a")

    assert _pod_process_owner() == f"pod-a:{os.getpid()}"


def test_blocking_store_endpoint_does_not_block_event_loop(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "direct")
    started = Event()
    release = Event()
    context = ApplicationContext(execution_token="x" * 32)

    def slow_list():
        started.set()
        release.wait(5)
        return []

    context.store.list_regional_clusters = slow_list
    app = create_app(context)

    async def scenario() -> None:
        async with asgi_client(app) as client:
            blocked = asyncio.create_task(
                client.get(
                    "/v1/regional/clusters",
                    headers={"X-GPU-Fault-Execution-Token": "x" * 32},
                )
            )
            while not started.is_set():
                await asyncio.sleep(0.001)
            health = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
            assert health.status_code == 200
            release.set()
            assert (await blocked).status_code == 200

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        app.state.store_io.close()


def test_store_endpoint_returns_503_when_executor_is_saturated(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "direct")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "1")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS", "0.01")
    started = Event()
    release = Event()
    context = ApplicationContext(execution_token="x" * 32)

    def slow_list():
        started.set()
        release.wait(5)
        return []

    context.store.list_regional_clusters = slow_list
    app = create_app(context)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        headers = {"X-GPU-Fault-Execution-Token": "x" * 32}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            blocked = asyncio.create_task(
                client.get("/v1/regional/clusters", headers=headers)
            )
            while not started.is_set():
                await asyncio.sleep(0.001)
            rejected = await client.get("/v1/regional/clusters", headers=headers)
            assert rejected.status_code == 503
            assert rejected.headers["Retry-After"] == "2"
            release.set()
            assert (await blocked).status_code == 200

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        app.state.store_io.close()


def test_store_endpoint_returns_503_when_writer_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "direct")
    context = ApplicationContext(execution_token="x" * 32)
    error_type = type(
        "ReadOnlySqlTransaction",
        (Exception,),
        {"__module__": "psycopg.errors", "sqlstate": "25006"},
    )

    def read_only_writer():
        raise error_type("cannot execute in a read-only transaction")

    context.store.list_regional_clusters = read_only_writer
    app = create_app(context)

    async def scenario() -> None:
        async with asgi_client(app) as client:
            response = await client.get(
                "/v1/regional/clusters",
                headers={"X-GPU-Fault-Execution-Token": "x" * 32},
            )
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "2"

    try:
        asyncio.run(scenario())
    finally:
        app.state.store_io.close()


def _admission_batcher(store, executor, **kwargs):
    from gpu_fault.app import _ProcessorAdmissionBatcher

    return _ProcessorAdmissionBatcher(
        store,
        executor,
        max_depth=100_000,
        max_cluster_depth=100_000,
        reserved_fault_depth=0,
        reserved_cluster_fault_depth=0,
        global_admission_guard=0,
        **kwargs,
    )


def test_admission_batch_does_not_inherit_a_stale_submitter_deadline() -> None:
    """One submitter's expired budget must not reject everyone else.

    ``submit`` starts the flush loop with ``asyncio.create_task``, which
    copies the calling request's context - including REQUEST_DEADLINE.
    The loop then outlives that request under load, so every later round
    ran its store call with a budget that had already expired and every
    request in it got a 503 that belonged to a stranger.
    """
    from types import SimpleNamespace

    calls = []
    started = Event()
    release = Event()
    executor = AsyncStoreExecutor(
        workers=2, max_in_flight=4, admission_timeout_seconds=2
    )
    batcher = _admission_batcher(SimpleNamespace(), executor, flush_delay_seconds=0)

    def enqueue_batch(items, **_kwargs):
        calls.append(len(items))
        if len(calls) == 1:
            started.set()
            release.wait(5)
        return [(item, None) for item in items]

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        first_item = SimpleNamespace(cluster_id="a", path="/v1/x")
        second_item = SimpleNamespace(cluster_id="a", path="/v1/x")
        # The request that happens to start the flush loop has a budget
        # that expires almost immediately.
        token = REQUEST_DEADLINE.set(time.monotonic() + 0.05)
        try:
            first = asyncio.create_task(batcher.submit(first_item))
        finally:
            REQUEST_DEADLINE.reset(token)
        while not started.is_set():
            await asyncio.sleep(0.001)
        # A second request with a full budget joins the next round.
        token = REQUEST_DEADLINE.set(time.monotonic() + 100)
        try:
            second = asyncio.create_task(batcher.submit(second_item))
        finally:
            REQUEST_DEADLINE.reset(token)
        await asyncio.sleep(0.1)  # let the first budget lapse
        release.set()
        assert await asyncio.wait_for(first, timeout=5) == (first_item, None)
        assert await asyncio.wait_for(second, timeout=5) == (second_item, None)
        assert calls == [1, 1]

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_admission_batch_rejects_lapsed_requests_before_the_transaction() -> None:
    """A request whose budget lapsed while queued must not be inserted.

    It is answered where it waited, so the counter says which queue ate
    the budget, and the rest of its round keeps its own deadline.
    """
    from types import SimpleNamespace

    calls = []
    started = Event()
    release = Event()
    executor = AsyncStoreExecutor(
        workers=2, max_in_flight=4, admission_timeout_seconds=2
    )
    batcher = _admission_batcher(SimpleNamespace(), executor, flush_delay_seconds=0)

    def enqueue_batch(items, **_kwargs):
        calls.append(len(items))
        if len(calls) == 1:
            started.set()
            release.wait(5)
        return [(item, None) for item in items]

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        blocking = SimpleNamespace(cluster_id="a", path="/v1/x")
        lapsed = SimpleNamespace(cluster_id="a", path="/v1/x")
        patient = SimpleNamespace(cluster_id="a", path="/v1/x")
        token = REQUEST_DEADLINE.set(time.monotonic() + 100)
        try:
            first = asyncio.create_task(batcher.submit(blocking))
        finally:
            REQUEST_DEADLINE.reset(token)
        while not started.is_set():
            await asyncio.sleep(0.001)
        token = REQUEST_DEADLINE.set(time.monotonic() + 0.02)
        try:
            doomed = asyncio.create_task(batcher.submit(lapsed))
        finally:
            REQUEST_DEADLINE.reset(token)
        token = REQUEST_DEADLINE.set(time.monotonic() + 100)
        try:
            survivor = asyncio.create_task(batcher.submit(patient))
        finally:
            REQUEST_DEADLINE.reset(token)
        await asyncio.sleep(0.05)
        release.set()
        assert await asyncio.wait_for(first, timeout=5) == (blocking, None)
        with pytest.raises(StoreIoCapacityExceeded):
            await asyncio.wait_for(doomed, timeout=5)
        assert await asyncio.wait_for(survivor, timeout=5) == (patient, None)
        assert batcher.expired_total == 1
        # The lapsed request never reached a transaction.
        assert calls == [1, 1]
        assert batcher.queue_wait_count == 2
        assert batcher.queue_wait_max_seconds > 0
        assert batcher.flush_count == 2
        assert batcher.rounds_total == 2
        assert batcher.pending_depth == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_admission_batch_sheds_on_arrival_when_the_queue_cannot_be_reached() -> None:
    """Shed at the door, not after the budget is gone.

    Burning the full 15s and then answering 503 keeps a slot, a decode
    thread and a queue entry busy for a client that stopped listening.
    """
    from types import SimpleNamespace

    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=2
    )
    batcher = _admission_batcher(
        SimpleNamespace(),
        executor,
        flush_delay_seconds=0,
        max_batch_size=1,
        max_flush_groups=1,
    )
    batcher.store.try_enqueue_processor_requests_batch = lambda items, **_kwargs: [
        (item, None) for item in items
    ]

    async def scenario() -> None:
        item = SimpleNamespace(cluster_id="a", path="/v1/x")
        # No round has run yet, so nothing is projected and nothing sheds.
        # This verifies projection state, not sub-millisecond scheduler latency.
        token = REQUEST_DEADLINE.set(time.monotonic() + 1)
        try:
            assert await batcher.submit(item) == (item, None)
            assert batcher.shed_total == 0
        finally:
            REQUEST_DEADLINE.reset(token)
        # One measured round of 5s ahead of a 1s budget must shed.
        batcher.round_seconds_ewma = 5.0
        token = REQUEST_DEADLINE.set(time.monotonic() + 1)
        try:
            with pytest.raises(StoreIoCapacityExceeded):
                await batcher.submit(item)
        finally:
            REQUEST_DEADLINE.reset(token)
        assert batcher.shed_total == 1
        # A request with room to spare still gets in.
        token = REQUEST_DEADLINE.set(time.monotonic() + 60)
        try:
            assert await batcher.submit(item) == (item, None)
        finally:
            REQUEST_DEADLINE.reset(token)
        assert batcher.shed_total == 1
        # Background callers without a deadline are never shed.
        assert await batcher.submit(item) == (item, None)
        assert batcher.shed_total == 1

    try:
        asyncio.run(scenario())
    finally:
        executor.close()


def test_admission_batch_scope_stall_does_not_delay_another_scope() -> None:
    """One cluster's counter row must not stop every other cluster.

    The flush loop used to select a set of per-cluster groups, await all
    of them with ``asyncio.gather``, and only then look at the queue
    again. Two burst runs of identical code and config differed only in
    that one had a flush sit 132s on its cluster's counter row: round
    time went 3.2s -> 44.9s, queue waits reached 43s, 2,216 requests were
    answered with a deadline rejection they had the budget to survive,
    and acceptance fell from 27,296 to 12,601. The stalled cluster was
    never the one being rejected.
    """
    from types import SimpleNamespace

    executor = AsyncStoreExecutor(
        workers=4, max_in_flight=8, admission_timeout_seconds=5
    )
    batcher = _admission_batcher(SimpleNamespace(), executor, flush_delay_seconds=0)
    stalled = Event()
    release = Event()

    def enqueue_batch(items, **_kwargs):
        if items[0].cluster_id == "slow":
            stalled.set()
            release.wait(10)
        return [(item, None) for item in items]

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        slow = SimpleNamespace(cluster_id="slow", path="/v1/x")
        fast = SimpleNamespace(cluster_id="fast", path="/v1/x")
        stuck = asyncio.create_task(batcher.submit(slow))
        while not stalled.is_set():
            await asyncio.sleep(0.001)
        started = time.monotonic()
        assert await asyncio.wait_for(batcher.submit(fast), timeout=3) == (fast, None)
        # The stalled cluster is still stalled; the point is that this
        # one did not have to wait for it.
        assert time.monotonic() - started < 2
        assert not stuck.done()
        assert batcher.in_flight_max >= 2
        release.set()
        assert await asyncio.wait_for(stuck, timeout=5) == (slow, None)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_admission_batch_keeps_one_flush_and_arrival_order_per_scope() -> None:
    """Ordering inside a cluster is the one thing pipelining must keep.

    Two concurrent batches for one cluster would also take that
    cluster's counter row twice at the same time, which is the convoy a
    batch exists to avoid.
    """
    from threading import Lock
    from types import SimpleNamespace

    executor = AsyncStoreExecutor(
        workers=4, max_in_flight=8, admission_timeout_seconds=5
    )
    batcher = _admission_batcher(
        SimpleNamespace(), executor, flush_delay_seconds=0, max_batch_size=1
    )
    order: list[int] = []
    guard = Lock()
    live = 0
    peak = 0

    def enqueue_batch(items, **_kwargs):
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        try:
            order.extend(item.seq for item in items)
            time.sleep(0.01)
            return [(item, None) for item in items]
        finally:
            with guard:
                live -= 1

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        items = [
            SimpleNamespace(cluster_id="a", path="/v1/x", seq=index)
            for index in range(6)
        ]
        results = await asyncio.gather(*(batcher.submit(item) for item in items))
        assert results == [(item, None) for item in items]
        assert order == [0, 1, 2, 3, 4, 5]
        assert peak == 1

    try:
        asyncio.run(scenario())
    finally:
        executor.close()


def test_admission_batch_rejects_a_stalled_entry_at_its_own_deadline() -> None:
    """A queued entry is answered when *its* budget lapses.

    Under the barrier the sweep only ran between rounds, so an entry
    behind a stalled flush was told its deadline had passed as long after
    the fact as the stall lasted - a 503 for a client that had already
    given up, and a queue slot held for it in the meantime. A scope-mate
    with budget left must still be admitted afterwards.
    """
    from types import SimpleNamespace

    executor = AsyncStoreExecutor(
        workers=4, max_in_flight=8, admission_timeout_seconds=5
    )
    batcher = _admission_batcher(SimpleNamespace(), executor, flush_delay_seconds=0)
    stalled = Event()
    release = Event()

    def enqueue_batch(items, **_kwargs):
        if not stalled.is_set():
            stalled.set()
            release.wait(10)
        return [(item, None) for item in items]

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        blocking = SimpleNamespace(cluster_id="a", path="/v1/x")
        doomed_item = SimpleNamespace(cluster_id="a", path="/v1/x")
        patient_item = SimpleNamespace(cluster_id="a", path="/v1/x")
        token = REQUEST_DEADLINE.set(time.monotonic() + 100)
        try:
            stuck = asyncio.create_task(batcher.submit(blocking))
        finally:
            REQUEST_DEADLINE.reset(token)
        while not stalled.is_set():
            await asyncio.sleep(0.001)
        token = REQUEST_DEADLINE.set(time.monotonic() + 0.2)
        try:
            doomed = asyncio.create_task(batcher.submit(doomed_item))
        finally:
            REQUEST_DEADLINE.reset(token)
        token = REQUEST_DEADLINE.set(time.monotonic() + 100)
        try:
            patient = asyncio.create_task(batcher.submit(patient_item))
        finally:
            REQUEST_DEADLINE.reset(token)
        started = time.monotonic()
        with pytest.raises(StoreIoCapacityExceeded):
            await asyncio.wait_for(doomed, timeout=3)
        # Answered at its own deadline, not when the stall cleared.
        assert time.monotonic() - started < 2
        assert not stuck.done()
        assert not patient.done()
        release.set()
        assert await asyncio.wait_for(stuck, timeout=5) == (blocking, None)
        assert await asyncio.wait_for(patient, timeout=5) == (patient_item, None)
        assert batcher.expired_total == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_admission_batch_holds_the_in_flight_cap_across_scopes() -> None:
    """Removing the barrier must not uncap concurrency against the pool.

    Every in-flight flush holds a ``store_io`` thread and a Postgres
    connection for as long as its transaction runs, so the cap is what
    keeps a fifty-cluster burst from opening fifty of them at once.
    """
    from threading import Lock
    from types import SimpleNamespace

    executor = AsyncStoreExecutor(
        workers=8, max_in_flight=16, admission_timeout_seconds=5
    )
    batcher = _admission_batcher(
        SimpleNamespace(),
        executor,
        flush_delay_seconds=0,
        max_batch_size=1,
        max_flush_groups=2,
    )
    release = Event()
    guard = Lock()
    live = 0
    peak = 0
    entered = Event()

    def enqueue_batch(items, **_kwargs):
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
            if live >= 2:
                entered.set()
        try:
            release.wait(10)
            return [(item, None) for item in items]
        finally:
            with guard:
                live -= 1

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        items = [
            SimpleNamespace(cluster_id=f"c{index}", path="/v1/x") for index in range(5)
        ]
        submissions = [asyncio.create_task(batcher.submit(item)) for item in items]
        while not entered.is_set():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.2)
        assert peak == 2
        assert batcher.in_flight == 2
        assert batcher.cap_waits_total > 0
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*submissions), timeout=10) == [
            (item, None) for item in items
        ]
        assert peak == 2
        assert batcher.in_flight_max == 2

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.close()


def test_metrics_name_the_cluster_whose_admission_flush_is_stalled() -> None:
    """A scrape has to be able to say *which* cluster is stalling.

    Scrapes land on a random uvicorn process, so the aggregate counters
    cannot be diffed back to a cluster - and a 132s flush and a 132s
    queue wait look identical in them. The per-scope series are bounded
    to a handful of rows so cluster count never becomes label
    cardinality.
    """
    from types import SimpleNamespace

    context = ApplicationContext(execution_token="x" * 32)
    app = create_app(context)
    batcher = app.state.processor_admission_batcher
    release = Event()
    stalled = Event()

    def enqueue_batch(items, **_kwargs):
        stalled.set()
        release.wait(10)
        return [(item, None) for item in items]

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        headers = {"X-GPU-Fault-Execution-Token": "x" * 32}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            stuck = asyncio.create_task(
                batcher.submit(SimpleNamespace(cluster_id='we"ird', path="/v1/x"))
            )
            while not stalled.is_set():
                await asyncio.sleep(0.001)
            # Queued behind the stalled flush for the same cluster,
            # which is the only entry a scope-mate can be.
            queued = asyncio.create_task(
                batcher.submit(SimpleNamespace(cluster_id='we"ird', path="/v1/x"))
            )
            await asyncio.sleep(0.05)
            body = (await client.get("/metrics", headers=headers)).text
            assert (
                "gpu_fault_processor_admission_batch_scope"
                '_in_flight_seconds{cluster_id="we\\"ird"}'
            ) in body
            assert (
                "gpu_fault_processor_admission_batch_scope_pending"
                '{cluster_id="we\\"ird"} 1'
            ) in body
            assert (
                "gpu_fault_processor_admission_batch_scope"
                '_wait_seconds{cluster_id="we\\"ird"}'
            ) in body
            release.set()
            assert (await asyncio.wait_for(stuck, timeout=5))[1] is None
            assert (await asyncio.wait_for(queued, timeout=5))[1] is None
            body = (await client.get("/metrics", headers=headers)).text
            assert (
                "gpu_fault_processor_admission_batch_scope"
                '_flush_seconds_max{cluster_id="'
            ) in body

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        app.state.store_io.close()


def test_admission_batch_still_coalesces_arrivals_into_one_transaction() -> None:
    """Pipelining must not turn a batcher into one insert per request.

    Under the barrier a round lasted seconds, so entries piled up behind
    it and every batch was wide for free. An arrival now wakes the
    dispatcher immediately, so the coalescing window is the only thing
    left keeping transactions wide.
    """
    from types import SimpleNamespace

    executor = AsyncStoreExecutor(
        workers=2, max_in_flight=4, admission_timeout_seconds=5
    )
    batcher = _admission_batcher(SimpleNamespace(), executor, flush_delay_seconds=0.05)
    sizes: list[int] = []

    def enqueue_batch(items, **_kwargs):
        sizes.append(len(items))
        return [(item, None) for item in items]

    batcher.store.try_enqueue_processor_requests_batch = enqueue_batch

    async def scenario() -> None:
        items = [SimpleNamespace(cluster_id="a", path="/v1/x") for _ in range(10)]
        assert await asyncio.gather(*(batcher.submit(item) for item in items)) == [
            (item, None) for item in items
        ]
        assert sizes == [10]

    try:
        asyncio.run(scenario())
    finally:
        executor.close()
