from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request

from gpu_fault.app import create_app
from gpu_fault.app.admission_runtime import max_compressed_request_bytes
from gpu_fault.app.middleware import dispatch
from gpu_fault.app.runtime import ProcessorDispatchState
from gpu_fault.channel_registry import GPU_INVENTORY_PATH
from gpu_fault.processor import (
    ProcessorCompletionSignals,
    ProcessorCoordinator,
    ProcessorRequestStatus,
)
from tests._builders import build_context, build_store, processor_request

UNQUEUED_PATH = "/v1/processor-response-wait-probe"


class Runner:
    """Stand in for an AsyncStoreExecutor without a thread pool."""

    async def run(self, function, *arguments, **keywords):
        return function(*arguments, **keywords)


class Batcher:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    async def submit(self, item):
        self.submitted.append(item)
        return item, "queued"


def _dependencies(store, *, timeout_seconds: float, signals=None) -> Any:
    runner = Runner()
    return dispatch.ProcessorDispatchDependencies(
        context=SimpleNamespace(store=store, execution_token=None),
        processor=SimpleNamespace(
            active_consumers=False,
            is_leader=lambda: True,
            # A processor without the attribute stands for the roles that cannot
            # have one: the ingress role runs no coordinator, and a request
            # completed on another replica is never signalled here.
            **({} if signals is None else {"completion_signals": signals}),
        ),
        state=ProcessorDispatchState(),
        store_io=runner,
        decode_io=runner,
        fault_store_io=runner,
        fault_decode_io=runner,
        processor_admission_batcher=Batcher(),
        fault_admission_batcher=Batcher(),
        evidence_admission_batcher=Batcher(),
        telemetry_spool_batcher=Batcher(),
        processor_replay_tracker=None,
        requires_processor=lambda _request: True,
        replay_authorized=lambda _request: False,
        returns_processor_receipt=lambda _path: False,
        is_fault_ingress_path=lambda _path: False,
        decode_json_body=lambda body, _encoding: (body, {}),
        processor_max_queue_depth=100,
        processor_max_cluster_queue_depth=10,
        processor_fault_reserved_queue_depth=10,
        processor_fault_reserved_cluster_depth=1,
        processor_global_admission_guard=10,
        processor_max_request_bytes=1024,
        processor_retry_after_seconds=2,
        processor_response_timeout_seconds=timeout_seconds,
        processor_queue_bypass_enabled=False,
        processor_queue_bypass_paths=set(),
        processor_admission_rejections={
            "global": 0,
            "cluster": 0,
            "global_reserved": 0,
            "cluster_reserved": 0,
        },
        processor_admission_rejections_by_path={},
        processor_queue_bypasses_by_path={},
        telemetry_spool_enabled=False,
        telemetry_spool_max_item_bytes=1024,
        telemetry_spool_rejections={"global": 0, "cluster": 0},
        telemetry_spool_admitted_by_path={},
        telemetry_request_budget_seconds=5.0,
    )


def _request(body: bytes = b"{}", *, headers=None, receive=None) -> Request:
    header_items = [(b"content-type", b"application/json")]
    for name, value in (headers or {}).items():
        header_items.append((name.lower().encode(), value.encode()))

    async def default_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": UNQUEUED_PATH,
            "raw_path": UNQUEUED_PATH.encode(),
            "query_string": b"",
            "headers": header_items,
            "root_path": "",
            "scheme": "http",
            "server": ("control-plane", 80),
            "client": ("10.0.0.5", 4242),
        },
        receive or default_receive,
    )


async def _no_call_next(_request):
    pytest.fail("the queued request bypassed the processor")


def _record_sleeps(monkeypatch) -> list[float]:
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay):
        # Slept for real: the loop's exit condition is wall-clock, so skipping
        # the wait would spin the poll thousands of times per second.
        slept.append(delay)
        await real_sleep(delay)

    monkeypatch.setattr(dispatch.asyncio, "sleep", sleep)
    return slept


def test_synchronous_wait_polls_with_capped_backoff(monkeypatch) -> None:
    """The store is polled fast at first, then progressively less often.

    A fixed 100ms interval meant one waiter cost ~1150 store reads over the full
    115s budget and every concurrent waiter retried in lockstep, so a burst of
    them kept the store I/O pool answering "not yet". The first poll now lands an
    order of magnitude sooner, and the interval grows to a cap instead.
    """

    polls: list[str] = []
    store = SimpleNamespace(
        get_processor_request=lambda request_id: (
            polls.append(request_id)
            or SimpleNamespace(status=ProcessorRequestStatus.PENDING)
        )
    )
    monkeypatch.setattr(dispatch.random, "uniform", lambda _low, _high: 0.0)
    slept = _record_sleeps(monkeypatch)

    response = asyncio.run(
        dispatch.dispatch_processor_request(
            _request(), _no_call_next, _dependencies(store, timeout_seconds=0.6)
        )
    )

    assert response.status_code == 503
    assert slept[:4] == pytest.approx([0.01, 0.016, 0.0256, 0.04096])
    assert max(slept) <= dispatch.RESPONSE_POLL_MAX_SECONDS
    # Holding the initial interval for the whole 0.6s would cost ~60 reads.
    assert 4 <= len(polls) <= 9, "the interval is not backing off"
    assert len(set(polls)) == 1, "each poll asked for a different request"


def test_synchronous_wait_jitters_every_interval(monkeypatch) -> None:
    """Concurrent waiters must not line up on the same poll instants.

    Without jitter, requests admitted together keep polling together for the
    whole wait, which is exactly the synchronised load the backoff is meant to
    break up.
    """

    store = SimpleNamespace(
        get_processor_request=lambda _request_id: SimpleNamespace(
            status=ProcessorRequestStatus.PENDING
        )
    )
    ranges: list[tuple[float, float]] = []

    def uniform(low, high):
        ranges.append((low, high))
        return high

    monkeypatch.setattr(dispatch.random, "uniform", uniform)
    slept = _record_sleeps(monkeypatch)

    asyncio.run(
        dispatch.dispatch_processor_request(
            _request(), _no_call_next, _dependencies(store, timeout_seconds=0.3)
        )
    )

    assert ranges[0] == (-dispatch.RESPONSE_POLL_JITTER, dispatch.RESPONSE_POLL_JITTER)
    assert len(ranges) == len(slept), "an interval was slept without jitter"
    assert slept[0] == pytest.approx(
        dispatch.RESPONSE_POLL_INITIAL_SECONDS * (1.0 + dispatch.RESPONSE_POLL_JITTER)
    )


def test_synchronous_wait_returns_the_processor_response(monkeypatch) -> None:
    statuses = iter((ProcessorRequestStatus.PENDING, ProcessorRequestStatus.COMPLETED))
    store = SimpleNamespace(
        get_processor_request=lambda _request_id: SimpleNamespace(
            status=next(statuses),
            response_content_type="application/json",
            response_status=201,
            response_body=lambda: b'{"accepted":true}',
        )
    )
    slept = _record_sleeps(monkeypatch)

    response = asyncio.run(
        dispatch.dispatch_processor_request(
            _request(), _no_call_next, _dependencies(store, timeout_seconds=30.0)
        )
    )

    assert response.status_code == 201
    assert response.body == b'{"accepted":true}'
    assert response.headers["Content-Type"] == "application/json"
    assert len(slept) == 1, "the completed response cost more than one backoff step"


def test_a_completion_in_this_process_ends_the_poll_interval(monkeypatch) -> None:
    """The waiter is handed its response instead of discovering it later.

    Both sides live in one process in queued mode, so the backed-off interval was
    pure latency on a response the process itself had already committed: the
    waiter was asleep, not blocked on anything. The interval below is longer than
    the whole test, so only the signal can produce an answer.
    """

    signals = ProcessorCompletionSignals()
    monkeypatch.setattr(dispatch, "RESPONSE_POLL_INITIAL_SECONDS", 30.0)
    monkeypatch.setattr(dispatch.random, "uniform", lambda _low, _high: 0.0)
    finished = threading.Event()
    reads: list[str] = []

    def complete_from_a_worker_thread(request_id: str) -> None:
        time.sleep(0.05)
        finished.set()
        signals.signal(request_id)

    def get_processor_request(request_id: str):
        reads.append(request_id)
        if finished.is_set():
            return SimpleNamespace(
                status=ProcessorRequestStatus.COMPLETED,
                response_content_type="application/json",
                response_status=200,
                response_body=lambda: b'{"done":true}',
            )
        if len(reads) == 1:
            threading.Thread(
                target=complete_from_a_worker_thread, args=(request_id,), daemon=True
            ).start()
        return SimpleNamespace(status=ProcessorRequestStatus.PENDING)

    started = time.monotonic()
    response = asyncio.run(
        dispatch.dispatch_processor_request(
            _request(),
            _no_call_next,
            _dependencies(
                SimpleNamespace(get_processor_request=get_processor_request),
                timeout_seconds=60.0,
                signals=signals,
            ),
        )
    )
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert response.body == b'{"done":true}'
    assert elapsed < dispatch.RESPONSE_POLL_INITIAL_SECONDS / 10
    assert len(reads) == 2, "the response cost more than the read the signal caused"
    assert signals.pending_waiters == 0, "the waiter outlived its request"


def test_a_waiter_that_is_never_signalled_still_polls(monkeypatch) -> None:
    """The signal is an accelerator, never the mechanism.

    A request completed on another replica produces no local signal at all, so
    with the registry in place the poll has to keep running on its own schedule
    and the wait has to end at the same deadline as before.
    """

    monkeypatch.setattr(dispatch.random, "uniform", lambda _low, _high: 0.0)
    reads: list[str] = []
    store = SimpleNamespace(
        get_processor_request=lambda request_id: (
            reads.append(request_id)
            or SimpleNamespace(status=ProcessorRequestStatus.PENDING)
        )
    )
    signals = ProcessorCompletionSignals()

    response = asyncio.run(
        dispatch.dispatch_processor_request(
            _request(),
            _no_call_next,
            _dependencies(store, timeout_seconds=0.3, signals=signals),
        )
    )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"
    assert len(reads) >= 4, "the interval stopped advancing without a signal"
    assert signals.pending_waiters == 0


def test_the_waiter_cap_refuses_registration_rather_than_growing() -> None:
    """Refusing a waiter costs it the latency it had; keeping it costs memory.

    This is per-request state reached from the request path, and the admission
    depth bounds what is queued, not how many callers are waiting at once.
    """

    signals = ProcessorCompletionSignals(max_waiters=1)

    async def two_waiters() -> tuple[Any, Any, int]:
        with signals.waiting("first") as first, signals.waiting("second") as second:
            return first, second, signals.pending_waiters

    first, second, pending = asyncio.run(two_waiters())

    assert first is not None
    assert second is None, "the cap admitted a waiter beyond it"
    assert (pending, signals.refused_total) == (1, 1)
    assert signals.pending_waiters == 0, "a refused waiter left state behind"

    with pytest.raises(ValueError, match="must be positive"):
        ProcessorCompletionSignals(max_waiters=0)


def test_a_signal_arrives_only_after_the_response_is_committed() -> None:
    """What the wake promises: the next read finds the response.

    A signal raised before the store write would wake the waiter into another
    "not yet" and leave it waiting out the interval it was supposed to skip. This
    drives a real terminal path of the coordinator -- the stale-superseded
    completion -- and reads the store at the instant the wake arrives.
    """

    store = build_store()
    stale = processor_request(
        GPU_INVENTORY_PATH,
        body=(
            '{"node_id":"node-a","observed_at":"'
            + (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            + '"}'
        ).encode(),
    )
    store.enqueue_processor_request(stale)
    claimed_at = datetime.now(timezone.utc)
    leadership = store.acquire_processor_leadership(
        "pod-a", now=claimed_at, lease_duration=timedelta(minutes=1)
    )
    claimed = store.claim_processor_requests(
        "pod-a",
        leadership.epoch,
        now=claimed_at,
        lease_duration=timedelta(seconds=120),
        limit=1,
    )[0]
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a",
        internal_token="completion-signal-token",
        active_consumers=False,
    )

    async def wait_for_the_signal() -> ProcessorRequestStatus:
        with processor.completion_signals.waiting(claimed.request_id) as wake:
            assert wake is not None
            worker = threading.Thread(
                target=processor._complete_if_stale, args=(claimed,), daemon=True
            )
            worker.start()
            await asyncio.wait_for(wake.wait(), timeout=10.0)
            # Read before the join: the claim is that the response is there when
            # the wake arrives, not merely once the worker has finished.
            status = store.get_processor_request(claimed.request_id).status
            worker.join(timeout=10.0)
            return status

    assert asyncio.run(wait_for_the_signal()) is ProcessorRequestStatus.COMPLETED


def test_declared_oversize_body_is_rejected_before_it_is_read() -> None:
    """A 413 must not require buffering the body the caller announced.

    The size check ran after ``await request.body()``, so announcing 200MB got
    200MB read and JSON-decoded first, and concurrent oversize callers each held
    that memory until the same rejection came back.
    """

    async def receive():
        pytest.fail("the oversize body was read before the size check")

    dependencies = _dependencies(SimpleNamespace(), timeout_seconds=30.0)
    request = _request(headers={"Content-Length": "4096"}, receive=receive)

    response = asyncio.run(
        dispatch.dispatch_processor_request(request, _no_call_next, dependencies)
    )

    assert response.status_code == 413
    assert dependencies.state.oversize_rejections == 1


def test_compressed_body_is_sized_after_decoding() -> None:
    """A compressed wire length is not a bound on the decoded size.

    gzip output can be marginally larger than its input, so rejecting a
    compressed body at ``processor_max_request_bytes`` would deny a legal request
    whose decoded body fits. Anything up to
    ``max_compressed_request_bytes`` therefore reaches the decoder, which
    enforces the limit on what it actually produced. See
    ``test_declared_oversize_compressed_body_is_rejected_before_it_is_read`` for
    the other side: past that bound the length check does apply, because no
    conforming stream could decode within budget.
    """

    dependencies = _dependencies(SimpleNamespace(), timeout_seconds=30.0)
    over_the_plain_limit = str(dependencies.processor_max_request_bytes + 1)
    assert int(over_the_plain_limit) <= max_compressed_request_bytes(
        dependencies.processor_max_request_bytes
    )
    request = _request(
        body=b"{}",
        headers={"Content-Length": over_the_plain_limit, "Content-Encoding": "gzip"},
    )
    decoded: list[str] = []
    dependencies = dispatch.ProcessorDispatchDependencies(
        **{
            **dependencies.__dict__,
            "decode_json_body": lambda body, encoding: (
                decoded.append(encoding) or (body, {})
            ),
            "returns_processor_receipt": lambda _path: True,
        }
    )

    response = asyncio.run(
        dispatch.dispatch_processor_request(request, _no_call_next, dependencies)
    )

    assert response.status_code == 202
    assert decoded == ["gzip"]
    assert dependencies.state.oversize_rejections == 0


def test_declared_oversize_compressed_body_is_rejected_before_it_is_read() -> None:
    """The pre-buffer length check applies to compressed bodies too.

    It used to return early for anything carrying a ``Content-Encoding``, so a
    caller announcing 200MB of gzip got 200MB buffered and only then measured —
    the exact hole ``test_declared_oversize_body_is_rejected_before_it_is_read``
    closed for plain bodies.
    """

    async def receive():
        pytest.fail("the oversize compressed body was read before the size check")

    dependencies = _dependencies(SimpleNamespace(), timeout_seconds=30.0)
    beyond_any_conforming_stream = max_compressed_request_bytes(
        dependencies.processor_max_request_bytes
    )
    request = _request(
        headers={
            "Content-Length": str(beyond_any_conforming_stream + 1),
            "Content-Encoding": "gzip",
        },
        receive=receive,
    )

    response = asyncio.run(
        dispatch.dispatch_processor_request(request, _no_call_next, dependencies)
    )

    assert response.status_code == 413
    assert dependencies.state.oversize_rejections == 1


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_response_timeout_fails_at_startup(monkeypatch, value: str) -> None:
    """The timeout is parsed once, at startup, not per request.

    Reading and parsing it inside the wait loop meant a typo in the ConfigMap
    started a healthy-looking Pod whose every synchronous request answered 500.
    """

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_RESPONSE_TIMEOUT_SECONDS", value)

    with pytest.raises(RuntimeError, match="response timeout are invalid"):
        create_app(build_context())


def test_unparsable_response_timeout_fails_at_startup(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_RESPONSE_TIMEOUT_SECONDS", "two minutes")

    with pytest.raises(ValueError, match="two minutes"):
        create_app(build_context())
