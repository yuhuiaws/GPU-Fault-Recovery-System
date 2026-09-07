"""Releasing a claimed request after a failure is a retry, and is booked
like one; a renewal error does not abandon the lease.

FINAL-建议汇总 F-D4 (P0-77D, P0-33C, P1-77E) and F-D5 (P1-77C, P0-33B).
``_release(item)`` put the row back to PENDING with ``created_at`` untouched,
so a deterministic completion failure -- or a ``lane-lease-changed`` fence --
made it the oldest row of its priority and re-executed it immediately, side
effects and all, forever. And the renewal thread returned on the first store
exception while the handler kept running, so a single 40001 let the lane
expire under a live execution.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from threading import Event

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorRequestStatus,
)
from gpu_fault.processor.replay_completion import finalize_replay_response
from tests._builders import build_store, processor_request

REQUEST_LEASE = __import__("datetime").timedelta(seconds=120)


def _claimed(store):
    request = store.enqueue_processor_request(
        processor_request("/v1/workflows/dispatch")
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    return request, claimed


def _processor(store, **overrides) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
        **overrides,
    )


def test_a_release_after_completion_failure_is_booked_as_a_retry(monkeypatch):
    store = build_store()
    request, claimed = _claimed(store)
    processor = _processor(store)

    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(
        store,
        "complete_active_processor_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("persistent")),
    )
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen", lambda *_args, **_kwargs: Response()
    )

    with pytest.raises(RuntimeError, match="persistent"):
        processor._execute(claimed, deadline=time.monotonic() + 10)

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 1
    assert current.not_before is not None
    assert current.not_before > datetime.now(timezone.utc)
    assert processor.metrics_snapshot()["completion_failure_releases_total"] == 1


def test_a_lane_lease_changed_fence_releases_with_a_backoff():
    store = build_store()
    request, claimed = _claimed(store)
    processor = _processor(store)

    finalize_replay_response(
        processor,
        claimed,
        status=409,
        content_type="application/json",
        retry_partition="lane-lease-changed",
        body=b"{}",
        started=time.monotonic(),
    )

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 1
    assert current.not_before is not None


def test_a_release_past_the_retry_horizon_completes_as_failed():
    store = build_store()
    request, claimed = _claimed(store)
    processor = _processor(
        store, lease=ProcessorLeaseSettings(retryable_response_max_age_seconds=0.001)
    )
    # Any measurable age is past a one-millisecond horizon.
    time.sleep(0.01)

    finalize_replay_response(
        processor,
        claimed,
        status=409,
        content_type="application/json",
        retry_partition="lane-lease-changed",
        body=b"{}",
        started=time.monotonic(),
    )

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.COMPLETED
    assert current.response_status is not None and current.response_status >= 500


def test_renewal_survives_a_store_exception_and_stops_only_when_fenced():
    store = build_store()
    _, claimed = _claimed(store)
    processor = _processor(
        store, lease=ProcessorLeaseSettings(request_renew_seconds=0.01)
    )
    calls: list[int] = []
    answers = iter([RuntimeError("40001"), True, True, False])

    def renew(*_args, **_kwargs):
        calls.append(1)
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    store.renew_active_processor_request = renew
    stop = Event()

    processor._renew_active_request(claimed, stop, deadline=time.monotonic() + 5)

    # One exception (continued), two successes, then fenced -> returned.
    assert len(calls) == 4
    snapshot = processor.metrics_snapshot()
    assert snapshot["renewal_errors_total"] == 1
    assert snapshot["renewal_fenced_total"] == 1
