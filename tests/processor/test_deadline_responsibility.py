"""An execution deadline is the request's fault first, the process's only
after a pattern; and the unhealthy latch lets go.

FINAL-建议汇总 F-D7 (P0-77B, P0-34A, P1-77G). One request past its deadline
latched ``_unhealthy_reason`` for the life of the process, which gated every
background service and -- through ``on_unhealthy`` -- ``os._exit(70)``. One
poisoned request therefore restarted the whole worker fleet, forever, while
the request itself went back to the head of its lane untouched.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from gpu_fault.processor import ProcessorCoordinator, ProcessorRequestStatus
from tests._builders import build_store, processor_request

REQUEST_LEASE = timedelta(seconds=120)


def _claim(store, path: str = "/v1/workflows/dispatch"):
    request = store.enqueue_processor_request(processor_request(path))
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    return request, claimed


def _processor(store, unhealthy, **overrides) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
        on_unhealthy=unhealthy.append,
        **overrides,
    )


def test_one_deadline_charges_the_request_not_the_process():
    store = build_store()
    unhealthy: list[str] = []
    processor = _processor(store, unhealthy)
    request, claimed = _claim(store)

    processor._mark_execution_deadline_exceeded(claimed)

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 1
    assert current.not_before is not None
    assert processor.is_healthy(), "expected processor.is_healthy() to be true"
    assert unhealthy == []
    assert processor.metrics_snapshot()["deadline_exceeded_total"] == 1


def test_the_process_gives_up_only_after_distinct_requests_keep_timing_out():
    store = build_store()
    unhealthy: list[str] = []
    processor = _processor(store, unhealthy, deadline_exceeded_process_threshold=3)
    claims = []
    for index in range(3):
        # Distinct nodes -> distinct lanes, so all three can be claimed at once.
        request = store.enqueue_processor_request(
            processor_request(
                "/v1/gpu-events/xid",
                body=('{"node_id":"node-' + str(index) + '","xid":79}').encode(),
            )
        )
        claims.append(
            (
                request,
                store.claim_active_processor_requests(
                    "pod-a:1",
                    now=datetime.now(timezone.utc),
                    lease_duration=REQUEST_LEASE,
                    limit=1,
                )[0],
            )
        )
    first, second, third = (claimed for _, claimed in claims)

    processor._mark_execution_deadline_exceeded(first)
    processor._mark_execution_deadline_exceeded(first)  # same request: no double count
    processor._mark_execution_deadline_exceeded(second)
    assert processor.is_healthy(), "expected processor.is_healthy() to be true"
    assert unhealthy == []

    processor._mark_execution_deadline_exceeded(third)

    assert not processor.is_healthy(), "expected processor.is_healthy() to be false"
    assert unhealthy == ["processor request execution deadline exceeded"]


def test_the_unhealthy_latch_expires():
    store = build_store()
    unhealthy: list[str] = []
    processor = _processor(
        store,
        unhealthy,
        deadline_exceeded_process_threshold=1,
        unhealthy_ttl_seconds=0.05,
    )
    _, claimed = _claim(store)

    processor._mark_execution_deadline_exceeded(claimed)
    assert not processor.is_healthy(), "expected processor.is_healthy() to be false"
    time.sleep(0.1)

    assert processor.is_healthy(), "expected processor.is_healthy() to be true"
    assert processor.unhealthy_reason is None
