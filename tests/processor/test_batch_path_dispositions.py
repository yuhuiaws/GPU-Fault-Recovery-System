"""The batch execution paths book each request on its own (B-4 / B-9, 2026-09-08).

``_process_telemetry_batch`` and ``_process_observation_batch`` completed a
whole batch with whatever status the handler returned - a 500 on one item was
committed as that item's final answer with no retry, no backoff and no entry
in the G1 completion ledger - and a single fenced completion (``None``) made
the handler release and count as error every item in the batch, including the
fifteen the same statement had just committed. Neither path renewed its lease
while the handler ran. These pins use the memory store, whose batch
completion now answers ``None`` per fenced request like Postgres (B-8).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorRequestStatus,
)
from tests._builders import attempt_observation, build_store, processor_request

INVENTORY = "/v1/collector-events/gpu-inventory"
OBSERVATIONS = "/v1/workload-observations"
LEASE = timedelta(seconds=120)


def _processor(store, **overrides) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
        **overrides,
    )


def _claim_inventory(store, count: int):
    for index in range(count):
        store.enqueue_processor_request(
            processor_request(
                INVENTORY,
                body=('{"node_id":"node-%d","gpus":[]}' % index).encode(),
                cluster_id="cluster-a",
            )
        )
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=LEASE, limit=count
    )
    assert len(claimed) == count
    return claimed


class _Response:
    def __init__(self, statuses: dict[str, int], *, delay: float = 0.0):
        self._statuses = statuses
        self._delay = delay
        self.status = 200
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        if self._delay:
            time.sleep(self._delay)
        return json.dumps(
            {
                "results": [
                    {"request_id": request_id, "status": status, "body": {}}
                    for request_id, status in self._statuses.items()
                ]
            }
        ).encode()


def _patch_urlopen(monkeypatch, statuses: dict[str, int], *, delay: float = 0.0):
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda *_args, **_kwargs: _Response(statuses, delay=delay),
    )


def _release_spy(store):
    calls: list[str] = []
    original = store.release_active_processor_request

    def release(request_id, *args, **kwargs):
        calls.append(request_id)
        return original(request_id, *args, **kwargs)

    store.release_active_processor_request = release
    return calls


def test_telemetry_batch_releases_only_the_fenced_item(monkeypatch):
    store = build_store()
    good, fenced = _claim_inventory(store, 2)
    processor = _processor(store)
    _patch_urlopen(monkeypatch, {good.request_id: 200, fenced.request_id: 200})
    releases = _release_spy(store)
    stale = fenced.model_copy(update={"lease_token": "not-the-lease-token"})

    processor._process_telemetry_batch([good, stale])

    assert (
        store.get_processor_request(good.request_id).status
        is ProcessorRequestStatus.COMPLETED
    )
    assert releases == [fenced.request_id], (
        "only the fenced request is released; the committed one is left alone"
    )
    snapshot = processor.metrics_snapshot()
    assert snapshot["processed"] == {"success": 1, "error": 1}


def test_observation_batch_releases_only_the_fenced_item():
    store = build_store()
    now = datetime.now(timezone.utc)
    for attempt in ("attempt-a", "attempt-b"):
        observation = attempt_observation("job-a", attempt, now)
        store.enqueue_processor_request(
            processor_request(
                OBSERVATIONS,
                body=observation.model_dump_json().encode(),
                cluster_id="cluster-a",
            )
        )
    good, fenced = store.claim_active_processor_requests(
        "pod-a:1", now=now, lease_duration=LEASE, limit=2
    )
    processor = _processor(store)
    releases = _release_spy(store)
    stale = fenced.model_copy(update={"lease_token": "not-the-lease-token"})

    processor._process_observation_batch([good, stale])

    assert (
        store.get_processor_request(good.request_id).status
        is ProcessorRequestStatus.COMPLETED
    )
    assert releases == [fenced.request_id]
    assert processor.metrics_snapshot()["processed"] == {"success": 1, "error": 1}


def test_telemetry_batch_retries_a_5xx_item_and_books_the_rest(monkeypatch):
    store = build_store()
    failed, good = _claim_inventory(store, 2)
    processor = _processor(store)
    _patch_urlopen(monkeypatch, {failed.request_id: 500, good.request_id: 200})

    processor._process_telemetry_batch([failed, good])

    retried = store.get_processor_request(failed.request_id)
    assert retried.status is ProcessorRequestStatus.PENDING
    assert retried.retry_count == 1
    assert retried.not_before is not None
    assert retried.not_before > datetime.now(timezone.utc)
    completed = store.get_processor_request(good.request_id)
    assert completed.status is ProcessorRequestStatus.COMPLETED
    assert completed.response_status == 200
    snapshot = processor.metrics_snapshot()
    assert snapshot["retry_rescheduled_total"] == 1
    assert snapshot["retry_rescheduled_by_path"] == {INVENTORY: 1}
    assert snapshot["completions_by_path_status"] == {INVENTORY: {"2xx": 1}}, (
        "the committed completion enters the G1 ledger"
    )
    assert snapshot["processed"] == {"success": 1, "error": 1}


def test_telemetry_batch_completes_a_5xx_item_past_the_retry_horizon(monkeypatch):
    store = build_store()
    [failed] = _claim_inventory(store, 1)
    processor = _processor(
        store, lease=ProcessorLeaseSettings(retryable_response_max_age_seconds=0.001)
    )
    time.sleep(0.01)
    _patch_urlopen(monkeypatch, {failed.request_id: 503})

    processor._process_telemetry_batch([failed])

    completed = store.get_processor_request(failed.request_id)
    assert completed.status is ProcessorRequestStatus.COMPLETED
    assert completed.response_status == 503
    snapshot = processor.metrics_snapshot()
    assert snapshot["retry_horizon_failures_total"] == 1
    assert snapshot["completions_by_path_status"] == {INVENTORY: {"5xx": 1}}
    assert snapshot["retry_rescheduled_total"] == 0


def test_telemetry_batch_renews_every_lease_while_the_handler_runs(monkeypatch):
    store = build_store()
    items = _claim_inventory(store, 2)
    processor = _processor(
        store,
        lease=ProcessorLeaseSettings(
            request_renew_seconds=0.01, request_max_execution_seconds=5
        ),
    )
    renewed: list[str] = []
    original = store.renew_active_processor_request

    def renew(request_id, *args, **kwargs):
        renewed.append(request_id)
        return original(request_id, *args, **kwargs)

    store.renew_active_processor_request = renew
    _patch_urlopen(monkeypatch, {item.request_id: 200 for item in items}, delay=0.15)

    processor._process_telemetry_batch(items)

    assert set(renewed) == {item.request_id for item in items}, renewed
    for item in items:
        assert (
            store.get_processor_request(item.request_id).status
            is ProcessorRequestStatus.COMPLETED
        )


@pytest.mark.parametrize("status", [408, 425, 429])
def test_telemetry_batch_treats_throttling_statuses_as_retryable(monkeypatch, status):
    store = build_store()
    [item] = _claim_inventory(store, 1)
    processor = _processor(store)
    _patch_urlopen(monkeypatch, {item.request_id: status})

    processor._process_telemetry_batch([item])

    current = store.get_processor_request(item.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 1
