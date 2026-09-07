"""The fault <-> observation interlock gets a liveness probe and a designed bound.

FINAL-建议汇总 F-D3 (P1-76B, P0-17A, P1-13E, P0-76B). A correlated fault row
waits while a workload observation on the same scope is PENDING or LEASED.
Nothing counted how many fault rows were being held back, and the only bound
on the wait was the 300 s retry horizon of the observation row - a number that
belongs to "how long do we retry a 5xx", not to "how long may an observation
hold faults back". Here the coordinator counts what the store reports as held
back, and an observation's retry horizon is bounded by its own stale limit.

The interlock predicate itself (``lease_expires_at > now()`` for LEASED
observation rows) lives in ``store/shared/processor_helpers.py`` and
``store/postgres/processor_claims.py``; the last test pins its absence and is
marked strict-xfail until that change lands.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
    ProcessorRequestStatus,
    ProcessorStaleSettings,
)
from gpu_fault.processor.replay_completion import finalize_replay_response
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import build_store, copy_model, processor_request

REQUEST_LEASE = timedelta(seconds=120)
OWNER = "pod-a:1"
FAULT_PATH = "/v1/collector-events/nvidia-kernel"
OBSERVATION_PATH = "/v1/workload-observations"


def _processor(store, **overrides) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token="processor-token",
        active_consumers=True,
        pools=ProcessorPoolSettings(fault_worker_count=1),
        **overrides,
    )


class _ProbingStore(InMemoryStore):
    """A store that can say how many fault rows the interlock is holding."""

    def __init__(self) -> None:
        super().__init__()
        self.probe_calls: list[datetime] = []
        self.blocked = 3

    def count_fault_rows_blocked_by_observation(self, *, now: datetime) -> int:
        self.probe_calls.append(now)
        return self.blocked


def _empty_fault_claim(processor: ProcessorCoordinator) -> None:
    processor._stream_idle_until.clear()
    processor._claim_active_by_pool(
        {"fault": 1, "observation": 0, "gpu": 0, "host": 0},
        lease_duration=timedelta(seconds=30),
    )


def test_fault_rows_held_back_by_observations_are_counted() -> None:
    store = _ProbingStore()
    processor = _processor(store)

    _empty_fault_claim(processor)

    metrics = processor.metrics_snapshot()
    assert len(store.probe_calls) == 1
    assert metrics["fault_rows_skipped_by_observation_total"] == 3
    assert metrics["fault_rows_blocked_by_observation"] == 3
    assert metrics["interlock_probes_total"] == 1


def test_the_interlock_probe_is_rate_limited() -> None:
    store = _ProbingStore()
    processor = _processor(store)

    _empty_fault_claim(processor)
    _empty_fault_claim(processor)
    assert len(store.probe_calls) == 1, (
        "a second empty claim within the window re-asked"
    )

    processor._next_interlock_probe_at = 0.0
    store.blocked = 2
    _empty_fault_claim(processor)

    metrics = processor.metrics_snapshot()
    assert len(store.probe_calls) == 2
    assert metrics["fault_rows_skipped_by_observation_total"] == 5
    assert metrics["fault_rows_blocked_by_observation"] == 2


class _WithoutProbe:
    """A store from before the probe existed: every other attribute passes through."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        if name == "count_fault_rows_blocked_by_observation":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_a_store_without_the_probe_is_left_alone() -> None:
    store = _WithoutProbe(build_store())
    processor = _processor(store)

    _empty_fault_claim(processor)

    metrics = processor.metrics_snapshot()
    assert metrics["fault_rows_skipped_by_observation_total"] == 0
    assert metrics["interlock_probes_total"] == 0


def _claimed_aged(store, path: str, *, age_seconds: float, body: bytes):
    request = copy_model(
        processor_request(path, body=body),
        created_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )
    store.enqueue_processor_request(request)
    claimed = store.claim_active_processor_requests(
        OWNER, now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    return request, claimed


_OBSERVATION_BODY = (
    b'{"job_id":"training-job","attempt_id":"training-job-a001",'
    b'"containers":[{"node_id":"node-a"}]}'
)


def test_an_observation_past_its_stale_limit_stops_retrying() -> None:
    """130 s old, stale at 120 s, generic horizon 300 s: finished, not retried."""

    store = build_store()
    request, claimed = _claimed_aged(
        store, OBSERVATION_PATH, age_seconds=130, body=_OBSERVATION_BODY
    )
    processor = _processor(
        store,
        lease=ProcessorLeaseSettings(retryable_response_max_age_seconds=300),
        stale=ProcessorStaleSettings(observation_stale_seconds=120),
    )

    finalize_replay_response(
        processor,
        claimed,
        status=503,
        content_type="application/json",
        retry_partition=None,
        body=b'{"detail":"writer is read-only"}',
        started=time.monotonic(),
    )

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.COMPLETED
    assert current.response_status == 503
    assert processor.metrics_snapshot()["retry_horizon_failures_total"] == 1


def test_a_telemetry_row_of_the_same_age_keeps_the_generic_horizon() -> None:
    store = build_store()
    request, claimed = _claimed_aged(
        store,
        "/v1/collector-events/gpu-metrics",
        age_seconds=130,
        body=b'{"node_id":"node-a"}',
    )
    processor = _processor(
        store,
        lease=ProcessorLeaseSettings(retryable_response_max_age_seconds=300),
        stale=ProcessorStaleSettings(observation_stale_seconds=120),
    )

    finalize_replay_response(
        processor,
        claimed,
        status=503,
        content_type="application/json",
        retry_partition=None,
        body=b"{}",
        started=time.monotonic(),
    )

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 1


def test_a_completion_failure_release_uses_the_observation_horizon(monkeypatch) -> None:
    store = build_store()
    request, claimed = _claimed_aged(
        store, OBSERVATION_PATH, age_seconds=130, body=_OBSERVATION_BODY
    )
    processor = _processor(
        store,
        lease=ProcessorLeaseSettings(retryable_response_max_age_seconds=300),
        stale=ProcessorStaleSettings(observation_stale_seconds=120),
    )

    processor._release(claimed, failure="completion failed: RuntimeError")

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.COMPLETED
    assert current.response_status == 503


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "interlock.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    from tests.store._postgres_processor_claim_support import (
        _truncate,
        postgres_store_instance,
    )

    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def test_expired_leased_observation_does_not_block_correlated_fault(store) -> None:
    """A crashed owner's LEASED observation must stop shadowing the fault
    once its lease has expired, not once somebody happens to reclaim it."""

    fault = processor_request(FAULT_PATH, body=b'{"node_id":"node-a"}')
    observation = processor_request(OBSERVATION_PATH, body=_OBSERVATION_BODY)
    assert set(fault.correlation_scope_keys) & set(observation.correlation_scope_keys)
    store.enqueue_processor_request(observation)
    store.enqueue_processor_request(fault)
    now = datetime.now(timezone.utc)

    first = store.claim_active_processor_requests(
        "pod-crashed:1",
        now=now,
        lease_duration=timedelta(seconds=1),
        limit=1,
        include_paths={OBSERVATION_PATH},
    )
    assert [item.request_id for item in first] == [observation.request_id]
    assert (
        store.claim_active_processor_requests(
            OWNER,
            now=now,
            lease_duration=REQUEST_LEASE,
            limit=1,
            include_paths={FAULT_PATH},
        )
        == []
    )

    after_expiry = store.claim_active_processor_requests(
        OWNER,
        now=now + timedelta(seconds=2),
        lease_duration=REQUEST_LEASE,
        limit=1,
        include_paths={FAULT_PATH},
    )
    assert [item.request_id for item in after_expiry] == [fault.request_id]
