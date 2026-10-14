"""A consumer that owns no notification shard is a poller, and says so.

FINAL-建议汇总 F-D11 (P1-14A, P1-75A, P2-75K, P3-75L, P1-28C, P0-38B). The
store's listener only relays payloads for the advisory-locked shard it owns;
with fewer shards than consumer processes the losers get nothing. Their
listener still reported ``on_state(True, None)``, which the coordinator took
as "notifications enabled": the idle ceilings stretched to the 5 s
notification fallback on a process that would never be woken, and
``/metrics`` said notifications were on. A process without a shard never
stretches the fault stream, and a shard owner only while no fault-tier row is
PENDING or LEASED with a lapsed lease (the queue trigger announces only
idle->busy edges, so a priority-0 row queued behind a pending one is never
announced; see test_fault_stream_wake.py); a shard owner still stretches the
routine streams.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
)
from gpu_fault.store import SqliteStore
from tests._builders import build_store, processor_request

AVAILABLE = {"fault": 1, "observation": 0, "gpu": 0, "host": 0}


def _processor(store) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        pools=ProcessorPoolSettings(fault_worker_count=1),
        lease=ProcessorLeaseSettings(
            poll_seconds=0.1,
            fault_idle_backoff_max_seconds=0.5,
            processor_notification_fallback_seconds=5.0,
        ),
        active_consumers=True,
    )


def test_a_process_without_a_shard_reports_notifications_disabled() -> None:
    processor = _processor(build_store())

    processor._set_notification_state(True, None)

    metrics = processor.metrics_snapshot()
    assert metrics["notifications_enabled"] == 0
    assert metrics["notification_listener_connected"] == 1
    assert metrics["notification_shard"] == -1

    processor._set_notification_state(True, 3)
    metrics = processor.metrics_snapshot()
    assert metrics["notifications_enabled"] == 1
    assert metrics["notification_listener_connected"] == 1
    assert metrics["notification_shard"] == 3

    # Losing the connection is a reconnect whether or not a shard was owned.
    processor._set_notification_state(False, None)
    processor._set_notification_state(True, None)
    processor._set_notification_state(False, None)
    metrics = processor.metrics_snapshot()
    assert metrics["notification_reconnects_total"] == 2
    assert metrics["notification_listener_connected"] == 0


def test_a_shardless_process_keeps_the_polling_fault_backoff() -> None:
    processor = _processor(build_store())
    processor._set_notification_state(True, None)

    processor._stream_idle_interval["fault"] = 4.0
    processor._claim_active_by_pool(AVAILABLE, lease_duration=timedelta(seconds=30))

    assert processor._stream_idle_interval["fault"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("fault_backlog", "interval"),
    [pytest.param(True, 0.5, id="backlog"), pytest.param(False, 5.0, id="idle")],
)
def test_a_shard_owner_keeps_fault_idle_only_behind_a_fault_backlog(
    tmp_path, fault_backlog: bool, interval: float
) -> None:
    # A store that answers the fault backlog query (the in-memory one does not).
    store = SqliteStore(str(tmp_path / "shard-owner.db"))
    if fault_backlog:
        # PENDING and deferred: a fault row admitted behind it raises no edge.
        store.enqueue_processor_request(
            processor_request(
                "/v1/attempts/terminal", body=b'{"attempt_id":"attempt-deferred"}'
            ).model_copy(
                update={"not_before": datetime.now(timezone.utc) + timedelta(hours=1)}
            )
        )
    processor = _processor(store)
    processor._set_notification_state(True, 0)

    processor._stream_idle_interval["fault"] = 4.0
    try:
        processor._claim_active_by_pool(AVAILABLE, lease_duration=timedelta(seconds=30))
    finally:
        store.close()

    assert processor._stream_idle_interval["fault"] == pytest.approx(interval)


def test_shardless_transitions_are_counted_once_per_episode() -> None:
    processor = _processor(build_store())

    # The listener re-reports its state every claim attempt while shardless.
    processor._set_notification_state(True, None)
    processor._set_notification_state(True, None)
    processor._set_notification_state(True, None)
    assert processor.metrics_snapshot()["notification_shardless_episodes_total"] == 1

    processor._set_notification_state(True, 2)
    processor._set_notification_state(True, None)
    assert processor.metrics_snapshot()["notification_shardless_episodes_total"] == 2
