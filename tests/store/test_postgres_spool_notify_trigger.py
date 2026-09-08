"""The spool NOTIFY trigger does not compare clocks (E-8).

Control-plane review 2026-09-08. The trigger only notified when
``NEW.available_at <= clock_timestamp()``, comparing the ingress clock that
stamped the row with the database clock; whenever the application clock led,
a brand-new row produced no wakeup and the consumer fell back to its 2-5 s
poll. It now notifies on every insert and on every update that makes the row
available no later than before (coalesce, abandon), and stays quiet when a
claim or release pushes the lease into the future.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import PostgresStore
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    _truncate()
    yield
    _truncate()


def _connections():
    import psycopg

    listener = psycopg.connect(POSTGRES_URL, autocommit=True)
    writer = psycopg.connect(POSTGRES_URL, autocommit=True)
    listener.execute("LISTEN gpu_fault_telemetry_spool")
    return listener, writer


def _notifications(listener, timeout: float = 1.0) -> list[str]:
    return [
        notification.payload
        for notification in listener.notifies(timeout=timeout, stop_after=16)
    ]


def _insert(writer, key: str, available_at: datetime) -> None:
    writer.execute(
        """
        INSERT INTO gpu_fault_telemetry_spool
            (spool_key, cluster_id, path, request_id, available_at,
             created_at, updated_at, payload)
        VALUES (%s, 'cluster-a', '/v1/gpu-metrics', %s, %s, now(), now(), '{}'::jsonb)
        ON CONFLICT (spool_key) DO UPDATE SET
            revision = gpu_fault_telemetry_spool.revision + 1,
            payload = excluded.payload,
            updated_at = now()
        """,
        (key, key, available_at),
    )


def test_a_row_stamped_by_a_leading_clock_still_notifies() -> None:
    listener, writer = _connections()
    try:
        ahead = datetime.now(timezone.utc) + timedelta(minutes=10)
        _insert(writer, "lane-1", ahead)
        payloads = _notifications(listener)
    finally:
        listener.close()
        writer.close()

    assert payloads and all("/v1/gpu-metrics" in item for item in payloads), payloads


def test_a_coalesce_notifies_but_a_lease_into_the_future_does_not() -> None:
    listener, writer = _connections()
    try:
        _insert(writer, "lane-2", datetime.now(timezone.utc))
        _notifications(listener)  # drain the insert's notification
        _insert(writer, "lane-2", datetime.now(timezone.utc))  # coalesce
        coalesced = _notifications(listener)
        writer.execute(
            """
            UPDATE gpu_fault_telemetry_spool
            SET available_at = now() + interval '2 minutes', lease_owner = 'w-1'
            WHERE spool_key = 'lane-2'
            """
        )
        leased = _notifications(listener, timeout=0.5)
        writer.execute(
            """
            UPDATE gpu_fault_telemetry_spool
            SET available_at = now(), lease_owner = NULL
            WHERE spool_key = 'lane-2'
            """
        )
        abandoned = _notifications(listener)
    finally:
        listener.close()
        writer.close()

    assert coalesced, "a coalesced sample must wake the consumer"
    assert leased == [], leased
    assert abandoned, "an abandoned row (available earlier) must wake the consumer"
