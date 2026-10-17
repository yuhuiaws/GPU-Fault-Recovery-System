"""The collection-error streak survives on PostgreSQL too (batch 126).

The streak start rides in the booked ``dcgm_field_completeness`` finding, i.e.
in the JSON payload of its ``gpu_finding_state`` row, so PostgreSQL needs no
schema change. These cases prove the payload round-trips through a fresh
connection per batch, and that the API's ingestion path on PostgreSQL opens no
incident for the live single-failure sequence and exactly one at the window.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from datetime import timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.gpu_metrics import GpuMetricBatch
from gpu_fault.store import PostgresStore
from tests._builders import asgi_client
from tests.metrics._collection_error_streak import (
    COMPLETENESS_KEY,
    DEAD_EXPORTER_OFFSETS,
    NOW,
    WINDOW_SECONDS,
    assert_streak_survives_the_store,
    clean_batch,
    erroring_batch,
)
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)


@pytest.fixture
def store() -> Iterator[PostgresStore]:
    yield from postgres_store_instance()


def _post(context: ApplicationContext, batch: GpuMetricBatch) -> dict:
    async def run():
        async with asgi_client(context) as client:
            return await client.post(
                GPU_METRICS_PATH, json=batch.model_dump(mode="json")
            )

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    return response.json()


def _incidents(store: PostgresStore) -> int:
    return sum(store.incident_state_counts().values())


def test_the_streak_start_survives_a_fresh_connection_per_batch(store) -> None:
    connections: list[PostgresStore] = []

    def reconnected() -> PostgresStore:
        if connections:
            connections[-1].close()
        connections.append(PostgresStore(POSTGRES_URL, initialize_schema=False))
        return connections[-1]

    try:
        assert_streak_survives_the_store(reconnected)
    finally:
        connections[-1].close()
    assert len(connections) > len(DEAD_EXPORTER_OFFSETS), "one connection per batch"


def test_one_failed_scrape_then_a_clean_one_opens_nothing_on_postgres(store) -> None:
    context = ApplicationContext(store=store)

    body = _post(context, erroring_batch("dcgm-pg-1", NOW))

    assert body["new_findings"] == [], "the failed scrape alone asks for nothing"
    state = store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is not None, "the gap is booked"
    assert state.finding.automatic_action == "NO_ACTION", state.finding
    _post(context, clean_batch("dcgm-pg-2", NOW + timedelta(seconds=15)))
    state = store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is None, "the clean scrape clears it"
    assert _incidents(store) == 0, "no incident for one failed scrape"
    assert store.list_workflows(limit=10) == [], "and no workflow"
    assert store.list_remote_commands() == [], "and no command"


def test_a_sustained_streak_opens_exactly_one_incident_on_postgres(store) -> None:
    context = ApplicationContext(store=store)

    counts = []
    for offset in (*DEAD_EXPORTER_OFFSETS, 300):
        _post(
            context, erroring_batch(f"dcgm-{offset}", NOW + timedelta(seconds=offset))
        )
        counts.append((offset, _incidents(store)))
    replay = _post(
        context,
        erroring_batch(
            f"dcgm-{WINDOW_SECONDS}", NOW + timedelta(seconds=WINDOW_SECONDS)
        ),
    )

    assert counts == [
        (offset, int(offset >= WINDOW_SECONDS))
        for offset in (*DEAD_EXPORTER_OFFSETS, 300)
    ], counts
    assert replay["duplicate"] is True, "the replayed escalation is recognised"
    assert _incidents(store) == 1, "a replay reaches the same incident"
    incident = store.get_incident_by_event("gpu-dcgm-0-dcgm-fields")
    assert incident is not None, "named after the streak's first batch"
    assert incident.policy_source == "SITE_DCGM_FIELD_COMPLETENESS"
    assert len(store.list_workflows(limit=10)) == 1, "one diagnostic workflow"
