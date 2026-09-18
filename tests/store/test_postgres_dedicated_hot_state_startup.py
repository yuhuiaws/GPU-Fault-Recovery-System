"""Dedicated hot-state mode checks for backfill gaps with EXISTS at startup.

Store review 2026-09-07, item J. Production runs ``hot_state_mode=dedicated``
and uvicorn ``--limit-max-requests`` makes process start routine, so the
startup check must be a yes/no per kind rather than the CLI's counting status
(a three-aggregate LEFT JOIN plus ``count(*)`` of every dedicated table).
"""

from __future__ import annotations

import os
from contextlib import closing

import pytest

from gpu_fault.store import PostgresStore
from tests.store import _postgres_processor_claim_support as shared
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    _truncate()
    PostgresStore(POSTGRES_URL).close()
    yield
    _truncate()


def _insert_legacy_row(kind: str, key: str, payload: str) -> None:
    import psycopg

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO gpu_fault_objects(kind, key, payload)"
                " VALUES (%s, %s, %s::jsonb)",
                (kind, key, payload),
            )


def test_a_dedicated_store_starts_when_no_legacy_row_lacks_its_twin() -> None:
    assert POSTGRES_URL is not None
    store = PostgresStore(
        POSTGRES_URL, initialize_schema=False, hot_state_mode="dedicated"
    )
    try:
        gaps = store.hot_state_backfill_gaps()
    finally:
        store.close()

    assert gaps == {
        "gpu_metric_latest": False,
        "gpu_metrics_batch": False,
        "attempt_observation": False,
        "training_progress": False,
    }


def test_a_dedicated_store_fails_closed_and_names_the_kind_with_a_gap() -> None:
    _insert_legacy_row(
        "gpu_metric_latest",
        "cluster-a/node-a/gpu-0",
        '{"cluster_id":"cluster-a","node_id":"node-a",'
        '"observed_at":"2026-09-07T00:00:00Z"}',
    )

    assert POSTGRES_URL is not None
    with pytest.raises(RuntimeError, match="gpu_metric_latest") as failure:
        PostgresStore(POSTGRES_URL, initialize_schema=False, hot_state_mode="dedicated")

    assert "training_progress" not in str(failure.value)
    # The counting status the CLI prints agrees with the gap the startup saw.
    store = PostgresStore(POSTGRES_URL, initialize_schema=False, hot_state_mode="dual")
    try:
        status = store.hot_state_migration_status()
        gaps = store.hot_state_backfill_gaps()
    finally:
        store.close()
    assert status["gpu_metric_latest"]["missing_or_mismatched"] == 1
    assert gaps == {
        "gpu_metric_latest": True,
        "gpu_metrics_batch": False,
        "attempt_observation": False,
        "training_progress": False,
    }


@pytest.mark.parametrize("schema_ready", [False, True], ids=["first-use", "reused"])
def test_shared_fixture_cleans_legacy_rows_before_dedicated_startup(
    monkeypatch: pytest.MonkeyPatch, schema_ready: bool
) -> None:
    monkeypatch.setattr(shared, "_SCHEMA_READY", schema_ready)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "dedicated")
    _insert_legacy_row(
        "gpu_metric_latest",
        "cluster-a/node-a/prior-test",
        '{"cluster_id":"cluster-a","node_id":"node-a",'
        '"observed_at":"2026-09-07T00:00:00Z"}',
    )
    assert POSTGRES_URL is not None
    with pytest.raises(RuntimeError, match="backfill is incomplete"):
        PostgresStore(POSTGRES_URL, initialize_schema=False, hot_state_mode="dedicated")

    with closing(shared.postgres_store_instance()) as instances:
        store = next(instances)
        assert store.hot_state_mode == "dedicated"
        assert not any(store.hot_state_backfill_gaps().values()), (
            "prior-test legacy rows reached the configured Store startup"
        )
