from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.gpu_metrics import GpuInventorySnapshot, GpuMetricsIngestionResult
from gpu_fault.store import SqliteStore
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)
from tests.store._cov95_compat_telemetry import CLUSTER, NODE, write_dataset


def test_ingestion_failure_has_explicit_memory_and_sqlite_transaction_semantics(
    compat_store,
):
    original = write_dataset(compat_store)
    later = NOW + timedelta(seconds=5)
    metric = original["metric"].model_copy(update={"observed_at": later})
    inventory = GpuInventorySnapshot.model_validate(
        {
            **original["inventory"].model_dump(),
            "snapshot_id": "later-inventory",
            "observed_at": later,
        }
    )
    result = GpuMetricsIngestionResult(batch_id="failing-batch", accepted_samples=1)
    batch_key = (CLUSTER, NODE, result.batch_id)

    with pytest.raises(RuntimeError, match="local downstream failure"):
        with compat_store.collector_ingestion_transaction(
            CLUSTER, NODE, result.batch_id
        ):
            assert compat_store.observe_gpu_metrics(
                [(original["key"], metric), (original["key"], original["metric"])]
            ) == [original["metric"], False], (
                "a replay inside the ingestion must not replace its newer sample"
            )
            assert compat_store.observe_gpu_inventory_snapshots(
                [inventory, original["inventory"]]
            ) == [original["inventory"], False], (
                "ingestion must observe its newer inventory before a stale replay"
            )
            assert (
                compat_store.save_gpu_inventory_snapshot(original["inventory"])
                == inventory
            ), "the single-snapshot API must expose the uncommitted inventory"
            assert compat_store.update_gpu_findings(
                [
                    (original["key"], original["finding"], NOW),
                    (original["key"], None, later),
                ]
            ) == [False, False], (
                "stale replay and clearing must not activate new findings"
            )
            duplicate = compat_store.save_gpu_metrics_batch(
                (CLUSTER, NODE, original["ingestion"].batch_id),
                original["ingestion"].model_copy(update={"accepted_samples": 99}),
            )
            assert duplicate.duplicate is True and duplicate.accepted_samples == 1, (
                "a duplicate batch result is immutable even inside a failed ingestion"
            )
            compat_store.save_gpu_metrics_batch(batch_key, result)
            raise RuntimeError("local downstream failure")

    transactional = isinstance(compat_store, SqliteStore)
    assert compat_store.list_gpu_metrics_latest(CLUSTER, NODE) == [
        original["metric"] if transactional else metric
    ], "SQLite must roll back; Memory's context intentionally has no rollback guarantee"
    assert compat_store.get_gpu_inventory_snapshot(CLUSTER, NODE) == (
        original["inventory"] if transactional else inventory
    ), "inventory and metric writes must obey the same compatibility-backend boundary"
    assert compat_store.list_gpu_findings(CLUSTER, NODE, active_only=True) == (
        [original["finding"]] if transactional else []
    ), "the finding clear must roll back with its SQLite ingestion"
    assert compat_store.get_gpu_metrics_batch(batch_key) == (
        None if transactional else result
    ), "a rolled-back SQLite ingestion cannot publish its batch completion record"
    assert (
        compat_store.get_gpu_metrics_batch(
            (CLUSTER, NODE, original["ingestion"].batch_id)
        )
        == original["ingestion"]
    ), "the original completed batch must survive either failure"


def test_empty_ingestion_batches_do_not_create_or_clear_existing_telemetry(
    compat_store,
):
    original = write_dataset(compat_store)

    with compat_store.collector_ingestion_transaction(CLUSTER, NODE, "empty-batch"):
        assert compat_store.observe_gpu_metrics([]) == [], (
            "an empty metric batch has no results"
        )
        assert compat_store.observe_gpu_inventory_snapshots([]) == [], (
            "an empty inventory batch is not an empty-device snapshot"
        )
        assert compat_store.update_gpu_findings([]) == [], (
            "an empty finding batch must not clear an existing active episode"
        )
        assert compat_store.save_collector_statuses_batch([]) == [], (
            "an empty status batch must not replace the last collector result"
        )
        assert compat_store.observe_telemetry_metrics([]) == [], (
            "an empty host batch must not erase existing samples"
        )

    assert compat_store.list_gpu_metrics_latest(CLUSTER, NODE) == [
        original["metric"]
    ], "empty ingestion must preserve the existing metric baseline"
    assert (
        compat_store.get_gpu_inventory_snapshot(CLUSTER, NODE) == original["inventory"]
    ), "empty ingestion must preserve the existing GPU inventory"
    assert (
        compat_store.get_gpu_finding_state(original["key"]).finding
        == original["finding"]
    ), "empty ingestion must leave the active finding untouched"
    assert compat_store.list_collector_statuses(CLUSTER, NODE) == [
        original["status"]
    ], "empty ingestion must preserve collector freshness and rejection evidence"
    assert compat_store.list_telemetry_metrics_latest(CLUSTER, NODE) == [
        original["host"]
    ], "empty ingestion must preserve host telemetry"
