from __future__ import annotations

from contextlib import closing

import pytest

from gpu_fault.store import InMemoryStore, NotFoundError, SqliteStore
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_telemetry import (
    CLUSTER,
    NODE,
    assert_persisted_dataset,
    write_dataset,
)


def test_sqlite_reconstruction_restores_records_but_not_its_in_memory_spool(tmp_path):
    path = str(tmp_path / "reconstructed.db")
    with closing(SqliteStore(path)) as first:
        expected = write_dataset(first)
        assert_persisted_dataset(first, expected)
        assert first.telemetry_spool_stats(now=NOW)["depth"] == 1
    with closing(SqliteStore(path)) as restarted:
        assert_persisted_dataset(restarted, expected)
        assert restarted.telemetry_spool_stats(now=NOW)["depth"] == 0
        assert restarted.processor_queue_stats(now=NOW)["depth"] == 1
        assert (
            restarted.list_training_progress_states(CLUSTER)[0].last_progress_at == NOW
        )
        assert (
            restarted.list_attempt_observation_states(CLUSTER)[0].first_observed_at
            == NOW
        )


def test_memory_records_are_instance_local_not_a_durable_backend():
    first, replacement = InMemoryStore(), InMemoryStore()
    expected = write_dataset(first)
    assert_persisted_dataset(first, expected)
    assert replacement.list_gpu_metrics_latest(CLUSTER, NODE) == []
    assert replacement.get_gpu_inventory_snapshot(CLUSTER, NODE) is None
    assert replacement.get_gpu_metrics_batch((CLUSTER, NODE, "compat-batch")) is None
    assert replacement.get_gpu_finding_state(expected["key"]) is None
    assert replacement.list_collector_statuses(CLUSTER) == []
    assert replacement.get_collector_metrics_snapshot() is None
    assert replacement.list_telemetry_metrics_latest(CLUSTER, NODE) == []
    assert replacement.list_training_progress(CLUSTER) == []
    assert replacement.list_attempt_observations(CLUSTER) == []
    assert replacement.telemetry_spool_stats(now=NOW)["depth"] == 0
    with pytest.raises(NotFoundError):
        replacement.get_processor_request("compat-queued")
    assert_persisted_dataset(first, expected)
