"""Serial public Store contracts against the allocated, guarded PostgreSQL 16."""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.gpu_metric_models import GpuHealthSeverity
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from gpu_fault.telemetry_models import TelemetryMetricLatest
from tests.store import _cov95_runtime_postgres as postgres
from tests.store._cov95_runtime_models import (
    CLUSTER,
    NODE,
    NOW,
    finding,
    ingestion,
    inventory,
    metric,
    metric_key,
    observation,
    progress,
    terminal,
)
from tests.store._cov95_runtime_postgres import peer_store

pg_store = postgres.pg_store


def test_gpu_batch_result_is_immutable_on_duplicate_retry(pg_store):
    key = (CLUSTER, NODE, "batch-example")
    original = ingestion()
    assert pg_store.get_gpu_metrics_batch(key) is None, "new batch unexpectedly exists"
    assert pg_store.save_gpu_metrics_batch(key, original) == original
    duplicate = pg_store.save_gpu_metrics_batch(key, ingestion(count=99))
    assert duplicate.duplicate is True, "retry was not identified as duplicate"
    assert duplicate.accepted_samples == 1, (
        "retry replaced the committed ingestion result"
    )
    assert pg_store.get_gpu_metrics_batch(key) == original
    assert pg_store.get_gpu_metrics_batch((CLUSTER, "other-node", key[2])) is None


def test_gpu_samples_keep_the_newest_value_within_and_across_batches(pg_store):
    first, newer = metric(), metric(71.0, 1)
    key = metric_key(first)
    assert pg_store.observe_gpu_metrics([]) == []
    assert pg_store.observe_gpu_metrics([(key, first), (key, newer), (key, first)]) == [
        None,
        first,
        False,
    ]
    assert pg_store.observe_gpu_metric(key, newer) is False, (
        "exact retry changed a sample"
    )
    assert pg_store.list_gpu_metrics_latest(CLUSTER, NODE) == [newer]
    assert pg_store.list_gpu_metrics_latest(CLUSTER, "other-node") == []


@pytest.mark.parametrize("pg_store", ["dual", "dedicated"], indirect=True)
def test_gpu_native_batch_refuses_mixed_cluster_node_scope(pg_store):
    first, other = metric(), metric(node="other-node")
    with pytest.raises(ValueError, match="one cluster/node"):
        pg_store.observe_gpu_metrics(
            [(metric_key(first), first), (metric_key(other), other)]
        )
    assert pg_store.list_gpu_metrics_latest(CLUSTER, NODE) == []
    assert pg_store.list_gpu_metrics_latest(CLUSTER, "other-node") == []


def test_gpu_inventory_is_monotonic_and_preserves_result_order(pg_store):
    first, newer = inventory(), inventory(1)
    assert pg_store.observe_gpu_inventory_snapshots([]) == []
    assert pg_store.observe_gpu_inventory_snapshots([first, newer, first]) == [
        None,
        first,
        False,
    ]
    assert pg_store.save_gpu_inventory_snapshot(first) == newer
    assert pg_store.get_gpu_inventory_snapshot(CLUSTER, NODE) == newer


def test_gpu_finding_activation_history_and_clear_are_distinct(pg_store):
    key = metric_key(metric())
    warning, repeated, critical = (
        finding(),
        finding(1),
        finding(2, severity=GpuHealthSeverity.CRITICAL),
    )
    assert pg_store.get_gpu_finding_states([]) == []
    assert pg_store.update_gpu_findings([]) == []
    assert pg_store.get_gpu_finding_state(key) is None
    assert pg_store.update_gpu_finding(key, warning, warning.observed_at) is True
    assert pg_store.update_gpu_finding(key, warning, warning.observed_at) is False
    assert pg_store.update_gpu_finding(key, repeated, repeated.observed_at) is False
    assert pg_store.get_gpu_finding_state(key).consecutive_breaches == 2
    assert pg_store.list_gpu_findings(CLUSTER, NODE, active_only=True) == [repeated]
    assert pg_store.update_gpu_finding(key, critical, critical.observed_at) is True
    assert pg_store.update_gpu_finding(key, None, NOW + timedelta(seconds=3)) is False
    assert pg_store.get_gpu_finding_state(key).consecutive_breaches == 0
    assert pg_store.list_gpu_findings(CLUSTER, NODE, active_only=True) == []
    assert {
        item.finding_id
        for item in pg_store.list_gpu_findings(CLUSTER, NODE, active_only=False)
    } == {warning.finding_id, critical.finding_id}
    assert pg_store.get_gpu_finding_states(
        [(CLUSTER, NODE, "GPU-missing", "temperature")]
    ) == [None]


def test_gpu_finding_batch_refuses_mixed_scope_before_any_write(pg_store):
    first, other = finding(), finding(node="other-node")
    with pytest.raises(ValueError, match="one cluster/node"):
        pg_store.update_gpu_findings(
            [
                (metric_key(metric()), first, NOW),
                (metric_key(metric(node="other-node")), other, NOW),
            ]
        )
    assert pg_store.list_gpu_findings(CLUSTER, NODE, active_only=False) == []


def test_training_progress_tracks_advancement_separately_from_freshness(pg_store):
    first, unchanged, advanced = progress(), progress(1), progress(2, step=2)
    assert pg_store.observe_training_progress(first) is None
    assert pg_store.observe_training_progress(first) is False
    assert pg_store.observe_training_progress(unchanged) == first
    state = pg_store.list_training_progress_states(CLUSTER, first.attempt_id)[0]
    assert state.last_progress_at == first.observed_at, (
        "unchanged step reset progress age"
    )
    assert pg_store.observe_training_progress(advanced) == unchanged
    assert pg_store.list_training_progress(CLUSTER) == [advanced]
    assert (
        pg_store.list_training_progress_states(CLUSTER)[0].last_progress_at
        == advanced.observed_at
    )
    assert pg_store.list_training_progress_states(CLUSTER, "missing-attempt") == []


def test_attempt_observations_reject_stale_samples_and_obey_scan_bounds(pg_store):
    first, newer = observation(), observation(2)
    assert pg_store.save_attempt_observations_batch([]) == []
    assert pg_store.save_attempt_observation(first) is True
    assert pg_store.save_attempt_observation(newer) is True
    assert pg_store.save_attempt_observation(first) is False
    other = observation(1, attempt="attempt-other", cluster="other-cluster")
    assert pg_store.save_attempt_observation(other) is True
    current = pg_store.list_attempt_observation_states(
        CLUSTER, limit=1, newest_first=True
    )
    assert [item.observation for item in current] == [newer]
    assert current[0].first_observed_at == first.observed_at
    assert pg_store.list_attempt_observations(CLUSTER) == [newer]
    assert pg_store.list_attempt_observation_states(CLUSTER, limit=0) == []
    assert [
        item.observation
        for item in pg_store.list_attempt_observation_states(
            limit=2, newest_first=False
        )
    ] == [other, newer]


def test_terminal_event_prevents_late_running_observation_from_resurrecting_attempt(
    pg_store,
):
    current = observation()
    assert pg_store.save_attempt_observation(current) is True
    event = terminal()
    assert pg_store.save_event_if_absent(event) is True
    assert pg_store.save_attempt_observation(observation(3)) is False
    stored = pg_store.list_attempt_observation_states(CLUSTER)[0].observation
    assert stored.workload_phase.value == "SUCCEEDED"
    assert pg_store.save_attempt_observation(observation(4)) is False


@pytest.mark.parametrize(
    "pg_store,legacy_visible",
    [("dual", True), ("dedicated", False)],
    indirect=["pg_store"],
)
def test_hot_readers_handle_legacy_only_rows_without_losing_dedicated_authority(
    pg_store, legacy_visible
):
    key = metric_key(metric())
    batch_key = (CLUSTER, NODE, "batch-example")
    with peer_store("legacy") as legacy:
        legacy.observe_gpu_metric(key, metric())
        legacy.save_gpu_metrics_batch(batch_key, ingestion())
        legacy.observe_training_progress(progress())
        legacy.save_attempt_observation(observation())
    assert pg_store.get_gpu_metrics_batch(batch_key) == (
        ingestion() if legacy_visible else None
    )
    assert pg_store.list_gpu_metrics_latest(CLUSTER, NODE) == (
        [metric()] if legacy_visible else []
    )
    assert pg_store.list_training_progress(CLUSTER, "attempt-example") == (
        [progress()] if legacy_visible else []
    )
    assert [
        item.observation
        for item in pg_store.list_attempt_observation_states(CLUSTER, limit=1)
    ] == ([observation()] if legacy_visible else [])
    newer = metric(72.0, 2)
    assert pg_store.observe_gpu_metric(key, newer) == (
        metric() if legacy_visible else None
    )
    assert pg_store.observe_training_progress(progress(2, step=2)) == (
        progress() if legacy_visible else None
    )
    assert pg_store.save_attempt_observation(observation(2)) is True
    assert pg_store.list_gpu_metrics_latest(CLUSTER, NODE) == [newer]
    with peer_store("legacy") as legacy:
        assert legacy.list_gpu_metrics_latest(CLUSTER, NODE) == (
            [newer] if legacy_visible else [metric()]
        )


def test_collector_status_batches_reject_stale_updates_and_keep_rejection_evidence(
    pg_store,
):
    first = CollectorStatus(
        cluster_id=CLUSTER,
        node_id=NODE,
        collector=CollectorKind.GPU_METRICS,
        observed_at=NOW,
        ingested_at=NOW,
        last_error_at=NOW,
        errors=["rejected-event:example"],
    )
    newer = first.model_copy(
        update={"observed_at": NOW + timedelta(seconds=1), "errors": []}
    )
    assert pg_store.save_collector_statuses_batch([]) == []
    assert pg_store.save_collector_statuses_batch([first, newer, first]) == [
        True,
        True,
        False,
    ]
    assert pg_store.save_collector_statuses_batch([first]) == [False]
    expected = newer.model_copy(update={"errors": first.errors})
    assert pg_store.list_collector_statuses(CLUSTER, NODE) == [expected]
    assert pg_store.list_collector_statuses(CLUSTER) == [expected]
    assert pg_store.list_collector_statuses(CLUSTER, "other-node") == []


def test_host_telemetry_batches_are_monotonic_and_node_scoped(pg_store):
    first = TelemetryMetricLatest(
        cluster_id=CLUSTER, node_id=NODE, observed_at=NOW, name="load", value=1.0
    )
    newer = first.model_copy(
        update={"observed_at": NOW + timedelta(seconds=1), "value": 2.0}
    )
    assert pg_store.observe_telemetry_metrics([]) == []
    assert pg_store.observe_telemetry_metrics([first, newer, first]) == [
        True,
        True,
        False,
    ]
    assert pg_store.observe_telemetry_metrics([first]) == [False]
    assert pg_store.list_telemetry_metrics_latest(CLUSTER, NODE) == [newer]
    assert pg_store.list_telemetry_metrics_latest(CLUSTER, "other-node") == []


@pytest.mark.parametrize("pg_store", ["dedicated"], indirect=True)
def test_closed_store_reports_observation_flush_failure(pg_store):
    from psycopg_pool import PoolClosed

    pg_store.close()
    with pytest.raises(PoolClosed, match="closed"):
        pg_store.save_attempt_observation(observation())
    with peer_store("dedicated") as peer:
        assert peer.list_attempt_observation_states(CLUSTER) == [], (
            "a failed group commit must not silently persist or acknowledge an observation"
        )


def test_observation_max_age_reclaims_abandoned_running_rows(pg_store):
    assert pg_store.save_attempt_observation(observation()) is True
    counts = pg_store.cleanup_hot_state(
        now=NOW + timedelta(days=90),
        attempt_observation_max_age=timedelta(days=30),
        limit=10,
    )
    assert pg_store.list_attempt_observation_states(CLUSTER) == [], (
        "a running observation that stopped refreshing must not bypass the age bound"
    )
    assert counts.get("attempt_observation_legacy", 0) + counts.get(
        "attempt_observation_stale", 0
    ) == (2 if pg_store.hot_state_mode == "dual" else 1), (
        "cleanup must account for each stored legacy/native copy it actually removed"
    )
