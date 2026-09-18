from __future__ import annotations

from datetime import timedelta

from gpu_fault.telemetry import CollectorKind, CollectorStatus
from gpu_fault.telemetry_models import TelemetryMetricLatest
from gpu_fault.training_models import TrainingProgressHeartbeat
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)
from tests.store._cov95_compat_telemetry import CLUSTER, GPU, NODE, finding


def test_host_metrics_normalize_node_device_without_collapsing_named_devices(
    compat_store,
):
    store = compat_store
    base = TelemetryMetricLatest(
        cluster_id=CLUSTER, node_id=NODE, observed_at=NOW, name="load", value=1.0
    )
    named = base.model_copy(update={"device": "cpu-1", "value": 2.0})
    alias = base.model_copy(
        update={"device": "", "value": 3.0, "observed_at": NOW + timedelta(seconds=1)}
    )
    other_node = base.model_copy(update={"node_id": "other-node"})
    assert (
        store.observe_telemetry_metrics([base, named, alias, other_node]) == [True] * 4
    )
    by_device = {
        item.device: item.value
        for item in store.list_telemetry_metrics_latest(CLUSTER, NODE)
    }
    assert by_device == {"": 3.0, "cpu-1": 2.0}
    assert store.observe_telemetry_metric(base) is False
    assert store.list_telemetry_metrics_latest(CLUSTER, "other-node") == [other_node]
    assert store.list_telemetry_metrics_latest("other-cluster", NODE) == []


def test_regressed_and_unknown_training_steps_preserve_distinct_progress_semantics(
    compat_store,
):
    store = compat_store
    first = TrainingProgressHeartbeat(
        cluster_id=CLUSTER, attempt_id="attempt", rank=0, observed_at=NOW, step=10
    )
    regressed = first.model_copy(
        update={"observed_at": NOW + timedelta(seconds=1), "step": 9}
    )
    unknown = first.model_copy(
        update={"observed_at": NOW + timedelta(seconds=2), "step": None}
    )
    restart = first.model_copy(
        update={"observed_at": NOW + timedelta(seconds=3), "step": 1}
    )
    store.observe_training_progress(first)
    store.observe_training_progress(regressed)
    assert (
        store.list_training_progress_states(CLUSTER, "attempt")[0].last_progress_at
        == NOW
    )
    store.observe_training_progress(unknown)
    assert (
        store.list_training_progress_states(CLUSTER, "attempt")[0].last_progress_at
        == unknown.observed_at
    )
    store.observe_training_progress(restart)
    current = store.list_training_progress_states(CLUSTER, "attempt")[0]
    assert current.last_progress_at == restart.observed_at
    assert current.heartbeat.step == 1
    assert store.observe_training_progress(regressed) is False
    assert store.list_training_progress("other-cluster") == []


def test_history_retention_does_not_clear_active_gpu_findings(compat_store):
    store = compat_store
    key = (CLUSTER, NODE, GPU, "temperature")
    first = finding("history-a", at=NOW - timedelta(days=3))
    changed = finding(
        "history-b", at=NOW - timedelta(days=2), automatic_action="VALIDATE"
    )
    assert store.update_gpu_finding(key, first, first.observed_at) is True
    assert store.update_gpu_finding(key, changed, changed.observed_at) is True
    assert (
        store.cleanup_hot_state(
            now=NOW, finding_history_retention=timedelta(days=1), limit=0
        )["gpu_finding_history"]
        == 0
    )
    assert (
        store.cleanup_hot_state(
            now=NOW, finding_history_retention=timedelta(days=1), limit=1
        )["gpu_finding_history"]
        == 1
    )
    assert [
        item.finding_id
        for item in store.list_gpu_findings(CLUSTER, NODE, active_only=False)
    ] == ["history-b"]
    assert store.list_gpu_findings(CLUSTER, NODE, active_only=True) == [changed]
    assert store.get_gpu_finding_state(key).consecutive_breaches == 2
    assert (
        store.cleanup_hot_state(
            now=NOW, finding_history_retention=timedelta(days=1), limit=1
        )["gpu_finding_history"]
        == 1
    )
    assert store.list_gpu_findings(CLUSTER, NODE, active_only=False) == []
    assert store.list_gpu_findings(CLUSTER, NODE, active_only=True) == [changed]


def test_collector_same_timestamp_success_clears_only_after_newer_error_time(
    compat_store,
):
    store = compat_store
    rejected = CollectorStatus(
        cluster_id=CLUSTER,
        node_id=NODE,
        collector=CollectorKind.NODE_LOGS,
        observed_at=NOW,
        ingested_at=NOW,
        last_error_at=NOW,
        errors=["rejected-event:compat"],
    )
    equal_success = rejected.model_copy(update={"last_success_at": NOW, "errors": []})
    newer_success = rejected.model_copy(
        update={
            "observed_at": NOW + timedelta(seconds=1),
            "last_success_at": NOW + timedelta(seconds=1),
            "errors": [],
        }
    )
    assert store.save_collector_statuses_batch([rejected, equal_success]) == [
        True,
        True,
    ]
    (current,) = store.list_collector_statuses(CLUSTER, NODE)
    assert current.errors == ["rejected-event:compat"]
    assert store.save_collector_status(newer_success) is True
    (current,) = store.list_collector_statuses(CLUSTER, NODE)
    assert current.errors == []
    assert current.last_error_at == NOW
    assert current.last_success_at == NOW + timedelta(seconds=1)
