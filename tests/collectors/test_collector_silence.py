from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.app import ApplicationContext, notify_silent_collectors
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import build_store

NOW = datetime(2026, 8, 14, 1, 45, tzinfo=timezone.utc)


def test_silent_collector_notifications_are_deduplicated() -> None:
    store = build_store()
    store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id="cluster-a",
            region="us-west-2",
            hyperpod_cluster_name="hp-a",
            eks_cluster_arn="arn:aws:eks:us-west-2:1:cluster/a",
            token_sha256="a" * 64,
        )
    )
    store.save_agent(
        SimpleNamespace(
            cluster_id="cluster-a",
            node_id="node-a",
            lifecycle_state=SimpleNamespace(value="ACTIVE"),
        )
    )
    store.save_collector_status(
        CollectorStatus(
            cluster_id="cluster-a",
            node_id="node-a",
            collector=CollectorKind.GPU_INVENTORY,
            observed_at=NOW,
            ingested_at=NOW,
            last_success_at=NOW - timedelta(seconds=30),
        )
    )
    context = ApplicationContext(store=store)
    context.regional_mode = True
    sent = []
    context.advisory_notifications.send = sent.append

    first = notify_silent_collectors(
        context,
        observed_at=NOW,
        silent_after_seconds={
            CollectorKind.GPU_INVENTORY: 180,
            CollectorKind.GPU_METRICS: 420,
            CollectorKind.HOST_TELEMETRY: 420,
            CollectorKind.NVIDIA_KERNEL: 600,
            CollectorKind.FABRIC_MANAGER_LOG: 600,
            CollectorKind.NODE_LOGS: 600,
        },
        alert_interval_seconds=3600,
    )
    second = notify_silent_collectors(
        context,
        observed_at=NOW + timedelta(minutes=1),
        silent_after_seconds={
            CollectorKind.GPU_INVENTORY: 180,
            CollectorKind.GPU_METRICS: 420,
            CollectorKind.HOST_TELEMETRY: 420,
            CollectorKind.NVIDIA_KERNEL: 600,
            CollectorKind.FABRIC_MANAGER_LOG: 600,
            CollectorKind.NODE_LOGS: 600,
        },
        alert_interval_seconds=3600,
    )

    assert first == 4
    assert second == 4
    assert len(store.list_notifications()) == 4
    assert len(set(sent)) == 4


def test_expired_active_agent_is_not_reported_silent() -> None:
    store = build_store()
    store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id="cluster-a",
            region="us-west-2",
            hyperpod_cluster_name="hp-a",
            eks_cluster_arn="arn:aws:eks:us-west-2:1:cluster/a",
            token_sha256="a" * 64,
        )
    )
    store.save_agent(
        SimpleNamespace(
            cluster_id="cluster-a",
            node_id="retired-node",
            lifecycle_state=SimpleNamespace(value="ACTIVE"),
            lease_expires_at=NOW - timedelta(seconds=1),
        )
    )
    context = ApplicationContext(store=store)
    context.regional_mode = True
    sent = []
    context.advisory_notifications.send = sent.append

    result = notify_silent_collectors(
        context,
        observed_at=NOW,
        silent_after_seconds={
            CollectorKind.GPU_INVENTORY: 180,
            CollectorKind.GPU_METRICS: 420,
            CollectorKind.HOST_TELEMETRY: 420,
        },
        alert_interval_seconds=3600,
    )

    assert result == 0
    assert store.list_notifications() == []
    assert sent == []
