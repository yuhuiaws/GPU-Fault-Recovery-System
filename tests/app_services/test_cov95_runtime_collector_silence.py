from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.app.collector_silence import notify_silent_collectors
from gpu_fault.collector_requirements import (
    COLLECTOR_SYSTEMD_UNITS,
    CollectorServiceState,
)
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.regional import RegionalClusterLifecycle
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import build_context
from tests.app_services._cov95_runtime_ingest import NOW
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A
from tests.fleet._support import heartbeat, registry, signed
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("threshold", [10.0, {CollectorKind.NVIDIA_KERNEL: 10.0}])
def test_silent_collector_is_not_requeued_within_the_same_notification_bucket(
    threshold: float | dict,
) -> None:
    context = build_context()
    context.store.save_regional_cluster(CLUSTER_A)
    services = {
        unit: CollectorServiceState(
            enabled="enabled" if kind is CollectorKind.NVIDIA_KERNEL else "disabled",
            active="active",
        )
        for kind, unit in COLLECTOR_SYSTEMD_UNITS.items()
    }
    registry(context.store, now=lambda: NOW).register(
        signed(heartbeat("node-a", observed_at=NOW, collector_services=services))
    )
    sent = []
    context.advisory_notifications = SimpleNamespace(send=sent.append)
    for at, expected in (
        (NOW, 1),
        (NOW + timedelta(seconds=1), 0),
        (NOW + timedelta(seconds=61), 1),
    ):
        assert (
            notify_silent_collectors(
                context,
                observed_at=at,
                silent_after_seconds=threshold,
                alert_interval_seconds=60,
            )
            == expected
        )
    assert len(sent) == len(context.store.list_notifications()) == 2
    assert len(set(sent)) == 2
    assert all(
        "node-a" in item.subject for item in context.store.list_notifications()
    ), "each warning must name its owned node"


@pytest.mark.parametrize(
    "state",
    [
        "fresh",
        "stale-success",
        "error-only",
        "inactive-cluster",
        "expired-agent",
        "draining-agent",
    ],
)
def test_collector_silence_respects_cluster_agent_and_success_freshness(
    state: str,
) -> None:
    context = build_context()
    context.store.save_regional_cluster(
        CLUSTER_A.model_copy(
            update={"lifecycle_state": RegionalClusterLifecycle.DRAINING}
            if state == "inactive-cluster"
            else {}
        )
    )
    services = {
        unit: CollectorServiceState(
            enabled="enabled" if kind is CollectorKind.NVIDIA_KERNEL else "disabled",
            active="active",
        )
        for kind, unit in COLLECTOR_SYSTEMD_UNITS.items()
    }
    record = registry(context.store, now=lambda: NOW).register(
        signed(heartbeat("node-a", observed_at=NOW, collector_services=services))
    )
    if state in {"expired-agent", "draining-agent"}:
        context.store.save_agent(
            record.model_copy(
                update=(
                    {"lease_expires_at": NOW - timedelta(seconds=1)}
                    if state == "expired-agent"
                    else {"lifecycle_state": AgentLifecycleState.DRAINING}
                )
            )
        )
    context.store.save_collector_status(
        CollectorStatus(
            cluster_id="cluster-a",
            node_id="node-a",
            collector=CollectorKind.NVIDIA_KERNEL,
            observed_at=NOW,
            ingested_at=NOW,
            last_success_at=(
                None
                if state == "error-only"
                else NOW - timedelta(seconds=11 if state == "stale-success" else 10)
            ),
        )
    )
    sent = []
    context.advisory_notifications = SimpleNamespace(send=sent.append)
    result = notify_silent_collectors(
        context, observed_at=NOW, silent_after_seconds=10, alert_interval_seconds=60
    )
    assert result == (1 if state in {"stale-success", "error-only"} else 0)
    assert len(sent) == result
