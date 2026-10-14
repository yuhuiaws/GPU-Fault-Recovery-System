"""The collector-silence scan alerts only for clusters that exist, as they are now.

Live (batch 100): the scan wrote drill advisories for synthetic perf
registrations -- every synthetic node is silent by construction, thousands of
advisories per perf hour -- and it wrote a cluster's advisories from the Agents
it had read earlier without looking again, so a scan that read before a perf
purge committed wrote 39 notifications (with delivery/result records and dedup
links) after it and the next suite's registration refused. The scan now skips
placeholder registrations through the identity refresh's own predicate and
re-reads the cluster and the Agent right before each write.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, NoReturn

import pytest

from gpu_fault.app import collector_silence
from gpu_fault.app import context as context_module
from gpu_fault.app.collector_silence import notify_silent_collectors
from gpu_fault.app.context import ApplicationContext
from gpu_fault.collector_requirements import (
    COLLECTOR_SYSTEMD_UNITS,
    CollectorServiceState,
)
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.regional import RegionalClusterLifecycle, RegionalClusterRegistration
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import CollectorKind
from tests._builders import build_context
from tests.app_services._cov95_runtime_ingest import NOW
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A
from tests.fleet._support import heartbeat, registry, signed

HOUR = 3600.0
# A perf suite's registration as regional_capacity_registry mints it: synthetic,
# run-owned, expiring, naming a HyperPod cluster and an account that do not exist.
PERF = RegionalClusterRegistration.model_validate(
    {
        **CLUSTER_A.model_dump(),
        "cluster_id": "perf-cap-000",
        "hyperpod_cluster_name": "perf-cap-000",
        "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/perf-cap-000",
        "synthetic": True,
        "synthetic_run_id": "unit-perf-run",
        "synthetic_expires_at": NOW + timedelta(hours=2),
    }
)


def kernel_collector_only() -> dict[str, CollectorServiceState]:
    return {
        unit: CollectorServiceState(
            enabled="enabled" if kind is CollectorKind.NVIDIA_KERNEL else "disabled",
            active="active",
        )
        for kind, unit in COLLECTOR_SYSTEMD_UNITS.items()
    }


def beat(context: ApplicationContext, at: datetime) -> Any:
    """One heartbeat of cluster-a's node-a; its kernel collector never reports."""

    return registry(context.store, now=lambda: at).register(
        signed(
            heartbeat(
                "node-a", observed_at=at, collector_services=kernel_collector_only()
            )
        )
    )


def silent_site(*registrations: RegionalClusterRegistration) -> ApplicationContext:
    """Every registration with one current Agent whose kernel collector is silent.

    The perf suite seeds its clusters' Agents as copies of a live one, which is
    what a registration other than cluster-a gets here.
    """

    context = build_context()
    for registration in registrations:
        context.store.save_regional_cluster(registration)
    record = beat(context, NOW)
    for registration in registrations:
        if registration.cluster_id != record.cluster_id:
            context.store.save_agent(
                record.model_copy(update={"cluster_id": registration.cluster_id})
            )
    return context


def capture(context: ApplicationContext) -> list[str]:
    sent: list[str] = []
    context.advisory_notifications = SimpleNamespace(send=sent.append)
    return sent


def scan(context: ApplicationContext, at: datetime = NOW) -> int:
    return notify_silent_collectors(
        context, observed_at=at, silent_after_seconds=10, alert_interval_seconds=HOUR
    )


def notified_clusters(context: ApplicationContext) -> list[str]:
    return sorted(item.cluster_name for item in context.store.list_notifications())


def test_a_synthetic_registration_gets_no_silence_advisory() -> None:
    context = silent_site(CLUSTER_A, PERF)
    sent = capture(context)

    assert scan(context) == 1
    assert notified_clusters(context) == ["cluster-a"], (
        "a placeholder cluster's nodes are silent by construction"
    )
    assert len(sent) == 1
    assert (
        collector_silence.is_placeholder_registration
        is context_module.is_placeholder_registration
    ), "the silence scan and the identity refresh must share one predicate"


def _purge_cluster(context: ApplicationContext, _monkeypatch: Any) -> None:
    context.store.delete_regional_cluster("cluster-a")


def _revoke_cluster(context: ApplicationContext, _monkeypatch: Any) -> None:
    context.store.save_regional_cluster(
        CLUSTER_A.model_copy(
            update={"lifecycle_state": RegionalClusterLifecycle.REVOKED}
        )
    )


def _delete_agent(context: ApplicationContext, monkeypatch: Any) -> None:
    # A perf purge deletes the Agent rows directly; the store has no per-Agent
    # delete, so the record's absence is what get_agent reports.
    def gone(cluster_id: str, node_id: str) -> NoReturn:
        raise NotFoundError(f"{cluster_id}/{node_id}")

    monkeypatch.setattr(context.store, "get_agent", gone)


def _revoke_agent(context: ApplicationContext, _monkeypatch: Any) -> None:
    record = context.store.get_agent("cluster-a", "node-a")
    context.store.save_agent(
        record.model_copy(update={"lifecycle_state": AgentLifecycleState.REVOKED})
    )


@pytest.mark.parametrize(
    "remove",
    [_purge_cluster, _revoke_cluster, _delete_agent, _revoke_agent],
    ids=["cluster-purged", "cluster-revoked", "agent-deleted", "agent-revoked"],
)
def test_a_cluster_or_agent_removed_after_the_scan_read_it_gets_no_advisory(
    monkeypatch: pytest.MonkeyPatch, remove: Callable[[ApplicationContext, Any], None]
) -> None:
    context = silent_site(CLUSTER_A)
    sent = capture(context)
    read_agents = context.store.list_agents

    def read_then_remove(cluster_id: str | None = None) -> Any:
        agents = read_agents(cluster_id)
        # The removal commits after the scan read the cluster's Agents and
        # before it writes their advisories.
        remove(context, monkeypatch)
        return agents

    monkeypatch.setattr(context.store, "list_agents", read_then_remove)

    assert scan(context) == 0
    assert context.store.list_notifications() == [], (
        "an advisory was written for a cluster or Agent removed mid-scan"
    )
    assert sent == []


def test_a_real_active_cluster_keeps_one_advisory_per_silent_collector_per_hour() -> (
    None
):
    context = silent_site(CLUSTER_A)
    sent = capture(context)

    per_scan = []
    for minute in range(61):
        at = NOW + timedelta(minutes=minute)
        beat(context, at)
        per_scan.append(scan(context, at))

    assert per_scan == [1] + [0] * 59 + [1], "one advisory per hour bucket"
    assert len(sent) == len(set(sent)) == 2
    assert notified_clusters(context) == ["cluster-a", "cluster-a"]
