"""Retired (non-ACTIVE) Agents never decide collector readiness or silence.

A node HyperPod terminated leaves ``collector_status`` rows nothing will ever
refresh. Once its Agent record is retired (``DRAINING``/``REVOKED``) those rows
must stop holding the cluster "not ready" or "silent"; the node is still listed
for operators under ``retired_nodes``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.collector_metrics import CollectorMetricsSnapshot
from gpu_fault.app.collector_silence import notify_silent_collectors
from gpu_fault.collector_requirements import (
    COLLECTOR_SYSTEMD_UNITS,
    collector_silent_thresholds,
)
from gpu_fault.fleet import AgentLifecycleState, AgentTransitionRequest, FleetRegistry
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import asgi_client, build_context, build_store
from tests.app_services._cov95_runtime_ingest import NOW
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A
from tests.fleet._support import heartbeat, registry, signed

EXECUTION_TOKEN = "e" * 32
HEADERS = {"X-GPU-Fault-Execution-Token": EXECUTION_TOKEN}
KIND = CollectorKind.NVIDIA_KERNEL
SERVICES = {
    unit: {"active": "active", "enabled": "enabled" if kind is KIND else "disabled"}
    for kind, unit in COLLECTOR_SYSTEMD_UNITS.items()
}
THRESHOLD = collector_silent_thresholds()[KIND]


def _register(fleet: FleetRegistry, node_id: str, *, now: datetime) -> None:
    fleet.register(
        signed(heartbeat(node_id, observed_at=now, collector_services=SERVICES))
    )


def _retire(fleet: FleetRegistry, node_id: str, *, state: AgentLifecycleState) -> None:
    """Retire through the registry's own lifecycle path, as the admin verb does."""

    record = fleet.store.get_agent("cluster-a", node_id)
    request = AgentTransitionRequest(
        expected_generation=record.generation,
        transition_id="workflow-reconcile/CHG-unit/0123456789abcdef",
        reason="operator reconciliation CHG-unit: node left Kubernetes and HyperPod",
    )
    fleet.drain_agent("cluster-a", node_id, request)
    if state is AgentLifecycleState.REVOKED:
        fleet.revoke_agent("cluster-a", node_id, request)
    assert fleet.store.get_agent("cluster-a", node_id).lifecycle_state is state, (
        "the fixture must leave the Agent in the requested lifecycle state"
    )


def _report(
    store: Any, node_id: str, last_success_at: datetime, *, now: datetime
) -> None:
    store.save_collector_status(
        CollectorStatus(
            cluster_id="cluster-a",
            node_id=node_id,
            collector=KIND,
            observed_at=now,
            ingested_at=now,
            last_success_at=last_success_at,
        )
    )


def _readiness(context: Any) -> dict[str, Any]:
    async def scenario() -> dict[str, Any]:
        async with asgi_client(context) as client:
            response = await client.get(
                "/v1/collector-readiness/cluster-a", headers=HEADERS
            )
        assert response.status_code == 200, response.text
        return response.json()

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    "state", [AgentLifecycleState.DRAINING, AgentLifecycleState.REVOKED]
)
def test_retired_agent_stale_receipts_do_not_hold_readiness(
    state: AgentLifecycleState,
) -> None:
    now = datetime.now(timezone.utc)
    store = build_store()
    fleet = registry(store, now=lambda: now)
    for node in ("node-a", "node-b"):
        _register(fleet, node, now=now)
    _report(store, "node-a", now, now=now)
    _report(store, "node-b", now - timedelta(seconds=THRESHOLD + 3600), now=now)
    _retire(fleet, "node-b", state=state)
    context = build_context(
        store=store, execution_token=EXECUTION_TOKEN, fleet_registry=fleet
    )

    result = _readiness(context)

    assert result["ready"] is True, "a retired Agent's stale rows must not gate"
    assert [node["node_id"] for node in result["nodes"]] == ["node-a"]
    (retired,) = result["retired_nodes"]
    assert retired["node_id"] == "node-b"
    assert retired["lifecycle_state"] == state.value
    assert retired["last_seen_at"] == now.isoformat()
    assert "ready" not in retired, "retired nodes carry no verdict"
    stale = retired["collectors"][KIND.value]
    assert stale["unit"] == COLLECTOR_SYSTEMD_UNITS[KIND]
    assert stale["age_seconds"] > THRESHOLD, "the stale receipt stays visible"
    assert len(store.list_collector_statuses("cluster-a", "node-b")) == 1, (
        "readiness is a reader; it must not delete the departed node's evidence"
    )


def test_active_agent_with_stale_receipts_still_fails_beside_a_retired_one() -> None:
    now = datetime.now(timezone.utc)
    store = build_store()
    fleet = registry(store, now=lambda: now)
    for node in ("node-a", "node-b"):
        _register(fleet, node, now=now)
    stale = now - timedelta(seconds=THRESHOLD + 10)
    _report(store, "node-a", stale, now=now)
    _report(store, "node-b", stale, now=now)
    _retire(fleet, "node-b", state=AgentLifecycleState.REVOKED)
    context = build_context(
        store=store, execution_token=EXECUTION_TOKEN, fleet_registry=fleet
    )

    result = _readiness(context)

    assert result["ready"] is False, "an ACTIVE Agent's stale receipt still gates"
    (active,) = result["nodes"]
    assert active["node_id"] == "node-a" and active["ready"] is False
    assert [node["node_id"] for node in result["retired_nodes"]] == ["node-b"]


def test_cluster_with_only_retired_agents_is_not_ready() -> None:
    now = datetime.now(timezone.utc)
    store = build_store()
    fleet = registry(store, now=lambda: now)
    _register(fleet, "node-b", now=now)
    _report(store, "node-b", now, now=now)
    _retire(fleet, "node-b", state=AgentLifecycleState.REVOKED)
    context = build_context(
        store=store, execution_token=EXECUTION_TOKEN, fleet_registry=fleet
    )

    result = _readiness(context)

    assert result["ready"] is False, "no reporting Agent is not readiness"
    assert result["nodes"] == []
    assert [node["node_id"] for node in result["retired_nodes"]] == ["node-b"]


def test_collector_metrics_exclude_retired_agent_rows() -> None:
    store = build_store()
    fleet = registry(store, now=lambda: NOW)
    for node in ("node-a", "node-b"):
        _register(fleet, node, now=NOW)
    _report(store, "node-a", NOW, now=NOW)
    _report(store, "node-b", NOW - timedelta(hours=3), now=NOW)
    _retire(fleet, "node-b", state=AgentLifecycleState.REVOKED)
    context = build_context(store=store)
    snapshot = CollectorMetricsSnapshot(
        context, owner_id="unit-reader", enabled=True, now=lambda: NOW
    )

    snapshot.refresh()

    assert {row["node_id"] for row in snapshot.details()} == {"node-a"}
    lines = snapshot.lines()
    silent = [
        line for line in lines if line.startswith("gpu_fault_collector_silent_nodes{")
    ]
    assert silent and all(line.endswith(" 0") for line in silent), (
        "a retired Agent's hours-old receipt must not count as a silent node"
    )
    ages = [
        float(line.rsplit(" ", 1)[1])
        for line in lines
        if line.startswith("gpu_fault_collector_last_success_age_seconds_max{")
    ]
    assert ages and max(ages) < THRESHOLD, (
        "the retired node's age must not feed the max-age gauge"
    )
    assert not any("node-b" in line for line in lines), (
        "a retired node must not appear in any collector gauge series"
    )


def test_collector_silence_skips_retired_agent() -> None:
    context = build_context()
    context.store.save_regional_cluster(CLUSTER_A)
    fleet = registry(context.store, now=lambda: NOW)
    for node in ("node-a", "node-b"):
        _register(fleet, node, now=NOW)
    _report(context.store, "node-a", NOW, now=NOW)
    _report(context.store, "node-b", NOW - timedelta(hours=3), now=NOW)
    _retire(fleet, "node-b", state=AgentLifecycleState.REVOKED)
    sent: list[str] = []
    context.advisory_notifications = SimpleNamespace(send=sent.append)

    result = notify_silent_collectors(
        context, observed_at=NOW, silent_after_seconds=10, alert_interval_seconds=60
    )

    assert result == 0, "a retired Agent is not a silent collector"
    assert sent == [] and context.store.list_notifications() == []
