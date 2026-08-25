from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.builtin_metric_contributors import orchestration_metric_lines
from gpu_fault.app.collector_metrics import CollectorMetricsSnapshot
from gpu_fault.app.metric_contributors import MetricContributorRegistry
from gpu_fault.app.metrics import collector_silence_lines
from tests._builders import build_store

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


def test_metric_contributors_render_in_registration_order(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpu_fault.app.metric_contributors.discover_plugins", lambda _group: {}
    )
    registry = MetricContributorRegistry()
    registry.register("first", lambda _runtime: ["first 1"])
    registry.register("second", lambda _runtime: ["second 2"])
    assert registry.names == ("first", "second")
    assert registry.render(object()) == ["first 1", "second 2"]


def test_duplicate_metric_contributor_fails_closed() -> None:
    registry = MetricContributorRegistry()
    registry.register("same", lambda _runtime: [])
    with pytest.raises(RuntimeError, match="duplicate metric"):
        registry.register("same", lambda _runtime: [])


def test_collector_metrics_use_background_snapshot_only() -> None:
    snapshot = SimpleNamespace(lines=lambda: ["snapshot 1"])
    runtime = SimpleNamespace(collector_metrics_snapshot=snapshot)
    assert collector_silence_lines(runtime) == ["snapshot 1"]


def test_ambiguous_attempt_metric_is_exported(monkeypatch) -> None:
    context = ApplicationContext(store=build_store())
    monkeypatch.setattr(
        context.orchestrator._evidence_operations,
        "ambiguous_attempt_ownership_total",
        lambda: 3,
    )

    lines = orchestration_metric_lines(SimpleNamespace(context=context))

    assert lines[-1] == ("gpu_fault_ambiguous_attempt_ownership_total 3")


def test_collector_metrics_top_n_is_bounded() -> None:
    snapshot = object.__new__(CollectorMetricsSnapshot)
    snapshot.top_n = 2
    rows = [
        {
            "cluster_id": "cluster-a",
            "node_id": f"node-{index}",
            "collector": "dcgm",
            "channel": "GPU_METRICS",
            "last_success_age_seconds": float(index),
            "silent": True,
        }
        for index in range(10)
    ]
    lines = snapshot._aggregate_lines(rows)
    assert (
        sum(line.startswith("gpu_fault_collector_silent_top_node") for line in lines)
        == 2
    )
    assert any(
        line.endswith(" 10")
        for line in lines
        if line.startswith("gpu_fault_collector_silent_nodes")
    )
    assert [row["node_id"] for row in snapshot._top_rows(rows)] == ["node-9", "node-8"]


def test_collector_metrics_snapshot_uses_shared_lease_and_record() -> None:
    context = ApplicationContext(store=build_store())
    owner = CollectorMetricsSnapshot(context, owner_id="owner-a", enabled=True)
    follower = CollectorMetricsSnapshot(context, owner_id="owner-b", enabled=True)

    owner.refresh()
    follower.refresh()

    assert owner.lines()
    assert follower.lines() == owner.lines()


def test_collector_metrics_snapshot_persists_bounded_details() -> None:
    context = ApplicationContext(store=build_store())
    snapshot = CollectorMetricsSnapshot(
        context, owner_id="owner-a", enabled=True, now=lambda: NOW
    )
    snapshot.top_n = 2
    rows = [
        {
            "cluster_id": "cluster-a",
            "node_id": f"node-{index}",
            "collector": "dcgm",
            "channel": "GPU_METRICS",
            "last_success_age_seconds": float(index),
            "silent": True,
        }
        for index in range(10)
    ]
    snapshot._rows = lambda _observed_at: rows

    snapshot.refresh()

    record = context.store.get_collector_metrics_snapshot()
    assert record is not None
    assert [row["node_id"] for row in record.details] == ["node-9", "node-8"]


def test_collector_metrics_snapshot_age_advances_without_refresh() -> None:
    context = ApplicationContext(store=build_store())
    clock = [NOW]
    snapshot = CollectorMetricsSnapshot(
        context, owner_id="owner-a", enabled=True, now=lambda: clock[0]
    )
    snapshot.refresh()
    clock[0] += timedelta(seconds=125)

    assert "gpu_fault_collector_metrics_snapshot_age_seconds 125" in snapshot.lines()


def test_disabled_role_does_not_export_collector_snapshot_metrics() -> None:
    snapshot = CollectorMetricsSnapshot(
        ApplicationContext(store=build_store()),
        owner_id="worker-a",
        enabled=False,
        now=lambda: NOW,
    )

    assert snapshot.lines() == []
