from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import metric_aggregation, metric_contributors
from gpu_fault.app.aurora_refresh_metrics import (
    AURORA_REFRESH_STATUS_FILENAME,
    aurora_credential_refresh_metric_lines,
)
from gpu_fault.app.collector_metrics import CollectorMetricsSnapshot
from gpu_fault.app.metric_contributors import MetricContributorRegistry
from tests._builders import build_context
from tests.fleet._support import NOW, heartbeat, registry, signed
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def test_plugin_aggregation_registration_is_idempotent_and_rejects_conflicting_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metric_aggregation, "STRATEGIES", dict(metric_aggregation.STRATEGIES)
    )
    name = "unit_runtime_plugin_queue"
    with pytest.raises(ValueError, match="name is required"):
        metric_aggregation.register("", metric_aggregation.SUM)
    assert metric_aggregation.strategy_for(name) is None
    metric_aggregation.register(name, metric_aggregation.SUM)
    metric_aggregation.register(name, metric_aggregation.SUM)
    with pytest.raises(RuntimeError, match="already registered as sum"):
        metric_aggregation.register(name, metric_aggregation.MAX)
    assert metric_aggregation.strategy_for(name) is metric_aggregation.SUM


def test_plugin_failure_keeps_registered_siblings_and_counts_each_failed_scrape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discoveries = []

    def broken(runtime: Any) -> list[str]:
        raise RuntimeError("synthetic metric source failure")

    def discover(group: Any) -> dict[str, Any]:
        discoveries.append(group)
        return {
            "unit-plugin": SimpleNamespace(
                load=lambda: lambda runtime: ["unit_plugin 7"]
            )
        }

    monkeypatch.setattr(metric_contributors, "discover_plugins", discover)
    contributors = MetricContributorRegistry()
    with pytest.raises(ValueError, match="name is required"):
        contributors.register("", broken)
    contributors.register("unit-failing", broken)
    with pytest.raises(RuntimeError, match="duplicate metric contributor"):
        contributors.register("unit-failing", lambda runtime: ["wrong_override 1"])
    for expected in (1, 2):
        lines = contributors.render(SimpleNamespace(background_services_enabled=True))
        assert "unit_plugin 7" in lines
        assert "wrong_override 1" not in lines
        assert contributors.error_counts() == {"unit-failing": expected}
    assert len(discoveries) == 1
    assert contributors.names == ("unit-failing", "unit-plugin")


@pytest.mark.parametrize(
    "finished", ["", [], "not-a-date", "2080-01-01T00:00:00", "2080-01-01T00:00:00Z"]
)
def test_refresh_status_never_uses_invalid_time_as_success_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, finished: Any
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "not-a-dsn"))
    (tmp_path / AURORA_REFRESH_STATUS_FILENAME).write_text(
        json.dumps(
            {
                "finished_at": finished,
                "status": "ok",
                "error": "unit-sensitive-value-that-must-not-be-rendered",
            }
        )
    )
    lines = aurora_credential_refresh_metric_lines(SimpleNamespace())
    assert "gpu_fault_aurora_credential_refresh_status_unreadable 1" in lines
    assert not any("last_success_age_seconds" in line for line in lines), (
        "invalid or future evidence must not publish a success age"
    )
    assert not any("last_run_ok" in line for line in lines), (
        "unknown refresh evidence must not claim a known outcome"
    )
    assert not any("unit-sensitive-value" in line for line in lines), (
        "refresh metrics must not expose arbitrary status-file values"
    )


def test_peer_lease_without_snapshot_does_not_invent_collector_health() -> None:
    fleet = registry()
    fleet.register(signed(heartbeat("node-a")))
    context = build_context(store=fleet.store)
    observed = [NOW]
    snapshot = CollectorMetricsSnapshot(
        context, owner_id="unit-reader", enabled=True, now=lambda: observed[0]
    )
    context.store.acquire_periodic_task_lease(
        "collector-metrics-snapshot",
        "unit-peer",
        now=NOW,
        lease_duration=timedelta(seconds=60),
    )
    snapshot.refresh()
    assert snapshot.details() == []
    assert (
        snapshot.lines()[-1] == "gpu_fault_collector_metrics_snapshot_age_seconds +Inf"
    )
    assert context.store.get_collector_metrics_snapshot() is None
    observed[0] = NOW + timedelta(hours=1)
    snapshot.refresh()
    assert snapshot.details() == [], (
        "expired agents cannot become current collector rows"
    )
    assert context.store.get_collector_metrics_snapshot() is not None
    assert snapshot.lines()[-1] == "gpu_fault_collector_metrics_snapshot_age_seconds 0"
