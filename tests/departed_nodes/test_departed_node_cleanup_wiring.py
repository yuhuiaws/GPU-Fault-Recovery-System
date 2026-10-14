"""Where the departed-node cleanup runs: one writer, the worker tier, /metrics.

The cleanup is constructed in every control-plane process of a regional site
but runs as a periodic job, under the ``departed-node-cleanup`` task lease, so
exactly one process in the fleet is the writer (the brief's single-writer
refusal); its counters reach ``/metrics`` through a fleet-level contributor
the ingress tier skips.
"""

from __future__ import annotations

import time
from datetime import timedelta
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.metric_aggregation import strategy_for
from gpu_fault.app.periodic_services import PeriodicServiceRunner
from gpu_fault.departed_node_cleanup import (
    CLEANUP_INTERVAL_SECONDS,
    DepartedNodeCleanup,
    DepartedNodeCleanupConfig,
    cleanup_enabled,
    departed_node_cleanup_from_context,
    departed_node_cleanup_metric_lines,
)
from gpu_fault.departed_node_provider import Boto3ProviderMembership
from gpu_fault.fleet import AgentLifecycleState
from tests._builders import build_store
from tests.departed_nodes._support import HP_CLUSTER, HP_NODE, hyperpod_departure


def _runner(context: Any, owner: str) -> PeriodicServiceRunner:
    processor = SimpleNamespace(
        is_healthy=lambda: True,
        active_consumers=True,
        is_leader=lambda: True,
        owner_id=owner,
    )
    return PeriodicServiceRunner(
        context=context,
        processor=processor,
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
    )


def test_a_second_replica_does_not_run_the_cleanup_while_the_first_holds_its_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    other = DepartedNodeCleanup(
        store,
        registry=world_.registry,
        incident_closure=world_.closure,
        provider=world_.provider,
    )
    first = _runner(
        SimpleNamespace(store=store, departed_node_cleanup=world_.cleanup), "pod-a:1"
    )
    second = _runner(
        SimpleNamespace(store=store, departed_node_cleanup=other), "pod-b:1"
    )
    for runner in (first, second):
        # Only the job under test; the others need a fuller context.
        monkeypatch.setattr(runner, "_JOBS", ("departed_nodes",))

    first.run_all_due(time.monotonic())
    second.run_all_due(time.monotonic())

    assert world_.cleanup.cycles_total == 1
    assert other.cycles_total == 0, "the lease holder is the one writer"
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.REVOKED
    )
    assert (
        "departed_nodes" in first.metrics_snapshot()["job_last_run_timestamp_seconds"]
    )


def test_a_failing_cycle_is_counted_against_the_job_and_the_loop_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken:
        interval_seconds = CLEANUP_INTERVAL_SECONDS

        def run_once(self) -> None:
            raise RuntimeError("store unavailable")

    runner = _runner(
        SimpleNamespace(store=build_store(), departed_node_cleanup=Broken()), "pod-a:1"
    )
    monkeypatch.setattr(runner, "_JOBS", ("departed_nodes",))

    runner.run_all_due(time.monotonic())

    assert runner.metrics_snapshot()["periodic_job_errors_total"] == {
        "departed_nodes": 1
    }


def test_no_cleanup_runs_outside_a_regional_control_plane() -> None:
    assert (
        departed_node_cleanup_from_context(SimpleNamespace(regional_mode=False)) is None
    )
    assert (
        departed_node_cleanup_from_context(
            SimpleNamespace(regional_mode=True, fleet_registry=None)
        )
        is None
    )
    runner = _runner(SimpleNamespace(store=build_store()), "pod-a:1")
    runner.run_all_due(time.monotonic())
    snapshot = runner.metrics_snapshot()
    assert "departed_nodes" not in snapshot["job_last_run_timestamp_seconds"]
    assert "departed_nodes" not in snapshot["periodic_job_errors_total"]


def test_the_regional_control_plane_builds_it_from_its_own_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED", "false")
    executor_config = SimpleNamespace(step_waiting_limit=lambda operation: 7200)
    context = SimpleNamespace(
        regional_mode=True,
        fleet_registry=object(),
        store=build_store(),
        incident_closure=object(),
        advisory_notifications=object(),
        production_executor_config=executor_config,
    )

    cleanup = departed_node_cleanup_from_context(context)

    assert isinstance(cleanup, DepartedNodeCleanup), cleanup
    assert cleanup.config.enabled is False
    assert cleanup.waiting_ttl == timedelta(seconds=7200)
    assert isinstance(cleanup.provider, Boto3ProviderMembership), cleanup.provider
    assert cleanup.notifications is context.advisory_notifications


@pytest.mark.parametrize(
    ("value", "expected"), [("", True), ("true", True), ("false", False)]
)
def test_the_switch_defaults_on(value: str, expected: bool) -> None:
    assert cleanup_enabled({"GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED": value}) is (
        expected
    )
    assert DepartedNodeCleanupConfig.from_environment({}).enabled is True


def test_every_rendered_family_has_an_aggregation_strategy() -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    world_.cleanup.run_once()

    lines = departed_node_cleanup_metric_lines(world_.cleanup)
    families = {line.split()[2] for line in lines if line.startswith("# TYPE ")}

    assert departed_node_cleanup_metric_lines(None) == []
    assert families == {
        "gpu_fault_departed_node_cleanup_enabled",
        "gpu_fault_departed_node_cleanup_cycles_total",
        "gpu_fault_departed_node_cleanup_last_cycle_timestamp_seconds",
        "gpu_fault_departed_node_cleanup_actions_total",
        "gpu_fault_departed_node_cleanup_refusals_total",
    }
    assert all(strategy_for(name) is not None for name in families), families
    assert (
        'gpu_fault_departed_node_cleanup_actions_total{action="agent_retired"} 1'
        in lines
    )


def test_the_contributor_is_registered_for_the_worker_tier_only() -> None:
    from gpu_fault.app.metrics import METRIC_CONTRIBUTORS

    assert "departed-node-cleanup" in METRIC_CONTRIBUTORS.fleet_level_names
