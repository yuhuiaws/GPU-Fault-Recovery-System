"""Every control loop exports when it last turned (ARCH-E E3, E5, E6).

The progress gauges -- pending age, error counters, drift -- are written by
the loops themselves, so a thread that died leaves each of them frozen at the
last healthy value it had. A wall-clock stamp taken at the top of every cycle
is the one signal a dead loop cannot keep fresh.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.app.builtin_metric_contributors import (
    control_loop_metric_lines,
    fleet_pin_drift_metric_lines,
)
from gpu_fault.app.collector_metrics import aggregate_lines
from gpu_fault.app.metrics import METRIC_CONTRIBUTORS
from gpu_fault.app.metrics_sections import (
    render_processor_metrics_1,
    render_spool_metrics_one,
)
from gpu_fault.app.periodic_services import PeriodicServiceRunner
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.fleet import FleetCompatibilityPolicy, FleetRegistry
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.fleet._support import CONFIG, NOW, SECRET, heartbeat, registry, signed


def _metric_value(lines: list[str], name: str) -> float:
    [line] = [item for item in lines if item.startswith(f"{name} ")]
    return float(line.split()[-1])


def test_the_workflow_dispatcher_stamps_every_cycle_it_starts() -> None:
    store = build_store()
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True, max_workers=1),
    )
    assert dispatcher.last_cycle_timestamp_seconds == 0.0, "nothing before a tick"

    before = time.time()
    dispatcher.run_once()

    assert dispatcher.last_cycle_timestamp_seconds >= before
    lines = control_loop_metric_lines(
        SimpleNamespace(context=SimpleNamespace(store=store, dispatcher=dispatcher))
    )
    # Rendered to the millisecond, so allow the truncation.
    assert (
        _metric_value(lines, "gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds")
        >= before - 0.001
    )


def test_a_dispatcher_that_never_ran_exports_zero_not_nothing() -> None:
    lines = control_loop_metric_lines(SimpleNamespace(context=SimpleNamespace()))

    assert (
        _metric_value(lines, "gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds")
        == 0.0
    )
    assert (
        _metric_value(lines, "gpu_fault_periodic_last_cycle_timestamp_seconds") == 0.0
    )
    assert (
        _metric_value(
            lines, "gpu_fault_notification_dispatch_last_cycle_timestamp_seconds"
        )
        == 0.0
    )


class _OneTickStop:
    """A stop event whose first wait says "keep going" and whose second says stop."""

    def __init__(self) -> None:
        self.waits = 0

    def wait(self, _timeout: float) -> bool:
        self.waits += 1
        return self.waits > 1

    def is_set(self) -> bool:
        return self.waits > 1


def _runner(stop) -> PeriodicServiceRunner:
    processor = SimpleNamespace(
        is_healthy=lambda: True,
        active_consumers=True,
        is_leader=lambda: True,
        owner_id="pod-a:1",
    )
    return PeriodicServiceRunner(
        context=SimpleNamespace(store=build_store()),
        processor=processor,
        stop=stop,
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
    )


def test_the_periodic_runner_stamps_every_tick_and_every_job_it_ran(
    monkeypatch,
) -> None:
    runner = _runner(_OneTickStop())
    ran: list[str] = []
    for name in runner._JOBS:
        monkeypatch.setattr(
            runner,
            f"_run_{name}",
            (lambda job: lambda now: ran.append(job) or job == "cleanup")(name),
        )

    before = time.time()
    runner.run()

    snapshot = runner.metrics_snapshot()
    assert snapshot["last_cycle_timestamp_seconds"] >= before
    assert set(snapshot["job_last_run_timestamp_seconds"]) == {"cleanup"}, (
        "only the job that reported it ran is stamped"
    )
    lines = control_loop_metric_lines(
        SimpleNamespace(context=SimpleNamespace(periodic_runner=runner))
    )
    assert _metric_value(lines, "gpu_fault_periodic_last_cycle_timestamp_seconds") >= (
        before - 0.001
    )
    assert any(
        line.startswith(
            'gpu_fault_periodic_job_last_run_timestamp_seconds{job="cleanup"} '
        )
        for line in lines
    ), lines


def test_the_runner_stamps_a_tick_it_sat_out_as_inactive() -> None:
    """The stamp answers "is the thread alive", not "did it own a lease"."""
    stop = _OneTickStop()
    runner = _runner(stop)
    runner.processor.is_healthy = lambda: False

    before = time.time()
    runner.run()

    assert runner.metrics_snapshot()["last_cycle_timestamp_seconds"] >= before
    assert runner.metrics_snapshot()["job_last_run_timestamp_seconds"] == {}


def test_processor_and_spool_stamps_render_only_once_the_processor_publishes_them():
    """The claim loop and spool consumer live in processor/**; the renderer
    is ready for their stamps and stays silent -- not zero -- until they exist."""
    runtime = {
        "active_consumer": 1,
        "notifications_enabled": 0,
        "notifications_received_total": 0,
        "notifications_filtered_total": 0,
        "notification_reconnects_total": 0,
        "notification_shard": -1,
        "notification_shard_count": 0,
        "in_flight": 0,
        "oldest_in_flight_seconds": 0.0,
        "healthy": 1,
        "spool": {},
    }
    io = SimpleNamespace(in_flight=0, max_in_flight=1, rejected_total=0)
    batcher = SimpleNamespace()

    silent: list[str] = []
    render_processor_metrics_1(silent, runtime, {}, 0, batcher, 0, 0, io, io)
    assert not any("claim_last_round_timestamp_seconds" in line for line in silent), (
        "an unpublished processor stamp must not render as zero"
    )

    stamped: list[str] = []
    render_processor_metrics_1(
        stamped,
        runtime,
        {"last_round_timestamp_seconds": 1_757_000_000.5},
        0,
        batcher,
        0,
        0,
        io,
        io,
    )
    assert (
        "gpu_fault_processor_claim_last_round_timestamp_seconds 1757000000.500"
        in stamped
    )

    spool = {
        "depth": 0,
        "leased": 0,
        "oldest_age_seconds": 0.0,
        "by_cluster": {},
        "payload_bytes": 0,
        "leased_bytes": 0,
    }
    spool_batcher = SimpleNamespace(pending_depth=0)
    silent_spool: list[str] = []
    render_spool_metrics_one(
        silent_spool, spool, {"spool": {}}, True, spool_batcher, {}, 0, 0
    )
    assert not any("consumer_last_cycle_timestamp" in line for line in silent_spool), (
        "an unpublished spool stamp must not render as zero"
    )
    stamped_spool: list[str] = []
    render_spool_metrics_one(
        stamped_spool,
        spool,
        {"spool": {"last_cycle_timestamp_seconds": 1_757_000_001.0}},
        True,
        spool_batcher,
        {},
        0,
        0,
    )
    assert (
        "gpu_fault_telemetry_spool_consumer_last_cycle_timestamp_seconds 1757000001.000"
    ) in stamped_spool


def test_pin_ahead_of_fleet_is_a_gauge_not_only_a_readiness_reason() -> None:
    store = build_store()
    fleet = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_version="0.9.0",
            required_artifact_sha256="d" * 64,
            required_config_digest=CONFIG,
        ),
        now=lambda: NOW,
    )
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))

    report = fleet.readiness("cluster-a", ["node-a", "node-b"])

    assert not report.ready, "expected report.ready to be falsy"
    lines = fleet_pin_drift_metric_lines(
        SimpleNamespace(context=SimpleNamespace(fleet_registry=fleet))
    )
    assert (
        'gpu_fault_fleet_pin_drift_nodes{cluster_id="cluster-a",'
        'kind="PIN_AHEAD_OF_FLEET"} 2'
    ) in lines
    assert (
        'gpu_fault_fleet_pin_drift_nodes{cluster_id="cluster-a",kind="NODE_STALE"} 0'
    ) in lines


def test_a_stale_node_counts_under_node_stale_and_a_fixed_pin_clears() -> None:
    fleet = registry()
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    fleet.register(signed(copy_model(heartbeat("node-c"), artifact_sha256="b" * 64)))

    fleet.readiness("cluster-a", ["node-c"])
    lines = fleet_pin_drift_metric_lines(
        SimpleNamespace(context=SimpleNamespace(fleet_registry=fleet))
    )
    assert (
        'gpu_fault_fleet_pin_drift_nodes{cluster_id="cluster-a",kind="NODE_STALE"} 1'
    ) in lines
    assert (
        'gpu_fault_fleet_pin_drift_nodes{cluster_id="cluster-a",'
        'kind="PIN_AHEAD_OF_FLEET"} 0'
    ) in lines

    # The node catches up; the next evaluation clears the gauge.
    fleet.register(signed(heartbeat("node-c", observed_at=NOW + timedelta(seconds=5))))
    fleet.readiness("cluster-a", ["node-c"])
    lines = fleet_pin_drift_metric_lines(
        SimpleNamespace(context=SimpleNamespace(fleet_registry=fleet))
    )
    assert (
        'gpu_fault_fleet_pin_drift_nodes{cluster_id="cluster-a",kind="NODE_STALE"} 0'
    ) in lines


def test_the_pin_drift_family_is_registered_and_empty_without_a_registry() -> None:
    assert "fleet-pin-drift" in METRIC_CONTRIBUTORS.names
    lines = fleet_pin_drift_metric_lines(
        SimpleNamespace(context=SimpleNamespace(fleet_registry=None))
    )
    assert [line for line in lines if not line.startswith("#")] == []


def test_collector_top_node_series_carry_no_rank_label() -> None:
    rows = [
        {
            "cluster_id": "cluster-a",
            "node_id": f"node-{index}",
            "collector": "dcgm",
            "channel": "GPU_METRICS",
            "last_success_age_seconds": float(index),
            "silent": True,
            "erroring": False,
        }
        for index in range(3)
    ]

    lines = aggregate_lines(rows, top_n=2)

    top = [
        line for line in lines if line.startswith("gpu_fault_collector_silent_top_node")
    ]
    assert len(top) == 2, top
    assert not any("rank=" in line for line in top), (
        "a rank label re-keys the series every time two nodes swap places"
    )
    assert top[0].startswith(
        'gpu_fault_collector_silent_top_node{cluster_id="cluster-a",collector="dcgm",'
        'channel="GPU_METRICS",node_id="node-2"}'
    ), top


class _ExplodingExecutor:
    def __init__(self) -> None:
        self.config = SimpleNamespace(executor_id="executor-exploding")

    def execute(self, request_id, request):
        raise RuntimeError("adapter wiring gap")


def test_dispatch_internal_errors_carry_a_last_seen_stamp_for_sampled_pods() -> None:
    """control-worker runs four processes; /metrics sums the counter over the
    Pod, but a process restart still resets it, so the stamp is what an alert
    can max() across Pods without reading a reset as an event."""
    store = build_store()
    created_at = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    store.save_incident_and_workflow(
        fault_incident(
            "inc-boom",
            "event-boom",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id="wf-boom",
            fencing_token=1,
            created_at=created_at,
            updated_at=created_at,
        ),
        workflow_request(
            "wf-boom",
            "inc-boom",
            status=WorkflowStatus.PENDING,
            fencing_token=1,
            official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
            created_at=created_at,
            updated_at=created_at,
        ),
    )
    dispatcher = WorkflowDispatcher(
        store,
        _ExplodingExecutor(),  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, max_workers=1),
    )
    assert dispatcher.internal_error_last_seen_timestamp_seconds == 0.0

    before = time.time()
    report = dispatcher.run_once()

    assert report.internal_errors == 1, report
    assert dispatcher.internal_error_last_seen_timestamp_seconds >= before - 0.001
    lines = control_loop_metric_lines(
        SimpleNamespace(context=SimpleNamespace(store=store, dispatcher=dispatcher))
    )
    assert (
        _metric_value(
            lines,
            "gpu_fault_workflow_dispatch_internal_error_last_seen_timestamp_seconds",
        )
        >= before - 0.001
    )
    assert (
        _metric_value(
            lines,
            "gpu_fault_workflow_dispatch_failure_handling_abandoned_last_seen_"
            "timestamp_seconds",
        )
        == 0.0
    )


class _BrokenLeaseStore:
    def acquire_periodic_task_lease(self, *args, **kwargs):
        raise RuntimeError("could not connect to server")


def test_periodic_errors_carry_last_seen_stamps_by_kind_and_job(monkeypatch) -> None:
    runner = _runner(_OneTickStop())
    runner.context = SimpleNamespace(store=_BrokenLeaseStore())

    def boom(now: float) -> bool:
        raise RuntimeError("training scan exploded")

    # Every other job is a no-op here: the broken store would otherwise fail
    # the ones that read it too, and the point is the stamp per job.
    for name in runner._JOBS:
        monkeypatch.setattr(runner, f"_run_{name}", lambda now: False)
    monkeypatch.setattr(runner, "_run_training", boom)
    before = time.time()

    assert runner._due("cleanup", 1000.0, 1.0) is False
    runner.run_all_due(1000.0)

    snapshot = runner.metrics_snapshot()
    assert snapshot["lease_error_last_seen_timestamp_seconds"] >= before
    assert set(snapshot["job_error_last_seen_timestamp_seconds"]) == {"training"}
    lines = control_loop_metric_lines(
        SimpleNamespace(context=SimpleNamespace(periodic_runner=runner))
    )
    assert (
        _metric_value(
            lines, "gpu_fault_periodic_lease_error_last_seen_timestamp_seconds"
        )
        >= before - 0.001
    )
    assert any(
        line.startswith(
            'gpu_fault_periodic_job_error_last_seen_timestamp_seconds{job="training"} '
        )
        for line in lines
    ), lines
