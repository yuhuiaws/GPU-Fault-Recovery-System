"""One scrape must not die with one contributor, one role, or one semaphore.

Control-plane review 2026-09-08, G-2 / G-8 / G-12 / A-6 / E-1 / F-6 / H2-1 /
H2-2 / H2-3. Before this: a store exception in any of the fourteen
contributors returned 500 for the whole endpoint, taking the process-local
gauges (processor health, dispatcher liveness) down with it; ``/metrics``
queued on the shared ``store_io`` semaphore behind the work it was supposed
to observe; ingress replicas rendered the same fleet-level table scans as the
workers; and the scan cache's 10 s TTL never survived the ~60 s between two
scrapes of the same uvicorn worker.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app import metrics as metrics_module
from gpu_fault.app.aurora_refresh_metrics import aurora_credential_refresh_metric_lines
from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    remote_command_metric_lines,
)
from gpu_fault.app.metric_contributors import MetricContributorRegistry
from gpu_fault.app.metric_scan_cache import (
    OPEN_WORKFLOW_STATUSES,
    TERMINAL_WORKFLOW_STATUSES,
    MetricScanCache,
    scan_ttl_seconds_from_env,
)
from gpu_fault.app.metrics_sections import render_pool_metrics
from gpu_fault.models import WorkflowStatus
from tests._builders import asgi_client, build_store

ERRORS = "gpu_fault_metrics_contributor_errors_total"


def _registry(monkeypatch) -> MetricContributorRegistry:
    monkeypatch.setattr(
        "gpu_fault.app.metric_contributors.discover_plugins", lambda _group: {}
    )
    return MetricContributorRegistry()


def _scrape(app) -> str:
    async def scenario() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    return asyncio.run(scenario())


def test_a_failing_contributor_is_isolated_and_counted(monkeypatch) -> None:
    registry = _registry(monkeypatch)
    registry.register("core", lambda _runtime: ["core 1"])

    def broken(_runtime):
        raise RuntimeError("PoolTimeout: couldn't get a connection after 10s")

    registry.register("broken", broken)
    registry.register("after", lambda _runtime: ["after 1"])
    runtime = SimpleNamespace(background_services_enabled=True)

    lines = registry.render(runtime)

    assert "core 1" in lines and "after 1" in lines, lines
    assert f'{ERRORS}{{contributor="broken"}} 1' in lines, lines
    # Every registered contributor carries a series from the first scrape, so
    # increase() has a baseline the moment one of them starts failing.
    assert f'{ERRORS}{{contributor="core"}} 0' in lines, lines
    assert f"# TYPE {ERRORS} counter" in lines

    assert f'{ERRORS}{{contributor="broken"}} 2' in registry.render(runtime)
    assert registry.error_counts() == {"broken": 2}


def test_fleet_level_contributors_are_skipped_without_background_services(
    monkeypatch,
) -> None:
    registry = _registry(monkeypatch)
    registry.register("core", lambda _runtime: ["core 1"])
    registry.register("closed-loop", lambda _runtime: ["fleet 1"], fleet_level=True)

    ingress = registry.render(SimpleNamespace(background_services_enabled=False))
    worker = registry.render(SimpleNamespace(background_services_enabled=True))
    bare = registry.render(SimpleNamespace())

    assert "core 1" in ingress and "fleet 1" not in ingress, ingress
    assert "fleet 1" in worker, worker
    # A runtime that does not say which role it is (plugin contributors and
    # tests build one directly) renders everything, as before.
    assert "fleet 1" in bare, bare
    assert registry.fleet_level_names == ("closed-loop",)


def test_builtin_fleet_level_registrations_cover_the_store_scanning_contributors() -> (
    None
):
    fleet_level = set(metrics_module.METRIC_CONTRIBUTORS.fleet_level_names)
    assert {
        "remote-command",
        "fleet-rollout",
        "orchestration",
        "closed-loop",
        "completion-state",
    } <= fleet_level, fleet_level
    # Per-replica facts stay on every role: the collector gauge is only
    # published by ingress and the core family is what the queue alerts read.
    assert {"core", "collector-silence", "postgres-pool"}.isdisjoint(fleet_level), (
        "per-replica families must not be fleet-level"
    )


def test_metrics_endpoint_does_not_queue_behind_the_store_io_semaphore(
    monkeypatch,
) -> None:
    app = create_app(ApplicationContext(store=build_store()))
    runtime = app.state.runtime

    async def refuse(*_args, **_kwargs):
        raise AssertionError("/metrics was rendered through the shared store_io")

    monkeypatch.setattr(runtime.store_io, "run", refuse)

    text = _scrape(app)

    assert "gpu_fault_processor_queue_depth " in text


def test_scan_ttl_default_holds_the_store_to_one_aggregate_per_pod_per_minute(
    monkeypatch,
) -> None:
    # Four uvicorn processes per Pod and a 15 s scrape interval: each process
    # answers about one scrape a minute, so a 60 s TTL is one store aggregate
    # per process per minute (H2-3) however the scrapes land.
    monkeypatch.delenv("GPU_FAULT_METRICS_SCAN_TTL_SECONDS", raising=False)
    assert scan_ttl_seconds_from_env() >= 60.0


class _StatusRecordingStore:
    def __init__(self, rows_per_call: int) -> None:
        self.status_calls: list[frozenset[WorkflowStatus] | None] = []
        self.limits: list[int] = []
        self._rows = rows_per_call

    def list_workflows(self, statuses=None, *, limit=100, newest_first=False):
        self.status_calls.append(None if statuses is None else frozenset(statuses))
        self.limits.append(limit)
        assert newest_first is True
        return [
            SimpleNamespace(request_id=f"wf-{len(self.status_calls)}-{index}")
            for index in range(min(self._rows, limit))
        ]


def test_workflow_detail_scan_never_asks_for_every_status_at_once() -> None:
    # G-2: ``list_workflows(statuses=None, newest_first=True)`` has no index --
    # every updated_at index is partial per status -- so it read and sorted the
    # whole workflow kind on disk. The open slice walks the executable/BLOCKED
    # partial indexes; the terminal slice walks ``gpu_fault_workflow_updated_all``.
    store = _StatusRecordingStore(rows_per_call=2)
    scan = MetricScanCache(store, workflow_limit=3, ttl_seconds=0.0).workflows()

    assert None not in store.status_calls, store.status_calls
    assert store.status_calls == [
        frozenset(OPEN_WORKFLOW_STATUSES),
        frozenset(TERMINAL_WORKFLOW_STATUSES),
    ]
    assert OPEN_WORKFLOW_STATUSES | TERMINAL_WORKFLOW_STATUSES == set(WorkflowStatus)
    assert store.limits == [4, 4]
    assert len(scan.workflows) == 4
    assert scan.limit == 3
    assert scan.truncated is False

    truncated = MetricScanCache(
        _StatusRecordingStore(rows_per_call=4), workflow_limit=3, ttl_seconds=0.0
    ).workflows()
    assert len(truncated.workflows) == 6
    assert truncated.truncated is True


def test_shared_store_aggregates_are_cached_for_the_ttl() -> None:
    clock = [0.0]
    calls = [0]

    def produce() -> dict[str, int]:
        calls[0] += 1
        return {"n": calls[0]}

    cache = MetricScanCache(
        SimpleNamespace(), ttl_seconds=60.0, monotonic=lambda: clock[0]
    )

    assert cache.shared("stats", produce) == {"n": 1}
    assert cache.shared("stats", produce) == {"n": 1}
    assert calls[0] == 1
    clock[0] = 61.0
    assert cache.shared("stats", produce) == {"n": 2}


def _counting(store, name: str, counter: dict[str, int]):
    original = getattr(store, name)

    def wrapped(*args, **kwargs):
        counter[name] = counter.get(name, 0) + 1
        return original(*args, **kwargs)

    setattr(store, name, wrapped)


def test_orphan_and_notification_aggregates_go_through_the_scan_cache() -> None:
    store = build_store()
    counter: dict[str, int] = {}
    for name in (
        "list_orphan_workflows",
        "list_incidents_with_missing_workflow",
        "notification_status_counts",
        "notification_delivery_stats",
    ):
        _counting(store, name, counter)
    runtime = SimpleNamespace(
        context=ApplicationContext(store=store),
        metric_scan_cache=MetricScanCache(store, ttl_seconds=600.0),
    )

    first = closed_loop_metric_lines(runtime)
    closed_loop_metric_lines(runtime)

    assert "gpu_fault_orphan_workflows 0" in first
    assert "gpu_fault_incident_dangling_workflow_pointers 0" in first
    assert counter == {
        "list_orphan_workflows": 1,
        "list_incidents_with_missing_workflow": 1,
        "notification_status_counts": 1,
        "notification_delivery_stats": 1,
    }, counter


def test_remote_command_stats_go_through_the_scan_cache() -> None:
    calls = [0]

    def remote_command_stats():
        calls[0] += 1
        return {
            "by_status": {"PENDING": 1},
            "by_cluster_status": {"cluster-a": {"PENDING": 1}},
            "oldest_unclaimed_age_seconds_by_cluster": {"cluster-a": 4.0},
            "executor_internal_error_total": 0,
            "executor_internal_error_last_seen_timestamp_seconds": 0.0,
            "unclaimed_expired_total": 0,
        }

    store = SimpleNamespace(remote_command_stats=remote_command_stats)
    runtime = SimpleNamespace(
        context=SimpleNamespace(regional_mode=True, store=store),
        metric_scan_cache=MetricScanCache(store, ttl_seconds=600.0),
    )

    lines = remote_command_metric_lines(runtime)
    remote_command_metric_lines(runtime)

    assert 'gpu_fault_remote_command_total{status="PENDING"} 1' in lines
    assert calls[0] == 1


def test_spool_depth_is_reported_from_the_table_even_when_the_spool_is_off(
    monkeypatch,
) -> None:
    # E-1: the drain gate read depth from ingress Pods already rolled to
    # GPU_FAULT_TELEMETRY_SPOOL=false, which reported a constant 0 without
    # looking at the table, so the remaining rows were abandoned as "drained".
    store = build_store()
    store.telemetry_spool_stats = lambda **_kwargs: {
        "depth": 3,
        "leased": 1,
        "oldest_age_seconds": 5.0,
        "by_cluster": {"cluster-a": 3},
        "payload_bytes": 10,
        "leased_bytes": 4,
    }
    app = create_app(ApplicationContext(store=store))
    assert app.state.runtime.telemetry_spool_enabled is False

    text = _scrape(app)

    assert "gpu_fault_telemetry_spool_enabled 0" in text
    assert "gpu_fault_telemetry_spool_depth 3" in text, text
    assert "gpu_fault_telemetry_spool_leased 1" in text


def _processor_runtime() -> dict:
    return {
        "processed": {"success": 0, "error": 0},
        "lane_wait": {"attempt": {"count": 0, "sum": 0.0, "max": 0.0}},
        "duration_buckets": [],
        "duration_count": 0,
        "duration_sum": 0.0,
    }


def test_live_pool_statistics_are_rendered_when_the_pool_reports_them() -> None:
    # G-7: the only pool view was a lifetime checkout summary whose max never
    # falls; psycopg_pool.get_stats() says whether slots are being lost right
    # now (rotated password, failover) and whether callers are queueing.
    lines: list[str] = []
    render_pool_metrics(
        lines,
        {
            "checkout_count": 1,
            "checkout_sum_seconds": 0.0,
            "checkout_max_seconds": 0.0,
            "pool_size": 8,
            "pool_available": 2,
            "requests_waiting": 3,
            "requests_errors_total": 4,
            "connections_errors_total": 5,
            "connections_lost_total": 6,
            "credential_rotations_total": 1,
            "credential_source_file": 1,
        },
        _processor_runtime(),
    )

    assert "gpu_fault_postgres_pool_size 8" in lines, lines
    assert "gpu_fault_postgres_pool_available 2" in lines
    assert "gpu_fault_postgres_pool_requests_waiting 3" in lines
    assert "gpu_fault_postgres_pool_requests_errors_total 4" in lines
    assert "gpu_fault_postgres_pool_connections_errors_total 5" in lines
    assert "gpu_fault_postgres_pool_connections_lost_total 6" in lines
    assert "gpu_fault_postgres_credential_rotations_total 1" in lines
    assert "# TYPE gpu_fault_postgres_pool_connections_errors_total counter" in lines


def test_a_store_without_pool_statistics_renders_none_of_them() -> None:
    lines: list[str] = []
    render_pool_metrics(
        lines,
        {"checkout_count": 0, "checkout_sum_seconds": 0.0, "checkout_max_seconds": 0.0},
        _processor_runtime(),
    )

    assert not any("gpu_fault_postgres_pool_size" in line for line in lines), lines
    assert not any("gpu_fault_postgres_credential_" in line for line in lines), lines


AGE = "gpu_fault_aurora_credential_refresh_last_success_age_seconds"
RUN_AGE = "gpu_fault_aurora_credential_refresh_last_run_age_seconds"


def _status_file(tmp_path, monkeypatch, payload: str):
    # The Deployment mounts the whole gpu-fault-aurora Secret at the directory
    # GPU_FAULT_STORE_URL_FILE points into; the status key is its sibling.
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "postgres-url"))
    (tmp_path / "last-refresh-status.json").write_text(payload, encoding="utf-8")


def test_aurora_refresh_status_file_exports_the_run_and_success_ages(
    tmp_path, monkeypatch
) -> None:
    # H1-2: the refresher CronJob's outcome only lived in Job logs. It now
    # writes ``last-refresh-status.json`` into the Secret every consumer mounts;
    # the age of the last success is the alertable fact.
    finished = datetime.now(timezone.utc) - timedelta(hours=2)
    _status_file(
        tmp_path,
        monkeypatch,
        json.dumps(
            {
                "status": "ok",
                "finished_at": finished.isoformat(),
                "error": None,
                "rotated": True,
                "restarted": False,
                "reason": "credentials updated",
            }
        ),
    )

    lines = aurora_credential_refresh_metric_lines(SimpleNamespace())

    age = next(float(line.split()[-1]) for line in lines if line.startswith(AGE + " "))
    assert 7100.0 < age < 7400.0, lines
    run_age = next(
        float(line.split()[-1]) for line in lines if line.startswith(RUN_AGE + " ")
    )
    assert 7100.0 < run_age < 7400.0, lines
    assert "gpu_fault_aurora_credential_refresh_last_run_ok 1" in lines
    assert f"# TYPE {AGE} gauge" in lines


def test_a_failed_refresh_run_withholds_the_success_age(tmp_path, monkeypatch) -> None:
    # The file only records the last run, so a failed run cannot say when the
    # last success was; the series goes absent and last_run_ok says why.
    _status_file(
        tmp_path,
        monkeypatch,
        json.dumps(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": "verify: password authentication failed for user ***",
            }
        ),
    )

    lines = aurora_credential_refresh_metric_lines(SimpleNamespace())

    assert "gpu_fault_aurora_credential_refresh_last_run_ok 0" in lines
    assert any(line.startswith(RUN_AGE + " ") for line in lines), (
        "a failed run still reports its run age"
    )
    assert not any(line.startswith(AGE + " ") for line in lines), (
        "a failed run must not report a success age"
    )


def test_aurora_refresh_metrics_are_absent_without_a_status_file(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "postgres-url"))
    assert aurora_credential_refresh_metric_lines(SimpleNamespace()) == []

    _status_file(tmp_path, monkeypatch, "{not json")
    lines = aurora_credential_refresh_metric_lines(SimpleNamespace())
    assert "gpu_fault_aurora_credential_refresh_status_unreadable 1" in lines
    assert not any(line.startswith(AGE + " ") for line in lines), (
        "an unreadable status file must not report a success age"
    )


@pytest.mark.parametrize("name", ["aurora-credential-refresh"])
def test_builtin_registry_carries_the_new_contributor(name: str) -> None:
    assert name in metrics_module.METRIC_CONTRIBUTORS.names
