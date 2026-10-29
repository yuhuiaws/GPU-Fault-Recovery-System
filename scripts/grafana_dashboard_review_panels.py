"""Panels added by the control-plane and recovery quality reviews.

Kept apart from ``grafana_dashboard_catalog.py`` only because that file sits at
its size ratchet; the catalog imports these and places them in its rows. Same
conventions: every series carries the control-plane selector, store-derived
families are read with ``max``/``sum by`` exactly as the alert rules read them,
and the alert names, thresholds and runbook anchors are attached by the
generator, never written here.

The selector helpers are duplicated rather than imported from the catalog to
avoid a circular import; the catalog's tests check that every plotted family
survives the ADOT keep filter and is emitted by an exporter.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTROL_PLANE_SELECTOR = (
    'control_plane_cluster=~"$control_plane_cluster",region=~"$region"'
)
BY_CONTROL_PLANE = "by (control_plane_cluster, region)"


@dataclass(frozen=True)
class Target:
    expr: str
    legend: str = ""


@dataclass(frozen=True)
class Panel:
    title: str
    targets: tuple[Target, ...]
    unit: str = "short"
    kind: str = "timeseries"
    description: str = ""
    alert_names: tuple[str, ...] | None = None


def series(metric: str, *selectors: str) -> str:
    return metric + "{" + ",".join((*selectors, CONTROL_PLANE_SELECTOR)) + "}"


def _panel(
    title: str,
    targets: tuple[tuple[str, str], ...],
    description: str,
    *,
    unit: str = "short",
    alert_names: tuple[str, ...] | None = None,
) -> Panel:
    return Panel(
        title,
        tuple(Target(expr, legend) for expr, legend in targets),
        unit=unit,
        description=description,
        alert_names=alert_names,
    )


def _increase(metric: str, window: str, by: str, *selectors: str) -> str:
    return f"sum by ({by}) (increase({series(metric, *selectors)}[{window}]))"


# The catalog's "Processor queue" row. The depth alerts are ratios of the caps
# admission enforces (the exported *_limit / *_fault_reserved_depth gauges), so
# their threshold lines sit on the share panels and hold for any configured
# cap; the panels in requests plot the same families in another unit and bind
# only the alert whose boundary they draw. Caps are read with ``min`` over
# replicas, like the rules read them.
CLUSTER_SELECTOR = 'cluster_id=~"$cluster_id"'
_DEPTH = series("gpu_fault_processor_queue_depth")
_LIMIT = series("gpu_fault_processor_queue_depth_limit")
_RESERVE = series("gpu_fault_processor_queue_fault_reserved_depth")
_CLUSTER_DEPTH = series("gpu_fault_processor_cluster_queue_depth", CLUSTER_SELECTOR)
_CLUSTER_LIMIT = series("gpu_fault_processor_cluster_queue_depth_limit")
_CLUSTER_RESERVE = series("gpu_fault_processor_cluster_queue_fault_reserved_depth")
_SITE = "{{control_plane_cluster}} {{region}}"


def _reserve_boundary(limit: str, reserve: str, *, share: bool) -> tuple[str, str]:
    """Where admission starts refusing evidence and routine requests."""
    value = f"1 - {reserve} / {limit}" if share else f"{limit} - {reserve}"
    return f"min {BY_CONTROL_PLANE} ({value})", f"{_SITE} fault reserve boundary"


QUEUE_DEPTH_PANELS = (
    _panel(
        "Processor queue use",
        (
            (
                f"max {BY_CONTROL_PLANE} ({_DEPTH}) / min {BY_CONTROL_PLANE} ({_LIMIT})",
                f"{_SITE} depth / limit",
            ),
            _reserve_boundary(_LIMIT, _RESERVE, share=True),
        ),
        "Incomplete processor requests across the region as a share of the "
        "limit admission enforces.",
        unit="percentunit",
    ),
    _panel(
        "Processor queue depth",
        (
            (f"max {BY_CONTROL_PLANE} ({_DEPTH})", f"{_SITE} depth"),
            (f"min {BY_CONTROL_PLANE} ({_LIMIT})", f"{_SITE} limit"),
            _reserve_boundary(_LIMIT, _RESERVE, share=False),
        ),
        "Incomplete processor requests across the region.",
        alert_names=(),
    ),
    _panel(
        "Oldest processor request age",
        (
            (
                f"max({series('gpu_fault_processor_queue_oldest_age_seconds')})",
                "oldest",
            ),
        ),
        "",
        unit="s",
    ),
    _panel(
        "Processor queue use by cluster",
        (
            (
                "max by (control_plane_cluster, region, cluster_id) "
                f"({_CLUSTER_DEPTH}) / on (control_plane_cluster, region) "
                f"group_left min {BY_CONTROL_PLANE} ({_CLUSTER_LIMIT})",
                "{{cluster_id}}",
            ),
            _reserve_boundary(_CLUSTER_LIMIT, _CLUSTER_RESERVE, share=True),
        ),
        "Each cluster's incomplete processor requests as a share of the "
        "per-cluster limit.",
        unit="percentunit",
    ),
    _panel(
        "Processor queue depth by cluster",
        (
            (f"max by (cluster_id) ({_CLUSTER_DEPTH})", "{{cluster_id}}"),
            (f"min {BY_CONTROL_PLANE} ({_CLUSTER_LIMIT})", f"{_SITE} limit"),
            _reserve_boundary(_CLUSTER_LIMIT, _CLUSTER_RESERVE, share=False),
        ),
        "",
        alert_names=("GpuFaultProcessorClusterBacklogCritical",),
    ),
    _panel(
        "Oldest request age by cluster",
        (
            (
                "max by (cluster_id) ("
                + series(
                    "gpu_fault_processor_cluster_queue_oldest_age_seconds",
                    CLUSTER_SELECTOR,
                )
                + ")",
                "{{cluster_id}}",
            ),
        ),
        "The claim window is FIFO across the region, so a storm in one "
        "cluster delays every other cluster; this is who is paying.",
        unit="s",
    ),
)


def _counter_scan_known() -> str:
    stamp = series("gpu_fault_processor_counter_drift_scan_timestamp_seconds")
    maximum_age = series("gpu_fault_processor_counter_drift_scan_max_age_seconds")
    age = f"time() - {stamp}"
    return (
        f"({stamp} > 0) and "
        f"({stamp} == on (control_plane_cluster, region) group_left "
        f"max {BY_CONTROL_PLANE} ({stamp})) "
        f"and ({age} >= 0) and ({maximum_age} > 0) "
        f"and ({age} <= {maximum_age}) "
        f"and ({series('gpu_fault_processor_counter_drift_abs')} >= 0) "
        f"and ({series('gpu_fault_processor_counter_mismatched_clusters')} >= 0)"
    )


def _counter_drift(metric: str) -> str:
    return (
        f"min_over_time((max {BY_CONTROL_PLANE} "
        f"({series(metric)} and ({_counter_scan_known()})))[5m:1m]) "
        "and on (control_plane_cluster, region) "
        f"max {BY_CONTROL_PLANE} ({_counter_scan_known()})"
    )


def _counter_scan_unavailable() -> str:
    workers = series(
        "up", 'job="gpu-fault-control-plane"', 'service_role="gpu-fault-control-worker"'
    )
    expected = f"max {BY_CONTROL_PLANE} ({workers} == 1)"
    return (
        f"({expected} unless on (control_plane_cluster, region) "
        f"max {BY_CONTROL_PLANE} ({_counter_scan_known()})) or (0 * {expected})"
    )


COUNTER_DRIFT_PANEL = _panel(
    "Processor counter drift",
    (
        (_counter_drift("gpu_fault_processor_counter_drift_abs"), "drift"),
        (
            _counter_drift("gpu_fault_processor_counter_mismatched_clusters"),
            "mismatched clusters",
        ),
    ),
    "Five-minute minimum from the newest complete, fresh scan. An old task "
    "owner cannot preserve repaired drift; unavailable scans are shown separately.",
    alert_names=("GpuFaultProcessorCounterDrift",),
)

COUNTER_SCAN_PANEL = _panel(
    "Processor counter scan unavailable",
    ((_counter_scan_unavailable(), "{{control_plane_cluster}} {{region}}"),),
    "1 when no current complete counter scan is available; 0 when one is known.",
    alert_names=("GpuFaultProcessorCounterDriftScanUnavailable",),
)

_TERMINAL_FAILURE = (
    f"max {BY_CONTROL_PLANE} (max_over_time("
    f"{series('gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds')}"
    "[15m]))"
)
NOTIFICATION_FAILURE_PANEL = _panel(
    "Terminal notification failure seen (15m)",
    (
        (
            f"((time() - {_TERMINAL_FAILURE}) < bool 900) "
            f"* ({_TERMINAL_FAILURE} > bool 0)",
            "{{control_plane_cluster}} {{region}}",
        ),
    ),
    "1 after a recorded terminal delivery failure in the last 15 minutes. "
    "Deleting older failures or retaining DEAD rows does not create an event.",
    alert_names=("GpuFaultNotificationDeliveryFailing",),
)

_AGGREGATION_DEGRADED = series("gpu_fault_metrics_aggregation_degraded")
_SCRAPED_CONTROL_PLANE = series("up", 'job="gpu-fault-control-plane"')
METRIC_COVERAGE_PANEL = _panel(
    "Process metric coverage incomplete",
    (
        (
            "max by (control_plane_cluster, region, pod, service_role) ("
            f"{_AGGREGATION_DEGRADED} or "
            f"(({_SCRAPED_CONTROL_PLANE} == 1) "
            "unless on (control_plane_cluster, region, pod, instance, service_role) "
            f"{_AGGREGATION_DEGRADED}))",
            "{{pod}} {{service_role}}",
        ),
    ),
    "1 for stale, missing or invalid process publications, or a missing merger "
    "verdict on a reachable Pod. Unknown process coverage is not zero activity.",
    alert_names=("GpuFaultMetricsAggregationIncomplete",),
)


def _containment_window_known() -> str:
    mean = series(
        "gpu_fault_closed_loop_milestone_window_mean_seconds",
        'milestone="containment"',
    )
    count = series(
        "gpu_fault_closed_loop_milestone_window_count", 'milestone="containment"'
    )
    complete = series(
        "gpu_fault_closed_loop_milestone_window_complete", 'milestone="containment"'
    )
    age = f"time() - {series('gpu_fault_closed_loop_window_end_timestamp_seconds')}"
    return (
        f"({complete} == 1) and "
        f"(({count} == 0) or (({count} > 0) and ({mean} >= 0))) "
        f"and ignoring (milestone) (({age} >= 0) and ({age} <= 120))"
    )


def _containment_window_mean() -> str:
    mean = series(
        "gpu_fault_closed_loop_milestone_window_mean_seconds",
        'milestone="containment"',
    )
    count = series(
        "gpu_fault_closed_loop_milestone_window_count", 'milestone="containment"'
    )
    return (
        f"max {BY_CONTROL_PLANE} "
        f"({mean} and ({count} > 0) and ({_containment_window_known()}))"
    )


def _containment_window_unavailable() -> str:
    workers = series(
        "up", 'job="gpu-fault-control-plane"', 'service_role="gpu-fault-control-worker"'
    )
    expected = f"({series('gpu_fault_workflow_scan_limit')} or ({workers} == 1))"
    return (
        f"count {BY_CONTROL_PLANE} ("
        f"{expected} unless on "
        "(control_plane_cluster, region, job, namespace, pod, instance) "
        f"({_containment_window_known()})) "
        f"or (0 * max {BY_CONTROL_PLANE} ({expected}))"
    )


CONTAINMENT_LATENCY_PANEL = _panel(
    "Containment latency (6h moving mean)",
    ((_containment_window_mean(), "{{control_plane_cluster}} {{region}}"),),
    "Creation to first successful containment in the event-time window. "
    "Empty, incomplete or stale windows have no mean.",
    unit="s",
    alert_names=("GpuFaultClosedLoopSlow",),
)

CONTAINMENT_WINDOW_PANEL = _panel(
    "Unavailable containment windows",
    ((_containment_window_unavailable(), "{{control_plane_cluster}} {{region}}"),),
    "Workers lacking a complete, fresh containment window. "
    "A missing latency mean does not establish healthy recovery.",
    alert_names=("GpuFaultClosedLoopWindowIncomplete",),
)


COLLECTOR_MEMORY_PANEL = _panel(
    "Collector memory",
    (
        (
            f"max {BY_CONTROL_PLANE} ({series('otelcol_process_memory_rss')})",
            "{{control_plane_cluster}} {{region}} rss",
        ),
    ),
    "Collector resident memory against its 256Mi limit (GOMEMLIMIT 220MiB); an "
    "OOMKill loop still lands a batch per restart, so absence alerts stay quiet "
    "(H2-4).",
    unit="bytes",
)

CONTRIBUTOR_FAILURES_PANEL = _panel(
    "/metrics contributor failures",
    (
        (
            _increase(
                "gpu_fault_metrics_contributor_errors_total", "10m", "pod, contributor"
            ),
            "{{pod}} {{contributor}}",
        ),
    ),
    "Scrapes on which one /metrics contributor raised and its families were "
    "skipped while the rest still rendered (G-8).",
)

CONSUMER_LIVENESS_PANEL = _panel(
    "Processor consumer loop liveness",
    (
        (
            "min by (pod) ("
            + series(
                "gpu_fault_processor_consumer_running",
                'pod=~"gpu-fault-control-worker-.*"',
            )
            + ")",
            "{{pod}} running",
        ),
        (
            "max by (pod) ("
            + series(
                "gpu_fault_processor_consumer_last_cycle_age_seconds",
                'pod=~"gpu-fault-control-worker-.*"',
            )
            + ")",
            "{{pod}} last cycle age (s)",
        ),
    ),
    "Whether each control-worker's queue consumer loop is alive and how long "
    "since it last started a claim cycle; a dead or wedged loop kept the health "
    "gauge at 1 before B-6.",
)

POOL_AND_CREDENTIAL_PANELS = (
    _panel(
        "PostgreSQL pool oversubscription ratio",
        (
            (
                "max by (service_role) ("
                + series("gpu_fault_postgres_pool_oversubscription_ratio")
                + ")",
                "{{service_role}}",
            ),
        ),
        "Threads that can hold a pooled connection over the pool ceiling, per "
        "role. A configuration constant, not an alert: both roles sit above 1 "
        "by design and admission_runtime logs the same figure as a startup "
        "WARNING; whether anyone actually waits is the next panel (G-7).",
        unit="percentunit",
    ),
    _panel(
        "PostgreSQL pool demand by consumer",
        (
            (
                "max by (service_role, consumer) ("
                + series("gpu_fault_postgres_pool_demand_connections")
                + ")",
                "{{service_role}} {{consumer}}",
            ),
            (
                "max by (service_role) ("
                + series("gpu_fault_postgres_pool_max_size")
                + ")",
                "{{service_role}} pool ceiling",
            ),
            (
                "max by (service_role) ("
                + series("gpu_fault_postgres_pool_headroom_connections")
                + ")",
                "{{service_role}} headroom",
            ),
        ),
        "The ratio's numerator broken down by thread family next to the pool "
        "ceiling and the remaining headroom (negative when the threads outnumber "
        "the pool); read this before changing a role's worker count or "
        "POOL_MAX_SIZE.",
    ),
    _panel(
        "PostgreSQL pool callers waiting",
        (
            (
                "max by (service_role, pod) ("
                + series("gpu_fault_postgres_pool_requests_waiting")
                + ")",
                "{{service_role}} {{pod}}",
            ),
        ),
        "Callers blocked on pool checkout right now; sustained above 0 the pool "
        "is saturated or losing slots (G-7).",
    ),
    _panel(
        "PostgreSQL pool connection errors",
        (
            (
                _increase(
                    "gpu_fault_postgres_pool_connections_errors_total",
                    "5m",
                    "service_role, pod",
                ),
                "{{service_role}} {{pod}}",
            ),
        ),
        "Failed attempts to open a pooled connection: a rotated password the "
        "Secret has not caught up with, or a failover (G-7 / CP-3).",
    ),
    _panel(
        "Aurora credential refresh",
        (
            (
                f"max {BY_CONTROL_PLANE} ("
                + series("gpu_fault_aurora_credential_refresh_last_success_age_seconds")
                + ")",
                "{{control_plane_cluster}} {{region}} last success age (s)",
            ),
            (
                f"min {BY_CONTROL_PLANE} ("
                + series("gpu_fault_aurora_credential_refresh_last_run_ok")
                + ")",
                "{{control_plane_cluster}} {{region}} last run ok",
            ),
        ),
        "Seconds since the hourly refresher CronJob last published the Aurora "
        "master password, and whether its latest run succeeded, from the status "
        "file in the mounted Secret (H1-2).",
        unit="s",
    ),
    _panel(
        "Aurora refresh evidence unknown",
        (
            (
                f"max {BY_CONTROL_PLANE} ("
                + series("gpu_fault_aurora_credential_refresh_status_unreadable")
                + ")",
                "{{control_plane_cluster}} {{region}}",
            ),
        ),
        "A configured status projection is missing, unreadable, malformed or "
        "future-dated. This is unknown evidence, not a successful refresh.",
        alert_names=("GpuFaultAuroraCredentialRefreshStatusUnknown",),
    ),
)

REVIEW_COUNTER_PANELS = (
    _panel(
        "Stale-fence remote command sweeps",
        (
            (
                _increase(
                    "gpu_fault_periodic_cleanup_rows_total",
                    "30m",
                    "control_plane_cluster, region",
                    'periodic_job="stale_fence_remote_commands"',
                ),
                "{{control_plane_cluster}} {{region}}",
            ),
        ),
        "LEASED remote commands whose workflow moved to another fencing token "
        "and whose lease lapsed unreported, failed by the periodic sweep with "
        "status_source=stale-fence (D-9).",
    ),
    _panel(
        "Notification delivery errors",
        (
            (
                _increase(
                    "gpu_fault_notification_delivery_errors_total",
                    "15m",
                    "control_plane_cluster, region",
                ),
                "{{control_plane_cluster}} {{region}}",
            ),
        ),
        "Outbox deliveries whose provider call or bookkeeping raised; the row "
        "was released for retry and the batch continued (F-3).",
    ),
    _panel(
        "Control-record archive errors",
        (
            (
                _increase(
                    "gpu_fault_control_record_archive_errors_total",
                    "1h",
                    "control_plane_cluster, region, reason",
                ),
                "{{reason}}",
            ),
        ),
        "Archive candidates that failed with an exception other than a safety "
        "refusal, by exception type (F-8).",
    ),
)


# The remediation budget row's per-workflow wait and the storm breaker
# (detailed design v2 §9.5).
_STORM = "gpu_fault_remediation_budget_storm_breaker"
_CLUSTER = 'cluster_id=~"$cluster_id"'
REMEDIATION_STORM_PANEL = _panel(
    "Remediation storm breaker",
    (
        (
            f"max by (cluster_id) ({series(_STORM + '_open', _CLUSTER)})",
            "{{cluster_id}} open",
        ),
        (
            f"max by (cluster_id) ({series(_STORM + '_held_workflows', _CLUSTER)})",
            "{{cluster_id}} held",
        ),
    ),
    "1 while a cluster's storm breaker holds its new destructive remediation, "
    "and the workflows it holds.",
)
REMEDIATION_LONGEST_WAIT_PANEL = _panel(
    "Longest remediation budget wait",
    (
        (
            "max by (scope_type) ("
            + series("gpu_fault_remediation_budget_longest_wait_seconds")
            + ")",
            "{{scope_type}}",
        ),
    ),
    "How long the oldest current wait has lasted, by the type of the scope that "
    "last refused it, from the refusal that began the wait.",
    unit="s",
)
