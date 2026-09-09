"""Panels added by the control-plane review 2026-09-08.

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


def series(metric: str, *selectors: str) -> str:
    return metric + "{" + ",".join((*selectors, CONTROL_PLANE_SELECTOR)) + "}"


def _panel(
    title: str,
    targets: tuple[tuple[str, str], ...],
    description: str,
    *,
    unit: str = "short",
) -> Panel:
    return Panel(
        title,
        tuple(Target(expr, legend) for expr, legend in targets),
        unit=unit,
        description=description,
    )


def _increase(metric: str, window: str, by: str, *selectors: str) -> str:
    return f"sum by ({by}) (increase({series(metric, *selectors)}[{window}]))"


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
                f"max {BY_CONTROL_PLANE} ("
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
                    'job="stale_fence_remote_commands"',
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
