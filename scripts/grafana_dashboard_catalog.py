"""Declarative table of the GPU fault Grafana dashboards.

One dashboard per ``amp-rules.yaml`` group plus an overview. Each panel names
its PromQL and legend; the alert thresholds, the alert names in the panel
description and the runbook anchors are NOT written here -- the generator in
``build-grafana-dashboards.py`` reads them out of the rule file and attaches
them to whichever panel plots the alert's metric families, so a threshold that
moves in the rule moves on the panel in the same commit.

Every series carries the ``control_plane_cluster``/``region`` selector because
ADOT stamps both labels on every sample it forwards and the alert rules group
on them; every per-cluster family carries the ``cluster_id`` selector as well.
The helpers below are the only way the table spells a selector, so a panel
cannot forget one.

Store-derived gauges are published identically by every replica, so they are
read with ``max`` (never ``sum``) exactly as the alert rules read them.
Every family plotted here must survive ADOT's keep filter, or the panel is
permanently empty; ``tests/test_grafana_dashboards.py`` checks that.
"""

from __future__ import annotations

from dataclasses import dataclass

from grafana_dashboard_review_panels import (
    COLLECTOR_MEMORY_PANEL,
    CONSUMER_LIVENESS_PANEL,
    CONTRIBUTOR_FAILURES_PANEL,
    POOL_AND_CREDENTIAL_PANELS,
    REVIEW_COUNTER_PANELS,
)

CONTROL_PLANE_SELECTOR = (
    'control_plane_cluster=~"$control_plane_cluster",region=~"$region"'
)
CLUSTER_SELECTOR = 'cluster_id=~"$cluster_id"'
BY_CONTROL_PLANE = "by (control_plane_cluster, region)"
#: Series the per-GPU-cluster collector (adot-dataplane.yaml) delivers carry
#: the cluster they were scraped in; the rules on them group on it too.
BY_GPU_CLUSTER = "by (control_plane_cluster, region, gpu_cluster)"


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


@dataclass(frozen=True)
class Row:
    title: str
    panels: tuple[Panel, ...]


@dataclass(frozen=True)
class Dashboard:
    uid: str
    title: str
    rows: tuple[Row, ...]
    group: str | None = None


def series(metric: str, *selectors: str) -> str:
    """``metric{control_plane_cluster=~..., region=~..., <selectors>}``."""
    return metric + "{" + ",".join((*selectors, CONTROL_PLANE_SELECTOR)) + "}"


def cluster_series(metric: str, *selectors: str) -> str:
    """A per-cluster family: ``series`` plus the ``cluster_id`` selector."""
    return series(metric, CLUSTER_SELECTOR, *selectors)


def increase_by_control_plane(metric: str, window: str) -> str:
    return f"sum {BY_CONTROL_PLANE} (increase({series(metric)}[{window}]))"


def _seconds_since(metric: str) -> str:
    """Age of a ``*_timestamp_seconds`` gauge, read the way the alerts read it.

    Process-level counters on the four-worker Pods are sampled one process per
    scrape, so the liveness and last-seen alerts compare ``time()`` against the
    newest stamp any process published instead of taking ``increase()``.
    """
    return f"time() - max {BY_CONTROL_PLANE} ({series(metric)})"


def _queue_depth_panels() -> tuple[Panel, ...]:
    return (
        Panel(
            "Processor queue depth",
            (Target(f"max({series('gpu_fault_processor_queue_depth')})", "depth"),),
            description="Incomplete processor requests across the region.",
        ),
        Panel(
            "Oldest processor request age",
            (
                Target(
                    f"max({series('gpu_fault_processor_queue_oldest_age_seconds')})",
                    "oldest",
                ),
            ),
            unit="s",
        ),
        Panel(
            "Processor queue depth by cluster",
            (
                Target(
                    "max by (cluster_id) "
                    f"({cluster_series('gpu_fault_processor_cluster_queue_depth')})",
                    "{{cluster_id}}",
                ),
            ),
        ),
        Panel(
            "Oldest request age by cluster",
            (
                Target(
                    "max by (cluster_id) ("
                    + cluster_series(
                        "gpu_fault_processor_cluster_queue_oldest_age_seconds"
                    )
                    + ")",
                    "{{cluster_id}}",
                ),
            ),
            unit="s",
            description=(
                "The claim window is FIFO across the region, so a storm in one "
                "cluster delays every other cluster; this is who is paying."
            ),
        ),
    )


def _remediation_budget_panels() -> tuple[Panel, ...]:
    return (
        Panel(
            "Remediation budget: active claims vs limit",
            (
                Target(
                    "max by (cluster_id) ("
                    + cluster_series(
                        "gpu_fault_remediation_budget_cluster_active_claims"
                    )
                    + ")",
                    "{{cluster_id}}",
                ),
                Target(
                    f"max({series('gpu_fault_remediation_budget_cluster_limit')})",
                    "limit",
                ),
            ),
        ),
        Panel(
            "Workflows waiting for cluster budget",
            (
                Target(
                    "max by (cluster_id) ("
                    + cluster_series(
                        "gpu_fault_remediation_budget_cluster_waiting_workflows"
                    )
                    + ")",
                    "{{cluster_id}}",
                ),
            ),
        ),
    )


def _store_io_panel() -> Panel:
    return Panel(
        "Store I/O saturation",
        (
            Target(
                "max by (pod) ("
                + series("gpu_fault_store_io_in_flight")
                + " / "
                + series("gpu_fault_store_io_max_in_flight")
                + ")",
                "{{pod}}",
            ),
        ),
        unit="percentunit",
        description=(
            "In-flight Store calls over the Pod's admission capacity. api-ha "
            "and control-worker run four uvicorn workers behind one port; "
            "/metrics sums both gauges over the Pod's live processes."
        ),
    )


OVERVIEW = Dashboard(
    uid="gpu-fault-overview",
    title="GPU Fault · Overview",
    rows=(
        Row(
            "Alerts",
            (
                Panel(
                    "Alerts firing",
                    (
                        Target(
                            'count by (alertname) (ALERTS{alertstate="firing"})',
                            "{{alertname}}",
                        ),
                    ),
                    kind="stat",
                    description=(
                        "Every AMP alert currently firing; the name is the "
                        "runbook card in docs/管理员日常运维.md §8."
                    ),
                ),
            ),
        ),
        Row(
            "Closed loop",
            (
                Panel(
                    "Incidents by state",
                    (
                        Target(
                            f"max by (state) ({series('gpu_fault_incidents_by_state')})",
                            "{{state}}",
                        ),
                    ),
                    description="ESCALATED is the operator queue.",
                ),
                Panel(
                    "Workflows by status",
                    (
                        Target(
                            f"max by (status) ({series('gpu_fault_workflow_total')})",
                            "{{status}}",
                        ),
                    ),
                ),
                Panel(
                    "Orphan workflows",
                    (
                        Target(
                            f"max({series('gpu_fault_orphan_workflows')})", "orphans"
                        ),
                    ),
                    description=(
                        "PENDING or SAFETY_PENDING workflows whose incident no "
                        "longer points at them."
                    ),
                ),
                Panel(
                    "Notification outbox depth",
                    (
                        Target(
                            f"max({series('gpu_fault_notification_outbox_depth')})",
                            "outbox",
                        ),
                    ),
                    description="Notifications without a terminal delivery result.",
                ),
            ),
        ),
        Row("Processor queue", _queue_depth_panels()),
        Row("Remediation budget", _remediation_budget_panels()),
        Row("Store I/O", (_store_io_panel(),)),
        Row(
            "Declared topology",
            (
                Panel(
                    "Declared GPU node counts",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_capacity_largest_cluster_node_count")
                            + ")",
                            "largest cluster",
                        ),
                        Target(
                            f"max({series('gpu_fault_capacity_managed_node_count')})",
                            "managed total",
                        ),
                    ),
                    kind="stat",
                    description=(
                        "The declared fleet topology the fault reserve and the "
                        "Aurora capacity floor derive from: node count of the "
                        "largest managed GPU cluster and across every managed "
                        "cluster. 0 means the release did not declare them."
                    ),
                ),
            ),
        ),
    ),
)

COLLECTOR_HEALTH = Dashboard(
    uid="gpu-fault-collector-health",
    title="GPU Fault · Collector health",
    group="gpu-fault-collector-health",
    rows=(
        Row(
            "Collector delivery",
            (
                Panel(
                    "Silent collector nodes",
                    (
                        Target(
                            "max by (cluster_id, collector, channel) ("
                            + cluster_series("gpu_fault_collector_silent_nodes")
                            + ")",
                            "{{cluster_id}} {{collector}}/{{channel}}",
                        ),
                    ),
                    description=(
                        "Nodes whose channel has not delivered a batch within its "
                        "silence threshold; the policy is blind to them."
                    ),
                ),
                Panel(
                    "Erroring collector nodes",
                    (
                        Target(
                            "max by (cluster_id, collector, channel) ("
                            + cluster_series("gpu_fault_collector_erroring_nodes")
                            + ")",
                            "{{cluster_id}} {{collector}}/{{channel}}",
                        ),
                    ),
                    description=(
                        "Nodes still delivering on time but reporting a collection "
                        "error every cycle."
                    ),
                ),
            ),
        ),
        Row(
            "Freshness",
            (
                Panel(
                    "Oldest last-success age by channel",
                    (
                        Target(
                            "max by (cluster_id, collector, channel) ("
                            + cluster_series(
                                "gpu_fault_collector_last_success_age_seconds_max"
                            )
                            + ")",
                            "{{cluster_id}} {{collector}}/{{channel}}",
                        ),
                    ),
                    unit="s",
                ),
                Panel(
                    "Collector metrics snapshot age",
                    (
                        Target(
                            "min("
                            + series("gpu_fault_collector_metrics_snapshot_age_seconds")
                            + ")",
                            "snapshot age",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Age of the collector-status snapshot the two gauges above "
                        "are computed from; stale means they are stale too."
                    ),
                ),
            ),
        ),
    ),
)

TELEMETRY_PIPELINE = Dashboard(
    uid="gpu-fault-telemetry-pipeline",
    title="GPU Fault · Telemetry pipeline",
    group="gpu-fault-telemetry-pipeline",
    rows=(
        Row(
            "Remote write",
            (
                Panel(
                    "Remote-write failed points",
                    (
                        Target(
                            f"sum {BY_CONTROL_PLANE} (rate("
                            + series("otelcol_exporter_send_failed_metric_points")
                            + "[5m]))",
                            "{{control_plane_cluster}} {{region}} failed/s",
                        ),
                        Target(
                            f"sum {BY_CONTROL_PLANE} (rate("
                            + series("otelcol_exporter_sent_metric_points")
                            + "[5m]))",
                            "{{control_plane_cluster}} {{region}} sent/s",
                        ),
                    ),
                    unit="short",
                ),
                Panel(
                    "Remote-write queue fill",
                    (
                        Target(
                            f"max {BY_CONTROL_PLANE} ("
                            + series("otelcol_exporter_queue_size")
                            + " / "
                            + series("otelcol_exporter_queue_capacity")
                            + ")",
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="percentunit",
                ),
                COLLECTOR_MEMORY_PANEL,
            ),
        ),
        Row(
            "Scrape liveness",
            (
                Panel(
                    "ADOT self-telemetry up",
                    (
                        Target(
                            series("up", 'job="gpu-fault-adot-self"'),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    kind="stat",
                    description=(
                        "0 or absent: the collector's own pipeline counters are "
                        "not reaching AMP, so the two panels above are blind."
                    ),
                ),
                Panel(
                    "Data-plane scrape up by GPU cluster",
                    (
                        Target(
                            f"max {BY_GPU_CLUSTER} ("
                            + series("up", 'job="gpu-fault-dataplane"')
                            + ")",
                            "{{gpu_cluster}}",
                        ),
                    ),
                    kind="stat",
                    description=(
                        "1 per GPU cluster whose ADOT collector "
                        "(adot-dataplane.yaml) scrapes at least one data-plane "
                        "target. 0 or a missing cluster: the Completion Watcher "
                        "row below is blind for it. A cluster whose release "
                        "skipped the collector (no adot_irsa_role_arn) never "
                        "appears here."
                    ),
                ),
            ),
        ),
        Row(
            "Completion Watcher",
            (
                Panel(
                    "Completion attempt state unavailable",
                    (
                        Target(
                            f"max {BY_GPU_CLUSTER} ("
                            + series("gpu_fault_completion_active_state_unavailable")
                            + ")",
                            "{{gpu_cluster}}",
                        ),
                    ),
                    kind="stat",
                    description=(
                        "1 while the routine attempt-state ConfigMap or the Role "
                        "that names it is refusing. Delivery keeps working, so "
                        "this is the only place the outage is visible."
                    ),
                ),
                Panel(
                    "Completion write-ahead append failures (10m increase)",
                    (
                        Target(
                            f"sum {BY_GPU_CLUSTER} (increase("
                            + series(
                                "gpu_fault_completion_outbox_append_failures_total"
                            )
                            + "[10m]))",
                            "{{gpu_cluster}}",
                        ),
                    ),
                    description=(
                        "Each step is a critical completion event that was "
                        "delivered without a write-ahead copy, so only a watcher "
                        "restart turns it into a lost or replayed event."
                    ),
                ),
            ),
        ),
    ),
)

CONTROL_PLANE_CAPACITY = Dashboard(
    uid="gpu-fault-control-plane-capacity",
    title="GPU Fault · Control-plane capacity",
    group="gpu-fault-control-plane-capacity",
    rows=(
        Row(
            "Scrape liveness",
            (
                Panel(
                    "Control-plane pods up",
                    (
                        Target(
                            series("up", 'job="gpu-fault-control-plane"'),
                            "{{pod}}",
                        ),
                    ),
                    description=(
                        "One series per scraped Pod across api-ha, control-worker "
                        "and telemetry-spool-worker."
                    ),
                ),
                CONTRIBUTOR_FAILURES_PANEL,
                CONSUMER_LIVENESS_PANEL,
            ),
        ),
        Row("PostgreSQL pool and credentials", POOL_AND_CREDENTIAL_PANELS),
        Row("Control-loop review counters", REVIEW_COUNTER_PANELS),
        Row("Processor queue", _queue_depth_panels()),
        Row(
            "Metric scan truncation",
            (
                Panel(
                    "Workflow metric scan truncated",
                    (
                        Target(
                            f"max({series('gpu_fault_workflow_scan_truncated')})",
                            "truncated",
                        ),
                    ),
                    description=(
                        "1 means the workflow detail gauges cover only the newest "
                        "slice of the table."
                    ),
                ),
                Panel(
                    "Attempt observation scan truncated",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_attempt_observation_scan_truncated")
                            + ")",
                            "truncated",
                        ),
                    ),
                ),
            ),
        ),
        Row(
            "Admission and Store I/O",
            (
                Panel(
                    "Processor admission rejections",
                    (
                        Target(
                            "sum by (pod) (rate("
                            + series("gpu_fault_processor_admission_rejections_total")
                            + "[5m]))",
                            "{{pod}}",
                        ),
                    ),
                    description="Requests rejected before enqueue, per second.",
                ),
                _store_io_panel(),
                Panel(
                    "Store I/O rejections",
                    (
                        Target(
                            "sum by (pod, reason) (rate("
                            + series("gpu_fault_store_io_rejections_total")
                            + "[5m]))",
                            "{{pod}} {{reason}}",
                        ),
                    ),
                ),
            ),
        ),
        Row(
            "Latency, leases and counters",
            (
                Panel(
                    "Processor request p95",
                    (
                        Target(
                            "histogram_quantile(0.95, sum by (le) (rate("
                            + series(
                                "gpu_fault_processor_request_processing_seconds_bucket"
                            )
                            + "[5m])))",
                            "p95",
                        ),
                    ),
                    unit="s",
                ),
                Panel(
                    "Expired leases reclaimed (10m)",
                    (
                        Target(
                            increase_by_control_plane(
                                "gpu_fault_processor_expired_leases_reclaimed_total",
                                "10m",
                            ),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                ),
                Panel(
                    "Processor counter drift",
                    (
                        Target(
                            f"max {BY_CONTROL_PLANE} (min_over_time("
                            + series("gpu_fault_processor_counter_drift_abs")
                            + "[5m]))",
                            "drift",
                        ),
                        Target(
                            f"max {BY_CONTROL_PLANE} (min_over_time("
                            + series("gpu_fault_processor_counter_mismatched_clusters")
                            + "[5m]))",
                            "mismatched clusters",
                        ),
                    ),
                    description=(
                        "Sustained for a full 5m window; a transient mismatch "
                        "during a reconcile is not drift."
                    ),
                ),
            ),
        ),
        Row(
            "Telemetry spool",
            (
                Panel(
                    "Spool NOTIFY listener enabled",
                    (
                        Target(
                            "min("
                            + series(
                                "gpu_fault_telemetry_spool_notifications_enabled",
                                'pod=~"gpu-fault-telemetry-spool-worker-.*"',
                            )
                            + ")",
                            "enabled",
                        ),
                    ),
                    kind="stat",
                ),
                Panel(
                    "Spool samples dropped (5m)",
                    (
                        Target(
                            "increase("
                            + series("gpu_fault_telemetry_spool_dropped_total")
                            + "[5m])",
                            "{{pod}}",
                        ),
                    ),
                ),
                Panel(
                    "Spool admissions rejected (5m)",
                    (
                        Target(
                            "sum(increase("
                            + series("gpu_fault_telemetry_spool_rejected_total")
                            + "[5m]))",
                            "rejected",
                        ),
                    ),
                ),
                Panel(
                    "Spool replay errors (5m)",
                    (
                        Target(
                            "increase("
                            + series("gpu_fault_telemetry_spool_errors_total")
                            + "[5m])",
                            "{{pod}}",
                        ),
                    ),
                ),
                Panel(
                    "Oldest spooled sample age",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_telemetry_spool_oldest_age_seconds")
                            + ")",
                            "oldest",
                        ),
                    ),
                    unit="s",
                ),
            ),
        ),
        Row(
            "Loop liveness",
            (
                Panel(
                    "Workflow dispatcher last cycle age",
                    (
                        Target(
                            _seconds_since(
                                "gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds"
                            ),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Seconds since any dispatcher replica stamped a cycle; "
                        "a lease held elsewhere still stamps."
                    ),
                ),
                Panel(
                    "Periodic runner last cycle age",
                    (
                        Target(
                            _seconds_since(
                                "gpu_fault_periodic_last_cycle_timestamp_seconds"
                            ),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="s",
                ),
                Panel(
                    "Processor healthy per worker Pod",
                    (
                        Target(
                            "min by (pod) ("
                            + series(
                                "gpu_fault_processor_healthy",
                                'pod=~"gpu-fault-control-worker-.*"',
                            )
                            + ")",
                            "{{pod}}",
                        ),
                    ),
                    description="0 is a control-worker whose processor reports unhealthy.",
                ),
                Panel(
                    "Active processor consumers",
                    (
                        Target(
                            f"sum({series('gpu_fault_processor_active_consumer')})",
                            "consumers",
                        ),
                    ),
                    description="0 means no replica is consuming the processor queue.",
                ),
                Panel(
                    "Spool consumer running and claim rounds (10m)",
                    (
                        Target(
                            "min by (pod) ("
                            + series(
                                "gpu_fault_telemetry_spool_consumer_running",
                                'pod=~"gpu-fault-telemetry-spool-worker-.*"',
                            )
                            + ")",
                            "{{pod}} running",
                        ),
                        Target(
                            "sum by (pod) (increase("
                            + series(
                                "gpu_fault_telemetry_spool_claim_rounds_total",
                                'pod=~"gpu-fault-telemetry-spool-worker-.*"',
                            )
                            + "[10m]))",
                            "{{pod}} claim rounds",
                        ),
                    ),
                    description=(
                        "A consumer that reports running but claims nothing for "
                        "10m is stalled, not idle: an idle spool still claims."
                    ),
                ),
            ),
        ),
        Row(
            "Processor and periodic errors",
            (
                Panel(
                    "Processor lease renewal failures (15m)",
                    (
                        Target(
                            increase_by_control_plane(
                                "gpu_fault_processor_renewal_fenced_total", "15m"
                            ),
                            "fenced",
                        ),
                        Target(
                            increase_by_control_plane(
                                "gpu_fault_processor_renewal_errors_total", "15m"
                            ),
                            "errors",
                        ),
                    ),
                ),
                Panel(
                    "Processor fault events rejected (15m)",
                    (
                        Target(
                            increase_by_control_plane(
                                "gpu_fault_processor_fault_rejections_total", "15m"
                            ),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    description=(
                        "Fault events the processor refused, per worker Pod "
                        "(summed over the Pod's four processes by /metrics)."
                    ),
                ),
                Panel(
                    "Periodic service error last seen age",
                    (
                        Target(
                            _seconds_since(
                                "gpu_fault_periodic_lease_error_last_seen_timestamp_seconds"
                            ),
                            "lease",
                        ),
                        Target(
                            "time() - max by (control_plane_cluster, region, job) ("
                            + series(
                                "gpu_fault_periodic_job_error_last_seen_timestamp_seconds"
                            )
                            + ")",
                            "{{job}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Seconds since the periodic runner last saw a lease or "
                        "job error; small is bad."
                    ),
                ),
            ),
        ),
    ),
)

REMOTE_COMMAND = Dashboard(
    uid="gpu-fault-remote-command",
    title="GPU Fault · Remote command",
    group="gpu-fault-remote-command",
    rows=(
        Row(
            "Claim latency",
            (
                Panel(
                    "Oldest unclaimed remote command",
                    (
                        Target(
                            "max by (cluster_id) ("
                            + cluster_series(
                                "gpu_fault_remote_command_oldest_unclaimed_seconds"
                            )
                            + ")",
                            "{{cluster_id}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "A PENDING command no cluster executor has claimed; the "
                        "recovery it carries is silently not happening."
                    ),
                ),
                Panel(
                    "Time since last executor internal error",
                    (
                        Target(
                            f"time() - max {BY_CONTROL_PLANE} ("
                            + series(
                                "gpu_fault_remote_command_executor_internal_error_"
                                "last_seen_timestamp_seconds"
                            )
                            + ")",
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Low is bad: the alert fires while the newest executor-side "
                        "defect is younger than its window."
                    ),
                ),
            ),
        ),
        Row(
            "Command census",
            (
                Panel(
                    "Remote commands by status",
                    (
                        Target(
                            "max by (status) ("
                            + series("gpu_fault_remote_command_total")
                            + ")",
                            "{{status}}",
                        ),
                    ),
                ),
                Panel(
                    "Remote commands by cluster and status",
                    (
                        Target(
                            "max by (cluster_id, status) ("
                            + cluster_series(
                                "gpu_fault_remote_command_by_cluster_total"
                            )
                            + ")",
                            "{{cluster_id}} {{status}}",
                        ),
                    ),
                ),
                Panel(
                    "Unclaimed commands dead-lettered",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_remote_command_unclaimed_expired")
                            + ")",
                            "expired",
                        ),
                    ),
                    kind="stat",
                ),
            ),
        ),
    ),
)

POLICY_COVERAGE = Dashboard(
    uid="gpu-fault-policy-coverage",
    title="GPU Fault · Policy coverage",
    group="gpu-fault-policy-coverage",
    rows=(
        Row(
            "Catalog coverage",
            (
                Panel(
                    "Unknown GPU product observations (15m)",
                    (
                        Target(
                            f"sum {BY_CONTROL_PLANE} (increase("
                            + series("gpu_fault_policy_unknown_product_total")
                            + "[15m]))",
                            "{{control_plane_cluster}} {{region}}",
                        ),
                        Target(
                            "sum by (product) (increase("
                            + series("gpu_fault_policy_unknown_product_total")
                            + "[15m]))",
                            "{{product}}",
                        ),
                    ),
                    description=(
                        "GPU products the catalog could not map to a family; the "
                        "policy is silent for every node reporting them."
                    ),
                ),
                Panel(
                    "CRITICAL findings without an incident",
                    (
                        Target(
                            "max by (reason) ("
                            + series("gpu_fault_gpu_findings_without_incident_total")
                            + ")",
                            "{{reason}}",
                        ),
                    ),
                ),
            ),
        ),
    ),
)

ORCHESTRATION_INVARIANTS = Dashboard(
    uid="gpu-fault-orchestration-invariants",
    title="GPU Fault · Orchestration invariants",
    group="gpu-fault-orchestration-invariants",
    rows=(
        Row(
            "Node ownership",
            (
                Panel(
                    "Active attempts per GPU node",
                    (
                        Target(
                            "max by (cluster_id, gpu_node) ("
                            + cluster_series(
                                "gpu_fault_ambiguous_attempt_ownership_current"
                            )
                            + ")",
                            "{{cluster_id}} {{gpu_node}}",
                        ),
                    ),
                    description="More than one fresh attempt owning a node.",
                ),
                Panel(
                    "Stale attempt observations",
                    (
                        Target(
                            "max by (cluster_id, gpu_node) ("
                            + cluster_series("gpu_fault_stale_attempt_observations")
                            + ")",
                            "{{cluster_id}} {{gpu_node}}",
                        ),
                    ),
                ),
            ),
        ),
        Row(
            "Fleet rollout fence",
            (
                Panel(
                    "Fleet rollout fence age",
                    (
                        Target(
                            "max by (cluster_id) ("
                            + cluster_series(
                                "gpu_fault_fleet_rollout_fence_age_seconds"
                            )
                            + ")",
                            "{{cluster_id}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "How long destructive remediation has been fenced for the "
                        "cluster by a non-terminal fleet deployment."
                    ),
                ),
                Panel(
                    "Fleet deployments never started",
                    (
                        Target(
                            "max by (cluster_id) ("
                            + cluster_series("gpu_fault_fleet_rollout_never_started")
                            + ")",
                            "{{cluster_id}}",
                        ),
                    ),
                ),
            ),
        ),
        Row(
            "Pointers, agents and registry",
            (
                Panel(
                    "Incident dangling workflow pointers",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_incident_dangling_workflow_pointers")
                            + ")",
                            "dangling",
                        ),
                    ),
                    description=(
                        "Incidents whose workflow_request_id names a workflow that "
                        "no longer exists."
                    ),
                ),
                Panel(
                    "Stale agents",
                    (
                        Target(
                            f"max({series('gpu_fault_stale_agents')})",
                            "stale",
                        ),
                    ),
                    description="Node agents past their heartbeat deadline.",
                ),
                Panel(
                    "Fleet pin drift nodes",
                    (
                        Target(
                            "max by (cluster_id, kind) ("
                            + cluster_series("gpu_fault_fleet_pin_drift_nodes")
                            + ")",
                            "{{cluster_id}} {{kind}}",
                        ),
                    ),
                    description=(
                        "PIN_AHEAD_OF_FLEET is a node pinned to a release the "
                        "fleet has not reached."
                    ),
                ),
                Panel(
                    "Regional registry Secret drift",
                    (
                        Target(
                            "max by (service_role) ("
                            + series("gpu_fault_regional_registry_secret_drift")
                            + ")",
                            "{{service_role}}",
                        ),
                    ),
                    description=(
                        "1 when the registry Secret's configured digest differs "
                        "from the durable head the replicas serve."
                    ),
                ),
            ),
        ),
    ),
)


def _outcome_counter_panels() -> tuple[Panel, ...]:
    return tuple(
        Panel(
            title,
            (
                Target(
                    increase_by_control_plane(metric, "30m"),
                    "{{control_plane_cluster}} {{region}}",
                ),
            ),
        )
        for title, metric in (
            (
                "Workflow lifetime exceeded (30m)",
                "gpu_fault_workflow_lifetime_exceeded_total",
            ),
            (
                "Job workflow node-busy timeouts (30m)",
                "gpu_fault_workflow_dispatch_node_busy_timeouts_total",
            ),
            (
                "Retry-horizon failures (30m)",
                "gpu_fault_processor_retry_horizon_failures_total",
            ),
            (
                "Branch escalation budget refusals (30m)",
                "gpu_fault_workflow_branch_escalation_budget_refusals_total",
            ),
        )
    )


RECOVERY_OUTCOME = Dashboard(
    uid="gpu-fault-recovery-outcome",
    title="GPU Fault · Recovery outcome",
    group="gpu-fault-recovery-outcome",
    rows=(
        Row(
            "Workflows",
            (
                Panel(
                    "Workflows by status",
                    (
                        Target(
                            f"max by (status) ({series('gpu_fault_workflow_total')})",
                            "{{status}}",
                        ),
                    ),
                ),
                Panel(
                    "BLOCKED workflows not reconciled",
                    (
                        Target(
                            f"max({series('gpu_fault_workflow_blocked_unreconciled')})",
                            "blocked",
                        ),
                    ),
                ),
                Panel(
                    "Oldest PENDING workflow age",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_workflow_pending_age_seconds_max")
                            + ")",
                            "pending age",
                        ),
                    ),
                    unit="s",
                ),
                Panel(
                    "Most overdue workflow",
                    (
                        Target(
                            f"max({series('gpu_fault_workflow_overdue_seconds')})",
                            "overdue",
                        ),
                    ),
                    unit="s",
                    description="Above zero means an execution deadline was not enforced.",
                ),
            ),
        ),
        Row(
            "Step waiting",
            (
                Panel(
                    "Step wait beyond its warning limit, by operation",
                    (
                        Target(
                            "max by (operation) ("
                            + series("gpu_fault_workflow_step_waiting_seconds")
                            + " - "
                            + series("gpu_fault_workflow_step_waiting_warning_seconds")
                            + ")",
                            "{{operation}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Positive means the oldest WAITING step of that operation "
                        "has outlived the limit configured for it."
                    ),
                ),
            ),
        ),
        Row("Counters (30m increase)", _outcome_counter_panels()),
        Row(
            "Closed loop and notifications",
            (
                Panel(
                    "Containment latency (6h moving mean)",
                    (
                        Target(
                            f"delta((max {BY_CONTROL_PLANE} ("
                            + series(
                                "gpu_fault_closed_loop_milestone_seconds_sum",
                                'milestone="containment"',
                            )
                            + "))[6h:5m]) / clamp_min(delta((max "
                            + BY_CONTROL_PLANE
                            + " ("
                            + series(
                                "gpu_fault_closed_loop_milestone_seconds_count",
                                'milestone="containment"',
                            )
                            + "))[6h:5m]), 1)",
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="s",
                    description="Workflow creation to successful containment.",
                ),
                Panel(
                    "Notification deliveries FAILED (15m)",
                    (
                        Target(
                            f"delta((max {BY_CONTROL_PLANE} ("
                            + series("gpu_fault_notification_total", 'status="FAILED"')
                            + "))[15m:1m])",
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                ),
                Panel(
                    "Notifications by status",
                    (
                        Target(
                            f"max by (status) ({series('gpu_fault_notification_total')})",
                            "{{status}}",
                        ),
                    ),
                ),
                Panel(
                    "Notification outbox depth",
                    (
                        Target(
                            f"max({series('gpu_fault_notification_outbox_depth')})",
                            "outbox",
                        ),
                    ),
                ),
            ),
        ),
        Row(
            "Remediation budget",
            (
                Panel(
                    "Workflows waiting for any remediation budget",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_remediation_budget_waiting_workflows")
                            + ")",
                            "waiting",
                        ),
                    ),
                ),
                Panel(
                    "Cluster budget saturation",
                    (
                        Target(
                            "max by (cluster_id) ("
                            + cluster_series(
                                "gpu_fault_remediation_budget_cluster_waiting_workflows"
                            )
                            + ")",
                            "{{cluster_id}} waiting",
                        ),
                        Target(
                            "max by (cluster_id) ("
                            + cluster_series(
                                "gpu_fault_remediation_budget_cluster_active_claims"
                            )
                            + ")",
                            "{{cluster_id}} active",
                        ),
                        Target(
                            f"max({series('gpu_fault_remediation_budget_cluster_limit')})",
                            "limit",
                        ),
                    ),
                    description=(
                        "Saturated when a cluster has waiters while its active "
                        "claims sit at the configured limit."
                    ),
                ),
            ),
        ),
        Row(
            "Completion and incidents",
            (
                Panel(
                    "Completion decisions by status",
                    (
                        Target(
                            "max by (status) ("
                            + series("gpu_fault_completion_decisions")
                            + ")",
                            "{{status}}",
                        ),
                    ),
                    description=(
                        "NO_ACTION and PLAN_CREATED are the only terminal "
                        "statuses; a rising count is load, not backlog."
                    ),
                ),
                Panel(
                    "Completion events without a decision",
                    (
                        Target(
                            "max("
                            + series("gpu_fault_completion_events_without_decision")
                            + ")",
                            "undecided",
                        ),
                    ),
                ),
                Panel(
                    "Incidents by state",
                    (
                        Target(
                            f"max by (state) ({series('gpu_fault_incidents_by_state')})",
                            "{{state}}",
                        ),
                    ),
                    description="ESCALATED is the operator queue the alert watches.",
                ),
            ),
        ),
        Row(
            "Dispatcher errors and notification outbox",
            (
                Panel(
                    "Dispatch internal error last seen age",
                    (
                        Target(
                            _seconds_since(
                                "gpu_fault_workflow_dispatch_internal_error_last_seen_timestamp_seconds"
                            ),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Seconds since a dispatch cycle last hit an internal "
                        "error; small is bad."
                    ),
                ),
                Panel(
                    "Failure handling abandoned last seen age",
                    (
                        Target(
                            _seconds_since(
                                "gpu_fault_workflow_dispatch_failure_handling_abandoned_last_seen_timestamp_seconds"
                            ),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Seconds since the dispatcher last gave up handling a "
                        "step failure; small is bad."
                    ),
                ),
                Panel(
                    "Oldest undelivered notification age",
                    (
                        Target(
                            "max("
                            + series(
                                "gpu_fault_notification_oldest_pending_age_seconds"
                            )
                            + ")",
                            "oldest pending",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Age of the oldest PENDING/RETRY/LEASED delivery, counted "
                        "from creation or the last operator re-queue."
                    ),
                ),
                Panel(
                    "Notification outbox draining",
                    (
                        Target(
                            "max by (status) ("
                            + series(
                                "gpu_fault_notification_delivery_total",
                                'status=~"PENDING|RETRY"',
                            )
                            + ")",
                            "{{status}}",
                        ),
                        Target(
                            _seconds_since(
                                "gpu_fault_notification_dispatch_last_cycle_timestamp_seconds"
                            ),
                            "dispatch cycle age (s)",
                        ),
                    ),
                    description=(
                        "Undelivered rows next to the seconds since the outbox "
                        "dispatcher last ran; rows with no cycle is a stuck outbox."
                    ),
                ),
                Panel(
                    "Expired notification last seen age",
                    (
                        Target(
                            _seconds_since(
                                "gpu_fault_notification_expired_last_seen_timestamp_seconds"
                            ),
                            "{{control_plane_cluster}} {{region}}",
                        ),
                    ),
                    unit="s",
                    description=(
                        "Seconds since an advisory expired unsent; small is bad."
                    ),
                ),
            ),
        ),
    ),
)

DASHBOARDS: tuple[Dashboard, ...] = (
    OVERVIEW,
    COLLECTOR_HEALTH,
    TELEMETRY_PIPELINE,
    CONTROL_PLANE_CAPACITY,
    REMOTE_COMMAND,
    POLICY_COVERAGE,
    ORCHESTRATION_INVARIANTS,
    RECOVERY_OUTCOME,
)
