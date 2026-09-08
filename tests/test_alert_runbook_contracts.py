from __future__ import annotations

from pathlib import Path

import yaml

from gpu_fault.app.collector_metrics import aggregate_lines
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = lazy_script_module(ROOT / "scripts/verify-regional-alerting.py")
AMP_RULES = ROOT / "deploy/observability/amp-rules.yaml"


def amp_rules() -> list[dict[str, object]]:
    payload = yaml.safe_load(AMP_RULES.read_text(encoding="utf-8"))
    return [rule for group in payload["groups"] for rule in group["rules"]]


def test_every_amp_alert_points_at_an_existing_runbook_card() -> None:
    defects = MODULE.runbook_defects(ROOT, amp_rules())

    assert defects == [], "every AMP alert needs a runbook triage card: " + "; ".join(
        defects
    )


def test_recovery_outcome_alerts_cover_the_business_closed_loop() -> None:
    """Component alerts cannot report "everything is up and nothing recovers"."""
    alerts = {str(rule["alert"]) for rule in amp_rules()}

    for alert in (
        "GpuFaultRecoveryBlockedBacklog",
        "GpuFaultClosedLoopSlow",
        "GpuFaultNotificationDeliveryFailing",
        "GpuFaultRemediationBudgetStarved",
    ):
        assert alert in alerts, f"recovery outcome alert is missing: {alert}"


COMPLETION_WATCHER_ALERTS = {
    # The 0/1 gauge for the routine attempt-state ConfigMap. Nothing on the
    # delivery path reads that object, so its outage is invisible everywhere
    # else; a restart inside the window re-derives attempts from live Pods and
    # resolves a finished attempt's node as IDLE, which drops STOP_WORKLOADS
    # from the reset plan.
    "GpuFaultCompletionActiveStateUnavailable": (
        "gpu_fault_completion_active_state_unavailable"
    ),
    # A failed write-ahead append no longer vetoes the POST, so this counter is
    # the only trace that a delivered critical event was never buffered.
    "GpuFaultCompletionOutboxAppendFailures": (
        "gpu_fault_completion_outbox_append_failures_total"
    ),
}


def test_the_completion_watcher_gauges_each_have_an_alert() -> None:
    """Both families reached AMP with no rule reading them (data-plane review).

    Each one reports a Completion Watcher ConfigMap that is refusing writes
    while everything else stays green, so neither condition is reachable from
    any other rule in the file.
    """
    alerts = {str(rule["alert"]) for rule in amp_rules()}

    for alert, metric in COMPLETION_WATCHER_ALERTS.items():
        assert alert in alerts, f"Completion Watcher alert is missing: {alert}"
        assert metric in _expression(alert), (
            f"{alert} must read {metric}: {_expression(alert)}"
        )


def test_the_completion_watcher_alerts_keep_the_grouping_labels() -> None:
    """A label-less aggregation merges every control plane into one page.

    Alertmanager groups on control_plane_cluster/region, so ``max(...)`` over
    the whole fleet would name only the first control plane that trips.
    """
    for alert in COMPLETION_WATCHER_ALERTS:
        assert "by (control_plane_cluster, region)" in _expression(alert), (
            f"{alert} aggregates without the Alertmanager grouping labels: "
            f"{_expression(alert)}"
        )


def test_the_completion_watcher_alerts_are_slow_warnings_with_a_card() -> None:
    """15m is roughly fifteen of the watcher's own once-a-minute retries.

    Long enough that a ConfigMap re-apply or a rollout clears by itself, and a
    warning rather than a page because nothing is lost while the watcher stays
    up: only a restart inside the window turns either gap into real damage.
    """
    for alert in COMPLETION_WATCHER_ALERTS:
        rule = _rule(alert)
        annotations = rule["annotations"]
        assert isinstance(annotations, dict), f"{alert} has no annotations mapping"
        assert rule["for"] == "15m", f"{alert} `for` is {rule['for']!r}, expected 15m"
        assert rule["labels"] == {"severity": "warning"}, (
            f"{alert} must be a warning, not {rule['labels']!r}"
        )
        expected = f"{MODULE.RUNBOOK_DOCUMENT}#{MODULE.markdown_anchor(alert)}"
        assert annotations["runbook_url"] == expected, (
            f"{alert} runbook anchor must be derived from its own name: "
            f"{annotations['runbook_url']!r} != {expected!r}"
        )


def test_the_active_state_alert_is_a_state_test_not_a_threshold() -> None:
    """The gauge is 0/1, so the rule fires on the state rather than on a count.

    ``== 1`` also keeps the rule out of the Grafana threshold-line contract,
    which only draws ``>``/``>=``/``<`` comparisons.
    """
    expression = _expression("GpuFaultCompletionActiveStateUnavailable")

    assert expression == (
        "max by (control_plane_cluster, region) "
        "(gpu_fault_completion_active_state_unavailable) == 1"
    ), f"unexpected expression shape: {expression}"


def test_the_outbox_append_failure_alert_reads_a_windowed_increase() -> None:
    """The counter only ever rises, so the alert has to read a window of it.

    A bare ``> 0`` on the counter would stay firing for the life of the process
    after one failure; ``increase(...[10m])`` resolves once appends recover.
    """
    expression = _expression("GpuFaultCompletionOutboxAppendFailures")

    assert expression == (
        "sum by (control_plane_cluster, region) "
        "(increase(gpu_fault_completion_outbox_append_failures_total[10m])) > 0"
    ), f"unexpected expression shape: {expression}"


def defects_for(alert: str, annotations: dict[str, str]) -> list[str]:
    """Report only the defects about ``alert``.

    ``runbook_defects`` also reports every card no rule in the argument
    mentions, so a one-rule probe would otherwise drown in orphan reports for
    the other 24 cards.
    """
    rules = [
        rule if rule["alert"] != alert else {"alert": alert, "annotations": annotations}
        for rule in amp_rules()
    ]
    return [defect for defect in MODULE.runbook_defects(ROOT, rules) if alert in defect]


def test_missing_annotation_is_a_defect() -> None:
    defects = defects_for("GpuFaultCollectorSilent", {"summary": "x"})

    assert defects == [
        "alert has no annotations.runbook_url: GpuFaultCollectorSilent"
    ], defects


def test_annotation_that_does_not_match_the_alert_name_is_a_defect() -> None:
    """A runbook_url pointing at some other card reads as valid to an operator."""
    defects = defects_for(
        "GpuFaultCollectorSilent",
        {"runbook_url": "docs/管理员日常运维.md#gpufaultstoreiorejected"},
    )

    assert len(defects) == 1, defects
    assert "expected" in defects[0], defects[0]


def test_a_runbook_card_without_an_alert_is_a_defect() -> None:
    """Triage prose no on-call path reaches is prose that rots unnoticed."""
    defects = MODULE.runbook_defects(
        ROOT,
        [
            {
                "alert": "GpuFaultCollectorSilent",
                "annotations": {
                    "runbook_url": ("docs/管理员日常运维.md#gpufaultcollectorsilent")
                },
            }
        ],
    )

    assert defects, "the remaining runbook cards must be reported as orphans"
    assert all("but no AMP alert fires it" in defect for defect in defects), defects
    assert not [defect for defect in defects if "GpuFaultCollectorSilent" in defect], (
        defects
    )


def test_every_amp_alert_carries_a_summary_and_a_description() -> None:
    """The notification body is built from these two, not from runbook_url.

    Fifteen rules used to have only a summary, so the SNS message an
    administrator received was a one-line title and a repository path.
    """
    defects = MODULE.annotation_defects(amp_rules())

    assert defects == [], "; ".join(defects)


def test_an_alert_without_a_description_is_a_defect() -> None:
    defects = MODULE.annotation_defects(
        [{"alert": "GpuFaultCollectorSilent", "annotations": {"summary": "x"}}]
    )

    assert defects == [
        "alert has no annotations.description: GpuFaultCollectorSilent"
    ], defects


def test_a_description_repeating_the_summary_is_a_defect() -> None:
    """Presence alone is satisfiable without telling the operator anything."""
    defects = MODULE.annotation_defects(
        [
            {
                "alert": "GpuFaultCollectorSilent",
                "annotations": {
                    "summary": "a collector is silent",
                    "description": "a collector is silent",
                },
            }
        ]
    )

    assert len(defects) == 1, defects
    assert "only repeats its summary" in defects[0], defects[0]


def test_an_alert_without_a_summary_is_a_defect() -> None:
    defects = MODULE.annotation_defects(
        [{"alert": "GpuFaultCollectorSilent", "annotations": {"description": "x"}}]
    )

    assert defects == ["alert has no annotations.summary: GpuFaultCollectorSilent"], (
        defects
    )


def test_every_store_derived_alert_aggregates_across_replicas() -> None:
    """One cluster-level fact must reach the operator as one alert entry.

    ``GpuFaultRecoveryBlockedBacklog`` shipped as a bare
    ``gpu_fault_workflow_total{status="BLOCKED"} > 3``. Every control-plane
    replica publishes that same Store-derived total, so a backlog of four
    workflows arrived as six identical entries -- one per ``gpu-fault-control-
    worker`` Pod -- and the count would grow with the Deployment. Nothing in the
    repository pinned ``expr``, so only review stood between that shape and
    production.
    """
    defects = MODULE.aggregation_defects(ROOT, amp_rules())

    assert defects == [], "; ".join(defects)


def test_an_unaggregated_store_derived_threshold_is_a_defect() -> None:
    """The exact expression that produced the duplicate notification."""
    defects = MODULE.aggregation_defects(
        ROOT,
        [
            {
                "alert": "GpuFaultRecoveryBlockedBacklog",
                "expr": 'gpu_fault_workflow_total{status="BLOCKED"} > 3',
            }
        ],
    )

    assert len(defects) == 1, defects
    assert "once per control-plane replica" in defects[0], defects[0]


def test_an_aggregation_that_drops_the_grouping_labels_is_a_defect() -> None:
    """Aggregating is not enough on its own.

    ``control_plane_cluster`` and ``region`` are relabelled per series and appear
    in ``amp-alertmanager.yaml``'s ``group_by``, so an aggregation that discards
    them trades duplicate notifications for merged ones.
    """
    defects = MODULE.aggregation_defects(
        ROOT,
        [
            {
                "alert": "GpuFaultRecoveryBlockedBacklog",
                "expr": "max by (cluster_id) (gpu_fault_workflow_blocked_unreconciled) > 3",
            }
        ],
    )

    assert len(defects) == 1, defects
    assert "loses the Alertmanager grouping labels" in defects[0], defects[0]
    assert "control_plane_cluster, region" in defects[0], defects[0]


def test_a_bare_aggregation_is_a_defect() -> None:
    """The mirror image of the duplicate-entry defect.

    ``min(x)`` satisfies "it aggregates" and carries no ``by`` clause for the
    label check to inspect, so it slipped through both halves while collapsing
    every control plane and region into one label-less series. Four rules shipped
    that way, including ``GpuFaultCollectorMetricsSnapshotStale``.
    """
    defects = MODULE.aggregation_defects(
        ROOT,
        [
            {
                "alert": "GpuFaultCollectorMetricsSnapshotStale",
                "expr": "min(gpu_fault_collector_metrics_snapshot_age_seconds) > 120",
            }
        ],
    )

    assert len(defects) == 1, defects
    assert "with no `by` clause" in defects[0], defects[0]
    assert "control_plane_cluster, region" in defects[0], defects[0]


def test_the_collector_families_are_store_derived() -> None:
    """`GpuFaultCollectorSilent` grouped on a label its metric never carries.

    ``collector_metrics.py`` aggregates to ``(cluster_id, collector, channel)``
    before publishing ``gpu_fault_collector_silent_nodes``, yet the alert grouped
    ``by (cluster_id, node_id, collector, channel)``. It escaped this check
    because the family list only read ``builtin_metric_contributors.py``, so the
    collector families were invisible to it.
    """
    families = MODULE.store_derived_families(ROOT)

    assert "gpu_fault_collector_silent_nodes" in families
    assert "gpu_fault_collector_metrics_snapshot_age_seconds" in families
    assert "gpu_fault_ingress_lane_in_flight" not in families, (
        "ingress lane occupancy is per-replica, not a Store-derived fact"
    )


def test_the_collector_silent_alert_does_not_group_on_a_missing_label() -> None:
    """Grouping on an absent label is silent: PromQL reads it as empty.

    ``GpuFaultCollectorSilent`` grouped ``by (cluster_id, node_id, collector,
    channel)`` on a gauge the snapshot publishes with only the other three, so the
    alert both dropped the Alertmanager grouping labels and named a label that
    never arrives. The published label set is read off the emitted lines rather
    than off the alert text, so the two cannot drift apart silently.
    """
    silent = next(
        rule for rule in amp_rules() if rule["alert"] == "GpuFaultCollectorSilent"
    )
    rows = [
        {
            "cluster_id": "gpu-a",
            "node_id": node_id,
            "collector": "dcgm",
            "channel": "metrics",
            "last_success_age_seconds": age,
            "silent": True,
            "erroring": False,
        }
        for node_id, age in (("node-a", 90.0), ("node-b", 30.0))
    ]

    emitted = [
        line
        for line in aggregate_lines(rows, top_n=3)
        if line.startswith("gpu_fault_collector_silent_nodes{")
    ]

    assert len(emitted) == 1, (
        "the aggregate gauge split per node, so it now carries node_id and the "
        f"alert would have to group on it: {emitted}"
    )
    assert "node_id" not in emitted[0], emitted[0]
    assert emitted[0].endswith(" 2"), (
        f"both silent nodes must fold into the one series: {emitted[0]}"
    )
    assert "node_id" not in str(silent["expr"])


def test_a_per_replica_family_is_left_alone() -> None:
    """Which Pod filled its queue is the point of a per-replica alert."""
    assert (
        MODULE.aggregation_defects(
            ROOT,
            [
                {
                    "alert": "GpuFaultProcessorQueueDepthHigh",
                    "expr": "gpu_fault_processor_queue_depth > 7000",
                }
            ],
        )
        == []
    )


def test_the_store_derived_family_list_is_read_from_the_metric_source() -> None:
    """Restating the list here would let it rot; it is derived instead.

    ``builtin_metric_contributors`` is exactly the module whose families come out
    of the Store, and its long HELP strings are written as adjacent literals that
    sometimes split inside a metric name, which is why the reader rejoins them.
    """
    families = MODULE.store_derived_families(ROOT)

    assert "gpu_fault_workflow_blocked_unreconciled" in families
    assert "gpu_fault_workflow_total" in families
    assert (
        "gpu_fault_remote_command_executor_internal_error_last_seen_timestamp_seconds"
        in families
    ), "a family split across two string literals was read as a truncated name"
    assert "gpu_fault_processor_queue_depth" not in families, (
        "a per-replica runtime family is not published from the Store"
    )


def test_the_blocked_backlog_alert_reads_the_unreconciled_gauge() -> None:
    """A lifetime accumulator cannot express "there is a backlog now".

    ``gpu_fault_workflow_total{status="BLOCKED"}`` counts every workflow ever
    persisted and BLOCKED is terminal, so the bucket only falls when an operator
    runs ``workflow-reconcile`` or the incident is archived out of the table. A
    backlog handled correctly therefore kept the threshold true forever and
    re-notified every ``repeat_interval``, while the description's "each one
    holds a GPU node" had already stopped being true.
    """
    rule = next(
        item
        for item in amp_rules()
        if item["alert"] == "GpuFaultRecoveryBlockedBacklog"
    )
    expression = " ".join(str(rule["expr"]).split())

    assert "gpu_fault_workflow_blocked_unreconciled" in expression
    assert "gpu_fault_workflow_total" not in expression
    assert "no verified restore successor" in str(rule["annotations"]["description"])


def adot_collector_config() -> dict:
    documents = yaml.safe_load_all(
        (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
            encoding="utf-8"
        )
    )
    collector = next(
        document
        for document in documents
        if document and document.get("kind") == "ConfigMap"
    )
    return yaml.safe_load(collector["data"]["collector.yaml"])


def test_the_collector_scrapes_its_own_delivery_counters() -> None:
    """A dropped remote_write batch must leave a trace AMP can alert on.

    ``retry_on_failure`` gives up after ``max_elapsed_time`` and drops the batch,
    and the only record is a counter on the collector's self-telemetry endpoint.
    That endpoint was configured but nothing scraped it, so an AMP workspace that
    was rejecting writes looked exactly like a fleet with nothing to report --
    while every other rule in amp-rules.yaml kept evaluating over the gaps.
    """
    config = adot_collector_config()
    jobs = {
        job["job_name"]: job
        for job in config["receivers"]["prometheus"]["config"]["scrape_configs"]
    }
    reader = config["service"]["telemetry"]["metrics"]["readers"][0]["pull"][
        "exporter"
    ]["prometheus"]

    assert "gpu-fault-adot-self" in jobs, (
        "the collector's own pipeline counters are unscraped, so a dropped "
        "remote_write batch reaches nobody"
    )
    self_job = jobs["gpu-fault-adot-self"]
    assert self_job["static_configs"][0]["targets"] == [
        f"{reader['host']}:{reader['port']}"
    ], "the self-scrape target and the telemetry reader must be the same endpoint"
    assert reader["host"] == "127.0.0.1", (
        "the reader has no authentication of its own and only this in-pod job "
        "reads it, so binding the pod IP publishes the collector's internals"
    )


def test_the_drop_counters_reach_amp_and_carry_the_grouping_labels() -> None:
    """The keep filter is the only thing that reaches AMP, and it is per job."""
    config = adot_collector_config()
    scrapes = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    self_job = next(job for job in scrapes if job["job_name"] == "gpu-fault-adot-self")
    added = {
        rule["target_label"]
        for rule in self_job["relabel_configs"]
        if rule.get("action") == "replace"
    }

    assert MODULE.keep_filter_defects(scrapes, amp_rules()) == []
    # `up` is synthesised from the target's label set, so these have to be target
    # relabels for GpuFaultAdotSelfMetricsMissing to group like every other rule.
    assert set(MODULE.GROUPING_LABELS) <= added, added
    assert MODULE.collector_label_defects(self_job) == []
    # Unlike a scraped target, the collector's own telemetry is labelled with its
    # service identity: service_instance_id is a UUID regenerated on every start
    # and service_version moves with the image. Left in, a restart or an image
    # bump resolves and re-fires these alerts as new series and drops any silence
    # an operator put in place mid-incident.
    dropped = {
        label
        for rule in self_job["metric_relabel_configs"]
        if rule.get("action") == "labeldrop"
        for label in str(rule["regex"]).split("|")
    }
    assert {"service_instance_id", "service_version"} <= dropped, dropped


def test_the_collector_internal_counters_are_selected_unsuffixed() -> None:
    """The suffixed name is the documented one and the wrong one.

    Upstream documents these counters as ``otelcol_exporter_*_total``, but the
    pinned aws-otel-collector v0.49.0 (core v0.156.0) publishes them unsuffixed
    on :8889 and ``add_metric_suffixes: false`` means nothing adds one before AMP.
    Both alerts shipped suffixed in a first draft, which is a rule that parses,
    validates and never fires -- read off the running image instead. If the image
    digest in ``release_artifacts.py`` moves, re-read the endpoint.
    """
    selected = {
        metric
        for rule in amp_rules()
        for metric in MODULE.AMP_METRIC.findall(str(rule["expr"]))
        if metric.startswith("otelcol_")
    }

    assert selected, "no rule reads the collector's own counters any more"
    assert "otelcol_exporter_send_failed_metric_points" in selected
    assert not [metric for metric in selected if metric.endswith("_total")], selected


def test_a_metric_no_scrape_job_keeps_is_reported_as_dropped() -> None:
    """The union is over jobs; a family no job keeps is dead configuration."""
    scrapes = [
        {
            "metric_relabel_configs": [
                {"action": "keep", "source_labels": ["__name__"], "regex": "up"}
            ]
        },
        {
            "metric_relabel_configs": [
                {
                    "action": "keep",
                    "source_labels": ["__name__"],
                    "regex": "otelcol_exporter_sent_metric_points_total",
                }
            ]
        },
    ]
    rules = [
        {
            "alert": "GpuFaultTelemetryRemoteWriteFailing",
            "expr": (
                "rate(otelcol_exporter_send_failed_metric_points_total[5m]) > 0 or "
                "rate(otelcol_exporter_sent_metric_points_total[5m]) > 0"
            ),
        }
    ]

    defects = MODULE.keep_filter_defects(scrapes, rules)

    assert defects == [
        "AMP rule metric is dropped by ADOT: GpuFaultTelemetryRemoteWriteFailing "
        "uses otelcol_exporter_send_failed_metric_points_total"
    ], defects


def test_collector_scope_labels_are_dropped_before_amp() -> None:
    """An ADOT upgrade must not re-fire every alert.

    ``otel_scope_version`` changes with the collector image and with nothing
    about the fleet, and alert identity is the whole label set, so leaving it in
    resolves and re-fires every alert and invalidates the silences an operator
    put in place mid-incident.
    """
    documents = list(
        yaml.safe_load_all(
            (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    collector = next(
        document
        for document in documents
        if document and document.get("kind") == "ConfigMap"
    )
    config = yaml.safe_load(collector["data"]["collector.yaml"])
    scrape = config["receivers"]["prometheus"]["config"]["scrape_configs"][0]

    assert MODULE.collector_label_defects(scrape) == []
    assert MODULE.collector_label_defects({"metric_relabel_configs": []}) == [
        "ADOT does not labeldrop the collector scope labels, so they enter alert "
        "identity: otel_scope_name, otel_scope_version"
    ]


def test_anchor_matches_github_heading_slug_rules() -> None:
    assert MODULE.markdown_anchor("GpuFaultStoreIoRejected") == (
        "gpufaultstoreiorejected"
    ), "alert names are alphanumeric, so the anchor is the lowercased name"
    assert MODULE.markdown_anchor("REG-14.1 CPU API Pod") == "reg-141-cpu-api-pod", (
        "spaces become dashes and the dot is dropped"
    )


# BATCH3: the periodic-job and closed-loop gauges added on 2026-09-06. Each one
# describes a condition every component alert is blind to -- a processor that
# died mid-lease, a shard counter that no longer matches the table, a completion
# event nobody triaged, an escalation nobody picked up, a workflow nobody
# dispatched -- so each needs its own rule and its own card.
BATCH3_ALERTS = {
    "GpuFaultProcessorExpiredLeasesReclaimed": (
        "increase(gpu_fault_processor_expired_leases_reclaimed_total[10m])"
    ),
    "GpuFaultProcessorCounterDrift": "gpu_fault_processor_counter_drift_abs",
    "GpuFaultCompletionPendingTriageBacklog": (
        'gpu_fault_completion_decisions{status="PENDING_TRIAGE"}'
    ),
    "GpuFaultCompletionEventsWithoutDecision": (
        "gpu_fault_completion_events_without_decision"
    ),
    "GpuFaultIncidentsAwaitingOperator": (
        'gpu_fault_incidents_by_state{state="ESCALATED"}'
    ),
    "GpuFaultWorkflowPendingAge": "gpu_fault_workflow_pending_age_seconds_max",
}


def _rule(alert: str) -> dict[str, object]:
    return next(rule for rule in amp_rules() if rule["alert"] == alert)


def _expression(alert: str) -> str:
    return " ".join(str(_rule(alert)["expr"]).split())


def test_the_periodic_job_gauges_each_have_an_alert() -> None:
    alerts = {str(rule["alert"]) for rule in amp_rules()}

    for alert, metric in BATCH3_ALERTS.items():
        assert alert in alerts, f"periodic-job alert is missing: {alert}"
        assert metric in _expression(alert), (alert, _expression(alert))


def test_the_pending_triage_backlog_waits_twice_the_deadline() -> None:
    """900s is the default GPU_FAULT_PENDING_TRIAGE_DEADLINE_SECONDS.

    The reconcile job gets one full deadline plus a scan interval to act; only a
    decision still PENDING_TRIAGE after twice that is a stall rather than a lag.
    """
    assert _rule("GpuFaultCompletionPendingTriageBacklog")["for"] == "30m"
    assert _rule("GpuFaultCompletionEventsWithoutDecision")["for"] == "30m"


def test_the_escalated_incident_reminder_is_slow_and_soft() -> None:
    """ESCALATED is a designed parking state, so this is a reminder, not a fault."""
    rule = _rule("GpuFaultIncidentsAwaitingOperator")

    assert rule["for"] == "24h"
    assert rule["labels"]["severity"] != "critical"  # type: ignore[index]


def test_the_pending_age_alert_fires_at_thirty_minutes() -> None:
    rule = _rule("GpuFaultWorkflowPendingAge")

    assert "> 1800" in _expression("GpuFaultWorkflowPendingAge")
    assert rule["for"] == "10m"


def test_the_counter_drift_alert_reads_both_drift_gauges_over_a_window() -> None:
    """A single scrape can straddle a batch commit; five minutes of drift cannot."""
    expression = _expression("GpuFaultProcessorCounterDrift")

    assert "min_over_time(gpu_fault_processor_counter_drift_abs[5m])" in expression
    assert (
        "min_over_time(gpu_fault_processor_counter_mismatched_clusters[5m])"
        in expression
    )


def test_the_batch3_alerts_are_not_a_severity_of_their_own() -> None:
    severities = {
        str(_rule(alert)["labels"]["severity"])  # type: ignore[index]
        for alert in BATCH3_ALERTS
    }

    assert severities <= {"warning", "critical"}, severities


def test_the_keep_list_admits_the_completion_incident_and_finding_families() -> None:
    """`gpu_fault_incident_.+` does not match `gpu_fault_incidents_by_state`.

    The `s` is enough to drop the whole family before AMP, so a rule on it would
    validate and never fire. The findings counter has no rule yet but is the
    only evidence that a CRITICAL finding was suppressed rather than lost.
    """
    scrapes = adot_collector_config()["receivers"]["prometheus"]["config"][
        "scrape_configs"
    ]
    probe = [
        {
            "alert": "Probe",
            "expr": (
                'gpu_fault_completion_decisions{status="PENDING_TRIAGE"} + '
                "gpu_fault_completion_events_without_decision + "
                'gpu_fault_incidents_by_state{state="ESCALATED"} + '
                "gpu_fault_gpu_findings_without_incident_total + "
                "gpu_fault_completion_pending_triage_reconciled_total"
            ),
        }
    ]

    assert MODULE.keep_filter_defects(scrapes, probe) == []
