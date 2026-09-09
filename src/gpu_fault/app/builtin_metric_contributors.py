from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Container, Sequence
from datetime import datetime, timedelta, timezone

from gpu_fault.app.metric_scan_cache import MetricScanCache, metric_scan_cache
from gpu_fault.app.runtime import AppRuntime
from gpu_fault.collector_requirements import agent_is_current
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.fleet_deployment import DeploymentStatus
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    NotificationDeliveryStatus,
    NotificationStatus,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store.contracts import ControlPlaneStore

# Upper bound on the rows either orphan inspection decodes per scrape; the
# gauges saturate there rather than let a pathological table grow the render.
ORPHAN_INSPECTION_LIMIT = 10_000


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def remote_command_metric_lines(
    runtime: AppRuntime,
) -> list[str]:
    context = runtime.context
    if not context.regional_mode:
        return []
    # A whole-kind GROUP BY per scrape (G-12); shared across scrapes for the
    # scan cache's TTL like the other fleet-level aggregates.
    remote = metric_scan_cache(runtime).shared(
        "remote_command_stats", context.store.remote_command_stats
    )
    lines = [
        "# HELP gpu_fault_remote_command_total Remote cluster commands by status.",
        "# TYPE gpu_fault_remote_command_total gauge",
    ]
    for status_value, count in sorted(remote["by_status"].items()):
        lines.append(
            f'gpu_fault_remote_command_total{{status="{status_value}"}} {count}'
        )
    # F-D12: the same counts per cluster, so one stuck cluster is not averaged
    # into the fleet total the alerts read.
    lines.extend(
        [
            "# HELP gpu_fault_remote_command_by_cluster_total Remote cluster "
            "commands by cluster and status.",
            "# TYPE gpu_fault_remote_command_by_cluster_total gauge",
        ]
    )
    by_cluster_status: dict[str, dict[str, int]] = remote.get("by_cluster_status", {})
    for cluster_id, statuses in sorted(by_cluster_status.items()):
        for status_value, count in sorted(statuses.items()):
            lines.append(
                "gpu_fault_remote_command_by_cluster_total"
                f'{{cluster_id="{_escape_label(cluster_id)}",'
                f'status="{status_value}"}} {count}'
            )
    lines.extend(
        [
            "# HELP gpu_fault_remote_command_oldest_unclaimed_seconds "
            "Age of the oldest PENDING remote command never claimed "
            "by a cluster executor.",
            "# TYPE gpu_fault_remote_command_oldest_unclaimed_seconds gauge",
        ]
    )
    for cluster_id, age in sorted(
        remote["oldest_unclaimed_age_seconds_by_cluster"].items()
    ):
        lines.append(
            "gpu_fault_remote_command_oldest_unclaimed_seconds"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {age:.6f}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remote_command_executor_internal_errors"
            "_total Retained remote commands failed by an executor-side defect.",
            "# TYPE gpu_fault_remote_command_executor_internal_errors_total gauge",
            "gpu_fault_remote_command_executor_internal_errors_total "
            f"{remote['executor_internal_error_total']}",
            "# HELP "
            "gpu_fault_remote_command_executor_internal_error_last_seen_"
            "timestamp_seconds Unix timestamp of the newest retained "
            "executor-side defect.",
            "# TYPE "
            "gpu_fault_remote_command_executor_internal_error_last_seen_"
            "timestamp_seconds gauge",
            "gpu_fault_remote_command_executor_internal_error_last_seen_"
            "timestamp_seconds "
            f"{remote['executor_internal_error_last_seen_timestamp_seconds']:.6f}",
            "# HELP gpu_fault_remote_command_unclaimed_expired "
            "Retained remote commands the periodic sweep failed with "
            "status_source=unclaimed-deadline-exceeded because no executor "
            "claimed them within GPU_FAULT_REMOTE_COMMAND_CLAIM_DEADLINE_SECONDS; "
            "falls as retention removes them.",
            "# TYPE gpu_fault_remote_command_unclaimed_expired gauge",
            "gpu_fault_remote_command_unclaimed_expired "
            f"{remote['unclaimed_expired_total']}",
        ]
    )
    lines.extend(_open_sibling_hold_lines(context))
    return lines


def _open_sibling_hold_lines(context: object) -> list[str]:
    """How often the regional adapter held a dispatch because another command
    for the same workflow step was still open (ARCH-D5/D6).

    The adapter is one of the executor's adapters rather than a context
    attribute, so the counter is summed over every adapter that carries it.
    Per process: only the replica whose dispatcher ran the step moves it."""

    executor = getattr(context, "workflow_executor", None)
    adapters = getattr(executor, "adapters", None) if executor is not None else None
    totals = {
        "open_sibling_holds_total": 0,
        "batched_commands_total": 0,
        "batched_steps_total": 0,
    }
    for adapter in adapters or ():
        for name in totals:
            value = getattr(adapter, name, None)
            if isinstance(value, int):
                totals[name] += value
    return [
        "# HELP gpu_fault_remote_command_open_sibling_holds_total dispatches held because another command for the same workflow step was still open (ARCH-D5).",
        "# TYPE gpu_fault_remote_command_open_sibling_holds_total counter",
        f"gpu_fault_remote_command_open_sibling_holds_total {totals['open_sibling_holds_total']}",
        # 性能 C: one compound command replaces one command per node-side
        # step; the steps counter is how many round trips it saved.
        "# HELP gpu_fault_remote_command_batched_commands_total compound remote commands minted for a contiguous run of node-side steps.",
        "# TYPE gpu_fault_remote_command_batched_commands_total counter",
        f"gpu_fault_remote_command_batched_commands_total {totals['batched_commands_total']}",
        "# HELP gpu_fault_remote_command_batched_steps_total workflow steps carried by a compound remote command beyond its head step.",
        "# TYPE gpu_fault_remote_command_batched_steps_total counter",
        f"gpu_fault_remote_command_batched_steps_total {totals['batched_steps_total']}",
    ]


def spare_reservation_metric_lines(runtime: AppRuntime) -> list[str]:
    """Spare-node reservations as of the spare-health controller's last scan
    (ARCH-A4c/A3).

    A reservation is a spare taken out of the pool for a failover that may
    never finish; the reclaimer releases the ones whose incident ended or
    whose TTL lapsed. The controller exists only where spare failover is
    enabled, so a deployment without it publishes no series rather than a
    zero that reads as "no reservations". The snapshot's ``observed_at`` is
    an ISO string, not an epoch, and is deliberately not rendered.
    """

    controller = getattr(runtime.context, "spare_health_controller", None)
    if controller is None:
        return []
    snapshot = controller.metrics_snapshot()

    def read(name: str) -> int:
        value = snapshot.get(name, 0)
        return value if isinstance(value, int) else 0

    return [
        "# HELP gpu_fault_spare_reservations_active Spare nodes currently reserved for a failover, as of the spare-health controller's last scan (ARCH-A4).",
        "# TYPE gpu_fault_spare_reservations_active gauge",
        f"gpu_fault_spare_reservations_active {read('spare_reservations_active')}",
        "# HELP gpu_fault_spare_reservations_reclaimed_total Spare reservations the spare-health controller released because their incident ended without consuming the spare or the reservation TTL lapsed (ARCH-A4).",
        "# TYPE gpu_fault_spare_reservations_reclaimed_total counter",
        "gpu_fault_spare_reservations_reclaimed_total "
        f"{read('spare_reservations_reclaimed_total')}",
    ]


def regional_registry_metric_lines(runtime: AppRuntime) -> list[str]:
    """Whether the regional registry Secret this process started from
    disagrees with the durable registry head (ARCH-H2/H3).

    The durable head wins -- join and remove publish there -- so drift means a
    release that only rewrote the Secret never reached the running registry.
    Only roles that run the registry runtime publish the flag; the runtime is
    bound to the context by the application factory."""

    registry_runtime = getattr(runtime.context, "regional_registry_runtime", None)
    if registry_runtime is None:
        return []
    role = _escape_label(str(getattr(registry_runtime, "service_role", "")))
    lines = [
        "# HELP gpu_fault_regional_registry_secret_drift 1 when the regional registry Secret this process started from differs from the durable registry head, which is authoritative; republish the registry from the release config to reconcile (ARCH-H2).",
        "# TYPE gpu_fault_regional_registry_secret_drift gauge",
        "gpu_fault_regional_registry_secret_drift"
        f'{{service_role="{role}"}} {int(bool(registry_runtime.secret_drift()))}',
    ]
    # A-7: the last refresh error is a gauge, not a readiness criterion -- one
    # closed connection (~1/min baseline) used to fail /healthz for the process.
    status = getattr(registry_runtime, "status", None)
    report = status() if callable(status) else None
    if isinstance(report, dict) and "error" in report:
        lines.extend(
            [
                "# HELP gpu_fault_regional_registry_refresh_error 1 while this process's latest regional registry refresh failed; readiness no longer flips on it while the snapshot is fresh (control-plane review 2026-09-08, A-7).",
                "# TYPE gpu_fault_regional_registry_refresh_error gauge",
                "gpu_fault_regional_registry_refresh_error"
                f'{{service_role="{role}"}} {int(bool(report.get("error")))}',
            ]
        )
    return lines


def _unresolved_fault_signal_lines(service: object) -> list[str]:
    """Kernel/fabric-manager lines that named an Xid/SXid the parser could
    not resolve to a number or class, by kind (ARCH-G7). Absent -- not zero --
    until the factory binds the ingestion service to the context."""

    totals = getattr(service, "unresolved_signal_totals", None) if service else None
    if not isinstance(totals, dict):
        return []
    lines = [
        "# HELP gpu_fault_ingest_unresolved_fault_signals_total Fault-layer log lines carrying an Xid/SXid token whose number or classification could not be parsed; each opened an operator-review finding instead of a policy decision, by kind (ARCH-G7).",
        "# TYPE gpu_fault_ingest_unresolved_fault_signals_total counter",
    ]
    for kind, count in sorted(totals.items()):
        lines.append(
            "gpu_fault_ingest_unresolved_fault_signals_total"
            f'{{kind="{_escape_label(str(kind))}"}} {int(count)}'
        )
    return lines


def fleet_rollout_metric_lines(
    runtime: AppRuntime,
) -> list[str]:
    """Expose how long a fleet rollout has been fencing each cluster.

    ``execution/fleet_preflight.py`` refuses every destructive remediation for a
    cluster while any fleet deployment there is non-terminal. That is correct and
    it is also completely silent: the hold is a ``WARNING`` log line on whichever
    worker happened to poll, and nothing in ``gpu-fault-admin status`` mentions
    fleet deployments at all. A rollout abandoned in ``PLANNED`` fenced a live
    cluster for 36 hours on 2026-09-03 without producing a single signal.

    Age rather than a count, because a count cannot distinguish a rollout that is
    working from one that is stuck -- a healthy regional roll holds a deployment
    open for minutes, and that is exactly what the fence is for.
    """

    context = runtime.context
    if not context.regional_mode:
        return []
    now = datetime.now(timezone.utc)
    # Only the cluster's open deployments, which is an indexed lookup and the
    # same one every agent heartbeat already makes -- not a table scan.
    ages: dict[str, float] = {}
    planned: Counter[str] = Counter()
    for cluster_id in context.store.list_regional_cluster_ids():
        for deployment in context.store.list_active_fleet_deployments(cluster_id):
            age = max(0.0, (now - deployment.created_at).total_seconds())
            ages[cluster_id] = max(ages.get(cluster_id, 0.0), age)
            if deployment.status is DeploymentStatus.PLANNED:
                planned[cluster_id] += 1
    lines = [
        "# HELP gpu_fault_fleet_rollout_fence_age_seconds Age of the oldest "
        "non-terminal fleet deployment, which is how long destructive "
        "remediation has been fenced for the cluster.",
        "# TYPE gpu_fault_fleet_rollout_fence_age_seconds gauge",
    ]
    for cluster_id, age in sorted(ages.items()):
        lines.append(
            "gpu_fault_fleet_rollout_fence_age_seconds"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {age:.6f}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_fleet_rollout_never_started Non-terminal fleet "
            "deployments that have not started a wave, so they fence the "
            "cluster without any rollout being in flight.",
            "# TYPE gpu_fault_fleet_rollout_never_started gauge",
        ]
    )
    for cluster_id in sorted(ages):
        lines.append(
            "gpu_fault_fleet_rollout_never_started"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {planned[cluster_id]}'
        )
    return lines


def fleet_pin_drift_metric_lines(runtime: AppRuntime) -> list[str]:
    """Nodes whose pinned artifact/version disagrees with the control plane,
    split into the two verdicts readiness reaches (ARCH-E E5).

    ``NODE_STALE`` is a node behind a pin its peers already run; the fix is
    on the node. ``PIN_AHEAD_OF_FLEET`` is a pin no live agent in the cluster
    runs: the control plane is waiting for a build that was never shipped,
    and every node action on that cluster fails readiness until the pin is
    corrected. The value is as of the cluster's latest readiness evaluation
    in this process, so it clears on the next evaluation after the fix.
    """

    registry = getattr(runtime.context, "fleet_registry", None)
    drift = getattr(registry, "pin_drift_nodes", None)
    lines = [
        "# HELP gpu_fault_fleet_pin_drift_nodes Nodes whose pinned agent version, artifact, policy, profile or config digest disagrees with the control plane, by cluster and drift kind, as of that cluster's latest readiness evaluation (ARCH-E E5).",
        "# TYPE gpu_fault_fleet_pin_drift_nodes gauge",
    ]
    if not isinstance(drift, dict):
        return lines
    for cluster_id, counts in sorted(drift.items()):
        for kind, count in sorted(counts.items()):
            lines.append(
                "gpu_fault_fleet_pin_drift_nodes"
                f'{{cluster_id="{_escape_label(str(cluster_id))}",'
                f'kind="{_escape_label(str(kind))}"}} {int(count)}'
            )
    return lines


def policy_metric_lines(runtime: AppRuntime) -> list[str]:
    lines = [
        "# HELP gpu_fault_policy_unknown_product_total "
        "GPU product observations that could not be mapped to a "
        "catalog product family.",
        "# TYPE gpu_fault_policy_unknown_product_total counter",
    ]
    for product, count in sorted(
        runtime.context.policy.unknown_product_counts().items()
    ):
        lines.append(
            "gpu_fault_policy_unknown_product_total"
            f'{{product="{_escape_label(product)}"}} {count}'
        )
    return lines


def control_loop_metric_lines(runtime: AppRuntime) -> list[str]:
    """Counters the dispatcher, executor, merge service and periodic runner
    keep in memory (F-L1). Each is an alertable statement about the control
    loop: a job stopped because its nodes stayed busy, a remediation that ran
    out of lifetime, an event recorded instead of planned."""

    ctx = runtime.context
    dispatcher = getattr(ctx, "dispatcher", None)
    executor = getattr(ctx, "workflow_executor", None)
    orchestrator = getattr(ctx, "orchestrator", None)
    merger = getattr(orchestrator, "_workflow_merger", None) if orchestrator else None
    periodic = getattr(ctx, "periodic_runner", None)
    lines: list[str] = []
    dispatch_counters = (
        (
            "node_busy_timeouts_total",
            "Job workflows failed by stopping the job because their nodes stayed under another remediation past the wait (F-N1).",
        ),
        (
            "failure_handling_abandoned_total",
            "Failed workflows whose failure handler kept raising and was given up on (F-A6).",
        ),
        (
            "plan_sync_misses_total",
            "Workflow status changes whose recovery plan row no longer existed (P2-80D).",
        ),
        (
            "internal_errors_total",
            "Dispatch attempts that raised an unrecognised error; the workflow stayed executable with a backoff (F-B4).",
        ),
    )
    for name, help_text in dispatch_counters:
        lines.extend(
            [
                f"# HELP gpu_fault_workflow_dispatch_{name} {help_text}",
                f"# TYPE gpu_fault_workflow_dispatch_{name} counter",
                f"gpu_fault_workflow_dispatch_{name} {getattr(dispatcher, name, 0) if dispatcher else 0}",
            ]
        )
    lines.extend(
        [
            "# HELP gpu_fault_workflow_lifetime_exceeded_total Workflows failed because their hard lifetime passed (F-N1).",
            "# TYPE gpu_fault_workflow_lifetime_exceeded_total counter",
            f"gpu_fault_workflow_lifetime_exceeded_total {getattr(executor, 'lifetime_exceeded_total', 0) if executor else 0}",
            "# HELP gpu_fault_workflow_merge_record_only_total Events recorded on an incident with no new steps, by reason (F-N1).",
            "# TYPE gpu_fault_workflow_merge_record_only_total counter",
        ]
    )
    for reason, attribute in (
        ("covered_read_only", "absorbed_record_only_total"),
        ("lifetime_exceeded", "lifetime_record_only_total"),
        ("workload_withdrawn", "withdrawn_record_only_total"),
    ):
        value = getattr(merger, attribute, 0) if merger is not None else 0
        lines.append(
            f'gpu_fault_workflow_merge_record_only_total{{reason="{reason}"}} {value}'
        )
    escalation = getattr(orchestrator, "_escalation", None) if orchestrator else None
    lines.extend(
        [
            "# HELP gpu_fault_hardware_escalation_chain_terminated_total Failed support-after workflows that were NOT escalated into another support workflow; the incident stayed ESCALATED for an operator (escalation chain bound, ARCH-ESCALATION-BOUND).",
            "# TYPE gpu_fault_hardware_escalation_chain_terminated_total counter",
            f"gpu_fault_hardware_escalation_chain_terminated_total {getattr(escalation, 'escalation_chain_terminated_total', 0) if escalation else 0}",
            "# HELP gpu_fault_hardware_escalation_containment_refused_total Escalations whose containment steps all failed with a safety rejection, so the support workflow was compiled without any isolation step (ARCH-ESCALATION-BOUND).",
            "# TYPE gpu_fault_hardware_escalation_containment_refused_total counter",
            f"gpu_fault_hardware_escalation_containment_refused_total {getattr(escalation, 'containment_refused_escalations_total', 0) if escalation else 0}",
        ]
    )
    closure = getattr(ctx, "incident_closure", None)
    lines.extend(
        [
            "# HELP gpu_fault_incident_operator_closed_total ESCALATED incidents an operator closed RECOVERED through POST /v1/incidents/{id}/close or gpu-fault-admin workflow-reconcile --close-incident (DESTR-018 product gap).",
            "# TYPE gpu_fault_incident_operator_closed_total counter",
            f"gpu_fault_incident_operator_closed_total {getattr(closure, 'operator_closed_total', 0) if closure else 0}",
            "# HELP gpu_fault_incident_auto_closed_by_restore_total ESCALATED incidents closed RECOVERED because a later workflow restored every node they named (DESTR-018 product gap).",
            "# TYPE gpu_fault_incident_auto_closed_by_restore_total counter",
            f"gpu_fault_incident_auto_closed_by_restore_total {getattr(closure, 'auto_closed_by_restore_total', 0) if closure else 0}",
        ]
    )
    store = getattr(ctx, "store", None)
    archiver = getattr(ctx, "control_record_archiver", None)
    lines.extend(
        [
            "# HELP gpu_fault_workflow_placement_holds_opened_total Job workflows opened because a running attempt was observed on a node under another remediation (rule A, case 2).",
            "# TYPE gpu_fault_workflow_placement_holds_opened_total counter",
            f"gpu_fault_workflow_placement_holds_opened_total {getattr(orchestrator, 'placement_holds_opened_total', 0) if orchestrator else 0}",
            "# HELP gpu_fault_workflow_placement_holds_dissolved_total Placement holds ended without executing because their nodes were freed inside the window (rule A, case 2).",
            "# TYPE gpu_fault_workflow_placement_holds_dissolved_total counter",
            f"gpu_fault_workflow_placement_holds_dissolved_total {getattr(dispatcher, 'placement_holds_dissolved_total', 0) if dispatcher else 0}",
            "# HELP gpu_fault_workflow_placement_holds_failed_total Workload observations whose placement hold could not be opened; the observation itself was still accepted (rule A, case 2).",
            "# TYPE gpu_fault_workflow_placement_holds_failed_total counter",
            f"gpu_fault_workflow_placement_holds_failed_total {getattr(orchestrator, 'placement_holds_failed_total', 0) if orchestrator else 0}",
            "# HELP gpu_fault_workflow_dispatch_deferred_total Rows a dispatch cycle scanned but never started because its deadline passed; they stayed PENDING (F-C7).",
            "# TYPE gpu_fault_workflow_dispatch_deferred_total counter",
            f"gpu_fault_workflow_dispatch_deferred_total {getattr(dispatcher, 'deferred_total', 0) if dispatcher else 0}",
            "# HELP gpu_fault_workflow_branch_escalation_budget_refusals_total Node branches retired to an operator because the cluster remediation budget could not take the next rung (F-N1).",
            "# TYPE gpu_fault_workflow_branch_escalation_budget_refusals_total counter",
            f"gpu_fault_workflow_branch_escalation_budget_refusals_total {getattr(executor, 'branch_escalation_budget_refusals_total', 0) if executor else 0}",
            "# HELP gpu_fault_health_signal_clock_regressions_total Host-health samples whose node timestamp went backwards while the control plane's clock moved on; judged on control-plane time (F-M2).",
            "# TYPE gpu_fault_health_signal_clock_regressions_total counter",
            f"gpu_fault_health_signal_clock_regressions_total {getattr(store, 'health_signal_clock_regressions_total', 0) if store is not None else 0}",
            "# HELP gpu_fault_ingest_stale_event_link_repairs_total Duplicate-event fast paths that found a dangling incident or workflow pointer and rebuilt the chain instead of failing the event (F-B7).",
            "# TYPE gpu_fault_ingest_stale_event_link_repairs_total counter",
            f"gpu_fault_ingest_stale_event_link_repairs_total {getattr(store, 'stale_event_link_repairs', 0) if store is not None else 0}",
            "# HELP gpu_fault_control_record_archive_withheld_total Incidents the archiver refused to archive, by safety reason (F-I1).",
            "# TYPE gpu_fault_control_record_archive_withheld_total counter",
        ]
    )
    withheld = (
        getattr(archiver, "withheld_total", None) if archiver is not None else None
    )
    for reason, count in sorted((withheld or {}).items()):
        lines.append(
            "gpu_fault_control_record_archive_withheld_total"
            f'{{reason="{_escape_label(reason)}"}} {count}'
        )
    lines.extend(_unresolved_fault_signal_lines(getattr(ctx, "fault_ingestion", None)))
    lines.extend(_dispatch_pending_state_lines(ctx, dispatcher))
    lines.extend(
        _notification_dispatch_lines(getattr(ctx, "advisory_notifications", None))
    )
    snapshot = periodic.metrics_snapshot() if periodic is not None else {}
    lines.extend(_control_loop_review_lines(dispatcher, archiver))
    lines.extend(_periodic_reconciliation_lines(snapshot))
    lines.extend(_processor_counter_mode_lines(store))
    lines.extend(
        [
            "# HELP gpu_fault_periodic_lease_errors_total Periodic-service task leases that could not be taken because of a store error (F-F1).",
            "# TYPE gpu_fault_periodic_lease_errors_total counter",
            f"gpu_fault_periodic_lease_errors_total {snapshot.get('periodic_lease_errors_total', 0)}",
            "# HELP gpu_fault_periodic_job_errors_total Periodic jobs that raised and were skipped for the tick, by job (F-F1).",
            "# TYPE gpu_fault_periodic_job_errors_total counter",
        ]
    )
    for job, count in sorted(snapshot.get("periodic_job_errors_total", {}).items()):
        lines.append(f'gpu_fault_periodic_job_errors_total{{job="{job}"}} {count}')
    lines.extend(_periodic_liveness_lines(snapshot))
    for name, help_text in (
        (
            "cleanup_rows_total",
            "Rows removed or expired by the periodic cleanup, by job (F-F2).",
        ),
        (
            "cleanup_budget_exhausted_total",
            "Cleanup rounds a job ended still saturated because its share of the budget ran out (F-F2).",
        ),
        (
            "cleanup_job_errors_total",
            "Cleanup rounds a job ended early on a store error (F-F2).",
        ),
    ):
        lines.extend(
            [
                f"# HELP gpu_fault_periodic_{name} {help_text}",
                f"# TYPE gpu_fault_periodic_{name} counter",
            ]
        )
        for job, count in sorted(snapshot.get(name, {}).items()):
            lines.append(
                f'gpu_fault_periodic_{name}{{job="{_escape_label(job)}"}} {count}'
            )
    return lines


def _labelled_counter(
    name: str, help_text: str, label: str, values: object
) -> list[str]:
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
    if isinstance(values, dict):
        for key, count in sorted(values.items()):
            lines.append(f'{name}{{{label}="{_escape_label(str(key))}"}} {int(count)}')
    return lines


def _control_loop_review_lines(dispatcher: object, archiver: object) -> list[str]:
    """Counters the control-plane review 2026-09-08 added (D-4, F-8).

    Each is a failure that used to be a log line and nothing else: a dispatcher
    sweep path that raised, an archiver run that failed on one incident. The
    archiver's success count sits beside its errors so a rate of zero can be
    told from a retention that is switched off.
    """

    lines = _labelled_counter(
        "gpu_fault_workflow_dispatch_sweep_errors_total",
        "Dispatcher sweep paths (abandoned generation, placement hold, node-busy hold/timeout, internal-error block/release) that raised and were counted rather than aborting the cycle, by path (D-4).",
        "path",
        getattr(dispatcher, "sweep_errors_total", None) if dispatcher else None,
    )
    archived = getattr(archiver, "archived_total", None) if archiver else None
    lines.extend(
        [
            "# HELP gpu_fault_control_record_archive_archived_total Incidents the archiver bundled to S3 and deleted, with their records (F-8).",
            "# TYPE gpu_fault_control_record_archive_archived_total counter",
            f"gpu_fault_control_record_archive_archived_total {int(archived) if isinstance(archived, int) else 0}",
        ]
    )
    lines.extend(
        _labelled_counter(
            "gpu_fault_control_record_archive_errors_total",
            "Archiver candidates that failed with an exception other than a safety refusal, by exception type; the run continued with the next candidate (F-8).",
            "reason",
            getattr(archiver, "errors_total", None) if archiver else None,
        )
    )
    return lines


def _notification_dispatch_lines(service: object) -> list[str]:
    """What the notification dispatcher gave up on, and when it last ran
    (ARCH-E E1/E3). Per process: only replicas that drain the outbox move
    these, so alerts sum or max them across the Deployment."""

    def read(name: str, default: float) -> float | int:
        if service is None:
            return default
        value = getattr(service, name, default)
        return value if isinstance(value, (int, float)) else default

    return [
        "# HELP gpu_fault_notification_expired_total Notifications retired unsent because they were claimed past their shelf life (GPU_FAULT_NOTIFICATION_TTL_SECONDS); the outbox drained slower than it filled or was blocked (ARCH-E E1).",
        "# TYPE gpu_fault_notification_expired_total counter",
        f"gpu_fault_notification_expired_total {read('expired_total', 0)}",
        "# HELP gpu_fault_notification_expired_last_seen_timestamp_seconds Unix time this process last retired a notification unsent past its shelf life; 0 if never. Alerts read max() of this rather than increase() of the counter (ARCH-E E4).",
        "# TYPE gpu_fault_notification_expired_last_seen_timestamp_seconds gauge",
        f"gpu_fault_notification_expired_last_seen_timestamp_seconds {read('expired_last_seen_timestamp_seconds', 0.0):.3f}",
        "# HELP gpu_fault_notification_dead_lettered_total Notifications retired DEAD after exhausting GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS delivery attempts; each is an administrator who was never reached (ARCH-E E1).",
        "# TYPE gpu_fault_notification_dead_lettered_total counter",
        f"gpu_fault_notification_dead_lettered_total {read('dead_lettered_total', 0)}",
        "# HELP gpu_fault_notification_suppressed_drills_total Drill notifications the dispatcher retired instead of mailing (GPU_FAULT_NOTIFICATION_DELIVER_DRILLS is off).",
        "# TYPE gpu_fault_notification_suppressed_drills_total counter",
        f"gpu_fault_notification_suppressed_drills_total {read('suppressed_drills_total', 0)}",
        "# HELP gpu_fault_notification_dispatch_last_cycle_timestamp_seconds Unix time this process last started a notification outbox dispatch cycle; 0 on replicas that never drain the outbox (ARCH-E E3).",
        "# TYPE gpu_fault_notification_dispatch_last_cycle_timestamp_seconds gauge",
        "gpu_fault_notification_dispatch_last_cycle_timestamp_seconds "
        f"{read('last_cycle_timestamp_seconds', 0.0):.3f}",
        "# HELP gpu_fault_notification_delivery_errors_total Outbox deliveries whose bookkeeping or provider call raised; the row was released for retry and the batch continued (control-plane review 2026-09-08, F-3).",
        "# TYPE gpu_fault_notification_delivery_errors_total counter",
        f"gpu_fault_notification_delivery_errors_total {read('delivery_errors_total', 0)}",
        "# HELP gpu_fault_notification_delivery_error_last_seen_timestamp_seconds Unix time this process last hit a delivery error; 0 if never (F-3).",
        "# TYPE gpu_fault_notification_delivery_error_last_seen_timestamp_seconds gauge",
        "gpu_fault_notification_delivery_error_last_seen_timestamp_seconds "
        f"{read('delivery_error_last_seen_timestamp_seconds', 0.0):.3f}",
    ]


# Filter reasons a dispatch cycle always has a name for; exported at zero so
# an alert can rate() them before the first row is ever set aside (F-L1).
DISPATCH_FILTER_REASONS = ("batch_limit", "preemption_pending")


def _dispatch_pending_state_lines(ctx: object, dispatcher: object) -> list[str]:
    """What the dispatcher knows about rows it did not start (F-L1)."""

    def read(name: str, default: float) -> float | int:
        if dispatcher is None:
            return default
        value = getattr(dispatcher, name, default)
        return value if isinstance(value, (int, float)) else default

    lines = [
        "# HELP gpu_fault_workflow_dispatch_preemption_pending_seen_total Dispatch cycles that saw a PENDING row set aside because a pre-emption on the same nodes was still open (F-A2).",
        "# TYPE gpu_fault_workflow_dispatch_preemption_pending_seen_total counter",
        f"gpu_fault_workflow_dispatch_preemption_pending_seen_total {read('preemption_pending_seen_total', 0)}",
        "# HELP gpu_fault_workflow_pending_age_seconds_max Age of the oldest PENDING workflow the last dispatch cycle scanned (F-A5).",
        "# TYPE gpu_fault_workflow_pending_age_seconds_max gauge",
        f"gpu_fault_workflow_pending_age_seconds_max {read('pending_age_seconds_max', 0.0):g}",
        "# HELP gpu_fault_workflow_pending_age_warnings_total Dispatch cycles whose oldest PENDING workflow was older than the warning threshold (F-A5).",
        "# TYPE gpu_fault_workflow_pending_age_warnings_total counter",
        f"gpu_fault_workflow_pending_age_warnings_total {read('pending_age_warnings_total', 0)}",
        "# HELP gpu_fault_workflow_retired_generation_awaiting_operator Retired workflow generations the last dispatch cycle left for an operator instead of superseding (F-A5).",
        "# TYPE gpu_fault_workflow_retired_generation_awaiting_operator gauge",
        f"gpu_fault_workflow_retired_generation_awaiting_operator {read('retired_generation_awaiting_operator', 0)}",
        "# HELP gpu_fault_workflow_dispatch_internal_error_last_seen_timestamp_seconds Unix time of this process's newest dispatch internal error (see gpu_fault_workflow_dispatch_internal_errors_total); 0 if none. Alerts read max() of this because a multi-process Pod is scraped one process at a time (ARCH-E E4).",
        "# TYPE gpu_fault_workflow_dispatch_internal_error_last_seen_timestamp_seconds gauge",
        f"gpu_fault_workflow_dispatch_internal_error_last_seen_timestamp_seconds {read('internal_error_last_seen_timestamp_seconds', 0.0):.3f}",
        "# HELP gpu_fault_workflow_dispatch_failure_handling_abandoned_last_seen_timestamp_seconds Unix time this process last gave up on a failed workflow's failure handler (see gpu_fault_workflow_dispatch_failure_handling_abandoned_total); 0 if never (ARCH-E E4).",
        "# TYPE gpu_fault_workflow_dispatch_failure_handling_abandoned_last_seen_timestamp_seconds gauge",
        f"gpu_fault_workflow_dispatch_failure_handling_abandoned_last_seen_timestamp_seconds {read('failure_handling_abandoned_last_seen_timestamp_seconds', 0.0):.3f}",
        "# HELP gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds Unix time this process last started a workflow dispatch cycle, lease held or not; 0 on replicas whose dispatcher never ran (ARCH-E E3).",
        "# TYPE gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds gauge",
        f"gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds {read('last_cycle_timestamp_seconds', 0.0):.3f}",
        "# HELP gpu_fault_workflow_dispatch_wakeup_last_seen_timestamp_seconds Unix time this process last turned a store wakeup into an early dispatch scan (see gpu_fault_workflow_dispatch_wakeups_total); 0 if never (ARCH-E E4).",
        "# TYPE gpu_fault_workflow_dispatch_wakeup_last_seen_timestamp_seconds gauge",
        f"gpu_fault_workflow_dispatch_wakeup_last_seen_timestamp_seconds {read('wakeup_last_seen_timestamp_seconds', 0.0):.3f}",
        "# HELP gpu_fault_workflow_dispatch_wakeups_total Store wakeups this process turned into an early dispatch scan, by channel; a remote-command payload counts only on SUCCEEDED/FAILED, the statuses that let a WAITING step advance (性能 A).",
        "# TYPE gpu_fault_workflow_dispatch_wakeups_total counter",
        *_dispatch_wakeup_channel_lines(
            "gpu_fault_workflow_dispatch_wakeups_total",
            getattr(dispatcher, "wakeups_total", None),
            0,
        ),
        "# HELP gpu_fault_workflow_dispatch_wakeup_listener_connected 1 while this process's LISTEN thread for the wakeup channel is connected, 0 while the dispatcher is polling only for it; any disconnected process makes the Pod's value 0.",
        "# TYPE gpu_fault_workflow_dispatch_wakeup_listener_connected gauge",
        *_dispatch_wakeup_channel_lines(
            "gpu_fault_workflow_dispatch_wakeup_listener_connected",
            getattr(dispatcher, "wakeup_listener_connected", None),
            False,
        ),
        "# HELP gpu_fault_workflow_dispatch_filtered_total Rows a dispatch cycle scanned and set aside, by reason, summed over the process (F-L1).",
        "# TYPE gpu_fault_workflow_dispatch_filtered_total counter",
    ]
    filtered = getattr(dispatcher, "filtered_total", None)
    totals: dict[str, int] = dict.fromkeys(DISPATCH_FILTER_REASONS, 0)
    if isinstance(filtered, dict):
        for reason, count in filtered.items():
            totals[str(reason)] = int(count)
    for reason, count in sorted(totals.items()):
        lines.append(
            "gpu_fault_workflow_dispatch_filtered_total"
            f'{{reason="{_escape_label(reason)}"}} {count}'
        )
    return lines


# Both channels are always rendered, zero included, so an alert on a channel
# that never connected has a sample to read (same reason as
# ``DISPATCH_FILTER_REASONS``).
DISPATCH_WAKEUP_CHANNELS = ("remote_command", "workflow_dispatch")


def _dispatch_wakeup_channel_lines(
    family: str, values: object, default: int | bool
) -> list[str]:
    totals: dict[str, int] = dict.fromkeys(DISPATCH_WAKEUP_CHANNELS, int(default))
    if isinstance(values, dict):
        for channel, value in values.items():
            totals[str(channel)] = int(value)
    return [
        f'{family}{{channel="{_escape_label(channel)}"}} {count}'
        for channel, count in sorted(totals.items())
    ]


def _periodic_liveness_lines(snapshot: dict[str, object]) -> list[str]:
    """When the periodic runner last ticked, and when each job last ran
    (ARCH-E E3). The runner stamp is per process and moves on every tick
    whether or not this replica owns any task lease; the per-job stamps move
    only on the replica that ran the job, so an alert reads their max across
    the Deployment."""

    stamp = snapshot.get("last_cycle_timestamp_seconds", 0.0)
    lines = [
        "# HELP gpu_fault_periodic_last_cycle_timestamp_seconds Unix time this process's periodic service runner last completed a tick; 0 where the runner never started (ARCH-E E3).",
        "# TYPE gpu_fault_periodic_last_cycle_timestamp_seconds gauge",
        "gpu_fault_periodic_last_cycle_timestamp_seconds "
        f"{stamp if isinstance(stamp, (int, float)) else 0.0:.3f}",
        "# HELP gpu_fault_periodic_job_last_run_timestamp_seconds Unix time this process last ran a periodic job, by job; only the task-lease holder moves it (ARCH-E E3).",
        "# TYPE gpu_fault_periodic_job_last_run_timestamp_seconds gauge",
    ]
    stamps = snapshot.get("job_last_run_timestamp_seconds", {})
    if isinstance(stamps, dict):
        for job, value in sorted(stamps.items()):
            if isinstance(value, (int, float)):
                lines.append(
                    "gpu_fault_periodic_job_last_run_timestamp_seconds"
                    f'{{job="{_escape_label(str(job))}"}} {value:.3f}'
                )
    lease_error = snapshot.get("lease_error_last_seen_timestamp_seconds", 0.0)
    lines.extend(
        [
            "# HELP gpu_fault_periodic_lease_error_last_seen_timestamp_seconds Unix time of this process's newest failed task-lease attempt (see gpu_fault_periodic_lease_errors_total); 0 if none. Alerts read max() of this because a multi-process Pod is scraped one process at a time (ARCH-E E4).",
            "# TYPE gpu_fault_periodic_lease_error_last_seen_timestamp_seconds gauge",
            "gpu_fault_periodic_lease_error_last_seen_timestamp_seconds "
            f"{lease_error if isinstance(lease_error, (int, float)) else 0.0:.3f}",
            "# HELP gpu_fault_periodic_job_error_last_seen_timestamp_seconds Unix time of this process's newest exception in a periodic job, by job (see gpu_fault_periodic_job_errors_total) (ARCH-E E4).",
            "# TYPE gpu_fault_periodic_job_error_last_seen_timestamp_seconds gauge",
        ]
    )
    job_errors = snapshot.get("job_error_last_seen_timestamp_seconds", {})
    if isinstance(job_errors, dict):
        for job, value in sorted(job_errors.items()):
            if isinstance(value, (int, float)):
                lines.append(
                    "gpu_fault_periodic_job_error_last_seen_timestamp_seconds"
                    f'{{job="{_escape_label(str(job))}"}} {value:.3f}'
                )
    return lines


def _periodic_reconciliation_lines(snapshot: dict[str, object]) -> list[str]:
    """The reconciliation jobs the periodic runner grew in batch 3: lease
    reclaim (F-D5) and processor counter drift (F-D10). The drift gauges are
    refreshed by the job, so a scrape never counts the queue table."""

    def read(name: str) -> int:
        value = snapshot.get(name, 0)
        return value if isinstance(value, int) else 0

    return [
        "# HELP gpu_fault_processor_expired_leases_reclaimed_total LEASED processor requests whose lease had lapsed and were handed back to PENDING by the periodic reclaim (F-D5).",
        "# TYPE gpu_fault_processor_expired_leases_reclaimed_total counter",
        f"gpu_fault_processor_expired_leases_reclaimed_total {read('processor_expired_leases_reclaimed_total')}",
        "# HELP gpu_fault_processor_counter_drift_abs Absolute gap between incomplete processor queue rows and the per-cluster counter table, as of the last drift scan (F-D10).",
        "# TYPE gpu_fault_processor_counter_drift_abs gauge",
        f"gpu_fault_processor_counter_drift_abs {read('processor_counter_drift_abs')}",
        "# HELP gpu_fault_processor_counter_mismatched_clusters Clusters whose processor counter disagrees with their queue rows, as of the last drift scan (F-D10).",
        "# TYPE gpu_fault_processor_counter_mismatched_clusters gauge",
        f"gpu_fault_processor_counter_mismatched_clusters {read('processor_counter_mismatched_clusters')}",
    ]


# Every value the processor counter mode can take; one series per value so a
# panel can select on the label without knowing which one is active.
PROCESSOR_COUNTER_MODES = ("dual", "partitioned")


def _processor_counter_mode_lines(store: object) -> list[str]:
    """Which processor counter table admission reads (store review 2026-09-07,
    item I). ``dual`` still locks the single per-cluster counter row on every
    enqueue, so the shards relieve nothing until ``partitioned``; production
    sat in ``dual`` with nothing exporting it. Only the Postgres store has the
    accessor; SQLite and memory stores emit no series."""

    read = getattr(store, "processor_counter_mode", None)
    if read is None:
        return []
    active = str(read())
    lines = [
        "# HELP gpu_fault_processor_counter_mode Processor admission counter mode; the 16 priority shards relieve counter-row contention only in partitioned mode; finalize requires an empty queue (gpu-fault-store-migrate --finalize-processor-counter-shards)",
        "# TYPE gpu_fault_processor_counter_mode gauge",
    ]
    modes = list(PROCESSOR_COUNTER_MODES)
    if active not in modes:
        # An unexpected value is worth a series of its own rather than a row
        # of zeros that reads as "no mode".
        modes.append(active)
    for mode in modes:
        lines.append(
            "gpu_fault_processor_counter_mode"
            f'{{mode="{_escape_label(mode)}"}} {int(mode == active)}'
        )
    return lines


def completion_state_metric_lines(runtime: AppRuntime) -> list[str]:
    """Where the completion path and the incidents stand, as gauges (F-L1).

    A terminal event without a decision is the poisoned shape of P0-48B; the
    ESCALATED incident bucket is the operator queue; a CRITICAL GPU finding
    closed without an incident is the one deliberate exception of F-M1.
    Each is a server-side count, never a decode of the rows.
    """

    store = runtime.context.store
    lines = [
        "# HELP gpu_fault_completion_decisions Completion decisions by status (F-G2).",
        "# TYPE gpu_fault_completion_decisions gauge",
    ]
    decisions = store.decision_status_counts()
    for status in DecisionStatus:
        lines.append(
            f'gpu_fault_completion_decisions{{status="{status.value}"}} '
            f"{decisions.get(status, 0)}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_completion_events_without_decision Terminal events with no completion decision row (P0-48B).",
            "# TYPE gpu_fault_completion_events_without_decision gauge",
            "gpu_fault_completion_events_without_decision "
            f"{store.count_completion_events_without_decision()}",
            "# HELP gpu_fault_incidents_by_state Fault incidents by state; ESCALATED is the operator queue.",
            "# TYPE gpu_fault_incidents_by_state gauge",
        ]
    )
    incidents = store.incident_state_counts()
    for state in IncidentState:
        lines.append(
            f'gpu_fault_incidents_by_state{{state="{state.value}"}} '
            f"{incidents.get(state, 0)}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_gpu_findings_without_incident_total New CRITICAL GPU findings deliberately closed without an incident of their own, by reason (F-M1).",
            "# TYPE gpu_fault_gpu_findings_without_incident_total counter",
        ]
    )
    gpu_metrics = getattr(runtime.context, "gpu_metrics", None)
    findings = getattr(gpu_metrics, "findings_without_incident", None)
    reasons: dict[str, int] = {"suppressed_by_composite": 0}
    if isinstance(findings, dict):
        for reason, count in findings.items():
            reasons[str(reason)] = int(count)
    for reason, count in sorted(reasons.items()):
        lines.append(
            "gpu_fault_gpu_findings_without_incident_total"
            f'{{reason="{_escape_label(reason)}"}} {count}'
        )
    return lines


def postgres_pool_metric_lines(runtime: AppRuntime) -> list[str]:
    """The role's connection demand against its pool, as a gauge (F-E5).

    The guard used to be one startup WARNING that nobody sees during a rolling
    release, and it undercounted the worker role so badly it never fired there.
    """

    estimate = getattr(runtime.context, "postgres_pool_capacity", None)
    if estimate is None:
        return []
    lines = [
        "# HELP gpu_fault_postgres_pool_max_size Configured per-process PostgreSQL pool ceiling.",
        "# TYPE gpu_fault_postgres_pool_max_size gauge",
        f"gpu_fault_postgres_pool_max_size {estimate.pool_max}",
        "# HELP gpu_fault_postgres_pool_demand_connections Threads of this role that can hold a pooled connection at once, by consumer (F-E5).",
        "# TYPE gpu_fault_postgres_pool_demand_connections gauge",
    ]
    for consumer, count in sorted(estimate.demand_by_consumer.items()):
        lines.append(
            "gpu_fault_postgres_pool_demand_connections"
            f'{{consumer="{_escape_label(consumer)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_postgres_pool_oversubscription_ratio Pooled connection demand divided by the pool ceiling; above 1 callers queue on checkout (F-E5).",
            "# TYPE gpu_fault_postgres_pool_oversubscription_ratio gauge",
            f"gpu_fault_postgres_pool_oversubscription_ratio {estimate.oversubscription_ratio:g}",
            "# HELP gpu_fault_postgres_unpooled_connections LISTEN connections opened beside the pool; they count against the server's max_connections, not the pool.",
            "# TYPE gpu_fault_postgres_unpooled_connections gauge",
            f"gpu_fault_postgres_unpooled_connections {estimate.unpooled_connections}",
        ]
    )
    headroom = getattr(estimate, "headroom", None)
    if isinstance(headroom, int):
        lines.extend(
            [
                "# HELP gpu_fault_postgres_pool_headroom_connections Pool ceiling minus the estimated connection demand; negative means the role's threads outnumber its pool (control-plane review 2026-09-08, A-5).",
                "# TYPE gpu_fault_postgres_pool_headroom_connections gauge",
                f"gpu_fault_postgres_pool_headroom_connections {headroom}",
            ]
        )
    return lines


def orchestration_metric_lines(
    runtime: AppRuntime,
) -> list[str]:
    evidence = runtime.context.orchestrator._evidence_operations
    cache = metric_scan_cache(runtime)
    scan = cache.observation_states()
    snapshot = evidence.ownership_metric_snapshot(
        agents=cache.agents(),
        observation_states=scan.states,
    )
    lines = [
        "# HELP gpu_fault_ambiguous_attempt_ownership_total "
        "Events rejected because more than one active attempt owned "
        "the target node.",
        "# TYPE gpu_fault_ambiguous_attempt_ownership_total counter",
        "gpu_fault_ambiguous_attempt_ownership_total "
        f"{evidence.ambiguous_attempt_ownership_total()}",
        "# HELP gpu_fault_ambiguous_attempt_ownership_current "
        "Fresh active attempts currently claiming one GPU node.",
        "# TYPE gpu_fault_ambiguous_attempt_ownership_current gauge",
    ]
    for (cluster_id, node_id), count in snapshot["current"].items():
        lines.append(
            "gpu_fault_ambiguous_attempt_ownership_current"
            f'{{cluster_id="{_escape_label(cluster_id)}",'
            f'gpu_node="{_escape_label(node_id)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_stale_attempt_observations "
            "Latest active attempt observations older than the ownership "
            "freshness window for one GPU node.",
            "# TYPE gpu_fault_stale_attempt_observations gauge",
        ]
    )
    for (cluster_id, node_id), count in snapshot["stale"].items():
        lines.append(
            "gpu_fault_stale_attempt_observations"
            f'{{cluster_id="{_escape_label(cluster_id)}",'
            f'gpu_node="{_escape_label(node_id)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_attempt_observation_scan_limit Attempt "
            "observations the ownership family is allowed to read per scrape.",
            "# TYPE gpu_fault_attempt_observation_scan_limit gauge",
            f"gpu_fault_attempt_observation_scan_limit {scan.limit}",
            "# HELP gpu_fault_attempt_observation_scan_size Attempt "
            "observations actually read for the ownership family.",
            "# TYPE gpu_fault_attempt_observation_scan_size gauge",
            f"gpu_fault_attempt_observation_scan_size {len(scan.states)}",
            "# HELP gpu_fault_attempt_observation_scan_truncated Set when the "
            "attempt observation table is larger than the scan budget, so the "
            "ownership and staleness families describe only its newest slice.",
            "# TYPE gpu_fault_attempt_observation_scan_truncated gauge",
            f"gpu_fault_attempt_observation_scan_truncated {int(scan.truncated)}",
        ]
    )
    return lines


def workflow_stall_metric_lines(
    workflows: Sequence[WorkflowRequest],
    now: datetime,
    terminal: Container[WorkflowStatus],
    config: ProductionExecutorConfig,
) -> list[str]:
    """Report a step that cannot finish, and a deadline that was not enforced.

    Step metrics were counts by status, so a step re-asking the same question
    forever looked exactly like a step that had just started waiting; and
    ``execution_deadline`` had no metric at all, which is why a workflow ten
    minutes past it could keep running with every dashboard green. Both are
    scoped to non-terminal workflows: a WAITING execution left on a workflow that
    has already ended is history, not a stall.

    The age is published next to the threshold that applies to it because the
    threshold is per-operation: a delegated replacement's ceiling is the provider
    window, everything else's is the default cap. An alert carrying its own copy
    of one number could only be right for one of those, and would go stale the
    first time either is retuned.
    """

    step_waiting_ages: dict[str, float] = {}
    warning_limits: dict[str, int] = {}
    overdue_seconds = 0.0
    for workflow in workflows:
        if workflow.status in terminal:
            continue
        if workflow.execution_deadline is not None:
            overdue_seconds = max(
                overdue_seconds,
                (now - workflow.execution_deadline).total_seconds(),
            )
        for execution in workflow.step_executions:
            if execution.status is not WorkflowStepStatus.WAITING:
                continue
            operation = execution.operation.value
            step_waiting_ages[operation] = max(
                step_waiting_ages.get(operation, 0.0),
                max(0.0, (now - execution.started_at).total_seconds()),
            )
            warning_limits[operation] = config.step_waiting_warning_limit(
                execution.operation
            )
    lines = [
        "# HELP gpu_fault_workflow_step_waiting_seconds Age of the oldest step "
        "still WAITING in a non-terminal workflow, by operation. A step whose "
        "own retry counter has stopped advancing shows up here and nowhere "
        "else.",
        "# TYPE gpu_fault_workflow_step_waiting_seconds gauge",
    ]
    lines.extend(
        "gpu_fault_workflow_step_waiting_seconds"
        f'{{operation="{_escape_label(operation)}"}} {age:.3f}'
        for operation, age in sorted(step_waiting_ages.items())
    )
    lines.extend(
        [
            "# HELP gpu_fault_workflow_step_waiting_warning_seconds How long this "
            "operation may wait before the wait itself is the problem. Emitted "
            "with the same labels as the age above so an alert compares the two "
            "series instead of hard-coding either threshold.",
            "# TYPE gpu_fault_workflow_step_waiting_warning_seconds gauge",
        ]
    )
    lines.extend(
        "gpu_fault_workflow_step_waiting_warning_seconds"
        f'{{operation="{_escape_label(operation)}"}} {limit}'
        for operation, limit in sorted(warning_limits.items())
    )
    lines.extend(
        [
            "# HELP gpu_fault_workflow_overdue_seconds How far the most overdue "
            "non-terminal workflow is past its execution deadline. Above zero "
            "means a deadline was not enforced.",
            "# TYPE gpu_fault_workflow_overdue_seconds gauge",
            f"gpu_fault_workflow_overdue_seconds {max(0.0, overdue_seconds):.3f}",
        ]
    )
    return lines


def _remediation_budget_lines(
    *,
    active_by_scope_type: Counter[str],
    active_by_cluster: Counter[str],
    waiting_by_scope_type: Counter[str],
    waiting_by_cluster: Counter[str],
    wait_total: int,
    waiting: int,
    config: ProductionExecutorConfig | None,
) -> list[str]:
    """Render the remediation budget families from the workflow scan tallies.

    The per-cluster families exist because the scope-type split cannot say
    *which* cluster is saturated (S1): a correlated whole-cluster fault drains
    at ``cluster_limit`` concurrent remediations, and an operator needs the
    active/limit pair and the waiting count for that cluster to estimate the
    drain time. Labels are the cluster id only -- never a node -- so the
    cardinality is bounded by the fleet, not by the fault.

    The limit gauge is omitted, not zeroed, when there is no executor policy
    to read: a fabricated zero would make the saturation alert's ``limit > 0``
    guard read as "never saturated".
    """

    lines = [
        "# HELP gpu_fault_remediation_budget_active_claims Active durable "
        "remediation claims by scope type.",
        "# TYPE gpu_fault_remediation_budget_active_claims gauge",
    ]
    for scope_type, count in sorted(active_by_scope_type.items()):
        lines.append(
            "gpu_fault_remediation_budget_active_claims"
            f'{{scope_type="{_escape_label(scope_type)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_cluster_active_claims RUNNING "
            "workflows with a live execution lease that hold this cluster's "
            "remediation budget claim.",
            "# TYPE gpu_fault_remediation_budget_cluster_active_claims gauge",
        ]
    )
    for cluster_id, count in sorted(active_by_cluster.items()):
        lines.append(
            "gpu_fault_remediation_budget_cluster_active_claims"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {count}'
        )
    if config is not None:
        lines.extend(
            [
                "# HELP gpu_fault_remediation_budget_cluster_limit Configured "
                "per-cluster remediation concurrency limit "
                "(GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER); a safety "
                "limit, the same for every cluster.",
                "# TYPE gpu_fault_remediation_budget_cluster_limit gauge",
                "gpu_fault_remediation_budget_cluster_limit "
                f"{config.remediation_budget.cluster_limit}",
            ]
        )
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_wait_total Workflow claim "
            "attempts held by a remediation budget.",
            "# TYPE gpu_fault_remediation_budget_wait_total gauge",
            f"gpu_fault_remediation_budget_wait_total {wait_total}",
            "# HELP gpu_fault_remediation_budget_waiting_workflows Workflows "
            "currently waiting for remediation capacity.",
            "# TYPE gpu_fault_remediation_budget_waiting_workflows gauge",
            f"gpu_fault_remediation_budget_waiting_workflows {waiting}",
            "# HELP gpu_fault_remediation_budget_waiting_workflows_by_scope "
            "Waiting workflows by the scope type of the budget that last "
            "refused them.",
            "# TYPE gpu_fault_remediation_budget_waiting_workflows_by_scope gauge",
        ]
    )
    for scope_type, count in sorted(waiting_by_scope_type.items()):
        lines.append(
            "gpu_fault_remediation_budget_waiting_workflows_by_scope"
            f'{{scope_type="{_escape_label(scope_type)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_remediation_budget_cluster_waiting_workflows "
            "Workflows waiting because this cluster's remediation budget was "
            "full when they last tried to claim.",
            "# TYPE gpu_fault_remediation_budget_cluster_waiting_workflows gauge",
        ]
    )
    for cluster_id, count in sorted(waiting_by_cluster.items()):
        lines.append(
            "gpu_fault_remediation_budget_cluster_waiting_workflows"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {count}'
        )
    return lines


def _orphan_inspection_counts(
    runtime: AppRuntime, cache: MetricScanCache
) -> tuple[int, int]:
    """The orphan inspection (F-B3 (4)): bounded server-side reads.

    A PENDING record younger than the aggregation window plus the processor
    drain wait is still normal churn, so those are excluded before counting.
    Both inspections walk a whole kind (the missing-workflow anti-join read
    every incident on every scrape, G-2), so they are shared across scrapes
    for the scan cache's TTL rather than re-run per scrape per process.
    """

    store = runtime.context.store
    orchestrator = runtime.context.orchestrator
    orphan_grace = timedelta(
        seconds=orchestrator.multi_node_aggregation_window_max_seconds
        + orchestrator.processor_drain_max_wait_seconds
    )
    orphans = cache.shared(
        "orphan_workflows",
        lambda: len(
            store.list_orphan_workflows(
                created_before=datetime.now(timezone.utc) - orphan_grace,
                limit=ORPHAN_INSPECTION_LIMIT,
            )
        ),
    )
    dangling = cache.shared(
        "dangling_incident_pointers",
        lambda: len(
            store.list_incidents_with_missing_workflow(limit=ORPHAN_INSPECTION_LIMIT)
        ),
    )
    return orphans, dangling


def closed_loop_metric_lines(runtime: AppRuntime) -> list[str]:
    store = runtime.context.store
    scan = metric_scan_cache(runtime).workflows()
    workflows = scan.workflows
    # The status gauge is a server-side aggregate, so it stays exact even when
    # the detail scan below only covers the newest slice of the table.
    workflow_statuses = {
        status.value: count for status, count in store.workflow_status_counts().items()
    }
    # Also a server-side aggregate, and the one the BLOCKED backlog alert reads:
    # the status gauge counts every workflow ever persisted, so its BLOCKED
    # bucket never falls just because the node came back.
    blocked_unreconciled = store.blocked_workflows_without_verified_restore()
    cache = metric_scan_cache(runtime)
    orphan_workflows, dangling_incident_pointers = _orphan_inspection_counts(
        runtime, cache
    )
    step_statuses: Counter[tuple[str, str]] = Counter()
    terminal_durations: dict[str, list[float]] = defaultdict(list)
    milestone_durations: dict[str, list[float]] = defaultdict(list)
    budget_scope_types: Counter[str] = Counter()
    budget_cluster_active: Counter[str] = Counter()
    budget_cluster_waiting: Counter[str] = Counter()
    budget_waiting_scope_types: Counter[str] = Counter()
    budget_wait_total = budget_waiting = 0
    now = datetime.now(timezone.utc)
    terminal = {
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    }
    budget_waiting_statuses = {
        WorkflowStatus.PENDING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.SAFETY_PENDING,
    }
    milestone_operations = {
        WorkflowOperation.MARK_UNSCHEDULABLE: "containment",
        WorkflowOperation.QUARANTINE: "containment",
        WorkflowOperation.VALIDATE_GPU: "validation",
        WorkflowOperation.VALIDATE_HOST: "validation",
        WorkflowOperation.VALIDATE_FABRIC: "validation",
        WorkflowOperation.RESTORE_SCHEDULING: "readmission",
        WorkflowOperation.RESTART_WORKLOAD: "workload_restart",
    }
    for workflow in workflows:
        if workflow.status in terminal:
            terminal_durations[workflow.status.value].append(
                max(0.0, (workflow.updated_at - workflow.created_at).total_seconds())
            )
        seen_milestones = set()
        for execution in workflow.step_executions:
            step_statuses[
                (
                    execution.operation.value,
                    execution.status.value,
                )
            ] += 1
            milestone = milestone_operations.get(execution.operation)
            if milestone is None or milestone in seen_milestones:
                continue
            if execution.status.value != "SUCCEEDED":
                continue
            seen_milestones.add(milestone)
            milestone_durations[milestone].append(
                max(
                    0.0,
                    (execution.updated_at - workflow.created_at).total_seconds(),
                )
            )
        if (
            workflow.status is WorkflowStatus.RUNNING
            and workflow.execution_lease_expires_at is not None
            and workflow.execution_lease_expires_at > now
        ):
            for scope in workflow.remediation_budget_claims:
                budget_scope_types[scope.split(":", 1)[0]] += 1
                if scope.startswith("cluster:"):
                    budget_cluster_active[scope.removeprefix("cluster:")] += 1
        budget_wait_total += workflow.remediation_budget_wait_count
        if (
            workflow.remediation_budget_last_blocked_reason is not None
            and workflow.status in budget_waiting_statuses
        ):
            budget_waiting += 1
            blocked_scope = workflow.remediation_budget_last_blocked_scope or ""
            if blocked_scope:
                budget_waiting_scope_types[blocked_scope.split(":", 1)[0]] += 1
            if blocked_scope.startswith("cluster:"):
                budget_cluster_waiting[blocked_scope.removeprefix("cluster:")] += 1

    lines = [
        "# HELP gpu_fault_workflow_total Persisted recovery workflows by status.",
        "# TYPE gpu_fault_workflow_total gauge",
    ]
    for workflow_status in WorkflowStatus:
        lines.append(
            f'gpu_fault_workflow_total{{status="{workflow_status.value}"}} '
            f"{workflow_statuses.get(workflow_status.value, 0)}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_workflow_blocked_unreconciled BLOCKED recovery "
            "workflows with no verified restore successor, so the GPU node is "
            "still out of the training pool.",
            "# TYPE gpu_fault_workflow_blocked_unreconciled gauge",
            f"gpu_fault_workflow_blocked_unreconciled {blocked_unreconciled}",
            "# HELP gpu_fault_orphan_workflows PENDING or SAFETY_PENDING "
            "workflows past the aggregation window whose incident is gone or "
            "names another workflow and which no successor names as "
            "predecessor: nothing will dispatch, fence or sweep them (F-B3).",
            "# TYPE gpu_fault_orphan_workflows gauge",
            f"gpu_fault_orphan_workflows {orphan_workflows}",
            "# HELP gpu_fault_incident_dangling_workflow_pointers Incidents "
            "whose workflow_request_id names a workflow row that does not "
            "exist (F-B3).",
            "# TYPE gpu_fault_incident_dangling_workflow_pointers gauge",
            f"gpu_fault_incident_dangling_workflow_pointers {dangling_incident_pointers}",
            "# HELP gpu_fault_workflow_scan_limit Workflows the step, duration "
            "and milestone families are allowed to read per scrape.",
            "# TYPE gpu_fault_workflow_scan_limit gauge",
            f"gpu_fault_workflow_scan_limit {scan.limit}",
            "# HELP gpu_fault_workflow_scan_size Workflows actually read for "
            "the step, duration and milestone families.",
            "# TYPE gpu_fault_workflow_scan_size gauge",
            f"gpu_fault_workflow_scan_size {len(workflows)}",
            "# HELP gpu_fault_workflow_scan_truncated Set when the workflow "
            "table is larger than the scan budget, so the step, duration and "
            "milestone families describe only the newest slice.",
            "# TYPE gpu_fault_workflow_scan_truncated gauge",
            f"gpu_fault_workflow_scan_truncated {int(scan.truncated)}",
            "# HELP gpu_fault_workflow_step_total Persisted workflow step outcomes.",
            "# TYPE gpu_fault_workflow_step_total gauge",
        ]
    )
    for (operation, step_status), count in sorted(step_statuses.items()):
        lines.append(
            "gpu_fault_workflow_step_total"
            f'{{operation="{operation}",status="{step_status}"}} {count}'
        )
    lines.extend(
        workflow_stall_metric_lines(
            workflows, now, terminal, runtime.context.production_executor_config
        )
    )
    lines.extend(
        [
            "# HELP gpu_fault_workflow_duration_seconds Terminal workflow duration.",
            "# TYPE gpu_fault_workflow_duration_seconds summary",
        ]
    )
    for status_value, values in sorted(terminal_durations.items()):
        _append_summary(
            lines,
            "gpu_fault_workflow_duration_seconds",
            values,
            status=status_value,
        )
    lines.extend(
        [
            "# HELP gpu_fault_closed_loop_milestone_seconds Time from workflow creation to a successful closed-loop milestone.",
            "# TYPE gpu_fault_closed_loop_milestone_seconds summary",
        ]
    )
    for milestone, values in sorted(milestone_durations.items()):
        _append_summary(
            lines,
            "gpu_fault_closed_loop_milestone_seconds",
            values,
            milestone=milestone,
        )
    lines.extend(
        _remediation_budget_lines(
            active_by_scope_type=budget_scope_types,
            active_by_cluster=budget_cluster_active,
            waiting_by_scope_type=budget_waiting_scope_types,
            waiting_by_cluster=budget_cluster_waiting,
            wait_total=budget_wait_total,
            waiting=budget_waiting,
            config=runtime.context.production_executor_config,
        )
    )

    lines.extend(_notification_lines(store, cache))

    stale_agents = sum(
        1
        for agent in cache.agents()
        if getattr(agent.lifecycle_state, "value", agent.lifecycle_state) == "ACTIVE"
        and not agent_is_current(agent, observed_at=now)
    )
    lines.extend(
        [
            "# HELP gpu_fault_stale_agents Agents whose heartbeat lease is absent or expired.",
            "# TYPE gpu_fault_stale_agents gauge",
            f"gpu_fault_stale_agents {stale_agents}",
        ]
    )
    return lines


def _notification_lines(store: ControlPlaneStore, cache: MetricScanCache) -> list[str]:
    """Render the business notification families from the Store aggregate.

    Read from ``notification_status_counts`` rather than the workflow detail
    scan, so the outbox depth stays exact on a table larger than the scan
    budget. Both aggregates are whole-kind LEFT JOIN GROUP BYs over kinds with
    no retention (F-6), so they are shared across scrapes for the cache TTL.
    """

    statuses = {
        status.value: count
        for status, count in cache.shared(
            "notification_status_counts", store.notification_status_counts
        ).items()
    }
    lines = [
        "# HELP gpu_fault_notification_total Persisted business notifications by delivery result.",
        "# TYPE gpu_fault_notification_total gauge",
    ]
    for status_value in NotificationStatus:
        lines.append(
            f'gpu_fault_notification_total{{status="{status_value.value}"}} '
            f"{statuses[status_value.value]}"
        )
    pending = sum(
        statuses[status]
        for status in {
            NotificationStatus.QUEUED.value,
            NotificationStatus.FAILED.value,
        }
    )
    lines.extend(
        [
            "# HELP gpu_fault_notification_outbox_depth Notifications without a terminal delivery result.",
            "# TYPE gpu_fault_notification_outbox_depth gauge",
            f"gpu_fault_notification_outbox_depth {pending}",
        ]
    )
    # ARCH-E E1: the outbox state machine itself. ``gpu_fault_notification_total``
    # reads the result rows, which are terminal verdicts; a notification being
    # retried has none, so the queue the dispatcher is actually working was
    # invisible and its age unmeasured.
    delivery = cache.shared(
        "notification_delivery_stats", store.notification_delivery_stats
    )
    lines.extend(
        [
            "# HELP gpu_fault_notification_delivery_total Notification outbox rows by delivery state as the outbox treats them: a row whose result is already SENT counts as SENT, a SKIPPED verdict on an undelivered row counts as DEAD (ARCH-E E1).",
            "# TYPE gpu_fault_notification_delivery_total gauge",
        ]
    )
    for status in NotificationDeliveryStatus:
        lines.append(
            f'gpu_fault_notification_delivery_total{{status="{status.value}"}} '
            f"{delivery['by_status'].get(status.value, 0)}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_notification_oldest_pending_age_seconds How long the oldest undelivered notification (PENDING, RETRY or LEASED) has waited since it was queued or last re-queued; 0 when nothing is undelivered (ARCH-E E1).",
            "# TYPE gpu_fault_notification_oldest_pending_age_seconds gauge",
            "gpu_fault_notification_oldest_pending_age_seconds "
            f"{delivery['oldest_pending_age_seconds']:.6f}",
        ]
    )
    return lines


def _append_summary(
    lines: list[str],
    name: str,
    values: list[float],
    **labels: str,
) -> None:
    if not values:
        return
    rendered = ",".join(
        f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items())
    )
    suffix = f"{{{rendered}}}" if rendered else ""
    lines.extend(
        [
            f"{name}_count{suffix} {len(values)}",
            f"{name}_sum{suffix} {sum(values):.6f}",
            f"{name}_max{suffix} {max(values):.6f}",
        ]
    )
