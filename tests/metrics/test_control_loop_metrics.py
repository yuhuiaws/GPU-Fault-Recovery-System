"""The counters the review added are visible on /metrics, not only in
process memory.

FINAL F-L1 (2026-09-06). The dispatcher, executor, merge service, processor
runtime, store I/O executor and periodic runner all grew counters during the
implementation rounds; each lived only in a snapshot dict or an attribute.
An alert cannot be written against an attribute.
"""

from __future__ import annotations

import asyncio

from gpu_fault.app import ApplicationContext, create_app
from tests._builders import asgi_client


def _scrape(app) -> str:
    async def scenario():
        async with asgi_client(app) as client:
            return (await client.get("/metrics")).text

    return asyncio.run(scenario())


def test_review_counters_reach_the_metrics_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-metrics")
    token = "processor-metrics-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    context.dispatcher.node_busy_timeouts_total = 2
    context.dispatcher.failure_handling_abandoned_total = 1
    context.dispatcher.plan_sync_misses_total = 3
    context.dispatcher.internal_errors_total = 4
    context.workflow_executor.lifetime_exceeded_total = 5
    merger = context.orchestrator._workflow_merger
    merger.absorbed_record_only_total = 6
    merger.lifetime_record_only_total = 7
    merger.withdrawn_record_only_total = 8
    merger.unsettled_host_resource_record_only_total = 14
    processor = app.state.processor
    processor._completion_failure_releases_total = 9
    processor._retry_horizon_failures_total = 10
    processor._renewal_errors_total = 11
    processor._renewal_fenced_total = 12
    app.state.store_io.rejected_by_reason["deadline"] = 13

    text = _scrape(app)

    for line in (
        "gpu_fault_workflow_dispatch_node_busy_timeouts_total 2",
        "gpu_fault_workflow_dispatch_failure_handling_abandoned_total 1",
        "gpu_fault_workflow_dispatch_plan_sync_misses_total 3",
        "gpu_fault_workflow_dispatch_internal_errors_total 4",
        "gpu_fault_workflow_lifetime_exceeded_total 5",
        'gpu_fault_workflow_merge_record_only_total{reason="covered_read_only"} 6',
        'gpu_fault_workflow_merge_record_only_total{reason="lifetime_exceeded"} 7',
        'gpu_fault_workflow_merge_record_only_total{reason="workload_withdrawn"} 8',
        'gpu_fault_workflow_merge_record_only_total{reason="unsettled_host_resource_incident"} 14',
        "gpu_fault_processor_completion_failure_releases_total 9",
        "gpu_fault_processor_retry_horizon_failures_total 10",
        "gpu_fault_processor_renewal_errors_total 11",
        "gpu_fault_processor_renewal_fenced_total 12",
        'gpu_fault_store_io_rejections_total{reason="deadline"} 13',
    ):
        assert line in text, line


def test_the_second_batch_of_review_counters_reaches_the_metrics_endpoint(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-metrics-2")
    token = "processor-metrics-token-" + "y" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    context.dispatcher.deferred_total = 21
    context.workflow_executor.branch_escalation_budget_refusals_total = 22
    context.store.health_signal_clock_regressions_total = 23
    context.store.stale_event_link_repairs = 24

    class _Archiver:
        withheld_total = {"incident has external successor": 25}

    context.control_record_archiver = _Archiver()
    processor = app.state.processor
    processor._fault_rows_skipped_by_observation_total = 26
    processor._fault_rows_blocked_by_observation = 27
    processor._interlock_probes_total = 28
    processor._notification_shardless_episodes_total = 29

    text = _scrape(app)

    for line in (
        "gpu_fault_workflow_dispatch_deferred_total 21",
        "gpu_fault_workflow_branch_escalation_budget_refusals_total 22",
        "gpu_fault_health_signal_clock_regressions_total 23",
        "gpu_fault_ingest_stale_event_link_repairs_total 24",
        'gpu_fault_control_record_archive_withheld_total{reason="incident has external successor"} 25',
        "gpu_fault_processor_fault_rows_skipped_by_observation_total 26",
        "gpu_fault_processor_fault_rows_blocked_by_observation 27",
        "gpu_fault_processor_interlock_probes_total 28",
        "gpu_fault_processor_notification_shardless_episodes_total 29",
        "gpu_fault_processor_notification_listener_connected 0",
    ):
        assert line in text, line


def test_escalation_chain_counters_reach_the_metrics_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-metrics")
    token = "processor-metrics-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    escalation = context.orchestrator._escalation
    escalation.escalation_chain_terminated_total = 3
    escalation.containment_refused_escalations_total = 4

    text = _scrape(app)

    for line in (
        "gpu_fault_hardware_escalation_chain_terminated_total 3",
        "gpu_fault_hardware_escalation_containment_refused_total 4",
    ):
        assert line in text, line


def test_incident_closure_counters_reach_the_metrics_endpoint(monkeypatch) -> None:
    """DESTR-018 product gap: the two ways an ESCALATED incident is closed --
    by the restore that freed its node, by an operator -- are each counted,
    and an alert can be written against the counter, not the attribute."""

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-metrics-closure")
    token = "processor-metrics-token-" + "z" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    context.incident_closure.operator_closed_total = 31
    context.incident_closure.auto_closed_by_restore_total = 32

    text = _scrape(app)

    for line in (
        "# TYPE gpu_fault_incident_operator_closed_total counter",
        "gpu_fault_incident_operator_closed_total 31",
        "# TYPE gpu_fault_incident_auto_closed_by_restore_total counter",
        "gpu_fault_incident_auto_closed_by_restore_total 32",
    ):
        assert line in text, line
