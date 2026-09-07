from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    orchestration_metric_lines,
    remote_command_metric_lines,
)
from gpu_fault.app.collector_metrics import CollectorMetricsSnapshot
from gpu_fault.app.metric_contributors import MetricContributorRegistry
from gpu_fault.app.metrics import collector_silence_lines
from gpu_fault.app.metrics_sections import render_capacity_metrics
from gpu_fault.models import (
    AdvisoryNotification,
    IncidentState,
    NotificationResult,
    NotificationStatus,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    asgi_client,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


def test_metric_contributors_render_in_registration_order(monkeypatch) -> None:
    monkeypatch.setattr(
        "gpu_fault.app.metric_contributors.discover_plugins", lambda _group: {}
    )
    registry = MetricContributorRegistry()
    registry.register("first", lambda _runtime: ["first 1"])
    registry.register("second", lambda _runtime: ["second 2"])
    assert registry.names == ("first", "second")
    assert registry.render(object()) == ["first 1", "second 2"]


def test_duplicate_metric_contributor_fails_closed() -> None:
    registry = MetricContributorRegistry()
    registry.register("same", lambda _runtime: [])
    with pytest.raises(RuntimeError, match="duplicate metric"):
        registry.register("same", lambda _runtime: [])


def test_collector_metrics_use_background_snapshot_only() -> None:
    snapshot = SimpleNamespace(lines=lambda: ["snapshot 1"])
    runtime = SimpleNamespace(collector_metrics_snapshot=snapshot)
    assert collector_silence_lines(runtime) == ["snapshot 1"]


def test_process_local_store_rejection_counter_has_process_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gpu_fault.app.metrics_sections.os.getpid", lambda: 4321)
    app = create_app(ApplicationContext(store=build_store()))

    async def fetch() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    metrics = asyncio.run(fetch())

    assert 'gpu_fault_store_io_rejections_total{process_id="4321"} 0' in metrics, (
        metrics
    )


def test_declared_node_counts_are_exported_as_capacity_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The renderer ships the declared topology to every role; a scrape can
    # then compare the fault reserve with the wave it has to hold. Labels are
    # deliberately absent: these are site-wide facts, never per node.
    monkeypatch.setenv("GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT", "1000")
    monkeypatch.setenv("GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT", "4000")
    lines: list[str] = []

    render_capacity_metrics(lines)

    assert "gpu_fault_capacity_largest_cluster_node_count 1000" in lines
    assert "gpu_fault_capacity_managed_node_count 4000" in lines
    assert "# TYPE gpu_fault_capacity_largest_cluster_node_count gauge" in lines
    assert "# TYPE gpu_fault_capacity_managed_node_count gauge" in lines

    # Wired into /metrics: an undeclared topology reads as an explicit 0.
    monkeypatch.delenv("GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT")
    monkeypatch.delenv("GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT")
    app = create_app(ApplicationContext(store=build_store()))

    async def fetch() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    metrics = asyncio.run(fetch())

    assert "gpu_fault_capacity_largest_cluster_node_count 0" in metrics, metrics
    assert "gpu_fault_capacity_managed_node_count 0" in metrics, metrics


def test_remote_internal_error_metrics_do_not_treat_retention_as_a_counter() -> None:
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            regional_mode=True,
            store=SimpleNamespace(
                remote_command_stats=lambda: {
                    "by_status": {"FAILED": 1},
                    "oldest_unclaimed_age_seconds_by_cluster": {},
                    "executor_internal_error_total": 1,
                    "executor_internal_error_last_seen_timestamp_seconds": 123.5,
                    "unclaimed_expired_total": 0,
                }
            ),
        )
    )

    lines = remote_command_metric_lines(runtime)

    assert (
        "# TYPE gpu_fault_remote_command_executor_internal_errors_total gauge" in lines
    )
    assert (
        "gpu_fault_remote_command_executor_internal_error_last_seen_"
        "timestamp_seconds 123.500000" in lines
    )


def test_ambiguous_attempt_metric_is_exported(monkeypatch) -> None:
    context = ApplicationContext(store=build_store())
    monkeypatch.setattr(
        context.orchestrator._evidence_operations,
        "ambiguous_attempt_ownership_total",
        lambda: 3,
    )
    monkeypatch.setattr(
        context.orchestrator._evidence_operations,
        "ownership_metric_snapshot",
        lambda **_: {
            "current": {("cluster-a", "node-a"): 2},
            "stale": {("cluster-a", "node-a"): 1},
        },
    )

    lines = orchestration_metric_lines(SimpleNamespace(context=context))

    assert "gpu_fault_ambiguous_attempt_ownership_total 3" in lines
    assert (
        "gpu_fault_ambiguous_attempt_ownership_current"
        '{cluster_id="cluster-a",gpu_node="node-a"} 2'
    ) in lines
    assert (
        "gpu_fault_stale_attempt_observations"
        '{cluster_id="cluster-a",gpu_node="node-a"} 1'
    ) in lines


def test_closed_loop_metrics_cover_outcomes_budgets_and_notifications() -> None:
    store = build_store()
    incident = fault_incident(
        "incident-metrics",
        "event-metrics",
        state=IncidentState.RECOVERED,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=30),
    )
    workflow = workflow_request(
        "workflow-metrics",
        incident.incident_id,
        WorkflowStatus.SUCCEEDED,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING),
        ],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowStepStatus.SUCCEEDED,
                updated_at=NOW + timedelta(seconds=5),
            ),
            workflow_step_execution(
                1,
                WorkflowOperation.RESTORE_SCHEDULING,
                WorkflowStepStatus.SUCCEEDED,
                updated_at=NOW + timedelta(seconds=20),
            ),
        ],
        remediation_budget_claims=["region", "cluster:cluster-a"],
        remediation_budget_wait_count=2,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=30),
    )
    incident = incident.model_copy(update={"workflow_request_id": workflow.request_id})
    store.save_incident_and_workflow(incident, workflow)
    # One RUNNING workflow with a live lease holds cluster-a's budget, and one
    # PENDING workflow was last refused by that same cluster scope: the
    # per-cluster families have to name cluster-a, and only cluster-a, for both.
    holder_incident = fault_incident(
        "incident-holder", "event-holder", state=IncidentState.ACTION_PENDING
    )
    holder = workflow_request(
        "workflow-holder",
        holder_incident.incident_id,
        WorkflowStatus.RUNNING,
        execution_owner_id="executor-a",
        execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        remediation_budget_claims=[
            "class:cluster-a:NODE_LIFECYCLE_MUTATION",
            "cluster:cluster-a",
            "node:cluster-a:node-a",
            "region",
        ],
    )
    store.save_incident_and_workflow(
        holder_incident.model_copy(update={"workflow_request_id": holder.request_id}),
        holder,
    )
    waiter_incident = fault_incident(
        "incident-waiter", "event-waiter", state=IncidentState.ACTION_PENDING
    )
    waiter = workflow_request(
        "workflow-waiter",
        waiter_incident.incident_id,
        WorkflowStatus.PENDING,
        remediation_budget_wait_count=3,
        remediation_budget_last_blocked_reason=(
            "remediation concurrency budget is full: "
            "scope=cluster:cluster-a active=5 limit=5"
        ),
        remediation_budget_last_blocked_scope="cluster:cluster-a",
    )
    store.save_incident_and_workflow(
        waiter_incident.model_copy(update={"workflow_request_id": waiter.request_id}),
        waiter,
    )
    notification = store.save_notification_if_absent(
        AdvisoryNotification(
            notification_id="notification-metrics",
            deduplication_key="notification-metrics",
            cluster_name="cluster-a",
            incident_id=incident.incident_id,
            subject="subject",
            body_text="body",
            support_case_draft="body",
            created_at=NOW,
        )
    )
    store.save_notification_result(
        NotificationResult(
            notification_id=notification.notification_id, status=NotificationStatus.SENT
        )
    )

    lines = closed_loop_metric_lines(
        SimpleNamespace(context=ApplicationContext(store=store))
    )

    assert 'gpu_fault_workflow_total{status="SUCCEEDED"} 1' in lines
    assert (
        'gpu_fault_closed_loop_milestone_seconds_count{milestone="containment"} 1'
    ) in lines
    assert (
        'gpu_fault_closed_loop_milestone_seconds_count{milestone="readmission"} 1'
    ) in lines
    assert "gpu_fault_remediation_budget_wait_total 5" in lines
    assert "gpu_fault_remediation_budget_waiting_workflows 1" in lines
    assert 'gpu_fault_remediation_budget_active_claims{scope_type="cluster"} 1' in lines
    # S1: the saturation has to be visible per cluster. The SUCCEEDED workflow
    # also names cluster:cluster-a in its claims and must not count.
    assert (
        'gpu_fault_remediation_budget_cluster_active_claims{cluster_id="cluster-a"} 1'
    ) in lines
    assert "gpu_fault_remediation_budget_cluster_limit 5" in lines
    assert (
        "gpu_fault_remediation_budget_cluster_waiting_workflows"
        '{cluster_id="cluster-a"} 1'
    ) in lines
    assert (
        'gpu_fault_remediation_budget_waiting_workflows_by_scope{scope_type="cluster"} 1'
    ) in lines
    assert not [
        line
        for line in lines
        if "remediation_budget_cluster" in line and "node" in line
    ], "per-cluster budget families must never carry a node label"
    assert 'gpu_fault_notification_total{status="SENT"} 1' in lines


def test_remediation_budget_cluster_limit_is_omitted_without_executor_config() -> None:
    """The limit gauge reports the executor policy; with no policy there is no
    number to report, and a fabricated zero would make the saturation alert's
    ``limit > 0`` guard read as "never saturated"."""

    store = build_store()
    context = ApplicationContext(store=store)
    context.production_executor_config = None

    lines = closed_loop_metric_lines(SimpleNamespace(context=context))

    assert not [
        line
        for line in lines
        if line.startswith("gpu_fault_remediation_budget_cluster_limit ")
    ]
    assert "gpu_fault_remediation_budget_waiting_workflows 0" in lines


def test_closed_loop_metrics_use_aggregate_notification_counts(monkeypatch) -> None:
    store = build_store()
    counts = {status: 0 for status in NotificationStatus}
    counts[NotificationStatus.SKIPPED] = 30_000
    monkeypatch.setattr(store, "notification_status_counts", lambda: counts)
    monkeypatch.setattr(
        store,
        "list_notifications",
        lambda: pytest.fail("metrics must not scan notifications individually"),
    )

    lines = closed_loop_metric_lines(
        SimpleNamespace(context=ApplicationContext(store=store))
    )

    assert 'gpu_fault_notification_total{status="SKIPPED"} 30000' in lines
    assert "gpu_fault_notification_outbox_depth 0" in lines


def test_blocked_backlog_gauge_is_a_server_side_aggregate(monkeypatch) -> None:
    """The BLOCKED backlog alert reads this, so the scan bound must not truncate it.

    The workflow detail scan is deliberately bounded, and the gauge the alert
    thresholds has to stay exact regardless: a backlog that fell outside the
    newest slice would read as zero and the alert would go silent while nodes
    were still held. ``gpu_fault_workflow_total`` cannot serve instead -- it
    counts the whole history and BLOCKED is terminal, so its bucket stays up
    after the node returns to the training pool.
    """

    store = build_store()
    store.save_incident(
        fault_incident("incident-held", "event-held", state=IncidentState.QUARANTINED)
    )
    store.save_workflow(
        workflow_request(
            "workflow-held", "incident-held", status=WorkflowStatus.BLOCKED
        )
    )
    monkeypatch.setattr(store, "list_workflows", lambda *_args, **_keywords: [])

    lines = closed_loop_metric_lines(
        SimpleNamespace(context=ApplicationContext(store=store))
    )

    assert "gpu_fault_workflow_blocked_unreconciled 1" in lines
    assert "gpu_fault_workflow_scan_size 0" in lines, (
        "the detail scan has to be empty for this to prove anything"
    )


def test_collector_metrics_top_n_is_bounded() -> None:
    snapshot = object.__new__(CollectorMetricsSnapshot)
    snapshot.top_n = 2
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
        for index in range(10)
    ]
    lines = snapshot._aggregate_lines(rows)
    assert (
        sum(line.startswith("gpu_fault_collector_silent_top_node") for line in lines)
        == 2
    )
    assert any(
        line.endswith(" 10")
        for line in lines
        if line.startswith("gpu_fault_collector_silent_nodes")
    )
    assert [row["node_id"] for row in snapshot._top_rows(rows)] == ["node-9", "node-8"]


def test_collector_metrics_snapshot_uses_shared_lease_and_record() -> None:
    context = ApplicationContext(store=build_store())
    owner = CollectorMetricsSnapshot(context, owner_id="owner-a", enabled=True)
    follower = CollectorMetricsSnapshot(context, owner_id="owner-b", enabled=True)

    owner.refresh()
    follower.refresh()

    assert owner.lines()
    assert follower.lines() == owner.lines()


def test_collector_metrics_snapshot_persists_bounded_details() -> None:
    context = ApplicationContext(store=build_store())
    snapshot = CollectorMetricsSnapshot(
        context, owner_id="owner-a", enabled=True, now=lambda: NOW
    )
    snapshot.top_n = 2
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
        for index in range(10)
    ]
    snapshot._rows = lambda _observed_at: rows

    snapshot.refresh()

    record = context.store.get_collector_metrics_snapshot()
    assert record is not None
    assert [row["node_id"] for row in record.details] == ["node-9", "node-8"]


def test_collector_metrics_snapshot_age_advances_without_refresh() -> None:
    context = ApplicationContext(store=build_store())
    clock = [NOW]
    snapshot = CollectorMetricsSnapshot(
        context, owner_id="owner-a", enabled=True, now=lambda: clock[0]
    )
    snapshot.refresh()
    clock[0] += timedelta(seconds=125)

    assert "gpu_fault_collector_metrics_snapshot_age_seconds 125" in snapshot.lines()


def test_disabled_role_does_not_export_collector_snapshot_metrics() -> None:
    snapshot = CollectorMetricsSnapshot(
        ApplicationContext(store=build_store()),
        owner_id="worker-a",
        enabled=False,
        now=lambda: NOW,
    )

    assert snapshot.lines() == []


def test_the_waiting_threshold_is_published_next_to_the_age() -> None:
    """The alert subtracts one series from the other, so both must exist per step.

    The threshold is per-operation -- a delegated replacement waits on the
    provider, everything else on the default cap -- so an alert carrying a single
    literal could only be right for one of them, and at the default cap it would
    fire on every healthy node replacement. Publishing the applicable threshold
    with the same labels is what lets the rule compare rather than hard-code, and
    what keeps it correct when either window is retuned.
    """

    store = build_store()
    now = datetime.now(timezone.utc)
    incident = fault_incident(
        "incident-waiting",
        "event-waiting",
        state=IncidentState.ACTION_PENDING,
        created_at=now - timedelta(seconds=900),
        updated_at=now,
    )
    operations = (
        WorkflowOperation.REPLACE_NODE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
    )
    workflow = workflow_request(
        "workflow-waiting",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        official_steps=[workflow_step(operation) for operation in operations],
        step_executions=[
            workflow_step_execution(
                index,
                operation,
                WorkflowStepStatus.WAITING,
                started_at=now - timedelta(seconds=900),
                updated_at=now,
            )
            for index, operation in enumerate(operations)
        ],
        created_at=now - timedelta(seconds=900),
        updated_at=now,
    )
    store.save_incident_and_workflow(
        incident.model_copy(update={"workflow_request_id": workflow.request_id}),
        workflow,
    )

    lines = closed_loop_metric_lines(
        SimpleNamespace(context=ApplicationContext(store=store))
    )

    thresholds = dict(
        line.removeprefix("gpu_fault_workflow_step_waiting_warning_seconds").split(" ")
        for line in lines
        if line.startswith("gpu_fault_workflow_step_waiting_warning_seconds{")
    )
    assert thresholds == {
        '{operation="QUIESCE_GPU_SERVICES"}': "300",
        '{operation="REPLACE_NODE"}': "1500",
    }
    ages = {
        line.partition(" ")[0].removeprefix("gpu_fault_workflow_step_waiting_seconds")
        for line in lines
        if line.startswith("gpu_fault_workflow_step_waiting_seconds{")
    }
    assert ages == set(thresholds), (
        "the two families must carry identical label sets or the rule's vector "
        "match silently drops the step it was meant to alert on"
    )


def test_waiting_step_age_and_unenforced_deadline_are_exported() -> None:
    """The two numbers a stalled step used to have nowhere to appear.

    Step metrics were counts by status, so a step re-asking the same question
    forever looked exactly like a step that had just started waiting; and the
    workflow's own deadline had no metric at all, which is why a workflow ten
    minutes past it could keep running with every dashboard green. Both are
    scoped to non-terminal workflows: a WAITING execution left behind on a
    workflow that has already ended is history, not a stall.
    """

    store = build_store()
    now = datetime.now(timezone.utc)
    for suffix, status, waiting_seconds, overdue in (
        ("stalled", WorkflowStatus.RUNNING, 1800, 600),
        ("done", WorkflowStatus.SUCCEEDED, 90000, 90000),
    ):
        incident = fault_incident(
            f"incident-{suffix}",
            f"event-{suffix}",
            state=IncidentState.ACTION_PENDING,
            created_at=now - timedelta(seconds=waiting_seconds),
            updated_at=now,
        )
        workflow = workflow_request(
            f"workflow-{suffix}",
            incident.incident_id,
            status,
            official_steps=[workflow_step(WorkflowOperation.REPLACE_NODE)],
            step_executions=[
                workflow_step_execution(
                    0,
                    WorkflowOperation.REPLACE_NODE,
                    WorkflowStepStatus.WAITING,
                    started_at=now - timedelta(seconds=waiting_seconds),
                    updated_at=now,
                )
            ],
            execution_deadline=now - timedelta(seconds=overdue),
            created_at=now - timedelta(seconds=waiting_seconds),
            updated_at=now,
        )
        store.save_incident_and_workflow(
            incident.model_copy(update={"workflow_request_id": workflow.request_id}),
            workflow,
        )

    lines = closed_loop_metric_lines(
        SimpleNamespace(context=ApplicationContext(store=store))
    )

    waiting = [
        line
        for line in lines
        if line.startswith("gpu_fault_workflow_step_waiting_seconds{")
    ]
    assert len(waiting) == 1
    series, _, age = waiting[0].partition(" ")
    assert series == (
        'gpu_fault_workflow_step_waiting_seconds{operation="REPLACE_NODE"}'
    )
    assert 1800 <= float(age) < 90000
    overdue_lines = [
        line for line in lines if line.startswith("gpu_fault_workflow_overdue_seconds ")
    ]
    assert len(overdue_lines) == 1
    assert 600 <= float(overdue_lines[0].split()[1]) < 90000
