from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    orchestration_metric_lines,
)
from gpu_fault.app.collector_metrics import CollectorMetricsSnapshot
from gpu_fault.app.metric_contributors import MetricContributorRegistry
from gpu_fault.app.metrics import collector_silence_lines
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
        lambda: {
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
    assert "gpu_fault_remediation_budget_wait_total 2" in lines
    assert 'gpu_fault_notification_total{status="SENT"} 1' in lines


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
