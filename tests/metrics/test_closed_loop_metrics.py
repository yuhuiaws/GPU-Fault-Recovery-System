"""Event-time means must not turn expiring snapshots into latency observations."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.builtin_metric_contributors import closed_loop_metric_lines
from gpu_fault.app.closed_loop_metrics import (
    MILESTONES,
    WINDOW_SECONDS,
    closed_loop_window_metric_lines,
    milestone_windows,
)
from gpu_fault.app.metric_aggregation import Strategy, strategy_for
from gpu_fault.app.metric_scan_cache import MetricScanCache, WorkflowScan
from gpu_fault.app.process_metrics import aggregate, parse_lines
from gpu_fault.models import (
    BlockedKind,
    WorkflowEvent,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from tests._builders import build_store
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_support import finding, memory_store

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
MEAN = "gpu_fault_closed_loop_milestone_window_mean_seconds"
COUNT = "gpu_fault_closed_loop_milestone_window_count"
COMPLETE = "gpu_fault_closed_loop_milestone_window_complete"
END = "gpu_fault_closed_loop_window_end_timestamp_seconds"
FAMILIES = {MEAN, COUNT, COMPLETE, END}
CONTAINMENT = '{milestone="containment"}'


def workflow(
    name: str = "workflow-a",
    *,
    at: datetime = NOW - timedelta(minutes=1),
    seconds: float = 600,
    operation: WorkflowOperation = WorkflowOperation.MARK_UNSCHEDULABLE,
) -> WorkflowRequest:
    created_at = at - timedelta(seconds=seconds)
    return WorkflowRequest(
        request_id=name,
        incident_id=f"incident-{name}",
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=1,
        created_at=created_at,
        updated_at=at,
        completed_operations=[operation],
        completed_step_indexes=[0],
        step_executions=[
            WorkflowStepExecution(
                step_index=0,
                operation=operation,
                status=WorkflowStepStatus.SUCCEEDED,
                started_at=created_at,
                updated_at=at,
            )
        ],
    )


def scan(*rows: WorkflowRequest, at: datetime = NOW) -> WorkflowScan:
    return WorkflowScan(
        workflows=rows,
        limit=20,
        truncated=False,
        window_seconds=7 * 24 * 3600,
        observed_at=at,
    )


def success_event(row: WorkflowRequest) -> WorkflowEvent:
    return WorkflowEvent(
        kind=WorkflowEventKind.STEP_ATTEMPT,
        operation=row.step_executions[0].operation,
        step_index=0,
        status="SUCCEEDED",
        at=row.step_executions[0].updated_at,
    )


def fresh_health_dag() -> WorkflowRequest:
    service, _absorbed = health_service(memory_store())
    _incident, row = service.ingest(
        finding(
            diagnostic_parameters={
                "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
                "capture_process_state": True,
            }
        )
    )
    assert row is not None, "the real health planner must produce a workflow"
    assert row.dag_enabled and row.dag_revision == 1, (
        "this regression exercises the production initial DAG shape"
    )
    return row


def test_fresh_unexecuted_health_dag_does_not_poison_the_window() -> None:
    row = fresh_health_dag()
    assert not row.events and not row.step_executions, (
        "a new health DAG has no execution history to lose"
    )
    windows = milestone_windows(scan(row, at=row.created_at))

    assert all(
        item.complete and item.count == 0 and item.mean_seconds is None
        for item in windows.values()
    ), "the initial empty plan is known empty, not missing historical successes"


@pytest.mark.parametrize(
    "changes",
    [
        {"dag_enabled": False},
        {"dag_revision": 2},
        {"execution_epoch": 1},
        {"merge_revision": 1},
        {"status": WorkflowStatus.RUNNING},
        {"status": WorkflowStatus.SUCCEEDED},
        {"execution_owner_id": "claimed"},
        {"execution_lease_expires_at": NOW},
        {"execution_deadline": NOW},
        {"lifetime_deadline_at": NOW},
        {"pending_failure_step_index": 0},
        {"completed_operations": [WorkflowOperation.MARK_UNSCHEDULABLE]},
        {"completed_step_indexes": [0]},
        {"inherited_step_indexes": [0]},
        {"superseded_step_indexes": [0]},
        {"branch_escalation_counts": {"node": 1}},
        {"exhausted_branch_ids": ["branch"]},
    ],
)
def test_missing_history_after_claim_or_rewrite_is_still_unknown(changes: dict) -> None:
    row = fresh_health_dag().model_copy(update=changes)

    windows = milestone_windows(scan(row, at=row.updated_at))

    assert not windows["containment"].complete, (
        "progress or rewrite evidence cannot be relabeled as a never-executed plan"
    )


def test_compile_blocked_initial_dag_is_known_unexecuted() -> None:
    row = fresh_health_dag().model_copy(
        update={
            "status": WorkflowStatus.BLOCKED,
            "blocked_kind": BlockedKind.NEEDS_OPERATOR,
        }
    )
    assert all(
        item.complete
        for item in milestone_windows(scan(row, at=row.updated_at)).values()
    ), "a compile-time block does not invent lost execution history"


@pytest.mark.parametrize("source", ["execution", "audit"])
def test_known_post_scan_success_is_excluded_without_losing_coverage(
    source: str,
) -> None:
    row = workflow(at=NOW + timedelta(seconds=1), seconds=600)
    if source == "audit":
        row = row.model_copy(
            update={"events": [success_event(row)], "step_executions": []}
        )
    window = milestone_windows(scan(row))["containment"]

    assert window.complete and window.count == 0 and window.mean_seconds is None, (
        "a timestamp beyond this scan is known out-of-window, not missing"
    )
    later = milestone_windows(scan(row, at=row.updated_at))["containment"]
    assert later.complete and later.count == 1 and later.mean_seconds == 600, (
        "the next scan includes the success exactly once"
    )


def test_completion_during_store_read_preserves_the_pre_read_boundary(
    monkeypatch,
) -> None:
    store = build_store()
    clock = [NOW]
    row = workflow(at=NOW + timedelta(seconds=1), seconds=600)

    def read(_statuses, **_kwargs):
        clock[0] = row.updated_at
        return [row]

    monkeypatch.setattr(store, "list_recent_workflows", read)
    cache = MetricScanCache(store, ttl_seconds=0, now=lambda: clock[0])
    observed = cache.workflows()
    window = milestone_windows(observed)["containment"]

    assert observed.observed_at == NOW, "the scan boundary precedes the Store read"
    assert window.complete and window.count == 0, (
        "a concurrent completion must not turn a complete scan into UNKNOWN"
    )
    next_window = milestone_windows(cache.workflows())["containment"]
    assert next_window.complete and next_window.count == 1, (
        "the completion becomes visible at the next event-time boundary"
    )


def test_expiring_two_100s_rows_and_adding_two_600s_rows_reports_600s() -> None:
    store = build_store()
    clock = [NOW]
    cache = MetricScanCache(store, ttl_seconds=0, now=lambda: clock[0])
    runtime = SimpleNamespace(
        context=ApplicationContext(store=store), metric_scan_cache=cache
    )
    for index in range(2):
        store.save_workflow(
            workflow(
                f"old-{index}",
                at=NOW - timedelta(days=7) + timedelta(seconds=30),
                seconds=100,
            )
        )
    before = closed_loop_metric_lines(runtime)
    clock[0] += timedelta(minutes=1)
    for index in range(2):
        store.save_workflow(workflow(f"new-{index}", at=clock[0], seconds=600))
    after = closed_loop_metric_lines(runtime)
    legacy = "gpu_fault_closed_loop_milestone_seconds"

    assert f"{legacy}_sum{CONTAINMENT} 200.000000" in before
    assert f"{legacy}_sum{CONTAINMENT} 1200.000000" in after
    assert f"{legacy}_count{CONTAINMENT} 2" in before
    assert f"{legacy}_count{CONTAINMENT} 2" in after
    assert f"{MEAN}{CONTAINMENT} 600.000000" in after
    assert f"{COUNT}{CONTAINMENT} 2" in after
    assert f"{COMPLETE}{CONTAINMENT} 1" in after
    assert closed_loop_metric_lines(runtime) == after


def test_window_uses_completion_time_not_a_later_workflow_update() -> None:
    old = workflow(at=NOW - timedelta(hours=7))
    touched = old.model_copy(update={"updated_at": NOW})

    window = milestone_windows(scan(touched))["containment"]

    assert window.count == 0
    assert window.mean_seconds is None
    assert window.complete is True


def test_old_workflow_created_before_window_can_complete_inside_it() -> None:
    row = workflow(seconds=8 * 3600)

    window = milestone_windows(scan(row))["containment"]

    assert window.count == 1
    assert window.mean_seconds == 8 * 3600
    assert window.complete is True


def test_window_is_open_at_lower_edge_and_closed_at_scan_time() -> None:
    rows = (
        workflow("outside", at=NOW - timedelta(seconds=WINDOW_SECONDS)),
        workflow("inside", at=NOW - timedelta(seconds=WINDOW_SECONDS - 1), seconds=100),
        workflow("upper", at=NOW, seconds=600),
        workflow("future", at=NOW + timedelta(minutes=2), seconds=1),
    )

    window = milestone_windows(scan(*rows))["containment"]

    assert window.count == 2
    assert window.mean_seconds == 350
    assert window.complete is True


def test_milestone_is_first_success_not_step_order_or_repeated_observation() -> None:
    row = workflow()
    later = row.step_executions[0].model_copy(
        update={
            "step_index": 1,
            "operation": WorkflowOperation.QUARANTINE,
            "updated_at": NOW,
        }
    )
    row = row.model_copy(
        update={
            "step_executions": [later, row.step_executions[0]],
            "events": [success_event(row), success_event(row)],
            "updated_at": NOW,
        }
    )
    reversed_row = row.model_copy(
        update={"step_executions": list(reversed(row.step_executions))}
    )

    first = milestone_windows(scan(row, row))["containment"]
    second = milestone_windows(scan(reversed_row))["containment"]

    assert first == second
    assert first.count == 1
    assert first.mean_seconds == 600
    assert first.complete is True


def test_inherited_containment_does_not_create_a_new_success() -> None:
    original = workflow("original")
    successor = workflow("successor", seconds=10)
    inherited = successor.step_executions[0].model_copy(
        update={"details": {"inherited_from_workflow_id": original.request_id}}
    )
    successor = successor.model_copy(update={"step_executions": [inherited]})

    window = milestone_windows(scan(original, successor))["containment"]

    assert window.count == 1
    assert window.mean_seconds == 600
    assert window.complete is True


def test_audit_success_survives_dag_rewrite_without_a_current_execution() -> None:
    row = workflow()
    rewritten = row.model_copy(
        update={
            "events": [success_event(row)],
            "step_executions": [],
            "completed_operations": [],
            "completed_step_indexes": [],
            "dag_revision": 1,
            "updated_at": NOW,
        }
    )

    window = milestone_windows(scan(rewritten))["containment"]

    assert window.count == 1
    assert window.mean_seconds == 600
    assert window.complete is True


@pytest.mark.parametrize("status", list(WorkflowStatus))
def test_success_counts_even_when_the_workflow_is_not_terminal(
    status: WorkflowStatus,
) -> None:
    row = workflow().model_copy(update={"status": status})

    window = milestone_windows(scan(row))["containment"]

    assert window.count == 1
    assert window.mean_seconds == 600
    assert window.complete is True


@pytest.mark.parametrize(
    "changes",
    [
        {"truncated": True},
        {"window_seconds": WINDOW_SECONDS - 1},
        {"observed_at": None},
        {"observed_at": NOW.replace(tzinfo=None)},
    ],
)
def test_incomplete_scan_never_publishes_partial_values(changes: dict) -> None:
    lines = closed_loop_window_metric_lines(replace(scan(workflow()), **changes))

    assert f"{MEAN}{CONTAINMENT} NaN" in lines
    assert f"{COUNT}{CONTAINMENT} NaN" in lines
    assert f"{COMPLETE}{CONTAINMENT} 0" in lines


@pytest.mark.parametrize("window_seconds", [0, WINDOW_SECONDS, 7 * 86400])
def test_sufficient_scan_windows_are_complete(window_seconds: int) -> None:
    result = milestone_windows(
        replace(scan(workflow()), window_seconds=window_seconds)
    )["containment"]

    assert result.complete is True
    assert result.mean_seconds == 600


@pytest.mark.parametrize(
    ("retention", "complete"),
    [
        (None, True),
        (timedelta(days=30), True),
        (timedelta(days=1), True),
        (timedelta(seconds=WINDOW_SECONDS), True),
        (timedelta(seconds=WINDOW_SECONDS - 1), False),
        (timedelta(0), False),
    ],
)
def test_retention_must_cover_the_window(
    retention: timedelta | None, complete: bool
) -> None:
    result = milestone_windows(scan(workflow()), retention=retention)["containment"]

    assert result.complete is complete


def test_contributor_uses_the_actual_archiver_retention() -> None:
    store = build_store()
    store.save_workflow(workflow())
    context = ApplicationContext(store=store)
    context.control_record_archiver = SimpleNamespace(retention=timedelta(hours=1))
    runtime = SimpleNamespace(
        context=context, metric_scan_cache=MetricScanCache(store, now=lambda: NOW)
    )

    lines = closed_loop_metric_lines(runtime)

    assert f"{COMPLETE}{CONTAINMENT} 0" in lines
    assert f"{COUNT}{CONTAINMENT} NaN" in lines


def test_empty_window_is_distinct_from_missing_coverage() -> None:
    lines = closed_loop_window_metric_lines(scan())

    for milestone in MILESTONES:
        label = f'{{milestone="{milestone}"}}'
        assert f"{MEAN}{label} NaN" in lines
        assert f"{COUNT}{label} 0" in lines
        assert f"{COMPLETE}{label} 1" in lines


def test_truncated_history_cannot_report_a_partial_first_success() -> None:
    row = workflow()
    row = row.model_copy(
        update={
            "events": [WorkflowEvent(kind=WorkflowEventKind.HISTORY_TRUNCATED, at=NOW)]
        }
    )

    lines = closed_loop_window_metric_lines(scan(row))

    assert f"{COMPLETE}{CONTAINMENT} 0" in lines
    assert f"{MEAN}{CONTAINMENT} NaN" in lines
    assert f"{COUNT}{CONTAINMENT} NaN" in lines


def test_a_proven_old_first_success_stays_outside_even_with_truncated_history() -> None:
    row = workflow(at=NOW - timedelta(hours=7)).model_copy(
        update={
            "updated_at": NOW,
            "events": [WorkflowEvent(kind=WorkflowEventKind.HISTORY_TRUNCATED, at=NOW)],
        }
    )

    window = milestone_windows(scan(row))["containment"]

    assert window.count == 0
    assert window.complete is True


@pytest.mark.parametrize(
    "changes",
    [
        {"step_executions": []},
        {"dag_revision": 1},
        {"created_at": NOW.replace(tzinfo=None)},
        {"updated_at": NOW - timedelta(days=1)},
    ],
)
def test_missing_or_inconsistent_completion_evidence_is_unknown(changes: dict) -> None:
    row = workflow().model_copy(update=changes)

    window = milestone_windows(scan(row))["containment"]

    assert window.complete is False


def test_conflicting_workflow_identity_is_not_deduplicated_as_healthy() -> None:
    row = workflow()
    other = row.model_copy(update={"created_at": row.created_at - timedelta(seconds=1)})

    window = milestone_windows(scan(row, other))["containment"]

    assert window.complete is False


def test_each_milestone_has_an_independent_first_success() -> None:
    rows = [
        workflow(f"workflow-{index}", operation=operation, seconds=100 * index)
        for index, operation in enumerate(
            (
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.RESTORE_SCHEDULING,
                WorkflowOperation.RESTART_WORKLOAD,
            ),
            start=1,
        )
    ]

    windows = milestone_windows(scan(*rows))

    assert {name: item.count for name, item in windows.items()} == {
        name: 1 for name in MILESTONES
    }
    assert windows["containment"].mean_seconds == 100
    assert windows["validation"].mean_seconds == 200
    assert windows["readmission"].mean_seconds == 300
    assert windows["workload_restart"].mean_seconds == 400


def test_pod_aggregation_keeps_mean_count_coverage_and_time_from_one_render() -> None:
    local = closed_loop_window_metric_lines(scan(workflow()))
    other = closed_loop_window_metric_lines(
        replace(scan(workflow("other", seconds=1200)), truncated=True)
    )

    merged = aggregate(parse_lines(local), [parse_lines(other)])

    assert parse_lines(merged).samples == parse_lines(local).samples
    assert set(parse_lines(merged).families) == FAMILIES
    assert all(strategy_for(name) is Strategy.ANY for name in FAMILIES), (
        "window mean, count, completeness and boundary must come from one Pod render"
    )
