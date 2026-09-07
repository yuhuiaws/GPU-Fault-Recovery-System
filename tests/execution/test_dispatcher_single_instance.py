"""The workflow dispatcher: one scanner per fleet, a horizon that cannot fill
up, a tick that a single bad row cannot abort, and bounded failure handling.

FINAL-建议汇总 batch 0-2: F-A1 (P0-79A), F-A2 (P0-79B), F-A3 (P1-79F /
P0-40A / P1-71C), F-A6 (P1-80B / P1-71E), F-A7 (P1-71D).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryAction,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state

NOW = datetime(2026, 9, 5, 19, 0, tzinfo=timezone.utc)
OP = WorkflowOperation.FREEZE_EVIDENCE


class _CountingStore:
    """Counts full-table workflow scans without changing the store's answers."""

    def __init__(self, store) -> None:
        self._store = store
        self.scans = 0

    def list_workflows(self, *args, **kwargs):
        self.scans += 1
        return self._store.list_workflows(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._store, name)


def _dispatcher(store, *, executor_id: str, **config):
    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], [OP], executor_id=executor_id),
        WorkflowDispatcherConfig(enabled=True, batch_size=100, max_workers=1, **config),
    )


def _mirror_plan(
    dispatcher: WorkflowDispatcher, workflow, status: WorkflowStatus
) -> None:
    dispatcher._sync_plan(workflow, status)


def _pending(store, request_id: str, incident_id: str, **values):
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        request_id,
        incident_id,
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        official_steps=[workflow_step(OP)],
        created_at=values.pop("created_at", NOW),
        updated_at=values.pop("updated_at", NOW),
        **values,
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


# ---------------------------------------------------------------- F-A1


def test_only_one_of_two_dispatchers_scans_when_the_lease_is_enabled() -> None:
    store = _CountingStore(build_store())
    _pending(store, "wf-1", "inc-1")
    first = _dispatcher(store, executor_id="executor-a", dispatch_lease_seconds=30)
    second = _dispatcher(store, executor_id="executor-b", dispatch_lease_seconds=30)

    first_report = first.run_once()
    scans_after_first = store.scans
    second_report = second.run_once()

    assert first_report.lease_held_by_other is False
    assert second_report.lease_held_by_other is True
    assert store.scans == scans_after_first
    assert second_report.scanned == 0


def test_the_dispatch_lease_hands_over_when_it_expires() -> None:
    store = build_store()
    _pending(store, "wf-1", "inc-1")
    first = _dispatcher(store, executor_id="executor-a", dispatch_lease_seconds=0.05)
    second = _dispatcher(store, executor_id="executor-b", dispatch_lease_seconds=0.05)
    first.run_once()
    _pending(store, "wf-2", "inc-2")
    time.sleep(0.1)

    report = second.run_once()

    assert report.lease_held_by_other is False
    assert store.get_workflow("wf-2").status is WorkflowStatus.SUCCEEDED


def test_without_a_lease_every_dispatcher_scans() -> None:
    store = _CountingStore(build_store())
    _pending(store, "wf-1", "inc-1")
    _dispatcher(store, executor_id="executor-a").run_once()
    scans_after_first = store.scans

    _dispatcher(store, executor_id="executor-b").run_once()

    assert store.scans > scans_after_first


def test_the_production_config_enables_the_lease(monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_WORKFLOW_DISPATCH_LEASE_SECONDS", raising=False)

    config = WorkflowDispatcherConfig.from_environment(True)

    assert config.dispatch_lease_seconds >= 3 * config.poll_interval_seconds


# ---------------------------------------------------------------- F-A2


class _RecordingExecutor:
    def __init__(self) -> None:
        self.requested: list[str] = []
        self.config = SimpleNamespace(executor_id="executor-recording")

    def execute(self, request_id, request):
        self.requested.append(request_id)
        raise WorkflowLeaseError("recorded only")


def test_dispatcher_sees_past_a_wall_of_filtered_rows() -> None:
    """P0-79B: LIMIT in SQL, filters in Python, oldest first."""

    store = build_store()
    for index in range(300):
        _pending(
            store,
            f"wf-blocked-{index:03d}",
            f"inc-blocked-{index:03d}",
            predecessor_workflow_id="wf-running-predecessor",
            created_at=NOW - timedelta(days=2),
            updated_at=NOW - timedelta(days=2),
        )
    store.save_workflow(
        workflow_request(
            "wf-running-predecessor",
            "inc-predecessor",
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            official_steps=[workflow_step(OP)],
            created_at=NOW - timedelta(days=3),
            updated_at=NOW - timedelta(days=3),
        )
    )
    _pending(store, "wf-eligible", "inc-eligible")
    executor = _RecordingExecutor()
    dispatcher = WorkflowDispatcher(
        store,
        executor,  # type: ignore[arg-type]
        # batch_size 5 -> a 100-row scan window, well inside the 300 blocked rows
        WorkflowDispatcherConfig(enabled=True, batch_size=5, max_workers=1),
    )

    report = dispatcher.run_once()

    assert "wf-eligible" in executor.requested
    assert report.filtered["predecessor"] == 300
    assert report.horizon_exhausted is False


def test_dispatch_report_names_every_filter_that_held_a_row_back() -> None:
    store = build_store()
    # The dispatcher compares not_before with the real clock, so "later" has to
    # be relative to it: NOW is a fixed date and NOW + 1 day expired on
    # 2026-09-06 19:00Z, failing the release gate that evening.
    _pending(
        store,
        "wf-later",
        "inc-later",
        not_before=datetime.now(timezone.utc) + timedelta(days=1),
    )
    _pending(store, "wf-behind", "inc-behind", predecessor_workflow_id="wf-open")
    store.save_workflow(
        workflow_request(
            "wf-open",
            "inc-open",
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            official_steps=[workflow_step(OP)],
        )
    )
    _pending(store, "wf-dangling", "inc-dangling", predecessor_workflow_id="wf-gone")
    executor = _RecordingExecutor()
    dispatcher = WorkflowDispatcher(
        store,
        executor,  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )

    report = dispatcher.run_once()

    assert report.filtered == {
        "not_before": 1,
        "predecessor": 1,
        "predecessor_missing": 1,
    }
    # A dangling predecessor is treated as terminal: the record is dispatchable
    # (and so is the RUNNING predecessor itself, which has no live owner).
    assert set(executor.requested) == {"wf-open", "wf-dangling"}


# ---------------------------------------------------------------- F-A3


def test_a_missing_predecessor_row_does_not_abort_the_tick() -> None:
    store = build_store()
    _pending(store, "wf-dangling", "inc-dangling", predecessor_workflow_id="wf-gone")
    _pending(store, "wf-fine", "inc-fine")
    executor = _RecordingExecutor()
    dispatcher = WorkflowDispatcher(
        store,
        executor,  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )

    dispatcher.run_once()

    assert sorted(executor.requested) == ["wf-dangling", "wf-fine"]


def test_one_poisoned_overdue_workflow_does_not_stop_the_watchdog_sweep() -> None:
    store = build_store()
    operations = [OP]
    _, poisoned = workflow_state(store, operations)
    now = datetime.now(timezone.utc)
    overdue = dict(
        status=WorkflowStatus.RUNNING,
        execution_owner_id=None,
        execution_lease_expires_at=None,
        execution_deadline=now - timedelta(seconds=600),
    )
    # A plan pointer to a plan that no longer exists poisons _sync_plan.
    store.save_workflow(copy_model(poisoned, source_plan_id="plan-gone", **overdue))
    healthy_incident = fault_incident(
        "inc-healthy",
        "event-healthy",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-healthy",
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )
    healthy = workflow_request(
        "wf-healthy",
        "inc-healthy",
        fencing_token=3,
        official_steps=[workflow_step(OP)],
        created_at=NOW,
        updated_at=NOW,
        **overdue,
    )
    store.save_incident_and_workflow(healthy_incident, healthy)

    _dispatcher(store, executor_id="executor-a").run_once()

    assert store.get_workflow(poisoned.request_id).status is WorkflowStatus.FAILED
    assert store.get_workflow("wf-healthy").status is WorkflowStatus.FAILED


# ---------------------------------------------------------------- F-A6


def test_failure_handler_exceptions_are_bounded_and_counted() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [OP])
    store.save_workflow(copy_model(workflow, status=WorkflowStatus.FAILED))
    calls = []

    def handler(failed):
        calls.append(failed.request_id)
        raise RuntimeError("escalation store unavailable")

    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], [OP]),
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, failure_handling_max_attempts=3
        ),
        failure_handler=handler,
    )

    for _ in range(6):
        dispatcher.run_once()

    assert len(calls) == 3
    current = store.get_workflow(workflow.request_id)
    assert current.failure_handled_at is not None
    assert current.failure_handling_attempts == 3
    assert dispatcher.failure_handling_abandoned_total == 1


# ---------------------------------------------------------------- F-A7


@pytest.mark.parametrize(
    ("workflow_status", "plan_status"),
    [
        (WorkflowStatus.SUPERSEDED, PlanStatus.SUPERSEDED),
        (WorkflowStatus.BLOCKED, PlanStatus.BLOCKED),
        (WorkflowStatus.FAILED, PlanStatus.FAILED),
        (WorkflowStatus.SUCCEEDED, PlanStatus.SUCCEEDED),
    ],
)
def test_plan_status_mirrors_the_workflow_status(workflow_status, plan_status) -> None:
    store = build_store()
    plan = RecoveryPlan(
        incident_id="inc-plan",
        attempt_id="attempt-plan",
        trigger="quick-triage:PASS",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": RecoveryAction.RESTART_WORKLOAD,
                "node_ids": ["node-a"],
                "execution_owner": "simulated-runtime",
            }
        ],
    )
    store.save_plan(plan)
    _, workflow = workflow_state(store, [OP])
    workflow = copy_model(workflow, source_plan_id=plan.plan_id)
    store.save_workflow(workflow)
    dispatcher = _dispatcher(store, executor_id="executor-a")

    _mirror_plan(dispatcher, workflow, workflow_status)

    assert store.get_plan(plan.plan_id).status is plan_status


def test_plan_mirror_yields_to_a_plan_another_writer_moved(monkeypatch) -> None:
    """ARCH-D5: the mirror write is a CAS on the plan it read.

    A plan moved between the dispatcher's read and its write is left as the
    other writer put it; the miss is counted and logged, not fatal.
    """
    store = build_store()
    plan = RecoveryPlan(
        incident_id="inc-plan-race",
        attempt_id="attempt-plan-race",
        trigger="quick-triage:PASS",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": RecoveryAction.RESTART_WORKLOAD,
                "node_ids": ["node-a"],
                "execution_owner": "simulated-runtime",
            }
        ],
    )
    store.save_plan(plan)
    _, workflow = workflow_state(store, [OP])
    workflow = copy_model(workflow, source_plan_id=plan.plan_id)
    store.save_workflow(workflow)
    dispatcher = _dispatcher(store, executor_id="executor-a")
    original_get_plan = store.get_plan

    def racing_get_plan(plan_id: str):
        read = original_get_plan(plan_id)
        # Another writer lands between this read and the mirror write.
        store.save_plan(copy_model(read, status=PlanStatus.SUPERSEDED))
        return read

    monkeypatch.setattr(store, "get_plan", racing_get_plan)

    _mirror_plan(dispatcher, workflow, WorkflowStatus.FAILED)

    assert store.get_plan(plan.plan_id).status is PlanStatus.SUPERSEDED, (
        "the mirror write overwrote a plan another writer had moved"
    )
    assert dispatcher.plan_sync_misses_total == 1, "the stale mirror was not counted"
