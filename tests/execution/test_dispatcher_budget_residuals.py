"""The dispatcher's residual remediation-budget gaps after batch 89.

* An internal-error release handed a row back in the status the scan saw, so a
  row that began its budgeted work in that dispatch went back to PENDING and
  held nothing for the 60-s backoff while its node was mid-remediation.
* A fresh storm: rows never refused had no recorded scope, so each was tried
  once -- 95 a tick at batch size 100 -- and another cluster waited about
  N / 95 ticks (six for N = 500) before its first try.
* An upgrade window: workflows an older release left RUNNING without their
  claims re-take them as holds, never refused, so a scope can sit above its
  limit; an older dispatcher still alternating with this one during the roll
  drops the claims again and admits on its own rule. This dispatcher reports
  the over-limit scope (gauge + log) and admits nothing new into it until it
  drains below the limit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import WorkflowStatus
from tests._builders import build_store, copy_model, workflow_step_execution
from tests.execution._budget_hold_support import (
    REBOOT,
    REBOOT_PLAN,
    STOP,
    ScriptedAdapter,
    all_workflows,
    budget_dispatcher,
    holders,
    seed,
)

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
CLUSTER = "cluster:cluster-a"
ONE_PER_CLUSTER = RemediationBudgetPolicy(region_limit=20, cluster_limit=1)
# The class limit sits above the cluster limit so the cluster scope is the one
# that fills.
FIVE_PER_CLUSTER = RemediationBudgetPolicy(cluster_limit=5, resource_class_limit=10)


class _WiringBugOnReboot(ScriptedAdapter):
    """``supports`` raises once, for the first reboot it is asked about: an
    error that says nothing about the record, so the dispatcher releases the
    row with a backoff instead of writing it BLOCKED."""

    def __init__(self) -> None:
        super().__init__(REBOOT_PLAN)
        self.armed = True

    def supports(self, step):
        if step.operation is REBOOT and self.armed:
            self.armed = False
            raise RuntimeError("adapter wiring bug")
        return super().supports(step)


def test_an_internal_error_after_a_budgeted_step_keeps_the_hold_through_the_backoff():
    store = build_store()
    a = seed(store, "a", created_at=NOW)
    b = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    adapter = _WiringBugOnReboot()
    dispatcher = budget_dispatcher(store, adapter, ONE_PER_CLUSTER, max_workers=1)

    report = dispatcher.run_once()

    assert report.internal_errors == 1, report
    released = store.get_workflow(a.request_id)
    assert released.completed_step_indexes == [0], "STOP ran in this dispatch"
    assert released.not_before is not None and released.execution_owner_id is None
    assert released.status is WorkflowStatus.RUNNING, (
        "a row that began its budgeted work went back to PENDING and dropped its hold"
    )
    assert store.remediation_budget_occupancy().get(CLUSTER) == 1

    dispatcher.run_once()
    assert store.get_workflow(b.request_id).status is WorkflowStatus.PENDING
    assert b.request_id not in adapter.started(), adapter.calls

    # The backoff ends: A takes its hold back and reboots; B waits for its end.
    store.save_workflow(copy_model(store.get_workflow(a.request_id), not_before=None))
    dispatcher.run_once()
    assert f"{a.request_id}/1/{REBOOT.value}" in adapter.calls
    assert b.request_id not in adapter.started(), adapter.calls
    adapter.released.add(a.request_id)
    dispatcher.run_once()
    assert store.get_workflow(a.request_id).status is WorkflowStatus.SUCCEEDED
    dispatcher.run_once()
    assert store.get_workflow(b.request_id).status is WorkflowStatus.RUNNING


def test_an_internal_error_before_any_budgeted_work_still_hands_the_row_back():
    """Nothing started: the row looks untouched, PENDING, and holds nothing."""

    store = build_store()
    a = seed(store, "a", created_at=NOW, operations=(REBOOT,))
    adapter = _WiringBugOnReboot()
    dispatcher = budget_dispatcher(store, adapter, ONE_PER_CLUSTER, max_workers=1)

    assert dispatcher.run_once().internal_errors == 1

    released = store.get_workflow(a.request_id)
    assert released.status is WorkflowStatus.PENDING
    assert released.step_executions == [] and released.not_before is not None
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 0


def _storm(store, size: int) -> list[str]:
    return [
        seed(
            store, f"storm-{index:03d}", created_at=NOW + timedelta(milliseconds=index)
        ).request_id
        for index in range(size)
    ]


def test_a_fresh_storm_does_not_hold_another_cluster_for_n_over_the_batch_ticks():
    """500 rows of cluster-a (limit 5) arrive at once, then one of cluster-b.

    The first tick admits five and the store records 95 refusals. From the
    second the tick's occupancy shows cluster-a full, so the guard works out
    the scopes of the rows never refused and passes them over as well,
    recording the refusing scope on each the way the store does: cluster-b
    starts on tick 2 (it took six), and every waiter is counted.
    """

    store = build_store()
    storm = _storm(store, 500)
    late = seed(
        store, "late", cluster_id="cluster-b", created_at=NOW + timedelta(hours=1)
    )
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, FIVE_PER_CLUSTER)

    dispatcher.run_once()
    report = dispatcher.run_once()

    assert store.get_workflow(late.request_id).status is WorkflowStatus.RUNNING, (
        report.filtered
    )
    waiting = [
        workflow
        for workflow in all_workflows(store)
        if workflow.request_id in storm and workflow.status is WorkflowStatus.PENDING
    ]
    assert len(waiting) == 495
    assert {item.remediation_budget_last_blocked_scope for item in waiting} == {CLUSTER}
    assert {item.remediation_budget_wait_count for item in waiting} == {1}
    assert sum(1 for item in waiting if item.request_id in adapter.started()) == 0
    assert len(holders(all_workflows(store), CLUSTER)) == 5

    dispatcher.run_once()
    assert {
        store.get_workflow(item.request_id).remediation_budget_wait_count
        for item in waiting
    } == {1}, "a waiter whose scope is still full was refused again"


def _legacy_running(store, count: int) -> list[str]:
    """RUNNING rows an older release left after their STOP: claims dropped,
    the reboot still to come (or under way)."""

    rows = []
    for index in range(count):
        row = seed(store, f"legacy-{index}", created_at=NOW + timedelta(seconds=index))
        store.save_workflow(
            copy_model(
                row,
                status=WorkflowStatus.RUNNING,
                completed_step_indexes=[0],
                completed_operations=[STOP],
                step_executions=[workflow_step_execution(0, STOP)],
            )
        )
        rows.append(row.request_id)
    return rows


def _over_limit_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "gpu_fault.execution.budget_guard"
        and record.levelno == logging.WARNING
    ]


def test_holds_re_taken_over_the_limit_are_reported_and_drain_before_new_work(caplog):
    store = build_store()
    legacy = _legacy_running(store, 7)
    waiter = seed(store, "waiter", created_at=NOW + timedelta(minutes=1))
    other = seed(
        store, "other", cluster_id="cluster-b", created_at=NOW + timedelta(minutes=2)
    )
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, FIVE_PER_CLUSTER, max_workers=1)

    dispatcher.run_once()  # the seven re-take their holds, never refused
    assert len(holders(all_workflows(store), CLUSTER)) == 7
    assert store.get_workflow(other.request_id).status is WorkflowStatus.RUNNING
    with caplog.at_level(logging.INFO, logger="gpu_fault.execution.budget_guard"):
        dispatcher.run_once()
        dispatcher.run_once()

    watch = dispatcher.budget_watch
    assert watch.over_limit == {CLUSTER: (7, 5)}
    assert watch.over_limit_by_scope_type() == {"cluster": 1}
    assert watch.over_limit_last_seen_timestamp_seconds > 0
    warnings = _over_limit_warnings(caplog)
    assert len(warnings) == 1, "one warning per change, not one per tick"
    assert "cluster:cluster-a" in warnings[0] and "holders=7 limit=5" in warnings[0]
    assert store.get_workflow(waiter.request_id).status is WorkflowStatus.PENDING

    # Two reboots end: five holders, at the limit -- no longer over it, still full.
    adapter.released.update(legacy[:2])
    dispatcher.run_once()
    dispatcher.run_once()
    assert watch.over_limit == {}
    assert store.get_workflow(waiter.request_id).status is WorkflowStatus.PENDING
    assert waiter.request_id not in adapter.started(), adapter.calls

    # One more ends: the scope has drained below its limit and the waiter starts.
    adapter.released.add(legacy[2])
    dispatcher.run_once()
    dispatcher.run_once()
    assert store.get_workflow(waiter.request_id).status is WorkflowStatus.RUNNING


def _old_dispatcher_tick(store, admitted: list[str]) -> None:
    """What an older dispatcher still alternating with this one writes: the
    claims of every RUNNING row past its STOP dropped, and rows admitted by its
    own count (it counted none of those) -- RUNNING with claims."""

    for workflow in all_workflows(store):
        if (
            workflow.status is WorkflowStatus.RUNNING
            and workflow.completed_step_indexes
        ):
            store.save_workflow(copy_model(workflow, remediation_budget_claims=[]))
    for request_id in admitted:
        store.save_workflow(
            copy_model(
                store.get_workflow(request_id),
                status=WorkflowStatus.RUNNING,
                remediation_budget_claims=[CLUSTER, "region"],
                remediation_budget_limits={CLUSTER: 5, "region": 20},
                remediation_budget_last_blocked_reason=None,
                remediation_budget_last_blocked_scope=None,
            )
        )


def test_an_older_dispatcher_in_the_roll_window_cannot_make_this_one_admit():
    """The residual roll window: the older dispatcher exceeds the limit on its
    own ticks; this one re-takes every hold, sees the excess and admits
    nothing new until the scope drains below the limit."""

    store = build_store()
    _legacy_running(store, 5)
    queued = [
        seed(
            store, f"queued-{index}", created_at=NOW + timedelta(minutes=index)
        ).request_id
        for index in range(1, 5)
    ]
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, FIVE_PER_CLUSTER, max_workers=1)

    for old_admits in (queued[:2], []):
        dispatcher.run_once()
        _old_dispatcher_tick(store, old_admits)
    dispatcher.run_once()
    dispatcher.run_once()

    assert dispatcher.budget_watch.over_limit == {CLUSTER: (7, 5)}
    for request_id in queued[2:]:
        assert store.get_workflow(request_id).status is WorkflowStatus.PENDING
        assert request_id not in adapter.started(), adapter.calls
