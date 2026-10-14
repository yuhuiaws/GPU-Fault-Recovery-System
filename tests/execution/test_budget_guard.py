"""``BudgetGuard``: which rows a dispatch tick passes over, and at what cost.

One occupancy read per tick; a waiter is passed over exactly while its scope
is full; a row never refused is passed over -- and recorded as a waiter, once
-- while a scope it would ask for is full, for an incident read cached across
ticks; RUNNING rows, holds, preempting successors and unknown scope kinds never
are. ``observe`` hands the scopes above their limit to the ``BudgetWatch``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from gpu_fault.execution import budget_guard
from gpu_fault.execution.budget_guard import BudgetGuard, BudgetWatch
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import WorkflowRequest, WorkflowStatus
from gpu_fault.store import NotFoundError
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._budget_hold_support import REBOOT, STOP

CLUSTER = "cluster:cluster-a"
POLICY = RemediationBudgetPolicy(cluster_limit=3, resource_class_limit=10)
CONFIG: Any = SimpleNamespace(remediation_budget=POLICY)


class _Store:
    def __init__(self, occupancy: dict[str, int]) -> None:
        self.occupancy = occupancy
        self.reads = 0
        self.incident_reads: list[str] = []
        self.saved: list[WorkflowRequest] = []
        self.save_error: Exception | None = None

    def remediation_budget_occupancy(self) -> dict[str, int]:
        self.reads += 1
        return dict(self.occupancy)

    def get_incident(self, incident_id: str):
        self.incident_reads.append(incident_id)
        if incident_id == "incident-gone":
            raise NotFoundError(incident_id)
        cluster = "cluster-b" if incident_id.endswith("-b") else "cluster-a"
        return fault_incident(incident_id, f"event-{incident_id}", cluster_id=cluster)

    def save_workflow(self, workflow: WorkflowRequest, *, expected=None) -> None:
        assert expected is not None, "a waiter is recorded by compare-and-set"
        if self.save_error is not None:
            raise self.save_error
        self.saved.append(workflow)


def _guard(
    occupancy: dict[str, int], watch: BudgetWatch | None = None
) -> tuple[BudgetGuard, _Store]:
    store = _Store(occupancy)
    return BudgetGuard(store, CONFIG, watch), store  # type: ignore[arg-type]


def _waiter(name: str, scope: str | None = CLUSTER, **values):
    return workflow_request(
        f"workflow-{name}",
        values.pop("incident_id", f"incident-{name}"),
        official_steps=[workflow_step(STOP), workflow_step(REBOOT)],
        remediation_budget_last_blocked_scope=scope,
        **values,
    )


def test_a_tick_reads_the_occupancy_once_and_nothing_else_without_a_full_scope():
    guard, store = _guard({CLUSTER: 2})

    passed = [guard.passes_over(_waiter(str(index), scope=None)) for index in range(5)]
    guard.observe()

    assert passed == [False] * 5
    assert store.reads == 1
    assert store.incident_reads == [] and store.saved == []


def test_a_full_scope_passes_over_every_waiter_with_one_read():
    guard, store = _guard({CLUSTER: 3})

    passed = [guard.passes_over(_waiter(str(index))) for index in range(4)]

    assert passed == [True] * 4
    assert store.reads == 1


def test_a_scope_with_room_lets_every_waiter_through():
    """No quota: the first waiter may still not claim (a fleet-rollout fence
    holds it without a claim), and the ones behind it must get the slot."""

    guard, store = _guard({CLUSTER: 2})

    passed = [guard.passes_over(_waiter(str(index))) for index in range(4)]

    assert passed == [False] * 4
    assert store.reads == 1


def test_scopes_are_counted_apart():
    guard, _ = _guard({CLUSTER: 3, "cluster:cluster-b": 0})

    assert guard.passes_over(_waiter("a")) is True
    assert guard.passes_over(_waiter("b", scope="cluster:cluster-b")) is False


def test_running_rows_holds_successors_and_unknown_kinds_are_never_passed_over():
    guard, store = _guard({CLUSTER: 3, "rack:7": 9})

    running = _waiter("running", status=WorkflowStatus.RUNNING)
    successor = _waiter("successor", preempt_predecessor=True)
    compensation = _waiter(
        "compensation", pending_failure_step_index=1, pending_failure_error="failed"
    )
    started = _waiter(
        "started",
        completed_step_indexes=[0],
        step_executions=[workflow_step_execution(0, STOP)],
    )
    unknown = _waiter("unknown", scope="rack:7")

    for workflow in (running, successor, compensation, started, unknown):
        assert guard.passes_over(workflow) is False, workflow.request_id
    assert guard.passes_over(_waiter("waiter")) is True
    assert store.reads == 1


def test_a_row_never_refused_is_passed_over_and_recorded_while_its_scope_is_full():
    watch = BudgetWatch()
    guard, store = _guard({CLUSTER: 3}, watch)
    fresh = _waiter("fresh", scope=None)

    assert guard.passes_over(fresh) is True

    (recorded,) = store.saved
    assert recorded.request_id == fresh.request_id
    assert recorded.remediation_budget_last_blocked_scope == CLUSTER
    assert recorded.remediation_budget_wait_count == 1
    assert recorded.remediation_budget_last_blocked_reason == (
        "remediation concurrency budget is full: "
        "scope=cluster:cluster-a active=3 limit=3"
    )
    assert watch.clusters == {"incident-fresh": "cluster-a"}

    # Another cluster's fresh row is judged on its own scopes.
    assert guard.passes_over(_waiter("other", scope=None, incident_id="x-b")) is False
    # A later tick finds the incident's cluster in the watch.
    later, later_store = _guard({CLUSTER: 3}, watch)
    assert later.passes_over(_waiter("fresh", scope=None)) is True
    assert later_store.incident_reads == []


def test_a_fresh_row_the_guard_cannot_judge_is_left_to_the_store():
    guard, store = _guard({CLUSTER: 3})

    owned = _waiter("owned", scope=None, execution_owner_id="executor-b")
    gone = _waiter("gone", scope=None, incident_id="incident-gone")
    assert guard.passes_over(owned) is False
    assert guard.passes_over(gone) is False
    store.save_error = StaleWriteError("workflow moved")
    assert guard.passes_over(_waiter("moved", scope=None)) is False
    assert store.saved == []


def test_the_store_calls_spent_on_fresh_rows_are_capped_per_tick(monkeypatch):
    monkeypatch.setattr(budget_guard, "FRESH_WAITER_STORE_CALLS", 3)
    guard, store = _guard({CLUSTER: 3})

    passed = [guard.passes_over(_waiter(str(index), scope=None)) for index in range(4)]

    # One incident read and one write each: the second row spends the last
    # call on its read, so it is left to the store, as is everything after.
    assert passed == [True, False, False, False]
    assert len(store.incident_reads) + len(store.saved) == 3


def test_observe_reports_scopes_above_their_limit_and_logs_each_change(caplog):
    watch = BudgetWatch()
    caplog.set_level(logging.INFO, logger="gpu_fault.execution.budget_guard")

    for occupancy in (
        {CLUSTER: 5, "region": 5, "node:cluster-a:n1": 1},
        {CLUSTER: 5, "region": 5},
        {CLUSTER: 3},
    ):
        guard, _ = _guard(occupancy, watch)
        guard.observe()
        if occupancy[CLUSTER] == 5:
            assert watch.over_limit == {CLUSTER: (5, 3)}
            assert watch.over_limit_by_scope_type() == {"cluster": 1}

    assert watch.over_limit == {}, "at the limit is full, not over it"
    assert watch.over_limit_last_seen_timestamp_seconds > 0
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2, messages
    assert "cluster:cluster-a holders=5 limit=3" in messages[0]
    assert "drained" in messages[1]

    watch.observe({CLUSTER: (4, 3)})
    watch.idle()
    assert watch.over_limit == {}, "a process without the dispatch lease reports none"
