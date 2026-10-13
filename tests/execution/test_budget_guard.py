"""``BudgetGuard``: which rows a dispatch tick passes over, and at what cost.

At most one occupancy read per tick, and none when no row carries a recorded
refusing scope; a waiter is passed over exactly while its scope is full;
RUNNING rows, holds, preempting successors and unknown scope kinds never are.
"""

from __future__ import annotations

from gpu_fault.execution.budget_guard import BudgetGuard
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import WorkflowStatus
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.execution._budget_hold_support import REBOOT, STOP

CLUSTER = "cluster:cluster-a"
POLICY = RemediationBudgetPolicy(cluster_limit=3)


class _OccupancyStore:
    def __init__(self, occupancy: dict[str, int]) -> None:
        self.occupancy = occupancy
        self.reads = 0

    def remediation_budget_occupancy(self) -> dict[str, int]:
        self.reads += 1
        return dict(self.occupancy)


def _guard(occupancy: dict[str, int]) -> tuple[BudgetGuard, _OccupancyStore]:
    store = _OccupancyStore(occupancy)
    return BudgetGuard(store, lambda: POLICY), store


def _waiter(name: str, scope: str | None = CLUSTER, **values):
    return workflow_request(
        f"workflow-{name}",
        f"incident-{name}",
        official_steps=[workflow_step(STOP), workflow_step(REBOOT)],
        remediation_budget_last_blocked_scope=scope,
        **values,
    )


def test_a_tick_without_budget_waiters_reads_no_occupancy():
    guard, store = _guard({CLUSTER: 3})

    passed = [guard.passes_over(_waiter(str(index), scope=None)) for index in range(5)]

    assert passed == [False] * 5
    assert store.reads == 0


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
