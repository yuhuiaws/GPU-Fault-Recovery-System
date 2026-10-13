"""The dispatcher under a remediation budget: limits hold across ticks, and a
storm that fills one scope cannot starve the rest of the fleet.

Batch 88: with the claims dropped after the first budgeted step, cluster limit
5 became 15 RUNNING workflows after three ticks and the action-production shape
climbed to the 100-row cap with a region budget of 20; the dispatcher's
100-oldest selection was the only concurrency limit. With the claims held, a
single-cluster storm parks its waiters at the head of that selection, so rows
whose recorded refusing scope is still full are passed over. The guard only
skips, never reorders: once a scope has room its waiters are candidates again
in eligibility order (FIFO within the scope is kept), all of them -- a waiter
that cannot claim (held by the fleet-rollout fence) must not starve the rest.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.fleet import FleetNodeReadiness, FleetReadinessReport
from gpu_fault.models import WorkflowStatus
from tests._builders import build_store
from tests.execution._budget_hold_support import (
    ACTION_MODEL,
    REBOOT_PLAN,
    ScriptedAdapter,
    all_workflows,
    budget_dispatcher,
    holders_by_cluster,
    seed,
)

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
# The class limit sits above the cluster limit so the cluster scope is the one
# that fills (and is recorded as the refusing scope) in the storm tests.
STORM_POLICY = RemediationBudgetPolicy(cluster_limit=5, resource_class_limit=10)


def _cluster_of(request_id: str, clusters: int) -> str:
    return f"cluster-{int(request_id.rsplit('-', 1)[1]) % clusters:03d}"


def test_a_cluster_limit_of_five_is_never_exceeded_across_ticks():
    """tick_budget.py: 60 workflows over 4 clusters, limit 5 per cluster."""

    store = build_store()
    clusters = 4
    for index in range(60):
        seed(
            store,
            f"{index:03d}",
            cluster_id=f"cluster-{index % clusters:03d}",
            created_at=NOW + timedelta(microseconds=index),
        )
    adapter = ScriptedAdapter(REBOOT_PLAN)
    policy = RemediationBudgetPolicy(
        region_limit=1000, cluster_limit=5, resource_class_limit=5
    )
    dispatcher = budget_dispatcher(store, adapter, policy)

    for tick in range(1, 5):
        dispatcher.run_once()
        held = holders_by_cluster(all_workflows(store))
        assert max(held.values()) <= 5, f"tick {tick}: {dict(held)}"
        running = Counter(
            _cluster_of(workflow.request_id, clusters)
            for workflow in all_workflows(store)
            if workflow.status is WorkflowStatus.RUNNING
        )
        assert max(running.values()) <= 5, f"tick {tick}: {dict(running)}"
        started = Counter(
            _cluster_of(request_id, clusters) for request_id in adapter.started()
        )
        assert max(started.values()) <= 5, f"tick {tick}: {dict(started)}"


def test_the_region_budget_bounds_the_action_production_shape():
    """32 clusters x 4 workflows, production defaults (region 20, class 2).

    Every workflow claims three resource classes with a limit of 2, so at most
    2 run per cluster, and the region limit cuts the 64 that would allow to 20.
    Releasing the reboots drains every wave: no waiter is starved.
    """

    store = build_store()
    clusters = 32
    for index in range(clusters * 4):
        seed(
            store,
            f"{index:03d}",
            cluster_id=f"cluster-{index % clusters:03d}",
            operations=ACTION_MODEL,
            created_at=NOW + timedelta(microseconds=index),
        )
    adapter = ScriptedAdapter(ACTION_MODEL)
    dispatcher = budget_dispatcher(store, adapter, RemediationBudgetPolicy())

    peak = 0
    for tick in range(1, 7):
        dispatcher.run_once()
        workflows = all_workflows(store)
        held = holders_by_cluster(workflows)
        region = sum(
            1
            for workflow in workflows
            if workflow.status is WorkflowStatus.RUNNING
            and "region" in workflow.remediation_budget_claims
        )
        assert region <= 20, f"tick {tick}: {region} workflows hold the region"
        assert max(held.values(), default=0) <= 2, f"tick {tick}: {dict(held)}"
        in_flight = adapter.started() - {
            workflow.request_id
            for workflow in workflows
            if workflow.status is WorkflowStatus.SUCCEEDED
        }
        assert len(in_flight) <= 20, f"tick {tick}: {len(in_flight)} started"
        peak = max(peak, region)
    assert peak == 20, "the region budget is the binding limit of this shape"

    for _ in range(60):
        adapter.released.update(adapter.started())
        dispatcher.run_once()
        statuses = Counter(workflow.status for workflow in all_workflows(store))
        if statuses[WorkflowStatus.SUCCEEDED] == clusters * 4:
            break
    assert statuses == Counter({WorkflowStatus.SUCCEEDED: clusters * 4}), statuses


def _storm(store, size: int) -> list[str]:
    return [
        seed(
            store, f"storm-{index:03d}", created_at=NOW + timedelta(milliseconds=index)
        ).request_id
        for index in range(size)
    ]


def test_a_storm_with_a_full_cluster_does_not_starve_another_cluster():
    """150 workflows in cluster-a (limit 5, all five slots held), then one in
    cluster-b: it starts on the very next tick, ahead of 145 older waiters."""

    store = build_store()
    storm = _storm(store, 150)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, STORM_POLICY)
    for _ in range(3):
        dispatcher.run_once()
    waiting = [
        workflow
        for workflow in all_workflows(store)
        if workflow.request_id in storm and workflow.status is WorkflowStatus.PENDING
    ]
    assert len(waiting) == 145
    assert {workflow.remediation_budget_last_blocked_scope for workflow in waiting} == {
        "cluster:cluster-a"
    }

    late = seed(
        store, "late", cluster_id="cluster-b", created_at=NOW + timedelta(hours=1)
    )
    report = dispatcher.run_once()

    assert store.get_workflow(late.request_id).status is WorkflowStatus.RUNNING
    assert late.request_id in adapter.started(), "cluster-b never reached an adapter"
    assert report.filtered.get("remediation_budget") == 145, report.filtered
    unchanged = [store.get_workflow(workflow.request_id) for workflow in waiting]
    assert [item.remediation_budget_wait_count for item in unchanged] == [
        item.remediation_budget_wait_count for item in waiting
    ], "a waiter whose scope is still full was dispatched again"


def test_a_freed_slot_goes_to_the_oldest_waiter_of_its_scope():
    """One worker executes the batch in its order, so the slot's winner is the
    scan's choice: the guard kept the scope's queue order."""

    store = build_store()
    storm = _storm(store, 150)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, STORM_POLICY, max_workers=1)
    for _ in range(3):
        dispatcher.run_once()
    running = [
        request_id
        for request_id in storm
        if store.get_workflow(request_id).status is WorkflowStatus.RUNNING
    ]
    assert len(running) == 5
    oldest_waiter = next(
        request_id
        for request_id in storm
        if store.get_workflow(request_id).status is WorkflowStatus.PENDING
    )

    adapter.released.add(running[0])
    dispatcher.run_once()  # the released workflow ends; the slot frees
    assert store.get_workflow(running[0]).status is WorkflowStatus.SUCCEEDED
    dispatcher.run_once()

    admitted = [
        request_id
        for request_id in storm
        if store.get_workflow(request_id).status is WorkflowStatus.RUNNING
        and request_id not in running
    ]
    assert admitted == [oldest_waiter]


class _FencedFleet:
    """Every node is ready; the ``fenced`` clusters have a rollout in flight."""

    def __init__(self, fenced: set[str]) -> None:
        self.fenced = fenced

    def fleet_rollout_fence_deployments(self, cluster_id: str) -> list[str]:
        return ["rollout-1"] if cluster_id in self.fenced else []

    def readiness(self, cluster_id: str, node_ids: list[str]) -> FleetReadinessReport:
        return FleetReadinessReport(
            cluster_id=cluster_id,
            ready=True,
            evaluated_at=NOW,
            nodes=[
                FleetNodeReadiness(
                    node_id=node_id,
                    ready=True,
                    generation=7,
                    endpoint=f"https://{node_id}:9099",
                    reasons=[],
                )
                for node_id in node_ids
            ],
        )


def test_a_waiter_the_rollout_fence_holds_does_not_starve_the_ones_behind_it():
    """Region limit 1: the oldest region waiter's cluster is fenced by a fleet
    rollout, so it is dispatched and returns without a claim; the next waiter
    must take the freed slot in the same tick."""

    store = build_store()
    holder = seed(store, "holder", cluster_id="cluster-h", created_at=NOW)
    fenced = seed(
        store, "fenced", cluster_id="cluster-f", created_at=NOW + timedelta(seconds=1)
    )
    behind = seed(
        store, "behind", cluster_id="cluster-g", created_at=NOW + timedelta(seconds=2)
    )
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(
        store, adapter, RemediationBudgetPolicy(region_limit=1), max_workers=1
    )
    dispatcher.run_once()
    for waiter in (fenced, behind):
        row = store.get_workflow(waiter.request_id)
        assert row.remediation_budget_last_blocked_scope == "region", row.request_id

    dispatcher.executor.fleet_registry = _FencedFleet({"cluster-f"})
    adapter.released.add(holder.request_id)
    dispatcher.run_once()  # the holder ends; the region slot frees
    assert store.get_workflow(holder.request_id).status is WorkflowStatus.SUCCEEDED
    dispatcher.run_once()

    assert store.get_workflow(fenced.request_id).status is WorkflowStatus.PENDING
    assert store.get_workflow(behind.request_id).status is WorkflowStatus.RUNNING
    assert behind.request_id in adapter.started(), adapter.calls
