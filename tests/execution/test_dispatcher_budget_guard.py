"""The dispatcher under a remediation budget: slots hold across ticks, no fixed
number caps a correlated fault, and a storm that fills one scope -- or trips
one cluster's storm breaker -- cannot starve the rest of the fleet.

Batch 88: with the claims dropped after the first budgeted step, a cluster
limit of 5 became 15 RUNNING workflows after three ticks; the dispatcher's
100-oldest selection was the only concurrency limit. With the claims held, a
storm parks its waiters at the head of that selection, so rows whose recorded
refusing scope is still full -- or whose cluster's breaker is open -- are
passed over. The guard only skips, never reorders: once a scope has room its
waiters are candidates again in eligibility order (FIFO within the scope is
kept). The region, per-cluster and per-resource-class limits batch 88 loaded
are retired: each cluster's storm breaker bounds a correlated fault instead.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import WorkflowStatus
from gpu_fault.remediation_storm import storm_scope
from tests._builders import build_store
from tests.execution._budget_hold_support import (
    ACTION_MODEL,
    REBOOT_PLAN,
    ScriptedAdapter,
    all_workflows,
    budget_dispatcher,
    holders,
    large_fleet,
    seed,
)

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
POLICY = RemediationBudgetPolicy()
SHARED_NODE = "node-shared"
NODE_SCOPE = f"node:cluster-a:{SHARED_NODE}"


def _cluster_of(request_id: str, clusters: int) -> str:
    return f"cluster-{int(request_id.rsplit('-', 1)[1]) % clusters:03d}"


@pytest.mark.usefixtures(large_fleet.__name__)
def test_one_node_slot_is_never_exceeded_across_ticks():
    """60 workflows on 4 nodes of one cluster: one per node at any time (a
    1000-node fleet keeps the breaker out of reach)."""

    store = build_store()
    nodes = 4
    for index in range(60):
        seed(
            store,
            f"{index:03d}",
            node_id=f"node-{index % nodes}",
            created_at=NOW + timedelta(microseconds=index),
        )
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, POLICY)

    for tick in range(1, 5):
        dispatcher.run_once()
        workflows = all_workflows(store)
        for node in range(nodes):
            held = holders(workflows, f"node:cluster-a:node-{node}")
            assert len(held) <= 1, f"tick {tick}: node-{node} held by {held}"
        assert len(adapter.started()) == nodes, f"tick {tick}: {adapter.calls}"


@pytest.mark.usefixtures(large_fleet.__name__)
def test_no_fixed_number_caps_the_action_production_shape():
    """32 clusters x 4 workflows, each on its own node: under the retired
    20 / 5 / 2 limits at most 20 ran (2 per cluster); with nothing but the
    per-node slot and a breaker a 1000-node fleet keeps out of reach, every
    one runs at once, and releasing the reboots drains them all."""

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
    batch = WorkflowDispatcherConfig.from_mapping({}, True).batch_size
    dispatcher = budget_dispatcher(store, adapter, POLICY, batch_size=batch)

    dispatcher.run_once()  # the derived batch of 346 holds all 128

    running = Counter(
        _cluster_of(workflow.request_id, clusters)
        for workflow in all_workflows(store)
        if workflow.status is WorkflowStatus.RUNNING
    )
    assert sum(running.values()) == clusters * 4, running
    assert set(running.values()) == {4}, running

    for _ in range(60):
        adapter.released.update(adapter.started())
        dispatcher.run_once()
        statuses = Counter(workflow.status for workflow in all_workflows(store))
        if statuses[WorkflowStatus.SUCCEEDED] == clusters * 4:
            break
    assert statuses == Counter({WorkflowStatus.SUCCEEDED: clusters * 4}), statuses


def _storm(store, size: int, **values) -> list[str]:
    return [
        seed(
            store,
            f"storm-{index:03d}",
            created_at=NOW + timedelta(milliseconds=index),
            **values,
        ).request_id
        for index in range(size)
    ]


def test_a_tripped_breaker_does_not_starve_another_cluster():
    """150 workflows of cluster-a on 150 nodes: without a registered fleet
    the breaker trips on the third and holds the rest; one late workflow of
    cluster-b starts on the very next tick, ahead of 148 older waiters."""

    store = build_store()
    storm = _storm(store, 150)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, POLICY)
    for _ in range(3):
        dispatcher.run_once()
    waiting = [
        workflow
        for workflow in all_workflows(store)
        if workflow.request_id in storm and workflow.status is WorkflowStatus.PENDING
    ]
    assert len(waiting) == 148
    assert {workflow.remediation_budget_last_blocked_scope for workflow in waiting} == {
        storm_scope("cluster-a")
    }

    late = seed(
        store, "late", cluster_id="cluster-b", created_at=NOW + timedelta(hours=1)
    )
    report = dispatcher.run_once()

    assert store.get_workflow(late.request_id).status is WorkflowStatus.RUNNING
    assert late.request_id in adapter.started(), "cluster-b never reached an adapter"
    assert report.filtered.get("remediation_budget") == 148, report.filtered
    unchanged = [store.get_workflow(workflow.request_id) for workflow in waiting]
    assert [item.remediation_budget_wait_count for item in unchanged] == [
        item.remediation_budget_wait_count for item in waiting
    ], "a waiter held by the open breaker was dispatched again"


def test_a_freed_slot_goes_to_the_oldest_waiter_of_its_scope():
    """One worker executes the batch in its order, so the slot's winner is the
    scan's choice: the guard kept the scope's queue order."""

    store = build_store()
    storm = _storm(store, 150, node_id=SHARED_NODE)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, POLICY, max_workers=1)
    for _ in range(3):
        dispatcher.run_once()
    running = [
        request_id
        for request_id in storm
        if store.get_workflow(request_id).status is WorkflowStatus.RUNNING
    ]
    assert running == storm[:1]
    assert holders(all_workflows(store), NODE_SCOPE) == running
    oldest_waiter = storm[1]

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
