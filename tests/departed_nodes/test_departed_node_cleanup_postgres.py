"""The departed-node cleanup on PostgreSQL, the store production runs on.

The memory store makes every write trivially atomic; these run the same cases
where the compare-and-set, the advisory lock that serializes two executors'
inventory reports, and the periodic task lease that makes one control-worker
process the writer are real.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.periodic_services import PeriodicServiceRunner
from gpu_fault.departed_node_cleanup import DepartedNodeCleanup
from gpu_fault.departed_nodes import DepartedNodeCleanupStatus
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared import departed_nodes as shared_departed_nodes
from gpu_fault_release import regional_release_probes as PROBES
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    EKS_NODE,
    HP_CLUSTER,
    HP_NODE,
    HP_PRESENT,
    ago,
    eks_departure,
    hyperpod_departure,
    inventory_report,
)
from tests.store._postgres_processor_claim_support import postgres_store_instance

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is required",
)


@pytest.fixture
def store() -> Iterator[PostgresStore]:
    yield from postgres_store_instance()


def _workflow_safety(monkeypatch: pytest.MonkeyPatch, store: Any) -> dict[str, Any]:
    from gpu_fault.app import ApplicationContext

    context = ApplicationContext(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    exec(compile(PROBES.probe_source("workflow_safety"), "<probe>", "exec"), {})
    monkeypatch.undo()
    return dict(json.loads(output.getvalue()))


def test_a_reclaimed_hyperpod_node_is_cleaned_on_postgres(
    store: PostgresStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_ = hyperpod_departure(store)
    assert _workflow_safety(monkeypatch, store)["blocker_count"] == 2

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.REVOKED
    )
    for request_id in ("workflow-hp-efa", "workflow-hp-xid"):
        assert store.get_workflow(request_id).status is WorkflowStatus.SUPERSEDED
    for incident_id in ("incident-hp-efa", "incident-hp-xid"):
        assert store.get_incident(incident_id).state is IncidentState.RECOVERED
    [stored] = store.list_departed_node_cleanups(HP_CLUSTER)
    assert stored == record, "the ledger line round-trips through gpu_fault_objects"
    assert _workflow_safety(monkeypatch, store)["blocker_count"] == 0
    assert world_.cleanup.run_once() == []
    assert len(world_.notifications.sent) == 1


def test_a_reclaimed_eks_node_is_cleaned_on_postgres(store: PostgresStore) -> None:
    world_ = eks_departure(store)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert store.get_agent(EKS_CLUSTER, EKS_NODE).lifecycle_state is (
        AgentLifecycleState.REVOKED
    )
    assert store.get_incident("incident-eks-efa").state is IncidentState.RECOVERED
    assert record.evidence["sources"]["ec2"]["state"] == "terminated"


def test_two_executors_reporting_at_once_lose_nothing(
    store: PostgresStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge reads the previous record: without the cluster's lock the
    sibling's report, read before the slow writer committed, would be
    overwritten by it (the chain count and the newer node list lost)."""

    base = ago(seconds=30)
    store.record_kubernetes_node_inventory(
        inventory_report("cluster-a", ["node-a", "node-b"], observed_at=base),
        received_at=base,
    )
    merge = shared_departed_nodes.merge_node_inventory
    entered = Event()

    def slow_merge(previous: Any, report: Any, *, received_at: Any) -> Any:
        merged = merge(previous, report, received_at=received_at)
        if report.executor_id == "executor-slow":
            entered.set()
            time.sleep(0.5)
        return merged

    monkeypatch.setattr(shared_departed_nodes, "merge_node_inventory", slow_merge)
    second = PostgresStore(
        os.environ["GPU_FAULT_TEST_POSTGRES_URL"], initialize_schema=False
    )
    first_at = ago(seconds=20)
    second_at = ago(seconds=10)
    try:
        slow = threading.Thread(
            target=store.record_kubernetes_node_inventory,
            args=(
                inventory_report(
                    "cluster-a",
                    ["node-a"],
                    observed_at=first_at,
                    executor_id="executor-slow",
                ),
            ),
            kwargs={"received_at": first_at},
        )
        slow.start()
        assert entered.wait(5), "the slow writer never reached its merge"
        second.record_kubernetes_node_inventory(
            inventory_report("cluster-a", ["node-a", "node-c"], observed_at=second_at),
            received_at=second_at,
        )
        slow.join(10)
    finally:
        second.close()

    record = store.get_kubernetes_node_inventory("cluster-a")
    assert record is not None
    assert record.node_names == ["node-a", "node-c"], "the newer report won"
    assert record.reports_in_chain == 3, "neither report was lost"
    assert record.absent["node-b"].since == first_at


def test_only_the_lease_holder_runs_the_cleanup_on_postgres(
    store: PostgresStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_ = hyperpod_departure(store)
    peer_store = PostgresStore(
        os.environ["GPU_FAULT_TEST_POSTGRES_URL"], initialize_schema=False
    )
    try:
        peer = DepartedNodeCleanup(
            peer_store,
            registry=world_.registry,
            incident_closure=world_.closure,
            provider=world_.provider,
        )
        runners = []
        for owner, context in (
            (
                "pod-a:1",
                SimpleNamespace(store=store, departed_node_cleanup=world_.cleanup),
            ),
            ("pod-b:1", SimpleNamespace(store=peer_store, departed_node_cleanup=peer)),
        ):
            runner = PeriodicServiceRunner(
                context=context,
                processor=SimpleNamespace(
                    is_healthy=lambda: True,
                    active_consumers=True,
                    is_leader=lambda: True,
                    owner_id=owner,
                ),
                stop=Event(),
                identity_registries=[],
                ingest_node_health_findings=lambda *args, **kwargs: None,
                notify_silent_collectors=lambda *args, **kwargs: None,
            )
            monkeypatch.setattr(runner, "_JOBS", ("departed_nodes",))
            runners.append(runner)
        for runner in runners:
            runner.run_all_due(time.monotonic())
    finally:
        peer_store.close()

    assert world_.cleanup.cycles_total == 1
    assert peer.cycles_total == 0


def test_a_restart_between_drain_and_revoke_resumes_on_postgres(
    store: PostgresStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_ = hyperpod_departure(store)
    real = world_.registry.revoke_agent
    calls = {"count": 0}

    def crash_once(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("control-worker restarted mid-cleanup")
        return real(*args, **kwargs)

    monkeypatch.setattr(world_.registry, "revoke_agent", crash_once)

    [first] = world_.cleanup.run_once()
    drained = store.get_agent(HP_CLUSTER, HP_NODE)
    [second] = world_.cleanup.run_once()
    revoked = store.get_agent(HP_CLUSTER, HP_NODE)

    assert first.status is DepartedNodeCleanupStatus.PENDING
    assert drained.lifecycle_state is AgentLifecycleState.DRAINING
    assert second.status is DepartedNodeCleanupStatus.CLEANED, second.blockers
    assert revoked.lifecycle_state is AgentLifecycleState.REVOKED
    assert revoked.generation == drained.generation, "no second drain"
    assert store.get_agent(HP_CLUSTER, HP_PRESENT).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )
