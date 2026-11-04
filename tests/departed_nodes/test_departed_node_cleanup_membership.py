"""A planned de-registration is not an operator alarm (batch 249).

Live 2026-11-04 (R18, the S69 remove/re-join proof): ``gpu-fault-admin
remove-cluster`` of the HyperPod cluster mailed ``[GPU node departed: operator
action needed] <cluster>: <node>`` for every node of the cluster, each naming
``--retire-departed-agents``. The removal publishes the cluster ``DRAINING``
first; from then on the executor's node report is refused (HTTP 423,
``regional_cluster_request_allowed``), and once the removal stops the node
runtime every Agent lease lapses while its heartbeat is minutes old. Every
Agent became a candidate, the lifecycle a refusal (``cluster_not_active``) and
each node one mail per refusal set; a re-join, the cluster ``PENDING`` over
the old Agent records and the old report, did it again.

The cleanup judges ACTIVE clusters only: any other lifecycle is a membership
operation under way (join, remove-cluster, uninstall) that owns the cluster's
records. These tests drive the removal and the re-join through the registry
revisions the release publishes and check what an operator sees -- the
notifications, the ``status`` lines, the refusal counter and the ledger gauges
-- and that a real departure on an ACTIVE cluster is judged and notified
exactly as before.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import pytest

from gpu_fault.departed_node_cleanup import (
    cleaned_notification,
    ledger_metric_lines,
    refused_notification,
)
from gpu_fault.departed_nodes import DepartedNodeCleanupStatus
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.kubernetes_node_inventory import NODE_INVENTORY_PATH
from gpu_fault.models import IncidentState, WorkflowStatus, WorkflowStepStatus
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    RegionalRegistryRevision,
)
from gpu_fault.regional_registry_runtime import regional_cluster_request_allowed
from gpu_fault.store import NotFoundError
from gpu_fault_release.regional_admin_checks_coverage import departed_node_lines
from tests._builders import build_store, workflow_step_execution
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    EKS_NODE,
    HP_CLUSTER,
    HP_NAME,
    HP_NODE,
    PLUGIN,
    World,
    agent,
    ago,
    eks_departure,
    hyperpod_departure,
    now,
    registration,
    world,
)

DRAINING = RegionalClusterLifecycle.DRAINING
REVOKED = RegionalClusterLifecycle.REVOKED
PENDING = RegionalClusterLifecycle.PENDING
ACTIVE = RegionalClusterLifecycle.ACTIVE
OPERATOR_SUBJECT = "[GPU node departed: operator action needed]"


def _members(provider: str) -> tuple[str, list[tuple[str, str]]]:
    """The cluster under removal and its (node, recorded instance) pairs."""

    if provider == "hyperpod":
        # The instance is in the node name; a HyperPod-era record carries the
        # Kubernetes Node UID.
        return HP_CLUSTER, [
            (f"hyperpod-i-0b00000{index}", f"node-uid-{index}") for index in (1, 2, 3)
        ]
    return EKS_CLUSTER, [
        (f"ip-10-0-50-{index}.us-west-2.compute.internal", f"i-0e0000b{index}")
        for index in (1, 2, 3)
    ]


def _publish(
    store: Any, *registrations: RegionalClusterRegistration, reason: str
) -> None:
    """One registry revision, as the release publishes it: it replaces the set."""

    try:
        generation = store.get_regional_registry_head().generation
    except NotFoundError:
        generation = 0
    store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=generation + 1,
            registrations=list(registrations),
            previous_generation=generation or None,
            required_member_ids=[],
            reason=reason,
        ),
        expected_generation=generation,
    )


class Clock:
    """The cleanup's clock; the store keeps the times its records were written."""

    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def _status_lines(store: Any) -> list[str]:
    """What ``gpu-fault-admin status`` prints from the ledger (``control_api``)."""

    clusters: dict[str, Any] = {}
    for record in store.list_departed_node_cleanups():
        clusters.setdefault(record.cluster_id, {"departed_node_cleanup": []})[
            "departed_node_cleanup"
        ].append(record.model_dump(mode="json"))
    return departed_node_lines(
        {"departed_node_cleanup_enabled": True, "clusters": clusters}
    )


def _open_nodes(store: Any) -> list[str]:
    """The ledger's open-line counts every worker renders (the age gauge beside
    them feeds ``GpuFaultDepartedNodeCleanupStuck``)."""

    return [
        line
        for line in ledger_metric_lines(
            store.list_departed_node_cleanups(), now=now(), enabled=True
        )
        if line.startswith("gpu_fault_departed_node_cleanup_open_nodes")
    ]


@pytest.mark.parametrize(
    ("lifecycle", "heartbeat", "node_report"),
    [
        (ACTIVE, True, True),
        (PENDING, True, False),
        (DRAINING, True, False),
        (REVOKED, False, False),
    ],
)
def test_outside_active_the_node_report_is_refused_so_nothing_proves_a_departure(
    lifecycle: RegionalClusterLifecycle, heartbeat: bool, node_report: bool
) -> None:
    """Why a cluster that is not ACTIVE is not judged: from the drain (or
    before a join activates) the control plane refuses the executor's node
    report (423), so the report the cleanup holds only ages; the Agents keep
    their leases while they run and lose them when the removal stops them."""

    item = registration(HP_CLUSTER, lifecycle=lifecycle)

    assert (
        regional_cluster_request_allowed(
            item, path="/v1/fleet/agents/heartbeat", method="POST"
        )
        is heartbeat
    )
    assert (
        regional_cluster_request_allowed(item, path=NODE_INVENTORY_PATH, method="POST")
        is node_report
    )


@pytest.mark.parametrize("provider", ["hyperpod", "eks-managed"])
def test_remove_cluster_mails_nobody_and_leaves_no_ledger_line(provider: str) -> None:
    """The live shape, cycle by cycle through the removal's registry revisions."""

    store = build_store()
    world_ = world(store)
    cluster_id, members = _members(provider)
    nodes = [node for node, _ in members]
    drained_at = ago(minutes=40)
    clock = Clock(drained_at)
    world_.cleanup.now = clock
    _publish(store, registration(cluster_id, provider=provider), reason="deploy")
    # The executor's reports up to the drain; the next one is refused (423).
    for minute in range(15, -1, -1):
        world_.report(cluster_id, nodes, at=drained_at - timedelta(minutes=minute))
    # KUBERNETES_QUIESCED stops the node runtime 4 minutes after the drain:
    # the last heartbeats, each with the default 90 s lease.
    stopped_at = drained_at + timedelta(minutes=4)
    for node, instance in members:
        store.save_agent(
            agent(
                cluster_id,
                node,
                instance_id=instance,
                last_seen=stopped_at,
                lease_seconds=90,
            )
        )
    assert world_.cleanup.run_once() == [], "live Agents in a fresh report"

    _publish(
        store,
        registration(cluster_id, provider=provider, lifecycle=DRAINING),
        reason=f"remove {cluster_id} draining",
    )
    seen: list[Any] = []
    status_during_removal: list[str] = []
    # 6 min: leases lapsed, heartbeats 2 min old, the report 6 min old;
    # 15 min: heartbeats past the 10-minute guard (a second refusal set).
    for minutes in (6, 9, 15):
        clock.at = drained_at + timedelta(minutes=minutes)
        seen.extend(world_.cleanup.run_once())
        status_during_removal.extend(_status_lines(store))
    _publish(
        store,
        registration(cluster_id, provider=provider, lifecycle=REVOKED),
        reason=f"remove {cluster_id} revoked",
    )
    clock.at = drained_at + timedelta(minutes=16)
    seen.extend(world_.cleanup.run_once())
    _publish(store, reason=f"remove {cluster_id} purged")
    clock.at = drained_at + timedelta(minutes=17)
    seen.extend(world_.cleanup.run_once())

    assert world_.notification_subjects() == [], "a removal is no departure"
    assert seen == [], [(item.node_id, item.blocker_classes) for item in seen]
    assert status_during_removal == [], status_during_removal
    assert store.list_departed_node_cleanups() == []
    assert world_.cleanup.refusals_total == Counter()
    assert world_.cleanup.cycles_total == 6
    assert world_.provider.calls == [], "nothing is put to the provider"
    assert [store.get_agent(cluster_id, node).lifecycle_state for node in nodes] == [
        AgentLifecycleState.ACTIVE
    ] * len(nodes), "the removal, not the cleanup, owns the Agent records"


def test_a_rejoin_over_the_old_records_mails_nobody_until_a_real_departure() -> None:
    """A cluster removed hours ago joins again under its id: the store still
    holds its Agent records (ACTIVE, lease long lapsed) and its last report.
    Nothing is judged while the join is PENDING; once ACTIVE, the node the
    provider replaced meanwhile is a departure like any other."""

    store = build_store()
    world_ = world(store)
    nodes = [f"hyperpod-i-0b00000{index}" for index in (1, 2, 3)]
    returning, replaced = nodes[:2], nodes[2]
    for minute in range(15, -1, -1):
        world_.report(HP_CLUSTER, nodes, at=ago(hours=3, minutes=minute))
    for index, node in enumerate(nodes, start=1):
        store.save_agent(
            agent(
                HP_CLUSTER,
                node,
                instance_id=f"node-uid-{index}",
                last_seen=ago(hours=2, minutes=56),
                lease_seconds=90,
            )
        )
    world_.provider.hyperpod[HP_NAME] = {"i-0b000001", "i-0b000002"}
    _publish(
        store,
        registration(HP_CLUSTER, lifecycle=PENDING),
        reason=f"join {HP_CLUSTER} pending",
    )

    pending = world_.cleanup.run_once()
    # The join's node runtime roll: the returning nodes' Agents heartbeat again.
    for index, node in enumerate(returning, start=1):
        store.save_agent(
            agent(
                HP_CLUSTER,
                node,
                instance_id=f"node-uid-{index}",
                last_seen=ago(seconds=10),
            )
        )
    rolled = world_.cleanup.run_once()

    assert world_.notification_subjects() == [], "a join is no departure"
    assert pending == [] and rolled == [], [*pending, *rolled]
    assert store.list_departed_node_cleanups() == []
    assert world_.cleanup.refusals_total == Counter()

    _publish(
        store,
        registration(HP_CLUSTER, lifecycle=ACTIVE),
        reason=f"join {HP_CLUSTER} active",
    )
    world_.reports_since(HP_CLUSTER, returning, minutes=12)
    [record] = world_.cleanup.run_once()

    assert record.node_id == replaced
    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert world_.notification_subjects() == [
        f"[NOTICE][GPU node departed: records ended] {HP_CLUSTER}: {replaced}"
    ]
    assert store.get_agent(HP_CLUSTER, replaced).lifecycle_state is (
        AgentLifecycleState.REVOKED
    )
    assert [
        store.get_agent(HP_CLUSTER, node).lifecycle_state for node in returning
    ] == [AgentLifecycleState.ACTIVE] * 2
    assert world_.cleanup.run_once() == [], "the returning nodes are no candidates"


def _changed_node(store: Any, request_id: str) -> None:
    """The workflow restarted the device plugin before the node left."""

    store.save_workflow(
        store.get_workflow(request_id).model_copy(
            update={
                "completed_operations": [PLUGIN],
                "completed_step_indexes": [0],
                "step_executions": [
                    workflow_step_execution(0, PLUGIN, WorkflowStepStatus.SUCCEEDED)
                ],
            }
        )
    )


def _draining_sibling(store: Any, world_: World) -> list[str]:
    """The HyperPod cluster of the same site, mid-removal (the live shape)."""

    nodes = [f"hyperpod-i-0b00000{index}" for index in (1, 2, 3)]
    store.save_regional_cluster(registration(HP_CLUSTER, lifecycle=DRAINING))
    for minute in range(20, 7, -1):
        world_.report(HP_CLUSTER, nodes, at=ago(minutes=minute))
    for index, node in enumerate(nodes, start=1):
        store.save_agent(
            agent(
                HP_CLUSTER,
                node,
                instance_id=f"node-uid-{index}",
                last_seen=ago(minutes=3),
                lease_seconds=90,
            )
        )
    return nodes


@pytest.mark.parametrize("outcome", ["cleaned", "refused"])
def test_a_departure_on_an_active_sibling_is_notified_exactly_as_before(
    outcome: str,
) -> None:
    """A spot reclaim on the ACTIVE plain-EKS cluster while the HyperPod
    cluster is being removed: the reclaim gets the very notice it always got
    -- the informational one when everything ended, the operator one (once)
    when a workflow that changed the node is left -- and the draining
    cluster none."""

    store = build_store()
    world_ = eks_departure(store)
    _draining_sibling(store, world_)
    if outcome == "refused":
        _changed_node(store, "workflow-eks-efa")

    [record] = world_.cleanup.run_once()

    [notification_id] = world_.notifications.sent
    sent = store.get_notification(notification_id)
    if outcome == "cleaned":
        assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
        expected = cleaned_notification(record)
        assert store.get_incident("incident-eks-efa").state is (IncidentState.RECOVERED)
    else:
        assert record.status is DepartedNodeCleanupStatus.REFUSED
        assert record.blocker_classes == ["workflow_changed_node"]
        expected = refused_notification(record)
        assert store.get_workflow("workflow-eks-efa").status is (WorkflowStatus.BLOCKED)
    assert (record.cluster_id, record.node_id) == (EKS_CLUSTER, EKS_NODE)
    assert (
        sent.deduplication_key,
        sent.subject,
        sent.body_text,
        sent.category,
        sent.priority,
    ) == (
        expected.deduplication_key,
        expected.subject,
        expected.body_text,
        expected.category,
        expected.priority,
    )
    assert store.list_departed_node_cleanups(HP_CLUSTER) == []
    again = world_.cleanup.run_once()
    assert [item.status for item in again] == (
        [] if outcome == "cleaned" else [DepartedNodeCleanupStatus.REFUSED]
    )
    assert len(world_.notifications.sent) == 1, "said once"
    assert world_.cleanup.refusals_total == (
        Counter() if outcome == "cleaned" else Counter({"workflow_changed_node": 1})
    )


def test_a_line_from_before_the_drain_stays_as_it_was_until_the_cluster_leaves() -> (
    None
):
    """A reclaim judged while the cluster was ACTIVE keeps its line and its one
    notification through the removal -- not rewritten, not mailed again --
    and leaves with the registration (``_prune``)."""

    store = build_store()
    world_ = hyperpod_departure(store)
    _changed_node(store, "workflow-hp-efa")
    [refused] = world_.cleanup.run_once()
    assert refused.status is DepartedNodeCleanupStatus.REFUSED
    assert refused.blocker_classes == ["workflow_changed_node"]
    status_before = _status_lines(store)
    gauges_before = _open_nodes(store)

    _publish(
        store,
        registration(HP_CLUSTER, lifecycle=DRAINING),
        reason=f"remove {HP_CLUSTER} draining",
    )
    assert world_.cleanup.run_once() == []
    _publish(
        store,
        registration(HP_CLUSTER, lifecycle=REVOKED),
        reason=f"remove {HP_CLUSTER} revoked",
    )
    assert world_.cleanup.run_once() == []

    assert store.list_departed_node_cleanups(HP_CLUSTER) == [refused]
    assert _status_lines(store) == status_before
    assert any(
        "manual: gpu-fault-admin workflow-reconcile" in line for line in status_before
    ), status_before
    assert _open_nodes(store) == gauges_before
    assert gauges_before, "the open REFUSED line is counted"
    assert world_.notification_subjects() == [
        f"{OPERATOR_SUBJECT} {HP_CLUSTER}: {HP_NODE}"
    ]
    assert world_.cleanup.refusals_total == Counter({"workflow_changed_node": 1})

    _publish(store, reason=f"remove {HP_CLUSTER} purged")
    assert world_.cleanup.run_once() == []
    assert store.list_departed_node_cleanups() == [], "gone with the registration"
    assert len(world_.notifications.sent) == 1
