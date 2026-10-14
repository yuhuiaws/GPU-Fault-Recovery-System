"""The control plane cleans up after a departed GPU node by itself.

Live 2026-10-14: three spot reclaims in 30 minutes, on both providers, each
leaving an ``ACTIVE`` Agent with an expired lease (``status`` coverage drift,
release verification rollback), a BLOCKED never-changed workflow (release
preflight ``workflow_safety`` refusal) and an ESCALATED incident -- and each
needed three hand-run ``workflow-reconcile`` commands. These tests run the
cleanup on the in-memory store: a reclaimed node is cleaned within one cycle
after the guard with no admin command, and every refusal the brief lists keeps
the records as they are and says why.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.departed_nodes import (
    GUARD_REASON,
    DepartedNodeCleanupStatus,
    DepartedNodeSpareRelease,
)
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RegionalClusterLifecycle
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release.regional_admin_checks_coverage import check_agent_coverage
from gpu_fault_release.regional_release_config import ReleaseError
from tests._builders import build_store, workflow_step_execution
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    EKS_INSTANCE,
    EKS_NODE,
    EKS_PRESENT,
    HP_CLUSTER,
    HP_INSTANCE,
    HP_NAME,
    HP_NODE,
    HP_PRESENT,
    PLUGIN,
    World,
    agent,
    ago,
    eks_departure,
    hyperpod_departure,
    pending_never_dispatched,
    registration,
    world,
)


def _record(world: World, cluster_id: str, node_id: str) -> Any:
    [record] = [
        item
        for item in world.store.list_departed_node_cleanups(cluster_id)
        if item.node_id == node_id
    ]
    return record


def _workflow_safety(monkeypatch: pytest.MonkeyPatch, store: Any) -> dict[str, Any]:
    """The release preflight's ``workflow_safety`` probe, run on ``store``."""

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


def _coverage(world: World, cluster_id: str, present: list[str]) -> None:
    """The ``control_api`` Agent coverage check of ``status`` / verification."""

    check_agent_coverage(
        cluster_id,
        {
            "expected_node_ids": present,
            "kubernetes_node_ids": present,
            "agents": [
                item.model_dump(mode="json")
                for item in world.store.list_agents(cluster_id)
            ],
        },
    )


@pytest.mark.parametrize("provider", ["hyperpod", "eks-managed"])
def test_a_reclaimed_node_is_cleaned_within_one_cycle_after_the_guard(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No admin command: the Agent retired, the never-changed workflows closed,
    the incidents RECOVERED -- and ``status``' coverage check and the release
    preflight's ``workflow_safety`` are clean afterwards."""

    store = build_store()
    if provider == "hyperpod":
        world_ = hyperpod_departure(store)
        cluster_id, node_id, present = HP_CLUSTER, HP_NODE, HP_PRESENT
        closed = ["workflow-hp-efa", "workflow-hp-xid"]
        incidents = ["incident-hp-efa", "incident-hp-xid"]
    else:
        world_ = eks_departure(store)
        cluster_id, node_id, present = EKS_CLUSTER, EKS_NODE, EKS_PRESENT
        closed = ["workflow-eks-efa"]
        incidents = ["incident-eks-efa"]
    before = _workflow_safety(monkeypatch, store)
    assert before["blocker_count"] == len(closed), before
    with pytest.raises(ReleaseError, match="Agent coverage drift"):
        _coverage(world_, cluster_id, [present])

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert record.blockers == [] and record.admin_commands == []
    assert record.reference and record.reference.startswith("departed-node-cleanup-")
    retired = store.get_agent(cluster_id, node_id)
    assert retired.lifecycle_state is AgentLifecycleState.REVOKED
    assert retired.generation == 5, "one drain bumped the generation once"
    assert retired.transition_id == f"departed-node-cleanup/{record.reference}"
    reason = retired.transition_reason or ""
    assert reason.startswith(
        f"operator reconciliation {record.reference}: retired Agent {node_id}"
    ), reason
    assert "actor departed-node-cleanup" in reason, reason
    label = "HyperPod" if provider == "hyperpod" else "EC2"
    assert f"after its node left Kubernetes and {label}" in reason, reason
    for request_id in closed:
        workflow = store.get_workflow(request_id)
        assert workflow.status is WorkflowStatus.SUPERSEDED, request_id
        event = workflow.events[-1]
        assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
        assert event.actor == "departed-node-cleanup"
        assert event.details["reference"] == record.reference
    for incident_id in incidents:
        incident = store.get_incident(incident_id)
        assert incident.state is IncidentState.RECOVERED, incident_id
        assert any("node departed:" in item for item in incident.reasons), (
            incident.reasons
        )
    assert store.get_agent(cluster_id, present).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    ), "the node still in Kubernetes is never touched"
    assert world_.notification_subjects() == [
        f"[NOTICE][GPU node departed: records ended] {cluster_id}: {node_id}"
    ]
    assert len(world_.provider.calls) == 1, "one provider read per cluster per cycle"
    after = _workflow_safety(monkeypatch, store)
    assert after["blocker_count"] == 0, after
    _coverage(world_, cluster_id, [present])
    assert world_.cleanup.actions_total["agent_retired"] == 1
    assert world_.cleanup.actions_total["workflow_closed"] == len(closed)
    assert world_.cleanup.actions_total["incident_closed"] == len(incidents)


def test_an_eks_instance_ec2_no_longer_knows_is_departed_too() -> None:
    store = build_store()
    world_ = eks_departure(store)
    del world_.provider.ec2[EKS_INSTANCE]

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert record.evidence["sources"]["ec2"]["state"] == "not-found"
    assert world_.provider.calls == [("ec2", (EKS_INSTANCE,))]


def test_a_second_cycle_finds_nothing_left_and_writes_nothing() -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    world_.cleanup.run_once()

    assert world_.cleanup.run_once() == [], "a cleaned node is no candidate"
    assert len(world_.notifications.sent) == 1
    assert world_.cleanup.actions_total["agent_retired"] == 1


def test_a_node_still_in_kubernetes_is_never_a_candidate() -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    world_.reports_since(HP_CLUSTER, [HP_PRESENT, HP_NODE], minutes=12)

    assert world_.cleanup.run_once() == []
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )
    assert store.get_workflow("workflow-hp-efa").status is WorkflowStatus.BLOCKED
    assert world_.provider.calls == [], "no provider read for a node in sight"


@pytest.mark.parametrize(
    ("provider", "listed"),
    [("hyperpod", True), ("eks-managed", "running"), ("eks-managed", "stopped")],
)
def test_a_provider_that_still_has_the_instance_keeps_everything(
    provider: str, listed: Any
) -> None:
    store = build_store()
    if provider == "hyperpod":
        world_ = hyperpod_departure(store)
        world_.provider.hyperpod[HP_NAME].add(HP_INSTANCE)
        cluster_id, node_id = HP_CLUSTER, HP_NODE
    else:
        world_ = eks_departure(store)
        world_.provider.ec2[EKS_INSTANCE] = str(listed)
        cluster_id, node_id = EKS_CLUSTER, EKS_NODE

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.PENDING
    assert record.blocker_classes == ["provider_still_lists"]
    assert record.actions == [] and record.reference is None
    assert store.get_agent(cluster_id, node_id).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )
    assert world_.notifications.sent == [], "a pending node pages nobody"


def test_a_failed_lookup_proves_nothing_and_the_next_cycle_retries() -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    world_.provider.failure = (
        "cannot list HyperPod nodes of gpu-hyperpod-hp: ThrottlingException"
    )

    [pending] = world_.cleanup.run_once()

    assert pending.status is DepartedNodeCleanupStatus.PENDING
    assert pending.blocker_classes == ["provider_unknown"]
    assert "ThrottlingException" in pending.blockers[0]
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )

    world_.provider.failure = None
    [cleaned] = world_.cleanup.run_once()

    assert cleaned.status is DepartedNodeCleanupStatus.CLEANED
    assert cleaned.first_seen_at == pending.first_seen_at


def test_a_young_heartbeat_or_a_live_lease_keeps_the_agent() -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=4))
    )

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.PENDING
    assert "agent_alive" in record.blocker_classes
    assert GUARD_REASON in record.blockers
    assert world_.provider.calls == [], "the provider is read only for a full case"
    assert store.get_workflow("workflow-hp-efa").status is WorkflowStatus.BLOCKED


def test_an_absence_shorter_than_the_guard_or_a_stale_report_proves_nothing() -> None:
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=30))
    )
    # Listed until five minutes ago, absent since.
    for minute in range(20, 5, -1):
        world_.report(HP_CLUSTER, [HP_NODE, HP_PRESENT], at=ago(minutes=minute))
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=5)

    [short] = world_.cleanup.run_once()

    assert short.status is DepartedNodeCleanupStatus.PENDING
    assert short.blocker_classes == ["kubernetes_unproven"]
    assert "of the 600s guard" in short.blockers[0]

    stale = build_store()
    stale_world = world(stale)
    stale.save_regional_cluster(registration(HP_CLUSTER))
    stale.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=40))
    )
    for minute in range(40, 9, -1):
        stale_world.report(HP_CLUSTER, [HP_PRESENT], at=ago(minutes=minute))

    [old] = stale_world.cleanup.run_once()

    assert old.status is DepartedNodeCleanupStatus.PENDING
    assert old.blocker_classes == ["kubernetes_unproven"]
    assert "Kubernetes node report is" in old.blockers[0]
    assert stale.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )


def test_no_report_at_all_is_reported_never_acted_on() -> None:
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=40))
    )

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.PENDING
    assert record.blocker_classes == ["kubernetes_unproven"]
    assert "no Kubernetes node report" in record.blockers[0]


def test_a_workflow_that_changed_a_node_is_left_to_the_operator() -> None:
    """The Agent is retired (it is the coverage drift), the workflow that
    restarted the device plugin is not closed, its incident stays open, and
    the notification names the exact reconcile command."""

    store = build_store()
    world_ = eks_departure(store)
    changed = store.get_workflow("workflow-eks-efa").model_copy(
        update={
            "completed_operations": [PLUGIN],
            "completed_step_indexes": [0],
            "step_executions": [
                workflow_step_execution(0, PLUGIN, WorkflowStepStatus.SUCCEEDED)
            ],
        }
    )
    store.save_workflow(changed)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert record.blocker_classes == ["workflow_changed_node"]
    assert record.actions == [f"agent-retired:{EKS_NODE}"]
    assert any(
        "--workflow-id workflow-eks-efa --dry-run" in item
        for item in record.admin_commands
    ), record.admin_commands
    assert store.get_workflow("workflow-eks-efa").status is WorkflowStatus.BLOCKED
    assert store.get_incident("incident-eks-efa").state is IncidentState.ESCALATED
    assert store.get_agent(EKS_CLUSTER, EKS_NODE).lifecycle_state is (
        AgentLifecycleState.REVOKED
    )
    assert world_.notification_subjects() == [
        f"[GPU node departed: operator action needed] {EKS_CLUSTER}: {EKS_NODE}"
    ]
    body = store.get_notification(world_.notifications.sent[0]).body_text
    assert "--workflow-id workflow-eks-efa" in body
    assert world_.cleanup.refusals_total["workflow_changed_node"] == 1
    world_.cleanup.run_once()
    assert len(world_.notifications.sent) == 1, "the same refusal is said once"
    assert world_.cleanup.refusals_total["workflow_changed_node"] == 1


def test_an_open_remote_command_keeps_the_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    real = store.list_remote_commands

    def with_open_command(**kwargs: Any) -> list[Any]:
        commands = list(real(**kwargs))
        if "workflow-hp-efa" in (kwargs.get("workflow_request_ids") or []):
            from types import SimpleNamespace

            commands.append(
                SimpleNamespace(
                    command_id="command-1",
                    workflow_request_id="workflow-hp-efa",
                    status=RemoteCommandStatus.LEASED,
                )
            )
        return commands

    monkeypatch.setattr(store, "list_remote_commands", with_open_command)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert "remote_command" in record.blocker_classes
    assert store.get_workflow("workflow-hp-efa").status is WorkflowStatus.BLOCKED
    assert store.get_incident("incident-hp-efa").state is IncidentState.ESCALATED
    assert store.get_workflow("workflow-hp-xid").status is (
        WorkflowStatus.SUPERSEDED
    ), "the other incident's never-dispatched record is still closed"


@pytest.mark.parametrize(
    "lifecycle", [RegionalClusterLifecycle.DRAINING, RegionalClusterLifecycle.PENDING]
)
def test_a_cluster_that_is_not_active_is_reported_not_acted_on(
    lifecycle: RegionalClusterLifecycle,
) -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    store.save_regional_cluster(registration(HP_CLUSTER, lifecycle=lifecycle))

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert "cluster_not_active" in record.blocker_classes
    assert f"is {lifecycle.value}" in record.blockers[0]
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )
    assert world_.provider.calls == []


def _quarantine(store: Any, incident_id: str, workflow_id: str) -> None:
    """The node was quarantined and the quarantining workflow ended."""

    incident = store.get_incident(incident_id)
    store.save_incident(
        incident.model_copy(update={"state": IncidentState.QUARANTINED})
    )
    store.save_workflow(
        store.get_workflow(workflow_id).model_copy(
            update={"status": WorkflowStatus.FAILED}
        )
    )


def test_a_quarantined_incident_of_a_departed_node_closes_on_the_departure() -> None:
    """The node that carried the quarantine is gone, and the isolation with it:
    the departure proof is the isolation evidence the closure service needs."""

    store = build_store()
    world_ = hyperpod_departure(store)
    _quarantine(store, "incident-hp-efa", "workflow-hp-efa")

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert "incident-closed:incident-hp-efa" in record.actions
    closed = store.get_incident("incident-hp-efa")
    assert closed.state is IncidentState.RECOVERED
    assert any("node departed:" in item for item in closed.reasons), closed.reasons
    assert (
        f"isolation no longer present on node {HP_NODE}: the node departed "
        f"(gone from Kubernetes and HyperPod, instance {HP_INSTANCE})"
    ) in closed.reasons
    [event] = [
        item
        for item in store.get_workflow("workflow-hp-efa").events
        if item.details.get("previous_incident_state") == "QUARANTINED"
    ]
    assert event.actor == "departed-node-cleanup"
    [evidence] = event.details["isolation_evidence"]
    assert evidence["exists"] is False
    assert evidence["departure"]["absent_from_provider"] is True, (
        "the audit keeps the departure the close rested on"
    )


def test_a_quarantined_incident_that_also_names_a_present_node_keeps_its_refusal() -> (
    None
):
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=30))
    )
    world_.provider.hyperpod[HP_NAME] = {"i-0a000002"}
    pending_never_dispatched(
        store, HP_CLUSTER, HP_NODE, suffix="pair", node_ids=[HP_NODE, HP_PRESENT]
    )
    _quarantine(store, "incident-pair", "workflow-pair")
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=12)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert record.blocker_classes == ["incident_names_other_nodes"]
    assert store.get_incident("incident-pair").state is IncidentState.QUARANTINED


def _spare_departure(store: Any) -> World:
    """``HP_NODE`` was a declared warm spare when Kubernetes last listed it."""

    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=30))
    )
    for minute in range(25, 14, -1):
        world_.report(
            HP_CLUSTER, [HP_NODE, HP_PRESENT], at=ago(minutes=minute), spares=[HP_NODE]
        )
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=14)
    return world_


def _spare_release(instance_id: str = HP_INSTANCE) -> DepartedNodeSpareRelease:
    return DepartedNodeSpareRelease(
        cluster_id=HP_CLUSTER,
        node_id=HP_NODE,
        instance_id=instance_id,
        change_reference="CHG-1",
        actor="ops@example",
        released_at=ago(seconds=1),
        declaration="released",
    )


def test_a_departed_spare_is_cleaned_and_its_declaration_reported() -> None:
    store = build_store()
    world_ = _spare_departure(store)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert record.blocker_classes == ["declared_spare"]
    assert any(
        "config spare" in item and "--release" in item for item in record.admin_commands
    ), record.admin_commands
    assert record.actions == [f"agent-retired:{HP_NODE}"], (
        "the spare's own records end like any departed node's"
    )
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.REVOKED
    )


def test_the_administrators_release_answers_the_spare_finding() -> None:
    store = build_store()
    world_ = _spare_departure(store)
    world_.cleanup.run_once()

    store.save_departed_node_spare_release(_spare_release("i-0a00000f"))
    [other_instance] = world_.cleanup.run_once()
    store.save_departed_node_spare_release(_spare_release())
    [released] = world_.cleanup.run_once()

    assert other_instance.blocker_classes == ["declared_spare"], (
        "a release of another instance answers another departure"
    )
    assert released.status is DepartedNodeCleanupStatus.CLEANED, released.blockers
    assert world_.notification_subjects()[-1] == (
        f"[NOTICE][GPU node departed: records ended] {HP_CLUSTER}: {HP_NODE}"
    )


def test_the_spare_finding_outlives_the_inventorys_memory_of_the_label() -> None:
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    start = ago(hours=25)
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=start)
    )
    world_.report(HP_CLUSTER, [HP_NODE, HP_PRESENT], at=start, spares=[HP_NODE])
    for minute in range(1, 31):
        world_.report(HP_CLUSTER, [HP_PRESENT], at=start + timedelta(minutes=minute))
    world_.cleanup.now = lambda: start + timedelta(minutes=30)
    [first] = world_.cleanup.run_once()
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=25 * 60 - 31)
    world_.cleanup.now = lambda: ago(seconds=0)

    [second] = world_.cleanup.run_once()

    assert first.blocker_classes == ["declared_spare"]
    assert second.blocker_classes == ["declared_spare"], (
        "the inventory forgot the label after 24 hours; the finding did not"
    )


def test_an_incident_that_also_names_a_present_node_is_left_alone() -> None:
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=30))
    )
    world_.provider.hyperpod[HP_NAME] = {"i-0a000002"}
    pending_never_dispatched(
        store, HP_CLUSTER, HP_NODE, suffix="pair", node_ids=[HP_NODE, HP_PRESENT]
    )
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=12)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert record.blocker_classes == ["incident_names_other_nodes"]
    assert store.get_workflow("workflow-pair").status is WorkflowStatus.PENDING
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.REVOKED
    )


def test_a_restart_between_the_drain_and_the_revoke_resumes_without_a_second_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    calls = {"revoke": 0}
    real_revoke = world_.registry.revoke_agent

    def crash_once(*args: Any, **kwargs: Any) -> Any:
        calls["revoke"] += 1
        if calls["revoke"] == 1:
            raise ValueError("control-worker restarted mid-cleanup")
        return real_revoke(*args, **kwargs)

    monkeypatch.setattr(world_.registry, "revoke_agent", crash_once)

    [first] = world_.cleanup.run_once()

    drained = store.get_agent(HP_CLUSTER, HP_NODE)
    assert drained.lifecycle_state is AgentLifecycleState.DRAINING
    assert first.status is DepartedNodeCleanupStatus.PENDING
    assert first.blocker_classes == ["record_moved"]

    [second] = world_.cleanup.run_once()

    revoked = store.get_agent(HP_CLUSTER, HP_NODE)
    assert second.status is DepartedNodeCleanupStatus.CLEANED, second.blockers
    assert revoked.lifecycle_state is AgentLifecycleState.REVOKED
    assert revoked.generation == drained.generation, "no second drain"
    assert revoked.transition_id == drained.transition_id
    assert second.reference == first.reference
    assert len(world_.notifications.sent) == 1
    assert world_.cleanup.actions_total["workflow_closed"] == 2, "closed once each"


def test_a_cleanup_interrupted_after_some_writes_resumes_where_it_stopped() -> None:
    """A fresh process (a restart, another replica) with the same store: what
    was ended stays ended, the rest is ended, nothing twice."""

    store = build_store()
    first = hyperpod_departure(store)
    store.save_workflow(
        store.get_workflow("workflow-hp-xid").model_copy(
            update={"execution_owner_id": "dispatcher-1"}
        )
    )
    [partial] = first.cleanup.run_once()
    assert partial.status is DepartedNodeCleanupStatus.PENDING
    assert store.get_workflow("workflow-hp-efa").status is WorkflowStatus.SUPERSEDED
    store.save_workflow(
        store.get_workflow("workflow-hp-xid").model_copy(
            update={"execution_owner_id": None}
        )
    )

    second = world(store)
    second.provider.hyperpod = first.provider.hyperpod
    [done] = second.cleanup.run_once()

    assert done.status is DepartedNodeCleanupStatus.CLEANED, done.blockers
    assert done.reference == partial.reference
    assert second.cleanup.actions_total["agent_retired"] == 0, "retired by the first"
    assert second.cleanup.actions_total["workflow_closed"] == 1
    assert [item for item in done.actions if item.startswith("workflow-closed")] == [
        "workflow-closed:workflow-hp-efa",
        "workflow-closed:workflow-hp-xid",
    ]


def test_the_switch_off_writes_nothing() -> None:
    store = build_store()
    world_ = hyperpod_departure(store, enabled=False)

    assert world_.cleanup.run_once() == []
    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )
    assert store.list_departed_node_cleanups() == []
    assert "gpu_fault_departed_node_cleanup_enabled 0" in world_.cleanup.metric_lines()


def test_a_pending_node_that_comes_back_is_forgotten() -> None:
    store = build_store()
    world_ = hyperpod_departure(store)
    world_.provider.failure = "ThrottlingException"
    world_.cleanup.run_once()
    assert (
        _record(world_, HP_CLUSTER, HP_NODE).status is DepartedNodeCleanupStatus.PENDING
    )

    world_.report(HP_CLUSTER, [HP_PRESENT, HP_NODE], at=ago(seconds=0))
    world_.cleanup.run_once()

    assert store.list_departed_node_cleanups(HP_CLUSTER) == []


def test_a_gap_in_the_reports_restarts_the_absence() -> None:
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=60))
    )
    for minute in range(60, 40, -1):
        world_.report(HP_CLUSTER, [HP_PRESENT], at=ago(minutes=minute))
    # Twenty silent minutes: the executor was down; the node could have come back.
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=4)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.PENDING
    assert record.blocker_classes == ["kubernetes_unproven"]


def test_a_long_refusal_keeps_its_reference_when_the_absence_window_moves() -> None:
    """After 24 hours the inventory forgets when an absence began and counts
    from its retention window instead; the departure is still the same one
    (the instance is gone for good), so its reference and the actions the
    ledger line already records must not change with that window."""

    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    start = ago(hours=25)
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=start)
    )
    world_.provider.hyperpod[HP_NAME] = {"i-0a000002"}
    pending_never_dispatched(
        store, HP_CLUSTER, HP_NODE, suffix="pair", node_ids=[HP_NODE, HP_PRESENT]
    )
    world_.report(HP_CLUSTER, [HP_NODE, HP_PRESENT], at=start)
    for minute in range(1, 31):
        world_.report(HP_CLUSTER, [HP_PRESENT], at=start + timedelta(minutes=minute))
    world_.cleanup.now = lambda: start + timedelta(minutes=30)
    [first] = world_.cleanup.run_once()
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=25 * 60 - 31)
    world_.cleanup.now = lambda: ago(seconds=0)

    [second] = world_.cleanup.run_once()

    assert first.status is DepartedNodeCleanupStatus.REFUSED, first.blockers
    assert first.actions == [f"agent-retired:{HP_NODE}"]
    assert second.absent_since != first.absent_since, (
        "the fixture must move the absence window"
    )
    assert second.reference == first.reference
    assert second.actions == first.actions


def test_records_of_a_removed_cluster_and_old_cleaned_lines_are_pruned() -> None:
    store = build_store()
    world_ = hyperpod_departure(store, cleaned_retention=timedelta(seconds=0))
    world_.cleanup.run_once()
    world_.cleanup.run_once()
    assert store.list_departed_node_cleanups() == []

    pending = eks_departure(store)
    pending.provider.failure = "AccessDenied"
    pending.cleanup.run_once()
    assert store.list_departed_node_cleanups(EKS_CLUSTER), "a pending line exists"
    store.delete_regional_cluster(EKS_CLUSTER)
    pending.cleanup.run_once()
    assert store.list_departed_node_cleanups(EKS_CLUSTER) == []


def test_an_eks_agent_without_a_recorded_instance_is_never_guessed() -> None:
    """No EC2 id, no question to ask EC2: the operator's, with the manual
    command -- it can never resolve on its own."""

    store = build_store()
    world_ = eks_departure(store)
    store.save_agent(
        agent(
            EKS_CLUSTER, EKS_NODE, instance_id="0b9b4c2e-uid", last_seen=ago(minutes=20)
        )
    )

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert record.blocker_classes == ["provider_identity_unknown"]
    assert any("--retire-departed-agents" in item for item in record.admin_commands), (
        record.admin_commands
    )
    assert world_.provider.calls == []


def test_a_node_whose_agent_holds_its_lease_is_never_a_candidate() -> None:
    """Whatever a report says, a heartbeating Agent is a live node: a node
    that joined after the latest report, or a report that went stale, must
    not put a healthy node in the ledger."""

    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(seconds=10))
    )
    pending_never_dispatched(store, HP_CLUSTER, HP_NODE, suffix="joined")
    world_.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=12)

    assert world_.cleanup.run_once() == []

    stale = build_store()
    stale_world = world(stale)
    stale.save_regional_cluster(registration(HP_CLUSTER))
    stale.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(seconds=10))
    )
    for minute in range(40, 20, -1):
        stale_world.report(HP_CLUSTER, [HP_PRESENT], at=ago(minutes=minute))

    assert stale_world.cleanup.run_once() == []
    assert stale.list_departed_node_cleanups() == []
