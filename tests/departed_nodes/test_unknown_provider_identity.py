"""A node whose provider identity is unknown: the administrator's exit, and its rule.

No EC2 instance id recorded (eks-managed) or no instance in a HyperPod node
name: the provider can never be asked, so the departed-node cleanup can never
prove the departure and never retires such an Agent by itself. The
administrator may accept the departure -- ``--accept-unknown-provider-identity``
-- once every Kubernetes node report has left the node out for 24 hours or
never listed it, and the Agent is silent past the guard; the cleanup names
that command only when it would be taken, and on the acceptance ends the rest
of what the node left (live: DESTR-020's alias on gpu-eks-cluster).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.departed_node_cleanup_actions import accept_identity_command
from gpu_fault.departed_nodes import (
    DepartedNodeCleanupStatus,
    DepartedNodeIdentityAcceptance,
    node_evidence,
    retire_departed_agent,
)
from gpu_fault.fleet import AgentLifecycleState, FleetRegistry
from gpu_fault.kubernetes_node_inventory import (
    NODE_INVENTORY_ABSENCE_RETENTION,
    UNKNOWN_PROVIDER_ABSENCE,
    merge_node_inventory,
    unknown_provider_exit,
)
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.workflow_reconcile import close_never_dispatched_workflow
from tests._builders import build_store
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    EKS_PRESENT,
    World,
    agent,
    ago,
    blocked_never_changed,
    inventory_report,
    pending_never_dispatched,
    registration,
    world,
)

ALIAS = "acceptance-alias-0a0b0c0d0e0f"
GUARD = timedelta(minutes=10)


def _alias_world(store: Any, *, with_agent: bool = False) -> World:
    """DESTR-020's residue: an incident for a node that never existed."""

    world_ = world(store)
    store.save_regional_cluster(registration(EKS_CLUSTER, provider="eks-managed"))
    if with_agent:
        store.save_agent(
            agent(EKS_CLUSTER, ALIAS, instance_id=None, last_seen=ago(minutes=40))
        )
    blocked_never_changed(store, EKS_CLUSTER, ALIAS, suffix="alias")
    world_.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=15)
    return world_


def _acceptance(node_id: str = ALIAS) -> DepartedNodeIdentityAcceptance:
    return DepartedNodeIdentityAcceptance(
        cluster_id=EKS_CLUSTER,
        node_id=node_id,
        change_reference="CHG-1",
        actor="ops@example",
        reason="DESTR-020 alias; never a node",
        accepted_at=ago(seconds=5),
        kubernetes={"open": True, "never_reported": True},
    )


# ----------------------------------------------------------------- the rule


def _inventory(*reports: tuple[list[str], timedelta]) -> Any:
    record = None
    for names, age in reports:
        at = ago(seconds=age.total_seconds())
        record = merge_node_inventory(
            record, inventory_report(EKS_CLUSTER, names, observed_at=at), received_at=at
        )
    return record


def _chain(names_before: list[str], names_after: list[str], *, left: float) -> Any:
    """A one-a-minute chain over ``left`` + 5 minutes: ``names_before`` then
    ``names_after`` for the last ``left`` minutes."""

    reports = [
        (names_before, timedelta(minutes=minute))
        for minute in range(int(left) + 5, int(left), -1)
    ]
    reports += [
        (names_after, timedelta(minutes=minute)) for minute in range(int(left), -1, -1)
    ]
    return _inventory(*reports)


def test_a_node_no_report_ever_listed_is_open_once_the_chain_covers_the_guard() -> None:
    covered = _chain([EKS_PRESENT], [EKS_PRESENT], left=12)
    young = _chain([EKS_PRESENT], [EKS_PRESENT], left=3)

    verdict = unknown_provider_exit(covered, ALIAS, now=ago(seconds=0), guard=GUARD)
    early = unknown_provider_exit(young, ALIAS, now=ago(seconds=0), guard=GUARD)

    assert verdict.open and verdict.never_reported, verdict
    assert verdict.absent_since == covered.chain_started_at
    assert not early.open and "of the 600s guard" in str(early.blocker)


def test_a_node_a_report_listed_waits_out_its_24_hours() -> None:
    left_an_hour_ago = _chain([ALIAS, EKS_PRESENT], [EKS_PRESENT], left=60)

    verdict = unknown_provider_exit(
        left_an_hour_ago, ALIAS, now=ago(seconds=0), guard=GUARD
    )

    assert not verdict.open and not verdict.never_reported
    assert verdict.opens_at is not None
    assert verdict.opens_at - verdict.absent_since == UNKNOWN_PROVIDER_ABSENCE
    assert "accepted 24 hours after that" in str(verdict.blocker)


@pytest.mark.parametrize(
    ("inventory", "match"),
    [
        pytest.param(None, "no Kubernetes node report", id="never-reported"),
        pytest.param(
            _inventory(([EKS_PRESENT], timedelta(minutes=20))),
            "latest Kubernetes node report is",
            id="stale",
        ),
        pytest.param(
            _inventory(([ALIAS], timedelta(minutes=1))),
            "still in Kubernetes",
            id="present",
        ),
    ],
)
def test_no_fresh_report_or_a_listed_node_keeps_the_exit_closed(
    inventory: Any, match: str
) -> None:
    verdict = unknown_provider_exit(inventory, ALIAS, now=ago(seconds=0), guard=GUARD)

    assert not verdict.open and match in str(verdict.blocker), verdict.blocker


def test_an_absence_older_than_the_chains_memory_is_open_but_not_never() -> None:
    """Pruned at 24 hours, the absence moved the retention window past the
    chain's start: open, but no longer claimed as never reported."""

    record = _chain([ALIAS, EKS_PRESENT], [EKS_PRESENT], left=15)
    record = record.model_copy(
        update={
            "absent": {},
            "retained_since": record.chain_started_at
            + NODE_INVENTORY_ABSENCE_RETENTION / 24,
        }
    )

    verdict = unknown_provider_exit(record, ALIAS, now=ago(seconds=0), guard=GUARD)

    assert verdict.open and not verdict.never_reported and verdict.absent_since is None


def _evidence(node_id: str) -> dict[str, Any]:
    return node_evidence(
        node_id,
        departed={
            "node_id": node_id,
            "exists": False,
            "provider": "eks-managed",
            "instance_id": None,
            "absent_from_provider": False,
            "blockers": [],
        },
        target={"cluster_id": EKS_CLUSTER, "region": "us-west-2"},
        aws_region=None,
    )


def _retire(store: Any, node_id: str, **overrides: Any) -> Any:
    record = store.get_agent(EKS_CLUSTER, node_id)
    values: dict[str, Any] = {
        "cluster_id": EKS_CLUSTER,
        "node_id": node_id,
        "expected_generation": record.generation,
        "expected_lifecycle": record.lifecycle_state.value,
        "expected_last_seen_at": record.last_seen_at,
        "evidence": _evidence(node_id),
        "transition_id": "workflow-reconcile/CHG-1/0123456789abcdef",
        "reason": "accepted",
        "now": ago(seconds=0),
        "accept_unknown_provider": True,
        **overrides,
    }
    return retire_departed_agent(FleetRegistry(store, "s" * 32), store, **values)


def test_the_accepted_retirement_rejudges_the_reports_and_the_identity() -> None:
    store = build_store()
    _alias_world(store, with_agent=True)
    store.save_agent(
        agent(
            EKS_CLUSTER, "known-node", instance_id="i-0e000009", last_seen=ago(hours=1)
        )
    )

    with pytest.raises(ValueError, match="provider identity is known"):
        _retire(store, "known-node")
    with pytest.raises(ValueError, match="does not prove the node gone"):
        _retire(store, ALIAS, accept_unknown_provider=False)
    revoked = _retire(store, ALIAS)

    assert revoked.lifecycle_state is AgentLifecycleState.REVOKED
    assert revoked.transition_id == "workflow-reconcile/CHG-1/0123456789abcdef"


def test_the_accepted_retirement_refuses_while_the_reports_do() -> None:
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(EKS_CLUSTER, provider="eks-managed"))
    store.save_agent(
        agent(EKS_CLUSTER, ALIAS, instance_id=None, last_seen=ago(hours=2))
    )
    for minute in range(70, 60, -1):
        world_.report(EKS_CLUSTER, [ALIAS, EKS_PRESENT], at=ago(minutes=minute))
    world_.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=60)

    with pytest.raises(ValueError, match="accepted 24 hours after that"):
        _retire(store, ALIAS)

    assert store.get_agent(EKS_CLUSTER, ALIAS).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )


# -------------------------------------------------------------- the cleanup


def test_the_cleanup_names_the_accepting_command_only_when_it_would_be_taken() -> None:
    store = build_store()
    world_ = _alias_world(store, with_agent=True)

    [open_line] = world_.cleanup.run_once()

    assert open_line.status is DepartedNodeCleanupStatus.REFUSED
    assert open_line.blocker_classes == ["provider_identity_unknown"]
    assert open_line.admin_commands == [accept_identity_command(ALIAS)]
    assert store.get_agent(EKS_CLUSTER, ALIAS).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    ), "the cleanup never retires such an Agent by itself"

    # A heartbeat inside the guard: the exit would refuse, so it is not named.
    store.save_agent(
        agent(
            EKS_CLUSTER,
            ALIAS,
            instance_id=None,
            last_seen=ago(minutes=4),
            lease_seconds=60,
        )
    )
    [waiting] = world_.cleanup.run_once()

    assert waiting.admin_commands == [], waiting.admin_commands
    assert any("is not open yet" in item for item in waiting.blockers), waiting.blockers


def test_a_node_that_left_the_reports_an_hour_ago_names_no_command_yet() -> None:
    store = build_store()
    world_ = world(store)
    store.save_regional_cluster(registration(EKS_CLUSTER, provider="eks-managed"))
    store.save_agent(
        agent(EKS_CLUSTER, ALIAS, instance_id=None, last_seen=ago(hours=2))
    )
    for minute in range(70, 60, -1):
        world_.report(EKS_CLUSTER, [ALIAS, EKS_PRESENT], at=ago(minutes=minute))
    world_.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=60)

    [record] = world_.cleanup.run_once()

    assert record.blocker_classes == ["provider_identity_unknown"]
    assert record.admin_commands == []
    assert any("accepted 24 hours after that" in item for item in record.blockers), (
        record.blockers
    )


def test_the_administrators_acceptance_lets_the_cleanup_end_the_rest() -> None:
    """DESTR-020's alias: no Agent record, an ESCALATED incident and a
    never-dispatched record for a node no report ever listed."""

    store = build_store()
    world_ = _alias_world(store)
    pending_never_dispatched(store, EKS_CLUSTER, ALIAS, suffix="alias-xid")
    [before] = world_.cleanup.run_once()
    store.save_departed_node_identity_acceptance(_acceptance())

    [after] = world_.cleanup.run_once()

    assert before.admin_commands == [accept_identity_command(ALIAS)]
    assert after.status is DepartedNodeCleanupStatus.CLEANED, after.blockers
    for incident_id in ("incident-alias", "incident-alias-xid"):
        closed = store.get_incident(incident_id)
        assert closed.state is IncidentState.RECOVERED, incident_id
        assert any(
            "departure accepted by ops@example (CHG-1)" in item
            for item in closed.reasons
        ), closed.reasons
    assert store.get_workflow("workflow-alias-xid").status is (
        WorkflowStatus.SUPERSEDED
    )
    assert "administrator accepted the departure" in str(
        store.get_workflow("workflow-alias-xid").preemption_reason
    )
    assert after.evidence["provider_identity"]["accepted"]["actor"] == "ops@example"
    assert world_.provider.calls == [], "nothing is ever put to the provider"


def test_an_agent_active_again_after_the_acceptance_is_never_retired_here() -> None:
    store = build_store()
    world_ = _alias_world(store, with_agent=True)
    store.save_departed_node_identity_acceptance(_acceptance())

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED
    assert any(
        "accepted the departure (CHG-1) but the node's Agent is ACTIVE" in item
        for item in record.blockers
    ), record.blockers
    assert store.get_agent(EKS_CLUSTER, ALIAS).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )
    assert store.get_incident("incident-alias").state is IncidentState.ESCALATED


def test_a_quarantined_incident_on_an_accepted_node_names_no_command() -> None:
    store = build_store()
    world_ = _alias_world(store)
    incident = store.get_incident("incident-alias")
    store.save_incident(
        incident.model_copy(update={"state": IncidentState.QUARANTINED})
    )
    store.save_workflow(
        store.get_workflow("workflow-alias").model_copy(
            update={"status": WorkflowStatus.FAILED}
        )
    )
    store.save_departed_node_identity_acceptance(_acceptance())

    [record] = world_.cleanup.run_once()

    assert record.blocker_classes == ["incident_not_closable"]
    assert record.admin_commands == [], "no close clears it without the provider"
    assert store.get_incident("incident-alias").state is IncidentState.QUARANTINED


def test_a_line_with_nothing_left_open_is_dropped() -> None:
    """The administrator ended the alias's incident by hand: no Agent, no open
    incident -- the line has nothing to report and goes."""

    store = build_store()
    world_ = _alias_world(store)
    world_.cleanup.run_once()
    world_.closure.close_incident(
        "incident-alias", reason="alias; never a node", operator="ops@example"
    )

    assert world_.cleanup.run_once() == []
    assert store.list_departed_node_cleanups(EKS_CLUSTER) == []


def test_the_node_listed_again_forgets_the_acceptance() -> None:
    store = build_store()
    world_ = _alias_world(store)
    world_.cleanup.run_once()
    store.save_departed_node_identity_acceptance(_acceptance())
    world_.report(EKS_CLUSTER, [ALIAS, EKS_PRESENT], at=ago(seconds=0))

    world_.cleanup.run_once()

    assert store.list_departed_node_identity_acceptances(EKS_CLUSTER) == []
    assert store.list_departed_node_cleanups(EKS_CLUSTER) == []


def test_an_acceptance_the_store_does_not_hold_closes_nothing() -> None:
    """The evidence's word that the administrator accepted the departure is
    not enough: the never-dispatched close reads the acceptance itself."""

    store = build_store()
    store.save_regional_cluster(registration(EKS_CLUSTER, provider="eks-managed"))
    request_id, _incident_id = pending_never_dispatched(
        store, EKS_CLUSTER, ALIAS, suffix="alias-xid"
    )
    workflow = store.get_workflow(request_id)
    item = {
        "request_id": request_id,
        "fencing_token": workflow.fencing_token,
        "execution_epoch": workflow.execution_epoch,
        "workflow_created_at": workflow.created_at.isoformat(),
        "departed_node_evidence": [
            {
                "node_id": ALIAS,
                "provider": "eks-managed",
                "absent_from_kubernetes": True,
                "absent_from_provider": False,
                "provider_identity": {
                    "known": False,
                    "accepted": {"actor": "ops@example", "change_reference": "CHG-1"},
                },
            }
        ],
    }

    def close() -> Any:
        return close_never_dispatched_workflow(
            store,
            item,
            reference="CHG-1",
            actor="ops@example",
            admin_plan_sha256=None,
            now=ago(seconds=0),
        )

    with pytest.raises(ValueError, match="holds no acceptance for"):
        close()
    assert store.get_workflow(request_id).status is WorkflowStatus.PENDING

    store.save_departed_node_identity_acceptance(_acceptance())
    closed, _incident, _warnings = close()

    assert closed.status is WorkflowStatus.SUPERSEDED
