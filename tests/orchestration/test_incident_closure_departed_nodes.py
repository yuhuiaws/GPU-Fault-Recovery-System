"""A QUARANTINED incident whose node departed closes on the departure proof.

Its isolation was a cordon and a taint on a Node object; a node gone from
Kubernetes and from its provider took both with it, so "provably departed" --
the rule the departed-node cleanup and ``workflow-reconcile`` share
(``departed_nodes``) -- is evidence that the isolation is gone. The closure
service judges it itself: the departure must show both absences for that very
node, and the node's Agent record must not be alive. Anything less keeps the
refusal a node the cluster no longer has always got.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.departed_nodes import node_evidence
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.models import IncidentState, WorkflowEventCode
from gpu_fault.orchestration.incident_closure import (
    IncidentClosureService,
    IncidentNotClosable,
    NodeIsolationEvidence,
)
from tests._builders import build_store, copy_model
from tests.departed_nodes._support import agent
from tests.orchestration._incident_closure_support import OPERATOR, _escalated_reset

GONE = "hyperpod-i-0a000001"
PRESENT = "hyperpod-i-0a000002"
TARGET = {
    "cluster_id": "cluster-a",
    "hyperpod_cluster_name": "hp-a",
    "region": "us-west-2",
}


def _quarantined(store: Any, *node_ids: str) -> Any:
    incident, _workflow = _escalated_reset(
        store, incident_id="inc-q", node_ids=node_ids
    )
    quarantined = copy_model(incident, state=IncidentState.QUARANTINED)
    store.save_incident(quarantined, expected=incident)
    return quarantined


def _departure(node_id: str = GONE, **overrides: Any) -> dict[str, Any]:
    departed = {
        "node_id": node_id,
        "exists": False,
        "provider": "hyperpod",
        "instance_id": node_id.removeprefix("hyperpod-"),
        "absent_from_provider": True,
        "restored": True,
        "blockers": [],
        **overrides,
    }
    return node_evidence(node_id, departed=departed, target=TARGET, aws_region=None)


def _gone(node_id: str = GONE, **overrides: Any) -> NodeIsolationEvidence:
    return NodeIsolationEvidence(
        node_id=node_id, exists=False, departure=_departure(node_id, **overrides)
    )


def _ago(**delta: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**delta)


def test_a_departed_node_clears_the_quarantine_and_the_audit_keeps_the_proof() -> None:
    store = build_store()
    incident = _quarantined(store, GONE)
    service = IncidentClosureService(store)

    closed, changed = service.close_incident(
        incident.incident_id,
        reason="node reclaimed by spot",
        operator=OPERATOR,
        reference="CHG-1",
        evidence=[_gone()],
    )

    assert changed is True and closed.state is IncidentState.RECOVERED
    assert store.get_incident("inc-q").reasons[-1] == (
        f"isolation no longer present on node {GONE}: the node departed (gone "
        "from Kubernetes and HyperPod, instance i-0a000001)"
    )
    [event] = [
        item
        for item in store.get_workflow(incident.workflow_request_id).events
        if item.code == WorkflowEventCode.INCIDENT_CLOSED.value
    ]
    [recorded] = event.details["isolation_evidence"]
    assert recorded["exists"] is False
    assert recorded["departure"]["sources"]["hyperpod"]["listed"] is False


@pytest.mark.parametrize(
    "lifecycle", [AgentLifecycleState.REVOKED, AgentLifecycleState.ACTIVE]
)
def test_a_retired_or_long_silent_agent_does_not_hold_the_close(
    lifecycle: AgentLifecycleState,
) -> None:
    store = build_store()
    incident = _quarantined(store, GONE)
    store.save_agent(
        agent(
            incident.cluster_id,
            GONE,
            instance_id=None,
            last_seen=_ago(minutes=30),
            lifecycle=lifecycle,
        )
    )

    closed, _ = IncidentClosureService(store).close_incident(
        incident.incident_id, reason="x", operator=OPERATOR, evidence=[_gone()]
    )

    assert closed.state is IncidentState.RECOVERED


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        pytest.param(
            _gone(
                absent_from_provider=False,
                blockers=["node is missing from Kubernetes but HyperPod still lists"],
            ),
            "departure is unproven \\(node is missing from Kubernetes but HyperPod",
            id="provider-still-lists",
        ),
        pytest.param(
            NodeIsolationEvidence(node_id=GONE, exists=False),
            "not in the cluster; its isolation state cannot be verified$",
            id="no-departure",
        ),
        pytest.param(
            NodeIsolationEvidence(
                node_id=GONE, exists=False, departure=_departure(PRESENT)
            ),
            "departure is unproven",
            id="another-nodes-departure",
        ),
    ],
)
def test_a_departure_short_of_proof_keeps_the_refusal(
    evidence: NodeIsolationEvidence, match: str
) -> None:
    store = build_store()
    incident = _quarantined(store, GONE)

    with pytest.raises(IncidentNotClosable, match=match):
        IncidentClosureService(store).close_incident(
            incident.incident_id, reason="x", operator=OPERATOR, evidence=[evidence]
        )

    assert store.get_incident("inc-q").state is IncidentState.QUARANTINED


def test_an_agent_still_alive_is_no_departure() -> None:
    store = build_store()
    incident = _quarantined(store, GONE)
    store.save_agent(
        agent(incident.cluster_id, GONE, instance_id=None, last_seen=_ago(minutes=1))
    )

    with pytest.raises(IncidentNotClosable, match="its Agent is not departed"):
        IncidentClosureService(store).close_incident(
            incident.incident_id, reason="x", operator=OPERATOR, evidence=[_gone()]
        )

    assert store.get_incident("inc-q").state is IncidentState.QUARANTINED


def test_a_departed_node_and_a_released_present_node_close_together() -> None:
    store = build_store()
    incident = _quarantined(store, GONE, PRESENT)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable, match="still cordoned"):
        service.close_incident(
            incident.incident_id,
            reason="x",
            operator=OPERATOR,
            evidence=[_gone(), NodeIsolationEvidence(PRESENT, unschedulable=True)],
        )
    closed, _ = service.close_incident(
        incident.incident_id,
        reason="x",
        operator=OPERATOR,
        evidence=[_gone(), NodeIsolationEvidence(PRESENT)],
    )

    assert closed.state is IncidentState.RECOVERED
    assert store.get_incident("inc-q").reasons[-2:] == [
        f"isolation no longer present on node {GONE}: the node departed (gone "
        "from Kubernetes and HyperPod, instance i-0a000001)",
        f"isolation no longer present on node {PRESENT}",
    ]


def test_the_departure_travels_through_the_scripts_mapping() -> None:
    mapping = _gone().as_dict()

    assert NodeIsolationEvidence.from_mapping(mapping) == _gone()
    assert "departure" not in NodeIsolationEvidence(PRESENT).as_dict(), (
        "a present node's evidence keeps its historical shape"
    )
