"""The control plane, not the node, decides what a node-log event is (F-B7).

``evaluate_logs`` used to mint ``event_id = log-<entry_id>`` from the id the
collector sent. Two nodes whose collectors produced the same ``entry_id`` --
same training-log path and offset, same journal line without a cursor, or
simply an old collector -- therefore shared one event: the second node's fault
hit the first node's incident in ``_existing_or_grouped`` and returned it, so
the second node never got a workflow, a quarantine or a drain (P0-38A).
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import NodeHealthPolicy, NodeLogBatch, NodeLogEntry
from tests._builders import build_store

NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
SHARED_ENTRY_ID = "e3b0c44298fc1c149afbf4c8996fb924"


def _batch(node_id: str, *, cluster_id: str = "cluster-a") -> NodeLogBatch:
    return NodeLogBatch(
        batch_id=f"logs-{node_id}-1",
        cluster_id=cluster_id,
        node_id=node_id,
        collected_at=NOW,
        runtime_profile_version="simulated-v1",
        entries=[
            NodeLogEntry(
                entry_id=SHARED_ENTRY_ID,
                source="training-log",
                observed_at=NOW,
                message="NCCL WARN collective timeout error",
                fields={"path": "/var/log/train.log"},
            )
        ],
    )


def test_same_entry_id_on_two_nodes_yields_two_event_ids() -> None:
    policy = NodeHealthPolicy(build_store())

    node_a = policy.evaluate_logs(_batch("node-a"))
    node_b = policy.evaluate_logs(_batch("node-b"))

    assert len(node_a) == len(node_b) == 1
    assert node_a[0].event_id != node_b[0].event_id
    assert node_a[0].event_id.startswith("log-"), (
        'expected node_a[0].event_id.startswith("log-") to be true'
    )
    assert "node-a" in node_a[0].event_id
    assert "node-b" in node_b[0].event_id
    assert node_a[0].finding_id == f"finding-{node_a[0].event_id}"


def test_same_entry_id_in_two_clusters_yields_two_event_ids() -> None:
    policy = NodeHealthPolicy(build_store())

    first = policy.evaluate_logs(_batch("node-a", cluster_id="cluster-a"))
    second = policy.evaluate_logs(_batch("node-a", cluster_id="cluster-b"))

    assert first[0].event_id != second[0].event_id


def test_event_id_is_stable_for_the_same_node_and_entry() -> None:
    # A re-posted batch (lost ack, boundary entry read twice) must still
    # de-duplicate: identity is a function of (cluster, node, entry), nothing
    # else.
    policy = NodeHealthPolicy(build_store())

    first = policy.evaluate_logs(_batch("node-a"))
    second = policy.evaluate_logs(_batch("node-a"))

    assert first[0].event_id == second[0].event_id


def test_event_id_cannot_be_forged_by_separator_ambiguity() -> None:
    # "a-b" / "c" and "a" / "b-c" flatten to the same readable prefix; the
    # identity digest keeps them apart.
    policy = NodeHealthPolicy(build_store())

    first = policy.evaluate_logs(_batch("c", cluster_id="a-b"))
    second = policy.evaluate_logs(_batch("b-c", cluster_id="a"))

    assert first[0].event_id != second[0].event_id


def test_second_node_with_the_same_entry_id_gets_its_own_incident_and_workflow() -> (
    None
):
    context = ApplicationContext()
    policy = context.node_health

    node_a_finding = policy.evaluate_logs(_batch("node-a"))[0]
    node_b_finding = policy.evaluate_logs(_batch("node-b"))[0]
    incident_a, workflow_a = context.orchestrator.ingest_node_health(node_a_finding)
    incident_b, workflow_b = context.orchestrator.ingest_node_health(node_b_finding)

    assert incident_a.incident_id != incident_b.incident_id
    assert incident_a.node_ids == ["node-a"]
    assert incident_b.node_ids == ["node-b"]
    assert workflow_a is not None and workflow_b is not None
    assert workflow_a.request_id != workflow_b.request_id
    assert context.store.get_incident_by_event(node_b_finding.event_id) == incident_b
