"""The node cleanup of a GPU cluster without nodes (``cleanup_zero_nodes``).

``uninstall`` and ``remove-cluster`` refused a cluster whose nodes had all left
with "node cleanup has no proven fleet targets": the fleet still held the
records of its departed nodes, and no listed node was a target. The empty
target set is now proven when the cluster's selector selects no node (the
release engine's proof), every record is retired or proven departed, and no
node carries anything of ours -- and nothing else changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import TOOLS
from tests.regional.test_cleanup_kubernetes import CONTEXT, Client, document, resource

KUBE = lazy_script_module(TOOLS / "cleanup_kubernetes.py")
ZERO = lazy_script_module(TOOLS / "cleanup_zero_nodes.py")

SELECTOR = "gpu-fault.io/control-plane=site-a"
CONFIG = {
    "aws_region": "us-west-2",
    "gpu_kubeconfig": "/secure/gpu.kubeconfig",
    "clusters": [
        {
            "cluster_id": "gpu-a",
            "context": "gpu-a",
            "provider": "eks-managed",
            "node_selector": SELECTOR,
            "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a",
        }
    ],
}
GROUP = {
    "name": "gpu-group",
    "status": "ACTIVE",
    "desired": 0,
    "min": 0,
    "max": 2,
    "instance_types": ["p5en.48xlarge"],
    "labels": {"gpu-fault.io/control-plane": "site-a"},
}
PARKED = {
    "parked": True,
    "provider": "eks-managed",
    "reason": "every managed node group the node selector covers is ACTIVE at "
    "desiredSize 0 and no Karpenter NodePool can launch a node it selects",
    "groups": [GROUP],
}
UNAVAILABLE = {
    "parked": False,
    "provider": "eks-managed",
    "reason": "node group gpu-group wants 2 node(s) (desiredSize 2): its nodes "
    "are lost or still joining",
    "groups": [{**GROUP, "desired": 2}],
}


def record(node: str, state: str) -> dict[str, Any]:
    return {
        "cluster_id": "gpu-a",
        "node_id": node,
        "node_instance_id": "i-00000000000000001",
        "lifecycle_state": state,
        "installed_unit_inventory": {"units": ["gpu-fault-node-agent.service"]},
    }


def zero_node_client() -> Client:
    """The cluster's GPU nodes are gone; a CPU node without our metadata stays."""

    client = Client()
    client.nodes = [resource("Node", "cpu-node", namespace=None)]
    return client


@pytest.fixture
def proofs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The proof the cleanup reads and the departures it proves, stubbed."""

    seen: dict[str, Any] = {"read": [], "proved": [], "proof": PARKED}

    def read(config: dict[str, Any], member: dict[str, Any]) -> Any:
        seen["read"].append((config, member))
        return seen["proof"]

    def departed(
        _config: Any, _document: Any, _context: str, agent: dict[str, Any]
    ) -> dict[str, Any]:
        seen["proved"].append(agent["node_id"])
        if agent["node_id"].startswith("running"):
            raise KUBE.CleanupStateError(
                f"an active fleet node is missing: {agent['node_id']}; EC2 "
                "instance i-00000000000000001 is running"
            )
        return {
            "cluster_id": "gpu-a",
            "node_id": agent["node_id"],
            "node_instance_id": agent["node_instance_id"],
            "provider": "eks-managed",
            "ec2_state": "terminated",
        }

    monkeypatch.setattr(ZERO, "read_proof", read)
    monkeypatch.setattr(KUBE, "departed_fleet_node", departed)
    return seen


@pytest.mark.parametrize(
    ("proof", "state"), [(PARKED, "parked"), (UNAVAILABLE, "unavailable")]
)
def test_a_cluster_without_nodes_is_an_empty_proven_target_set(
    proofs: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    proof: dict[str, Any],
    state: str,
) -> None:
    """Retired records, a departed ACTIVE one and a departed DRAINING one, on a
    cluster whose selector selects no node: nothing to clean, said and kept."""

    proofs["proof"] = proof
    client, journal = zero_node_client(), document()
    journal["fleet_snapshot"] = [
        record("retired-a", "REVOKED"),
        record("retired-b", "REVOKED"),
        record("left-active", "ACTIVE"),
        record("left-draining", "DRAINING"),
    ]

    targets = KUBE.node_targets(client, journal, CONTEXT, "gpu-a", config=CONFIG)

    assert targets == {}, "no node of a cluster without nodes is a target"
    assert journal["node_targets"][CONTEXT] == {}, "the empty set is recorded"
    entry = journal["zero_node_clusters"]["gpu-a"]
    assert entry == {
        "state": state,
        "provider": "eks-managed",
        "reason": proof["reason"],
        "groups": proof["groups"],
        "retired_agents": ["retired-a", "retired-b"],
        "departed_agents": ["left-active", "left-draining"],
    }, entry
    assert proofs["proved"] == ["left-active", "left-draining"], (
        "every record not yet retired is proven departed, DRAINING included"
    )
    assert [item["node_id"] for item in journal["departed_fleet_nodes"]] == [
        "left-active",
        "left-draining",
    ], "both departures are journaled like any other"
    assert proofs["read"] == [(CONFIG, CONFIG["clusters"][0])], (
        "the proof is read once, of the bound config entry"
    )
    narration = capsys.readouterr().err
    assert "zero-node-gpu-cluster" in narration and "step=node-cleanup" in narration, (
        narration
    )
    assert f"state={state}" in narration and "retired=2 departed=2" in narration, (
        narration
    )


def test_a_selected_or_unreadable_node_list_keeps_the_refusal(
    proofs: dict[str, Any],
) -> None:
    proofs["proof"] = None
    client, journal = zero_node_client(), document()
    journal["fleet_snapshot"] = [record("retired-a", "REVOKED")]

    with pytest.raises(KUBE.CleanupStateError, match="no proven fleet targets"):
        KUBE.node_targets(client, journal, CONTEXT, "gpu-a", config=CONFIG)
    assert "zero_node_clusters" not in journal, "nothing is recorded on a refusal"


def test_a_record_whose_instance_the_provider_still_has_keeps_the_refusal(
    proofs: dict[str, Any],
) -> None:
    """A DRAINING record whose instance EC2 still runs: the node runtime could
    still be there, outside Kubernetes, so the cleanup is not proven."""

    client, journal = zero_node_client(), document()
    journal["fleet_snapshot"] = [
        record("retired-a", "REVOKED"),
        record("running-draining", "DRAINING"),
    ]

    with pytest.raises(KUBE.CleanupStateError, match="EC2 instance .* is running"):
        KUBE.node_targets(client, journal, CONTEXT, "gpu-a", config=CONFIG)
    assert "zero_node_clusters" not in journal, "nothing is recorded on a refusal"


@pytest.mark.parametrize("state", ["RETIRED", None])
def test_a_record_neither_retired_nor_provable_keeps_the_refusal(
    proofs: dict[str, Any], state: str | None
) -> None:
    client, journal = zero_node_client(), document()
    journal["fleet_snapshot"] = [record("odd", "REVOKED")]
    journal["fleet_snapshot"][0]["lifecycle_state"] = state

    with pytest.raises(KUBE.CleanupStateError, match="neither retired nor provable"):
        KUBE.node_targets(client, journal, CONTEXT, "gpu-a", config=CONFIG)


def test_a_node_carrying_our_state_keeps_its_refusal(proofs: dict[str, Any]) -> None:
    """Anything of ours on a node the targets do not name still refuses."""

    client, journal = zero_node_client(), document()
    client.nodes[0]["metadata"]["labels"] = {"gpu-fault.io/managed": "true"}
    journal["fleet_snapshot"] = [record("retired-a", "REVOKED")]

    with pytest.raises(KUBE.CleanupStateError, match="unregistered node runtime"):
        KUBE.node_targets(client, journal, CONTEXT, "gpu-a", config=CONFIG)


def test_without_a_bound_config_nothing_is_proven(proofs: dict[str, Any]) -> None:
    client, journal = zero_node_client(), document()
    journal["fleet_snapshot"] = [record("retired-a", "REVOKED")]

    with pytest.raises(KUBE.CleanupStateError, match="no proven fleet targets"):
        KUBE.node_targets(client, journal, CONTEXT, "gpu-a")
    assert proofs["read"] == [], "no proof read without the bound cluster entry"


def test_the_skipped_node_cleanup_names_the_cluster_without_nodes(
    proofs: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, journal = zero_node_client(), document()
    journal["fleet_snapshot"] = [record("retired-a", "REVOKED")]
    monkeypatch.setattr(KUBE, "CleanupClient", lambda _config, _context: client)
    state_path = tmp_path / "state.json"

    KUBE.run_node_cleanup(
        CONFIG,
        journal,
        state_path,
        context=CONTEXT,
        cluster_id="gpu-a",
        image="example.invalid/cleanup@sha256:" + "0" * 64,
        mode="uninstall",
        host_script_b64="",
    )

    record_entry = journal["node_cleanup"][CONTEXT]
    assert record_entry["status"] == "SKIPPED", record_entry
    assert record_entry["reason"] == (
        "GPU cluster without nodes (parked): no node is selected and every fleet "
        "record is retired or proven departed; nothing to clean"
    ), record_entry
    assert client.calls == [], "no DaemonSet for a cluster without nodes"
    saved = json.loads(state_path.read_text())
    assert saved["zero_node_clusters"]["gpu-a"]["state"] == "parked", (
        "the zero-node record is checkpointed with the skip"
    )


def test_the_proof_is_read_from_the_bound_entry_with_the_engine_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``read_proof`` hands the release engine's reader the entry, the GPU
    kubeconfig of the bound config and the site's region."""

    from gpu_fault.admin import bootstrap_zero_nodes

    calls: list[dict[str, Any]] = []

    def proof(runner: Any, **arguments: Any) -> dict[str, Any]:
        calls.append(arguments)
        return PARKED

    monkeypatch.setattr(bootstrap_zero_nodes, "zero_node_proof_for_entry", proof)

    assert ZERO.read_proof(CONFIG, CONFIG["clusters"][0]) == PARKED
    assert calls == [
        {
            "gpu_kubeconfig": Path("/secure/gpu.kubeconfig"),
            "entry": CONFIG["clusters"][0],
            "region": "us-west-2",
        }
    ], calls
