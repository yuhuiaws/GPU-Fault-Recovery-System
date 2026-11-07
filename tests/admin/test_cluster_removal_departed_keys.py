"""``remove-cluster`` of a cluster whose GPU node-key Secret still holds keys of
departed nodes (``cluster_removal_departed_keys``).

The Executor's node-key sync never deletes a key, so a cluster that lost a node
since its last deploy kept that node's key, and the removal refused it as "GPU
node-key ownership is incomplete or rotating". A key is now accepted when its
node is proven departed -- absent from the cluster's unfiltered node list and
gone from its provider -- and recorded with its digest and evidence; anything
less refuses, naming the node and its blocker. The CPU entries removed are
still exactly the selected nodes'.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import cluster_removal_departed_keys as departed_keys
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.node_key_proof import node_key_digest
from gpu_fault.departed_nodes import (
    NO_HYPERPOD_INSTANCE_BLOCKER,
    NO_RECORDED_INSTANCE_BLOCKER,
)
from tests.admin.test_admin_cluster_removal_keys import NodeKeyScenario, encoded

ROOT = Path(__file__).resolve().parents[2]
CURRENT = "hyperpod-i-00000000"
DEPARTED = "hyperpod-i-00000001"
SECOND = "hyperpod-i-00000002"
EKS_DEPARTED = "ip-10-0-0-1.us-east-1.compute.internal"
EKS_INSTANCE = "i-00000003"
GPU_KUBECTL = ["kubectl", "--kubeconfig", "/state/gpu.kubeconfig", "--context", "c"]
HYPERPOD = {"cluster_id": "gpu-a", "context": "c", "hyperpod_cluster_name": "hp-a"}
EKS = {"cluster_id": "gpu-a", "context": "c", "provider": "eks-managed"}
SITE = SimpleNamespace(
    release_config={
        "cpu_kubeconfig": "/state/cpu.kubeconfig",
        "namespace": "gpu-fault-system",
        "aws_region": "us-east-1",
    },
    environment={},
    repository_root=ROOT,
)


def answer(value: object) -> str:
    if isinstance(value, Exception):
        raise value
    return value if isinstance(value, str) else json.dumps(value)


class Reads:
    """Every read the departed-key proof makes, answered from fixtures; any
    other command, and any write, fails the test."""

    def __init__(
        self,
        *,
        nodes: object = (CURRENT,),
        listed: object = ("i-00000000",),
        states: object = None,
        recorded: object = None,
    ) -> None:
        self.nodes = list(nodes) if isinstance(nodes, tuple) else nodes
        self.listed = list(listed) if isinstance(listed, tuple) else listed
        self.states = {} if states is None else states
        self.recorded = {} if recorded is None else recorded
        self.commands: list[list[str]] = []

    def run(self, arguments: Sequence[str], **options: Any) -> str:
        command = list(arguments)
        self.commands.append(command)
        assert not options.get("mutate"), f"the proof only reads: {command}"
        if command[0] == "kubectl" and "nodes" in command:
            assert "-l" not in command, "the node list is read unfiltered"
            return answer(self.nodes)
        if command[0] == "kubectl" and "pod" in command:
            return "fixture-ingress"
        if command[0] == "kubectl" and "exec" in command:
            assert options.get("sensitive") is True, "the fleet is read privately"
            asked = json.loads(options["input_text"])["node_ids"]
            recorded = answer(self.recorded)
            return json.dumps(
                {
                    "mode": "agent-instance-ids",
                    "instance_ids": {
                        node: value
                        for node, value in json.loads(recorded).items()
                        if node in asked
                    },
                }
            )
        if command[:3] == ["aws", "sagemaker", "list-cluster-nodes"]:
            listed = answer(self.listed)
            return json.dumps(
                {
                    "ClusterNodeSummaries": [
                        {"InstanceId": i} for i in json.loads(listed)
                    ]
                }
            )
        if command[:3] == ["aws", "ec2", "describe-instances"]:
            states = json.loads(answer(self.states))
            asked = command[command.index("--filters") + 1].split("Values=")[1]
            return json.dumps([[i, states[i]] for i in asked.split(",") if i in states])
        raise AssertionError(f"unexpected command {command}")

    def asked(self, *prefix: str) -> bool:
        return any(command[: len(prefix)] == list(prefix) for command in self.commands)


def prove(
    reads: Reads, extras: dict[str, str], target: dict[str, Any] = HYPERPOD
) -> dict[str, Any]:
    return departed_keys.departed_node_keys(
        SITE,  # type: ignore[arg-type]
        target,
        "gpu-a",
        extras,
        {DEPARTED: "c" * 64},
        runner=reads,  # type: ignore[arg-type]
        gpu_kubectl=GPU_KUBECTL,
    )


def test_a_departed_hyperpod_node_key_is_recorded_with_its_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reads = Reads()

    record = prove(reads, {DEPARTED: "a" * 64, SECOND: "b" * 64})

    assert record[DEPARTED]["sha256"] == "a" * 64, record
    assert record[DEPARTED]["cpu_sha256"] == "c" * 64, "the CPU entry is recorded"
    assert record[SECOND]["cpu_sha256"] is None, "no CPU entry, none recorded"
    evidence = record[DEPARTED]["evidence"]
    assert {key: evidence[key] for key in evidence if key != "read_at"} == {
        "absent_from_kubernetes": True,
        "provider": "hyperpod",
        "instance_id": "i-00000001",
        "absent_from_provider": True,
    }, evidence
    assert set(evidence["read_at"]) == {"kubernetes", "hyperpod"}, evidence
    narration = capsys.readouterr().err
    assert departed_keys.NARRATION in narration, narration
    assert "kept=2 cpu_entries_kept=1" in narration, narration
    assert f"nodes={DEPARTED},{SECOND}" in narration, narration


@pytest.mark.parametrize("state", ["terminated", None], ids=["terminated", "unknown"])
def test_a_departed_eks_node_key_is_recorded_with_its_evidence(
    state: str | None,
) -> None:
    reads = Reads(
        nodes=("ip-10-0-0-2.us-east-1.compute.internal",),
        states={EKS_INSTANCE: state} if state else {},
        recorded={EKS_DEPARTED: EKS_INSTANCE},
    )

    record = prove(reads, {EKS_DEPARTED: "a" * 64}, EKS)

    evidence = record[EKS_DEPARTED]["evidence"]
    assert evidence["provider"] == "eks-managed", evidence
    assert evidence["instance_id"] == EKS_INSTANCE, evidence
    assert evidence["ec2_state"] == (state or "not-found"), evidence
    assert set(evidence["read_at"]) == {"kubernetes", "ec2"}, evidence


UNPROVEN = {
    "still-in-kubernetes": (
        HYPERPOD,
        DEPARTED,
        {"nodes": (CURRENT, DEPARTED)},
        "still in Kubernetes outside the cluster's node selector",
    ),
    "kubernetes-unreadable": (
        HYPERPOD,
        DEPARTED,
        {"nodes": BootstrapError("the API server is unreachable")},
        "node list cannot be read: the API server is unreachable",
    ),
    "kubernetes-ambiguous": (
        HYPERPOD,
        DEPARTED,
        {"nodes": '{"items": []}'},
        "node list cannot be read: the node list has ambiguous node names",
    ),
    "hyperpod-api-error": (
        HYPERPOD,
        DEPARTED,
        {"listed": BootstrapError("ThrottlingException")},
        "provider membership unknown: cannot list HyperPod nodes of gpu-a",
    ),
    "hyperpod-still-lists": (
        HYPERPOD,
        DEPARTED,
        {"listed": ("i-00000000", "i-00000001")},
        "HyperPod still lists instance i-00000001",
    ),
    "hyperpod-name-without-instance": (
        HYPERPOD,
        "node-retired",
        {},
        NO_HYPERPOD_INSTANCE_BLOCKER,
    ),
    "ec2-api-error": (
        EKS,
        EKS_DEPARTED,
        {
            "states": BootstrapError("UnauthorizedOperation"),
            "recorded": {EKS_DEPARTED: EKS_INSTANCE},
        },
        "provider membership unknown: cannot describe EC2 instances in us-east-1",
    ),
    "ec2-still-has-it": (
        EKS,
        EKS_DEPARTED,
        {"states": {EKS_INSTANCE: "stopped"}, "recorded": {EKS_DEPARTED: EKS_INSTANCE}},
        f"EC2 instance {EKS_INSTANCE} is stopped",
    ),
    "no-recorded-instance-id": (EKS, EKS_DEPARTED, {}, NO_RECORDED_INSTANCE_BLOCKER),
    "recorded-id-not-ec2": (
        EKS,
        EKS_DEPARTED,
        {"recorded": {EKS_DEPARTED: "0b6e0c4a-node-uid"}},
        NO_RECORDED_INSTANCE_BLOCKER,
    ),
    "fleet-unreadable": (
        EKS,
        EKS_DEPARTED,
        {"recorded": BootstrapError("exec failed")},
        "cannot read the instance ids the fleet recorded: exec failed",
    ),
}


@pytest.mark.parametrize(
    ("target", "node", "fixture", "blocker"), UNPROVEN.values(), ids=UNPROVEN
)
def test_a_key_whose_departure_is_unproven_refuses_naming_the_node_and_why(
    capsys: pytest.CaptureFixture[str],
    target: dict[str, Any],
    node: str,
    fixture: dict[str, Any],
    blocker: str,
) -> None:
    """Never "departed" by default: a failed or ambiguous read, a missing
    identity and a provider that still has the instance all refuse."""

    reads = Reads(**fixture)

    with pytest.raises(BootstrapError) as refused:
        prove(reads, {node: "a" * 64}, target)

    message = str(refused.value)
    assert message.startswith(
        "the GPU node-key Secret holds keys of nodes not proven departed: "
    ), message
    assert f"{node}: " in message and blocker in message, message
    assert departed_keys.NARRATION not in capsys.readouterr().err, "nothing kept"


def test_a_node_still_in_kubernetes_is_never_put_to_the_provider() -> None:
    reads = Reads(nodes=(CURRENT, DEPARTED))

    with pytest.raises(BootstrapError, match="still in Kubernetes"):
        prove(reads, {DEPARTED: "a" * 64})
    assert not reads.asked("aws"), "a node Kubernetes has is no departed node"


def test_the_refusal_names_eight_nodes_then_counts_the_rest() -> None:
    nodes = [f"node-{index:02d}" for index in range(11)]

    with pytest.raises(BootstrapError) as refused:
        prove(Reads(), dict.fromkeys(nodes, "a" * 64))

    message = str(refused.value)
    assert all(f"{node}: " in message for node in nodes[:8]), message
    assert "node-08: " not in message and message.endswith("; and 3 more"), message


@pytest.mark.parametrize(
    "record",
    [
        [],
        {},
        {"node-a": {"sha256": "a" * 64}},
        {DEPARTED: {"sha256": "A" * 64}},
        {DEPARTED: "a" * 64},
    ],
    ids=["not-a-map", "empty", "a-selected-node", "not-a-digest", "no-entry"],
)
def test_a_malformed_departed_key_record_is_no_proof(record: object) -> None:
    binding = {departed_keys.DEPARTED_KEYS: record}

    with pytest.raises(BootstrapError, match="no complete node-key digest proof"):
        departed_keys.recorded_departed_keys(binding, ["node-a"])


class DepartedKeyRemoval(NodeKeyScenario):
    """A HyperPod removal whose GPU Secret holds two departed nodes' keys; the
    CPU Secret still holds one of them."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        self.gpu["data"][DEPARTED] = encoded("departed-one-fixture")
        self.gpu["data"][SECOND] = encoded("departed-two-fixture")
        self.cpu["data"][DEPARTED] = encoded("departed-one-fixture")
        self.listed = ["i-00000009"]
        self.provider_reads = 0

    def run(self, arguments: Sequence[str], **kwargs: Any) -> str:
        if list(arguments[:3]) == ["aws", "sagemaker", "list-cluster-nodes"]:
            self.provider_reads += 1
            return json.dumps(
                {"ClusterNodeSummaries": [{"InstanceId": i} for i in self.listed]}
            )
        return super().run(arguments, **kwargs)


def test_a_removal_keeps_and_records_the_keys_of_departed_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario = DepartedKeyRemoval(tmp_path, monkeypatch)
    kept = {
        node: value for node, value in scenario.cpu["data"].items() if node != "node-a"
    }

    result = removal.remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED", result
    binding = scenario.state()["evidence"]["DISCOVERED"]["node_key_ownership"]
    assert set(binding["expected_key_sha256"]) == {"node-a"}, binding
    recorded = binding[departed_keys.DEPARTED_KEYS]
    assert recorded[DEPARTED]["sha256"] == node_key_digest(
        encoded("departed-one-fixture")
    ), recorded
    assert recorded[DEPARTED]["cpu_sha256"] == recorded[DEPARTED]["sha256"], recorded
    assert recorded[SECOND]["cpu_sha256"] is None, recorded
    assert recorded[SECOND]["evidence"]["absent_from_provider"] is True, recorded
    assert scenario.cpu["data"] == kept, (
        "only the selected node's CPU entry is deleted; a departed node's is kept"
    )
    removed = [
        operation["path"]
        for patch in scenario.key_patches
        for operation in patch
        if operation["op"] == "remove"
    ]
    assert removed == ["/data/node-a"], removed
    assert "kept=2 cpu_entries_kept=1" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "outcome"),
    [("digest", "refused"), ("new-key", "refused"), ("gone", "completed")],
)
def test_a_resume_rechecks_the_recorded_departed_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, outcome: str
) -> None:
    scenario = DepartedKeyRemoval(tmp_path, monkeypatch)
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="cleanup"):
        removal.remove_cluster(scenario.request())
    original = copy.deepcopy(scenario.state()["evidence"]["DISCOVERED"])
    if change == "digest":
        scenario.gpu["data"][DEPARTED] = encoded("replacement-fixture")
    elif change == "new-key":
        scenario.gpu["data"]["hyperpod-i-00000004"] = encoded("newcomer-fixture")
    else:
        scenario.gpu["data"].pop(SECOND)
    scenario.failure = None
    reads = scenario.provider_reads

    if outcome == "refused":
        with pytest.raises(BootstrapError, match="incarnation or ownership changed"):
            removal.remove_cluster(scenario.request())
        assert scenario.key_patches == [], "a changed record authorized no deletion"
    else:
        assert removal.remove_cluster(scenario.request())["phase"] == "COMPLETED"
    assert scenario.provider_reads == reads, "a resume proves nothing anew"
    assert scenario.state()["evidence"]["DISCOVERED"] == original, (
        "the resume kept the original record"
    )


def test_an_unproven_departed_key_refuses_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = DepartedKeyRemoval(tmp_path, monkeypatch)
    scenario.listed = ["i-00000001"]

    with pytest.raises(BootstrapError) as refused:
        removal.remove_cluster(scenario.request())

    assert f"{DEPARTED}: node is missing from Kubernetes but HyperPod still lists" in (
        str(refused.value)
    ), refused.value
    assert scenario.calls == ["snapshot"], "nothing ran past discovery"
    assert scenario.key_patches == [], "no key was deleted"


def test_a_removal_of_a_cluster_without_nodes_binds_and_deletes_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole remove-cluster state machine over a cluster whose selector
    selects no node: no ownership binding, no CPU key deleted, every step run;
    the GPU keys of its departed nodes go with the namespace."""

    scenario = DepartedKeyRemoval(tmp_path, monkeypatch)
    monkeypatch.setattr(removal, "_target_nodes", lambda *_a, **_k: [])
    before = copy.deepcopy(scenario.cpu["data"])

    result = removal.remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED", result
    state = scenario.state()
    assert state["evidence"]["DISCOVERED"]["nodes"] == [], state["evidence"]
    assert state["evidence"]["DISCOVERED"]["node_key_ownership"] == {}, state
    assert state["evidence"]["CONTROL_REGISTRY_REMOVED"] == {
        "node_action_keys_removed": 0
    }, state["evidence"]
    assert scenario.key_patches == [] and scenario.cpu["data"] == before, (
        "no CPU key is deleted for a cluster without nodes"
    )
    assert scenario.provider_reads == 0, "no binding, nothing to prove"
    assert "unregister" in scenario.calls and "ownership-release" in scenario.calls
