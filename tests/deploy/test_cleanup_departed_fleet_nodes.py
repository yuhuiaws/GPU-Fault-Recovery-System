"""PREFLIGHT's proof that a missing ACTIVE fleet node left with its instance.

An uninstall or remove-cluster lets an ACTIVE Agent record whose node is gone
from Kubernetes pass only when the provider no longer has the instance -- the
departed-node rule the control plane's cleanup, ``workflow-reconcile`` and the
release engine share (``gpu_fault.departed_nodes``). The cleanup tool knew only
the HyperPod half: it read the instance out of a ``hyperpod-<instance-id>``
node name, so a departed node of an eks-managed cluster (named ``ip-...`` or
by its instance, the instance recorded in the Agent record) was refused, and
it compared the record's ``node_instance_id`` with that instance although the
HyperPod installer records the Kubernetes Node UID there, so a real HyperPod
record was refused too.

These cases drive ``node_targets`` and the ``capture`` entry point with the
fleet rows the cleanup probe really returns and a scripted AWS CLI.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import TOOLS
from tests.deploy.test_cov95_tools_cleanup_kubernetes import (
    HYPERPOD_CONFIG,
    hyperpod_listing,
    install_transport,
)
from tests.regional.test_cleanup_kubernetes import CONTEXT, Client, document

KUBE = lazy_script_module(TOOLS / "cleanup_kubernetes.py")
STATE = lazy_script_module(TOOLS / "cleanup_state.py")
REGION = "us-west-2"
EKS_CONFIG: dict[str, Any] = {
    "cpu_kubeconfig": "/dev/null",
    "aws_region": REGION,
    "clusters": [
        {
            "cluster_id": "gpu-a",
            "context": "gpu-a",
            "provider": "eks-managed",
            "region": REGION,
            "eks_cluster_arn": f"arn:aws:eks:{REGION}:111122223333:cluster/gpu-eks",
        }
    ],
}
EKS_NODE = "ip-10-0-33-205.us-west-2.compute.internal"
EKS_INSTANCE = "i-00000000000000002"
NODE_UID = "6f1d3c2a-8b7e-4a5d-9c0f-1e2d3c4b5a69"
HYPERPOD_INSTANCE = "i-00000000000000001"
HYPERPOD_NODE = f"hyperpod-{HYPERPOD_INSTANCE}"
UNITS = {"units": ["gpu-fault-node-agent.service"]}


def departed(node_id: str, node_instance_id: str | None) -> dict[str, Any]:
    """A fleet probe row of an ACTIVE Agent whose node Kubernetes lost."""

    state = document()
    state["fleet_snapshot"].append(
        {
            "cluster_id": "gpu-a",
            "node_id": node_id,
            "node_instance_id": node_instance_id,
            "lifecycle_state": "ACTIVE",
            "installed_unit_inventory": UNITS,
        }
    )
    return state


def ec2_rows(state: str | None) -> list[list[str]]:
    """``describe-instances --query Reservations[].Instances[].[InstanceId,State.Name]``."""

    return [] if state is None else [[EKS_INSTANCE, state]]


def describe_instances() -> list[str]:
    return [
        "aws",
        "ec2",
        "describe-instances",
        "--region",
        REGION,
        "--filters",
        f"Name=instance-id,Values={EKS_INSTANCE}",
        "--query",
        "Reservations[].Instances[].[InstanceId,State.Name]",
        "--output",
        "json",
    ]


@pytest.mark.parametrize(
    "answer,state", [("terminated", "terminated"), (None, "not-found")]
)
def test_a_departed_eks_node_whose_instance_ec2_reports_gone_is_journaled(
    monkeypatch: pytest.MonkeyPatch, answer: str | None, state: str
) -> None:
    client, document_ = Client(), departed(EKS_NODE, EKS_INSTANCE)
    calls = install_transport(monkeypatch, ec2_rows(answer))

    assert KUBE.node_targets(
        client, document_, CONTEXT, "gpu-a", config=EKS_CONFIG
    ) == {"managed-node": "uid-managed-node"}, "the departed node is no cleanup target"
    assert document_["departed_fleet_nodes"] == [
        {
            "cluster_id": "gpu-a",
            "node_id": EKS_NODE,
            "node_instance_id": EKS_INSTANCE,
            "provider": "eks-managed",
            "ec2_state": state,
        }
    ], "the departure is journaled with the EC2 answer"
    record = document_["provider_nodes"]["gpu-a"]
    assert (record["provider"], record["region"], record["instance_states"]) == (
        "eks-managed",
        REGION,
        {EKS_INSTANCE: state},
    ), record
    assert record["sha256"] == STATE.ec2_instance_states_digest({EKS_INSTANCE: state})
    assert [call[0] for call in calls] == [describe_instances()], (
        "one read of the recorded instance, by filter, in the cluster's Region"
    )
    assert 0 < calls[0][1]["timeout_seconds"] <= 45, "bounded like the HyperPod read"

    # A re-capture reads the journaled answer instead of asking EC2 again.
    del document_["node_targets"]
    KUBE.node_targets(client, document_, CONTEXT, "gpu-a", config=EKS_CONFIG)
    assert len(calls) == 1, "EC2 is asked once per instance"
    assert len(document_["departed_fleet_nodes"]) == 1, "the journal is not duplicated"
    assert client.mutations() == [], "the capture writes nothing"


@pytest.mark.parametrize(
    "state", ["pending", "running", "shutting-down", "stopping", "stopped"]
)
def test_a_missing_eks_node_whose_instance_ec2_still_has_fails_closed(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    client, document_ = Client(), departed(EKS_NODE, EKS_INSTANCE)
    install_transport(monkeypatch, ec2_rows(state))

    with pytest.raises(
        KUBE.CleanupStateError,
        match=(
            f"an active fleet node is missing: {EKS_NODE}; "
            f"EC2 instance {EKS_INSTANCE} is {state}"
        ),
    ):
        KUBE.node_targets(client, document_, CONTEXT, "gpu-a", config=EKS_CONFIG)

    assert "node_targets" not in document_, "nothing is recorded on a refusal"
    assert "departed_fleet_nodes" not in document_, "no departure either"
    assert client.mutations() == [], "the refusal writes nothing"


def eks_config(**changes: Any) -> dict[str, Any]:
    config = copy.deepcopy(EKS_CONFIG)
    for key, value in changes.items():
        if value is None:
            del config["clusters"][0][key]
        else:
            config["clusters"][0][key] = value
    return config


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("no-recorded-instance", "the fleet recorded no EC2 instance id"),
        ("node-uid-recorded", "the fleet recorded no EC2 instance id"),
        ("no-region", "eks-managed cluster has no bound Region"),
        ("unknown-provider", "cluster provider must be one of"),
        ("other-context", "outside the bound config scope"),
        ("no-config", "cleanup cluster has no bound config"),
        (
            "aws-exit",
            "cannot describe EC2 instances in us-west-2: "
            "UnauthorizedOperation: <sensitive output",
        ),
        ("aws-missing", r"EC2 instance lookup failed \(FileNotFoundError\)"),
        ("not-json", "EC2 instance description is invalid JSON"),
        ("invalid-rows", "EC2 instance description is invalid"),
    ],
)
def test_a_missing_eks_node_without_ec2_proof_keeps_refusing(
    monkeypatch: pytest.MonkeyPatch, variant: str, reason: str
) -> None:
    recorded: str | None = EKS_INSTANCE
    config: dict[str, Any] | None = EKS_CONFIG
    answer: Any = ec2_rows("terminated")
    code = 0
    if variant == "no-recorded-instance":
        recorded = None
    elif variant == "node-uid-recorded":
        recorded = NODE_UID
    elif variant == "no-region":
        config = {**eks_config(region=None), "aws_region": None}
    elif variant == "unknown-provider":
        config = eks_config(provider="eks")
    elif variant == "other-context":
        config = eks_config(context="gpu-b")
    elif variant == "no-config":
        config = None
    elif variant == "aws-exit":
        code = 1
    elif variant == "not-json":
        answer = "not json"
    elif variant == "invalid-rows":
        answer = [{"InstanceId": EKS_INSTANCE}]
    client, document_ = Client(), departed(EKS_NODE, recorded)
    calls = install_transport(monkeypatch, answer, code)
    if variant == "aws-exit":

        def denied(arguments: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
            calls.append((arguments, {}))
            return subprocess.CompletedProcess(
                arguments,
                254,
                "",
                "An error occurred (UnauthorizedOperation) when calling the "
                "DescribeInstances operation: You are not authorized",
            )

        monkeypatch.setattr(KUBE, "run_command", denied)
    elif variant == "aws-missing":

        def missing(arguments: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
            calls.append((arguments, {}))
            raise FileNotFoundError("aws")

        monkeypatch.setattr(KUBE, "run_command", missing)

    with pytest.raises(
        KUBE.CleanupStateError,
        match=f"an active fleet node is missing: {EKS_NODE}; .*{reason}",
    ):
        KUBE.node_targets(client, document_, CONTEXT, "gpu-a", config=config)

    if variant not in {"aws-exit", "aws-missing", "not-json", "invalid-rows"}:
        assert calls == [], "no EC2 call without a recorded, bound instance identity"
    assert "departed_fleet_nodes" not in document_, "no departure is journaled"
    assert "provider_nodes" not in document_, "no provider answer is journaled"
    assert client.mutations() == [], "the refusal writes nothing"


def test_a_tampered_ec2_journal_is_refused_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, document_ = Client(), departed(EKS_NODE, EKS_INSTANCE)
    calls = install_transport(monkeypatch, ec2_rows("running"))
    STATE.attach_ec2_instance_states(
        document_, "gpu-a", region=REGION, states={EKS_INSTANCE: "running"}
    )
    document_["provider_nodes"]["gpu-a"]["instance_states"][EKS_INSTANCE] = "terminated"

    with pytest.raises(
        KUBE.CleanupStateError, match="captured EC2 instance states are invalid"
    ):
        KUBE.node_targets(client, document_, CONTEXT, "gpu-a", config=EKS_CONFIG)
    assert calls == [], "a journal that does not match its digest is not re-asked"


def test_a_hyperpod_listing_journaled_for_an_eks_cluster_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, document_ = Client(), departed(EKS_NODE, EKS_INSTANCE)
    install_transport(monkeypatch, ec2_rows("terminated"))
    STATE.attach_provider_nodes(
        document_,
        "gpu-a",
        hyperpod_cluster_name="hp-gpu-a",
        region=REGION,
        instance_ids=[HYPERPOD_INSTANCE],
    )

    with pytest.raises(
        KUBE.CleanupStateError, match="captured EC2 instance states are invalid"
    ):
        KUBE.node_targets(client, document_, CONTEXT, "gpu-a", config=EKS_CONFIG)


def test_a_departed_hyperpod_record_carrying_the_node_uid_is_proven_by_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HyperPod installer passes the Kubernetes Node UID as
    ``--node-instance-id``; only the node name carries the EC2 instance."""

    client, document_ = Client(), departed(HYPERPOD_NODE, NODE_UID)
    calls = install_transport(monkeypatch, hyperpod_listing("i-00000000000000002"))

    assert KUBE.node_targets(
        client, document_, CONTEXT, "gpu-a", config=HYPERPOD_CONFIG
    ) == {"managed-node": "uid-managed-node"}, "the departed node is no target"
    assert document_["departed_fleet_nodes"] == [
        {
            "cluster_id": "gpu-a",
            "node_id": HYPERPOD_NODE,
            "node_instance_id": HYPERPOD_INSTANCE,
            "provider": "hyperpod",
        }
    ], "the departure names the instance from the node name"
    assert len(calls) == 1, "one HyperPod listing"


def test_a_hyperpod_cluster_never_asks_ec2_and_an_eks_cluster_never_lists_hyperpod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider of the bound cluster picks the proof, not the node name."""

    eks_named_like_hyperpod = departed(HYPERPOD_NODE, EKS_INSTANCE)
    calls = install_transport(monkeypatch, ec2_rows("terminated"))
    KUBE.node_targets(
        Client(), eks_named_like_hyperpod, CONTEXT, "gpu-a", config=EKS_CONFIG
    )
    assert [call[0][:2] for call in calls] == [["aws", "ec2"]], calls

    hyperpod_named_like_eks = departed(EKS_NODE, EKS_INSTANCE)
    with pytest.raises(
        KUBE.CleanupStateError, match="node name carries no HyperPod instance id"
    ):
        KUBE.node_targets(
            Client(), hyperpod_named_like_eks, CONTEXT, "gpu-a", config=HYPERPOD_CONFIG
        )
    assert len(calls) == 1, "no provider call for an unknown HyperPod identity"


def test_capture_entrypoint_journals_a_departed_eks_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(EKS_CONFIG))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"cpu": {"resources": []}, **document()["inventory_snapshot"]})
    )
    path = tmp_path / "state.json"
    state = STATE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="gpu",
        mode="clean",
        node_mode="uninstall",
        cluster_ids=["gpu-a"],
    )
    fixture = departed(EKS_NODE, EKS_INSTANCE)
    state.update(
        {
            key: fixture[key]
            for key in ("cluster_uids", "namespace_snapshots", "fleet_snapshot")
        }
    )
    STATE.atomic_write(path, state)
    client = Client()
    monkeypatch.setattr(KUBE, "CleanupClient", lambda _config, _context: client)
    calls = install_transport(monkeypatch, ec2_rows(None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cleanup_kubernetes",
            "--config",
            str(config),
            "--state-file",
            str(path),
            "--context",
            CONTEXT,
            "--cluster-id",
            "gpu-a",
            "capture",
        ],
    )

    assert KUBE.main() == 0
    persisted = STATE.read_state(path)
    assert persisted["node_targets"] == {CONTEXT: {"managed-node": "uid-managed-node"}}
    assert [
        (entry["node_id"], entry["provider"], entry["ec2_state"])
        for entry in persisted["departed_fleet_nodes"]
    ] == [(EKS_NODE, "eks-managed", "not-found")], persisted["departed_fleet_nodes"]
    assert STATE.verify_ec2_instance_states(persisted["provider_nodes"]["gpu-a"]) == {
        EKS_INSTANCE: "not-found"
    }, "the bound journal carries the EC2 answer under its digest"
    assert len(calls) == 1, "one EC2 read"
    assert client.mutations() == [], "the capture writes nothing"
