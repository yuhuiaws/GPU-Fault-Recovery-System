"""The deploy's bootstrap over a GPU cluster without nodes: its node keys.

Live, an eks-managed GPU cluster scaled to zero failed the deploy in its
bootstrap, before any roll: ``node_keys:<cluster>`` ran the node-key helper,
which refuses an empty node inventory ("HyperPod node inventory is empty or
invalid"). The release engine itself skips that cluster's node work; the
bootstrap now asks the same proof in the same run
(``gpu_fault.admin.bootstrap_zero_nodes``) and, for a cluster without nodes --
parked or unavailable -- provisions no key, keeps the keys it has and says so.
A node list that cannot be read proves nothing: the helper runs, and refuses,
as it always did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin import bootstrap_zero_nodes as ZERO
from gpu_fault.admin.bootstrap_common import ReadOnlyProbeRunner
from tests.admin._zero_node_bootstrap_support import (
    EKS_GROUP,
    ZeroNodeSite,
    eks_cluster,
    hyperpod_cluster,
)


def _keys(site: ZeroNodeSite, cluster=None, cluster_id: str = "gpu-eks"):
    return ZERO.zero_node_keys(
        site,
        gpu_kubeconfig=Path("/state/gpu.kubeconfig"),
        cluster=cluster or eks_cluster(),
        cluster_id=cluster_id,
    )


def _line(capsys: pytest.CaptureFixture[str]) -> str:
    return next(
        item
        for item in capsys.readouterr().err.splitlines()
        if item.startswith("zero-node-gpu-cluster ")
    )


@pytest.mark.parametrize(
    "desired,state", [(0, "parked"), (2, "unavailable")], ids=["parked", "unavailable"]
)
def test_a_cluster_without_nodes_gets_no_key_and_says_so(
    capsys: pytest.CaptureFixture[str], desired: int, state: str
) -> None:
    site = ZeroNodeSite(desired=desired)

    result = _keys(site)

    assert result is not None, "a cluster without nodes has no key to provision"
    assert result["cluster_id"] == "gpu-eks"
    entry = result["zero_nodes"]
    assert entry["state"] == state, entry
    assert [group["name"] for group in entry["groups"]] == [EKS_GROUP]
    line = _line(capsys)
    for field in (
        "cluster=gpu-eks",
        f"state={state}",
        "step=node_keys",
        'recorded="no node, no key to provision"',
        "reason=",
    ):
        assert field in line, line
    assert site.helper_runs() == [], "the node-key helper never runs"
    assert site.mutating() == [], "nothing is written: no key deleted or rotated"
    assert not any("secret" in command for command, _m, _e in site.commands), (
        "the key Secrets are not even read"
    )


def test_the_proof_is_the_release_engines_and_needs_only_read_only_calls() -> None:
    """The same reads as ``regional_parked_cluster`` -- the node list, then the
    node groups and the NodePools -- through a probe runner, which refuses any
    mutation."""

    site = ZeroNodeSite()

    result = _keys(ReadOnlyProbeRunner(site))  # type: ignore[arg-type]

    assert result is not None and result["zero_nodes"]["state"] == "parked"
    issued = [command[:3] for command, _mutate, _env in site.commands]
    assert ["aws", "eks", "list-nodegroups"] in issued
    assert ["aws", "eks", "describe-nodegroup"] in issued
    reads = site.commands_for("gpu-eks")
    assert reads[0][-5:] == [
        "nodes",
        "-l",
        "gpu-fault.io/control-plane=site-a",
        "-o",
        "json",
    ], "the node list under the recorded selector comes first"
    assert any("nodepools.karpenter.sh" in command for command in reads), (
        "the Karpenter NodePools are read, as the engine reads them"
    )


def test_a_cluster_with_a_node_is_provisioned_as_before() -> None:
    site = ZeroNodeSite()

    assert _keys(site, hyperpod_cluster(), "gpu-a") is None
    assert [command[:3] for command, _m, _e in site.commands] == [
        ["kubectl", "--kubeconfig", "/state/gpu.kubeconfig"]
    ], "only the node list is read; no provider is asked about a cluster with nodes"


def test_a_node_list_that_cannot_be_read_proves_nothing() -> None:
    """Then the task runs exactly as before -- and its helper refuses."""

    assert _keys(ZeroNodeSite(unreadable=True)) is None


def test_a_hyperpod_cluster_scaled_to_zero_is_proven_from_describe_cluster(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class HyperPodAtZero(ZeroNodeSite):
        def run(self, arguments, **kwargs):  # type: ignore[no-untyped-def]
            command = [str(item) for item in arguments]
            if command[:3] == ["aws", "sagemaker", "describe-cluster"]:
                self.commands.append((command, False, {}))
                return json.dumps(
                    {
                        "InstanceGroups": [
                            {
                                "InstanceGroupName": "gpu-spot-group",
                                "InstanceType": "ml.p5en.48xlarge",
                                "CurrentCount": 0,
                                "TargetCount": 0,
                                "Status": "InService",
                            }
                        ]
                    }
                )
            if command[0] == "kubectl" and "nodes" in command:
                self.commands.append((command, False, {}))
                return json.dumps({"apiVersion": "v1", "kind": "List", "items": []})
            return super().run(arguments, **kwargs)

    site = HyperPodAtZero()

    result = _keys(site, hyperpod_cluster("hp-gpu"), "hp-gpu")

    assert result is not None and result["zero_nodes"]["state"] == "parked", result
    assert result["zero_nodes"]["provider"] == "hyperpod"
    assert "state=parked" in _line(capsys)
