"""The batch join of plain EKS GPU clusters (provider ``eks-managed``).

``deploy`` hands every GPU ARN it adds to the batch join -- one ARN or several,
including the second ARN of an EKS-first first deploy -- and ``join-cluster``
does so for two or more. A plain EKS ``spec.clusters[]`` entry carries no
``hyperpodClusterName``; the batch candidate writer compared that absent name
like a real one, so any two plain EKS entries "conflicted" and the batch failed
with "batch join contains a conflicting cluster" for every new member.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import deploy_command
from gpu_fault.admin.bootstrap_discovery import same_cluster_entry
from gpu_fault.admin.site import load_site
from tests.admin._cov95_join_support import ACCOUNT, REGION
from tests.admin.test_admin_eks_managed_join import EksJoinScenario, eks_target

SECOND_EKS_NODE = "ip-10-0-40-7.us-west-2.compute.internal"


def _second_eks_target(scenario: EksJoinScenario) -> str:
    second = replace(
        eks_target(),
        input_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-f",
        eks_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-f",
        eks_name="gpu-f",
        vpc_id="vpc-gpu-f",
        context="gpu-f",
    )
    scenario.targets[second.eks_arn] = second
    scenario.node_names["f"] = SECOND_EKS_NODE
    return second.eks_arn


class _DeployHooks:
    """The two hooks ``deploy_command.join_gpu_clusters`` calls, wired to the
    real batch join and the scenario's command boundary."""

    def __init__(self, scenario: EksJoinScenario) -> None:
        self.scenario = scenario
        self.results: list[dict[str, Any]] = []

    def load_site(self, path, *, repository_root=None):
        return load_site(path, repository_root=repository_root)

    def join_clusters(self, requests):
        result = batch.join_clusters(requests, runner_factory=lambda: self.scenario)
        self.results.append(result)
        return result


def _deploy_join(scenario: EksJoinScenario, *arns: str) -> dict[str, Any]:
    hooks = _DeployHooks(scenario)
    deploy_command.join_gpu_clusters(
        scenario.path, arns, repository_root=None, hooks=hooks
    )
    (result,) = hooks.results
    return result


def _site_entries(scenario: EksJoinScenario) -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(scenario.path.read_text())
    return {item["clusterId"]: item for item in document["spec"]["clusters"]}


def test_one_batch_joins_two_plain_eks_clusters_into_a_plain_eks_site(
    tmp_path, monkeypatch, capsys
) -> None:
    scenario = EksJoinScenario(tmp_path, monkeypatch)
    second = _second_eks_target(scenario)

    result = _deploy_join(scenario, eks_target().eks_arn, second)

    assert result["phase"] == "COMPLETED", result
    assert result["joined"] == ["gpu-e", "gpu-f"], (
        "both plain EKS clusters must join in the one batch"
    )
    assert result["failed"] == {}, "no member may fail as a conflicting cluster"
    assert scenario.cluster_states == {
        "gpu-a": "ACTIVE",
        "gpu-e": "ACTIVE",
        "gpu-f": "ACTIVE",
    }, "the registry must end with both new members ACTIVE beside the old one"
    entries = _site_entries(scenario)
    assert sorted(entries) == ["gpu-a", "gpu-e", "gpu-f"], (
        "site.yaml must list the existing and both joined clusters"
    )
    for cluster_id in ("gpu-e", "gpu-f"):
        assert entries[cluster_id]["provider"] == "eks-managed", (
            f"{cluster_id} must be recorded as plain EKS"
        )
        assert "hyperpodClusterName" not in entries[cluster_id], (
            f"no HyperPod name may be invented for {cluster_id}"
        )
    assert '"phase": "COMPLETED"' in capsys.readouterr().out, (
        "deploy prints the batch summary"
    )


def test_the_deploy_path_joins_one_plain_eks_cluster_into_a_plain_eks_site(
    tmp_path, monkeypatch
) -> None:
    """A deploy adding one ARN, or the follow-up join of an EKS-first first
    deploy's second ARN, takes the batch path even for a single cluster."""

    scenario = EksJoinScenario(tmp_path, monkeypatch)

    result = _deploy_join(scenario, eks_target().eks_arn)

    assert (result["phase"], result["joined"]) == ("COMPLETED", ["gpu-e"]), result
    assert scenario.cluster_states == {"gpu-a": "ACTIVE", "gpu-e": "ACTIVE"}, (
        "the joined plain EKS cluster must be registered beside the existing one"
    )
    assert sorted(_site_entries(scenario)) == ["gpu-a", "gpu-e"], (
        "site.yaml must list both plain EKS clusters"
    )


def _entry(cluster_id: str, *, hyperpod: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "clusterId": cluster_id,
        "context": f"ctx-{cluster_id}",
        "eksClusterArn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{cluster_id}",
    }
    if hyperpod is None:
        value["provider"] = "eks-managed"
    else:
        value["hyperpodClusterName"] = hyperpod
    return value


@pytest.mark.parametrize(
    ("item", "target", "same"),
    [
        pytest.param(_entry("eks-1"), _entry("eks-2"), False, id="two-plain-eks"),
        pytest.param(
            _entry("hp-1", hyperpod="hp-1"), _entry("eks-2"), False, id="hp-eks"
        ),
        pytest.param(
            _entry("eks-1"), _entry("hp-2", hyperpod="hp-2"), False, id="eks-hp"
        ),
        pytest.param(
            _entry("hp-1", hyperpod="hp-1"),
            _entry("hp-2", hyperpod="hp-2"),
            False,
            id="two-hyperpod",
        ),
        pytest.param(
            _entry("hp-1", hyperpod="shared"),
            _entry("hp-2", hyperpod="shared"),
            True,
            id="same-hyperpod-name",
        ),
        pytest.param(
            _entry("eks-1"),
            {**_entry("eks-2"), "clusterId": "eks-1"},
            True,
            id="same-cluster-id",
        ),
        pytest.param(
            _entry("eks-1"),
            {**_entry("eks-2"), "eksClusterArn": _entry("eks-1")["eksClusterArn"]},
            True,
            id="same-eks-arn",
        ),
        pytest.param(
            _entry("eks-1"),
            {**_entry("eks-2"), "context": "ctx-eks-1"},
            True,
            id="same-context",
        ),
        pytest.param(_entry("eks-1"), _entry("eks-1"), True, id="itself"),
    ],
)
def test_cluster_entries_conflict_only_on_an_identity_they_both_carry(
    item: dict[str, Any], target: dict[str, Any], same: bool
) -> None:
    assert same_cluster_entry(item, target) is same, (
        "an absent HyperPod name must never make two entries the same cluster, "
        "and every identity both entries carry still must"
    )
    assert same_cluster_entry(target, item) is same, "the judgement is symmetric"
