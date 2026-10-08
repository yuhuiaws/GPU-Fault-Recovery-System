"""The BOOT-029 driver asks the product's own discovery for each GPU ARN's provider."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from scripts.e2e.regional import boot029_providers as providers

HYPERPOD_EKS = "arn:aws:eks:us-west-2:123456789012:cluster/hyperpod-orchestrator"
HYPERPOD = "arn:aws:sagemaker:us-west-2:123456789012:cluster/hp-cluster"
PLAIN_EKS = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-eks-cluster"
SECOND_PLAIN_EKS = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-eks-cluster-b"


class HyperPodInventory(CommandRunner):
    """The Region's SageMaker clusters, each naming its orchestrator EKS ARN."""

    def __init__(self, orchestrators: dict[str, str]) -> None:
        super().__init__()
        self.orchestrators = orchestrators
        self.calls: list[tuple[str, ...]] = []

    def aws_json(
        self,
        region: str,
        *arguments: str,
        mutate: bool = False,
        sensitive: bool = False,
    ) -> dict[str, Any]:
        assert region == "us-west-2", "discovery reads the ARN's own Region"
        assert arguments[0] == "sagemaker", "provider discovery is a SageMaker read"
        assert mutate is False, "provider discovery never mutates"
        self.calls.append(arguments)
        if arguments[1] == "list-clusters":
            return {
                "ClusterSummaries": [
                    {"ClusterName": name} for name in sorted(self.orchestrators)
                ]
            }
        assert arguments[1:3] == ("describe-cluster", "--cluster-name"), arguments
        name = arguments[3]
        return {
            "ClusterName": name,
            "ClusterArn": f"arn:aws:sagemaker:us-west-2:123456789012:cluster/{name}",
            "Orchestrator": {"Eks": {"ClusterArn": self.orchestrators[name]}},
        }


def test_an_eks_arn_is_hyperpod_only_when_one_hyperpod_cluster_orchestrates_it() -> (
    None
):
    runner = HyperPodInventory({"hp-cluster": HYPERPOD_EKS, "other": "arn:aws:eks:x"})

    pairs = providers.gpu_cluster_providers([HYPERPOD_EKS, PLAIN_EKS], runner=runner)

    assert pairs == [("hyperpod", HYPERPOD_EKS), ("eks-managed", PLAIN_EKS)], (
        "the orchestrated EKS ARN is HyperPod, the un-orchestrated one plain EKS"
    )
    assert runner.calls[0] == ("sagemaker", "list-clusters"), (
        "the inventory is the product's own read"
    )


def test_a_sagemaker_arn_is_hyperpod_without_any_read() -> None:
    runner = HyperPodInventory({})

    pairs = providers.gpu_cluster_providers([HYPERPOD], runner=runner)

    assert pairs == [("hyperpod", HYPERPOD)], pairs
    assert runner.calls == [], "a SageMaker ARN names its provider itself"


def test_two_hyperpod_clusters_on_one_eks_cluster_are_refused_not_guessed() -> None:
    runner = HyperPodInventory({"hp-a": HYPERPOD_EKS, "hp-b": HYPERPOD_EKS})

    with pytest.raises(BootstrapError, match="at most one HyperPod cluster"):
        providers.gpu_cluster_providers([HYPERPOD_EKS], runner=runner)


def test_other_services_and_empty_lists_are_refused() -> None:
    runner = HyperPodInventory({})
    with pytest.raises(BootstrapError, match="EKS or SageMaker ARN"):
        providers.gpu_cluster_provider(
            "arn:aws:ec2:us-west-2:123456789012:instance/i-0000000000000000",
            runner=runner,
        )
    with pytest.raises(BootstrapError, match="at least one GPU cluster"):
        providers.gpu_cluster_providers([], runner=runner)
    with pytest.raises(BootstrapError, match="at least one GPU cluster"):
        providers.provider_groups([])


def test_groups_keep_the_operator_order_and_the_first_provider_deploys() -> None:
    groups = providers.provider_groups(
        [
            ("hyperpod", HYPERPOD_EKS),
            ("eks-managed", PLAIN_EKS),
            ("hyperpod", HYPERPOD),
            ("eks-managed", SECOND_PLAIN_EKS),
        ]
    )

    assert groups.first_provider == "hyperpod", groups
    assert groups.first_deploy == (HYPERPOD_EKS, HYPERPOD), (
        "every cluster of the first ARN's provider is in the first deploy, in order"
    )
    assert groups.joined == (("eks-managed", (PLAIN_EKS, SECOND_PLAIN_EKS)),), groups
    assert groups.joined_providers == ("eks-managed",), groups
    assert groups.joined_arns == (PLAIN_EKS, SECOND_PLAIN_EKS), groups

    eks_first = providers.provider_groups(
        [("eks-managed", PLAIN_EKS), ("hyperpod", HYPERPOD_EKS)]
    )
    assert eks_first.first_provider == "eks-managed", eks_first
    assert eks_first.joined == (("hyperpod", (HYPERPOD_EKS,)),), eks_first

    single = providers.provider_groups([("hyperpod", HYPERPOD_EKS)])
    assert single.joined == () and single.first_deploy == (HYPERPOD_EKS,), single


def test_cli_prints_one_provider_line_per_arn_and_refuses_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inventory = HyperPodInventory({"hp-cluster": HYPERPOD_EKS})
    monkeypatch.setattr(providers, "CommandRunner", lambda: inventory)
    monkeypatch.setattr(
        sys,
        "argv",
        ["boot029_providers", "--gpu-arn", HYPERPOD_EKS, "--gpu-arn", PLAIN_EKS],
    )

    providers.main()

    out = capsys.readouterr().out.splitlines()
    assert out == [f"hyperpod\t{HYPERPOD_EKS}", f"eks-managed\t{PLAIN_EKS}"], (
        "one tab-separated line per ARN, in the operator's order"
    )

    duplicate = HyperPodInventory({"hp-a": HYPERPOD_EKS, "hp-b": HYPERPOD_EKS})
    monkeypatch.setattr(providers, "CommandRunner", lambda: duplicate)
    monkeypatch.setattr(sys, "argv", ["boot029_providers", "--gpu-arn", HYPERPOD_EKS])
    with pytest.raises(SystemExit) as raised:
        providers.main()
    assert raised.value.code == 1, "a refusal exits non-zero"
    assert "BOOT-029 provider discovery refused:" in capsys.readouterr().err
