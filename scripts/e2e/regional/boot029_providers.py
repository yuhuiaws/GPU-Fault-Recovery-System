"""Which provider each GPU cluster of a BOOT-029 run stands on.

A first ``gpu-fault-admin deploy`` writes one site Runtime Profile for every
GPU cluster it creates, so its GPU clusters must share one provider; a cluster
of the other provider joins afterwards with ``join-cluster`` and gets its own
profile version (``bootstrap_common.require_single_provider``). The driver
therefore has to know, before stage 1 and stage 5, which of the operator's
``--gpu-cluster-arn`` values are HyperPod clusters and which are plain EKS
(``eks-managed``) clusters -- and an EKS ARN alone does not say: a HyperPod
cluster is named by its orchestrator EKS ARN just like a plain EKS cluster.

The answer is the product's own read (``bootstrap_discovery.find_hyperpod_for_eks``,
``sagemaker list-clusters`` plus one ``describe-cluster`` per HyperPod cluster of
the Region, read-only): an EKS ARN that exactly one HyperPod cluster names as
its orchestrator is ``hyperpod``, one that none names is ``eks-managed``, and a
SageMaker ARN is ``hyperpod`` without a read. Nothing is guessed from names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Sequence

from gpu_fault.admin.bootstrap_common import Arn, BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_discovery import find_hyperpod_for_eks
from gpu_fault.cluster_scope import ClusterProvider

HYPERPOD = ClusterProvider.HYPERPOD.value
EKS_MANAGED = ClusterProvider.EKS_MANAGED.value


def gpu_cluster_provider(arn: str, *, runner: CommandRunner | None = None) -> str:
    """``hyperpod`` or ``eks-managed`` for one GPU cluster ARN, as deploy sees it."""

    parsed = Arn.parse(arn)
    if parsed.service == "sagemaker":
        return HYPERPOD
    if parsed.service != "eks":
        raise BootstrapError(
            f"a GPU cluster is identified by its EKS or SageMaker ARN, not "
            f"{parsed.service}"
        )
    hyperpod = find_hyperpod_for_eks(
        runner or CommandRunner(), eks_arn=arn, region=parsed.region
    )
    return HYPERPOD if hyperpod is not None else EKS_MANAGED


def gpu_cluster_providers(
    arns: Sequence[str], *, runner: CommandRunner | None = None
) -> list[tuple[str, str]]:
    """``(provider, arn)`` for every ARN, in the operator's order."""

    if not arns:
        raise BootstrapError("at least one GPU cluster ARN is required")
    active = runner or CommandRunner()
    return [(gpu_cluster_provider(arn, runner=active), arn) for arn in arns]


@dataclass(frozen=True)
class ProviderGroups:
    """The run's GPU clusters as the first deploy and the later joins see them.

    ``first_deploy`` holds every cluster of the first ARN's provider, in the
    operator's order; ``joined`` holds one ``(provider, arns)`` group per
    further provider, in order of first appearance. A single-provider list has
    no ``joined`` group and the first deploy names every cluster, exactly as
    before mixed sites existed.
    """

    first_provider: str
    first_deploy: tuple[str, ...]
    joined: tuple[tuple[str, tuple[str, ...]], ...] = field(default_factory=tuple)

    @property
    def joined_providers(self) -> tuple[str, ...]:
        return tuple(provider for provider, _ in self.joined)

    @property
    def joined_arns(self) -> tuple[str, ...]:
        return tuple(arn for _, arns in self.joined for arn in arns)


def provider_groups(pairs: Sequence[tuple[str, str]]) -> ProviderGroups:
    if not pairs:
        raise BootstrapError("at least one GPU cluster ARN is required")
    first_provider = pairs[0][0]
    by_provider: dict[str, list[str]] = {}
    for provider, arn in pairs:
        by_provider.setdefault(provider, []).append(arn)
    return ProviderGroups(
        first_provider=first_provider,
        first_deploy=tuple(by_provider.pop(first_provider)),
        joined=tuple((provider, tuple(arns)) for provider, arns in by_provider.items()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-arn",
        action="append",
        required=True,
        help="GPU cluster ARN, once per managed GPU cluster in the run's order",
    )
    args = parser.parse_args()
    try:
        pairs = gpu_cluster_providers(list(args.gpu_arn))
    except (BootstrapError, OSError, ValueError) as exc:
        parser.exit(1, f"BOOT-029 provider discovery refused: {exc}\n")
    for provider, arn in pairs:
        print(f"{provider}\t{arn}")


if __name__ == "__main__":
    main()
