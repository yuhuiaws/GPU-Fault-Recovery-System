"""A read-only discovery double for first deploys driven through the CLI.

The public ``deploy`` discovers the request's clusters
(``gpu_fault.admin.first_deploy_precheck``) before it records the request; a
test that runs a first deploy through ``admin_cli.run`` answers that discovery
here instead of reaching ``aws``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin.bootstrap_common import Arn, ClusterIdentity

SITE_SELECTOR = "gpu-fault.io/control-plane=example-site"


def discovered_identity(
    cluster_arn: str, *, role: str, context: str = "", provider: str = "hyperpod"
) -> ClusterIdentity:
    """What discovery returns for ``cluster_arn``: a HyperPod-orchestrated
    cluster, or a plain EKS one for ``provider="eks-managed"``. An EKS ARN and
    a SageMaker ARN with the same cluster name describe the same cluster."""

    arn = Arn.parse(cluster_arn)
    name = arn.resource_name
    scope = f"{arn.partition}:{{}}:{arn.region}:{arn.account}:cluster/"
    plain = provider == "eks-managed"
    return ClusterIdentity(
        input_arn=cluster_arn,
        role=role,
        region=arn.region,
        account_id=arn.account,
        hyperpod_arn="" if plain else "arn:" + scope.format("sagemaker") + name,
        hyperpod_name="" if plain else f"hp-{name}",
        eks_arn="arn:" + scope.format("eks") + name,
        eks_name=name,
        vpc_id=f"vpc-{name}",
        subnet_ids=(f"subnet-{name}",),
        node_recovery="" if plain else "None",
        context=context,
        provider=provider,
        node_selector=SITE_SELECTOR if role == "gpu" else "",
        node_selector_source="label-convention" if role == "gpu" else "",
    )


def stub_first_deploy_discovery(
    monkeypatch: pytest.MonkeyPatch, *, providers: Mapping[str, str] | None = None
) -> list[tuple[str, str]]:
    """Answer the CLI's ``discover_cluster``; returns ``(arn, role)`` per call.

    ``providers`` maps an ARN to ``eks-managed``; every other ARN is HyperPod.
    """

    asked: list[tuple[str, str]] = []

    def discover(
        _runner: Any, *, cluster_arn: str, role: str, context: str = "", **_options: Any
    ) -> ClusterIdentity:
        asked.append((cluster_arn, role))
        return discovered_identity(
            cluster_arn,
            role=role,
            context=context,
            provider=(providers or {}).get(cluster_arn, "hyperpod"),
        )

    monkeypatch.setattr(admin_cli, "discover_cluster", discover)
    return asked
