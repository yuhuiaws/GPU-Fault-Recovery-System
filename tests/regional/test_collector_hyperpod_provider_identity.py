"""HyperPod provider IDs bind the ARN resource, not ordinary EKS AZ names."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from scripts.e2e.regional.collector_reboot_evidence import (
    RebootEvidenceError,
    capture_reboot_scope,
)
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from tests.regional.test_collector_reboot_evidence import (
    HP,
    HP_ARN,
    INSTANCE,
    NODE,
    REGION,
    ROLE,
    ReadOnlyRegional,
)


def hyperpod_region() -> ReadOnlyRegional:
    regional = ReadOnlyRegional()
    regional.data["node"]["spec"]["providerID"] = (
        "aws:///usw2-az3/sagemaker/cluster/hyperpod-"
        f"{HP_ARN.rsplit('/', 1)[-1]}-{INSTANCE}"
    )
    regional.data["node"]["metadata"]["labels"] = {
        "topology.kubernetes.io/region": REGION,
        "topology.k8s.aws/zone-id": "usw2-az3",
        "sagemaker.amazonaws.com/cluster-name": HP,
    }
    return regional


def test_native_hyperpod_provider_id_maps_to_exact_arn_and_instance() -> None:
    regional = hyperpod_region()
    result = capture_reboot_scope(
        cast(RegionalLiveFixture, regional),
        node=NODE,
        hyperpod_cluster=HP,
        executor_role_arn=ROLE,
    )
    assert result["instance_id"] == INSTANCE
    assert result["cluster_arn"] == HP_ARN
    assert result["node_provider_id"] == regional.data["node"]["spec"]["providerID"]


@pytest.mark.parametrize(
    "provider",
    [
        f"aws:///usw2-az3/sagemaker/cluster/hyperpod-foreign-{INSTANCE}",
        f"aws:///usw2-az0/sagemaker/cluster/hyperpod-hp-unique-resource-{INSTANCE}",
        "aws:///usw2-az3/sagemaker/cluster/hyperpod-hp-unique-resource-i-short",
        f"aws:///usw2-az3/other/cluster/hyperpod-hp-unique-resource-{INSTANCE}",
        f"aws:///usw2-az3/sagemaker/cluster/hyperpod-hp-unique-resource-{INSTANCE}/other",
    ],
)
def test_other_hyperpod_paths_do_not_supply_instance_authority(provider: str) -> None:
    regional = hyperpod_region()
    regional.data["node"]["spec"]["providerID"] = provider
    with pytest.raises(RebootEvidenceError, match="providerID"):
        capture_reboot_scope(
            cast(RegionalLiveFixture, regional),
            node=NODE,
            hyperpod_cluster=HP,
            executor_role_arn=ROLE,
        )


@pytest.mark.parametrize(
    "label",
    [
        "topology.kubernetes.io/region",
        "topology.k8s.aws/zone-id",
        "sagemaker.amazonaws.com/cluster-name",
    ],
)
@pytest.mark.parametrize("present", [False, True])
def test_hyperpod_provider_topology_must_agree(label: str, present: bool) -> None:
    regional = hyperpod_region()
    labels = regional.data["node"]["metadata"]["labels"]
    if present:
        labels[label] = "other"
    else:
        labels.pop(label)
    with pytest.raises(RebootEvidenceError, match="topology"):
        capture_reboot_scope(
            cast(RegionalLiveFixture, regional),
            node=NODE,
            hyperpod_cluster=HP,
            executor_role_arn=ROLE,
        )


def test_provider_id_rewrite_is_rejected_even_if_instance_suffix_is_unchanged() -> None:
    regional = hyperpod_region()
    changed = deepcopy(regional.data["node"])
    changed["spec"]["providerID"] = f"aws:///{REGION}a/{INSTANCE}"
    regional.second["node"] = changed
    with pytest.raises(RebootEvidenceError, match="Node identity changed"):
        capture_reboot_scope(
            cast(RegionalLiveFixture, regional),
            node=NODE,
            hyperpod_cluster=HP,
            executor_role_arn=ROLE,
        )
