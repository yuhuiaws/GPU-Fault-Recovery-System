from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional.ha011_cpu_nodes import cpu_node
from scripts.e2e.regional.ha011_resources import CpuKubernetes
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)
from tests.regional._cov95_ha011_support import install_kubernetes, settings_at


@pytest.mark.parametrize("instance_type", ["c5.4xlarge", "ml.c5.4xlarge"])
def test_cpu_inventory_preserves_the_raw_instance_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, instance_type: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    node = fake.object("Node", "cpu-node")
    node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = instance_type
    original = copy.deepcopy(node)

    bound = cpu_node(CpuKubernetes(settings), "cpu-node")

    assert bound["instance_type"] == instance_type, (
        "family normalization must not rewrite the bound instance label"
    )
    assert bound["uid"] == original["metadata"]["uid"], (
        "CPU placement must remain bound to the observed node UID"
    )
    assert bound["inventory_sha256"] == contracts.digest(
        {
            "labels": original["metadata"]["labels"],
            "provider": original["spec"]["providerID"],
            "capacity": original["status"]["capacity"],
            "allocatable": original["status"]["allocatable"],
        }
    ), "the inventory digest must bind the complete raw labels and resources"
    assert node == original, "CPU classification must not modify node inventory"
    assert all(args[0] == "get" for _, args in fake.calls), (
        "CPU inventory binding must remain read-only"
    )


def test_plain_and_hyperpod_labels_produce_distinct_bound_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    labels = fake.object("Node", "cpu-node")["metadata"]["labels"]
    kubernetes = CpuKubernetes(settings)
    labels["node.kubernetes.io/instance-type"] = "c5.4xlarge"
    plain = cpu_node(kubernetes, "cpu-node")
    labels["node.kubernetes.io/instance-type"] = "ml.c5.4xlarge"
    hyperpod = cpu_node(kubernetes, "cpu-node")

    assert plain["inventory_sha256"] != hyperpod["inventory_sha256"], (
        "a raw instance-label change must invalidate the planned inventory digest"
    )
    assert contracts.digest(plain) != contracts.digest(hyperpod), (
        "equivalent CPU families must not collapse distinct bound identities"
    )


@pytest.mark.parametrize(
    "instance_type",
    [
        "ml.p5.48xlarge",
        "ml.inf1.6xlarge",
        "ml.inf2.48xlarge",
        "ml.unknown.4xlarge",
        "unknown.4xlarge",
        "ml.ml.c5.4xlarge",
        "ml..c5.4xlarge",
        "prefix.ml.c5.4xlarge",
        "mlc5.4xlarge",
        "ML.c5.4xlarge",
        "ml.C5.4xlarge",
        "ml.",
        "",
    ],
)
def test_accelerator_unknown_and_malformed_instance_families_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, instance_type: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    fake.object("Node", "cpu-node")["metadata"]["labels"][
        "node.kubernetes.io/instance-type"
    ] = instance_type

    with pytest.raises(contracts.ProofError, match="explicitly CPU-only"):
        cpu_node(CpuKubernetes(settings), "cpu-node")

    assert not fake.created, "unapproved instance families must fail before mutation"


@pytest.mark.parametrize("inventory", ["capacity", "allocatable"])
@pytest.mark.parametrize(
    "resource,value",
    [
        ("nvidia.com/gpu", "1"),
        ("nvidia.com/mig-1g.10gb", "1"),
        ("aws.amazon.com/neuron", "1"),
        ("example.com/fpga", "1"),
        ("nvidia.com/gpu", "unknown"),
        ("cpu", "2000m"),
        ("cpu", "NaN"),
    ],
)
def test_hyperpod_cpu_family_does_not_bypass_resource_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inventory: str,
    resource: str,
    value: str,
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    node = fake.object("Node", "cpu-node")
    node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = "ml.c5.4xlarge"
    node["status"][inventory][resource] = value

    with pytest.raises(contracts.ProofError):
        cpu_node(CpuKubernetes(settings), "cpu-node")

    assert not fake.created, "invalid resources must fail before CPU placement"


@pytest.mark.parametrize(
    "change",
    [
        "gpu-label",
        "not-ready",
        "region",
        "provider",
        "cordon",
        "taint",
        "future-lease",
        "stale-lease",
        "naive-lease",
        "lease-holder",
        "lease-owner",
    ],
)
def test_hyperpod_cpu_family_keeps_node_and_lease_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    node = fake.object("Node", "cpu-node")
    lease = fake.object("Lease", "cpu-node")
    node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = "ml.c5.4xlarge"
    if change == "gpu-label":
        node["metadata"]["labels"]["nvidia.com/gpu.product"] = "accelerator"
    elif change == "not-ready":
        node["status"]["conditions"][0]["status"] = "False"
    elif change == "region":
        node["metadata"]["labels"]["topology.kubernetes.io/region"] = "us-east-1"
    elif change == "provider":
        node["spec"]["providerID"] = "unknown://host"
    elif change == "cordon":
        node["spec"]["unschedulable"] = True
    elif change == "taint":
        node["spec"]["taints"] = [{"effect": "NoSchedule"}]
    elif change == "future-lease":
        lease["spec"]["renewTime"] = "2099-01-01T00:00:00+00:00"
    elif change == "stale-lease":
        lease["spec"]["renewTime"] = "2000-01-01T00:00:00+00:00"
    elif change == "naive-lease":
        lease["spec"]["renewTime"] = "2026-01-01T00:00:00"
    elif change == "lease-holder":
        lease["spec"]["holderIdentity"] = "other"
    else:
        lease["metadata"]["ownerReferences"] = [{"uid": "old-node"}]

    with pytest.raises(contracts.ProofError):
        cpu_node(CpuKubernetes(settings), "cpu-node")

    assert not fake.created, "node and lease guard failures must precede placement"
