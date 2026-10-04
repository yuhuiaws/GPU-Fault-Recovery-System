"""Lifecycle paths beyond join for a plain EKS (``eks-managed``) GPU cluster.

Failure-domain mapping, node-key provisioning, ``remove-cluster`` identity and
registry proofs, ``uninstall`` registry validation and the installation registry
builder all learned that a GPU cluster may have no HyperPod behind it. Every
case keeps a HyperPod twin that must behave exactly as before.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import cluster_removal_preserved as preserved
from gpu_fault.admin import failure_domain_map as maps
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin import uninstall
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal import RemoveClusterRequest
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall_types import UninstallRequest
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin.test_admin_site import site_file
from tests.deploy.test_node_action_key_provisioning import MODULE as KEYS

REGION = "us-east-1"
ACCOUNT = "123456789012"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-e"
HYPERPOD_SELECTOR = "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
EKS_SELECTOR = "nvidia.com/gpu.present=true"
FQDN = "ip-10-0-33-205.us-west-2.compute.internal"


def _mixed_site(tmp_path: Path):
    """The test site plus one plain EKS cluster ``gpu-e``."""

    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    hyperpod = document["spec"]["clusters"][0]
    eks = {
        key: value for key, value in hyperpod.items() if key != "hyperpodClusterName"
    }
    eks.update(
        {
            "clusterId": "gpu-e",
            "context": "gpu-e",
            "provider": "eks-managed",
            "nodeSelector": EKS_SELECTOR,
            "eksClusterArn": EKS_ARN,
            "executorIrsaRoleArn": f"arn:aws:iam::{ACCOUNT}:role/executor-e",
        }
    )
    document["spec"]["clusters"].append(eks)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_site(path)


def _entry(site, cluster_id: str) -> dict[str, Any]:
    return next(
        dict(item)
        for item in site.release_config["clusters"]
        if item["cluster_id"] == cluster_id
    )


def _node(name: str, labels: dict[str, str]) -> dict[str, Any]:
    return {
        "kind": "Node",
        "apiVersion": "v1",
        "metadata": {"name": name, "uid": "uid-" + name, "labels": labels},
    }


EKS_LABELS = {
    "nvidia.com/gpu.present": "true",
    "node-type": "gpu",
    "eks.amazonaws.com/nodegroup": "gpu-p5en-efa-spot-node-group",
    "topology.kubernetes.io/zone": "us-west-2c",
}
CPU_LABELS = {"node-type": "cpu", "eks.amazonaws.com/nodegroup": "cpu-node-group"}


# --- failure-domain map ---------------------------------------------------------------


def test_failure_domain_listing_selects_each_provider_on_its_own_selector(
    tmp_path: Path,
) -> None:
    site = _mixed_site(tmp_path)
    calls: list[list[str]] = []

    def kubectl(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(list(args))
        context = args[args.index("--context") + 1]
        selector = args[args.index("-l") + 1]
        if context == "gpu-e":
            items = [_node(FQDN, EKS_LABELS)]
        else:
            items = [_node("hyperpod-i-1", {maps.HYPERPOD_CLUSTER_LABEL: "hp-gpu-a"})]
        assert selector in {HYPERPOD_SELECTOR, EKS_SELECTOR}, selector
        return subprocess.CompletedProcess(args, 0, json.dumps({"items": items}), "")

    eks_nodes = maps.cluster_nodes(site, "gpu-e", run=kubectl)
    hyperpod_nodes = maps.cluster_nodes(site, "gpu-a", run=kubectl)

    assert [args[-2:] for args in calls] == [
        ["-l", EKS_SELECTOR],
        ["-l", HYPERPOD_SELECTOR],
    ], "each cluster must be listed on its own provider selector"
    assert set(eks_nodes) == {FQDN}, "the FQDN GPU node is the eks-managed inventory"
    assert set(hyperpod_nodes) == {"hyperpod-i-1"}, "HyperPod inventory unchanged"
    result = maps.build_failure_domain_map(
        site,
        list_nodes=lambda current, cluster_id: maps.cluster_nodes(
            current, cluster_id, run=kubectl
        ),
    )
    assert result.mapping["gpu-e"] == {FQDN: "gpu-p5en-efa-spot-node-group"}, (
        "an eks-managed node maps to its node group, the instance-group analogue"
    )
    assert result.label_keys[:2] == (
        "sagemaker.amazonaws.com/instance-group-name",
        "eks.amazonaws.com/nodegroup",
    ), "a mixed site reads the node group label after the instance group label"


def test_failure_domain_listing_refuses_a_node_the_selector_does_not_match(
    tmp_path: Path,
) -> None:
    site = _mixed_site(tmp_path)

    def kubectl(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        # A proxy that ignored the selector and returned the CPU node too.
        items = [_node(FQDN, EKS_LABELS), _node("ip-10-0-1-1.internal", CPU_LABELS)]
        return subprocess.CompletedProcess(args, 0, json.dumps({"items": items}), "")

    with pytest.raises(BootstrapError, match="conflicting node identity"):
        maps.cluster_nodes(site, "gpu-e", run=kubectl)


# --- node action keys -------------------------------------------------------------------


def _key_runner(items: list[dict[str, Any]], calls: list[list[str]]):
    def runner(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(list(arguments))
        document = {
            "apiVersion": "v1",
            "kind": "NodeList",
            "metadata": {},
            "items": items,
        }
        return subprocess.CompletedProcess(arguments, 0, json.dumps(document), "")

    return runner


def test_node_key_inventory_uses_the_selector_and_keeps_the_hyperpod_default() -> None:
    calls: list[list[str]] = []
    client = KEYS.SecretClient(
        KEYS.Scope(("kubectl", "--context", "gpu-e"), "gpu-fault-system"),
        _key_runner([_node(FQDN, EKS_LABELS)], calls),
    )
    assert client.nodes("gpu-e", EKS_SELECTOR) == {FQDN: "uid-" + FQDN}, (
        "an eks-managed inventory is keyed by the FQDN node name"
    )
    assert calls[-1][calls[-1].index("-l") + 1] == EKS_SELECTOR, (
        "the selector must be pushed to the API server"
    )
    hyperpod_calls: list[list[str]] = []
    hyperpod = KEYS.SecretClient(
        KEYS.Scope(("kubectl",), "gpu-fault-system"),
        _key_runner(
            [_node("hyperpod-i-1", {maps.HYPERPOD_CLUSTER_LABEL: "hp-gpu-a"})],
            hyperpod_calls,
        ),
    )
    assert hyperpod.nodes("hp-gpu-a") == {"hyperpod-i-1": "uid-hyperpod-i-1"}, (
        "HyperPod inventory must still work without a selector argument"
    )
    assert (
        hyperpod_calls[-1][hyperpod_calls[-1].index("-l") + 1] == HYPERPOD_SELECTOR
    ), "the HyperPod default selector is the historical label text"


def test_node_key_inventory_rejects_nodes_outside_the_selector() -> None:
    client = KEYS.SecretClient(
        KEYS.Scope(("kubectl",), "gpu-fault-system"),
        _key_runner([_node(FQDN, EKS_LABELS), _node("cpu-1", CPU_LABELS)], []),
    )
    with pytest.raises(KEYS.ProvisionError, match="invalid or ambiguous"):
        client.nodes("gpu-e", EKS_SELECTOR)


# --- remove-cluster -------------------------------------------------------------------------


def _resource(
    site_id: str,
    key: str,
    resource_type: str,
    resource_id: str,
    *,
    arn: str | None = None,
    policy: InstallationResourceDeletePolicy = InstallationResourceDeletePolicy.PRESERVE,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id=site_id,
        resource_key=key,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_arn=arn,
        region=REGION,
        account_id=ACCOUNT,
        ownership=(
            InstallationResourceOwnership.CREATED
            if policy is InstallationResourceDeletePolicy.DELETE
            else InstallationResourceOwnership.EXTERNAL
        ),
        delete_policy=policy,
        created_at=now,
        updated_at=now,
    )


def _snapshot(
    site_id: str, *resources: InstallationResource
) -> InstallationResourceSnapshot:
    value = InstallationResourceSnapshot(site_id=site_id, resources=list(resources))
    return value.model_copy(update={"source_sha256": value.digest()})


def _eks_registry(site, *, with_hyperpod: bool = False) -> InstallationResourceSnapshot:
    site_id = site.registry_site_id
    resources = [
        _resource(site_id, "cluster/gpu-e/eks", "gpu_eks", "gpu-e", arn=EKS_ARN),
        _resource(
            site_id,
            "aws/iam/executor/gpu-e/role",
            "iam_role",
            "executor-e",
            arn=f"arn:aws:iam::{ACCOUNT}:role/executor-e",
            policy=InstallationResourceDeletePolicy.DELETE,
        ),
    ]
    if with_hyperpod:
        resources.append(
            _resource(site_id, "cluster/gpu-e/hyperpod", "gpu_hyperpod", "ghost")
        )
    return _snapshot(site_id, *resources)


def test_removal_registry_proof_accepts_a_plain_eks_cluster_without_hyperpod(
    tmp_path: Path,
) -> None:
    site = _mixed_site(tmp_path)
    request = RemoveClusterRequest(
        site=site, cluster_id="gpu-e", confirmation=removal.CONFIRMATION
    )
    target = _entry(site, "gpu-e")
    identity = {"eks_arn": EKS_ARN, "hyperpod_arn": ""}

    removal.validate_target_registry(request, target, _eks_registry(site), identity)

    with pytest.raises(BootstrapError, match="identity is incomplete"):
        removal.validate_target_registry(
            request, target, _eks_registry(site, with_hyperpod=True), identity
        )
    with pytest.raises(BootstrapError, match="names a HyperPod ARN"):
        removal.validate_target_registry(
            request,
            target,
            _eks_registry(site),
            {"eks_arn": EKS_ARN, "hyperpod_arn": "arn:aws:sagemaker:x:y:cluster/z"},
        )


def test_removal_registry_proof_still_requires_the_hyperpod_record_for_hyperpod(
    tmp_path: Path,
) -> None:
    site = _mixed_site(tmp_path)
    request = RemoveClusterRequest(
        site=site, cluster_id="gpu-a", confirmation=removal.CONFIRMATION
    )
    target = _entry(site, "gpu-a")
    site_id = site.registry_site_id
    without_hyperpod = _snapshot(
        site_id,
        _resource(
            site_id,
            "cluster/gpu-a/eks",
            "gpu_eks",
            "gpu-a",
            arn=target["eks_cluster_arn"],
        ),
        _resource(
            site_id,
            "aws/iam/executor/gpu-a/role",
            "iam_role",
            "executor",
            arn=target["executor_irsa_role_arn"],
            policy=InstallationResourceDeletePolicy.DELETE,
        ),
    )
    with pytest.raises(BootstrapError, match="identity is incomplete"):
        removal.validate_target_registry(
            request,
            target,
            without_hyperpod,
            {"eks_arn": target["eks_cluster_arn"], "hyperpod_arn": "arn:x"},
        )


def test_removal_identity_of_a_plain_eks_cluster_never_describes_sagemaker(
    tmp_path: Path, monkeypatch
) -> None:
    site = _mixed_site(tmp_path)
    request = RemoveClusterRequest(
        site=site,
        cluster_id="gpu-e",
        confirmation=removal.CONFIRMATION,
        gpu_cluster_arn=EKS_ARN,
    )
    target = _entry(site, "gpu-e")
    aws_calls: list[tuple[str, str]] = []

    class Runner:
        def aws_json(self, _region, service, operation, *_arguments, **_options):
            aws_calls.append((service, operation))
            raise AssertionError("plain EKS removal identity must not call AWS")

    def run_command(arguments, **_kwargs):
        if "config" in arguments:
            cpu = "--context" not in arguments
            return subprocess.CompletedProcess(
                arguments, 0, f"https://{'cpu' if cpu else 'gpu'}.example", ""
            )
        if "namespace" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps({"metadata": {"uid": "ns-uid"}}), ""
            )
        raise AssertionError(f"unexpected command {arguments}")

    monkeypatch.setattr(removal, "run_command", run_command)
    monkeypatch.setattr(
        removal,
        "_namespace_document",
        lambda _kubectl, _ns: {"metadata": {"uid": "ns"}},
    )
    networks = {
        "eks_endpoint": "https://gpu.example",
        "eks_created_at": "2026-01-01T00:00:00Z",
        "vpc_id": "vpc-gpu-e",
    }
    cpu_network = {
        "eks_endpoint": "https://cpu.example",
        "eks_created_at": "cpu-created",
        "vpc_id": "vpc-cpu",
    }
    identity, namespace_uid = removal.removal_identity(
        request, target, Runner(), networks, cpu_network
    )
    assert aws_calls == [], "no sagemaker describe may run for a plain EKS cluster"
    assert identity["hyperpod_arn"] == "", "the identity records no HyperPod ARN"
    assert identity["eks_arn"] == EKS_ARN, "the EKS ARN is the live identity"
    assert namespace_uid == "ns", "the GPU namespace identity is still read"
    with pytest.raises(BootstrapError, match="differs from the live cluster identity"):
        removal.removal_identity(
            RemoveClusterRequest(
                site=site,
                cluster_id="gpu-e",
                confirmation=removal.CONFIRMATION,
                gpu_cluster_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/x",
            ),
            target,
            Runner(),
            networks,
            cpu_network,
        )


def test_preserved_hyperpod_check_is_skipped_for_a_plain_eks_cluster(
    tmp_path: Path, monkeypatch
) -> None:
    site = _mixed_site(tmp_path)
    request = RemoveClusterRequest(
        site=site, cluster_id="gpu-e", confirmation=removal.CONFIRMATION
    )
    monkeypatch.setattr(
        preserved,
        "json_command",
        lambda *_a, **_k: pytest.fail("a plain EKS removal must not describe HyperPod"),
    )
    preserved.verify_preserved_hyperpod(request, _entry(site, "gpu-e"))


# --- uninstall ----------------------------------------------------------------------------------


def test_uninstall_registry_validation_accepts_a_site_with_a_plain_eks_cluster(
    tmp_path: Path,
) -> None:
    site = _mixed_site(tmp_path)
    request = UninstallRequest(
        site=site, cpu_disposition="keep", confirmation="UNINSTALL_GPU_FAULT"
    )
    config = site.release_config
    site_id = site.registry_site_id
    gpu_a = _entry(site, "gpu-a")
    rows = [
        _resource(
            site_id, "cluster/cpu-eks", "cpu_eks", "control", arn=config["cpu_eks_arn"]
        ),
        _resource(site_id, "cluster/cpu-hyperpod", "cpu_hyperpod", "control"),
        _resource(
            site_id,
            "aws/aurora/cluster",
            "aurora_cluster",
            config["health"]["aurora_cluster_id"],
            policy=InstallationResourceDeletePolicy.DELETE,
        ),
        _resource(
            site_id,
            "cluster/gpu-a/eks",
            "gpu_eks",
            "gpu-a",
            arn=gpu_a["eks_cluster_arn"],
        ),
        _resource(site_id, "cluster/gpu-a/hyperpod", "gpu_hyperpod", "hp-gpu-a"),
        _resource(site_id, "cluster/gpu-e/eks", "gpu_eks", "gpu-e", arn=EKS_ARN),
    ]
    uninstall.validate_registry(request, _snapshot(site_id, *rows))
    # The HyperPod cluster still needs its gpu_hyperpod row.
    with pytest.raises(BootstrapError, match="cluster identity differs from site"):
        uninstall.validate_registry(
            request,
            _snapshot(site_id, *[row for row in rows if row.resource_id != "hp-gpu-a"]),
        )


# --- installation registry builder --------------------------------------------------------------


def test_registry_builder_records_no_hyperpod_for_a_plain_eks_cluster(
    tmp_path: Path,
) -> None:
    site = _mixed_site(tmp_path)
    rows = registry.external_cluster_resources(
        site, site_id="test-site", region=REGION, account_id=ACCOUNT
    )
    keys = {row.resource_key for row in rows}
    assert "cluster/gpu-e/eks" in keys, "the plain EKS cluster itself is recorded"
    assert "cluster/gpu-e/hyperpod" not in keys, "no gpu_hyperpod row for plain EKS"
    assert {"cluster/gpu-a/eks", "cluster/gpu-a/hyperpod"} <= keys, (
        "the HyperPod cluster keeps both of its rows"
    )
    hyperpod_row = next(
        row for row in rows if row.resource_key == "cluster/gpu-a/hyperpod"
    )
    assert hyperpod_row.resource_id == "hp-gpu-a", "HyperPod row content unchanged"


def test_removal_state_accepts_a_plain_eks_discovery_without_hyperpod_arn() -> None:
    from gpu_fault.admin import cluster_removal_state as state_module

    assert hasattr(state_module, "is_eks_managed"), (
        "the removal journal must consult the provider before demanding a HyperPod ARN"
    )
    assert state_module.is_eks_managed({"provider": "eks-managed"}), "provider read"
    assert not state_module.is_eks_managed({"hyperpod_cluster_name": "hp"}), (
        "a HyperPod target still demands its HyperPod ARN"
    )
    _ = SimpleNamespace  # keep the import honest for future state fixtures
