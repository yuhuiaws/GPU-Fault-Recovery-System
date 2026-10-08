"""The GPU kubeconfig entries a cluster owns: pruned by remove-cluster, reclaimed
by join-cluster only with proof.

``aws eks update-kubeconfig --alias`` names the context by the alias and the
cluster and user entries by the EKS ARN, so a first deploy's
``gpu-fault-gpu-1-<eks>`` alias and a join's ``gpu-fault-gpu-<eks>-<sha8>``
alias of the same cluster share one cluster and one user entry -- the shape the
live mixed site had when its first re-join met its own stale alias."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_kubeconfig as kube
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.cluster_removal_state import SCHEMA_VERSION, STEP_DEPENDENCIES
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file

ACCOUNT = "123456789012"
HP_EKS = f"arn:aws:eks:us-east-1:{ACCOUNT}:cluster/eks-hp"
EKS = f"arn:aws:eks:us-east-1:{ACCOUNT}:cluster/gpu-eks"
HP_FIRST_DEPLOY = "gpu-fault-gpu-1-eks-hp"
HP_JOIN = "gpu-fault-gpu-eks-hp-3fa3ebd3"
EKS_JOIN = "gpu-fault-gpu-gpu-eks-aa3805df"
HP_SERVER = "https://hp.example.invalid"
EKS_SERVER = "https://eks.example.invalid"
HP_CA = base64.b64encode(b"ca-hp").decode()
EKS_CA = base64.b64encode(b"ca-eks").decode()


def _kubeconfig(*, current: str = HP_JOIN) -> dict[str, Any]:
    """Three contexts over two clusters, as the live site's gpu.kubeconfig had."""

    return {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": current,
        "contexts": [
            {"name": HP_FIRST_DEPLOY, "context": {"cluster": HP_EKS, "user": HP_EKS}},
            {"name": EKS_JOIN, "context": {"cluster": EKS, "user": EKS}},
            {"name": HP_JOIN, "context": {"cluster": HP_EKS, "user": HP_EKS}},
        ],
        "clusters": [
            {
                "name": HP_EKS,
                "cluster": {"server": HP_SERVER, "certificate-authority-data": HP_CA},
            },
            {
                "name": EKS,
                "cluster": {"server": EKS_SERVER, "certificate-authority-data": EKS_CA},
            },
        ],
        "users": [
            {"name": HP_EKS, "user": {"exec": {"command": "aws"}}},
            {"name": EKS, "user": {"exec": {"command": "aws"}}},
        ],
    }


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "gpu.kubeconfig"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    return path


def _names(document: dict[str, Any], key: str) -> list[str]:
    return sorted(item["name"] for item in document.get(key, []))


def test_entries_of_a_cluster_cover_both_alias_schemes_and_only_unshared_bodies() -> (
    None
):
    document = _kubeconfig()

    hyperpod = kube.cluster_entries(document, context=HP_JOIN, eks_arn=HP_EKS)
    eks = kube.cluster_entries(document, context=EKS_JOIN, eks_arn=EKS)

    assert hyperpod.contexts == (HP_FIRST_DEPLOY, HP_JOIN), (
        "the first-deploy alias and the join alias of one cluster both belong to it"
    )
    assert hyperpod.clusters == (HP_EKS,) and hyperpod.users == (HP_EKS,), hyperpod
    assert eks == kube.KubeconfigEntries((EKS_JOIN,), (EKS,), (EKS,)), eks


def test_removing_one_cluster_keeps_the_other_clusters_entries_and_current_context(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _kubeconfig(current=HP_JOIN))
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    evidence = kube.prune_cluster_entries(
        path,
        context=HP_JOIN,
        eks_arn=HP_EKS,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        remaining_contexts={EKS_JOIN},
    )

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _names(document, "contexts") == [EKS_JOIN], "only the eks context remains"
    assert _names(document, "clusters") == [EKS] and _names(document, "users") == [EKS]
    assert document["current-context"] == EKS_JOIN, (
        "current-context moves to the remaining cluster instead of dangling"
    )
    assert evidence["contexts"] == [HP_FIRST_DEPLOY, HP_JOIN], evidence
    assert evidence["clusters"] == [HP_EKS] and evidence["users"] == [HP_EKS], evidence
    assert evidence["before_sha256"] == before, evidence
    assert evidence["after_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence["released"] is True and evidence["current_context"] == EKS_JOIN
    assert path.stat().st_mode & 0o777 == 0o600, "the kubeconfig stays private"

    again = kube.prune_cluster_entries(
        path,
        context=HP_JOIN,
        eks_arn=HP_EKS,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        remaining_contexts={EKS_JOIN},
    )
    assert again["contexts"] == [] and again["released"] is True, (
        "a second prune finds nothing of the cluster and changes nothing"
    )
    assert again["before_sha256"] == again["after_sha256"] == evidence["after_sha256"]


def test_removing_the_eks_cluster_leaves_both_hyperpod_aliases(tmp_path: Path) -> None:
    path = _write(tmp_path, _kubeconfig(current=EKS_JOIN))

    evidence = kube.prune_cluster_entries(
        path,
        context=EKS_JOIN,
        eks_arn=EKS,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        remaining_contexts={HP_JOIN},
    )

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _names(document, "contexts") == sorted([HP_FIRST_DEPLOY, HP_JOIN]), (
        "the HyperPod cluster's aliases are not the eks cluster's entries"
    )
    assert _names(document, "clusters") == [HP_EKS], document
    assert document["current-context"] == HP_JOIN, document
    assert evidence["contexts"] == [EKS_JOIN], evidence


def test_removing_the_last_cluster_drops_current_context(tmp_path: Path) -> None:
    document = _kubeconfig(current=EKS_JOIN)
    document["contexts"] = [document["contexts"][1]]
    document["clusters"] = [document["clusters"][1]]
    document["users"] = [document["users"][1]]
    path = _write(tmp_path, document)

    kube.prune_cluster_entries(
        path,
        context=EKS_JOIN,
        eks_arn=EKS,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        remaining_contexts=set(),
    )

    pruned = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "current-context" not in pruned, "no context is left to be current"
    assert (
        pruned["contexts"] == [] and pruned["clusters"] == [] and pruned["users"] == []
    )


def test_prune_refuses_the_cpu_kubeconfig_and_a_shared_remaining_context(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _kubeconfig())
    before = path.read_bytes()

    with pytest.raises(BootstrapError, match="CPU kubeconfig"):
        kube.prune_cluster_entries(
            path,
            context=HP_JOIN,
            eks_arn=HP_EKS,
            cpu_kubeconfig=path,
            remaining_contexts=set(),
        )
    with pytest.raises(BootstrapError, match="remaining cluster still uses"):
        kube.prune_cluster_entries(
            path,
            context=HP_JOIN,
            eks_arn=HP_EKS,
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            remaining_contexts={HP_FIRST_DEPLOY},
        )
    assert path.read_bytes() == before, "a refusal changes nothing"


def test_a_missing_or_foreign_kubeconfig_is_handled_without_guessing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.kubeconfig"
    evidence = kube.prune_cluster_entries(
        missing,
        context=HP_JOIN,
        eks_arn=HP_EKS,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        remaining_contexts=set(),
    )
    assert evidence["released"] is True and evidence["before_sha256"] is None, evidence
    assert not missing.exists(), "nothing is created for a missing kubeconfig"
    broken = tmp_path / "broken.kubeconfig"
    broken.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(BootstrapError, match="one mapping"):
        kube.load_kubeconfig(broken)


def completed_removal_journal(
    state_dir: Path, *, cluster_id: str, context: str, eks_arn: str
) -> Path:
    """A COMPLETED remove-cluster journal that read_removal_state accepts."""

    target = {
        "cluster_id": cluster_id,
        "context": context,
        "eks_cluster_arn": eks_arn,
        "hyperpod_cluster_name": "hp-" + cluster_id,
        "token_file": str(state_dir / "secure" / f"{cluster_id}.token"),
    }
    steps = sorted(STEP_DEPENDENCIES)
    journal = {
        "schema_version": SCHEMA_VERSION,
        "site_id": "test-site",
        "cluster_id": cluster_id,
        "site_identity": {},
        "target": target,
        "attempt_id": "a" * 32,
        "phase": "COMPLETED",
        "completed_steps": steps,
        "step_completed_at": {step: "2026-01-01T00:00:00+00:00" for step in steps},
        "evidence": {
            "DISCOVERED": {
                "target": target,
                "provider_identity": {
                    "eks_arn": eks_arn,
                    "hyperpod_arn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:cluster/hp",
                },
                "namespace_uid": "namespace-uid",
                "registry_digest": "d" * 64,
                "token_sha256": "t" * 64,
            }
        },
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    path = state_dir / "remove-cluster" / cluster_id / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(journal), encoding="utf-8")
    return path


class DescribeCluster:
    """The one AWS read the alias proof makes."""

    def __init__(self, *, endpoint: str = EKS_SERVER, authority: str = EKS_CA) -> None:
        self.endpoint = endpoint
        self.authority = authority
        self.calls: list[tuple[str, ...]] = []

    def aws_json(self, region: str, *arguments: str, **_options: Any) -> dict[str, Any]:
        self.calls.append((region, *arguments))
        assert arguments[:2] == ("eks", "describe-cluster"), "only describe-cluster"
        return {
            "cluster": {
                "arn": EKS,
                "endpoint": self.endpoint,
                "certificateAuthority": {"data": self.authority},
            }
        }


def _eks_target() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=EKS,
        role="gpu",
        region="us-east-1",
        account_id=ACCOUNT,
        hyperpod_arn="",
        hyperpod_name="",
        eks_arn=EKS,
        eks_name="gpu-eks",
        vpc_id="vpc-eks",
        subnet_ids=("subnet-a",),
        node_recovery="",
        context=EKS_JOIN,
        provider="eks-managed",
        node_selector="eks.amazonaws.com/nodegroup=gpu",
    )


def test_a_released_alias_of_the_same_cluster_is_reclaimed_with_proof(
    tmp_path: Path,
) -> None:
    """Joined, removed, joined again: the identical alias is taken over when the
    kubeconfig cluster entry proves the same endpoint and CA and the removal
    journal is COMPLETED for this EKS cluster."""
    site = load_site(site_file(tmp_path))
    path = _write(tmp_path, _kubeconfig())
    completed_removal_journal(
        tmp_path, cluster_id="gpu-eks", context=EKS_JOIN, eks_arn=EKS
    )
    runner = DescribeCluster()

    evidence = kube.reclaim_released_alias(
        runner,  # type: ignore[arg-type]
        site=site,
        target=_eks_target(),
        cluster_id="gpu-eks",
        kubeconfig=path,
        registry_states=lambda: pytest.fail("the journal already proves release"),
    )

    assert evidence["context"] == EKS_JOIN and evidence["cluster_entry"] == EKS, (
        evidence
    )
    assert evidence["endpoint"] == EKS_SERVER, evidence
    assert (
        evidence["certificate_authority_sha256"]
        == hashlib.sha256(EKS_CA.encode()).hexdigest()
    ), "the CA is recorded as a digest, never in clear"
    assert evidence["released_by"] == {
        "remove_cluster_attempt": "a" * 32,
        "completed_at": "2026-01-01T00:00:00+00:00",
    }, evidence
    assert runner.calls == [
        ("us-east-1", "eks", "describe-cluster", "--name", "gpu-eks")
    ]


def test_a_first_deployed_then_removed_cluster_reclaims_through_an_absent_registry(
    tmp_path: Path,
) -> None:
    """No removal journal under the join's cluster id (the first deploy named the
    cluster differently): the control-plane registry holding no registration is
    the other proof of release."""
    site = load_site(site_file(tmp_path))
    path = _write(tmp_path, _kubeconfig())

    evidence = kube.reclaim_released_alias(
        DescribeCluster(),  # type: ignore[arg-type]
        site=site,
        target=_eks_target(),
        cluster_id="gpu-eks",
        kubeconfig=path,
        registry_states=lambda: {"gpu-a": "ACTIVE"},
    )

    assert evidence["released_by"] == {"registry": "absent"}, evidence


@pytest.mark.parametrize(
    "shape,message",
    [
        ("other-endpoint", "another server endpoint"),
        ("other-ca", "another certificate authority"),
        ("registered", "still registered with the control plane"),
        ("managed", "still managed by the site"),
        ("no-cluster-entry", "no cluster entry"),
    ],
)
def test_an_alias_without_proof_or_release_stays_refused(
    tmp_path: Path, shape: str, message: str
) -> None:
    site = load_site(site_file(tmp_path))
    document = _kubeconfig()
    if shape == "no-cluster-entry":
        document["clusters"] = [document["clusters"][0]]
    path = _write(tmp_path, document)
    runner = DescribeCluster(
        endpoint="https://other.example.invalid"
        if shape == "other-endpoint"
        else EKS_SERVER,
        authority=base64.b64encode(b"other").decode()
        if shape == "other-ca"
        else EKS_CA,
    )
    cluster_id = "gpu-a" if shape == "managed" else "gpu-eks"
    before = path.read_bytes()

    with pytest.raises(kube.KubeconfigAliasRefused, match=message):
        kube.reclaim_released_alias(
            runner,  # type: ignore[arg-type]
            site=site,
            target=_eks_target(),
            cluster_id=cluster_id,
            kubeconfig=path,
            registry_states=lambda: {"gpu-eks": "ACTIVE"},
        )
    assert path.read_bytes() == before, "a refusal never rewrites the kubeconfig"
