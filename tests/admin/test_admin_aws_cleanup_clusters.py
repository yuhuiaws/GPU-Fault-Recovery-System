from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)
from tests.admin._aws_cleanup_support import ACCOUNT, REGION, Aws, absent, resource
from tests.admin.test_admin_site import site_file

CPU_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/control"


@pytest.fixture
def site(tmp_path: Path) -> RenderedSite:
    return load_site(site_file(tmp_path))


def cpu_records() -> tuple[InstallationResource, InstallationResource]:
    return (
        resource(
            "cpu_hyperpod",
            "control",
            policy=InstallationResourceDeletePolicy.PRESERVE,
            ownership=InstallationResourceOwnership.EXTERNAL,
        ),
        resource(
            "cpu_eks",
            "control",
            arn=CPU_ARN,
            policy=InstallationResourceDeletePolicy.PRESERVE,
            ownership=InstallationResourceOwnership.EXTERNAL,
        ),
    )


class CpuAws(Aws):
    def __init__(self) -> None:
        self.cluster: dict[str, Any] = {
            "name": "control",
            "arn": CPU_ARN,
            "createdAt": "2026-01-01T00:00:00Z",
            "status": "ACTIVE",
        }
        self.hyperpod: dict[str, Any] = {
            "ClusterName": "control",
            "ClusterArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/cpu-id",
            "ClusterStatus": "InService",
            "Orchestrator": {"Eks": {"ClusterArn": CPU_ARN}},
        }
        self.children: dict[str, dict[str, str]] = {
            "nodegroup": {"workers-a": "ACTIVE", "workers-b": "ACTIVE"},
            "fargate-profile": {"profile-a": "ACTIVE", "profile-b": "ACTIVE"},
            "addon": {"eks-pod-identity-agent": "ACTIVE"},
        }
        self.deleted_children: list[str] = []
        self.hp_gone = False
        self.eks_gone = False
        self.recreate_eks = False
        self.deleting_hp_reads = 0
        self.deleting_eks_reads = 0
        super().__init__(
            {
                ("sagemaker", "describe-cluster"): self.describe_hyperpod,
                ("sagemaker", "delete-cluster"): self.delete_hyperpod,
                ("eks", "describe-cluster"): self.describe_cluster,
                ("eks", "delete-cluster"): self.delete_cluster,
                ("eks", "list-pod-identity-associations"): {"associations": []},
                ("eks", "list-nodegroups"): lambda _args: {
                    "nodegroups": list(self.children["nodegroup"])
                },
                ("eks", "list-fargate-profiles"): lambda _args: {
                    "fargateProfileNames": list(self.children["fargate-profile"])
                },
                ("eks", "list-addons"): lambda _args: {
                    "addons": list(self.children["addon"])
                },
                **{
                    ("eks", f"describe-{kind}"): self.describe_child
                    for kind in self.children
                },
                **{
                    ("eks", f"delete-{kind}"): self.delete_child
                    for kind in self.children
                },
            }
        )

    def describe_cluster(self, _arguments: list[str]) -> Any:
        if self.cluster["status"] == "DELETING":
            self.deleting_eks_reads += 1
            self.eks_gone = self.deleting_eks_reads >= 3
        return (
            absent("ResourceNotFoundException")
            if self.eks_gone
            else {"cluster": self.cluster}
        )

    def describe_hyperpod(self, _arguments: list[str]) -> Any:
        if self.hyperpod["ClusterStatus"] == "Deleting":
            self.deleting_hp_reads += 1
            self.hp_gone = self.deleting_hp_reads >= 2
        return absent("ResourceNotFound") if self.hp_gone else self.hyperpod

    def delete_hyperpod(self, _arguments: list[str]) -> dict[str, Any]:
        self.hp_gone = True
        if self.recreate_eks:
            self.cluster["createdAt"] = "2026-02-01T00:00:00Z"
        return {}

    def describe_child(self, arguments: list[str]) -> Any:
        kind = arguments[2].removeprefix("describe-")
        name = arguments[arguments.index(f"--{kind}-name") + 1]
        children = self.children[kind]
        if children.get(name) == "DELETING":
            if kind == "nodegroup":
                assert {"workers-a", "workers-b"} <= set(self.deleted_children)
            children.pop(name)
        if name not in children:
            return absent("ResourceNotFoundException")
        field, name_field = {
            "nodegroup": ("nodegroup", "nodegroupName"),
            "fargate-profile": ("fargateProfile", "fargateProfileName"),
            "addon": ("addon", "addonName"),
        }[kind]
        return {
            field: {
                name_field: name,
                "clusterName": "control",
                "status": children[name],
            }
        }

    def delete_child(self, arguments: list[str]) -> dict[str, Any]:
        kind = arguments[2].removeprefix("delete-")
        name = arguments[arguments.index(f"--{kind}-name") + 1]
        if kind == "fargate-profile":
            assert not self.children["nodegroup"]
            assert "DELETING" not in self.children[kind].values()
        if kind == "addon":
            assert not self.children["fargate-profile"]
        self.children[kind][name] = "DELETING"
        self.deleted_children.append(name)
        return {}

    def delete_cluster(self, _arguments: list[str]) -> dict[str, Any]:
        assert self.hp_gone, "CPU EKS deletion started before HyperPod was absent"
        assert all(not children for children in self.children.values()), (
            "CPU EKS deletion started before all child resources were absent"
        )
        self.eks_gone = True
        return {}


def test_cpu_nodegroups_form_a_wave_but_fargate_profiles_remain_serial(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = CpuAws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.deleted_children == [
        "workers-a",
        "workers-b",
        "profile-a",
        "profile-b",
        "eks-pod-identity-agent",
    ]
    assert aws.eks_gone, "CPU EKS cluster was not deleted"
    assert [call[1:3] for call in aws.mutations][0] == ["sagemaker", "delete-cluster"]
    assert [call[1:3] for call in aws.mutations][-1] == ["eks", "delete-cluster"]


def test_cpu_deletion_retries_already_deleting_clusters_without_another_delete(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = CpuAws()
    aws.hyperpod["ClusterStatus"] = "Deleting"
    aws.cluster["status"] = "DELETING"
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.hp_gone and aws.eks_gone
    assert aws.mutations == []


def test_cpu_eks_read_failure_blocks_hyperpod_deletion(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = CpuAws()
    aws.responses[("eks", "describe-cluster")] = absent("AccessDenied")
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="AccessDenied"):
        ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.mutations == []
    assert not any(call[1] == "sagemaker" for call in aws.calls), (
        "HyperPod was accessed despite a failed CPU EKS read"
    )


def test_cpu_hyperpod_cannot_point_at_a_preserved_gpu_eks(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = CpuAws()
    aws.hyperpod["Orchestrator"]["Eks"]["ClusterArn"] = (
        f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-a"
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="live binding"):
        ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.mutations == []


def test_eks_recreation_during_hyperpod_wait_blocks_further_deletion(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = CpuAws()
    aws.recreate_eks = True
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="incarnation"):
        ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert [call[1:3] for call in aws.mutations] == [["sagemaker", "delete-cluster"]]
