from __future__ import annotations

from copy import deepcopy

import pytest

from gpu_fault.admin import aws_cleanup_clusters as clusters
from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin._aws_cleanup_support import absent, resource
from tests.admin.test_admin_aws_cleanup_aurora import CLUSTER, SNAPSHOT, AuroraAws
from tests.admin.test_admin_aws_cleanup_clusters import CpuAws, cpu_records
from tests.admin.test_admin_site import site_file


@pytest.fixture
def site(tmp_path):
    return load_site(site_file(tmp_path))


def delete_aurora(site, *, policy="retain"):
    return ResourceCleaner(site).delete_aurora(
        resource("aurora_cluster", CLUSTER),
        final_snapshot_policy=policy,
        final_snapshot_identifier=SNAPSHOT,
    )


def test_absent_hyperpod_does_not_prevent_exact_cpu_eks_cleanup(site, monkeypatch):
    aws = CpuAws()
    aws.hp_gone = True
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.eks_gone is True
    assert not any(
        call[:3] == ["aws", "sagemaker", "delete-cluster"] for call in aws.mutations
    ), "already absent CPU HyperPod received another delete"


def test_absent_cpu_eks_is_not_deleted_again(site, monkeypatch):
    aws = CpuAws()
    aws.eks_gone = True
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    clusters.delete_eks_cluster(ResourceCleaner(site), "control")
    assert aws.mutations == []


def test_cpu_eks_disappearance_after_children_skips_cluster_delete(site, monkeypatch):
    aws = CpuAws()
    original = aws.describe_cluster

    def describe(arguments):
        if all(not children for children in aws.children.values()):
            aws.eks_gone = True
        return original(arguments)

    aws.responses["eks", "describe-cluster"] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.eks_gone is True
    assert not any(
        call[:3] == ["aws", "eks", "delete-cluster"] for call in aws.mutations
    ), "CPU EKS deletion was repeated after observed disappearance"


def test_absent_aurora_with_missing_retained_snapshot_is_not_success(site, monkeypatch):
    aws = AuroraAws()
    aws.gone = True
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="final snapshot is unavailable"):
        delete_aurora(site)
    assert aws.mutations == []


def test_aurora_without_deletion_protection_does_not_modify_it(site, monkeypatch):
    aws = AuroraAws()
    aws.database["DeletionProtection"] = False
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    assert delete_aurora(site) == SNAPSHOT
    assert not any(call[2] == "modify-db-cluster" for call in aws.mutations), (
        "already disabled deletion protection was modified"
    )


def test_explicit_skip_snapshot_reaches_only_the_fake_cluster_deletion(
    site, monkeypatch
):
    aws = AuroraAws()

    def delete(arguments):
        assert "--skip-final-snapshot" in arguments
        assert not aws.instances, "Aurora deletion preceded instance quiescence"
        assert aws.database["DeletionProtection"] is False
        aws.gone = True
        return {}

    aws.responses["rds", "delete-db-cluster"] = delete
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    assert delete_aurora(site, policy="skip") is None
    assert aws.snapshot is None
    assert not any(call[2] == "describe-db-cluster-snapshots" for call in aws.calls), (
        "skip-snapshot deletion made an unrelated snapshot query"
    )


@pytest.mark.parametrize("change", ["promoted-writer", "disappeared"])
def test_aurora_writer_change_after_inventory_blocks_instance_deletion(
    site, monkeypatch, change
):
    aws = AuroraAws()
    original = aws.describe_instances

    def inventory(arguments):
        response = original(arguments)
        if "--filters" in arguments:
            if change == "disappeared":
                aws.gone = True
            else:
                for member in aws.database["DBClusterMembers"]:
                    member["IsClusterWriter"] = (
                        member["DBInstanceIdentifier"] == "reader-a"
                    )
        return response

    aws.responses["rds", "describe-db-instances"] = inventory
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="writer changed during cleanup"):
        delete_aurora(site)
    assert aws.mutations == []


@pytest.mark.parametrize(
    "change", ["disappeared", "gained-member", "gained-after-disable"]
)
def test_aurora_final_delete_rechecks_membership_after_each_barrier(
    site, monkeypatch, change
):
    aws = AuroraAws()
    original = aws.describe_cluster
    disable = aws.disable_protection

    def describe(arguments):
        if not aws.instances:
            if change == "disappeared":
                aws.gone = True
            elif change == "gained-member":
                aws.database["DBClusterMembers"] = [
                    {"DBInstanceIdentifier": "new-member", "IsClusterWriter": True}
                ]
        return original(arguments)

    def changed_protection(arguments):
        result = disable(arguments)
        if change == "gained-after-disable":
            aws.database["DBClusterMembers"] = [
                {"DBInstanceIdentifier": "new-member", "IsClusterWriter": True}
            ]
        return result

    aws.responses["rds", "describe-db-clusters"] = describe
    aws.responses["rds", "modify-db-cluster"] = changed_protection
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    if change == "disappeared":
        assert delete_aurora(site, policy="skip") is None
    else:
        with pytest.raises(BootstrapError, match="still has members|gained members"):
            delete_aurora(site)
    assert not any(call[2] == "delete-db-cluster" for call in aws.mutations), (
        "changed final Aurora state authorized cluster deletion"
    )


@pytest.mark.parametrize(
    "change", ["already-deleting", "disappeared", "became-deleting"]
)
def test_aurora_reader_recheck_does_not_repeat_an_existing_delete(
    site, monkeypatch, change
):
    aws = AuroraAws()
    original = aws.describe_instances
    changed = False
    if change == "already-deleting":
        aws.instances["reader-a"]["DBInstanceStatus"] = "deleting"
        aws.deleted_instances.append("reader-a")

    def describe(arguments):
        nonlocal changed
        if (
            change != "already-deleting"
            and "--filters" not in arguments
            and arguments[arguments.index("--db-instance-identifier") + 1] == "reader-a"
            and not changed
        ):
            changed = True
            aws.deleted_instances.append("reader-a")
            if change == "disappeared":
                aws.instances.pop("reader-a")
                aws.absent_instances.append("reader-a")
                aws.database["DBClusterMembers"] = [
                    member
                    for member in aws.database["DBClusterMembers"]
                    if member["DBInstanceIdentifier"] != "reader-a"
                ]
                return absent("DBInstanceNotFound")
            aws.instances["reader-a"]["DBInstanceStatus"] = "deleting"
            return {"DBInstances": [deepcopy(aws.instances["reader-a"])]}
        return original(arguments)

    aws.responses["rds", "describe-db-instances"] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    assert delete_aurora(site) == SNAPSHOT
    assert not any(
        call[2] == "delete-db-instance" and "reader-a" in call for call in aws.mutations
    ), "reader already being removed received another delete request"
