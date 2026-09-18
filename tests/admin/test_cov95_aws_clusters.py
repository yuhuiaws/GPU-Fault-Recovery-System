from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import pytest

from gpu_fault.admin import aws_cleanup_clusters as clusters
from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin._aws_cleanup_support import ACCOUNT, REGION, absent, resource
from tests.admin.test_admin_aws_cleanup_aurora import CLUSTER, SNAPSHOT, AuroraAws
from tests.admin.test_admin_aws_cleanup_clusters import CpuAws, cpu_records
from tests.admin.test_admin_site import site_file


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(
        aws_commands,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=lambda _seconds: None),
    )
    return load_site(site_file(tmp_path))


@pytest.mark.parametrize(
    "plane,updates",
    [
        ("eks", {"name": "foreign"}),
        ("eks", {"createdAt": None}),
        ("eks", {"status": ""}),
        ("hyperpod", {"ClusterStatus": ""}),
        ("hyperpod", {"ClusterArn": f"arn:aws:iam::{ACCOUNT}:role/example"}),
    ],
)
def test_cpu_incarnation_validation_precedes_deletion(
    site, monkeypatch, plane, updates
):
    aws = CpuAws()
    (aws.cluster if plane == "eks" else aws.hyperpod).update(updates)
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="identity|incarnation|binding"):
        ResourceCleaner(site).prepare_cpu_delete(*cpu_records())
    assert aws.mutations == []


@pytest.mark.parametrize("plane", ["eks", "hyperpod"])
def test_cpu_preparation_cannot_bind_absent_cluster(site, monkeypatch, plane):
    aws = CpuAws()
    aws.eks_gone = plane == "eks"
    aws.hp_gone = plane == "hyperpod"
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="absent cluster incarnation"):
        ResourceCleaner(site).prepare_cpu_delete(*cpu_records())
    assert aws.mutations == []


def test_cpu_preparation_rejects_registry_hyperpod_incarnation_drift(site, monkeypatch):
    aws = CpuAws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    hyperpod, eks = cpu_records()
    hyperpod = hyperpod.model_copy(
        update={"resource_arn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/other"}
    )
    with pytest.raises(BootstrapError, match="registry binding has drifted"):
        ResourceCleaner(site).prepare_cpu_delete(hyperpod, eks)
    assert aws.mutations == []


@pytest.mark.parametrize(
    "operation", [clusters.cpu_eks_identity, clusters.cpu_hyperpod_identity]
)
def test_cpu_identity_helpers_reject_other_names_before_transport(
    site, monkeypatch, operation
):
    aws = CpuAws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="does not match"):
        operation(ResourceCleaner(site), "foreign")
    assert aws.calls == []


@pytest.mark.parametrize(
    "listing",
    [
        "list-pod-identity-associations",
        "list-nodegroups",
        "list-fargate-profiles",
        "list-addons",
    ],
)
def test_cpu_eks_disappearance_during_child_listing_does_not_trigger_more_deletes(
    site, monkeypatch, listing
):
    aws = CpuAws()

    def disappeared(_arguments):
        aws.eks_gone = True
        return absent("ResourceNotFoundException")

    aws.responses[("eks", listing)] = disappeared
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.eks_gone, "fake EKS disappearance was not observed"
    assert not any(call[1:3] == ["eks", "delete-cluster"] for call in aws.mutations), (
        "EKS deletion was requested after confirmed disappearance"
    )
    assert aws.calls[-1][2] == listing


@pytest.mark.parametrize("identifier", [None, "", 7])
def test_cpu_association_without_identity_blocks_child_cleanup(
    site, monkeypatch, identifier
):
    aws = CpuAws()
    aws.responses[("eks", "list-pod-identity-associations")] = {
        "associations": [{"associationId": identifier}]
    }
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="association ID"):
        ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.deleted_children == []
    assert [call[1:3] for call in aws.mutations] == [["sagemaker", "delete-cluster"]]


def test_cpu_associations_are_removed_before_nodegroups(site, monkeypatch):
    aws = CpuAws()
    aws.responses[("eks", "list-pod-identity-associations")] = {
        "associations": [{"associationId": "a-example"}]
    }
    aws.responses[("eks", "delete-pod-identity-association")] = {}
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    operations = [call[2] for call in aws.mutations]
    assert operations.index("delete-pod-identity-association") < operations.index(
        "delete-nodegroup"
    )


@pytest.mark.parametrize(
    "field,value",
    [("nodegroupName", "foreign"), ("clusterName", "foreign"), ("status", "")],
)
def test_cpu_child_identity_drift_prevents_deletion_of_that_child(
    site, monkeypatch, field, value
):
    aws = CpuAws()
    aws.responses[("eks", "describe-nodegroup")] = {
        "nodegroup": {
            "nodegroupName": "workers-a",
            "clusterName": "control",
            "status": "ACTIVE",
            field: value,
        }
    }
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="nodegroup identity"):
        ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert aws.deleted_children == []


@pytest.mark.parametrize("state", ["DELETING", "ABSENT"])
def test_cpu_child_already_removing_is_waited_not_deleted_again(
    site, monkeypatch, state
):
    aws = CpuAws()
    original = aws.describe_child
    seen = False

    def describe(arguments):
        nonlocal seen
        if arguments[arguments.index("--fargate-profile-name") + 1] != "profile-a":
            return original(arguments)
        if state == "DELETING" and not seen:
            seen = True
            return {
                "fargateProfile": {
                    "fargateProfileName": "profile-a",
                    "clusterName": "control",
                    "status": "DELETING",
                }
            }
        aws.children["fargate-profile"].pop("profile-a", None)
        return absent("ResourceNotFoundException")

    aws.responses[("eks", "describe-fargate-profile")] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete_cpu_cluster(*cpu_records())
    assert "profile-a" not in aws.deleted_children
    assert aws.eks_gone, "CPU cleanup did not finish after the already-removing child"


@pytest.mark.parametrize(
    "field,value",
    [
        ("Engine", "mysql"),
        ("DeletionProtection", "false"),
        ("Status", ""),
        ("DBClusterArn", f"arn:aws:rds:{REGION}:{ACCOUNT}:cluster:foreign"),
    ],
)
def test_aurora_cluster_state_drift_prevents_every_deletion(
    site, monkeypatch, field, value
):
    aws = AuroraAws()
    aws.database[field] = value
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="identity or state has drifted"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("DBClusterIdentifier", "foreign"),
        ("DbiResourceId", ""),
        ("DBInstanceStatus", ""),
    ],
)
def test_aurora_inventory_requires_each_instance_identity(
    site, monkeypatch, field, value
):
    aws = AuroraAws()
    aws.instances["reader-a"][field] = value
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="identity or state"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("DBInstanceIdentifier", "foreign"),
        ("DbiResourceId", "new-incarnation"),
        ("DBClusterIdentifier", "foreign"),
        ("DBInstanceStatus", ""),
    ],
)
def test_aurora_instance_recheck_rejects_drift_after_inventory(
    site, monkeypatch, field, value
):
    aws = AuroraAws()
    original = aws.describe_instances

    def describe(arguments):
        document = original(arguments)
        if "--filters" not in arguments and isinstance(document, dict):
            document = copy.deepcopy(document)
            document["DBInstances"][0][field] = value
        return document

    aws.responses[("rds", "describe-db-instances")] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="identity changed|status is unavailable"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("Status", "failed"),
        ("SnapshotType", "automated"),
        ("DBClusterSnapshotIdentifier", "other"),
    ],
)
def test_aurora_snapshot_status_and_identity_are_not_inferred(
    site, monkeypatch, field, value
):
    aws = AuroraAws()
    aws.snapshot = {**aws.final_snapshot(), field: value}
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="snapshot"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


@pytest.mark.parametrize("status", ["creating", "copying"])
def test_absent_aurora_waits_for_bound_final_snapshot_availability(
    site, monkeypatch, status
):
    aws = AuroraAws()
    aws.gone = True
    states = iter([status, "available"])

    def describe(_arguments):
        return {
            "DBClusterSnapshots": [
                {**aws.final_snapshot(), "Status": next(states, "available")}
            ]
        }

    aws.responses[("rds", "describe-db-cluster-snapshots")] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    entry = resource(
        "aurora_cluster",
        CLUSTER,
        attributes={"db_cluster_resource_id": "cluster-incarnation-a"},
    )
    assert (
        ResourceCleaner(site).delete_aurora(
            entry, final_snapshot_policy="retain", final_snapshot_identifier=SNAPSHOT
        )
        == SNAPSHOT
    )
    assert aws.mutations == []
    assert sum(call[2] == "describe-db-cluster-snapshots" for call in aws.calls) >= 2


def test_aurora_preparation_requires_incarnation_even_when_deletion_is_already_absent(
    site, monkeypatch
):
    aws = AuroraAws()
    aws.gone = True
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="proven incarnation"):
        ResourceCleaner(site).prepare_aurora_delete(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="skip",
            final_snapshot_identifier="unused",
        )
    assert aws.mutations == []


@pytest.mark.parametrize(
    "field,value",
    [("final_snapshot_policy", "skip"), ("final_snapshot_identifier", "other")],
)
def test_saved_aurora_snapshot_policy_cannot_change_on_retry(
    site, monkeypatch, field, value
):
    aws = AuroraAws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="binding has drifted"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER, attributes={field: value}),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.calls == []
