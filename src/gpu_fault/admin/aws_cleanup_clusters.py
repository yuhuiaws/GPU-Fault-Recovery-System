"""Bounded CPU-cluster and Aurora deletion lifecycles."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.aws_cleanup_helpers import (
    object_field,
    objects,
    ordered_aurora_instances,
    single_object,
    strict_tags,
    strings,
)
from gpu_fault.admin.aws_commands import (
    FinalSnapshotPolicy,
    checked_command,
    json_command,
    wait_until,
)
from gpu_fault.admin.bootstrap_common import BootstrapError, assert_site_tag
from gpu_fault.installation_resources import InstallationResource

if TYPE_CHECKING:
    from gpu_fault.admin.aws_cleanup import ClusterDeletion, ResourceProbe


def aurora_snapshot(
    cleaner: ResourceProbe,
    identifier: str,
    *,
    cluster_id: str,
    cluster_resource_id: str | None = None,
) -> dict[str, Any] | None:
    document = json_command(
        cleaner._aws(
            "rds",
            "describe-db-cluster-snapshots",
            "--db-cluster-snapshot-identifier",
            identifier,
        ),
        not_found=("DBClusterSnapshotNotFoundFault",),
    )
    if document is None:
        return None
    snapshot = single_object(document, "DBClusterSnapshots")
    if (
        snapshot.get("DBClusterSnapshotIdentifier") != identifier
        or snapshot.get("DBClusterIdentifier") != cluster_id
        or snapshot.get("SnapshotType") != "manual"
        or (
            cluster_resource_id is not None
            and snapshot.get("DbClusterResourceId") != cluster_resource_id
        )
    ):
        raise BootstrapError(
            "Aurora final snapshot identity does not match the cluster"
        )
    if snapshot.get("Status") not in {"creating", "copying", "available"}:
        raise BootstrapError("Aurora final snapshot status is unavailable or failed")
    return snapshot


def _wait_aurora_snapshot(
    cleaner: ClusterDeletion,
    identifier: str,
    cluster_id: str,
    identity: str | None,
) -> None:
    wait_until(
        lambda: (
            (
                snapshot := aurora_snapshot(
                    cleaner,
                    identifier,
                    cluster_id=cluster_id,
                    cluster_resource_id=identity,
                )
            )
            is not None
            and snapshot["Status"] == "available"
        ),
        description=f"Aurora final snapshot {identifier} availability",
        timeout_seconds=3600,
        interval_seconds=15,
    )


def _bound_aurora(
    cleaner: ClusterDeletion,
    cluster: InstallationResource,
    identity: str,
) -> dict[str, Any] | None:
    database = cleaner._aurora_cluster(cluster.resource_id)
    if database is None:
        return None
    if (
        database.get("DBClusterIdentifier") != cluster.resource_id
        or database.get("DbClusterResourceId") != identity
        or database.get("Engine") != "aurora-postgresql"
        or not isinstance(database.get("DeletionProtection"), bool)
        or not isinstance(database.get("Status"), str)
        or not database["Status"]
        or database.get("DBClusterArn")
        != cleaner._ownership._arn(cluster, "rds", f"cluster:{cluster.resource_id}")
    ):
        raise BootstrapError("Aurora cluster identity or state has drifted")
    cleaner._ownership.validate_arn(str(database.get("DBClusterArn") or ""))
    assert_site_tag(
        strict_tags(database.get("TagList")),
        site_id=cleaner._ownership.site_id,
        description=cluster.resource_key,
    )
    return database


def _delete_aurora_instances(
    cleaner: ClusterDeletion,
    cluster: InstallationResource,
    database: dict[str, Any],
    identity: str,
) -> None:
    document = json_command(
        cleaner._aws(
            "rds",
            "describe-db-instances",
            "--filters",
            f"Name=db-cluster-id,Values={cluster.resource_id}",
        )
    )
    instances = ordered_aurora_instances(
        database, objects(document or {}, "DBInstances")
    )
    writers = {
        member["DBInstanceIdentifier"]
        for member in objects(database, "DBClusterMembers")
        if member["IsClusterWriter"]
    }
    for instance in instances:
        if (
            instance.get("DBClusterIdentifier") != cluster.resource_id
            or not instance.get("DbiResourceId")
            or not instance.get("DBInstanceStatus")
        ):
            raise BootstrapError("Aurora instance identity or state is unavailable")
        cleaner._ownership.validate_arn(str(instance.get("DBInstanceArn") or ""))
        assert_site_tag(
            strict_tags(instance.get("TagList")),
            site_id=cleaner._ownership.site_id,
            description="Aurora instance",
        )
    # Submit independent readers together, but confirm the whole reader wave
    # absent before touching the writer.
    for writer_wave in (False, True):
        wave = [
            instance
            for instance in instances
            if (instance["DBInstanceIdentifier"] in writers) == writer_wave
        ]
        for instance in wave:
            instance_id = instance["DBInstanceIdentifier"]
            if instance["DBInstanceStatus"] == "deleting":
                continue
            current = _bound_aurora(cleaner, cluster, identity)
            if (
                current is None
                or {
                    member["DBInstanceIdentifier"]
                    for member in objects(current, "DBClusterMembers")
                    if member.get("IsClusterWriter") is True
                }
                != writers
            ):
                raise BootstrapError("Aurora writer changed during cleanup")
            live_document = json_command(
                cleaner._aws(
                    "rds",
                    "describe-db-instances",
                    "--db-instance-identifier",
                    instance_id,
                ),
                not_found=("DBInstanceNotFound",),
            )
            if live_document is None:
                continue
            live = single_object(live_document, "DBInstances")
            if (
                live.get("DBInstanceIdentifier") != instance_id
                or live.get("DbiResourceId") != instance["DbiResourceId"]
                or live.get("DBClusterIdentifier") != cluster.resource_id
            ):
                raise BootstrapError("Aurora instance identity changed during cleanup")
            assert_site_tag(
                strict_tags(live.get("TagList")),
                site_id=cleaner._ownership.site_id,
                description="Aurora instance",
            )
            if live.get("DBInstanceStatus") != "deleting":
                if not live.get("DBInstanceStatus"):
                    raise BootstrapError("Aurora instance status is unavailable")
                checked_command(
                    cleaner._aws(
                        "rds",
                        "delete-db-instance",
                        "--db-instance-identifier",
                        instance_id,
                        "--skip-final-snapshot",
                        "--delete-automated-backups",
                    ),
                    not_found=("DBInstanceNotFound",),
                )
        for instance in wave:
            instance_id = instance["DBInstanceIdentifier"]
            wait_until(
                lambda: not cleaner._exists_command(
                    cleaner._aws(
                        "rds",
                        "describe-db-instances",
                        "--db-instance-identifier",
                        instance_id,
                    ),
                    not_found=("DBInstanceNotFound",),
                ),
                description=f"Aurora instance {instance_id} deletion",
                timeout_seconds=3600,
                interval_seconds=15,
            )


def _aurora_preflight(
    cleaner: ClusterDeletion,
    cluster: InstallationResource,
    *,
    final_snapshot_policy: FinalSnapshotPolicy,
    final_snapshot_identifier: str,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if final_snapshot_policy not in {"retain", "skip"}:
        raise BootstrapError("unknown Aurora final snapshot policy")
    retained = final_snapshot_identifier if final_snapshot_policy == "retain" else None
    if retained is not None and (
        re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,254}", retained) is None
        or retained.endswith("-")
        or "--" in retained
    ):
        raise BootstrapError("invalid Aurora final snapshot identifier")
    if (
        cluster.resource_type != "aurora_cluster"
        or cluster.resource_id
        != cleaner.site.release_config["health"].get("aurora_cluster_id")
    ):
        raise BootstrapError("Aurora deletion target does not match the site")
    for key, value in (
        ("final_snapshot_policy", final_snapshot_policy),
        ("final_snapshot_identifier", final_snapshot_identifier),
    ):
        if key in cluster.attributes and cluster.attributes[key] != value:
            raise BootstrapError("saved Aurora final snapshot binding has drifted")
    present = cleaner._ownership.before_delete(cluster, cleaner.exists)
    database = (
        cleaner._aurora_cluster(cluster.resource_id) if present is not None else None
    )
    identity = cluster.attributes.get("db_cluster_resource_id")
    if database is not None:
        observed = str(database.get("DbClusterResourceId") or "")
        if not observed or (identity is not None and identity != observed):
            raise BootstrapError("Aurora cluster incarnation is unavailable")
        identity = observed
        database = _bound_aurora(cleaner, cluster, identity)
    if retained:
        snapshot = aurora_snapshot(
            cleaner,
            retained,
            cluster_id=cluster.resource_id,
            cluster_resource_id=identity,
        )
        if database is None:
            if snapshot is None:
                raise BootstrapError(
                    "Aurora cluster is absent but its final snapshot is unavailable"
                )
            if snapshot["Status"] != "available":
                _wait_aurora_snapshot(cleaner, retained, cluster.resource_id, identity)
            return None, identity, retained
        if snapshot is not None and database.get("Status") != "deleting":
            raise BootstrapError("Aurora final snapshot identifier already exists")
    return database, identity, retained


def prepare_aurora_deletion(
    cleaner: ClusterDeletion,
    cluster: InstallationResource,
    *,
    final_snapshot_policy: FinalSnapshotPolicy,
    final_snapshot_identifier: str,
) -> dict[str, str]:
    """Return a read-only binding for the caller to persist before deletion."""
    _database, identity, _retained = _aurora_preflight(
        cleaner,
        cluster,
        final_snapshot_policy=final_snapshot_policy,
        final_snapshot_identifier=final_snapshot_identifier,
    )
    if identity is None:
        raise BootstrapError(
            "cannot prepare Aurora deletion without a proven incarnation"
        )
    return {
        "site_id": cluster.site_id,
        "region": cleaner.region,
        "account_id": cleaner._ownership.cpu.account,
        "cluster_id": cluster.resource_id,
        "cluster_arn": cleaner._ownership._arn(
            cluster, "rds", f"cluster:{cluster.resource_id}"
        ),
        "db_cluster_resource_id": identity,
        "final_snapshot_policy": final_snapshot_policy,
        "final_snapshot_identifier": final_snapshot_identifier,
    }


def delete_aurora_cluster(
    cleaner: ClusterDeletion,
    cluster: InstallationResource,
    *,
    final_snapshot_policy: FinalSnapshotPolicy,
    final_snapshot_identifier: str,
) -> str | None:
    database, identity, retained = _aurora_preflight(
        cleaner,
        cluster,
        final_snapshot_policy=final_snapshot_policy,
        final_snapshot_identifier=final_snapshot_identifier,
    )
    if database is None:
        return retained
    if identity is None:
        raise BootstrapError("Aurora deletion lacks a proven incarnation")
    if database.get("Status") != "deleting":
        _delete_aurora_instances(cleaner, cluster, database, identity)
    database = _bound_aurora(cleaner, cluster, identity)
    if database is not None and database.get("Status") != "deleting":
        if objects(database, "DBClusterMembers"):
            raise BootstrapError(
                "Aurora cluster still has members before final deletion"
            )
        if database["DeletionProtection"]:
            checked_command(
                cleaner._aws(
                    "rds",
                    "modify-db-cluster",
                    "--db-cluster-identifier",
                    cluster.resource_id,
                    "--no-deletion-protection",
                    "--apply-immediately",
                ),
                not_found=("DBClusterNotFoundFault",),
            )
            wait_until(
                lambda: (
                    (current := _bound_aurora(cleaner, cluster, identity)) is None
                    or not current["DeletionProtection"]
                ),
                description=f"Aurora cluster {cluster.resource_id} deletion protection disablement",
                timeout_seconds=900,
                interval_seconds=10,
            )
        database = _bound_aurora(cleaner, cluster, identity)
        if database is not None and database.get("Status") != "deleting":
            if objects(database, "DBClusterMembers"):
                raise BootstrapError("Aurora cluster gained members during cleanup")
            arguments = cleaner._aws(
                "rds",
                "delete-db-cluster",
                "--db-cluster-identifier",
                cluster.resource_id,
                "--delete-automated-backups",
            )
            arguments.extend(
                ["--final-db-snapshot-identifier", retained]
                if retained
                else ["--skip-final-snapshot"]
            )
            checked_command(arguments, not_found=("DBClusterNotFoundFault",))
    cleaner.wait_absent(cluster, timeout_seconds=3600)
    if retained:
        _wait_aurora_snapshot(cleaner, retained, cluster.resource_id, identity)
    return retained


def cpu_eks_identity(
    cleaner: ClusterDeletion, name: str, *, expected_created_at: object = None
) -> dict[str, Any] | None:
    if name != cleaner._ownership.cpu.resource_name:
        raise BootstrapError("EKS deletion target does not match the CPU cluster")
    document = json_command(
        cleaner._aws("eks", "describe-cluster", "--name", name),
        not_found=("ResourceNotFoundException",),
    )
    if document is None:
        return None
    cluster = object_field(document, "cluster")
    if (
        cluster.get("arn") != cleaner.site.release_config["cpu_eks_arn"]
        or cluster.get("name") != name
        or not isinstance(cluster.get("createdAt"), str)
        or not cluster["createdAt"].strip()
        or not cluster.get("status")
        or (
            expected_created_at is not None
            and cluster["createdAt"] != expected_created_at
        )
    ):
        raise BootstrapError("CPU EKS identity or incarnation has drifted")
    return cluster


def cpu_hyperpod_identity(
    cleaner: ClusterDeletion, name: str, *, expected_arn: str | None = None
) -> dict[str, Any] | None:
    if name != cleaner.site.release_config["cpu_hyperpod_cluster_name"]:
        raise BootstrapError("HyperPod deletion target does not match the CPU cluster")
    document = json_command(
        cleaner._aws("sagemaker", "describe-cluster", "--cluster-name", name),
        not_found=("ResourceNotFound",),
    )
    if document is None:
        return None
    eks = object_field(object_field(document, "Orchestrator"), "Eks")
    if (
        document.get("ClusterName") != name
        or eks.get("ClusterArn") != cleaner.site.release_config["cpu_eks_arn"]
        or not document.get("ClusterStatus")
        or (expected_arn is not None and document.get("ClusterArn") != expected_arn)
    ):
        raise BootstrapError("CPU HyperPod/EKS live binding has drifted")
    arn = str(document.get("ClusterArn") or "")
    cleaner._ownership.validate_arn(arn)
    if not arn.startswith(
        f"arn:{cleaner._ownership.cpu.partition}:sagemaker:{cleaner.region}:"
        f"{cleaner._ownership.cpu.account}:cluster/"
    ):
        raise BootstrapError("CPU HyperPod incarnation is unavailable")
    return document


def prepare_cpu_deletion(
    cleaner: ClusterDeletion,
    hyperpod: InstallationResource,
    eks: InstallationResource,
) -> dict[str, Any]:
    cleaner._ownership.validate_cpu_pair(hyperpod, eks)
    eks_live = cpu_eks_identity(cleaner, eks.resource_id)
    hyperpod_live = cpu_hyperpod_identity(cleaner, hyperpod.resource_id)
    if eks_live is None or hyperpod_live is None:
        raise BootstrapError("CPU cleanup cannot bind an absent cluster incarnation")
    if hyperpod.resource_arn not in (None, hyperpod_live["ClusterArn"]):
        raise BootstrapError("CPU HyperPod registry binding has drifted")
    return {
        "cpu_eks_created_at": eks_live["createdAt"],
        "cpu_hyperpod_arn": hyperpod_live["ClusterArn"],
    }


def delete_hyperpod_cluster(
    cleaner: ClusterDeletion, name: str, *, expected_arn: str | None = None
) -> None:
    document = cpu_hyperpod_identity(cleaner, name, expected_arn=expected_arn)
    if document is None:
        return
    arn = document["ClusterArn"]
    if document["ClusterStatus"] != "Deleting":
        checked_command(
            cleaner._aws("sagemaker", "delete-cluster", "--cluster-name", arn),
            not_found=("ResourceNotFound",),
        )
    wait_until(
        lambda: cpu_hyperpod_identity(cleaner, name, expected_arn=arn) is None,
        description=f"HyperPod cluster {name} deletion",
        timeout_seconds=3600,
        interval_seconds=15,
    )


def _request_eks_child_deletion(
    cleaner: ClusterDeletion, cluster_name: str, kind: str, name: str
) -> None:
    field, identity = {
        "nodegroup": ("nodegroup", "nodegroupName"),
        "fargate-profile": ("fargateProfile", "fargateProfileName"),
        "addon": ("addon", "addonName"),
    }[kind]
    document = json_command(
        cleaner._aws(
            "eks",
            f"describe-{kind}",
            "--cluster-name",
            cluster_name,
            f"--{kind}-name",
            name,
        ),
        not_found=("ResourceNotFoundException",),
    )
    if document is None:
        return
    child = object_field(document, field)
    if (
        child.get(identity) != name
        or child.get("clusterName") != cluster_name
        or not child.get("status")
    ):
        raise BootstrapError(f"CPU EKS {kind} identity or state has drifted")
    if child["status"] != "DELETING":
        checked_command(
            cleaner._aws(
                "eks",
                f"delete-{kind}",
                "--cluster-name",
                cluster_name,
                f"--{kind}-name",
                name,
            ),
            not_found=("ResourceNotFoundException",),
        )


def _wait_eks_child(
    cleaner: ClusterDeletion, cluster_name: str, kind: str, name: str
) -> None:
    wait_until(
        lambda: not cleaner._exists_command(
            cleaner._aws(
                "eks",
                f"describe-{kind}",
                "--cluster-name",
                cluster_name,
                f"--{kind}-name",
                name,
            ),
            not_found=("ResourceNotFoundException",),
        ),
        description=f"EKS {kind} {name} deletion",
        timeout_seconds=3600,
        interval_seconds=15,
    )


def delete_eks_cluster(
    cleaner: ClusterDeletion, name: str, *, expected_created_at: object = None
) -> None:
    cluster = cpu_eks_identity(cleaner, name, expected_created_at=expected_created_at)
    if cluster is None:
        return
    if cluster["status"] != "DELETING":
        associations = json_command(
            cleaner._aws(
                "eks", "list-pod-identity-associations", "--cluster-name", name
            ),
            not_found=("ResourceNotFoundException",),
        )
        if associations is None:
            return
        for association in objects(associations, "associations"):
            identifier = association.get("associationId")
            if not isinstance(identifier, str) or not identifier:
                raise BootstrapError("CPU Pod Identity association ID is unavailable")
            checked_command(
                cleaner._aws(
                    "eks",
                    "delete-pod-identity-association",
                    "--cluster-name",
                    name,
                    "--association-id",
                    identifier,
                ),
                not_found=("ResourceNotFoundException",),
            )
        nodegroups = json_command(
            cleaner._aws("eks", "list-nodegroups", "--cluster-name", name),
            not_found=("ResourceNotFoundException",),
        )
        if nodegroups is None:
            return
        names = strings(nodegroups, "nodegroups")
        for nodegroup in names:
            _request_eks_child_deletion(cleaner, name, "nodegroup", nodegroup)
        for nodegroup in names:
            _wait_eks_child(cleaner, name, "nodegroup", nodegroup)
        profiles = json_command(
            cleaner._aws("eks", "list-fargate-profiles", "--cluster-name", name),
            not_found=("ResourceNotFoundException",),
        )
        if profiles is None:
            return
        # EKS permits only one Fargate profile deletion in progress per cluster.
        for profile in strings(profiles, "fargateProfileNames"):
            _request_eks_child_deletion(cleaner, name, "fargate-profile", profile)
            _wait_eks_child(cleaner, name, "fargate-profile", profile)
        addons = json_command(
            cleaner._aws("eks", "list-addons", "--cluster-name", name),
            not_found=("ResourceNotFoundException",),
        )
        if addons is None:
            return
        for addon in strings(addons, "addons"):
            _request_eks_child_deletion(cleaner, name, "addon", addon)
            _wait_eks_child(cleaner, name, "addon", addon)
        current = cpu_eks_identity(
            cleaner, name, expected_created_at=cluster["createdAt"]
        )
        if current is not None and current["status"] != "DELETING":
            checked_command(
                cleaner._aws("eks", "delete-cluster", "--name", name),
                not_found=("ResourceNotFoundException",),
            )
    wait_until(
        lambda: cpu_eks_identity(
            cleaner, name, expected_created_at=cluster["createdAt"]
        )
        is None,
        description=f"EKS cluster {name} deletion",
        timeout_seconds=3600,
        interval_seconds=15,
    )
