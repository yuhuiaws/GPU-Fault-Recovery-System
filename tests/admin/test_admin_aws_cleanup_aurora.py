from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner, ordered_aurora_instances
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deadlines import current_deadline
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from tests.admin._aws_cleanup_support import (
    ACCOUNT,
    REGION,
    TAGS,
    Aws,
    absent,
    resource,
)
from tests.admin.test_admin_site import site_file

CLUSTER = "gpu-fault-aurora"
CLUSTER_ARN = f"arn:aws:rds:{REGION}:{ACCOUNT}:cluster:{CLUSTER}"
SNAPSHOT = "gpu-fault-final"


@pytest.fixture
def site(tmp_path: Path) -> RenderedSite:
    return load_site(site_file(tmp_path))


class AuroraAws(Aws):
    def __init__(self) -> None:
        self.database: dict[str, Any] = {
            "DBClusterIdentifier": CLUSTER,
            "DBClusterArn": CLUSTER_ARN,
            "DbClusterResourceId": "cluster-incarnation-a",
            "Engine": "aurora-postgresql",
            "Status": "available",
            "DeletionProtection": True,
            "TagList": deepcopy(TAGS),
            "DBClusterMembers": [
                {"DBInstanceIdentifier": name, "IsClusterWriter": name == "writer"}
                for name in ("reader-a", "writer", "reader-b")
            ],
        }
        self.instances = {
            name: {
                "DBInstanceIdentifier": name,
                "DBInstanceArn": f"arn:aws:rds:{REGION}:{ACCOUNT}:db:{name}",
                "DbiResourceId": f"db-incarnation-{name}",
                "DBClusterIdentifier": CLUSTER,
                "DBInstanceStatus": "available",
                "TagList": deepcopy(TAGS),
            }
            for name in ("reader-a", "writer", "reader-b")
        }
        self.snapshot: dict[str, Any] | None = None
        self.gone = False
        self.deleted_instances: list[str] = []
        self.absent_instances: list[str] = []
        self.fail_instance: str | None = None
        self.retag_after_readers = False
        self.recreate_after_readers = False
        super().__init__(
            {
                ("rds", "describe-db-clusters"): self.describe_cluster,
                ("rds", "list-tags-for-resource"): lambda _arguments: {
                    "TagList": self.database["TagList"]
                },
                ("rds", "describe-db-instances"): self.describe_instances,
                ("rds", "describe-db-cluster-snapshots"): self.describe_snapshot,
                ("rds", "delete-db-instance"): self.delete_instance,
                ("rds", "modify-db-cluster"): self.disable_protection,
                ("rds", "delete-db-cluster"): self.delete_cluster,
            }
        )

    def describe_cluster(self, _arguments: list[str]) -> Any:
        return (
            absent("DBClusterNotFoundFault")
            if self.gone
            else {"DBClusters": [self.database]}
        )

    def describe_instances(self, arguments: list[str]) -> Any:
        if "--filters" in arguments:
            return {"DBInstances": list(self.instances.values())}
        name = arguments[arguments.index("--db-instance-identifier") + 1]
        instance = self.instances.get(name)
        if instance is not None and instance["DBInstanceStatus"] == "deleting":
            if name.startswith("reader"):
                assert {"reader-a", "reader-b"} <= set(self.deleted_instances)
            self.instances.pop(name)
            self.absent_instances.append(name)
            self.database["DBClusterMembers"] = [
                item
                for item in self.database["DBClusterMembers"]
                if item["DBInstanceIdentifier"] != name
            ]
            if set(self.absent_instances) == {"reader-a", "reader-b"}:
                if self.retag_after_readers:
                    self.database["TagList"] = []
                if self.recreate_after_readers:
                    self.database["DbClusterResourceId"] = "replacement-cluster"
            instance = None
        return (
            absent("DBInstanceNotFound")
            if instance is None
            else {"DBInstances": [instance]}
        )

    def describe_snapshot(self, _arguments: list[str]) -> Any:
        return (
            absent("DBClusterSnapshotNotFoundFault")
            if self.snapshot is None
            else {"DBClusterSnapshots": [self.snapshot]}
        )

    def delete_instance(self, arguments: list[str]) -> Any:
        name = arguments[arguments.index("--db-instance-identifier") + 1]
        if name == self.fail_instance:
            return absent("AccessDenied")
        if name == "writer":
            assert set(self.absent_instances) == {"reader-a", "reader-b"}
        assert self.instances[name]["DBInstanceStatus"] != "deleting"
        self.deleted_instances.append(name)
        self.instances[name]["DBInstanceStatus"] = "deleting"
        return {}

    def disable_protection(self, _arguments: list[str]) -> dict[str, Any]:
        assert not self.instances, (
            "deletion protection was disabled before all Aurora instances were absent"
        )
        self.database["DeletionProtection"] = False
        return {}

    def final_snapshot(self) -> dict[str, Any]:
        return {
            "DBClusterSnapshotIdentifier": SNAPSHOT,
            "DBClusterIdentifier": CLUSTER,
            "DbClusterResourceId": self.database["DbClusterResourceId"],
            "SnapshotType": "manual",
            "Status": "available",
        }

    def delete_cluster(self, arguments: list[str]) -> dict[str, Any]:
        assert not self.instances, (
            "Aurora cluster deletion started before all instances were absent"
        )
        assert not self.database["DeletionProtection"]
        assert "--skip-final-snapshot" not in arguments
        assert (
            arguments[arguments.index("--final-db-snapshot-identifier") + 1] == SNAPSHOT
        )
        self.gone = True
        self.snapshot = self.final_snapshot()
        return {}


def test_retry_of_a_deleting_cluster_only_waits_for_absence_and_bound_snapshot(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = AuroraAws()
    aws.database["Status"] = "deleting"
    aws.database["DeletionProtection"] = False
    for instance in aws.instances.values():
        instance["DBInstanceStatus"] = "deleting"
    reads = []
    original_describe = aws.describe_cluster

    def describe(arguments):
        reads.append(arguments)
        if len(reads) >= 5:
            aws.gone = True
            aws.snapshot = aws.final_snapshot()
        return original_describe(arguments)

    aws.responses[("rds", "describe-db-clusters")] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    monkeypatch.setattr(aws_commands.time, "sleep", lambda _seconds: None)
    result = ResourceCleaner(site).delete_aurora(
        resource("aurora_cluster", CLUSTER),
        final_snapshot_policy="retain",
        final_snapshot_identifier=SNAPSHOT,
    )
    assert result == SNAPSHOT
    assert aws.mutations == []
    assert len(reads) >= 5


def test_aurora_reader_wave_finishes_before_writer_and_final_snapshot(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = AuroraAws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    cluster = resource("aurora_cluster", CLUSTER)
    result = cleaner.delete_aurora(
        cluster, final_snapshot_policy="retain", final_snapshot_identifier=SNAPSHOT
    )
    assert result == SNAPSHOT
    assert aws.deleted_instances == ["reader-a", "reader-b", "writer"]
    assert [call[2] for call in aws.mutations] == [
        "delete-db-instance",
        "delete-db-instance",
        "delete-db-instance",
        "modify-db-cluster",
        "delete-db-cluster",
    ]
    cleaner.delete_aurora(
        cluster, final_snapshot_policy="retain", final_snapshot_identifier=SNAPSHOT
    )
    assert len(aws.mutations) == 5


@pytest.mark.parametrize(
    ("policy", "identifier"),
    [
        ("unknown", SNAPSHOT),
        ("retain", ""),
        ("retain", "bad--snapshot"),
        ("retain", "bad-"),
    ],
)
def test_aurora_rejects_unknown_snapshot_inputs_before_any_api(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, policy: Any, identifier: str
) -> None:
    aws = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="snapshot"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy=policy,
            final_snapshot_identifier=identifier,
        )
    assert aws.calls == []


@pytest.mark.parametrize(
    "updates",
    [
        {"delete_policy": Policy.PRESERVE},
        {"resource_id": "other-aurora"},
        {"resource_type": "gpu_eks"},
        {"region": "us-west-2"},
    ],
)
def test_aurora_preserved_or_rebound_targets_do_not_reach_aws(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, updates: dict[str, Any]
) -> None:
    aws = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER).model_copy(update=updates),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.calls == []


@pytest.mark.parametrize(
    "snapshot_updates",
    [
        {},
        {"DBClusterIdentifier": "other-cluster"},
        {"DbClusterResourceId": "previous-incarnation"},
    ],
)
def test_snapshot_collisions_are_rejected_before_instance_deletion(
    site: RenderedSite,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_updates: dict[str, Any],
) -> None:
    aws = AuroraAws()
    aws.snapshot = {**aws.final_snapshot(), **snapshot_updates}
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="snapshot"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


@pytest.mark.parametrize("missing", ["writer", "instances", "cluster_identity", "tags"])
def test_unknown_aurora_membership_or_ownership_prevents_all_mutations(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    aws = AuroraAws()
    if missing == "writer":
        for member in aws.database["DBClusterMembers"]:
            member["IsClusterWriter"] = False
    elif missing == "instances":
        aws.instances.pop("reader-a")
    elif missing == "cluster_identity":
        aws.database.pop("DbClusterResourceId")
    else:
        aws.instances["reader-a"]["TagList"] = []
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


@pytest.mark.parametrize("drift", ["retag", "recreate"])
def test_aurora_identity_is_rechecked_between_reader_and_writer_waves(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    aws = AuroraAws()
    aws.retag_after_readers = drift == "retag"
    aws.recreate_after_readers = drift == "recreate"
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.deleted_instances == ["reader-a", "reader-b"]
    assert aws.database["DeletionProtection"] is True
    assert not aws.gone, "Aurora cluster was deleted after its identity changed"


def test_aurora_partial_delete_retries_only_the_remaining_instance(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = AuroraAws()
    aws.fail_instance = "writer"
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    cluster = resource("aurora_cluster", CLUSTER)
    with pytest.raises(BootstrapError, match="AccessDenied"):
        cleaner.delete_aurora(
            cluster, final_snapshot_policy="retain", final_snapshot_identifier=SNAPSHOT
        )
    assert aws.deleted_instances == ["reader-a", "reader-b"]
    aws.fail_instance = None
    assert (
        cleaner.delete_aurora(
            cluster, final_snapshot_policy="retain", final_snapshot_identifier=SNAPSHOT
        )
        == SNAPSHOT
    )
    assert aws.deleted_instances == ["reader-a", "reader-b", "writer"]


def test_an_absent_cluster_does_not_accept_another_clusters_snapshot(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = AuroraAws()
    aws.gone = True
    aws.snapshot = {**aws.final_snapshot(), "DBClusterIdentifier": "other"}
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="snapshot identity"):
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


def test_absent_cluster_snapshot_uses_recorded_incarnation_when_available(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = AuroraAws()
    aws.gone = True
    aws.snapshot = aws.final_snapshot()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="snapshot identity"):
        ResourceCleaner(site).delete_aurora(
            resource(
                "aurora_cluster",
                CLUSTER,
                attributes={"db_cluster_resource_id": "previous"},
            ),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


@pytest.mark.parametrize(
    "members",
    [
        [],
        [{"DBInstanceIdentifier": "writer", "IsClusterWriter": "false"}],
        [{"DBInstanceIdentifier": "writer", "IsClusterWriter": True}] * 2,
        [
            {"DBInstanceIdentifier": "writer", "IsClusterWriter": True},
            {"DBInstanceIdentifier": "unlisted-reader", "IsClusterWriter": False},
        ],
    ],
)
def test_reader_ordering_refuses_unknown_or_incomplete_membership(
    members: list[dict[str, Any]],
) -> None:
    with pytest.raises(BootstrapError):
        ordered_aurora_instances(
            {"DBClusterMembers": members}, [{"DBInstanceIdentifier": "writer"}]
        )


def test_aurora_wait_exceeds_900_seconds_with_bounded_per_poll_commands(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    budgets: list[float] = []
    aws = AuroraAws()
    aws.database["Status"] = "deleting"
    aws.database["DBClusterMembers"] = []
    aws.instances.clear()

    def describe(_arguments: list[str]) -> Any:
        deadline = current_deadline()
        if deadline is not None:
            budgets.append(deadline.remaining())
        if clock[0] >= 1300:
            aws.gone = True
            aws.snapshot = aws.final_snapshot()
        return (
            absent("DBClusterNotFoundFault")
            if aws.gone
            else {"DBClusters": [aws.database]}
        )

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    aws.responses[("rds", "describe-db-clusters")] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    monkeypatch.setattr(aws_commands.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(aws_commands.time, "sleep", sleep)
    assert (
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
        == SNAPSHOT
    )
    assert clock[0] == 1300
    assert len(budgets) == 241
    assert max(budgets) == 3600
    assert min(budgets) == 2400
    assert aws.mutations == []


def test_prepare_returns_a_read_only_binding_that_fences_later_deletion(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = AuroraAws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    cluster = resource("aurora_cluster", CLUSTER)
    binding = cleaner.prepare_aurora_delete(
        cluster, final_snapshot_policy="retain", final_snapshot_identifier=SNAPSHOT
    )
    assert binding == {
        "site_id": "test-site",
        "region": REGION,
        "account_id": ACCOUNT,
        "cluster_id": CLUSTER,
        "cluster_arn": CLUSTER_ARN,
        "db_cluster_resource_id": "cluster-incarnation-a",
        "final_snapshot_policy": "retain",
        "final_snapshot_identifier": SNAPSHOT,
    }
    assert aws.mutations == []
    aws.database["DbClusterResourceId"] = "replacement"
    with pytest.raises(BootstrapError, match="incarnation"):
        cleaner.delete_aurora(
            cluster.model_copy(update={"attributes": binding}),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.mutations == []


def test_a_saved_retention_binding_cannot_be_changed_to_skip(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="snapshot binding"):
        ResourceCleaner(site).delete_aurora(
            resource(
                "aurora_cluster",
                CLUSTER,
                attributes={
                    "final_snapshot_policy": "retain",
                    "final_snapshot_identifier": SNAPSHOT,
                },
            ),
            final_snapshot_policy="skip",
            final_snapshot_identifier=SNAPSHOT,
        )
    assert aws.calls == []


def test_retry_waits_for_a_pending_retained_snapshot_after_cluster_disappears(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    budgets: list[float] = []
    aws = AuroraAws()
    aws.gone = True
    aws.snapshot = {**aws.final_snapshot(), "Status": "creating"}

    def describe(_arguments: list[str]) -> Any:
        deadline = current_deadline()
        if deadline is not None:
            budgets.append(deadline.remaining())
        if clock[0] >= 1300:
            aws.snapshot = aws.final_snapshot()
        return {"DBClusterSnapshots": [aws.snapshot]}

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    aws.responses[("rds", "describe-db-cluster-snapshots")] = describe
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    monkeypatch.setattr(aws_commands.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(aws_commands.time, "sleep", sleep)
    assert (
        ResourceCleaner(site).delete_aurora(
            resource("aurora_cluster", CLUSTER),
            final_snapshot_policy="retain",
            final_snapshot_identifier=SNAPSHOT,
        )
        == SNAPSHOT
    )
    assert clock[0] == 1300
    assert len(budgets) == 81
    assert max(budgets) == 3600
    assert min(budgets) == 2400
    assert aws.mutations == []
