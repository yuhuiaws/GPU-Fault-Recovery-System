from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.test_admin_bootstrap_aurora import InstanceCreationRunner


class DelayedPrimaryRunner(InstanceCreationRunner):
    def __init__(self, *, resumed: bool, membership_missing: bool = False) -> None:
        super().__init__(
            {"aurora-a-writer": "creating"} if resumed else {},
            cluster_status="available",
        )
        self.membership_missing = membership_missing

    def aws_json(self, region: str, *arguments: str, **options: Any) -> dict:
        result = super().aws_json(region, *arguments, **options)
        if (
            arguments[1] == "describe-db-clusters"
            and self.membership_missing
            and self.primary is None
        ):
            result["DBClusters"][0]["DBClusterMembers"] = []
        return result

    def run(self, arguments: Sequence[str], **options: Any) -> str:
        result = super().run(arguments, **options)
        if arguments[2] == "create-db-instance":
            name = arguments[arguments.index("--db-instance-identifier") + 1]
            if name == "aurora-a-writer":
                self.primary = None
        elif arguments[2] == "describe-db-instances":
            self.primary = "aurora-a-writer"
        return result


def ensure(runner: InstanceCreationRunner) -> list[str]:
    return aurora.ensure_serverless_instances(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        availability_zones=["us-east-1a", "us-east-1b"],
        safe_name=lambda value, maximum: value[:maximum],
        wait=False,
    )


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("membership_missing", [False, True])
def test_initial_writer_designation_waits_for_creation(
    resumed: bool, membership_missing: bool
) -> None:
    runner = DelayedPrimaryRunner(
        resumed=resumed, membership_missing=membership_missing
    )

    assert ensure(runner) == ["aurora-a-writer", "aurora-a-reader"]
    assert runner.instances == {
        "aurora-a-writer": "available",
        "aurora-a-reader": "creating",
    }
    assert runner.transitions == (
        ([] if resumed else [("create", "aurora-a-writer")])
        + [("available", "aurora-a-writer"), ("create", "aurora-a-reader")]
    ), "reader creation must follow observed primary readiness"


def test_initial_writer_failure_does_not_create_reader() -> None:
    runner = DelayedPrimaryRunner(resumed=True)
    runner.fail_wait = "db-instance-available"

    with pytest.raises(BootstrapError, match="source readiness failed"):
        ensure(runner)

    assert runner.transitions == []
    assert "aurora-a-reader" not in runner.instances


def test_ready_cluster_without_primary_is_not_a_first_creation() -> None:
    runner = InstanceCreationRunner(
        {"aurora-a-writer": "available"}, cluster_status="available"
    )

    with pytest.raises(BootstrapError, match="replica source primary"):
        ensure(runner)

    assert runner.transitions == []


@pytest.mark.parametrize("stale_reads", [2, aurora.PRIMARY_MEMBERSHIP_ATTEMPTS])
def test_primary_membership_lag_is_bounded(
    monkeypatch: pytest.MonkeyPatch, stale_reads: int
) -> None:
    class StaleMembership(DelayedPrimaryRunner):
        reads = 0

        def aws_json(self, region: str, *arguments: str, **options: Any) -> dict:
            result = super().aws_json(region, *arguments, **options)
            if arguments[1] == "describe-db-clusters" and self.primary is not None:
                self.reads += 1
                if self.reads <= stale_reads:
                    result["DBClusters"][0]["DBClusterMembers"][0][
                        "IsClusterWriter"
                    ] = False
            return result

    sleeps: list[float] = []
    monkeypatch.setattr(aurora, "time", SimpleNamespace(sleep=sleeps.append))
    runner = StaleMembership(resumed=True)
    if stale_reads == aurora.PRIMARY_MEMBERSHIP_ATTEMPTS:
        with pytest.raises(BootstrapError, match="replica source primary"):
            ensure(runner)
        assert runner.transitions == [("available", "aurora-a-writer")]
        assert runner.reads == aurora.PRIMARY_MEMBERSHIP_ATTEMPTS
    else:
        ensure(runner)
        assert runner.instances["aurora-a-reader"] == "creating"
    assert sleeps == [aurora.PRIMARY_MEMBERSHIP_POLL_SECONDS] * min(
        stale_reads, aurora.PRIMARY_MEMBERSHIP_ATTEMPTS - 1
    )


@pytest.mark.parametrize("identity", ["foreign", "ambiguous", "malformed"])
def test_creation_wait_does_not_authorize_an_unknown_primary(identity: str) -> None:
    class UnknownPrimary(DelayedPrimaryRunner):
        def aws_json(self, region: str, *arguments: str, **options: Any) -> dict:
            result = super().aws_json(region, *arguments, **options)
            if arguments[1] == "describe-db-clusters":
                members = result["DBClusters"][0]["DBClusterMembers"]
                if identity == "foreign":
                    members[0]["DBInstanceIdentifier"] = "foreign"
                elif identity == "ambiguous":
                    members.append(
                        {"DBInstanceIdentifier": "other", "IsClusterWriter": False}
                    )
                else:
                    members[0]["IsClusterWriter"] = None
            return result

    runner = UnknownPrimary(resumed=True)
    with pytest.raises(BootstrapError, match="replica source primary"):
        ensure(runner)
    assert runner.transitions == []
