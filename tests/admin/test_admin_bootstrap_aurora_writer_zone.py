"""The Aurora writer follows the CPU control plane's availability zone.

Live 2026-09-27: a fresh bootstrap put the writer in us-west-2b under a control
plane entirely in us-west-2a; the §8.4 50-cluster load then returned ~3x the
reserved 503s until the cluster was failed over (user ruling 2026-09-28: place
the writer with the control plane at creation).
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora

NODES = {
    "items": [
        {"metadata": {"labels": {"topology.kubernetes.io/zone": "us-east-1b"}}},
        {"metadata": {"labels": {"topology.kubernetes.io/zone": "us-east-1b"}}},
        {
            "metadata": {
                "labels": {"failure-domain.beta.kubernetes.io/zone": "us-east-1a"}
            }
        },
        {"metadata": {"labels": {}}},
    ]
}


def test_control_plane_node_zones_reads_both_zone_labels_and_skips_unlabelled() -> None:
    assert aurora.control_plane_node_zones(NODES) == [
        "us-east-1b",
        "us-east-1b",
        "us-east-1a",
    ]
    assert aurora.control_plane_node_zones({"items": []}) == []


def test_writer_first_zones_moves_the_dominant_control_plane_zone_first() -> None:
    zones = ["us-east-1a", "us-east-1b"]
    assert aurora.writer_first_zones(
        zones, ["us-east-1b", "us-east-1b", "us-east-1a"]
    ) == ["us-east-1b", "us-east-1a"]
    assert aurora.writer_first_zones(zones, ["us-east-1a"]) == zones
    # No control-plane zone has a private subnet: leave the order alone.
    assert aurora.writer_first_zones(zones, ["us-east-1c"]) == zones
    assert aurora.writer_first_zones(zones, []) == zones
    assert zones == ["us-east-1a", "us-east-1b"], "the input must not be mutated"


class _Cluster:
    """A two-member cluster; ``failover-db-cluster`` swaps the writer after one poll."""

    def __init__(
        self,
        writer: str,
        zones: dict[str, str],
        statuses: dict[str, str] | None = None,
        *,
        never_settle: bool = False,
    ):
        self.writer = writer
        self.zones = zones
        self.statuses = statuses or {name: "available" for name in zones}
        self.never_settle = never_settle
        self.calls: list[list[str]] = []
        self.mutations: list[list[str]] = []
        self.pending_target: str | None = None
        self.pending_polls = 0

    def run(self, arguments: Sequence[str], **kwargs: Any) -> str:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        if kwargs.get("mutate"):
            self.mutations.append(argv)
            if argv[2] == "failover-db-cluster":
                self.pending_target = argv[
                    argv.index("--target-db-instance-identifier") + 1
                ]
            return ""
        return json.dumps(self.aws_json("us-east-1", *argv[1:]))

    def aws_json(self, _region: str, *arguments: str, **_kwargs: Any) -> dict:
        self.calls.append(["aws", *arguments])
        if arguments[1] == "describe-db-clusters":
            status = "available"
            if self.pending_target is not None:
                # One poll sees the failover in flight; the next reports the new writer.
                self.pending_polls += 1
                if self.never_settle or self.pending_polls == 1:
                    status = "failing-over"
                else:
                    self.writer, self.pending_target = self.pending_target, None
            return {
                "DBClusters": [
                    {
                        "DBClusterIdentifier": "aurora-a",
                        "Status": status,
                        "DBClusterMembers": [
                            {
                                "DBInstanceIdentifier": name,
                                "IsClusterWriter": name == self.writer,
                            }
                            for name in self.zones
                        ],
                    }
                ]
            }
        assert arguments[1] == "describe-db-instances", arguments
        return {
            "DBInstances": [
                {
                    "DBInstanceIdentifier": name,
                    "DBInstanceStatus": self.statuses[name],
                    "AvailabilityZone": zone,
                }
                for name, zone in self.zones.items()
            ]
        }


def _affinity(cluster: _Cluster, control_zones: Sequence[str], **kwargs: Any) -> dict:
    return aurora.ensure_writer_zone_affinity(
        cluster,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        instance_ids=["aurora-a-writer", "aurora-a-reader"],
        control_plane_zones=control_zones,
        sleep=lambda _seconds: None,
        **kwargs,
    )


def test_a_writer_already_in_the_control_plane_zone_is_left_alone() -> None:
    cluster = _Cluster(
        "aurora-a-writer",
        {"aurora-a-writer": "us-east-1a", "aurora-a-reader": "us-east-1b"},
    )

    report = _affinity(cluster, ["us-east-1a"])

    assert report["action"] == "none" and report["writer_zone"] == "us-east-1a", report
    assert cluster.mutations == [], "no failover when the writer is already placed"


def test_a_writer_outside_the_control_plane_zone_is_failed_over_to_the_member_inside() -> (
    None
):
    cluster = _Cluster(
        "aurora-a-writer",
        {"aurora-a-writer": "us-east-1b", "aurora-a-reader": "us-east-1a"},
    )

    report = _affinity(cluster, ["us-east-1a", "us-east-1a"])

    assert report == {
        "action": "failover",
        "target": "aurora-a-reader",
        "writer": "aurora-a-writer",
        "writer_zone": "us-east-1b",
        "control_plane_zones": ["us-east-1a"],
    }
    assert [argv[2] for argv in cluster.mutations] == ["failover-db-cluster"]
    assert cluster.writer == "aurora-a-reader", (
        "the call must wait until RDS reports the new writer"
    )


def test_no_available_member_in_the_control_plane_zone_means_no_failover() -> None:
    cluster = _Cluster(
        "aurora-a-writer",
        {"aurora-a-writer": "us-east-1b", "aurora-a-reader": "us-east-1a"},
        statuses={"aurora-a-writer": "available", "aurora-a-reader": "creating"},
    )

    report = _affinity(cluster, ["us-east-1a"])

    assert report["action"] == "none" and "no available member" in report["reason"], (
        report
    )
    assert cluster.mutations == [], "a creating replica must not be a failover target"
    assert _affinity(cluster, [])["reason"] == "control-plane zones unknown"


def test_a_failover_that_never_settles_is_reported_not_waited_forever(
    monkeypatch,
) -> None:
    cluster = _Cluster(
        "aurora-a-writer",
        {"aurora-a-writer": "us-east-1b", "aurora-a-reader": "us-east-1a"},
        never_settle=True,
    )
    monkeypatch.setattr(aurora.time, "monotonic", lambda: 10_000.0)

    with pytest.raises(aurora.BootstrapError, match="did not settle"):
        _affinity(cluster, ["us-east-1a"], timeout_seconds=0.0)


def test_the_writer_is_promotion_tier_0_and_the_reader_tier_1() -> None:
    from tests.admin.test_admin_bootstrap_aurora import InstanceCreationRunner

    runner = InstanceCreationRunner()
    aurora.ensure_serverless_instances(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        availability_zones=["us-east-1a", "us-east-1b"],
        safe_name=lambda value, maximum: value[:maximum],
    )
    tiers = [
        (
            arguments[arguments.index("--db-instance-identifier") + 1],
            arguments[arguments.index("--promotion-tier") + 1],
        )
        for arguments in runner.calls
        if "--promotion-tier" in arguments
    ]
    assert tiers == [("aurora-a-writer", "0"), ("aurora-a-reader", "1")], tiers
