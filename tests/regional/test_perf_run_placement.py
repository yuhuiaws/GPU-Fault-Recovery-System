"""Every capacity run records where its measured components ran.

The burst suite read CloudWatch for ``GPU_FAULT_AURORA_INSTANCE`` and fell back
to a default instance name that is not the site's writer (its aurora.json
showed a 0.5 ACU instance beside a 124 ACU writer), and no run recorded the
writer or its Availability Zone. The writer is now resolved from the Aurora
cluster the run names, and run.json/summary.json carry the writer instance,
its AZ and the control-plane Pods' node AZs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_capacity_database as database
from scripts.perf import regional_capacity_placement as placement
from scripts.perf import regional_capacity_suite as capacity
from tests.regional._perf_caller_support import (
    PLACEMENT,
    CapacityWire,
    action_args,
    capacity_args,
)


def describe(
    writer: str = "aurora-a-1", members: tuple[str, ...] = ("aurora-a-1", "aurora-a-2")
):
    zones = {"aurora-a-1": "us-west-2b", "aurora-a-2": "us-west-2a"}
    calls: list[tuple[str, ...]] = []

    def aws_json(*arguments: str) -> dict[str, Any]:
        calls.append(arguments)
        assert arguments[:2] in {
            ("rds", "describe-db-clusters"),
            ("rds", "describe-db-instances"),
        }, "placement only reads RDS descriptions"
        assert arguments[2:4] == ("--region", "us-west-2"), arguments
        if arguments[1] == "describe-db-clusters":
            assert arguments[4:] == ("--db-cluster-identifier", "aurora-a")
            return {
                "DBClusters": [
                    {
                        "DBClusterMembers": [
                            {
                                "DBInstanceIdentifier": member,
                                "IsClusterWriter": member == writer,
                            }
                            for member in members
                        ]
                    }
                ]
            }
        instance = arguments[-1]
        return {"DBInstances": [{"AvailabilityZone": zones[instance]}]}

    return aws_json, calls


def test_the_writer_is_resolved_from_the_named_cluster_with_its_zone() -> None:
    aws_json, calls = describe(writer="aurora-a-1")

    writer = placement.aurora_writer(
        aws_json, cluster_id="aurora-a", aws_region="us-west-2"
    )

    assert writer == {
        "cluster_id": "aurora-a",
        "writer_instance_id": "aurora-a-1",
        "writer_availability_zone": "us-west-2b",
        "instances": {
            "aurora-a-1": {"role": "writer", "availability_zone": "us-west-2b"},
            "aurora-a-2": {"role": "reader", "availability_zone": "us-west-2a"},
        },
    }
    assert len(calls) == 3


def test_a_cluster_without_exactly_one_writer_fails_closed() -> None:
    aws_json, _calls = describe(writer="none")

    with pytest.raises(RuntimeError, match="0 writer instances"):
        placement.aurora_writer(aws_json, cluster_id="aurora-a", aws_region="us-west-2")


def control_plane(*args: str, **_kwargs: Any) -> str:
    if args[:2] == ("get", "nodes"):
        return json.dumps(
            {
                "items": [
                    {
                        "metadata": {
                            "name": f"node-{zone}",
                            "labels": {"topology.kubernetes.io/zone": zone},
                        }
                    }
                    for zone in ("us-west-2a", "us-west-2b")
                ]
            }
        )
    assert args[:2] == ("get", "pod"), args
    pods = [
        ("gpu-fault-api-ha-1", "gpu-fault-api-ha", "node-us-west-2a"),
        ("gpu-fault-api-ha-2", "gpu-fault-api-ha", "node-us-west-2b"),
        ("gpu-fault-control-worker-1", "gpu-fault-control-worker", "node-us-west-2a"),
        ("unrelated-1", "unrelated", "node-us-west-2a"),
    ]
    return json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": name, "labels": {"app": app}},
                    "spec": {"nodeName": node},
                }
                for name, app, node in pods
            ]
        }
    )


def test_control_plane_pods_are_mapped_to_their_node_zones() -> None:
    zones = placement.control_plane_zones(
        control_plane, prefixes=placement.CONTROL_POD_PREFIXES
    )

    assert zones["recorded"] is True
    assert zones["pods"]["gpu-fault-api-ha-2"] == {
        "role": "gpu-fault-api-ha",
        "node": "node-us-west-2b",
        "zone": "us-west-2b",
    }
    assert "unrelated-1" not in zones["pods"]
    assert zones["zones"] == {
        "us-west-2a": {"gpu-fault-api-ha": 1, "gpu-fault-control-worker": 1},
        "us-west-2b": {"gpu-fault-api-ha": 1},
    }


def test_unreadable_control_plane_zones_are_recorded_as_missing() -> None:
    zones = placement.control_plane_zones(
        lambda *_args, **_kwargs: "not json", prefixes=placement.CONTROL_POD_PREFIXES
    )

    assert zones == {"recorded": False, "error": "JSONDecodeError"}


def test_the_start_placement_lands_in_run_json_and_the_end_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"run_id": "run-a"}))
    snapshots = iter(
        [
            {"aurora": {"writer_instance_id": "aurora-a-1"}, "control_plane": {}},
            {"aurora": {"writer_instance_id": "aurora-a-2"}, "control_plane": {}},
        ]
    )
    monkeypatch.setattr(placement, "run_placement", lambda _cluster: next(snapshots))

    start = placement.start_placement("aurora-a", tmp_path)
    section = placement.placement_section(start, "aurora-a")

    run = json.loads((tmp_path / "run.json").read_text())
    assert run == {"run_id": "run-a", "placement": start}
    assert section["start"] == start
    assert section["end"]["aurora"]["writer_instance_id"] == "aurora-a-2"
    assert section["aurora_writer_changed"] is True, (
        "a failover during the run must be visible in the summary"
    )


def test_an_unreadable_end_placement_does_not_fail_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_cluster: str) -> dict[str, Any]:
        raise RuntimeError("rds unavailable")

    monkeypatch.setattr(placement, "run_placement", unavailable)
    start = {"aurora": {"writer_instance_id": "aurora-a-1"}}

    section = placement.placement_section(start, "aurora-a")

    assert section["end"] == {"error": "RuntimeError: rds unavailable"}
    assert section["aurora_writer_changed"] is None


def test_cluster_metrics_use_the_named_cluster_not_a_writer_name_suffix() -> None:
    dimensions: list[str] = []

    class Completed:
        stdout = json.dumps({"Datapoints": []})

    def run(argv: list[str], **_kwargs: Any) -> Completed:
        dimensions.append(argv[argv.index("--dimensions") + 1])
        return Completed()

    database.aurora_window(
        run,
        aurora_instance="aurora-a-1",
        aurora_cluster_id="aurora-a",
        aws_region="us-west-2",
        start=1_000.0,
        end=1_060.0,
    )

    assert "Name=DBClusterIdentifier,Value=aurora-a" in dimensions
    assert "Name=DBInstanceIdentifier,Value=aurora-a-1" in dimensions
    assert all("gpu-fault-aurora" not in item for item in dimensions), (
        "no metric may fall back to the old default instance name"
    )


def test_the_burst_suite_refuses_a_run_without_an_aurora_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    monkeypatch.setattr(capacity, "AURORA_CLUSTER_ID", "")

    with pytest.raises(SystemExit) as refused:
        capacity.main(capacity_args("all", tmp_path))

    assert refused.value.code == 2, "the refusal is a usage error"
    assert wire.events == [], "nothing may be registered or created before it"
    assert wire.objects == {}


def test_the_burst_suite_resolves_its_writer_before_registering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    windows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        capacity,
        "start_placement",
        lambda cluster, artifacts: wire.events.append(f"placement:{cluster}")
        or PLACEMENT,
    )
    monkeypatch.setattr(
        capacity, "aurora_window", lambda *_args, **kwargs: windows.append(kwargs) or {}
    )

    assert capacity.main(capacity_args("all", tmp_path)) == 0

    assert wire.events[0] == "placement:test-aurora", (
        "the writer must be resolved before the synthetic registration"
    )
    assert windows == [{"writer": PLACEMENT["aurora"]}], (
        "aurora.json must read the resolved writer"
    )


@pytest.mark.parametrize("named", [True, False], ids=["named", "unset"])
def test_the_action_suite_records_the_placement_when_the_round_names_aurora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: bool
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    snapshot = {
        "aurora": {"cluster_id": "aurora-a", "writer_instance_id": "aurora-a-1"},
        "control_plane": {"recorded": True},
    }
    monkeypatch.setattr(placement, "run_placement", lambda _cluster: dict(snapshot))
    if named:
        monkeypatch.setenv("GPU_FAULT_AURORA_CLUSTER_ID", "aurora-a")

    assert action.main(action_args(tmp_path)) == 0
    assert wire.events.count("publish") == 1, "the run registered once"

    run = json.loads((tmp_path / "run.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    if named:
        assert run["placement"] == snapshot
        assert summary["placement"]["start"] == snapshot
        assert summary["placement"]["aurora_writer_changed"] is False
    else:
        assert "placement" not in run
        assert summary["placement"] == {
            "recorded": False,
            "reason": "GPU_FAULT_AURORA_CLUSTER_ID is not set",
        }, "a round without the cluster still runs and says so"
