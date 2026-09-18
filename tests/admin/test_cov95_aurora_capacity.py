from __future__ import annotations

import copy
import json
import subprocess
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.admin import aurora_capacity as capacity
from gpu_fault.admin.config import AdminConfigError, AuroraCapacityConfig


class AuroraTransport:
    def __init__(self):
        self.calls = []
        self.clock = 100.0
        self.late = False
        self.cluster_reads = 0
        self.cluster = {
            "Status": "available",
            "ServerlessV2ScalingConfiguration": {
                "MinCapacity": 8.0,
                "MaxCapacity": 32.0,
            },
            "DBClusterMembers": [
                {"DBInstanceIdentifier": "reader", "IsClusterWriter": False},
                {"DBInstanceIdentifier": "writer", "IsClusterWriter": True},
            ],
        }
        self.instance = {
            "DBInstanceStatus": "available",
            "DBInstanceClass": "db.serverless",
        }
        self.points = [{"Timestamp": datetime.now(UTC).isoformat(), "Maximum": 8.0}]

    def __call__(self, region, *arguments, mutate=False):
        self.calls.append((region, arguments, mutate))
        operation = arguments[1]
        if operation == "describe-db-clusters":
            self.cluster_reads += 1
            if self.late and self.cluster_reads == 2:
                self.clock += 2
            return {"DBClusters": [copy.deepcopy(self.cluster)]}
        if operation == "describe-db-instances":
            return {"DBInstances": [copy.deepcopy(self.instance)]}
        if operation == "get-metric-statistics":
            return {"Datapoints": copy.deepcopy(self.points)}
        if operation == "create-db-cluster":
            return {
                "DBCluster": {"DBClusterIdentifier": "example", "Status": "creating"}
            }
        raise AssertionError("unexpected fake Aurora request")

    def run(self, arguments, **_options):
        region = arguments[arguments.index("--region") + 1]
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps(self(region, *arguments[1:])), ""
        )

    def sleep(self, seconds):
        self.clock += seconds


@pytest.fixture
def transport(monkeypatch):
    value = AuroraTransport()
    monkeypatch.setattr(
        capacity,
        "time",
        SimpleNamespace(monotonic=lambda: value.clock, sleep=value.sleep),
    )
    monkeypatch.setattr(
        capacity,
        "subprocess",
        SimpleNamespace(**{**vars(subprocess), "run": value.run}),
    )
    return value


def test_capacity_wait_rejects_success_returning_after_its_deadline(transport):
    """COV95-ADMIN-004: a late positive RDS read cannot authorize the next stage."""
    transport.late = True
    with pytest.raises(AdminConfigError, match="did not converge"):
        capacity.request_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="example",
            desired=AuroraCapacityConfig(),
            timeout_seconds=1,
            poll_seconds=1,
            aws_json=transport,
        )
    assert all(not mutate for _region, _arguments, mutate in transport.calls), (
        "late capacity proof caused mutation"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("ServerlessV2ScalingConfiguration", ["invalid"]),
        ("DBClusterMembers", []),
        ("DBClusterMembers", [None, None]),
        ("DBClusterMembers", [{"DBInstanceIdentifier": ""}, {}]),
        (
            "DBClusterMembers",
            [
                {"DBInstanceIdentifier": "a", "IsClusterWriter": True},
                {"DBInstanceIdentifier": "b", "IsClusterWriter": True},
            ],
        ),
    ],
)
def test_capacity_observation_requires_complete_cluster_roles(transport, field, value):
    transport.cluster[field] = value
    with pytest.raises(AdminConfigError, match="invalid|exactly one|identifier"):
        capacity.observe_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="example",
            include_observed_capacity=False,
            aws_json=transport,
        )
    assert all(not mutate for _region, _arguments, mutate in transport.calls), (
        "invalid cluster observation caused mutation"
    )


@pytest.mark.parametrize(
    "points",
    [
        {},
        [None],
        [{"Timestamp": None, "Maximum": 8}],
        [{"Timestamp": "invalid", "Maximum": 8}],
        [{"Timestamp": "2026-09-12T00:00:00", "Maximum": 8}],
        [{"Timestamp": "2026-09-12T00:00:00Z", "Maximum": "8"}],
    ],
)
def test_unusable_capacity_samples_cannot_prove_scale_up(transport, points):
    transport.points = points
    with pytest.raises(
        AdminConfigError, match="did not converge|datapoints are invalid"
    ):
        capacity.await_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="example",
            desired=AuroraCapacityConfig(),
            modified=False,
            timeout_seconds=2,
            poll_seconds=1,
            aws_json=transport,
        )
    assert all(not mutate for _region, _arguments, mutate in transport.calls), (
        "unusable capacity samples caused mutation"
    )


@pytest.mark.parametrize("age", [timedelta(minutes=4), timedelta(minutes=-2)])
def test_stale_or_future_metric_is_not_a_current_capacity_proof(transport, age):
    transport.points[0]["Timestamp"] = (datetime.now(UTC) - age).isoformat()
    with pytest.raises(AdminConfigError, match="did not converge"):
        capacity.await_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="example",
            desired=AuroraCapacityConfig(),
            modified=False,
            timeout_seconds=2,
            poll_seconds=1,
            aws_json=transport,
        )


@pytest.mark.parametrize(
    "shape", [[], {}, {"DBClusters": []}, {"DBClusters": [{}, {}]}]
)
def test_capacity_lookup_requires_one_object(transport, monkeypatch, shape):
    monkeypatch.setattr(
        capacity.subprocess,
        "run",
        lambda arguments, **_options: subprocess.CompletedProcess(
            arguments, 0, json.dumps(shape), ""
        ),
    )
    with pytest.raises(AdminConfigError, match="non-object|exactly one"):
        capacity.observe_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="example",
            include_observed_capacity=False,
        )


@pytest.mark.parametrize(
    "status,output,error", [(1, "", "example denied"), (1, "", ""), (0, "invalid", "")]
)
def test_default_aurora_transport_failures_are_not_empty_success(
    transport, monkeypatch, status, output, error
):
    monkeypatch.setattr(
        capacity.subprocess,
        "run",
        lambda arguments, **_options: subprocess.CompletedProcess(
            arguments, status, output, error
        ),
    )
    with pytest.raises(AdminConfigError, match="command failed|invalid JSON"):
        capacity.observe_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="example",
            include_observed_capacity=False,
        )


@pytest.mark.parametrize("mode", ["create", "reconcile"])
def test_capacity_main_uses_only_the_installed_fake_transport(transport, capsys, mode):
    arguments = [
        mode,
        "--region",
        "us-east-1",
        "--cluster-id",
        "example",
        "--min-acu",
        "8",
        "--max-acu",
        "32",
    ]
    if mode == "create":
        arguments += [
            "--subnet-group",
            "example",
            "--security-group",
            "sg-example",
            "--parameter-group",
            "example",
        ]
    assert capacity.main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert (
        report["cluster_id"] == "example"
        if mode == "create"
        else report["modified"] is False
    )
    assert transport.calls, "capacity entrypoint never reached the fake AWS transport"


def test_capacity_main_rejects_invalid_window_before_any_aws_call(transport, capsys):
    assert (
        capacity.main(
            [
                "create",
                "--region",
                "us-east-1",
                "--cluster-id",
                "example",
                "--min-acu",
                "100",
                "--max-acu",
                "8",
                "--subnet-group",
                "example",
                "--security-group",
                "sg-example",
                "--parameter-group",
                "example",
            ]
        )
        == 1
    )
    assert "minAcu must not exceed maxAcu" in capsys.readouterr().err
    assert transport.calls == []
