from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpu_fault import admin_aurora_capacity as aurora
from gpu_fault.admin_config import AdminConfigError, AuroraCapacityConfig


def _cluster(min_acu: float, max_acu: float) -> dict:
    return {
        "DBClusters": [
            {
                "Status": "available",
                "ServerlessV2ScalingConfiguration": {
                    "MinCapacity": min_acu,
                    "MaxCapacity": max_acu,
                },
                "DBClusterMembers": [
                    {"DBInstanceIdentifier": "writer-a", "IsClusterWriter": True},
                    {"DBInstanceIdentifier": "reader-a", "IsClusterWriter": False},
                ],
            }
        ]
    }


def test_reconcile_scales_up_and_waits_for_both_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = False
    commands: list[tuple[str, ...]] = []

    def fake_aws(_region: str, *arguments: str, **_kwargs):
        nonlocal modified
        commands.append(arguments)
        operation = arguments[1]
        if operation == "describe-db-clusters":
            return _cluster(8.0, 32.0) if modified else _cluster(0.5, 8.0)
        if operation == "describe-db-instances":
            return {
                "DBInstances": [
                    {
                        "DBInstanceStatus": "available",
                        "DBInstanceClass": "db.serverless",
                    }
                ]
            }
        if operation == "modify-db-cluster":
            modified = True
            return {}
        assert operation == "get-metric-statistics"
        return {
            "Datapoints": [
                {
                    "Timestamp": datetime.now(UTC).isoformat(),
                    "Maximum": 8.0 if modified else 0.5,
                }
            ]
        }

    monkeypatch.setattr(aurora, "_aws_json", fake_aws)

    result = aurora.reconcile_aurora_capacity(
        aws_region="us-east-1",
        cluster_id="aurora-a",
        expected=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
        desired=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert result["modified"] is True
    assert {item["actual_acu"] for item in result["after"]["instances"]} == {8.0}
    modify = next(args for args in commands if args[1] == "modify-db-cluster")
    assert "MinCapacity=8,MaxCapacity=32" in modify


def test_reconcile_accepts_an_already_applied_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = []

    def fake_aws(_region: str, *arguments: str, **_kwargs):
        operations.append(arguments[1])
        if arguments[1] == "describe-db-clusters":
            return _cluster(0.5, 8.0)
        if arguments[1] == "describe-db-instances":
            return {
                "DBInstances": [
                    {
                        "DBInstanceStatus": "available",
                        "DBInstanceClass": "db.serverless",
                    }
                ]
            }
        raise AssertionError(arguments)

    monkeypatch.setattr(aurora, "_aws_json", fake_aws)

    result = aurora.reconcile_aurora_capacity(
        aws_region="us-east-1",
        cluster_id="aurora-a",
        expected=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        desired=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert result["modified"] is False
    assert "modify-db-cluster" not in operations
    assert "get-metric-statistics" not in operations


def test_reconcile_rejects_unreviewed_live_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_aws(_region: str, *arguments: str, **_kwargs):
        if arguments[1] == "describe-db-clusters":
            return _cluster(16.0, 64.0)
        if arguments[1] == "describe-db-instances":
            return {
                "DBInstances": [
                    {
                        "DBInstanceStatus": "available",
                        "DBInstanceClass": "db.serverless",
                    }
                ]
            }
        raise AssertionError(arguments)

    monkeypatch.setattr(aurora, "_aws_json", fake_aws)

    with pytest.raises(AdminConfigError, match="differs from both"):
        aurora.reconcile_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="aurora-a",
            expected=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
            desired=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
            timeout_seconds=1,
            poll_seconds=0,
        )
