from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpu_fault.admin import aurora_capacity as aurora
from gpu_fault.admin.config import AdminConfigError, AuroraCapacityConfig


def _cluster(min_acu: float, max_acu: float, *, status: str = "available") -> dict:
    return {
        "DBClusters": [
            {
                "Status": status,
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


def _scale_down_cluster(
    statuses: list[str], *, terminal: str = "available"
) -> tuple[dict, list[int]]:
    """An RDS stub that only starts moving after ``modify-db-cluster`` returns.

    ``statuses`` is consumed one entry per poll; once exhausted every later poll
    reports ``terminal``.
    """

    state = {"modified": False, "status": "available"}
    polls = [0]
    remaining = list(statuses)

    def fake_aws(_region: str, *arguments: str, **_kwargs):
        operation = arguments[1]
        if operation == "modify-db-cluster":
            state["modified"] = True
            return {}
        if operation == "describe-db-clusters":
            if not state["modified"]:
                return _cluster(8.0, 32.0)
            polls[0] += 1
            # The requested window is reported immediately; only the status
            # betrays that the change has not finished.
            state["status"] = remaining.pop(0) if remaining else terminal
            return _cluster(0.5, 8.0, status=str(state["status"]))
        if operation == "describe-db-instances":
            return {
                "DBInstances": [
                    {
                        "DBInstanceStatus": "available",
                        "DBInstanceClass": "db.serverless",
                    }
                ]
            }
        raise AssertionError(arguments)

    return {"fake_aws": fake_aws}, polls


def test_reconcile_waits_out_a_flip_that_starts_after_the_modify_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scaling down must not hand a ``modifying`` cluster to the release.

    The observed-ACU check only shields scale-ups. After a scale-down the live ACU
    sits trivially above the new floor, so nothing else stops the loop from
    accepting the first poll -- which still reads ``available`` with the new window
    because ``--apply-immediately`` answers before RDS moves. Accepting it lets
    ``_run_automatic_release`` start against a cluster about to go ``modifying``.
    """

    stub, polls = _scale_down_cluster(
        ["available", "modifying", "modifying", "available"]
    )
    monkeypatch.setattr(aurora, "_aws_json", stub["fake_aws"])

    result = aurora.reconcile_aurora_capacity(
        aws_region="us-east-1",
        cluster_id="aurora-a",
        expected=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        desired=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
        timeout_seconds=60,
        poll_seconds=0,
    )

    assert result["modified"] is True
    assert polls[0] == 6, (
        "the loop stopped on the first quiet poll instead of requiring "
        f"{aurora.CAPACITY_SETTLE_STABLE_POLLS} consecutive ones: {polls[0]} polls"
    )


def test_reconcile_reports_a_scale_down_that_never_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster wedged in ``modifying`` must surface, not be reported converged.

    The clock is faked rather than the timeout shortened: ``timeout_seconds=0``
    raises before the first observation, so the test would pass no matter what the
    cluster reports, and a real short timeout would spin the loop against the wall
    clock for however long it took.
    """

    stub, polls = _scale_down_cluster([], terminal="modifying")
    monkeypatch.setattr(aurora, "_aws_json", stub["fake_aws"])
    elapsed = iter([0.0, 30.0, 60.0, 90.0])
    monkeypatch.setattr(aurora.time, "monotonic", lambda: next(elapsed))

    with pytest.raises(AdminConfigError, match="did not converge"):
        aurora.reconcile_aurora_capacity(
            aws_region="us-east-1",
            cluster_id="aurora-a",
            expected=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
            desired=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
            timeout_seconds=60,
            poll_seconds=0,
        )

    assert polls[0] > 0, "the loop timed out without ever observing the cluster"


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
