from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import aurora_capacity as aurora
from gpu_fault.admin.config import AdminConfigError, AuroraCapacityConfig

ROOT = Path(__file__).resolve().parents[2]


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


# ---------------------------------------------------------------------------
# One write path. A rollback once wrote 0.5/8 ACU through a path the deploy did
# not know about and the deploy could not correct it; the fix added yet another
# path instead of collapsing them. These pin the collapse.
# ---------------------------------------------------------------------------

# Spelled in two halves so this file is not itself a hit.
SCALING_FLAG = "--serverless-v2-" + "scaling-configuration"
SKIPPED_DIRECTORIES = frozenset({"node_modules", "dist", "build", "__pycache__"})


def _repo_text_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if path.suffix not in {".py", ".sh", ".md"} or not path.is_file():
            continue
        parents = path.relative_to(ROOT).parts[:-1]
        if any(part.startswith(".") or part in SKIPPED_DIRECTORIES for part in parents):
            continue
        files.append(path)
    return files


def test_the_scaling_flag_is_written_from_exactly_one_module() -> None:
    """Every ``modify``/``create`` that sets the ACU window goes through
    ``aurora_capacity``; bash, the perf harness, bootstrap and the manual all
    call it rather than spelling the flag themselves."""

    assert aurora.SERVERLESS_V2_SCALING_FLAG == SCALING_FLAG
    hits = sorted(
        path.relative_to(ROOT).as_posix()
        for path in _repo_text_files()
        if SCALING_FLAG in path.read_text(encoding="utf-8", errors="ignore")
    )

    assert hits == ["src/gpu_fault/admin/aurora_capacity.py"], hits


def _value(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def _spec(**overrides: Any) -> aurora.AuroraClusterSpec:
    values: dict[str, Any] = {
        "cluster_id": "gpu-fault-site-a-aurora",
        "subnet_group": "gpu-fault-site-a-aurora",
        "security_group_ids": ("sg-aurora",),
        "parameter_group": "gpu-fault-site-a-aurora-pg",
        "capacity": AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        "site_id": "site-a",
    }
    values.update(overrides)
    return aurora.AuroraClusterSpec(**values)


def test_create_arguments_pin_the_engine_and_carry_both_tags() -> None:
    """Bootstrap tagged ``gpu-fault:site-id`` (what ``aws_cleanup`` and the
    registry read) and deploy.sh tagged ``Application`` (what the console
    filter uses); a cluster from either path must satisfy both readers."""

    arguments = aurora.create_db_cluster_arguments(_spec())

    assert arguments[:2] == ["rds", "create-db-cluster"]
    assert "--region" not in arguments, "the caller supplies the region"
    assert _value(arguments, "--db-cluster-identifier") == "gpu-fault-site-a-aurora"
    assert _value(arguments, "--engine-version") == aurora.AURORA_ENGINE_VERSION
    assert aurora.AURORA_ENGINE_VERSION == "16.8"
    assert _value(arguments, "--engine-mode") == "provisioned"
    assert _value(arguments, SCALING_FLAG) == "MinCapacity=8,MaxCapacity=32"
    assert _value(arguments, "--db-subnet-group-name") == "gpu-fault-site-a-aurora"
    assert _value(arguments, "--vpc-security-group-ids") == "sg-aurora"
    assert (
        _value(arguments, "--db-cluster-parameter-group-name")
        == "gpu-fault-site-a-aurora-pg"
    )
    assert _value(arguments, "--enable-cloudwatch-logs-exports") == "postgresql"
    for flag in (
        "--manage-master-user-password",
        "--storage-encrypted",
        "--deletion-protection",
        "--copy-tags-to-snapshot",
        "--enable-iam-database-authentication",
    ):
        assert flag in arguments, flag
    tags = arguments[arguments.index("--tags") + 1 :]
    assert sorted(tags) == [
        "Key=Application,Value=gpu-fault-control-plane",
        "Key=gpu-fault:site-id,Value=site-a",
    ]


def test_create_arguments_without_a_site_carry_only_the_application_tag() -> None:
    arguments = aurora.create_db_cluster_arguments(_spec(site_id=None))

    tags = arguments[arguments.index("--tags") + 1 :]
    assert tags == ["Key=Application,Value=gpu-fault-control-plane"]


def test_create_arguments_validate_the_window_before_rendering() -> None:
    """``create-db-cluster`` accepts any syntactically valid pair, so rendering
    first would turn a rejected config into a live cluster."""

    with pytest.raises(AdminConfigError, match="0.5 ACU increments"):
        aurora.create_db_cluster_arguments(
            _spec(capacity=AuroraCapacityConfig(min_acu=0.7, max_acu=32.0))
        )


def test_scaling_configuration_validates_before_rendering() -> None:
    assert (
        aurora.scaling_configuration(AuroraCapacityConfig(min_acu=0.5, max_acu=32.0))
        == "MinCapacity=0.5,MaxCapacity=32"
    )

    with pytest.raises(AdminConfigError, match="0.5 ACU increments"):
        aurora.scaling_configuration(AuroraCapacityConfig(min_acu=0.7, max_acu=32.0))


class FakeRds:
    """An injected AWS caller: records every call with its ``mutate`` flag and
    plays a cluster that only reports the new window after the modify."""

    def __init__(self, min_acu: float, max_acu: float) -> None:
        self.window = (min_acu, max_acu)
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def __call__(
        self, _region: str, *arguments: str, mutate: bool = False
    ) -> dict[str, Any]:
        self.calls.append((arguments, mutate))
        operation = arguments[1]
        if operation == "describe-db-clusters":
            return _cluster(*self.window)
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
            window = _value(list(arguments), SCALING_FLAG)
            minimum, maximum = (float(item.split("=")[1]) for item in window.split(","))
            self.window = (minimum, maximum)
            return {}
        assert operation == "get-metric-statistics"
        return {
            "Datapoints": [
                {"Timestamp": datetime.now(UTC).isoformat(), "Maximum": self.window[0]}
            ]
        }

    def operations(self) -> list[str]:
        return [arguments[1] for arguments, _mutate in self.calls]


def test_reconcile_without_a_reviewed_expectation_accepts_the_live_window() -> None:
    """Bootstrap and the perf harness have no plan/apply record to compare
    against; they reconcile whatever is live to the desired window. Only the
    modify is a mutation, so a dry-run runner can hold it back."""

    rds = FakeRds(0.5, 8.0)

    result = aurora.reconcile_aurora_capacity(
        aws_region="us-east-1",
        cluster_id="aurora-a",
        desired=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        timeout_seconds=5,
        poll_seconds=0,
        aws_json=rds,
    )

    assert result["modified"] is True
    assert (result["after"]["min_acu"], result["after"]["max_acu"]) == (8.0, 32.0)
    mutations = [arguments[1] for arguments, mutate in rds.calls if mutate]
    assert mutations == ["modify-db-cluster"]


def test_reconcile_can_leave_the_settle_wait_to_a_dry_run_caller() -> None:
    """A dry run holds the modify back, so waiting for it would wait forever."""

    rds = FakeRds(0.5, 8.0)

    result = aurora.reconcile_aurora_capacity(
        aws_region="us-east-1",
        cluster_id="aurora-a",
        desired=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        timeout_seconds=5,
        poll_seconds=0,
        aws_json=rds,
        wait_for_settle=False,
    )

    operations = rds.operations()
    assert result["modified"] is True
    assert operations.count("modify-db-cluster") == 1
    assert operations[operations.index("modify-db-cluster") + 1 :] == []


def test_the_module_cli_reconciles_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """deploy.sh and the manual call ``python -m gpu_fault.admin.aurora_capacity``
    instead of spelling the modify themselves."""

    rds = FakeRds(0.5, 8.0)
    monkeypatch.setattr(aurora, "_aws_json", rds)

    status = aurora.main(
        [
            "reconcile",
            "--region",
            "us-east-1",
            "--cluster-id",
            "aurora-a",
            "--min-acu",
            "8",
            "--max-acu",
            "32",
            "--timeout-seconds",
            "5",
            "--poll-seconds",
            "0",
        ]
    )

    assert status == 0
    document = json.loads(capsys.readouterr().out)
    assert document["modified"] is True
    assert rds.window == (8.0, 32.0)


def test_the_module_cli_creates_through_the_shared_builder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    issued: list[tuple[str, tuple[str, ...]]] = []

    def fake_aws(region: str, *arguments: str, mutate: bool = False) -> dict[str, Any]:
        issued.append((region, arguments))
        return {
            "DBCluster": {"DBClusterIdentifier": arguments[3], "Status": "creating"}
        }

    monkeypatch.setattr(aurora, "_aws_json", fake_aws)

    status = aurora.main(
        [
            "create",
            "--region",
            "us-west-2",
            "--cluster-id",
            "gpu-fault-aurora",
            "--subnet-group",
            "gpu-fault-aurora",
            "--security-group",
            "sg-1",
            "--parameter-group",
            "gpu-fault-aurora-pg",
            "--min-acu",
            "8",
            "--max-acu",
            "32",
        ]
    )

    assert status == 0
    expected = aurora.create_db_cluster_arguments(
        aurora.AuroraClusterSpec(
            cluster_id="gpu-fault-aurora",
            subnet_group="gpu-fault-aurora",
            security_group_ids=("sg-1",),
            parameter_group="gpu-fault-aurora-pg",
            capacity=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        )
    )
    assert issued == [("us-west-2", tuple(expected))]
    document = json.loads(capsys.readouterr().out)
    assert document["cluster_id"] == "gpu-fault-aurora"
    assert document["status"] == "creating"


def test_the_module_cli_reports_a_rejected_window_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = aurora.main(
        [
            "reconcile",
            "--region",
            "us-east-1",
            "--cluster-id",
            "aurora-a",
            "--min-acu",
            "0.7",
            "--max-acu",
            "32",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "0.5 ACU increments" in captured.err
