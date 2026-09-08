from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.config import AuroraCapacityConfig
from gpu_fault.admin.config_file import initialize_desired_admin_config

# The error code RDS answers a describe of something that is not there with.
NOT_FOUND = {
    "describe-db-instances": "DBInstanceNotFound",
    "describe-db-cluster-parameter-groups": "DBParameterGroupNotFound",
}


class Runner:
    """Records the AWS calls instead of making them, and plays a two-instance
    Serverless v2 cluster whose window only changes after ``modify-db-cluster``.

    ``missing`` names the describes that answer "not there" the way RDS does:
    a failed read carrying the not-found code. The ensure paths decide from
    that one read, so the fake has no separate existence switch to get wrong.
    """

    def __init__(
        self,
        statuses: Sequence[str] = (),
        *,
        window: tuple[float, float] = (0.5, 8.0),
        missing: Sequence[str] = (),
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        # Each entry is the cluster status one settle poll observes; the list is
        # consumed in order and the last value repeats forever.
        self.statuses = list(statuses) or ["available"]
        self.window = window
        self.missing = frozenset(missing)

    def run(self, arguments: Sequence[str], **_keywords: Any) -> str:
        self.calls.append(tuple(arguments))
        return ""

    def _absent(self, operation: str) -> None:
        if operation in self.missing:
            raise BootstrapError(
                f"command failed (254): aws: An error occurred "
                f"({NOT_FOUND[operation]}) when calling the operation"
            )

    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> dict:
        self.calls.append(("aws", *arguments))
        operation = arguments[1]
        self._absent(operation)
        if operation == "describe-db-clusters":
            status = self.statuses[0]
            if len(self.statuses) > 1:
                self.statuses.pop(0)
            return {
                "DBClusters": [
                    {
                        "Status": status,
                        "ServerlessV2ScalingConfiguration": {
                            "MinCapacity": self.window[0],
                            "MaxCapacity": self.window[1],
                        },
                        "DBClusterMembers": [
                            {"DBInstanceIdentifier": "w", "IsClusterWriter": True},
                            {"DBInstanceIdentifier": "r", "IsClusterWriter": False},
                        ],
                    }
                ]
            }
        if operation == "modify-db-cluster":
            window = arguments[arguments.index("--apply-immediately") - 1]
            low, high = (float(item.split("=")[1]) for item in window.split(","))
            self.window = (low, high)
            return {}
        if operation == "get-metric-statistics":
            return {
                "Datapoints": [
                    {
                        "Timestamp": datetime.now(UTC).isoformat(),
                        "Maximum": self.window[0],
                    }
                ]
            }
        return {
            "DBInstances": [
                {"DBInstanceStatus": "available", "DBInstanceClass": "db.serverless"}
            ]
        }


def _operations(runner: Runner) -> list[str]:
    return [arguments[2] for arguments in runner.calls]


def _modify(runner: Runner) -> tuple[str, ...]:
    modifies = [call for call in runner.calls if call[2] == "modify-db-cluster"]
    assert len(modifies) == 1, _operations(runner)
    return modifies[0]


def test_bootstrap_reads_the_desired_capacity_rather_than_a_default(
    tmp_path: Path,
) -> None:
    """Bootstrap scales Aurora to the approved desired config, not to a constant.

    ``bootstrap_aurora_capacity`` is the only place the bootstrap path learns the
    ACU window, so a hardcoded fallback here would silently undo an approved
    capacity change on the next deploy.
    """

    initialize_desired_admin_config(tmp_path)

    capacity = aurora.bootstrap_aurora_capacity(tmp_path)

    assert (capacity.min_acu, capacity.max_acu) == (8.0, 32.0)


def test_reconcile_leaves_a_cluster_that_already_matches_alone() -> None:
    """Matching capacity must issue no call at all.

    ``modify-db-cluster --apply-immediately`` is a mutation on the live database
    even when the values are unchanged, so an unconditional call would make every
    deploy touch Aurora.
    """

    runner = Runner()

    aurora.reconcile_existing_capacity(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        cluster={
            "ServerlessV2ScalingConfiguration": {
                "MinCapacity": 8.0,
                "MaxCapacity": 32.0,
            }
        },
        capacity=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
    )

    assert runner.calls == []


@pytest.mark.parametrize(
    "cluster",
    [
        {"ServerlessV2ScalingConfiguration": {"MinCapacity": 0.5, "MaxCapacity": 8.0}},
        # A cluster created before Serverless v2 reports no scaling block at all;
        # reading that as "already correct" would leave it unscaled forever.
        {},
    ],
)
def test_reconcile_scales_a_cluster_that_does_not_match(
    cluster: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The modify goes through the runner (so ``--dry-run`` can hold it back)
    and is the same call ``aurora_capacity`` makes for ``config apply``: one
    writer, one settle rule."""

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    runner = Runner()

    aurora.reconcile_existing_capacity(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        cluster=cluster,
        capacity=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
    )

    modify = _modify(runner)
    assert "MinCapacity=8,MaxCapacity=32" in modify
    assert "--apply-immediately" in modify
    assert runner.window == (8.0, 32.0)
    # A scale-up is not done until CloudWatch shows the instances at the floor.
    assert "get-metric-statistics" in _operations(runner)


def test_reconcile_waits_out_a_flip_that_starts_after_the_modify_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capacity change must not hand a ``modifying`` cluster to the next step.

    ``modify-db-cluster --apply-immediately`` returns while the cluster still
    reads ``available`` and already reports the requested window, so one quiet
    poll proves nothing. This is not hypothetical: a deploy that scaled Aurora
    then failed its own ``aurora`` preflight check a few hundred lines later,
    because RDS flipped to ``modifying`` in between.
    """

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    # The first status is the pre-modify observation; the loop then sees the
    # late flip and needs three quiet polls after it.
    runner = Runner(["available", "modifying", "modifying", "available"])

    aurora.reconcile_existing_capacity(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        cluster={
            "ServerlessV2ScalingConfiguration": {"MinCapacity": 0.5, "MaxCapacity": 8.0}
        },
        capacity=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
    )

    operations = _operations(runner)
    polls = operations[operations.index("modify-db-cluster") :].count(
        "describe-db-clusters"
    )
    assert polls == 5, (
        "the settle loop stopped on the first quiet poll instead of requiring "
        f"{aurora.CAPACITY_SETTLE_STABLE_POLLS} consecutive ones: {polls} polls"
    )


def test_a_scale_that_never_settles_is_a_bootstrap_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared reconciler raises the admin-config error type; bootstrap's
    callers handle ``BootstrapError``, so it must arrive as one."""

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    elapsed = iter([0.0, 600.0, 1200.0, 1800.0, 2400.0])
    monkeypatch.setattr(aurora.time, "monotonic", lambda: next(elapsed))
    runner = Runner(["modifying"])

    with pytest.raises(aurora.BootstrapError, match="did not converge"):
        aurora.reconcile_existing_capacity(
            runner,
            aws_region="us-east-1",
            cluster_id="aurora-a",
            cluster={},
            capacity=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        )


def test_settle_gives_up_rather_than_polling_a_stuck_cluster_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster wedged in ``modifying`` has to surface, not hang the deploy."""

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    runner = Runner(["modifying"])

    with pytest.raises(aurora.BootstrapError, match="did not settle within"):
        aurora.wait_for_capacity_to_settle(
            runner, aws_region="us-east-1", cluster_id="aurora-a", timeout_seconds=0.0
        )


def test_missing_instances_are_created_in_their_own_availability_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer and reader must not land in the same AZ.

    Both replicas in one zone makes the Aurora failover the alerting runbook
    depends on a no-op, and the zone is only pinned at create time.
    """

    runner = Runner(missing=["describe-db-instances"])

    instance_ids = aurora.ensure_serverless_instances(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        availability_zones=["us-east-1a", "us-east-1b"],
        safe_name=lambda value, maximum: value[:maximum],
    )

    assert instance_ids == ["aurora-a-writer", "aurora-a-reader"]
    assert _operations(runner) == [
        "describe-db-instances",
        "create-db-instance",
        "describe-db-instances",
        "create-db-instance",
        "wait",
        "wait",
    ]
    zones = [
        arguments[arguments.index("--availability-zone") + 1]
        for arguments in runner.calls
        if "--availability-zone" in arguments
    ]
    assert zones == ["us-east-1a", "us-east-1b"]


def test_existing_instances_are_waited_for_but_not_recreated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running bootstrap must be idempotent and still confirm availability.

    ``create-db-instance`` on an existing identifier fails the whole bootstrap,
    and skipping the wait would let the deploy continue against an instance still
    in ``creating``.
    """

    runner = Runner()

    aurora.ensure_serverless_instances(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        availability_zones=["us-east-1a", "us-east-1b"],
        safe_name=lambda value, maximum: value[:maximum],
    )

    assert _operations(runner) == [
        "describe-db-instances",
        "describe-db-instances",
        "wait",
        "wait",
    ]


def test_a_throttled_instance_read_is_not_read_as_a_missing_instance() -> None:
    """``create-db-instance`` on an identifier that exists fails the bootstrap,
    so a describe that failed for any reason but not-found must stop here."""

    class Throttled(Runner):
        def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> dict:
            self.calls.append(("aws", *arguments))
            raise BootstrapError(
                "command failed (254): aws: An error occurred (Throttling) "
                "when calling the DescribeDBInstances operation: Rate exceeded"
            )

    runner = Throttled()

    with pytest.raises(BootstrapError, match="Throttling"):
        aurora.ensure_serverless_instances(
            runner,
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
        )

    assert "create-db-instance" not in _operations(runner)


def test_a_missing_availability_zone_is_refused_rather_than_defaulted() -> None:
    """One zone for two instances is a configuration error, not a placement.

    ``zip`` without ``strict`` would silently create only the writer, leaving a
    single-instance cluster that reads as a successful bootstrap.
    """

    runner = Runner(missing=["describe-db-instances"])

    with pytest.raises(ValueError, match="argument 2 is shorter"):
        aurora.ensure_serverless_instances(
            runner,
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a"],
            safe_name=lambda value, maximum: value[:maximum],
        )


class DiagnosticsRunner(Runner):
    """Plays the parameter-group side of RDS: the group's current values and
    the engine family, plus a quiet cluster for the settle wait."""

    def __init__(
        self, parameters: dict[str, str] | None = None, *, missing: Sequence[str] = ()
    ) -> None:
        super().__init__(missing=missing)
        self.parameters = dict(parameters or {})

    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> dict:
        self.calls.append(("aws", *arguments))
        self._absent(arguments[1])
        if arguments[1] == "describe-db-engine-versions":
            return {
                "DBEngineVersions": [{"DBParameterGroupFamily": "aurora-postgresql16"}]
            }
        if arguments[1] == "describe-db-cluster-parameters":
            return {
                "Parameters": [
                    {"ParameterName": name, "ParameterValue": value}
                    for name, value in self.parameters.items()
                ]
            }
        if arguments[1] == "describe-db-clusters":
            return {"DBClusters": [{"Status": "available"}]}
        return {"DBInstances": [{"DBInstanceStatus": "available"}]}


def _settled(parameters: dict[str, str]) -> dict[str, str]:
    return {
        **{name: value for name, value, _method in aurora.DIAGNOSTIC_PARAMETERS},
        **parameters,
    }


def test_a_missing_parameter_group_is_created_for_the_engine_family_and_filled() -> (
    None
):
    """The default group cannot be modified, so a fresh cluster gets its own
    group, in the family of its engine version, with the four diagnostic
    parameters set in one call."""

    runner = DiagnosticsRunner(missing=["describe-db-cluster-parameter-groups"])

    group = aurora.ensure_cluster_parameter_group(
        runner,
        aws_region="us-west-2",
        cluster_id="aurora-a",
        engine_version="16.8",
        safe_name=lambda value, maximum: value[:maximum],
    )

    assert group == "aurora-a-pg"
    assert _operations(runner) == [
        "describe-db-cluster-parameter-groups",
        "describe-db-engine-versions",
        "create-db-cluster-parameter-group",
        "modify-db-cluster-parameter-group",
    ]
    create = runner.calls[2]
    assert (
        create[create.index("--db-parameter-group-family") + 1] == "aurora-postgresql16"
    )
    modify = runner.calls[3]
    parameters = modify[modify.index("--parameters") + 1 :]
    assert (
        "ParameterName=log_lock_waits,ParameterValue=1,ApplyMethod=immediate"
        in parameters
    )
    assert (
        "ParameterName=shared_preload_libraries,ParameterValue=pg_stat_statements,"
        "ApplyMethod=pending-reboot"
    ) in parameters


def test_a_settled_parameter_group_is_left_alone() -> None:
    """A routine deploy must not modify a group whose values already match:
    the modify is a live mutation even when nothing changes."""

    runner = DiagnosticsRunner(_settled({"work_mem": "65536"}))

    aurora.ensure_cluster_parameter_group(
        runner,
        aws_region="us-west-2",
        cluster_id="aurora-a",
        engine_version="16.8",
        safe_name=lambda value, maximum: value[:maximum],
    )

    assert _operations(runner) == [
        "describe-db-cluster-parameter-groups",
        "describe-db-cluster-parameters",
    ]


def test_only_the_drifted_parameters_are_rewritten() -> None:
    """An operator's other parameters and the already-correct ones stay out of
    the modify call."""

    runner = DiagnosticsRunner(_settled({"log_lock_waits": "0"}))

    aurora.ensure_cluster_parameter_group(
        runner,
        aws_region="us-west-2",
        cluster_id="aurora-a",
        engine_version="16.8",
        safe_name=lambda value, maximum: value[:maximum],
    )

    modify = runner.calls[-1]
    assert modify[2] == "modify-db-cluster-parameter-group"
    assert modify[modify.index("--parameters") + 1 :] == (
        "ParameterName=log_lock_waits,ParameterValue=1,ApplyMethod=immediate",
    )


def test_an_aligned_cluster_needs_no_modify() -> None:
    runner = DiagnosticsRunner()

    changed = aurora.reconcile_cluster_diagnostics(
        runner,
        aws_region="us-west-2",
        cluster_id="aurora-a",
        cluster={
            "DBClusterParameterGroup": "aurora-a-pg",
            "EnabledCloudwatchLogsExports": ["postgresql"],
        },
        parameter_group="aurora-a-pg",
    )

    assert changed is False
    assert runner.calls == []


def test_a_cluster_on_the_default_group_gets_the_group_and_the_log_export_in_one_online_modify() -> (
    None
):
    """One ``modify-db-cluster --apply-immediately`` carries both changes; no
    instance is rebooted (the preload library stays pending-reboot), and the
    settle wait follows because the modify answers before the cluster flips."""

    runner = DiagnosticsRunner()

    changed = aurora.reconcile_cluster_diagnostics(
        runner,
        aws_region="us-west-2",
        cluster_id="aurora-a",
        cluster={"DBClusterParameterGroup": "default.aurora-postgresql16"},
        parameter_group="aurora-a-pg",
    )

    assert changed is True
    modify = runner.calls[0]
    assert modify[2] == "modify-db-cluster"
    assert (
        modify[modify.index("--db-cluster-parameter-group-name") + 1] == "aurora-a-pg"
    )
    assert (
        modify[modify.index("--cloudwatch-logs-export-configuration") + 1]
        == '{"EnableLogTypes":["postgresql"]}'
    )
    assert "--apply-immediately" in modify
    assert not any("reboot" in item for call in runner.calls for item in call), (
        "diagnostics must never reboot an instance; the preload library waits for the operator"
    )
    assert "describe-db-clusters" in _operations(runner), "the settle wait ran"


def test_an_existing_group_is_read_from_the_projected_parameter_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``aws rds describe-db-cluster-parameters --query Parameters[...]`` answers
    with the bare list, not ``{"Parameters": [...]}``. The first live deploy
    created the group and never listed it; the second one did and crashed."""

    # The group's existence is now probed through the runner
    # (``describe_or_absent``); ``DiagnosticsRunner`` answers that describe with
    # a document, so the group reads as existing without patching subprocess.
    del monkeypatch

    class ProjectedRunner(DiagnosticsRunner):
        def aws_json(self, region: str, *arguments: str, **keywords: Any) -> Any:
            value = super().aws_json(region, *arguments, **keywords)
            if arguments[1] == "describe-db-cluster-parameters":
                return value["Parameters"]
            return value

    runner = ProjectedRunner(_settled({"log_lock_waits": "0"}))

    aurora.ensure_cluster_parameter_group(
        runner,
        aws_region="us-west-2",
        cluster_id="aurora-a",
        engine_version="16.8",
        safe_name=lambda value, maximum: value[:maximum],
    )

    modify = runner.calls[-1]
    assert modify[2] == "modify-db-cluster-parameter-group", modify
    assert modify[modify.index("--parameters") + 1 :] == (
        "ParameterName=log_lock_waits,ParameterValue=1,ApplyMethod=immediate",
    ), modify
