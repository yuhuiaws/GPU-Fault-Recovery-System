from __future__ import annotations

import json
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora
from gpu_fault.admin import bootstrap_common, execution
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_task_inputs import task_input_spec
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
        if arguments[2].startswith("describe-"):
            return json.dumps(self.aws_json("us-east-1", *arguments[1:]))
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
                {
                    "DBInstanceIdentifier": (
                        arguments[arguments.index("--db-instance-identifier") + 1]
                        if "--db-instance-identifier" in arguments
                        else "aurora-a-writer"
                    ),
                    "DBInstanceStatus": "available",
                    "DBInstanceClass": "db.serverless",
                }
            ]
        }


class InstanceCreationRunner(Runner):
    """Rejects replica creates until the actual primary and cluster are ready."""

    def __init__(
        self,
        instances: dict[str, str] | None = None,
        *,
        primary: str | None = None,
        cluster_status: str = "creating",
        fail_wait: str | None = None,
    ) -> None:
        super().__init__()
        self.instances = dict(instances or {})
        self.primary = primary
        self.cluster_status = cluster_status
        self.fail_wait = fail_wait
        self.transitions: list[tuple[str, str]] = []

    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> dict:
        self.calls.append(("aws", *arguments))
        if arguments[1] == "describe-db-clusters":
            return {
                "DBClusters": [
                    {
                        "DBClusterIdentifier": "aurora-a",
                        "Status": self.cluster_status,
                        "DBClusterMembers": [
                            {
                                "DBInstanceIdentifier": name,
                                "IsClusterWriter": name == self.primary,
                            }
                            for name in self.instances
                        ],
                    }
                ]
            }
        assert arguments[1] == "describe-db-instances", arguments
        selected = self.instances
        if "--db-instance-identifier" in arguments:
            name = arguments[arguments.index("--db-instance-identifier") + 1]
            if name not in selected:
                raise BootstrapError("DBInstanceNotFound")
            selected = {name: selected[name]}
        return {
            "DBInstances": [
                {
                    "DBInstanceIdentifier": name,
                    "DBClusterIdentifier": "aurora-a",
                    "DBInstanceStatus": status,
                }
                for name, status in selected.items()
            ]
        }

    def run(self, arguments: Sequence[str], **_keywords: Any) -> str:
        if arguments[2] == "create-db-instance":
            self.calls.append(tuple(arguments))
            name = arguments[arguments.index("--db-instance-identifier") + 1]
            if self.instances:
                assert self.cluster_status == "available", "cluster is not ready"
                assert self.instances.get(self.primary or "") == "available", (
                    "replica creation preceded primary readiness"
                )
            else:
                self.primary = name
            self.instances[name] = "creating"
            self.transitions.append(("create", name))
        elif arguments[2].startswith("describe-"):
            is_cluster = arguments[2] == "describe-db-clusters"
            waiter = "db-cluster-available" if is_cluster else "db-instance-available"
            if waiter == self.fail_wait:
                raise BootstrapError("source readiness failed")
            if is_cluster:
                self.cluster_status = "available"
                name = arguments[arguments.index("--db-cluster-identifier") + 1]
            else:
                name = arguments[arguments.index("--db-instance-identifier") + 1]
                self.instances[name] = "available"
            self.transitions.append(("available", name))
            return json.dumps(self.aws_json("us-east-1", *arguments[1:]))
        else:
            pytest.fail(f"unexpected operation: {arguments[2]}")
        return ""


class WaitBudgetRunner(InstanceCreationRunner):
    """Fake transport using the real timeout policy, without starting binaries."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 100.0
        self.request_seconds: dict[str, float] = {}
        self.pending_polls: dict[str, int] = {}
        self.poll_counts: dict[str, int] = {}
        self.sleeps: list[float] = []
        self.waiter_timeouts: list[tuple[str, float]] = []
        self.waiter_deadlines: list[execution.Deadline | None] = []
        self.wait_barrier: threading.Barrier | None = None

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def run_command(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        **_keywords: Any,
    ) -> subprocess.CompletedProcess[str]:
        timeout = execution.command_timeout(arguments, timeout_seconds)
        if arguments[2].startswith("describe-") and timeout_seconds is not None:
            waiter = (
                "db-cluster-available"
                if arguments[2] == "describe-db-clusters"
                else "db-instance-available"
            )
            self.waiter_timeouts.append((waiter, timeout))
            self.waiter_deadlines.append(execution.current_deadline())
            if self.wait_barrier is not None:
                self.wait_barrier.wait()
            duration = self.request_seconds.get(waiter, 0.0)
            self.now += min(duration, timeout)
            if duration > timeout:
                raise subprocess.TimeoutExpired(arguments, timeout)
            self.poll_counts[waiter] = self.poll_counts.get(waiter, 0) + 1
            if self.poll_counts[waiter] > self.pending_polls.get(waiter, 0):
                return subprocess.CompletedProcess(
                    arguments, 0, self.run(arguments), ""
                )
        if arguments[2].startswith("describe-"):
            value = self.aws_json("us-east-1", *arguments[1:])
            return subprocess.CompletedProcess(arguments, 0, json.dumps(value), "")
        self.run(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")


@pytest.fixture
def wait_budget_runner(monkeypatch: pytest.MonkeyPatch) -> WaitBudgetRunner:
    runner = WaitBudgetRunner()
    monkeypatch.delenv(execution.DEADLINE_ENV, raising=False)
    monkeypatch.delenv(execution.DEADLINE_LABEL_ENV, raising=False)
    monkeypatch.setattr(execution.time, "monotonic", lambda: runner.now)
    monkeypatch.setattr(aurora.time, "sleep", runner.sleep)
    monkeypatch.setattr(bootstrap_common, "run_command", runner.run_command)
    return runner


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


def _ensure_instances(
    runner: Runner,
    zones: Sequence[str] = ("us-east-1a", "us-east-1b"),
    *,
    wait: bool = True,
) -> list[str]:
    return aurora.ensure_serverless_instances(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        availability_zones=list(zones),
        safe_name=lambda value, maximum: value[:maximum],
        wait=wait,
    )


def _zones(runner: Runner) -> list[str]:
    return [
        arguments[arguments.index("--availability-zone") + 1]
        for arguments in runner.calls
        if "--availability-zone" in arguments
    ]


def test_missing_instances_are_created_in_their_own_availability_zone() -> None:
    """The initial writer takes the first zone and its replica the second."""

    runner = InstanceCreationRunner()

    instance_ids = _ensure_instances(runner)

    assert instance_ids == ["aurora-a-writer", "aurora-a-reader"]
    assert runner.instances == {
        "aurora-a-writer": "available",
        "aurora-a-reader": "available",
    }
    assert _operations(runner).count("create-db-instance") == 2
    create = next(call for call in runner.calls if call[2] == "create-db-instance")
    assert create[create.index("--db-instance-identifier") + 1] == "aurora-a-writer"
    assert runner.primary == "aurora-a-writer"
    assert _zones(runner) == ["us-east-1a", "us-east-1b"]


def test_existing_instances_are_waited_for_but_not_recreated() -> None:
    runner = InstanceCreationRunner(
        {"aurora-a-writer": "creating", "aurora-a-reader": "creating"},
        primary="aurora-a-writer",
    )

    instance_ids = _ensure_instances(runner)

    assert instance_ids == ["aurora-a-writer", "aurora-a-reader"]
    assert runner.instances == {
        "aurora-a-writer": "available",
        "aurora-a-reader": "available",
    }
    assert _operations(runner) == ["describe-db-instances"] * 4


@pytest.mark.parametrize("writer_status", ["creating", "available"])
def test_an_existing_writer_is_not_recreated(writer_status: str) -> None:
    """``create-db-instance`` on an existing identifier fails the whole
    bootstrap, even when a resumed foundation still has to add the reader."""

    runner = InstanceCreationRunner(
        {"aurora-a-writer": writer_status},
        primary="aurora-a-writer",
        cluster_status="available",
    )

    instance_ids = _ensure_instances(runner, wait=False)

    assert instance_ids == ["aurora-a-writer", "aurora-a-reader"]
    assert [
        name for transition, name in runner.transitions if transition == "create"
    ] == ["aurora-a-reader"], "the existing primary was recreated"
    assert runner.instances == {
        "aurora-a-writer": "available",
        "aurora-a-reader": "creating",
    }
    assert _zones(runner) == ["us-east-1b"]


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
        _ensure_instances(runner)

    assert "create-db-instance" not in _operations(runner)


def test_a_missing_availability_zone_is_refused_rather_than_defaulted() -> None:
    """One zone for two instances is a configuration error, not a placement:
    both members in one zone make the Aurora failover the runbook depends on a
    no-op, and the reader's zone would only be missed once ``aurora_ready``
    reached for it, minutes later."""

    runner = Runner(missing=["describe-db-instances"])

    with pytest.raises(ValueError, match="argument 2 is shorter"):
        _ensure_instances(runner, ["us-east-1a"])

    assert runner.calls == [], "invalid placement must fail before any AWS calls"


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


# --- instance readiness off the critical path -----------------------------------------


class ReadinessRunner(Runner):
    """Plays the instance side of RDS for the split ensure/await path.

    ``instance_statuses`` answers the one filtered describe the readiness code
    issues; ``secret_answers`` scripts ``get-secret-value`` (an exception is
    raised, a string returned); ``secret_present`` is what ``kubectl get secret``
    sees. ``wait_barrier`` lets a test prove both instance waits were in flight
    at the same time: a barrier of two only opens when both threads reach it.
    """

    def __init__(
        self,
        *,
        instance_statuses: dict[str, str],
        secret_answers: Sequence[str | Exception] = (),
        secret_present: bool = True,
        wait_barrier: threading.Barrier | None = None,
        **keywords: Any,
    ) -> None:
        super().__init__(**keywords)
        self.instance_statuses = dict(instance_statuses)
        self.secret_answers = list(secret_answers)
        self.secret_present = secret_present
        self.wait_barrier = wait_barrier
        self.wait_threads: list[int] = []

    def run(self, arguments: Sequence[str], **_keywords: Any) -> str:
        self.calls.append(tuple(arguments))
        if arguments[0] == "kubectl":
            return "gpu-fault-aurora" if self.secret_present else ""
        if arguments[2] == "describe-db-instances":
            self.wait_threads.append(threading.get_ident())
            if self.wait_barrier is not None:
                self.wait_barrier.wait()
            name = arguments[arguments.index("--db-instance-identifier") + 1]
            self.instance_statuses[name] = "available"
            return json.dumps(
                {
                    "DBInstances": [
                        {"DBInstanceIdentifier": name, "DBInstanceStatus": "available"}
                    ]
                }
            )
        return ""

    def aws_text(self, _region: str, *arguments: str, **_keywords: Any) -> str:
        self.calls.append(("aws", *arguments))
        answer = self.secret_answers.pop(0) if self.secret_answers else _SECRET
        if isinstance(answer, Exception):
            raise answer
        return answer

    def aws_json(self, region: str, *arguments: str, **keywords: Any) -> dict:
        if arguments[1] == "describe-db-instances" and "--filters" in arguments:
            self.calls.append(("aws", *arguments))
            return {
                "DBInstances": [
                    {"DBInstanceIdentifier": name, "DBInstanceStatus": status}
                    for name, status in self.instance_statuses.items()
                ]
            }
        value = super().aws_json(region, *arguments, **keywords)
        if arguments[1] == "describe-db-clusters":
            value["DBClusters"][0].update(
                {
                    "Endpoint": "c.cluster.example",
                    "MasterUserSecret": {
                        "SecretArn": "arn:aws:secretsmanager:x",
                        "KmsKeyId": "arn:aws:kms:x",
                    },
                }
            )
        return value


_SECRET = '{"username": "gpu_fault", "password": "p"}'
_NOT_FOUND = BootstrapError(
    "command failed (254): aws: An error occurred (ResourceNotFoundException) "
    "when calling the GetSecretValue operation: Secrets Manager can't find the "
    "specified secret."
)
_AVAILABLE = {"aurora-a-writer": "available", "aurora-a-reader": "available"}
_READER_CREATING = {"aurora-a-writer": "available", "aurora-a-reader": "creating"}


def _await_ready(runner: ReadinessRunner) -> aurora.AuroraReadiness:
    return aurora.await_aurora_ready(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        instance_ids=["aurora-a-writer", "aurora-a-reader"],
    )


def test_foundation_waits_for_the_writer_but_defers_reader_readiness() -> None:
    """The final readiness wait is optional, the replica source gate is not."""

    runner = InstanceCreationRunner()

    instance_ids = _ensure_instances(runner, wait=False)

    assert instance_ids == ["aurora-a-writer", "aurora-a-reader"]
    assert runner.instances == {
        "aurora-a-writer": "available",
        "aurora-a-reader": "creating",
    }
    assert runner.transitions == [
        ("create", "aurora-a-writer"),
        ("available", "aurora-a-writer"),
        ("available", "aurora-a"),
        ("create", "aurora-a-reader"),
    ], "reader creation preceded writer and cluster readiness"


class RdsOrderRunner(InstanceCreationRunner):
    """Use one resource state across foundation, readiness and secret reads."""

    def aws_json(self, region: str, *arguments: str, **keywords: Any) -> dict:
        value = super().aws_json(region, *arguments, **keywords)
        if arguments[1] == "describe-db-clusters":
            value["DBClusters"][0].update(
                {
                    "Endpoint": "c.cluster.example",
                    "MasterUserSecret": {
                        "SecretArn": "arn:aws:secretsmanager:x",
                        "KmsKeyId": "arn:aws:kms:x",
                    },
                }
            )
        return value

    def aws_text(self, _region: str, *arguments: str, **_keywords: Any) -> str:
        assert self.cluster_status == "available", "secret read before cluster ready"
        assert self.instances == _AVAILABLE, "secret read before both instances ready"
        self.calls.append(("aws", *arguments))
        self.transitions.append(("secret", "aurora-a"))
        return _SECRET


@pytest.mark.parametrize("writer_exists", [False, True])
def test_the_reader_is_created_only_once_the_cluster_and_writer_are_available(
    writer_exists: bool,
) -> None:
    """Fresh and resumed foundations gate the reader, then readiness gates secrets."""

    runner = RdsOrderRunner(
        {"aurora-a-writer": "creating"} if writer_exists else {},
        primary="aurora-a-writer" if writer_exists else None,
    )

    instance_ids = _ensure_instances(runner, wait=False)

    assert instance_ids == ["aurora-a-writer", "aurora-a-reader"]
    assert runner.instances == _READER_CREATING
    assert "get-secret-value" not in _operations(runner)

    ready = aurora.await_aurora_ready(
        runner, aws_region="us-east-1", cluster_id="aurora-a", instance_ids=instance_ids
    )

    initial_creates = [] if writer_exists else [("create", "aurora-a-writer")]
    assert runner.transitions == initial_creates + [
        ("available", "aurora-a-writer"),
        ("available", "aurora-a"),
        ("create", "aurora-a-reader"),
        ("available", "aurora-a-reader"),
        ("secret", "aurora-a"),
    ], "the reader or master secret crossed its readiness barrier"
    assert _zones(runner) == (
        ["us-east-1b"] if writer_exists else ["us-east-1a", "us-east-1b"]
    ), "the reader did not take the second zone"
    assert ready.username == "gpu_fault"


@pytest.mark.parametrize("failure", ["db-instance-available", "db-cluster-available"])
def test_failed_primary_readiness_prevents_reader_create_and_resumes(
    failure: str,
) -> None:
    runner = InstanceCreationRunner(fail_wait=failure)
    arguments = {
        "aws_region": "us-east-1",
        "cluster_id": "aurora-a",
        "availability_zones": ["us-east-1a", "us-east-1b"],
        "safe_name": lambda value, maximum: value[:maximum],
        "wait": False,
    }

    with pytest.raises(BootstrapError, match="source readiness failed"):
        aurora.ensure_serverless_instances(runner, **arguments)

    assert set(runner.instances) == {"aurora-a-writer"}
    runner.fail_wait = None
    aurora.ensure_serverless_instances(runner, **arguments)

    creates = [
        call[call.index("--db-instance-identifier") + 1]
        for call in runner.calls
        if call[2] == "create-db-instance"
    ]
    assert creates == ["aurora-a-writer", "aurora-a-reader"]
    assert runner.instances["aurora-a-reader"] == "creating"


def test_repair_waits_for_the_actual_primary_after_a_failover() -> None:
    runner = InstanceCreationRunner(
        {"aurora-a-reader": "rebooting"}, primary="aurora-a-reader"
    )

    aurora.ensure_serverless_instances(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        availability_zones=["us-east-1a", "us-east-1b"],
        safe_name=lambda value, maximum: value[:maximum],
        wait=False,
    )

    assert runner.primary == "aurora-a-reader"
    assert runner.instances == {
        "aurora-a-reader": "available",
        "aurora-a-writer": "creating",
    }
    assert runner.transitions == [
        ("available", "aurora-a-reader"),
        ("available", "aurora-a"),
        ("create", "aurora-a-writer"),
    ], "repair did not wait for the actual primary before creating its replica"


def test_unknown_primary_prevents_replica_creation() -> None:
    runner = InstanceCreationRunner(
        {"aurora-a-writer": "available"}, cluster_status="available"
    )
    with pytest.raises(BootstrapError, match="replica source primary"):
        aurora.ensure_serverless_instances(
            runner,
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )
    assert "create-db-instance" not in _operations(runner)


@pytest.mark.parametrize("changed", ["primary", "writer-status"])
def test_replica_source_is_revalidated_after_the_last_poll(changed: str) -> None:
    class ChangedSource(InstanceCreationRunner):
        def run(self, arguments: Sequence[str], **keywords: Any) -> str:
            output = super().run(arguments, **keywords)
            if arguments[2] == "describe-db-clusters":
                if changed == "primary":
                    self.primary = "other-primary"
                else:
                    self.instances["aurora-a-writer"] = "rebooting"
            return output

    runner = ChangedSource({"aurora-a-writer": "creating"}, primary="aurora-a-writer")

    with pytest.raises(BootstrapError, match="replica source changed"):
        aurora.ensure_serverless_instances(
            runner,
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )

    assert runner.transitions == [
        ("available", "aurora-a-writer"),
        ("available", "aurora-a"),
    ], "source revalidation was skipped after the waiters succeeded"
    assert "create-db-instance" not in _operations(runner), (
        "changed source identity/health authorized a replica create"
    )


def test_a_reader_unknown_to_an_old_checkpoint_is_waited_for_not_invented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness cannot invent a placement when a checkpoint member is not listed."""

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    runner = ReadinessRunner(instance_statuses={"aurora-a-writer": "available"})

    _await_ready(runner)

    operations = _operations(runner)
    assert "create-db-instance" not in operations, operations
    waits = [
        call
        for call in runner.calls
        if call[2] == "describe-db-instances" and "--db-instance-identifier" in call
    ]
    assert [call[call.index("--db-instance-identifier") + 1] for call in waits] == [
        "aurora-a-reader"
    ]
    assert runner.instance_statuses == _AVAILABLE
    assert runner.calls.index(waits[0]) < operations.index("get-secret-value"), (
        "the master secret was read before the unlisted reader became available"
    )


@pytest.mark.parametrize("pending_instance", ["aurora-a-writer", "aurora-a-reader"])
def test_failed_instance_readiness_prevents_secret_read(pending_instance: str) -> None:
    class FailedInstance(ReadinessRunner):
        def run(self, arguments: Sequence[str], **_keywords: Any) -> str:
            self.calls.append(tuple(arguments))
            assert arguments[2] == "describe-db-instances"
            assert (
                arguments[arguments.index("--db-instance-identifier") + 1]
                == pending_instance
            )
            return json.dumps(
                {
                    "DBInstances": [
                        {
                            "DBInstanceIdentifier": pending_instance,
                            "DBInstanceStatus": "failed",
                        }
                    ]
                }
            )

    runner = FailedInstance(
        instance_statuses={**_AVAILABLE, pending_instance: "creating"}
    )

    with pytest.raises(BootstrapError, match="terminal failure"):
        _await_ready(runner)

    assert _operations(runner) == ["describe-db-instances"] * 2, (
        "a failed instance authorized creation or a master secret read"
    )


def test_both_instance_waits_are_in_flight_at_the_same_time() -> None:
    """Existing instances may be observed concurrently; this does not bypass
    the primary-readiness dependency when creating a new replica."""

    runner = ReadinessRunner(
        instance_statuses={}, wait_barrier=threading.Barrier(2, timeout=5.0)
    )

    aurora.await_serverless_instances(
        runner,
        aws_region="us-east-1",
        instance_ids=["aurora-a-writer", "aurora-a-reader"],
    )

    assert _operations(runner) == ["describe-db-instances"] * 2
    assert len(set(runner.wait_threads)) == 2, "both waits ran on one thread"


@pytest.mark.parametrize("waiter", ["db-instance-available", "db-cluster-available"])
def test_rds_waiters_allow_polling_window_and_api_overhead(
    wait_budget_runner: WaitBudgetRunner, waiter: str
) -> None:
    wait_budget_runner.pending_polls[waiter] = 59
    wait_budget_runner.request_seconds[waiter] = 1.5

    with execution.deadline_scope(
        "bootstrap/aurora", task_input_spec("aurora").deadline_seconds
    ):
        aurora.ensure_serverless_instances(
            CommandRunner(),
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )

    assert wait_budget_runner.now == 100 + 1860
    assert wait_budget_runner.poll_counts[waiter] == 60
    assert wait_budget_runner.sleeps == [30] * 59
    assert all(
        timeout == 120 for _name, timeout in wait_budget_runner.waiter_timeouts
    ), "each describe must retain the ordinary request timeout"
    assert wait_budget_runner.transitions == [
        ("create", "aurora-a-writer"),
        ("available", "aurora-a-writer"),
        ("available", "aurora-a"),
        ("create", "aurora-a-reader"),
    ], "per-poll waits bypassed the replica source barrier"
    assert wait_budget_runner.instances["aurora-a-reader"] == "creating"


@pytest.mark.parametrize("root_seconds,task_seconds", [(10, 3600), (21600, 10)])
def test_parallel_instance_waiters_keep_the_remaining_parent_deadline(
    wait_budget_runner: WaitBudgetRunner, root_seconds: int, task_seconds: int
) -> None:
    wait_budget_runner.wait_barrier = threading.Barrier(2, timeout=5.0)

    with (
        execution.deployment_deadline("deploy", root_seconds),
        execution.deadline_scope("bootstrap/aurora_ready", task_seconds) as deadline,
    ):
        wait_budget_runner.now += 5
        aurora.await_serverless_instances(
            CommandRunner(),
            aws_region="us-east-1",
            instance_ids=["aurora-a-writer", "aurora-a-reader"],
        )

    assert wait_budget_runner.waiter_timeouts == [("db-instance-available", 5)] * 2
    assert [
        observed.expires if observed else None
        for observed in wait_budget_runner.waiter_deadlines
    ] == [deadline.expires] * 2, "parallel polling lost the inherited deadline"


def test_cluster_wait_uses_only_the_budget_left_by_the_writer(
    wait_budget_runner: WaitBudgetRunner,
) -> None:
    wait_budget_runner.request_seconds = {
        "db-instance-available": 20,
        "db-cluster-available": 20,
    }

    with (
        execution.deadline_scope("bootstrap/aurora", 30),
        pytest.raises(BootstrapError, match="exceeded its time budget"),
    ):
        aurora.ensure_serverless_instances(
            CommandRunner(),
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )

    assert wait_budget_runner.waiter_timeouts == [
        ("db-instance-available", 30),
        ("db-cluster-available", 10),
    ]
    assert set(wait_budget_runner.instances) == {"aurora-a-writer"}


def test_expired_parent_prevents_instance_waiter_start(
    wait_budget_runner: WaitBudgetRunner,
) -> None:
    with execution.deadline_scope("bootstrap/aurora_ready", 10):
        wait_budget_runner.now += 11
        with pytest.raises(BootstrapError, match="exceeded its time budget"):
            aurora.await_serverless_instances(
                CommandRunner(),
                aws_region="us-east-1",
                instance_ids=["aurora-a-writer", "aurora-a-reader"],
            )

    assert wait_budget_runner.waiter_timeouts == []
    assert wait_budget_runner.calls == []


@pytest.mark.parametrize("waiter", ["db-instance-available", "db-cluster-available"])
def test_rds_waiters_stop_at_their_operation_budget(
    wait_budget_runner: WaitBudgetRunner, waiter: str
) -> None:
    wait_budget_runner.pending_polls[waiter] = 60
    wait_budget_runner.request_seconds[waiter] = 6

    with pytest.raises(BootstrapError, match="exceeded its time budget"):
        aurora.ensure_serverless_instances(
            CommandRunner(),
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )

    assert wait_budget_runner.now == 100 + 2100
    assert wait_budget_runner.poll_counts[waiter] == 59
    assert wait_budget_runner.sleeps[-1] == 6
    assert set(wait_budget_runner.instances) == {"aurora-a-writer"}


def test_an_available_cluster_is_proved_ready_by_describes_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun against a live site must not pay for an ``rds wait`` per
    instance: one filtered describe says both are available, the secret is read
    once, and nothing is created or waited for."""

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    runner = ReadinessRunner(instance_statuses=_AVAILABLE)

    ready = _await_ready(runner)

    assert _operations(runner) == [
        "describe-db-instances",
        "describe-db-clusters",
        "get-secret-value",
    ]
    assert ready == aurora.AuroraReadiness(
        endpoint="c.cluster.example",
        secret_arn="arn:aws:secretsmanager:x",
        kms_key_arn="arn:aws:kms:x",
        username="gpu_fault",
        password="p",
    )
    assert "gpu_fault" not in repr(ready), "the credentials leaked into the repr"


def test_only_the_instances_still_creating_are_waited_for_and_the_secret_follows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    runner = ReadinessRunner(instance_statuses=_READER_CREATING)

    _await_ready(runner)

    operations = _operations(runner)
    waits = [
        call
        for call in runner.calls
        if call[2] == "describe-db-instances" and "--db-instance-identifier" in call
    ]
    assert [call[call.index("--db-instance-identifier") + 1] for call in waits] == [
        "aurora-a-reader"
    ], "an available instance was waited for, or a creating one was not"
    assert operations.index("describe-db-instances", 1) < operations.index(
        "get-secret-value"
    ), "the master secret was read before the instances were available"
    assert "create-db-instance" not in operations, (
        "a reader that already exists (still creating) was created again"
    )


def test_the_master_secret_read_retries_until_the_secret_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RDS reports the managed secret's ARN before Secrets Manager can serve its
    value; a read that lands in that window must try again, not fail the
    deploy and not fall back to an empty credential."""

    naps: list[float] = []
    monkeypatch.setattr(aurora.time, "sleep", naps.append)
    runner = ReadinessRunner(
        instance_statuses=_AVAILABLE, secret_answers=[_NOT_FOUND, _NOT_FOUND, _SECRET]
    )

    ready = _await_ready(runner)

    assert ready.username == "gpu_fault"
    assert _operations(runner).count("get-secret-value") == 3
    assert naps == [aurora.MASTER_SECRET_READ_POLL_SECONDS] * 2


def test_a_master_secret_that_never_appears_fails_closed_after_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    runner = ReadinessRunner(
        instance_statuses=_AVAILABLE,
        secret_answers=[_NOT_FOUND] * (aurora.MASTER_SECRET_READ_ATTEMPTS + 5),
    )

    with pytest.raises(BootstrapError, match="master user secret is not readable"):
        _await_ready(runner)

    assert (
        _operations(runner).count("get-secret-value")
        == aurora.MASTER_SECRET_READ_ATTEMPTS
    ), "the secret read did not stop at its bound"


def test_a_secret_read_denied_by_iam_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only "not there yet" is worth another attempt; a permission error will
    read the same on every try and must surface at once with its own message."""

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    denied = BootstrapError(
        "command failed (254): aws: An error occurred (AccessDeniedException) "
        "when calling the GetSecretValue operation"
    )
    runner = ReadinessRunner(
        instance_statuses=_AVAILABLE, secret_answers=[denied, _SECRET]
    )

    with pytest.raises(BootstrapError, match="AccessDeniedException"):
        _await_ready(runner)

    assert _operations(runner).count("get-secret-value") == 1


def _probe(runner: ReadinessRunner) -> None:
    aurora.assert_aurora_ready(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        instance_ids=["aurora-a-writer", "aurora-a-reader"],
        kubeconfig=Path("/state/cpu.kubeconfig"),
        namespace="gpu-fault-system",
    )


def test_the_readiness_probe_passes_on_an_available_cluster_with_its_secret() -> None:
    runner = ReadinessRunner(instance_statuses=_AVAILABLE)

    _probe(runner)

    aws = [call[2] for call in runner.calls if call[0] == "aws"]
    assert aws == ["describe-db-instances"], "the probe issued more than one RDS read"
    kubectl = [call for call in runner.calls if call[0] == "kubectl"]
    assert len(kubectl) == 1 and "gpu-fault-aurora" in kubectl[0], kubectl
    assert "--ignore-not-found" in kubectl[0], "a missing Secret would fail the probe"


def test_the_readiness_probe_reads_a_creating_instance_as_drift() -> None:
    runner = ReadinessRunner(instance_statuses=_READER_CREATING)

    with pytest.raises(BootstrapMutationRequired, match="aurora-a-reader"):
        _probe(runner)

    assert _operations(runner) == ["describe-db-instances"], (
        "the read-only probe started readiness polling or read the secret"
    )
    assert runner.wait_threads == [], "the read-only probe waited"


def test_the_readiness_probe_reads_a_missing_secret_as_drift() -> None:
    runner = ReadinessRunner(instance_statuses=_AVAILABLE, secret_present=False)

    with pytest.raises(BootstrapMutationRequired, match="gpu-fault-aurora"):
        _probe(runner)


def test_a_drifted_window_waits_for_creating_instances_before_the_modify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final instance readiness is deferred unless capacity reconciliation needs
    both members; a matching window costs neither a wait nor a describe."""

    monkeypatch.setattr(aurora.time, "sleep", lambda _seconds: None)
    runner = ReadinessRunner(
        instance_statuses={
            "aurora-a-writer": "creating",
            "aurora-a-reader": "available",
        }
    )
    cluster = {
        "ServerlessV2ScalingConfiguration": {"MinCapacity": 0.5, "MaxCapacity": 8}
    }

    aurora.reconcile_existing_capacity(
        runner,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        cluster=cluster,
        capacity=AuroraCapacityConfig(min_acu=16.0, max_acu=64.0),
        instance_ids=["aurora-a-writer", "aurora-a-reader"],
    )

    operations = _operations(runner)
    polls = [
        call
        for call in runner.calls
        if call[2] == "describe-db-instances"
        and "--db-instance-identifier" in call
        and call[call.index("--db-instance-identifier") + 1] == "aurora-a-writer"
    ]
    assert len(polls) == 1
    assert runner.calls.index(polls[0]) < operations.index("modify-db-cluster"), (
        "the capacity mutation preceded instance readiness"
    )

    settled = ReadinessRunner(
        instance_statuses={"aurora-a-writer": "creating", "aurora-a-reader": "creating"}
    )
    aurora.reconcile_existing_capacity(
        settled,
        aws_region="us-east-1",
        cluster_id="aurora-a",
        cluster=cluster,
        capacity=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
        instance_ids=["aurora-a-writer", "aurora-a-reader"],
    )
    assert settled.calls == [], "a matching window still touched RDS"
