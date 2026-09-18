from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any, Sequence, cast

import pytest

from gpu_fault.admin import bootstrap_aurora as aurora
from gpu_fault.admin import bootstrap_common, execution
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from tests.admin import test_admin_bootstrap_aurora as support

WaitBudgetRunner = support.WaitBudgetRunner
wait_budget_runner = support.wait_budget_runner


@pytest.mark.parametrize("waiter", ["db-instance-available", "db-cluster-available"])
def test_rds_waiters_stop_at_the_models_attempt_limit(
    wait_budget_runner: WaitBudgetRunner, waiter: str
) -> None:
    wait_budget_runner.pending_polls[waiter] = 60

    with pytest.raises(BootstrapError, match="exceeded 60 attempts"):
        aurora.ensure_serverless_instances(
            CommandRunner(),
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )

    assert wait_budget_runner.poll_counts[waiter] == 60
    assert wait_budget_runner.sleeps == [30] * 59
    assert set(wait_budget_runner.instances) == {"aurora-a-writer"}, (
        "attempt exhaustion authorized a dependent reader create"
    )


def test_rds_polling_uses_installed_model_interval_and_attempts(
    wait_budget_runner: WaitBudgetRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.loaders import Loader

    model = Loader().load_service_model("rds", "waiters-2")
    model["waiters"]["DBInstanceAvailable"].update(delay=7, maxAttempts=3)
    monkeypatch.setattr(Loader, "load_service_model", lambda *_args: model)
    wait_budget_runner.instances["aurora-a-writer"] = "creating"
    wait_budget_runner.pending_polls["db-instance-available"] = 3

    with pytest.raises(BootstrapError, match="exceeded 3 attempts"):
        aurora.await_serverless_instances(
            CommandRunner(), aws_region="us-east-1", instance_ids=["aurora-a-writer"]
        )

    assert wait_budget_runner.poll_counts == {"db-instance-available": 3}
    assert wait_budget_runner.sleeps == [7, 7]


@pytest.mark.parametrize("writer_status", ["creating", "available"])
def test_read_only_runner_refuses_instance_and_cluster_waits(
    wait_budget_runner: WaitBudgetRunner, writer_status: str
) -> None:
    wait_budget_runner.instances["aurora-a-writer"] = writer_status
    wait_budget_runner.primary = "aurora-a-writer"

    with pytest.raises(BootstrapMutationRequired):
        aurora.ensure_serverless_instances(
            ReadOnlyProbeRunner(CommandRunner()),
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )

    assert wait_budget_runner.waiter_timeouts == [], (
        "a read-only probe started a readiness poll"
    )
    assert wait_budget_runner.sleeps == [], "a read-only probe entered waiter sleep"
    assert wait_budget_runner.transitions == [], "a read-only probe mutated RDS"


def test_rds_sleep_is_bounded_by_remaining_task_time(
    wait_budget_runner: WaitBudgetRunner,
) -> None:
    wait_budget_runner.instances["aurora-a-writer"] = "creating"
    wait_budget_runner.pending_polls["db-instance-available"] = 60
    wait_budget_runner.request_seconds["db-instance-available"] = 2

    with (
        execution.deadline_scope("aurora-task", 10),
        pytest.raises(BootstrapError, match="exceeded its time budget"),
    ):
        aurora.await_serverless_instances(
            CommandRunner(), aws_region="us-east-1", instance_ids=["aurora-a-writer"]
        )

    assert wait_budget_runner.sleeps == [8]
    assert wait_budget_runner.poll_counts == {"db-instance-available": 1}
    assert wait_budget_runner.now == 110


def test_rds_sleep_respects_the_inherited_hard_stop(
    wait_budget_runner: WaitBudgetRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    wait_budget_runner.instances["aurora-a-writer"] = "creating"
    wait_budget_runner.pending_polls["db-instance-available"] = 60
    monkeypatch.setenv(execution.HARD_DEADLINE_ENV, str(wait_budget_runner.now + 7))

    with pytest.raises(BootstrapError, match="exceeded its time budget"):
        aurora.await_serverless_instances(
            CommandRunner(), aws_region="us-east-1", instance_ids=["aurora-a-writer"]
        )

    assert wait_budget_runner.waiter_timeouts == [("db-instance-available", 7)]
    assert wait_budget_runner.sleeps == [7]
    assert wait_budget_runner.now == 107


def test_rds_wait_inherits_recovery_without_extending_its_reserve(
    wait_budget_runner: WaitBudgetRunner,
) -> None:
    wait_budget_runner.instances["aurora-a-writer"] = "creating"
    wait_budget_runner.pending_polls["db-instance-available"] = 60

    with execution.deployment_deadline("deploy", 10, recovery_seconds=10):
        wait_budget_runner.now += 11
        with (
            execution.recovery_deadline("aurora recovery"),
            pytest.raises(BootstrapError, match="exceeded its time budget"),
        ):
            aurora.await_serverless_instances(
                CommandRunner(),
                aws_region="us-east-1",
                instance_ids=["aurora-a-writer"],
            )

    assert wait_budget_runner.waiter_timeouts == [("db-instance-available", 9)]
    assert wait_budget_runner.sleeps == [9]
    assert wait_budget_runner.now == 120


def test_a_stalled_rds_request_keeps_the_ordinary_command_limit(
    wait_budget_runner: WaitBudgetRunner,
) -> None:
    wait_budget_runner.request_seconds["db-instance-available"] = 121

    with pytest.raises(BootstrapError, match="exceeded its time budget"):
        aurora.await_serverless_instances(
            CommandRunner(), aws_region="us-east-1", instance_ids=["aurora-a-writer"]
        )

    assert wait_budget_runner.waiter_timeouts == [("db-instance-available", 120)]
    assert wait_budget_runner.now == 220
    assert wait_budget_runner.sleeps == [], "a timed-out request was retried"


def test_rds_success_after_the_waiter_deadline_is_not_accepted(
    wait_budget_runner: WaitBudgetRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def late_reply(
        arguments: Sequence[str], **keywords: Any
    ) -> subprocess.CompletedProcess[str]:
        result = wait_budget_runner.run_command(arguments, **keywords)
        wait_budget_runner.now += 2100
        return result

    monkeypatch.setattr(bootstrap_common, "run_command", late_reply)

    with pytest.raises(BootstrapError, match="exceeded its time budget"):
        aurora.await_serverless_instances(
            CommandRunner(), aws_region="us-east-1", instance_ids=["aurora-a-writer"]
        )

    assert wait_budget_runner.poll_counts == {"db-instance-available": 1}
    assert wait_budget_runner.sleeps == [], "a late success restarted the waiter"


class PollResponseRunner(support.InstanceCreationRunner):
    def __init__(self, kind: str, response: object) -> None:
        super().__init__(
            {"aurora-a-writer": "creating" if kind == "instance" else "available"},
            primary="aurora-a-writer",
        )
        self.operation = f"describe-db-{kind}s"
        self.response = response
        self.polls = 0

    def run(self, arguments: Sequence[str], **keywords: Any) -> str:
        if arguments[2] == self.operation:
            self.calls.append(tuple(arguments))
            self.polls += 1
            if isinstance(self.response, Exception):
                raise self.response
            return json.dumps(self.response)
        return super().run(arguments, **keywords)


@pytest.mark.parametrize("kind", ["instance", "cluster"])
@pytest.mark.parametrize(
    "status",
    [
        "deleted",
        "deleting",
        "failed",
        "incompatible-restore",
        "incompatible-parameters",
    ],
)
def test_aws_terminal_waiter_acceptors_block_replica_creation(
    kind: str, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = f"DB{kind.capitalize()}"
    runner = PollResponseRunner(
        kind,
        {
            f"{resource}s": [
                {
                    f"{resource}Identifier": (
                        "aurora-a-writer" if kind == "instance" else "aurora-a"
                    ),
                    "DBInstanceStatus" if kind == "instance" else "Status": status,
                }
            ]
        },
    )
    naps: list[float] = []
    monkeypatch.setattr(time, "sleep", naps.append)

    with pytest.raises(BootstrapError, match=f"terminal failure.*{status}"):
        aurora.ensure_serverless_instances(
            cast(CommandRunner, runner),
            aws_region="us-east-1",
            cluster_id="aurora-a",
            availability_zones=["us-east-1a", "us-east-1b"],
            safe_name=lambda value, maximum: value[:maximum],
            wait=False,
        )

    assert runner.polls == 1
    assert naps == [], "a terminal AWS state was retried"
    assert not any(call[2] == "create-db-instance" for call in runner.calls), (
        "a terminal primary/cluster state authorized a replica create"
    )


@pytest.mark.parametrize("code", ["DBInstanceNotFound", "AccessDenied", "Throttling"])
def test_waiter_aws_errors_are_terminal_without_an_sdk_retry(
    code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.session import Session

    error = BootstrapError(code)
    runner = PollResponseRunner("instance", error)
    monkeypatch.setattr(
        Session,
        "create_client",
        lambda *_args, **_kwargs: pytest.fail("waiter created an SDK network client"),
    )

    with pytest.raises(BootstrapError, match=code) as raised:
        aurora.await_serverless_instances(
            cast(CommandRunner, runner),
            aws_region="us-east-1",
            instance_ids=["aurora-a-writer"],
        )

    assert raised.value is error, "the original AWS failure was discarded"
    assert runner.polls == 1


@pytest.mark.parametrize(
    "entries",
    [
        [{"DBInstanceIdentifier": "other", "DBInstanceStatus": "available"}],
        [{"DBInstanceIdentifier": "aurora-a-writer", "DBInstanceStatus": "available"}]
        * 2,
        [{"DBInstanceStatus": "available"}],
        [None],
        None,
    ],
)
def test_polling_refuses_unbound_or_ambiguous_instance_identity(
    entries: object,
) -> None:
    runner = PollResponseRunner("instance", {"DBInstances": entries})

    with pytest.raises(BootstrapError, match="identity differs"):
        aurora.await_serverless_instances(
            cast(CommandRunner, runner),
            aws_region="us-east-1",
            instance_ids=["aurora-a-writer"],
        )

    assert runner.polls == 1


def test_empty_instance_list_is_not_a_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = PollResponseRunner("instance", {"DBInstances": []})
    naps: list[float] = []
    monkeypatch.setattr(time, "sleep", naps.append)

    with pytest.raises(BootstrapError, match="exceeded 60 attempts"):
        aurora.await_serverless_instances(
            cast(CommandRunner, runner),
            aws_region="us-east-1",
            instance_ids=["aurora-a-writer"],
        )

    assert runner.polls == 60
    assert naps == [30] * 59


def test_waiter_sleeps_outside_all_fake_transport_api_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.session import Session

    slots = threading.BoundedSemaphore(8)
    polls: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    def request(
        arguments: Sequence[str], **_keywords: Any
    ) -> subprocess.CompletedProcess[str]:
        with slots:
            assert arguments[:3] == ["aws", "rds", "describe-db-instances"], (
                "a long CLI waiter retained the transport's API slot"
            )
            polls.append(tuple(arguments))
            return subprocess.CompletedProcess(
                arguments,
                0,
                json.dumps(
                    {
                        "DBInstances": [
                            {
                                "DBInstanceIdentifier": "aurora-a-writer",
                                "DBInstanceStatus": (
                                    "available" if len(polls) == 2 else "creating"
                                ),
                            }
                        ]
                    }
                ),
                "",
            )

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        acquired = 0
        try:
            for _index in range(8):
                assert slots.acquire(blocking=False), (
                    "RDS poll sleep held capacity needed by independent AWS work"
                )
                acquired += 1
        finally:
            for _index in range(acquired):
                slots.release()

    monkeypatch.setattr(bootstrap_common, "run_command", request)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(
        Session,
        "create_client",
        lambda *_args, **_kwargs: pytest.fail("waiter created an SDK network client"),
    )

    aurora.await_serverless_instances(
        CommandRunner(), aws_region="us-east-1", instance_ids=["aurora-a-writer"]
    )

    assert len(polls) == 2
    assert sleeps == [30]


@pytest.mark.parametrize(
    "fatal_type", [ProcessSupervisionLost, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("fatal_first", [False, True])
def test_parallel_waiters_do_not_downgrade_fatal_control_flow(
    monkeypatch: pytest.MonkeyPatch, fatal_type: type[BaseException], fatal_first: bool
) -> None:
    ordinary = BootstrapError("ordinary AWS failure")
    fatal = fatal_type("fatal waiter failure")
    identifiers = ["aurora-a-writer", "aurora-a-reader"]
    failures = dict(
        zip(
            identifiers,
            (fatal, ordinary) if fatal_first else (ordinary, fatal),
            strict=True,
        )
    )
    started = threading.Barrier(2, timeout=5)
    completed: list[str] = []
    lock = threading.Lock()

    def fail_request(
        arguments: Sequence[str], **_keywords: Any
    ) -> subprocess.CompletedProcess[str]:
        identifier = arguments[arguments.index("--db-instance-identifier") + 1]
        try:
            started.wait()
            raise failures[identifier]
        finally:
            with lock:
                completed.append(identifier)

    monkeypatch.setattr(bootstrap_common, "run_command", fail_request)

    with pytest.raises(fatal_type) as raised:
        aurora.await_serverless_instances(
            CommandRunner(), aws_region="us-east-1", instance_ids=identifiers
        )

    assert raised.value is fatal, "an ordinary waiter failure hid fatal control flow"
    assert sorted(completed) == sorted(identifiers), "a started waiter was not drained"
