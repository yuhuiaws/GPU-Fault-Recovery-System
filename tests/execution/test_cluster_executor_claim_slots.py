"""``ClusterActionExecutor.run`` claims into free slots; it never waits for a batch.

The loop used to be ``run_once`` in a ``while``: claim up to ``batch_size``, wait
until every claimed command had reported, claim again. Each result was posted
when its command finished, but nothing new was claimed until the batch's longest
command was done, so a reboot of fifteen to forty-five minutes held its short
siblings' next steps and every new command behind it -- the quantizer batch 87
removed from the perf harness's simulated executor (``run_claim_loop``).

``LeasingControlPlane`` is the boundary these tests observe: PENDING commands
are claimable, a claim leases at most ``max_commands`` of them, a WAITING result
makes the command claimable again, a verdict closes it and may queue the
workflow's next step, and ``wait_seconds`` holds an empty claim until something
becomes claimable (bounded here). The executor and its lease lifecycle are the
real ones.
"""

from __future__ import annotations

import logging
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.cluster_executor.lease import SHUTDOWN_RELEASE_STATUS_SOURCE
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from tests.execution.test_cluster_executor_lease_and_report import (
    CLUSTER,
    EXECUTOR,
    FENCING_TOKEN,
    remote_command,
)

LOGGER_NAME = "gpu_fault.cluster_executor"
# ``eventually`` must keep sleeping for real while a test replaces time.sleep.
REAL_SLEEP = time.sleep


@dataclass(frozen=True)
class Claim:
    max_commands: int
    wait_seconds: float
    leased_before: int
    command_ids: tuple[str, ...]
    at: float


@dataclass(frozen=True)
class Report:
    command_id: str
    lease_token: str | None
    fencing_token: int
    result: RemoteCommandResult


class LeasingControlPlane:
    """The claim, renew and result routes with the store's lease semantics."""

    cluster_id = CLUSTER

    def __init__(
        self,
        commands: list[RemoteActionCommand],
        *,
        next_steps: dict[str, RemoteActionCommand] | None = None,
        hold_seconds: float = 0.2,
    ) -> None:
        self._condition = threading.Condition()
        self.pending = list(commands)
        self.next_steps = dict(next_steps or {})
        self.hold_seconds = hold_seconds
        self.leased: dict[str, RemoteActionCommand] = {}
        self.claims: list[Claim] = []
        self.renewals: list[tuple[str, str, int]] = []
        self.reports: list[Report] = []
        self.claim_errors: list[BaseException] = []
        self.failed_claims = 0

    def add(self, command: RemoteActionCommand) -> None:
        with self._condition:
            self.pending.append(command)
            self._condition.notify_all()

    def fail_next_claims(self, error: BaseException, count: int) -> None:
        with self._condition:
            self.claim_errors.extend([error] * count)

    def claim(
        self,
        executor_id: str,
        *,
        execution_owners: list[str],
        max_commands: int,
        lease_seconds: int,
        wait_seconds: float = 0,
    ) -> list[RemoteActionCommand]:
        at = time.monotonic()
        with self._condition:
            if self.claim_errors:
                self.failed_claims += 1
                raise self.claim_errors.pop(0)
            if wait_seconds > 0:
                self._condition.wait_for(
                    lambda: bool(self.pending),
                    timeout=min(wait_seconds, self.hold_seconds),
                )
            batch = self.pending[:max_commands]
            del self.pending[:max_commands]
            expires = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            leased = [
                command.model_copy(
                    update={
                        "status": RemoteCommandStatus.LEASED,
                        "lease_owner": executor_id,
                        "lease_expires_at": expires,
                    }
                )
                for command in batch
            ]
            self.claims.append(
                Claim(
                    max_commands=max_commands,
                    wait_seconds=wait_seconds,
                    leased_before=len(self.leased),
                    command_ids=tuple(command.command_id for command in leased),
                    at=at,
                )
            )
            self.leased.update((command.command_id, command) for command in leased)
            return leased

    def renew(
        self, command: RemoteActionCommand, executor_id: str, lease_seconds: int
    ) -> RemoteActionCommand:
        with self._condition:
            self.renewals.append((command.command_id, executor_id, lease_seconds))
            return command.model_copy(
                update={
                    "lease_expires_at": datetime.now(timezone.utc)
                    + timedelta(seconds=lease_seconds)
                }
            )

    def complete(
        self, command: RemoteActionCommand, result: RemoteCommandResult
    ) -> RemoteActionCommand:
        with self._condition:
            self.reports.append(
                Report(
                    command.command_id,
                    command.lease_token,
                    command.fencing_token,
                    result,
                )
            )
            self.leased.pop(command.command_id, None)
            if result.status is RemoteCommandStatus.WAITING:
                # Claimable again at once, carrying the record it posted.
                self.pending.append(
                    command.model_copy(
                        update={
                            "status": RemoteCommandStatus.WAITING,
                            "lease_owner": None,
                            "lease_expires_at": None,
                            "result_details": dict(result.details),
                        }
                    )
                )
            elif command.command_id in self.next_steps:
                self.pending.append(self.next_steps.pop(command.command_id))
            self._condition.notify_all()
        return command

    def reported_ids(self) -> list[str]:
        with self._condition:
            return [report.command_id for report in self.reports]

    def renewals_of(self, command_id: str) -> list[tuple[str, str, int]]:
        with self._condition:
            return [item for item in self.renewals if item[0] == command_id]


class GatedAdapter:
    """Holds each gated command until the test releases it; records the order."""

    owner = "owner-a"

    def __init__(
        self,
        *gated: str,
        waiting: frozenset[str] = frozenset(),
        unreachable: frozenset[str] = frozenset(),
    ) -> None:
        self.gates = {command_id: threading.Event() for command_id in gated}
        self.waiting = waiting
        self.unreachable = unreachable
        self.lock = threading.Lock()
        self.started: list[str] = []
        self.active = 0
        self.max_active = 0

    def supports(self, _step: Any) -> bool:
        return True

    def release(self, command_id: str) -> None:
        self.gates[command_id].set()

    def release_all(self) -> None:
        for gate in self.gates.values():
            gate.set()

    def started_ids(self) -> list[str]:
        with self.lock:
            return list(self.started)

    def execute(self, context: Any) -> WorkflowStepOutcome:
        command_id = context.idempotency_key.removeprefix("idem-")
        with self.lock:
            self.started.append(command_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            gate = self.gates.get(command_id)
            if gate is not None and not gate.wait(10):
                raise RuntimeError(f"{command_id} was never released")
            if command_id in self.unreachable:
                raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure")
        finally:
            with self.lock:
                self.active -= 1
        if command_id in self.waiting:
            return WorkflowStepOutcome(status=WorkflowStepStatus.WAITING)
        return WorkflowStepOutcome(status=WorkflowStepStatus.SUCCEEDED)


class FastRenewalTick(threading.Event):
    """The renewer's tick (a third of the lease, at most 30 s), 500 times faster."""

    def wait(self, timeout: float | None = None) -> bool:
        return super().wait(None if timeout is None else timeout / 500)


class LoopThread(threading.Thread):
    """``executor.run()`` on its own thread, keeping whatever it raised."""

    def __init__(self, executor: ClusterActionExecutor) -> None:
        super().__init__(name="claim-loop-under-test", daemon=True)
        self.executor = executor
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.executor.run()
        except BaseException as exc:  # noqa: BLE001 - re-raised by the test
            self.error = exc


@contextmanager
def running(
    executor: ClusterActionExecutor, adapter: GatedAdapter
) -> Iterator[LoopThread]:
    loop = LoopThread(executor)
    loop.start()
    try:
        yield loop
    finally:
        executor.request_stop("test finished")
        adapter.release_all()
        loop.join(10)
    assert not loop.is_alive(), "run() did not return after the stop and the drain"
    assert loop.error is None, f"run() raised: {loop.error!r}"


def eventually(condition: Callable[[], bool], message: str) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, message
        REAL_SLEEP(0.005)


def slot_executor(
    client: LeasingControlPlane, adapter: GatedAdapter, tmp_path: Path, **overrides: Any
) -> ClusterActionExecutor:
    settings: dict[str, Any] = {
        "poll_seconds": 0.02,
        "liveness_interval_seconds": 0.05,
        "claim_state_path": str(tmp_path / "claim-state.json"),
        "liveness_state_path": str(tmp_path / "executor-loop-alive"),
        **overrides,
    }
    return ClusterActionExecutor(
        client,
        [adapter],
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        **settings,
    )


def finished_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage().split()[4].removeprefix("command=")
        for record in caplog.records
        if record.name == LOGGER_NAME
        and record.getMessage().startswith("regional cluster executor finished")
    ]


def test_a_long_command_no_longer_holds_its_batch_siblings_next_steps(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The batch's short commands report, their next steps are claimed and
    finished, and only then does the reboot that shared their claim end."""

    client = LeasingControlPlane(
        [
            remote_command("reboot"),
            remote_command("drain-a"),
            remote_command("drain-b"),
        ],
        next_steps={
            "drain-a": remote_command("validate-a"),
            "drain-b": remote_command("validate-b"),
        },
    )
    adapter = GatedAdapter("reboot")
    executor = slot_executor(
        client, adapter, tmp_path, batch_size=3, max_concurrent_commands=3
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        with running(executor, adapter):
            eventually(
                lambda: {"validate-a", "validate-b"} <= set(client.reported_ids()),
                "the next steps of the short commands waited for the reboot",
            )
            eventually(
                lambda: {"validate-a", "validate-b"} <= set(finished_lines(caplog)),
                "the loop never concluded the next steps on their own",
            )
            assert "reboot" not in client.reported_ids(), "the reboot is still running"
            adapter.release("reboot")
            eventually(
                lambda: "reboot" in client.reported_ids(), "the reboot never reported"
            )

    reported = client.reported_ids()
    assert sorted(reported) == [
        "drain-a",
        "drain-b",
        "reboot",
        "validate-a",
        "validate-b",
    ], reported
    assert reported[-1] == "reboot", f"results must land in duration order: {reported}"
    assert reported.index("drain-a") < reported.index("validate-a"), reported
    assert reported.index("drain-b") < reported.index("validate-b"), reported
    assert finished_lines(caplog)[-1] == "reboot", (
        "each command is concluded when it reports, not after its whole batch: "
        f"{finished_lines(caplog)}"
    )
    first = client.claims[0]
    assert (first.max_commands, first.command_ids) == (
        3,
        ("reboot", "drain-a", "drain-b"),
    ), first
    assert all(
        report.result.status is RemoteCommandStatus.SUCCEEDED
        for report in client.reports
    ), client.reports


def test_a_freed_slot_is_claimed_again_at_once_within_the_batch_bounds(
    tmp_path: Path,
) -> None:
    """Six short commands cycle through the one slot the long command leaves,
    and a claim never asks for more than the slots that are free."""

    shorts = [remote_command(f"short-{index}") for index in range(6)]
    client = LeasingControlPlane([remote_command("long"), *shorts])
    adapter = GatedAdapter("long")
    executor = slot_executor(
        client, adapter, tmp_path, batch_size=2, max_concurrent_commands=3
    )

    with running(executor, adapter):
        eventually(
            lambda: {command.command_id for command in shorts}
            <= set(client.reported_ids()),
            "the short commands never got the free slot",
        )
        assert "long" not in client.reported_ids(), "the long command is running"
        adapter.release("long")
        eventually(lambda: "long" in client.reported_ids(), "long never reported")

    assert adapter.max_active == 2, (
        "in flight is bounded by min(batch_size, max_concurrent_commands): "
        f"{adapter.max_active}"
    )
    assert client.claims[0].command_ids == ("long", "short-0"), client.claims[0]
    assert all(
        1 <= claim.max_commands <= 2 - claim.leased_before
        and len(claim.command_ids) <= claim.max_commands
        for claim in client.claims
    ), f"a claim asked for more than the free slots: {client.claims}"
    reused = [
        claim
        for claim in client.claims
        if claim.command_ids and claim.command_ids[0].startswith("short-")
    ]
    assert [claim.command_ids for claim in reused] == [
        (f"short-{index}",) for index in range(1, 6)
    ], f"each freed slot must be claimed into on its own: {client.claims}"
    assert all(claim.max_commands == 1 for claim in reused), reused


def test_a_long_command_keeps_renewing_while_short_ones_cycle_past_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The long command's renewer keeps its lease alive the whole time, and the
    lease, fencing and the one report per command are what they always were."""

    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", FastRenewalTick)
    shorts = [remote_command(f"short-{index}") for index in range(4)]
    client = LeasingControlPlane([remote_command("long"), *shorts])
    adapter = GatedAdapter("long")
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=2,
        max_concurrent_commands=2,
        lease_seconds=30,
    )

    with running(executor, adapter):
        eventually(
            lambda: len(client.renewals_of("long")) >= 6
            and {command.command_id for command in shorts}
            <= set(client.reported_ids()),
            "the long command stopped renewing while the others ran",
        )
        adapter.release("long")
        eventually(lambda: "long" in client.reported_ids(), "long never reported")

    assert set(client.renewals_of("long")) == {("long", EXECUTOR, 30)}, (
        "every renewal is this executor's, for the configured lease"
    )
    assert sorted(report.command_id for report in client.reports) == sorted(
        ["long", *(command.command_id for command in shorts)]
    ), "exactly one report per command"
    for report in client.reports:
        assert report.result.status is RemoteCommandStatus.SUCCEEDED, report
        assert report.result.lease_token == report.lease_token, report
        assert report.lease_token == f"lease-{report.command_id}", report
        assert report.fencing_token == FENCING_TOKEN, report
    counters = executor.metrics_snapshot()
    assert {
        name: counters[name]
        for name in (
            "lease_renewal_failures",
            "lease_lost_total",
            "results_withheld_total",
            "reported_failures",
            "unexpected_failures",
        )
    } == dict.fromkeys(
        (
            "lease_renewal_failures",
            "lease_lost_total",
            "results_withheld_total",
            "reported_failures",
            "unexpected_failures",
        ),
        0,
    ), counters


def test_sigterm_drains_the_commands_in_flight_and_claims_nothing_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Stop claiming, let the leases lapse, still report what was done -- and
    return only once the last command in flight has reported."""

    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", FastRenewalTick)
    client = LeasingControlPlane([remote_command("reboot"), remote_command("drain")])
    adapter = GatedAdapter("reboot", "drain")
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=3,
        max_concurrent_commands=3,
        lease_seconds=30,
    )

    def logged(fragment: str) -> bool:
        return any(fragment in record.getMessage() for record in caplog.records)

    previous = signal.getsignal(signal.SIGTERM)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        try:
            executor.install_signal_handlers()
            with running(executor, adapter) as loop:
                eventually(
                    lambda: set(adapter.started_ids()) == {"reboot", "drain"},
                    "both commands should be running",
                )
                signal.raise_signal(signal.SIGTERM)
                reason = executor.stop_reason
                eventually(
                    lambda: logged("draining before it stops"),
                    "the loop never entered the drain",
                )
                client.add(remote_command("late"))
                eventually(
                    lambda: logged(
                        "lease left to lapse during shutdown: command=reboot"
                    ),
                    "the in-flight reboot kept renewing after SIGTERM",
                )
                renewals_after_stop = len(client.renewals_of("reboot"))
                adapter.release("drain")
                eventually(
                    lambda: "drain" in client.reported_ids(), "drain never reported"
                )
                assert loop.is_alive(), "run() returned with the reboot in flight"
                adapter.release("reboot")
                loop.join(10)
                assert not loop.is_alive(), "run() did not return after the drain"
        finally:
            signal.signal(signal.SIGTERM, previous)

    assert reason == "SIGTERM received", reason
    assert client.reported_ids() == ["drain", "reboot"], client.reported_ids()
    assert all(
        report.result.status is RemoteCommandStatus.SUCCEEDED
        for report in client.reports
    ), "work done before the stop is still reported under the lease it holds"
    assert [command.command_id for command in client.pending] == ["late"], (
        "nothing is claimed after the stop"
    )
    assert "late" not in adapter.started_ids(), adapter.started_ids()
    assert len(client.renewals_of("reboot")) == renewals_after_stop, (
        "the lease of a command in flight is left to lapse, not renewed"
    )
    messages = [record.getMessage() for record in caplog.records]
    stopped = next(
        index
        for index, message in enumerate(messages)
        if message.startswith("regional cluster executor stopped claiming")
    )
    finished_reboot = next(
        index
        for index, message in enumerate(messages)
        if message.startswith("regional cluster executor finished command=reboot")
    )
    assert finished_reboot < stopped, "run() stopped before its last command reported"


def test_a_claim_that_returns_after_sigterm_hands_its_commands_straight_back(
    tmp_path: Path,
) -> None:
    """The long-poll claim on the wire when SIGTERM lands can still lease work
    to the exiting process; the loop releases it at once and runs nothing."""

    record = {"node_action_command_id": "idem-command-a/node-a/agent-1"}

    class StoppedDuringTheClaim(LeasingControlPlane):
        def claim(self, executor_id: str, **kwargs: Any) -> list[RemoteActionCommand]:
            commands = super().claim(executor_id, **kwargs)
            if commands:
                executor.request_stop("SIGTERM received")
            return commands

    client = StoppedDuringTheClaim(
        [remote_command("command-a", result_details=dict(record))]
    )
    adapter = GatedAdapter()
    executor = slot_executor(client, adapter, tmp_path)

    executor.run()

    assert adapter.started_ids() == [], "a stopping executor runs nothing new"
    (released,) = client.reports
    assert released.result.status is RemoteCommandStatus.WAITING, released
    assert released.result.status_source == SHUTDOWN_RELEASE_STATUS_SOURCE, released
    assert released.result.details == record, "the command's own record is echoed"
    assert len(client.claims) == 1, "nothing is claimed after the stop"


def test_claim_failures_back_off_without_touching_the_command_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreachable claim route backs off as it always did, the command
    already running still reports, and the freed slot is claimed once the route
    answers again."""

    backoffs: list[float] = []

    def sleep(seconds: float) -> None:
        backoffs.append(seconds)
        REAL_SLEEP(0.001)

    monkeypatch.setattr("gpu_fault.cluster_executor.executor.time.sleep", sleep)
    client = LeasingControlPlane([remote_command("reboot")])
    adapter = GatedAdapter("reboot")
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=2,
        max_concurrent_commands=2,
        poll_seconds=0.05,
        claim_backoff_max_seconds=0.2,
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        with running(executor, adapter):
            eventually(
                lambda: adapter.started_ids() == ["reboot"], "reboot never started"
            )
            client.fail_next_claims(
                ClusterExecutorError("control plane unreachable", status_code=503), 4
            )
            eventually(lambda: len(backoffs) == 4, "the claims never backed off")
            during_failures = list(backoffs)
            adapter.release("reboot")
            eventually(
                lambda: client.reported_ids() == ["reboot"],
                "the command in flight must still report",
            )
            client.add(remote_command("next"))
            eventually(
                lambda: "next" in client.reported_ids(),
                "the freed slot was never claimed again",
            )

    assert during_failures == [0.05, 0.1, 0.2, 0.2], (
        "the claim-failure back-off is unchanged while a command runs: "
        f"{during_failures}"
    )
    failures = [
        record
        for record in caplog.records
        if "claim failed" in record.getMessage()
        or "still failing" in record.getMessage()
    ]
    assert [record.levelno for record in failures] == [
        logging.ERROR,
        logging.ERROR,
        logging.WARNING,
        logging.ERROR,
    ], [record.getMessage() for record in failures]
    assert all("claim failed" in record.getMessage() for record in failures), [
        record.getMessage() for record in failures
    ]
    assert client.reports[0].result.status is RemoteCommandStatus.SUCCEEDED


def test_the_network_back_off_claims_nothing_while_commands_are_in_flight(
    tmp_path: Path,
) -> None:
    """A replica whose network is degraded stops pulling work, as the batch
    loop did, and claims again once a command it holds makes progress."""

    client = LeasingControlPlane([remote_command("reboot")])
    adapter = GatedAdapter("reboot", unreachable=frozenset({"probe"}))
    # Two slots: with the probe running both are taken, so its WAITING release
    # cannot meet a claim before the loop has judged it.
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=2,
        max_concurrent_commands=2,
        transport_degraded_backoff_after=1,
    )

    with running(executor, adapter):
        eventually(lambda: adapter.started_ids() == ["reboot"], "reboot never started")
        client.add(remote_command("probe"))
        eventually(
            lambda: executor.consecutive_transport_degraded_cycles == 1,
            "the unreachable probe never counted as a degraded cycle",
        )
        claims = len(client.claims)
        REAL_SLEEP(0.3)
        assert len(client.claims) == claims, (
            "a network-degraded replica claimed with a command in flight"
        )
        assert [command.command_id for command in client.pending] == ["probe"], (
            "the released probe stays for a sibling"
        )
        adapter.release("reboot")
        eventually(
            lambda: client.reported_ids()[:2] == ["probe", "reboot"]
            and len(client.reported_ids()) >= 3,
            "progress must end the back-off and the probe be claimed again",
        )

    assert client.reports[0].result.status is RemoteCommandStatus.WAITING
    assert client.reports[0].result.status_source == "executor-retryable-transport"


def test_a_held_command_with_free_slots_still_polls_under_long_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free slots and a held claim on the control plane must not bring back the
    2026-09-04 spin: a command that only waits is claimed once per idle wait."""

    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise StopIteration

    monkeypatch.setattr("gpu_fault.cluster_executor.executor.time.sleep", sleep)
    client = LeasingControlPlane([remote_command("held")], hold_seconds=5)
    adapter = GatedAdapter(waiting=frozenset({"held"}))
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        poll_seconds=2,
        claim_wait_seconds=20,
        batch_size=5,
        max_concurrent_commands=5,
    )

    with pytest.raises(StopIteration):
        executor.run()

    assert sleeps == [2, 2, 2], "a held command takes the idle path every time"
    # Before the loop has seen the first WAITING, the claim the control plane
    # holds may take the command once more; after that, one run per idle wait.
    assert len(adapter.started_ids()) <= len(sleeps) + 3, adapter.started_ids()
    assert all(claim.wait_seconds == 20 for claim in client.claims), client.claims
    assert executor.last_cycle_advanced is False


def test_an_empty_long_poll_answered_at_once_is_paced_with_commands_in_flight(
    tmp_path: Path,
) -> None:
    """A control plane that will not hold the claim (its per-cluster waiter
    cap, no wakeup hub) answers every long-poll empty and at once; with free
    slots beside a running command the loop must not spin on it."""

    client = LeasingControlPlane([remote_command("long")], hold_seconds=0)
    adapter = GatedAdapter("long")
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=3,
        max_concurrent_commands=3,
        claim_wait_seconds=20,
        poll_seconds=0.1,
    )

    with running(executor, adapter):
        eventually(lambda: adapter.started_ids() == ["long"], "long never started")
        REAL_SLEEP(1.0)
        assert "long" not in client.reported_ids(), "long ran the whole window"

    assert all(claim.wait_seconds == 20 for claim in client.claims), client.claims
    empty = [claim for claim in client.claims if not claim.command_ids]
    gaps = [later.at - earlier.at for earlier, later in zip(empty, empty[1:])]
    assert len(empty) >= 3, f"the loop stopped claiming: {client.claims}"
    assert min(gaps) >= 0.09, (
        f"empty answers re-claimed sooner than poll_seconds (a spin): {gaps}"
    )
