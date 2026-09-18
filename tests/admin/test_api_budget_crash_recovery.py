"""Crash real helpers against the private, temporary deploy-host IPC ledger."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import api_budget as budget
from gpu_fault.admin.deadlines import deadline_scope
from gpu_fault.admin.execution import run_command

WINDOWS = (
    ("before-stopping-commit", "active", False),
    ("after-stopping-commit", "stopping", True),
    ("before-stop", "stopping", True),
    ("after-stop", "stopping", True),
    ("before-parked-commit", "stopping", True),
    ("after-parked-commit", "parked", True),
    ("helper-active", "parked", True),
    ("before-resuming-commit", "parked", True),
    ("after-resuming-commit", "resuming", True),
    ("before-cont", "resuming", True),
    ("after-cont", "resuming", True),
    ("before-active-commit", "resuming", True),
    ("after-active-commit", "active", False),
)
CANCELLATIONS = (
    ("before-killing-commit", "parked"),
    ("after-killing-commit", "killing"),
    ("before-kill", "killing"),
    ("after-kill", "killing"),
)


def process_stat(pid: int) -> tuple[str, str]:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return "", ""
    return fields[0], fields[19]


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_process_stat_observes_exit_during_procfs_read(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    def exited(_path: Path) -> str:
        raise error("owned test process exited")

    with monkeypatch.context() as observation:
        observation.setattr(Path, "read_text", exited)
        assert process_stat(123) == ("", ""), (
            "a confirmed process exit is not a probe failure"
        )


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_process_stat_does_not_treat_unknown_read_errors_as_exit(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    def unreadable(_path: Path) -> str:
        raise error("procfs is unreadable")

    with monkeypatch.context() as observation:
        observation.setattr(Path, "read_text", unreadable)
        with pytest.raises(error):
            process_stat(123)


def wait_until(predicate: Callable[[], bool], message: str) -> None:
    deadline = time.monotonic() + 8
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(0.01)


def record(directory: Path, name: str, **values: object) -> None:
    temporary = directory / f"{name}.pending"
    temporary.write_text(json.dumps(values))
    temporary.replace(directory / f"{name}.json")


def bind_command(identifier: str) -> None:
    root = budget.budget_root()
    assert root is not None, "fake CLI lost its test IPC scope"
    with sqlite3.connect(root / "budget.sqlite3") as database:
        database.execute(
            "UPDATE leases SET command_pid=?,command_start=? WHERE id=?",
            (os.getpid(), process_stat(os.getpid())[1], identifier),
        )


def crash_at(directory: Path, configured: str, reached: str) -> None:
    if configured != reached:
        return
    record(directory, "crash", window=reached, pid=os.getpid())
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("SIGKILL did not terminate the owned fake helper")


def instrument_helper(directory: Path, parent_id: str, window: str) -> None:
    connect = sqlite3.connect
    send_signal = signal.pidfd_send_signal
    resumed = False

    class Connection(sqlite3.Connection):
        def commit(self) -> None:
            row = self.execute(
                "SELECT state FROM leases WHERE id=?", (parent_id,)
            ).fetchone()
            state = str(row[0]) if row else ""
            checkpoint = state in {"stopping", "parked", "resuming", "killing"} or (
                state == "active" and resumed
            )
            if checkpoint:
                crash_at(directory, window, f"before-{state}-commit")
            super().commit()
            if state == "parked":
                record(directory, "parked", pid=os.getpid())
            if checkpoint:
                crash_at(directory, window, f"after-{state}-commit")

    def connection(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return connect(*args, **kwargs, factory=Connection)

    def signalled(descriptor: int, signum: int) -> None:
        nonlocal resumed
        action = {
            signal.SIGSTOP: "stop",
            signal.SIGCONT: "cont",
            signal.SIGKILL: "kill",
        }[signal.Signals(signum)]
        crash_at(directory, window, f"before-{action}")
        send_signal(descriptor, signum)
        resumed = resumed or signum == signal.SIGCONT
        crash_at(directory, window, f"after-{action}")

    patcher = pytest.MonkeyPatch()
    patcher.setattr(sqlite3, "connect", connection)
    patcher.setattr(signal, "pidfd_send_signal", signalled)


def fake_process(role: str, directory: Path, arguments: list[str]) -> None:
    if role == "helper":
        parent_id, window, raw_weight = arguments
        instrument_helper(directory, parent_id, window)
        scope = (
            deadline_scope("cancelled helper", 1)
            if window in dict(CANCELLATIONS)
            else contextlib.nullcontext()
        )
        with (
            scope,
            budget.api_slot(
                "aws", weight=int(raw_weight), parent_id=parent_id
            ) as identifier,
        ):
            record(directory, "helper", pid=os.getpid(), lease=identifier)
            if window == "live-command":
                command = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "command",
                        str(directory),
                        str(identifier),
                    ]
                )
                wait_until(
                    lambda: (directory / "command.json").exists(),
                    "fake helper's CLI did not bind",
                )
                crash_at(directory, window, "live-command")
                command.wait(timeout=8)
            crash_at(directory, window, "helper-active")
        raise AssertionError("configured helper crash window was not reached")
    if role == "command":
        bind_command(arguments[0])
        record(directory, "command", pid=os.getpid())
        wait_until(
            lambda: (directory / "release-command").exists(),
            "test did not release the orphaned fake CLI",
        )
        return
    if role == "lender":
        parent_id, window, weight = arguments
        bind_command(parent_id)

        def worker() -> None:
            while True:
                time.sleep(0.01)

        import threading

        threading.Thread(target=worker, daemon=True).start()
        record(directory, "lender", pid=os.getpid(), lease=parent_id)
        helper = subprocess.Popen(
            [sys.executable, __file__, "helper", str(directory), *arguments]
        )
        helper.wait(timeout=10)
        wait_until(
            lambda: (directory / "release").exists(),
            "test did not release the recovered lender",
        )
        return
    if role == "owner":
        window, parent_weight, child_weight = arguments
        with budget.api_slot("aws", weight=int(parent_weight)) as owner_id:
            lender = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "lender",
                    str(directory),
                    str(owner_id),
                    window,
                    child_weight,
                ]
            )
            code = lender.wait(timeout=12)
            if window in dict(CANCELLATIONS):
                wait_until(
                    lambda: (directory / "release").exists(),
                    "test did not release the cancelled lender's owner",
                )
                assert code == -signal.SIGKILL, "unfunded lender was not terminated"
            else:
                assert code == 0, "fake lender failed after recovery"
        return
    raise AssertionError("unknown fake process role")


@contextlib.contextmanager
def running_lender(
    directory: Path,
    window: str,
    parent_weight: int,
    child_weight: int,
    *,
    before_crash: Callable[[], None] | None = None,
) -> Iterator[Future[subprocess.CompletedProcess[str]]]:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_command,
            [
                sys.executable,
                __file__,
                "owner",
                str(directory),
                window,
                str(parent_weight),
                str(child_weight),
            ],
            timeout_seconds=15,
        )
        try:
            if before_crash is not None:
                before_crash()
            wait_until(
                lambda: (directory / "crash.json").exists() or future.done(),
                "fake helper did not reach its crash window",
            )
            if future.done():
                result = future.result()
                pytest.fail(f"fake owner exited before the crash: {result.stderr}")
            crashed_pid = int(json.loads((directory / "crash.json").read_text())["pid"])
            wait_until(
                lambda: process_stat(crashed_pid)[0] in {"", "Z", "X"},
                "SIGKILL did not finish the owned helper",
            )
            yield future
        finally:
            (directory / "release").touch()
            (directory / "release-command").touch()
            # Exercise public admission to finish any interrupted transition.
            with budget.api_slot("http"):
                pass
            result = future.result(timeout=20)
            assert result.returncode == 0, result.stderr


def observe_signals(monkeypatch: pytest.MonkeyPatch, pid: int) -> list[int]:
    descriptors: dict[int, int] = {}
    sent: list[int] = []

    def open_pidfd(target: int) -> int:
        descriptor = os.pidfd_open(target)
        descriptors[descriptor] = target
        return descriptor

    def send_signal(descriptor: int, signum: int) -> None:
        assert descriptors[descriptor] == pid, (
            "recovery tried to signal a PID outside the proven lender"
        )
        sent.append(signum)
        signal.pidfd_send_signal(descriptor, signum)

    monkeypatch.setattr(
        budget, "os", SimpleNamespace(**{**vars(os), "pidfd_open": open_pidfd})
    )
    monkeypatch.setattr(
        budget,
        "signal",
        SimpleNamespace(**{**vars(signal), "pidfd_send_signal": send_signal}),
    )
    return sent


@pytest.mark.parametrize("window,state,excluded", WINDOWS)
@pytest.mark.parametrize("parent_weight,child_weight", [(1, 4), (4, 1)])
def test_sigkill_windows_recover_without_uncharged_runnable_lenders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
    state: str,
    excluded: bool,
    parent_weight: int,
    child_weight: int,
) -> None:
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        with running_lender(tmp_path, window, parent_weight, child_weight):
            lender = json.loads((tmp_path / "lender.json").read_text())
            with sqlite3.connect(root / "budget.sqlite3") as database:
                actual, branch = database.execute(
                    "SELECT state,lent_to FROM leases WHERE id=?", (lender["lease"],)
                ).fetchone()
                charged = database.execute(
                    "SELECT COALESCE(SUM(weight),0) FROM leases "
                    "WHERE state IN ('active','stopping','resuming')"
                ).fetchone()[0]
            assert actual == state, "crash did not leave the expected durable state"
            assert bool(branch) == excluded, "crash lost its sibling exclusion"
            if state == "parked":
                wait_until(
                    lambda: process_stat(lender["pid"])[0] == "T",
                    "uncharged lender was not stopped",
                )
                tasks = list(Path(f"/proc/{lender['pid']}/task").iterdir())
                assert len(tasks) >= 2, "test did not exercise a multithreaded CLI"
                assert all(
                    (task / "stat").read_text().rsplit(")", 1)[1].split()[0] == "T"
                    for task in tasks
                ), "parked lender retained a runnable thread"
                expected = (
                    max(parent_weight, child_weight)
                    if window in {"helper-active", "before-resuming-commit"}
                    else 0
                )
            else:
                expected = parent_weight
            assert charged == expected, "durable ledger lost its weight reservation"
            with monkeypatch.context() as recovery:
                sent = observe_signals(recovery, lender["pid"])
                with budget.api_slot("http"):
                    with sqlite3.connect(root / "budget.sqlite3") as database:
                        restored = database.execute(
                            "SELECT state,lent_to,weight FROM leases WHERE id=?",
                            (lender["lease"],),
                        ).fetchone()
                        aws_weight = database.execute(
                            "SELECT COALESCE(SUM(weight),0) FROM leases "
                            "WHERE backend='aws' AND state IN "
                            "('active','stopping','resuming')"
                        ).fetchone()[0]
                assert restored == ("active", None, parent_weight), (
                    "dead helper recovery did not restore the lender's reservation"
                )
                assert aws_weight == parent_weight, "recovery double-charged a loan"
                assert all(signum == signal.SIGCONT for signum in sent), (
                    "funded lender recovery sent an unexpected signal"
                )
                assert bool(sent) == excluded, "recovery replayed a completed handoff"
            wait_until(
                lambda: process_stat(lender["pid"])[0] not in {"T", "t"},
                "funded lender remained stopped after recovery",
            )
        with sqlite3.connect(root / "budget.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0, (
                "crash test left a lease behind"
            )


def test_dead_helper_keeps_live_cli_charged_until_actual_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        with running_lender(tmp_path, "live-command", 4, 1):
            lender = json.loads((tmp_path / "lender.json").read_text())
            command = json.loads((tmp_path / "command.json").read_text())
            with monkeypatch.context() as recovery:
                sent = observe_signals(recovery, lender["pid"])
                with budget.api_slot("http"):
                    with sqlite3.connect(root / "budget.sqlite3") as database:
                        assert database.execute(
                            "SELECT state FROM leases WHERE id=?", (lender["lease"],)
                        ).fetchone() == ("parked",), (
                            "dead shim resumed a parent over its live CLI"
                        )
                        assert database.execute(
                            "SELECT SUM(weight) FROM leases WHERE state='active' "
                            "AND backend='aws'"
                        ).fetchone() == (4,), "live orphaned CLI lost its charge"
                assert sent == [], "live orphaned CLI permitted a parent signal"
                (tmp_path / "release-command").touch()
                wait_until(
                    lambda: process_stat(command["pid"])[0] in {"", "Z", "X"},
                    "owned orphan CLI did not exit",
                )
                with budget.api_slot("http"):
                    pass
                assert sent == [signal.SIGCONT], (
                    "actual CLI exit did not recover the dead helper's loan"
                )


@pytest.mark.parametrize("window,state", CANCELLATIONS)
def test_crashed_cancellation_never_resumes_an_unfunded_lender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str, state: str
) -> None:
    with budget.deployment_api_budget(), contextlib.ExitStack() as holders:
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        holders.enter_context(budget.api_slot("aws", weight=7))

        def consume_capacity() -> None:
            wait_until(
                lambda: (tmp_path / "parked.json").exists(),
                "heavier helper did not park its lender",
            )
            holders.enter_context(budget.api_slot("aws"))

        with running_lender(tmp_path, window, 1, 4, before_crash=consume_capacity):
            lender = json.loads((tmp_path / "lender.json").read_text())
            with sqlite3.connect(root / "budget.sqlite3") as database:
                assert database.execute(
                    "SELECT state FROM leases WHERE id=?", (lender["lease"],)
                ).fetchone() == (state,), "cancellation crash lost its durable state"
                assert database.execute(
                    "SELECT SUM(weight) FROM leases WHERE backend='aws' "
                    "AND state IN ('active','stopping','resuming')"
                ).fetchone() == (8,), "cancellation crash lost occupied capacity"
            with monkeypatch.context() as recovery:
                sent = observe_signals(recovery, lender["pid"])
                with budget.api_slot("http"):
                    pass
                wait_until(
                    lambda: process_stat(lender["pid"])[0] in {"", "Z", "X"},
                    "unfunded stopped lender was not terminated",
                )
                assert all(signum == signal.SIGKILL for signum in sent), (
                    "cancellation resumed an uncharged lender"
                )
                if window != "after-kill":
                    assert sent, "interrupted cancellation was not recovered"


def test_recovery_identity_drift_keeps_charge_and_never_signals_unrelated_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "test budget was not created"
        with running_lender(tmp_path, "before-cont", 4, 1):
            lender = json.loads((tmp_path / "lender.json").read_text())
            with subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(15)"]
            ) as unrelated:
                with sqlite3.connect(root / "budget.sqlite3") as database:
                    original = database.execute(
                        "SELECT command_pid,command_start FROM leases WHERE id=?",
                        (lender["lease"],),
                    ).fetchone()
                    database.execute(
                        "UPDATE leases SET command_pid=?,command_start=? WHERE id=?",
                        (unrelated.pid, "mismatched-start", lender["lease"]),
                    )
                try:
                    with monkeypatch.context() as recovery:
                        sent = observe_signals(recovery, lender["pid"])
                        with budget.api_slot("http"):
                            with sqlite3.connect(root / "budget.sqlite3") as database:
                                assert database.execute(
                                    "SELECT state,weight FROM leases WHERE id=?",
                                    (lender["lease"],),
                                ).fetchone() == ("resuming", 4), (
                                    "identity drift discarded a live owner's reservation"
                                )
                        assert sent == [], "identity drift authorized a PID signal"
                        assert unrelated.poll() is None, (
                            "loan recovery affected an unrelated test process"
                        )
                finally:
                    with sqlite3.connect(root / "budget.sqlite3") as database:
                        database.execute(
                            "UPDATE leases SET command_pid=?,command_start=? WHERE id=?",
                            (*original, lender["lease"]),
                        )
                    unrelated.kill()
                    unrelated.wait(timeout=3)


if __name__ == "__main__":
    fake_process(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
