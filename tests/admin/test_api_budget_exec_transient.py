"""A zombie thread-group leader is not an exited CLI.

Live on 2026-09-19 (deploy #18) a shim's lease reaper read a sibling's freshly
launched ``aws`` as state Z: the snap launcher is a Go program that execs from a
non-leader thread, and the kernel reports the dying leader as a zombie under the
process's own pid and start time until the exec'ing thread takes the pid over.
The reaper then found the command alive again and failed its own, innocent
command with ``cannot release a running API command``. These tests drive the
reaper through ``api_slot`` and watch the ledger: a live command's reservation
must survive, a real zombie's must go, and a sibling never fails on either.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import api_budget as budget

START = "4242"
FAKE_PID = 4321


def process_start(pid: int) -> str:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[19] if fields[0] not in {"Z", "X"} else ""


class FakeProc:
    """A scripted procfs for one pid: ``stat`` reads come from a sequence."""

    def __init__(self, reads: list[Any], tasks: dict[int, str]) -> None:
        self.reads = list(reads)
        self.tasks = dict(tasks)
        self.stat_reads = 0
        self.waits: list[float] = []

    @staticmethod
    def stat_line(state: str, start: str = START) -> str:
        fields = [state, "1", *(["0"] * 17), start]
        return f"{FAKE_PID} (aws) " + " ".join(fields)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = self
        real_read, real_iterdir = Path.read_text, Path.iterdir

        def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
            parts = path.parts
            if len(parts) >= 3 and parts[1] == "proc" and parts[2] == str(FAKE_PID):
                if len(parts) == 4 and parts[3] == "stat":
                    proc.stat_reads += 1
                    item = proc.reads.pop(0) if len(proc.reads) > 1 else proc.reads[0]
                    if isinstance(item, BaseException):
                        raise item
                    return str(item)
                if len(parts) == 6 and parts[3] == "task" and parts[5] == "stat":
                    state = proc.tasks.get(int(parts[4]))
                    if state is None:
                        raise FileNotFoundError(path)
                    return proc.stat_line(state)
            return real_read(path, *args, **kwargs)

        def iterdir(path: Path):
            parts = path.parts
            if len(parts) == 4 and parts[1] == "proc" and parts[2] == str(FAKE_PID):
                return iter([path / str(tid) for tid in proc.tasks])
            return real_iterdir(path)

        monkeypatch.setattr(Path, "read_text", read_text)
        monkeypatch.setattr(Path, "iterdir", iterdir)
        # The confirmation pacing is recorded instead of slept.
        monkeypatch.setattr(budget, "_confirmation_wait", proc.waits.append)


def seed_lease(root: Path, identifier: str, command_pid: int, start: str) -> None:
    with sqlite3.connect(root / "budget.sqlite3") as database:
        database.execute(
            "INSERT INTO leases(id,backend,weight,pid,pid_start,command_pid,"
            "command_start,state) VALUES(?,'aws',1,?,?,?,?,'active')",
            (identifier, os.getpid(), process_start(os.getpid()), command_pid, start),
        )
        database.execute(
            "INSERT INTO calls(id,phase,backend,wait_seconds) VALUES(?,'fixture','aws',0)",
            (identifier,),
        )


def ledger(root: Path, identifier: str) -> tuple[bool, int]:
    with sqlite3.connect(root / "budget.sqlite3") as database:
        held = database.execute(
            "SELECT 1 FROM leases WHERE id=?", (identifier,)
        ).fetchone()
        finished = database.execute(
            "SELECT finished FROM calls WHERE id=?", (identifier,)
        ).fetchone()[0]
    return held is not None, int(finished)


def reap_once(monkeypatch: pytest.MonkeyPatch, proc: FakeProc) -> tuple[bool, int]:
    """One sibling admission: the reaper probes the seeded lease and returns the ledger."""
    identifier = "b" * 32
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "the regression requires an owned temporary ledger"
        seed_lease(root, identifier, FAKE_PID, START)
        proc.install(monkeypatch)
        with budget.api_slot("http"):
            pass
        return ledger(root, identifier)


def test_zombie_leader_beside_a_live_task_keeps_the_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc([FakeProc.stat_line("Z")], {FAKE_PID: "Z", 4330: "R"})
    held, finished = reap_once(monkeypatch, proc)
    assert held and finished == 0, (
        "a zombie leader with a running sibling task was reaped as an exit"
    )
    assert proc.waits == [], "a live task settled the question without waiting"


def test_zombie_that_turns_runnable_again_keeps_the_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc(
        [FakeProc.stat_line("Z"), FakeProc.stat_line("Z"), FakeProc.stat_line("R")],
        {FAKE_PID: "Z"},
    )
    held, finished = reap_once(monkeypatch, proc)
    assert held and finished == 0, (
        "the exec'ing thread took the pid back and the reaper still released it"
    )
    assert proc.waits == list(budget.EXIT_CONFIRMATION_DELAYS[:2]), (
        "confirmation did not pace its re-reads through the shared delays"
    )


def test_persistent_zombie_is_reaped_after_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc([FakeProc.stat_line("Z")], {FAKE_PID: "Z"})
    held, finished = reap_once(monkeypatch, proc)
    assert not held and finished == 1, (
        "a zombie that outlasted every confirmation delay is a real zombie"
    )
    # The reaper confirms once, and the release re-probes once more.
    assert proc.waits[: len(budget.EXIT_CONFIRMATION_DELAYS)] == list(
        budget.EXIT_CONFIRMATION_DELAYS
    ), "a persistent zombie must be confirmed across the whole delay schedule"
    assert proc.stat_reads >= 1 + len(budget.EXIT_CONFIRMATION_DELAYS), (
        "each confirmation delay must be followed by a fresh procfs read"
    )


def test_zombie_that_disappears_is_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FakeProc(
        [FakeProc.stat_line("Z"), FileNotFoundError("reaped")], {FAKE_PID: "Z"}
    )
    held, finished = reap_once(monkeypatch, proc)
    assert not held and finished == 1, "a zombie reaped during confirmation is an exit"
    assert len(proc.waits) == 1, "disappearance ends the confirmation early"


@pytest.mark.parametrize("state", ["S", "R", "T", "D"])
def test_a_running_or_stopped_command_is_never_probed_twice(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    proc = FakeProc([FakeProc.stat_line(state)], {FAKE_PID: state})
    held, finished = reap_once(monkeypatch, proc)
    assert held and finished == 0, f"a {state} command was reaped"
    assert proc.waits == [], "a decisive first read must not wait"


def test_unknown_reads_during_confirmation_keep_the_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc(
        [FakeProc.stat_line("Z"), PermissionError("procfs unreadable")], {FAKE_PID: "Z"}
    )
    held, finished = reap_once(monkeypatch, proc)
    assert held and finished == 0, "an unreadable re-read was taken as proof of exit"


def test_reaper_keeps_a_sibling_reservation_whose_command_is_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every confirmation read stays Z, then the release re-probe finds it running."""
    reads = [FakeProc.stat_line("Z")] * (1 + len(budget.EXIT_CONFIRMATION_DELAYS))
    reads.append(FakeProc.stat_line("R"))
    proc = FakeProc(reads, {FAKE_PID: "Z"})
    held, finished = reap_once(monkeypatch, proc)
    assert held and finished == 0, (
        "a reaper released a sibling's reservation while its command ran, or "
        "failed its own admission over it"
    )


def exec_from_a_worker_thread() -> subprocess.Popen[bytes]:
    """The main thread (leader) idles while a worker thread execs the process."""
    program = (
        "import os, threading\n"
        "threading.Thread(target=lambda: os.execv('/bin/sh', "
        "['sh', '-c', 'echo started; exec sleep 0.2'])).start()\n"
        "threading.Event().wait()\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", program], stdout=subprocess.PIPE, start_new_session=True
    )


def test_a_live_process_execing_from_a_worker_thread_never_reads_as_exited() -> None:
    """Real kernel behaviour: the leader dies first, the process lives on.

    Three probes race each exec in a tight loop. The process is alive until at
    least 0.2 s after it prints ``started``; an exit verdict before then is the
    live defect (the pre-fix probe returned one for roughly one launch in six).
    """
    premature: list[float] = []
    launches = 24
    for _ in range(launches):
        process = exec_from_a_worker_thread()
        start = process_start(process.pid)
        assert start, "the fixture process must have an observable start time"
        stop = threading.Event()
        verdicts: list[float] = []

        def probe(pid: int = process.pid, begin: str = start) -> None:
            while not stop.is_set():
                if budget._identity_gone(pid, begin, allow_reuse=False):
                    verdicts.append(time.monotonic())
                    return

        threads = [threading.Thread(target=probe) for _ in range(3)]
        for thread in threads:
            thread.start()
        assert process.stdout is not None
        process.stdout.readline()
        started_at = time.monotonic()
        process.wait(timeout=30)
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
        premature.extend(at for at in verdicts if at < started_at + 0.15)
        assert budget._identity_gone(process.pid, start) is True, (
            "a reaped process must read as exited"
        )
    assert premature == [], (
        f"{len(premature)} probe(s) declared a live exec'ing process exited"
    )
