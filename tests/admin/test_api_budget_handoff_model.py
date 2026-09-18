from __future__ import annotations

import os
import signal
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import api_budget as budget

PARENT = "a" * 32


class ProcessPath:
    def __init__(self, model, path):
        self.model = model
        self.path = Path(path)
        self.name = self.path.name

    def __truediv__(self, part):
        return ProcessPath(self.model, self.path / part)

    def stat(self):
        return SimpleNamespace()

    def iterdir(self):
        return iter(
            ProcessPath(self.model, self.path / name) for name in ("101", "105")
        )

    def read_text(self):
        pid = int(self.path.parts[2])
        parent, start = {
            1: (0, "1"),
            100: (1, "10"),
            101: (100, "11"),
            102: (101, "12"),
            999: (1, "99"),
        }[pid]
        state = self.model.state if pid == 101 else "S"
        if self.path.parent.name == "105" and self.model.worker_state is not None:
            state = self.model.worker_state
        fields = [state, str(parent), "0", "0", *(["0"] * 15), start]
        return f"{pid} (model CLI) " + " ".join(fields)


class ProcessModel:
    def __init__(self, database=None):
        self.database = database
        self.state = "S"
        self.worker_state = None
        self.signals = []
        self.closed = []
        self.os = SimpleNamespace(**vars(os))
        self.os.getpid = lambda: 102
        self.os.pidfd_open = self.open_pidfd
        self.os.close = self.close
        self.signal = SimpleNamespace(**vars(signal))
        self.signal.pidfd_send_signal = self.send_signal

    def path(self, value):
        text = str(value)
        return ProcessPath(self, text) if text.startswith("/proc/") else Path(value)

    def open_pidfd(self, pid):
        assert pid == 101, "handoff attempted to signal a process outside its lender"
        return 10001

    def close(self, descriptor):
        if descriptor == 10001:
            self.closed.append(descriptor)
        else:
            os.close(descriptor)

    def send_signal(self, descriptor, signum):
        assert descriptor == 10001, "handoff used an unverified process descriptor"
        if self.database is not None:
            with sqlite3.connect(self.database) as database:
                state, branch, weight = database.execute(
                    "SELECT state,lent_to,weight FROM leases WHERE id=?", (PARENT,)
                ).fetchone()
                assert state == (
                    "stopping" if signum == signal.SIGSTOP else "resuming"
                ), "signal preceded its durable handoff intent"
                assert branch, "signal lost durable sibling exclusion"
                assert (
                    database.execute(
                        "SELECT SUM(weight) FROM leases WHERE backend='aws' "
                        "AND state IN ('active','stopping','resuming')"
                    ).fetchone()[0]
                    >= weight
                ), "signal made a lender runnable without capacity"
        self.signals.append(signum)
        self.state = "T" if signum == signal.SIGSTOP else "S"


def seed_lender(root, weight):
    with sqlite3.connect(root / "budget.sqlite3") as database:
        database.execute(
            "INSERT INTO leases(id,backend,weight,pid,pid_start,command_pid,"
            "command_start,depth,state) VALUES(?,'aws',?,100,'10',101,'11',0,'active')",
            (PARENT, weight),
        )


@pytest.mark.parametrize("parent_weight,child_weight", [(1, 1), (1, 4), (4, 1), (4, 4)])
@pytest.mark.parametrize("failed", [False, True])
def test_handoff_stops_lender_and_returns_its_reserved_capacity(
    monkeypatch, parent_weight, child_weight, failed
):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        seed_lender(root, parent_weight)
        model = ProcessModel(root / "budget.sqlite3")
        with monkeypatch.context() as context:
            context.setattr(budget, "os", model.os)
            context.setattr(budget, "signal", model.signal)
            context.setattr(budget, "Path", model.path)

            def borrow():
                with budget.api_slot(
                    "aws", weight=child_weight, parent_id=PARENT
                ) as child:
                    assert model.state == "T"
                    with sqlite3.connect(root / "budget.sqlite3") as database:
                        assert database.execute(
                            "SELECT state,lent_to FROM leases WHERE id=?", (PARENT,)
                        ).fetchone() == ("parked", child)
                        assert database.execute(
                            "SELECT weight,state,parent_id FROM leases WHERE id=?",
                            (child,),
                        ).fetchone() == (
                            max(parent_weight, child_weight),
                            "active",
                            PARENT,
                        )
                        assert database.execute(
                            "SELECT SUM(weight) FROM leases WHERE state='active'"
                        ).fetchone()[0] == max(parent_weight, child_weight)
                    if failed:
                        raise RuntimeError("modeled helper failure")

            if failed:
                with pytest.raises(RuntimeError, match="modeled helper"):
                    borrow()
            else:
                borrow()
            assert model.signals == [signal.SIGSTOP, signal.SIGCONT]
            assert model.state == "S"
            assert model.closed == [10001, 10001], (
                "STOP and durable resume must each close their verified pidfd"
            )
            with sqlite3.connect(root / "budget.sqlite3") as database:
                assert database.execute(
                    "SELECT id,state,lent_to,weight FROM leases"
                ).fetchall() == [(PARENT, "active", None, parent_weight)]
            assert budget.statistics()["peak_admitted_weight"]["aws"] <= 8


@pytest.mark.parametrize("invalid", ["unrelated", "stale", "depth", "stopped"])
def test_invalid_lender_never_surrenders_capacity(monkeypatch, invalid):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        seed_lender(root, 4)
        model = ProcessModel()
        with sqlite3.connect(root / "budget.sqlite3") as database:
            if invalid == "unrelated":
                database.execute("UPDATE leases SET pid=999,pid_start='99'")
            elif invalid == "stale":
                database.execute("UPDATE leases SET command_start='different'")
            elif invalid == "depth":
                database.execute("UPDATE leases SET depth=7")
            else:
                model.state = "T"
        with monkeypatch.context() as context:
            context.setattr(budget, "os", model.os)
            context.setattr(budget, "signal", model.signal)
            context.setattr(budget, "Path", model.path)
            with pytest.raises(budget.ApiBudgetError):
                with budget.api_slot("aws", parent_id=PARENT):
                    pytest.fail("invalid lender authorized a helper")
            assert model.signals == []
            with sqlite3.connect(root / "budget.sqlite3") as database:
                assert database.execute(
                    "SELECT id,state,weight,lent_to FROM leases"
                ).fetchall() == [(PARENT, "active", 4, None)]


def test_one_stopped_thread_does_not_authorize_an_api_loan(monkeypatch):
    from gpu_fault.admin.execution import deadline_scope

    with budget.deployment_api_budget():
        root = budget.budget_root()
        seed_lender(root, 4)
        model = ProcessModel(root / "budget.sqlite3")
        model.worker_state = "S"
        with monkeypatch.context() as context:
            context.setattr(budget, "os", model.os)
            context.setattr(budget, "signal", model.signal)
            context.setattr(budget, "Path", model.path)
            # The proof waits min(2 s, remaining deadline) for every thread to
            # acknowledge STOP, so the deadline is what keeps this test short.
            # 50 ms was too short to be deterministic: under release-gate load
            # (16 workers + Postgres shards) the deadline expired before the
            # slot's first remaining-time check and the case failed with
            # "deadline expired" instead of "did not stop" (deploy #6).
            with deadline_scope("thread stop proof", 1.0):
                with pytest.raises(budget.ApiBudgetError, match="did not stop"):
                    with budget.api_slot("aws", parent_id=PARENT):
                        pytest.fail("leader-only STOP proof authorized an API helper")
            assert model.signals == [signal.SIGSTOP, signal.SIGCONT], (
                "failed all-thread proof did not restore its charged lender"
            )
            assert budget.statistics()["backends"] == {}, (
                "helper was admitted before all threads acknowledged STOP"
            )
            with sqlite3.connect(root / "budget.sqlite3") as database:
                assert database.execute(
                    "SELECT id,state,weight,lent_to FROM leases"
                ).fetchall() == [(PARENT, "active", 4, None)], (
                    "failed STOP proof left a loan or lost capacity"
                )


@pytest.mark.parametrize("after_signal", [False, True])
def test_interrupted_resume_is_retried_with_capacity_and_exclusion(
    monkeypatch, after_signal
):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        seed_lender(root, 4)
        model = ProcessModel(root / "budget.sqlite3")
        failed = False

        def interrupted(descriptor, signum):
            nonlocal failed
            if signum == signal.SIGCONT and not failed:
                failed = True
                if after_signal:
                    model.send_signal(descriptor, signum)
                raise InterruptedError("modeled signal interruption")
            model.send_signal(descriptor, signum)

        model.signal.pidfd_send_signal = interrupted
        with monkeypatch.context() as context:
            context.setattr(budget, "os", model.os)
            context.setattr(budget, "signal", model.signal)
            context.setattr(budget, "Path", model.path)
            with budget.api_slot("aws", parent_id=PARENT):
                pass
            assert model.signals == [signal.SIGSTOP, signal.SIGCONT] + (
                [signal.SIGCONT] if after_signal else []
            ), "resume retry lost its signal or replayed STOP"
            assert model.closed == [10001, 10001, 10001], (
                "resume interruption leaked a verified descriptor"
            )
            with sqlite3.connect(root / "budget.sqlite3") as database:
                assert database.execute(
                    "SELECT id,state,weight,lent_to FROM leases"
                ).fetchall() == [(PARENT, "active", 4, None)], (
                    "interrupted resume did not finish its durable transition"
                )
