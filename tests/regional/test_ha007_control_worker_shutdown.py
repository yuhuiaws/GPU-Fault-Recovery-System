"""HA-007: budgets from the generated manifest, an over-budget request, no orphan child."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.lifecycle import ShutdownCoordinator
from scripts.e2e.regional import run_ha007_control_worker_shutdown as ha007

ROOT = Path(__file__).resolve().parents[2]


def test_budgets_are_read_from_the_generated_control_worker_manifest() -> None:
    budgets = ha007.control_worker_budgets()
    assert budgets["lifespan_budget_seconds"] == 130.0
    assert budgets["kubernetes_grace_seconds"] == 240.0
    sources = budgets["sources"]
    assert sources["lifespan_budget_seconds"].endswith(
        "gpu-fault-control-worker-config-core.yaml"
    ), sources
    assert sources["kubernetes_grace_seconds"].endswith(
        "gpu-fault-control-worker.yaml"
    ), sources
    assert not hasattr(ha007, "LIFESPAN_BUDGET_SECONDS"), (
        "the lifespan budget must come from the manifest, not a module constant"
    )
    assert not hasattr(ha007, "KUBERNETES_GRACE_SECONDS"), (
        "the grace period must come from the manifest, not a module constant"
    )


def test_budgets_fail_closed_when_the_manifest_lacks_the_values(tmp_path: Path) -> None:
    generated = tmp_path / "deploy/control-plane/regional/generated"
    generated.mkdir(parents=True)
    (generated / "gpu-fault-control-worker.yaml").write_text(
        "kind: Deployment\nspec:\n  template:\n    spec: {}\n"
    )
    (generated / "gpu-fault-control-worker-config-core.yaml").write_text(
        "kind: ConfigMap\ndata: {}\n"
    )
    with pytest.raises(RuntimeError, match="terminationGracePeriodSeconds"):
        ha007.control_worker_budgets(tmp_path)


def test_default_durations_include_one_request_longer_than_the_budget() -> None:
    budget = ha007.control_worker_budgets()["lifespan_budget_seconds"]
    outcomes = [ha007.expected_outcome(d, budget) for d in ha007.DEFAULT_DURATIONS]
    assert "deadline-exceeded" in outcomes
    assert "completed" in outcomes


def test_probe_reports_verdict_and_exercises_the_deadline_path(tmp_path: Path) -> None:
    report = ha007.run_probe(
        tmp_path,
        [0.01, 0.6],
        budgets={"lifespan_budget_seconds": 0.2, "kubernetes_grace_seconds": 10},
    )

    assert "status" not in report
    assert report["verdict"] == "PASS", report
    assert [item["expected_outcome"] for item in report["runs"]] == [
        "completed",
        "deadline-exceeded",
    ]
    assert report["runs"][0]["completed"] == 1
    over = report["runs"][1]
    assert over["shutdown_failures"] == [ha007.WORKER_THREAD_NAME]
    assert over["completed"] == 0
    assert over["returncode"] != 0
    assert over["shutdown_seconds"] >= 0.2
    assert report["lifespan_budget_seconds"] == 0.2
    document = json.loads((tmp_path / f"{ha007.CASE_ID}.json").read_text())
    assert document["verdict"] == "PASS"


def test_evaluate_run_flags_the_wrong_behaviour_for_each_outcome() -> None:
    in_budget = {
        "duration_seconds": 1.0,
        "elapsed_seconds": 1.1,
        "returncode": 0,
        "completed": 1,
        "shutdown_failures": [],
        "shutdown_seconds": 1.0,
    }
    assert (
        ha007.evaluate_run(
            in_budget, lifespan_budget_seconds=5, kubernetes_grace_seconds=10
        )
        == []
    )
    assert ha007.evaluate_run(
        {**in_budget, "completed": 0},
        lifespan_budget_seconds=5,
        kubernetes_grace_seconds=10,
    ) == ["in-budget request did not complete before exit"]
    assert ha007.evaluate_run(
        {**in_budget, "elapsed_seconds": 11},
        lifespan_budget_seconds=5,
        kubernetes_grace_seconds=10,
    ) == ["Pod exit exceeded terminationGracePeriodSeconds"]

    over = {
        "duration_seconds": 8.0,
        "elapsed_seconds": 5.1,
        "returncode": 1,
        "completed": 0,
        "shutdown_failures": [ha007.WORKER_THREAD_NAME],
        "shutdown_seconds": 5.0,
    }
    assert (
        ha007.evaluate_run(over, lifespan_budget_seconds=5, kubernetes_grace_seconds=10)
        == []
    )
    swallowed = ha007.evaluate_run(
        {**over, "returncode": 0, "shutdown_failures": []},
        lifespan_budget_seconds=5,
        kubernetes_grace_seconds=10,
    )
    assert (
        "over-budget request was not reported as the coordinator's failure" in swallowed
    )
    assert "over-budget request exited zero; lifespan did not raise" in swallowed


def test_child_exits_on_its_own_when_no_signal_arrives(tmp_path: Path) -> None:
    started = tmp_path / "started.txt"
    result = tmp_path / "result.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/e2e/regional/run_ha007_control_worker_shutdown.py"),
            "--child",
            "--duration",
            "0.01",
            "--started-file",
            str(started),
            "--result-file",
            str(result),
            "--lifespan-budget-seconds",
            "1",
            "--signal-timeout-seconds",
            "0.3",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        returncode = process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGKILL)
        pytest.fail("child without SIGTERM stayed alive: an orphan")
    assert returncode == 3, process.stdout.read() if process.stdout else returncode
    assert json.loads(result.read_text())["signal_timed_out"] is True


def test_child_leaves_no_thread_waiting_past_grace(tmp_path: Path) -> None:
    started = time.monotonic()
    report = ha007.run_probe(
        tmp_path,
        [0.5],
        budgets={"lifespan_budget_seconds": 0.1, "kubernetes_grace_seconds": 3},
    )
    assert report["verdict"] == "PASS", report
    assert time.monotonic() - started < 3, (
        "the over-budget child must exit at the budget, not at the request end"
    )


def test_event_wait_can_return_false_after_the_signal_flag_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeliveredAtTimeout(threading.Condition):
        def wait(self, timeout: float | None = None) -> bool:
            # Model handler dispatch after a timed C wait, before lock reacquisition.
            self.release()
            try:
                event.set()
            finally:
                self.acquire()
            return False

    with monkeypatch.context() as patch:
        patch.setattr(threading, "Condition", DeliveredAtTimeout)
        event = threading.Event()
    assert event.wait(0.01) is False, (
        "Event.wait returns the Condition timeout result without rereading the flag"
    )
    assert event.is_set() is True, (
        "a false timed-wait result does not prove that the handler never ran"
    )


@pytest.mark.parametrize("delivery", ["wait-boundary", "dispatch-point", "absent"])
def test_child_signal_dispatch_preserves_the_deadline_without_condition_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, delivery: str
) -> None:
    now = 0.0
    signal_timeout = 0.055
    handlers = []
    sleeps: list[float] = []
    waits: list[float | None] = []
    joins: list[float] = []
    budgets: list[float] = []

    class DeliveredAtTimeout(threading.Condition):
        def wait(self, timeout: float | None = None) -> bool:
            nonlocal now
            waits.append(timeout)
            now += float(timeout or 0)
            if delivery != "absent":
                self.release()
                try:
                    handlers[0](signal.SIGTERM, None)
                finally:
                    self.acquire()
            return False

    with monkeypatch.context() as patch:
        patch.setattr(threading, "Condition", DeliveredAtTimeout)
        event = threading.Event()

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds
        if delivery == "dispatch-point" or (
            delivery == "wait-boundary" and now >= signal_timeout
        ):
            handlers[0](signal.SIGTERM, None)

    class OverBudgetThread:
        def __init__(self, *, target, name: str, daemon: bool) -> None:
            assert callable(target) and daemon is True, (
                "the child must retain a real request target and daemon semantics"
            )
            self.name = name
            self.started = False

        def start(self) -> None:
            self.started = True

        def join(self, *, timeout: float) -> None:
            nonlocal now
            assert self.started, "shutdown must join the already started request"
            joins.append(timeout)
            now += timeout

        def is_alive(self) -> bool:
            return self.started

    def coordinator(budget: float) -> ShutdownCoordinator:
        budgets.append(budget)
        return ShutdownCoordinator(budget, now=lambda: now)

    monkeypatch.setattr(ha007, "Event", lambda: event)
    monkeypatch.setattr(ha007, "Thread", OverBudgetThread)
    monkeypatch.setattr(
        ha007,
        "signal",
        SimpleNamespace(
            SIGTERM=signal.SIGTERM, signal=lambda _sig, fn: handlers.append(fn)
        ),
    )
    monkeypatch.setattr(
        ha007, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep)
    )
    monkeypatch.setattr(ha007, "ShutdownCoordinator", coordinator)
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ha007-signal-boundary",
            "--child",
            "--duration",
            "2",
            "--started-file",
            str(tmp_path / "started"),
            "--result-file",
            str(result_path),
            "--lifespan-budget-seconds",
            "1",
            "--signal-timeout-seconds",
            str(signal_timeout),
        ],
    )
    assert ha007.main() == (3 if delivery == "absent" else 1), (
        "an observed signal must reach the coordinator, not the orphan timeout exit"
    )
    result = json.loads(result_path.read_text())
    assert result["signal_timed_out"] is (delivery == "absent"), (
        "only absence of the signal flag may claim a signal timeout"
    )
    assert waits == [], (
        "the signal handler must not reenter an Event condition held by the main thread"
    )
    assert sleeps and all(0 < seconds <= 0.02 for seconds in sleeps), (
        "short sleeps must provide bounded Python signal dispatch points"
    )
    assert sum(sleeps) <= signal_timeout, "polling must not extend the signal deadline"
    if delivery == "absent":
        assert now == signal_timeout and not joins and not budgets, (
            "an orphan must exit at its original deadline without starting shutdown"
        )
    else:
        assert budgets == [1.0], (
            "the coordinator must retain the original one-second budget"
        )
        assert joins == pytest.approx([1.0], rel=0, abs=1e-15), (
            "the remaining join time may differ only by floating-point subtraction roundoff"
        )
        assert result["shutdown_failures"] == [ha007.WORKER_THREAD_NAME], (
            "an over-budget request must remain the coordinator's explicit failure"
        )
        assert result["completed"] == 0, (
            "the pending request must not be marked complete"
        )
