"""HA-007: budgets from the generated manifest, an over-budget request, no orphan child."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

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
