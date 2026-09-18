"""Offline COLLECT-016 drainage, report completeness and deadline regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from gpu_fault.admin import deadlines
from gpu_fault.models import Environment
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from scripts.e2e.regional import collector_action_guard as guard
from scripts.e2e.regional import run_collect016_training_recovery as runner

JOB = "c016-owned-a"
ATTEMPTS = {"a-original", "a-restarted"}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    for module in (runner, guard, deadlines):
        monkeypatch.setattr(module, "time", value)
    return value


def observation(
    attempt: str = "a-original", *, phase: WorkloadPhase = WorkloadPhase.STOPPED
) -> dict[str, Any]:
    return AttemptObservation(
        cluster_id="cluster-a",
        environment=Environment.HYPERPOD_EKS,
        job_id=JOB,
        attempt_id=attempt,
        workload_phase=phase,
        observed_at=datetime.now(timezone.utc),
        expected_critical_ranks=3,
        runtime_profile_version="profile-a",
        containers=(
            [
                ContainerObservation(
                    pod_uid=f"{attempt}-pod-{rank}",
                    pod_name=f"worker-{rank}",
                    container_name="trainer",
                    role="worker",
                    rank=rank,
                    node_id=f"node-{rank}",
                    gpu_count=8,
                )
                for rank in range(3)
            ]
            if phase in {WorkloadPhase.PENDING, WorkloadPhase.RUNNING}
            else []
        ),
    ).model_dump(mode="json")


def snapshot(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"observations": list(rows)}


@pytest.fixture
def regional() -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(cluster_id="cluster-a"),
        store_snapshot=Mock(
            return_value=snapshot(observation(), observation("a-restarted"))
        ),
    )


def drain(
    regional: SimpleNamespace, path: Path, *, timeout: int = 30
) -> dict[str, Any]:
    return runner.wait_a_workload_drained(
        regional,  # type: ignore[arg-type]
        job_id=JOB,
        expected_attempt_ids=ATTEMPTS,
        case_dir=path,
        timeout_seconds=timeout,
    )


def test_drain_waits_for_both_attempts_and_records_the_bound_proof(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path
) -> None:
    regional.store_snapshot.side_effect = [
        snapshot(
            observation(phase=WorkloadPhase.RUNNING),
            observation("a-restarted", phase=WorkloadPhase.RUNNING),
        ),
        snapshot(
            observation(), observation("a-restarted", phase=WorkloadPhase.RUNNING)
        ),
        snapshot(observation(), observation("a-restarted")),
    ]

    result = drain(regional, tmp_path)

    assert result["drained"] is True
    assert result["job_id"] == JOB and result["cluster_id"] == "cluster-a"
    assert result["expected_attempt_ids"] == sorted(ATTEMPTS)
    assert [len(entry["live"]) for entry in result["entries"]] == [2, 1, 0]
    assert clock.sleeps == [5, 5]
    assert json.loads((tmp_path / "a-drain.json").read_text()) == result
    assert all(
        call.kwargs == {"job_id": JOB, "queue_attempts": 1}
        for call in regional.store_snapshot.call_args_list
    ), "drain must query all attempts for the bound job"


@pytest.mark.parametrize("phase", list(WorkloadPhase))
def test_complete_terminated_ranks_or_terminal_tombstones_prove_drain(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path, phase: WorkloadPhase
) -> None:
    row = observation(phase=phase)
    for container in row["containers"]:
        container.update(terminated=True, exit_code=0)
    regional.store_snapshot.return_value = snapshot(row, observation("a-restarted"))

    assert drain(regional, tmp_path)["drained"] is True
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "environment", [Environment.KUBERNETES, Environment.EKS, Environment.HYPERPOD_EKS]
)
def test_drain_accepts_supported_kubernetes_observation_environments(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path, environment: Environment
) -> None:
    for row in regional.store_snapshot.return_value["observations"]:
        row["environment"] = environment.value

    assert drain(regional, tmp_path)["drained"] is True
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "report",
    [
        None,
        [],
        {},
        {"observations": None},
        {"observations": False},
        {"observations": {}},
        {"observations": "unknown"},
        {"observations": []},
        {"observations": [None]},
        {"observations": [{}]},
    ],
)
def test_missing_or_invalid_reports_never_become_drain_proof(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path, report: Any
) -> None:
    regional.store_snapshot.return_value = report

    with pytest.raises(runner.RegionalFixtureError, match="drain"):
        drain(regional, tmp_path)

    proof = json.loads((tmp_path / "a-drain.json").read_text())
    assert proof["drained"] is False
    assert regional.store_snapshot.call_count == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "change",
    [
        "cluster",
        "job",
        "attempt",
        "empty-attempt",
        "duplicate-attempt",
        "missing-attempt",
        "unknown-phase",
        "missing-phase",
        "missing-containers",
        "null-containers",
        "empty-active-containers",
        "missing-terminated",
        "missing-critical",
        "empty-pod",
        "blank-profile",
        "missing-exit-code",
        "text-terminated",
        "integer-terminated",
        "partial-ranks",
        "duplicate-ranks",
        "duplicate-container",
        "invalid-time",
        "naive-time",
        "wrong-environment",
    ],
)
def test_partial_or_unbound_observations_fail_closed(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path, change: str
) -> None:
    row = observation(phase=WorkloadPhase.RUNNING)
    for container in row["containers"]:
        container.update(terminated=True, exit_code=0)
    rows = [row, observation("a-restarted")]
    if change in {"cluster", "job", "attempt"}:
        row[f"{change}_id"] = "foreign"
    elif change == "empty-attempt":
        row["attempt_id"] = ""
    elif change == "duplicate-attempt":
        rows.append(deepcopy(row))
    elif change == "missing-attempt":
        rows.pop()
    elif change == "unknown-phase":
        row["workload_phase"] = "UNKNOWN"
    elif change == "missing-phase":
        row.pop("workload_phase")
    elif change == "missing-containers":
        row.pop("containers")
    elif change == "null-containers":
        row["containers"] = None
    elif change == "empty-active-containers":
        row["containers"] = []
    elif change == "missing-terminated":
        row["containers"][0].pop("terminated")
    elif change == "missing-critical":
        row["containers"][0].pop("critical")
    elif change == "empty-pod":
        row["containers"][0]["pod_uid"] = ""
    elif change == "blank-profile":
        row["runtime_profile_version"] = " "
    elif change == "missing-exit-code":
        row["containers"][0].pop("exit_code")
    elif change == "text-terminated":
        row["containers"][0]["terminated"] = "true"
    elif change == "integer-terminated":
        row["containers"][0]["terminated"] = 1
    elif change == "partial-ranks":
        row["containers"].pop()
    elif change == "duplicate-ranks":
        row["containers"][1]["rank"] = row["containers"][0]["rank"]
    elif change == "duplicate-container":
        row["containers"][1] = deepcopy(row["containers"][0])
    elif change == "invalid-time":
        row["observed_at"] = "unknown"
    elif change == "naive-time":
        row["observed_at"] = "2026-09-15T00:00:00"
    else:
        row["environment"] = Environment.HYPERPOD_SLURM.value
    regional.store_snapshot.return_value = snapshot(*rows)

    with pytest.raises(runner.RegionalFixtureError, match="drain"):
        drain(regional, tmp_path)

    assert json.loads((tmp_path / "a-drain.json").read_text())["drained"] is False
    assert clock.sleeps == []


@pytest.mark.parametrize("phase", [WorkloadPhase.PENDING, WorkloadPhase.RUNNING])
def test_stale_or_unplaced_live_container_still_prevents_drain(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path, phase: WorkloadPhase
) -> None:
    row = observation(phase=phase)
    row["observed_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    for container in row["containers"]:
        container["node_id"] = None
    regional.store_snapshot.return_value = snapshot(row, observation("a-restarted"))

    with pytest.raises(runner.RegionalFixtureError, match="total deadline"):
        drain(regional, tmp_path, timeout=12)

    assert clock.now == 12 and clock.sleeps == [5, 5, 2]
    proof = json.loads((tmp_path / "a-drain.json").read_text())
    assert proof["drained"] is False
    assert all(entry["live"] for entry in proof["entries"]), (
        "unplaced active containers must remain blockers in every sample"
    )


def test_default_drain_budget_outlives_missing_attempt_grace(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path
) -> None:
    def report(**_kwargs: Any) -> dict[str, Any]:
        phase = WorkloadPhase.RUNNING if clock.now < 340 else WorkloadPhase.STOPPED
        return snapshot(
            observation(phase=phase), observation("a-restarted", phase=phase)
        )

    regional.store_snapshot.side_effect = report
    result = runner.wait_a_workload_drained(
        regional,  # type: ignore[arg-type]
        job_id=JOB,
        expected_attempt_ids=ATTEMPTS,
        case_dir=tmp_path,
    )

    assert result["drained"] is True and clock.now == 340
    assert runner.A_WORKLOAD_DRAIN_TIMEOUT_SECONDS == 600


def test_snapshot_time_is_inside_one_total_deadline_and_cleanup_scope_is_restored(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path
) -> None:
    budgets: list[float] = []

    def report(**_kwargs: Any) -> dict[str, Any]:
        deadline = deadlines.current_deadline()
        assert deadline is not None
        budgets.append(deadline.remaining())
        clock.now += 8
        return snapshot(
            observation(
                phase=WorkloadPhase.RUNNING
                if len(budgets) == 1
                else WorkloadPhase.STOPPED
            ),
            observation("a-restarted"),
        )

    regional.store_snapshot.side_effect = report
    with deadlines.deadline_scope("parent", 100) as parent:
        with pytest.raises(runner.RegionalFixtureError, match="total deadline"):
            drain(regional, tmp_path, timeout=20)
        assert deadlines.current_deadline() is parent
        assert parent.remaining() == 79

    assert budgets == [20, 7]
    assert deadlines.current_deadline() is None
    assert json.loads((tmp_path / "a-drain.json").read_text())["drained"] is False


@pytest.mark.parametrize("limiter", ["parent", "maintenance"])
def test_drain_cannot_extend_a_shorter_existing_deadline(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path, limiter: str
) -> None:
    regional.store_snapshot.return_value = snapshot(
        observation(phase=WorkloadPhase.RUNNING), observation("a-restarted")
    )
    scope = (
        deadlines.deadline_scope("short parent", 12)
        if limiter == "parent"
        else guard.action_window(datetime.now(timezone.utc) + timedelta(seconds=12))
    )
    with scope:
        with pytest.raises(runner.RegionalFixtureError, match="deadline|maintenance"):
            drain(regional, tmp_path, timeout=600)

    assert clock.now <= 12 and regional.store_snapshot.call_count == 3
    assert json.loads((tmp_path / "a-drain.json").read_text())["drained"] is False


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_drain_timeout_is_rejected_before_reading(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path, timeout: Any
) -> None:
    with pytest.raises(runner.RegionalFixtureError, match="finite"):
        drain(regional, tmp_path, timeout=timeout)
    regional.store_snapshot.assert_not_called()
    assert clock.sleeps == []


def test_read_failure_keeps_negative_evidence_and_never_retries_as_empty(
    regional: SimpleNamespace, clock: Clock, tmp_path: Path
) -> None:
    regional.store_snapshot.side_effect = runner.RegionalFixtureError("read failed")

    with pytest.raises(runner.RegionalFixtureError, match="read failed"):
        drain(regional, tmp_path)

    assert json.loads((tmp_path / "a-drain.json").read_text())["drained"] is False
    assert regional.store_snapshot.call_count == 1
    assert deadlines.current_deadline() is None and clock.sleeps == []
