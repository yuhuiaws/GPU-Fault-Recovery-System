"""The late-XID fixture drains predecessors without delaying its killed attempt."""

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
from scripts.e2e.regional import run_collect021_late_xid_after_pod_death as runner
from tests.regional import _cov95_collect_training as training_support
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401

training_case = training_support.training_case
JOB = "own-job"
ATTEMPT = "shared-attempt-name"
NODE = "node-a"


def observation(
    *,
    job: str = JOB,
    attempt: str = ATTEMPT,
    node: str = NODE,
    phase: WorkloadPhase = WorkloadPhase.RUNNING,
) -> dict[str, Any]:
    return AttemptObservation(
        cluster_id="cluster-a",
        environment=Environment.HYPERPOD_EKS,
        job_id=job,
        attempt_id=attempt,
        workload_phase=phase,
        observed_at=datetime.now(timezone.utc),
        runtime_profile_version="profile-a",
        expected_critical_ranks=1,
        containers=[
            ContainerObservation(
                pod_uid=f"{job}/{attempt}/pod",
                pod_name=f"{job}-worker",
                container_name="trainer",
                role="worker",
                rank=0,
                node_id=node,
                gpu_count=8,
            )
        ],
    ).model_dump(mode="json")


def blockers(state: object, *, own_idle: bool = False) -> list[dict[str, str]]:
    return runner.node_attempt_blockers(
        state,
        cluster_id="cluster-a",
        node=NODE,
        own_job_id=JOB,
        own_attempt_id=ATTEMPT,
        own_must_be_idle=own_idle,
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    for module in (runner, guard, deadlines):
        monkeypatch.setattr(module, "time", value)
    return value


@pytest.fixture
def regional() -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(cluster_id="cluster-a"),
        store_snapshot=Mock(return_value={"observations": [observation()]}),
    )


def wait(regional: Any, path: Path, timeout: int = 30) -> dict[str, Any]:
    return runner.wait_node_clear_of_foreign_attempts(
        regional,
        node=NODE,
        own_job_id=JOB,
        own_attempt_id=ATTEMPT,
        case_dir=path,
        timeout_seconds=timeout,
        poll_seconds=10,
    )


def test_same_attempt_name_in_a_different_job_is_not_exempt() -> None:
    assert blockers(
        {"observations": [observation(), observation(job="foreign-job")]}
    ) == [{"job_id": "foreign-job", "attempt_id": ATTEMPT}]


@pytest.mark.parametrize("phase", list(WorkloadPhase))
def test_only_complete_off_node_or_terminal_foreign_observations_clear(
    phase: WorkloadPhase,
) -> None:
    foreign = observation(job="foreign-job", node="node-b", phase=phase)
    assert blockers({"observations": [observation(), foreign]}) == []


@pytest.mark.parametrize(
    "change", ["absent-containers", "partial-ranks", "unplaced", "stale-on-node"]
)
def test_incomplete_or_stale_active_foreign_attempts_still_block(change: str) -> None:
    foreign = observation(job="foreign-job", node="node-b")
    if change == "absent-containers":
        foreign["containers"] = []
    elif change == "partial-ranks":
        foreign["expected_critical_ranks"] = 2
    elif change == "unplaced":
        foreign["containers"][0]["node_id"] = None
    else:
        foreign["containers"][0]["node_id"] = NODE
        foreign["observed_at"] = "2020-01-01T00:00:00Z"
    assert blockers({"observations": [observation(), foreign]}), (
        "absence or old timestamps do not prove a foreign attempt has drained"
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing-report",
        "missing-list",
        "empty-list",
        "missing-own",
        "foreign-cluster",
        "duplicate",
        "unknown-phase",
        "missing-containers",
        "missing-termination",
        "invalid-rank",
        "naive-time",
        "missing-profile",
    ],
)
def test_malformed_or_unbound_reports_never_prove_an_idle_node(change: str) -> None:
    own = observation()
    state: Any = {"observations": [own]}
    if change == "missing-report":
        state = None
    elif change == "missing-list":
        state = {}
    elif change == "empty-list":
        state["observations"] = []
    elif change == "missing-own":
        own["job_id"] = "other-job"
    elif change == "foreign-cluster":
        own["cluster_id"] = "other-cluster"
    elif change == "duplicate":
        state["observations"].append(deepcopy(own))
    elif change == "unknown-phase":
        own["workload_phase"] = "UNKNOWN"
    elif change == "missing-containers":
        own.pop("containers")
    elif change == "missing-termination":
        own["containers"][0].pop("terminated")
    elif change == "invalid-rank":
        own["containers"][0]["rank"] = True
    elif change == "naive-time":
        own["observed_at"] = "2026-09-15T00:00:00"
    else:
        own["runtime_profile_version"] = ""
    with pytest.raises(runner.RegionalFixtureError):
        blockers(state)


def test_owned_attempt_must_still_be_idle_at_the_post_death_read() -> None:
    with pytest.raises(runner.RegionalFixtureError, match="became active"):
        blockers({"observations": [observation()]}, own_idle=True)
    assert (
        blockers(
            {"observations": [observation(phase=WorkloadPhase.FAILED)]}, own_idle=True
        )
        == []
    )


def test_wait_rechecks_all_attempts_and_persists_a_bound_history(
    regional: Any, clock: Clock, tmp_path: Path
) -> None:
    regional.store_snapshot.side_effect = [
        {"observations": [observation(), observation(job="foreign-job")]},
        {
            "observations": [
                observation(),
                observation(job="foreign-job", phase=WorkloadPhase.STOPPED),
            ]
        },
    ]
    proof = wait(regional, tmp_path)
    assert proof["clear"] is True
    assert proof["job_id"] == JOB and proof["attempt_id"] == ATTEMPT
    assert [len(item["blockers"]) for item in proof["entries"]] == [1, 0]
    assert clock.sleeps == [10]
    assert (
        json.loads((tmp_path / "foreign-attempts-before-kill.json").read_text())
        == proof
    )
    assert all(
        call.kwargs == {"node": NODE, "queue_attempts": 1}
        for call in regional.store_snapshot.call_args_list
    ), "node drainage must read all attempts, not filter away foreign jobs"


def test_wait_does_not_start_another_read_after_total_deadline(
    regional: Any, clock: Clock, tmp_path: Path
) -> None:
    regional.store_snapshot.return_value["observations"].append(
        observation(job="foreign-job")
    )
    with pytest.raises(runner.RegionalFixtureError, match="total deadline"):
        wait(regional, tmp_path, timeout=3)
    assert clock.sleeps == [3]
    assert regional.store_snapshot.call_count == 1
    assert not json.loads((tmp_path / "foreign-attempts-before-kill.json").read_text())[
        "clear"
    ]


def test_late_returning_read_cannot_produce_clear_evidence(
    regional: Any, clock: Clock, tmp_path: Path
) -> None:
    def read(**kwargs: Any) -> dict[str, Any]:
        clock.now += 61
        return {"observations": [observation()]}

    regional.store_snapshot.side_effect = read
    with pytest.raises(runner.RegionalFixtureError, match="total deadline"):
        wait(regional, tmp_path, timeout=120)
    assert clock.sleeps == []


def test_slow_receipt_write_cannot_return_clear_after_the_deadline(
    regional: Any, clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = runner.write_json_atomic

    def delayed_write(path: Path, value: dict[str, Any]) -> None:
        write(path, value)
        if value.get("clear") is True:
            clock.now += 31

    monkeypatch.setattr(runner, "write_json_atomic", delayed_write)
    with pytest.raises(runner.RegionalFixtureError, match="total deadline"):
        wait(regional, tmp_path)
    assert not json.loads((tmp_path / "foreign-attempts-before-kill.json").read_text())[
        "clear"
    ]


def test_wait_honors_parent_and_maintenance_deadlines(
    regional: Any, clock: Clock, tmp_path: Path
) -> None:
    regional.store_snapshot.return_value["observations"].append(
        observation(job="foreign-job")
    )
    with guard.action_window(datetime.now(timezone.utc) + timedelta(seconds=20)):
        with deadlines.deadline_scope("parent", 3):
            with pytest.raises(runner.RegionalFixtureError, match="total deadline"):
                wait(regional, tmp_path)
    assert clock.sleeps == [3]


@pytest.mark.parametrize("training_case", [runner], indirect=True)
def test_complete_case_checks_foreign_attempts_before_kill_and_before_xid(
    training_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = training_case
    original = runner.wait_for_decision
    bounds = []
    started = datetime.now(timezone.utc)

    def capture_bound(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bounds.append(kwargs["observed_after"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "wait_for_decision", capture_bound)
    assert (
        runner.execute_case(
            harness.settings,
            harness.root,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
        == 0
    )
    calls = [call[0] for call in harness.calls]
    reads = [index for index, call in enumerate(calls) if call == "node-attempt-read"]
    assert len(reads) == 2
    assert reads[0] < calls.index("kill-workload") < reads[1] < calls.index("write-xid")
    assert len(bounds) == 1 and started <= bounds[0] <= datetime.now(timezone.utc), (
        "the shared Store boundary owns clock-skew adjustment; callers pass raw time"
    )


@pytest.mark.parametrize("training_case", [runner], indirect=True)
@pytest.mark.parametrize("stage", ["before-kill", "before-xid"])
def test_bad_node_report_stops_next_mutation_and_keeps_cleanup(
    training_case: Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    harness = training_case
    original = harness.regional.store_snapshot

    def read(**kwargs: Any) -> dict[str, Any]:
        if kwargs == {"node": NODE, "queue_attempts": 1} and harness.killed == (
            stage == "before-xid"
        ):
            return {"observations": []}
        return original(**kwargs)

    monkeypatch.setattr(harness.regional, "store_snapshot", read)
    assert (
        runner.execute_case(
            harness.settings,
            harness.root,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
        == 1
    )
    calls = [call[0] for call in harness.calls]
    assert "write-xid" not in calls
    assert ("kill-workload" in calls) is (stage == "before-xid")
    assert "workload-delete" in calls and "collector-cleanup" in calls


@pytest.mark.parametrize("training_case", [runner], indirect=True)
def test_post_death_receipt_expiry_refuses_xid_and_keeps_cleanup(
    training_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = training_case
    for module in (deadlines, guard):
        monkeypatch.setattr(module, "time", harness.clock)
    write = runner.write_json_atomic

    def delayed_write(path: Path, value: dict[str, Any]) -> None:
        write(path, value)
        if path.name == "foreign-attempts-before-xid.json":
            harness.clock.now += 61

    monkeypatch.setattr(runner, "write_json_atomic", delayed_write)
    assert (
        runner.execute_case(
            harness.settings,
            harness.root,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
        == 1
    )
    calls = [call[0] for call in harness.calls]
    assert "kill-workload" in calls and "write-xid" not in calls
    assert "workload-delete" in calls and "collector-cleanup" in calls
