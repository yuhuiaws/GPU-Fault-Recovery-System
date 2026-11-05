"""The Agent-lifetime passive sampler: one preemptible sample per target per epoch.

``CommunicationRuntime.tick`` samples every target of the current lease once
per 30-second epoch, least recently sampled first, stops at a withdrawn,
replaced or expired lease and at a stop request, and keeps the newest eight
results for the health view. Nothing samples without a verified service
envelope. The supervisor and the GPU activity fence are fakes that record
what the runtime asked of them; the runtime module's clock is a fake the test
moves, so no epoch boundary or lease expiry depends on the wall clock.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import communication_runtime
from gpu_fault.node_agent.communication_runtime import (
    CommunicationRuntime,
    bounded_request,
)
from gpu_fault.node_agent.observation_limits import OBSERVATION_REQUEST_BUDGET_BYTES
from gpu_fault.node_agent.observation_supervisor import ObservationIncomplete
from tests.node_agent._support import command

VERIFIED = {"verified": True, "cpu_cores": 2, "memory_bytes": 1 << 30}


class Clock:
    """``time`` as the runtime module sees it; only the test moves it."""

    def __init__(self) -> None:
        self.wall = 30.0 * 60_000_000
        self.mono = 5_000.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono


class Activity:
    """The GPU activity fence: records admissions and releases."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.admitted: list[tuple[Any, str, str]] = []
        self.released: list[str] = []

    def admit(self, operation: Any, key: str, incident: str) -> str:
        if self.failure is not None:
            raise self.failure
        self.admitted.append((operation, key, incident))
        return key

    def release(self, activity: str) -> None:
        self.released.append(activity)


class Supervisor:
    """Answers each sample, runs ``during`` inside it, releases like the real one."""

    def __init__(
        self, during: Any = None, failures: dict[str, Exception] | None = None
    ):
        self.during = during
        self.failures = failures or {}
        self.requests: list[dict[str, Any]] = []

    def run(
        self,
        request: dict[str, Any],
        timeout: float,
        *,
        release: Any = lambda: None,
        preemptible: bool = False,
        slot_wait: float = 0.0,
    ) -> dict[str, Any]:
        attempt = request["target"]["attempt_id"]
        self.requests.append(
            {"request": request, "timeout": timeout, "preemptible": preemptible}
        )
        try:
            if attempt in self.failures:
                raise self.failures[attempt]
            if self.during is not None:
                self.during()
            return {"status": "SAMPLED", "attempt": attempt}
        finally:
            release()

    def sampled(self) -> list[str]:
        return [item["request"]["target"]["attempt_id"] for item in self.requests]


class World:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **values):
        self.limits = dict(VERIFIED)
        self.clock = Clock()
        monkeypatch.setattr(
            "gpu_fault.node_agent.communication_runtime.service_limits",
            lambda: dict(self.limits),
        )
        monkeypatch.setattr(communication_runtime, "time", self.clock)
        root = tmp_path / "proc"
        if values.get("boot", "boot") is not None:
            boot = root / "sys/kernel/random/boot_id"
            boot.parent.mkdir(parents=True)
            boot.write_text(values.get("boot", "boot") + "\n")
        self.activity = values.get("activity") or Activity()
        self.runtime = CommunicationRuntime(
            SimpleNamespace(
                node_ids={"node-a"},
                agent_generation=1,
                proc_root=root,
                infiniband_root=tmp_path / "ib",
                diagnostic_output_dir=tmp_path / "evidence",
                python_stack_tool="",
                diagnostic_s3_uri=None,
                diagnostic_retention_seconds=600,
                diagnostic_max_archives=10,
                gpu_activity=self.activity,
            )
        )
        self.supervisor = values.get("supervisor") or Supervisor()
        self.runtime.supervisor = self.supervisor

    @property
    def epoch(self) -> int:
        return int(self.clock.wall // 30)


def plan(*attempts: str, boot: str = "boot") -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "boot_id": boot,
        "generation": 1,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=60)).isoformat(),
        "targets": [
            {"job_id": "job", "attempt_id": attempt, "cgroup_paths": ["/training/job"]}
            for attempt in attempts
        ],
        "expected_target_count": len(attempts),
    }


def test_a_tick_samples_every_target_once_per_epoch(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    world.runtime.accept_plan(plan("a1", "a2"))

    world.runtime.tick()

    epoch = world.epoch
    assert world.supervisor.sampled() == ["a1", "a2"], world.supervisor.requests
    first = world.supervisor.requests[0]
    assert first["timeout"] == 12 and first["preemptible"] is True, first
    assert first["request"]["operation"] == "sample", first
    assert first["request"]["settings"] == world.runtime.settings, first
    assert first["request"]["plan"]["boot_id"] == "boot", first
    assert first["request"]["target"]["cgroup_paths"] == ["/training/job"], first
    assert world.activity.admitted == [
        (Op.COLLECT_HUNG_SUMMARY, f"passive/a1/{epoch}", "passive-observation"),
        (Op.COLLECT_HUNG_SUMMARY, f"passive/a2/{epoch}", "passive-observation"),
    ], world.activity.admitted
    assert world.activity.released == [f"passive/a1/{epoch}", f"passive/a2/{epoch}"]
    assert world.runtime.health()["samples"] == {
        "a1": {"status": "SAMPLED", "attempt": "a1"},
        "a2": {"status": "SAMPLED", "attempt": "a2"},
    }

    world.runtime.tick()
    assert world.supervisor.sampled() == ["a1", "a2"], "one sample per epoch"

    world.clock.wall += 30
    world.runtime.tick()
    assert world.supervisor.sampled() == ["a1", "a2", "a1", "a2"], "next epoch"


@pytest.mark.parametrize("state", ["unverified", "no-lease", "expired-lease"])
def test_nothing_is_sampled_without_a_verified_envelope_and_a_live_lease(
    tmp_path, monkeypatch, state
):
    world = World(tmp_path, monkeypatch)
    if state != "no-lease":
        world.runtime.accept_plan(plan("a1"))
    if state == "unverified":
        world.limits = {"verified": False, "reason": "NO_DELEGATED_CGROUP"}
    elif state == "expired-lease":
        world.clock.mono += 61

    world.runtime.tick()

    assert world.supervisor.requests == [], state
    assert world.activity.admitted == [], state
    health = world.runtime.health()
    assert health["samples"] == {}, health
    assert health["service_limits"]["verified"] is (state != "unverified"), health


@pytest.mark.parametrize("change", ["withdrawn", "replaced", "expired"])
def test_a_lease_that_ends_mid_tick_stops_before_the_next_target(
    tmp_path, monkeypatch, change
):
    world = World(tmp_path, monkeypatch)

    def end_lease() -> None:
        if change == "withdrawn":
            world.runtime.accept_plan(None)
        elif change == "replaced":
            world.runtime.accept_plan(plan("b1"))
        else:
            world.clock.mono += 61

    world.supervisor.during = end_lease
    world.runtime.accept_plan(plan("a1", "a2"))

    world.runtime.tick()

    assert world.supervisor.sampled() == ["a1"], change
    assert list(world.runtime.health()["samples"]) == ["a1"], change


def test_a_stop_request_ends_the_tick_and_the_run_loop(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    stop = threading.Event()
    world.supervisor.during = stop.set
    world.runtime.accept_plan(plan("a1", "a2"))

    world.runtime.run(stop)

    assert world.supervisor.sampled() == ["a1"], "stop ends the tick at once"
    # The target the stop skipped is the least recently sampled one, so the
    # next epoch samples it first.
    world.supervisor.during = None
    world.clock.wall += 30
    world.runtime.tick()
    assert world.supervisor.sampled() == ["a1", "a2", "a1"], world.supervisor.requests


def test_a_stopped_run_loop_samples_nothing(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    world.runtime.accept_plan(plan("a1"))
    stop = threading.Event()
    stop.set()

    world.runtime.run(stop)

    assert world.supervisor.requests == [], world.supervisor.requests


@pytest.mark.parametrize(
    "where,failure,reason",
    [
        pytest.param(
            "admit",
            ValueError("GPU maintenance fence is held"),
            "GPU maintenance fence is held",
            id="fence-refused",
        ),
        pytest.param(
            "supervisor",
            RuntimeError("observation worker could not start"),
            "observation worker could not start",
            id="worker-not-started",
        ),
        pytest.param("supervisor", OSError("e" * 300), "e" * 128, id="bounded-reason"),
    ],
)
def test_a_sample_that_cannot_run_is_recorded_unavailable_and_the_tick_goes_on(
    tmp_path, monkeypatch, where, failure, reason
):
    activity = Activity(failure if where == "admit" else None)
    supervisor = Supervisor(failures={"a1": failure} if where == "supervisor" else {})
    world = World(tmp_path, monkeypatch, activity=activity, supervisor=supervisor)
    world.runtime.accept_plan(plan("a1", "a2"))

    world.runtime.tick()

    samples = world.runtime.health()["samples"]
    assert samples["a1"] == {"status": "UNAVAILABLE", "reason": reason}, samples
    if where == "admit":
        assert samples["a2"]["status"] == "UNAVAILABLE", samples
        assert supervisor.requests == [] and activity.released == [], activity.released
    else:
        assert samples["a2"] == {"status": "SAMPLED", "attempt": "a2"}, samples
        epoch = world.epoch
        assert activity.released == [f"passive/a1/{epoch}", f"passive/a2/{epoch}"]


def test_the_health_view_keeps_the_newest_eight_samples(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    world.runtime.accept_plan(plan(*(f"a{index}" for index in range(8))))
    world.runtime.tick()
    world.runtime.accept_plan(plan("b0", "b1"))

    world.runtime.tick()

    assert list(world.runtime.health()["samples"]) == [
        *(f"a{index}" for index in range(2, 8)),
        "b0",
        "b1",
    ]


def test_a_node_without_a_readable_boot_id_accepts_no_lease(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch, boot=None)

    assert world.runtime.boot_id == "", world.runtime.boot_id
    for boot in ("boot", ""):
        with pytest.raises(ValueError, match="scope identity mismatch"):
            world.runtime.accept_plan(plan("a1", boot=boot))
    assert world.runtime.health()["lease_valid"] is False, world.runtime.health()


def test_withdrawing_the_lease_clears_the_scope(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    world.runtime.accept_plan(plan("a1"))
    assert world.runtime.health()["assigned_targets"] == 1, world.runtime.health()

    world.runtime.accept_plan(None)
    world.runtime.tick()

    health = world.runtime.health()
    assert health["lease_valid"] is False, health
    assert health["assigned_targets"] == 0 == health["expected_targets"], health
    assert world.supervisor.requests == [], "a withdrawn lease is never sampled"


def test_a_command_is_refused_before_admission_without_a_verified_envelope(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    world.limits = {
        "verified": False,
        "reason": "BOUNDED_CGROUP_V2_ENVELOPE_UNAVAILABLE",
    }

    with pytest.raises(
        ObservationIncomplete, match="RESOURCE_LIMITS_UNVERIFIED"
    ) as raised:
        world.runtime.run_command("triage", command(Op.COLLECT_HUNG_TRIAGE))

    assert raised.value.action_details == {
        "status": "RESOURCE_LIMITS_UNVERIFIED",
        "evidence_complete": False,
    }
    assert world.activity.admitted == [] and world.supervisor.requests == [], (
        "nothing is admitted or started for a refused command"
    )
    assert world.runtime.health()["service_limits"]["verified"] is False


OVER_BUDGET = "x" * (OBSERVATION_REQUEST_BUDGET_BYTES + 1)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"operation": "triage", "command": OVER_BUDGET}, id="no-command"),
        pytest.param(
            {"operation": "triage", "command": {"parameters": OVER_BUDGET}},
            id="no-parameters",
        ),
        pytest.param(
            {
                "operation": "bundle",
                "command": {"parameters": {"raw_flights_by_node": OVER_BUDGET}},
            },
            id="nothing-context-only",
        ),
    ],
)
def test_an_over_budget_request_without_context_only_parameters_is_left_whole(value):
    request, trimmed = bounded_request(value)

    assert request is value, "only context-only parameters may ever be trimmed"
    assert trimmed == {}, trimmed
