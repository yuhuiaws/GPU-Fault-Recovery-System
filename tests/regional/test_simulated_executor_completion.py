"""The simulated executor completes every claimed command on its own clock.

Batch 80 (r4 timeline): the claim loop waited for its whole claimed batch, so
one 120 s simulated RESTART_NODE parked every other workflow of its cluster
and the integrated p99 moved in harness-made steps. Each command now reports
when its own simulated duration elapses and a freed slot is claimed again at
once; leases, renewals, fencing tokens and result payloads are unchanged.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from scripts.perf import benchmark_regional_action_executor as executor

WAIT_SECONDS = 15.0


@pytest.fixture(autouse=True)
def executor_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTOR_PROTOCOL_VERSION", "2")
    monkeypatch.setenv("EXECUTOR_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("EXECUTOR_COMPATIBILITY_DIGEST", "b" * 64)


class VirtualClock:
    """Simulated action time: a sleeper wakes only when the test advances."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleepers: dict[str, float] = {}
        self.condition = threading.Condition()

    def sleep(self, seconds: float) -> None:
        name = threading.current_thread().name
        deadline = time.monotonic() + 60
        with self.condition:
            wake_at = self.now + seconds
            self.sleepers[name] = wake_at
            self.condition.notify_all()
            try:
                while self.now < wake_at and time.monotonic() < deadline:
                    self.condition.wait(timeout=0.5)
            finally:
                del self.sleepers[name]
                self.condition.notify_all()

    def advance(self, value: float) -> None:
        with self.condition:
            self.now = value
            self.condition.notify_all()

    def wait_until(self, predicate: Callable[[], bool]) -> bool:
        with self.condition:
            return self.condition.wait_for(predicate, timeout=WAIT_SECONDS)


class FakeControlPlane:
    """Claim and result endpoints; a finished step releases the next one."""

    def __init__(self, clock: VirtualClock, offers: list[list[dict[str, Any]]]):
        self.clock = clock
        self.offers = offers
        self.follow_ups: dict[str, dict[str, Any]] = {}
        self.results: list[str] = []
        self.claims: list[int] = []
        self.payloads: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if path == "/v1/regional/executors/claim":
                self.claims.append(int(payload["max_commands"]))
                commands = self.offers.pop(0) if self.offers else []
                return {"commands": commands[: int(payload["max_commands"])]}
            command_id = path.split("/")[-2]
            self.payloads.append(payload)
            self.results.append(command_id)
            follow_up = self.follow_ups.pop(command_id, None)
            if follow_up is not None:
                self.offers.append([follow_up])
        with self.clock.condition:
            self.clock.condition.notify_all()
        return {}


def command(command_id: str, operation: str) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "lease_token": f"lease-{command_id}",
        "step": {"operation": operation, "node_ids": ["node-a"]},
    }


def simulation(control: FakeControlPlane, clock: VirtualClock) -> Any:
    return executor.SimulationState(
        client=control,  # type: ignore[arg-type]
        executor_id="action-executor-000",
        operation_delays=dict(executor.DELAYS),
        delay_scale=1.0,
        lease_seconds=120,
        renewal_interval=3600.0,
        inject_renew_failure_pending=False,
        sleep=clock.sleep,
    )


def start_loop(state: Any, *, workers: int, expected: int) -> dict[str, Any]:
    outcome: dict[str, Any] = {}

    def run() -> None:
        outcome["tally"] = executor.run_claim_loop(
            state,
            workers=workers,
            expected=expected,
            max_seconds=WAIT_SECONDS * 4,
            idle_exit_seconds=0,
        )

    worker = threading.Thread(target=run, name="claim-loop", daemon=True)
    worker.start()
    outcome["thread"] = worker
    return outcome


def test_mixed_duration_batch_completes_in_duration_order_while_claiming_more() -> None:
    clock = VirtualClock()
    control = FakeControlPlane(
        clock,
        [
            [
                command("reboot", "RESTART_NODE"),
                command("reset", "RESET_GPU"),
                command("cordon", "MARK_UNSCHEDULABLE"),
            ]
        ],
    )
    control.follow_ups["reset"] = command("restore", "RESTORE_GPU_SERVICES")
    state = simulation(control, clock)
    outcome = start_loop(state, workers=8, expected=4)

    assert clock.wait_until(lambda: len(clock.sleepers) == 3), (
        "the claimed batch did not start concurrently"
    )
    clock.advance(1.0)
    assert clock.wait_until(lambda: control.results == ["cordon"]), (
        "the 0.5 s command did not report on its own while the others ran"
    )
    clock.advance(9.0)
    assert clock.wait_until(lambda: control.results == ["cordon", "reset"]), (
        "the 8 s reset did not report while the reboot ran"
    )
    assert clock.wait_until(lambda: len(clock.sleepers) == 2), (
        "the follow-up command waited for the 120 s reboot of the same batch"
    )
    clock.advance(12.0)
    assert clock.wait_until(
        lambda: control.results == ["cordon", "reset", "restore"]
    ), "the follow-up did not complete while the reboot was still running"
    clock.advance(121.0)
    outcome["thread"].join(timeout=WAIT_SECONDS)

    assert control.results == ["cordon", "reset", "restore", "reboot"], (
        "commands must report in the order their own durations elapse"
    )
    tally = outcome["tally"]
    assert tally.completed_ids == {"cordon", "reset", "restore", "reboot"}
    assert tally.duplicate_claims == 0
    assert tally.result_errors == 0
    assert state.max_active_commands == 3
    assert [item["lease_token"] for item in control.payloads] == [
        "lease-cordon",
        "lease-reset",
        "lease-restore",
        "lease-reboot",
    ], "every result must stay fenced by its own claim's lease token"
    assert {item["status"] for item in control.payloads} == {"SUCCEEDED"}
    assert all(item["details"]["simulated"] is True for item in control.payloads), (
        "result reporting must keep the simulated marker"
    )


def test_claims_never_exceed_the_free_worker_slots() -> None:
    clock = VirtualClock()
    control = FakeControlPlane(
        clock, [[command(f"reset-{index}", "RESET_GPU") for index in range(5)]]
    )
    state = simulation(control, clock)
    outcome = start_loop(state, workers=2, expected=3)

    assert clock.wait_until(lambda: len(clock.sleepers) == 2), (
        "the executor did not fill its two worker slots"
    )
    control.offers.append([command("reset-late", "RESET_GPU")])
    claims_while_full = len(control.claims)
    clock.advance(4.0)
    time.sleep(0.3)
    assert len(control.claims) == claims_while_full, (
        "the executor claimed while every worker slot was busy"
    )
    clock.advance(9.0)
    assert clock.wait_until(lambda: len(control.results) == 2), (
        "the first two commands did not report"
    )
    assert clock.wait_until(lambda: len(clock.sleepers) == 1), (
        "a freed slot was not claimed again"
    )
    clock.advance(20.0)
    outcome["thread"].join(timeout=WAIT_SECONDS)

    assert control.claims[0] == 2, "the first claim must ask for the free slots only"
    assert all(1 <= value <= 2 for value in control.claims), control.claims
    assert state.max_active_commands == 2
    assert outcome["tally"].completed_ids == {"reset-0", "reset-1", "reset-late"}
