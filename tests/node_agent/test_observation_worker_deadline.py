"""The read-only worker plans a triage from what is left of the deadline when it plans.

The supervisor's deadline runs from the command's arrival on the node; the
wait for the node's single worker slot and the start of the guard and of the
worker -- two interpreters, the package's imports -- run on it. The supervisor
used to hand the worker the seconds left at the spawn
(``OBSERVATION_BUDGET_SECONDS``), and the worker planned the triage within
them less ``WORKER_BUDGET_MARGIN_SECONDS``: the start was never subtracted, and
after a slot wait of more than 5.5 s of the triage's 28 the 2.5 s margin was
all that was left for it. On a node whose CPU a training job saturates the
start alone takes seconds (from spawn to the worker's first line 3.3 s with
eight busy loops on its CPU, 5.3 s while the guard still imported the worker),
so the triage ran past the deadline and came back ``PARTIAL_TIMEOUT``. The
supervisor now hands over the deadline itself (``OBSERVATION_DEADLINE_VARIABLE``,
a ``time.monotonic()`` reading) and the worker reads what is left right before
it plans (b214).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from gpu_fault.node_agent import observation_supervisor, observation_worker
from gpu_fault.node_agent.observation_limits import (
    OBSERVATION_DEADLINE_VARIABLE,
    WORKER_BUDGET_MARGIN_SECONDS,
)
from gpu_fault.node_agent.observation_supervisor import ObservationSupervisor
from tests.node_agent import _slow_start_observation_worker as slow_start
from tests.node_agent.test_communication_monitor import (
    REAL_WORKER_BUDGET_SECONDS,
    use_imported_package,
)
from tests.node_agent.test_observation_slot_contention import triage_command

#: The triage's own budget (``triage_command``).
COMMAND_BUDGET_SECONDS = 20
#: When the supervisor spawned the guard, on the injected clock, and what was
#: left of the deadline then: a triage (28 s) that waited the longest slot wait
#: it may (``slot_wait_for(28)`` = 8.4 s) for the passive sample to leave.
SPAWNED_AT = 1000.0
LEFT_AT_SPAWN = 28.0 - 8.4


def _request(tmp_path, triage_seconds: float | None = None) -> dict:
    command = triage_command().model_dump(mode="json")
    if triage_seconds is not None:
        command["parameters"]["triage_timeout_seconds"] = triage_seconds
    return {
        "operation": "triage",
        "settings": {
            "proc_root": str(tmp_path / "proc"),
            "infiniband_root": str(tmp_path / "ib"),
            "diagnostic_output_dir": str(tmp_path / "diagnostics"),
            "python_stack_tool": "",
        },
        "command": command,
    }


def _planned(tmp_path, monkeypatch, deadline: float | None, now: float) -> float:
    """The budget the worker plans with ``deadline`` handed and the clock at ``now``."""

    seen: list[float] = []

    def collect(self, cmd):
        seen.append(cmd.parameters["triage_timeout_seconds"])
        return {"ranks": []}

    monkeypatch.setattr(
        observation_worker.ObservationAgent, "_collect_hung_triage", collect
    )
    observation_worker.execute(
        _request(tmp_path), deadline=deadline, monotonic=lambda: now
    )
    assert len(seen) == 1, seen
    return seen[0]


@pytest.mark.parametrize(
    ("start_seconds", "planned"),
    [
        (0.0, 17.1),  # no start at all: what the seconds left at the spawn gave
        (0.3, 16.8),  # the measured start of guard and worker, idle
        (4.0, 13.1),  # a CPU-starved node: four seconds to start
    ],
)
def test_a_slow_start_is_subtracted_from_the_plan(
    tmp_path, monkeypatch, start_seconds, planned
):
    """The plan is what is left when the worker plans, less the margin, so the
    triage and its trimmed result end by the deadline however long the start
    took; planned from the spawn, it overran the deadline by the start."""

    deadline = SPAWNED_AT + LEFT_AT_SPAWN
    planned_at = SPAWNED_AT + start_seconds
    budget = _planned(tmp_path, monkeypatch, deadline, planned_at)

    assert budget == pytest.approx(planned), budget
    assert planned_at + budget + WORKER_BUDGET_MARGIN_SECONDS <= deadline + 1e-9, (
        "the triage and its margin end by the deadline"
    )
    from_spawn = LEFT_AT_SPAWN - WORKER_BUDGET_MARGIN_SECONDS
    overrun = planned_at + from_spawn + WORKER_BUDGET_MARGIN_SECONDS - deadline
    assert overrun == pytest.approx(start_seconds), overrun


@pytest.mark.parametrize("start_seconds", [1.0, 3.0, 5.0])
def test_a_start_that_used_up_the_deadline_still_plans_the_floor(
    tmp_path, monkeypatch, start_seconds
):
    """Less than the margin and the 2 s floor left (3 s left at the spawn): the
    triage still runs its 2 s floor, and the supervisor's deadline hands back
    its last checkpoint (``test_the_deadline_hands_back_the_last_checkpoint``)
    -- the start does not turn the triage into nothing."""

    budget = _planned(
        tmp_path, monkeypatch, SPAWNED_AT + 3.0, SPAWNED_AT + start_seconds
    )
    assert budget == 2.0, budget


def test_without_a_near_deadline_the_command_keeps_its_own_budget(
    tmp_path, monkeypatch
):
    assert _planned(tmp_path, monkeypatch, None, SPAWNED_AT) == COMMAND_BUDGET_SECONDS
    assert (
        _planned(tmp_path, monkeypatch, SPAWNED_AT + 28.0, SPAWNED_AT + 0.3)
        == COMMAND_BUDGET_SECONDS
    ), "27.7 s left less the margin is more than the triage's own 20 s"
    assert (
        _planned(tmp_path, monkeypatch, float("inf"), SPAWNED_AT)
        == COMMAND_BUDGET_SECONDS
    ), "no finite deadline caps nothing"


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({OBSERVATION_DEADLINE_VARIABLE: repr(4317125.912036128)}, 4317125.912036128),
        ({}, None),
        ({OBSERVATION_DEADLINE_VARIABLE: "soon"}, None),
        ({OBSERVATION_DEADLINE_VARIABLE: "inf"}, None),
        ({OBSERVATION_DEADLINE_VARIABLE: "nan"}, None),
        # The variable this one replaced carries no deadline.
        ({"OBSERVATION_BUDGET_SECONDS": "9.500"}, None),
    ],
)
def test_the_worker_reads_the_deadline_it_was_handed(environment, expected):
    assert observation_worker.handed_deadline(environment) == expected


def test_the_supervisor_hands_over_its_deadline_and_a_slow_start_shrinks_the_plan(
    tmp_path, monkeypatch
):
    """The real supervisor and the real worker's ``main``, behind a start
    ``START_SECONDS`` slower than this host's: the worker plans the triage
    within what is left after it, and answers before the deadline.

    The command asks for as long as the supervisor allows
    (``REAL_WORKER_BUDGET_SECONDS``, the longest budget the product gives a
    worker), so the deadline alone decides the plan and no start this host can
    take exhausts it: with 48 busy loops on the test's CPU a 20 s budget was
    gone before the worker planned (18 s), and the 2 s floor was all it got.
    """

    use_imported_package(monkeypatch)
    monkeypatch.setattr(
        observation_supervisor,
        "observation_guard",
        SimpleNamespace(__name__="tests.node_agent._slow_start_observation_worker"),
    )
    timeout = REAL_WORKER_BUDGET_SECONDS
    before = time.monotonic()
    result = ObservationSupervisor().run(_request(tmp_path, timeout), timeout)
    after = time.monotonic()

    assert result["status"] == "PLANNED", result
    # The deadline itself: the supervisor's start plus its timeout.
    assert before + timeout <= result["deadline"] <= after + timeout, (
        before,
        result["deadline"],
        after,
    )
    assert result["planned_at"] - before >= slow_start.START_SECONDS, result
    # The seconds left at the spawn would have planned the timeout less the
    # margin; the start is subtracted, and so is whatever this host added.
    assert result["planned_seconds"] <= (
        timeout - slow_start.START_SECONDS - WORKER_BUDGET_MARGIN_SECONDS
    ), result
    # And the plan fits: what it planned ends, with the margin, by the
    # deadline. (The record is taken just after the plan; the second covers
    # this host's scheduling between the two, the overrun it replaces was 3 s.)
    assert (
        result["planned_at"] + result["planned_seconds"] + WORKER_BUDGET_MARGIN_SECONDS
        <= result["deadline"] + 1.0
    ), result
