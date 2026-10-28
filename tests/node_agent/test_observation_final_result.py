"""A final result is final once its worker tree exits, however long that takes.

After the worker's final line and the end of its output, the supervisor waited
0.1 s for the worker tree to exit and otherwise handed back
``PARTIAL_TIMEOUT`` with ``evidence_complete=false`` -- for a result that was
complete, with most of the deadline left. With 48 busy loops on the test's CPU
the exiting process stayed off the CPU for longer than that between closing its
stdout and being reaped: the bundle preemption test
(``test_a_bundle_preempts_the_passive_sample_holding_the_shared_slot``) raised
``ObservationIncomplete: PARTIAL_TIMEOUT`` 20 s into a 120 s budget, and the
slow-start triage came back ``PARTIAL_TIMEOUT`` with 100 s left. On a GPU node a
training job, or the Agent's own cgroup quota, holds the guard off the CPU the
same way. The supervisor now waits for an exiting tree up to the deadline
(b214).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from gpu_fault.node_agent import observation_supervisor
from gpu_fault.node_agent.observation_supervisor import ObservationSupervisor
from tests.node_agent import _exiting_observation_worker as exiting
from tests.node_agent._observation_deadline import DeadlineClock
from tests.node_agent.test_communication_monitor import (
    REAL_WORKER_BUDGET_SECONDS,
    use_imported_package,
)


def _exiting_worker(monkeypatch) -> None:
    use_imported_package(monkeypatch)
    monkeypatch.setattr(
        observation_supervisor,
        "observation_guard",
        SimpleNamespace(__name__="tests.node_agent._exiting_observation_worker"),
    )


def test_a_final_result_is_not_partial_while_its_worker_tree_exits(monkeypatch):
    """The tree leaves 1 s after the end of its output (ten times the 0.1 s the
    supervisor used to allow): the answer is the worker's, complete."""

    _exiting_worker(monkeypatch)
    result = ObservationSupervisor().run(
        {"operation": "summary", "settings": {}, "exit_after": 1.0},
        REAL_WORKER_BUDGET_SECONDS,
    )
    assert result == exiting.RESULT, result


def test_an_exiting_worker_tree_is_waited_for_no_longer_than_the_deadline(
    tmp_path, monkeypatch
):
    """A tree that never leaves after its final line: at the deadline the
    supervisor hands back what it read, marked incomplete, as before."""

    _exiting_worker(monkeypatch)
    checkpointed = tmp_path / "checkpointed"
    clock = DeadlineClock(REAL_WORKER_BUDGET_SECONDS, when=checkpointed.exists)
    clock.install(monkeypatch)
    result = ObservationSupervisor().run(
        {"operation": "summary", "settings": {}, "checkpointed": str(checkpointed)},
        REAL_WORKER_BUDGET_SECONDS,
    )
    returned = time.monotonic()

    assert clock.expired_at is not None, ("the worker never answered", result)
    assert returned - clock.expired_at < 2.5, (returned - clock.expired_at, result)
    assert result["status"] == "PARTIAL_TIMEOUT", result
    assert result["evidence_complete"] is False, result
    assert result["worker_returncode"] is None, result
    assert result["read_only"] is True, "what the worker answered is handed back"
