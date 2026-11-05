"""The observation supervisor when a worker, its output or the slot misbehave.

The supervisor runs every read-only diagnostic of the node in one worker tree
behind one slot, and its answer is the only evidence the control plane gets:
whatever went wrong must come back as a result that names the cause, the
component and the stage, and the slot must be free again exactly when the tree
has gone. ``test_observation_slot_contention`` and
``test_observation_final_result`` cover contention and deadlines with real
workers; these tests cover the rest: output over the result budget or not a
result, a worker without an output pipe, a request refused for its size after
it waited for the slot, a slot held by an activity that already left, and a
tree that ignores SIGTERM -- abandoned at its deadline, preempted in vain, and
holding the slot until it really exits.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.node_agent import observation_supervisor
from gpu_fault.node_agent.observation_limits import (
    OBSERVATION_DEADLINE_VARIABLE,
    OBSERVATION_REQUEST_BUDGET_BYTES,
    OBSERVATION_RESULT_BUDGET_BYTES,
    WORKER_NOT_STARTED,
    WORKER_OUTPUT,
)
from gpu_fault.node_agent.observation_supervisor import (
    SLOT_BUSY,
    ObservationSupervisor,
    request_overflow,
)
from tests.node_agent._observation_deadline import DeadlineClock, WatchedSlot

#: Long enough that no host reaches it on its own; the tests that need the
#: deadline to pass move the supervisor's clock (``DeadlineClock``).
BUDGET_SECONDS = 120.0
STAND_IN = "tests.node_agent._cov95_observation_stand_in"


def use_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        observation_supervisor, "observation_guard", SimpleNamespace(__name__=STAND_IN)
    )


def slot_is_free(supervisor: ObservationSupervisor, timeout: float = 0.0) -> bool:
    acquired = (
        supervisor.slot.acquire(timeout=timeout)
        if timeout
        else supervisor.slot.acquire(blocking=False)
    )
    if acquired:
        supervisor.slot.release()
    return bool(acquired)


def join_reapers() -> None:
    for thread in threading.enumerate():
        if thread.name == "observation-reaper":
            thread.join(timeout=30)
            assert not thread.is_alive(), "the reaper outlived its worker tree"


def test_an_oversized_request_without_parameters_names_only_its_size() -> None:
    message = request_overflow({"operation": "summary", "command": "not-a-command"}, 9)
    assert message == (
        "observation input exceeds budget: "
        f"9 > {OBSERVATION_REQUEST_BUDGET_BYTES} bytes"
    ), message
    assert "largest parameters" not in request_overflow(
        {"command": {"parameters": {}}}, 9
    ), "an empty parameter map has nothing to weigh"


def test_a_slot_held_by_an_activity_that_already_left_is_refused_as_such() -> None:
    """The holder record is cleared just before the slot is given back; a
    command that arrives in between is refused without a holder to name."""

    supervisor = ObservationSupervisor()
    assert supervisor.slot.acquire(blocking=False), "the test holds the slot"
    released: list[str] = []
    try:
        result = supervisor.run(
            {"operation": "triage"},
            5.0,
            release=lambda: released.append("triage"),
            slot_wait=0.05,
        )
    finally:
        supervisor.slot.release()

    assert result["status"] == SLOT_BUSY, result
    assert result["error"].startswith(
        "observation worker slot held by an activity that already left; waited "
    ), result["error"]
    assert result["slot_holder"] is None, result
    assert result["worker_started"] is False, result
    assert result["failure_stage"] == WORKER_NOT_STARTED, result
    assert result["slot_wait_allowed_seconds"] == 0.05, result
    assert released == ["triage"], "a refused command still releases its admission"
    assert supervisor.last_result is result, "the refusal is the last result"


def test_a_request_refused_for_its_size_after_a_slot_wait_records_the_wait() -> None:
    supervisor = ObservationSupervisor()
    slot = WatchedSlot(supervisor.slot)
    supervisor.slot = slot  # type: ignore[assignment]
    assert slot.acquire(blocking=False), "the test holds the slot"

    def give_back_after_the_wait_began() -> None:
        slot.waiting.wait(30)
        time.sleep(0.05)
        slot.release()

    holder = threading.Thread(target=give_back_after_the_wait_began)
    holder.start()
    try:
        result = supervisor.run(
            {
                "operation": "bundle",
                "settings": {"blob": "x" * OBSERVATION_REQUEST_BUDGET_BYTES},
            },
            30.0,
            slot_wait=20.0,
        )
    finally:
        holder.join(timeout=30)

    assert result["status"] == "UNAVAILABLE", result
    assert result["error"].startswith(
        "ValueError: observation input exceeds budget: "
    ), result["error"]
    assert "largest parameters" not in result["error"], "the request has no command"
    assert result["slot_wait_seconds"] >= 0.04, result
    assert result["failure_stage"] == WORKER_NOT_STARTED, result
    assert result["worker_started"] is False, result
    assert supervisor.holder is None, supervisor.holder
    assert slot_is_free(supervisor), "the slot is given back with the refusal"


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("oversize", "ValueError: observation output exceeds budget"),
        ("not_a_result", "ValueError: invalid observation result"),
    ],
)
def test_worker_output_the_supervisor_cannot_take_is_refused_as_worker_output(
    monkeypatch: pytest.MonkeyPatch, mode: str, error: str
) -> None:
    use_stand_in(monkeypatch)
    supervisor = ObservationSupervisor()
    released: list[str] = []

    result = supervisor.run(
        {
            "operation": "summary",
            "stand_in": mode,
            "size": OBSERVATION_RESULT_BUDGET_BYTES + 64 * 1024,
        },
        BUDGET_SECONDS,
        release=lambda: released.append(mode),
    )

    assert result["status"] == "UNAVAILABLE", result
    assert result["error"] == error, result
    assert result["failed_component"] == "observation_supervisor", result
    assert result["failure_stage"] == WORKER_OUTPUT, result
    assert result["worker_started"] is True, result
    assert result["evidence_complete"] is False, result
    assert "worker_tree_running" not in result, "the worker tree was stopped"
    assert released == [mode], released
    assert supervisor.holder is None, supervisor.holder
    assert slot_is_free(supervisor), "the slot is free once the tree has gone"


class PipelessProcess:
    """A started guard whose stdout pipe the platform did not create."""

    pid = 0
    stdout = None

    def __init__(self) -> None:
        self.waited: list[float | None] = []

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        self.waited.append(timeout)
        return 0

    def terminate(self) -> None:
        raise AssertionError("an exited guard is never signalled")


def test_a_worker_without_an_output_pipe_fails_loudly_and_frees_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[dict[str, Any]] = []
    process = PipelessProcess()

    def popen(arguments: list[str], **kwargs: Any) -> PipelessProcess:
        started.append({"arguments": arguments, **kwargs})
        return process

    monkeypatch.setattr(
        observation_supervisor,
        "subprocess",
        SimpleNamespace(
            Popen=popen,
            PIPE=subprocess.PIPE,
            DEVNULL=subprocess.DEVNULL,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEY", "never handed to a worker")
    supervisor = ObservationSupervisor()
    released: list[str] = []

    with pytest.raises(RuntimeError, match="observation output pipe was not created"):
        supervisor.run(
            {"operation": "summary"}, 10.0, release=lambda: released.append("summary")
        )

    [spawn] = started
    assert spawn["arguments"] == [
        sys.executable,
        "-m",
        observation_supervisor.observation_guard.__name__,
    ], spawn
    assert spawn["stdout"] == subprocess.PIPE and spawn["start_new_session"] is True
    assert "GPU_FAULT_NODE_ACTION_KEY" not in spawn["env"], "secrets stay behind"
    assert float(spawn["env"][OBSERVATION_DEADLINE_VARIABLE]) > 0, spawn["env"]
    assert process.waited == [0.2], "the exited guard is reaped, never killed"
    assert released == ["summary"], released
    assert supervisor.holder is None, supervisor.holder
    assert slot_is_free(supervisor), "the slot is free after the failure"


def test_a_tree_that_ignores_sigterm_keeps_the_slot_until_it_really_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abandoned at its deadline, the tree still holds the slot: a triage that
    preempts it in vain is refused naming both, and only the tree's exit
    frees the slot (the reaper), with the sample's admission released then."""

    use_stand_in(monkeypatch)
    started = tmp_path / "started"
    DeadlineClock(BUDGET_SECONDS, when=started.exists).install(monkeypatch)
    supervisor = ObservationSupervisor()
    released: list[str] = []
    pid = None
    try:
        result = supervisor.run(
            {"operation": "sample", "stand_in": "deaf", "started": str(started)},
            BUDGET_SECONDS,
            release=lambda: released.append("sample"),
            preemptible=True,
        )
        holder = supervisor.holder
        assert holder is not None, "the abandoned tree is still the holder"
        pid = holder["process"].pid
        assert started.exists(), "the deadline passed once the stand-in ran"

        assert result["status"] == "PARTIAL_TIMEOUT", result
        assert result["worker_tree_running"] is True, result
        assert result["evidence_complete"] is False, result
        assert result["worker_returncode"] is None, result
        assert holder["abandoned"] is True, holder
        assert holder["operation"] == "sample", holder
        assert released == [], "the sample's admission is held with the slot"
        assert not slot_is_free(supervisor), "the running tree keeps the slot"

        refusal = supervisor.run(
            {"operation": "triage"},
            BUDGET_SECONDS,
            release=lambda: released.append("triage"),
            slot_wait=0.05,
        )

        assert refusal["status"] == SLOT_BUSY, refusal
        assert "observation worker slot held by sample for " in refusal["error"]
        assert (
            refusal["error"]
            .split("; waited ")[0]
            .endswith(
                " s (preempted, not yet exited) (abandoned by its caller at its deadline)"
            )
        ), refusal["error"]
        assert refusal["slot_holder"]["preempted"] is True, refusal
        assert refusal["slot_holder"]["abandoned"] is True, refusal
        assert holder["preempted_by"] == "triage", holder
        assert released == ["triage"], released
    finally:
        # The stand-in ignores SIGTERM for good: kill it even when an assertion
        # failed before its holder record was read.
        if pid is None and started.exists():
            text = started.read_text().strip()
            pid = int(text) if text.isdecimal() else None
        if pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)

    assert slot_is_free(supervisor, timeout=30), "the reaper frees the slot"
    join_reapers()
    assert supervisor.holder is None, "the reaper clears its own holder"
    assert released == ["triage", "sample"], released
