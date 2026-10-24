import subprocess
import sys
import time
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent.diagnostic_process import DiagnosticProcessRunner
from gpu_fault.node_agent.operations.collection_budget import (
    bounded_read,
    collection_lease,
)


@pytest.mark.parametrize("case", ["timeout", "bytes", "valid", "nonzero"])
def test_real_owned_diagnostic_subprocess_is_time_and_output_bounded(case):
    runner = DiagnosticProcessRunner(kill_grace_seconds=0.1)
    scripts = {
        "timeout": "import time; time.sleep(20)",
        "bytes": "print('x' * 2000000)",
        "valid": "print('sample')",
        "nonzero": "raise SystemExit(3)",
    }
    started = time.monotonic()
    if case in {"timeout", "bytes"}:
        with pytest.raises(
            subprocess.TimeoutExpired if case == "timeout" else ValueError
        ):
            runner([sys.executable, "-c", scripts[case]], timeout=0.3)
    else:
        result = runner([sys.executable, "-c", scripts[case]], timeout=2)
        assert result.returncode == (3 if case == "nonzero" else 0), result
    assert time.monotonic() - started < 3, (
        "owned child must not block collector completion"
    )


def test_blocked_reads_hold_their_slot_but_return_incomplete_without_thread_growth():
    release = Event()
    completed = [Event() for _ in range(4)]

    def reader(index):
        release.wait(3)
        completed[index].set()

    try:
        for index in range(4):
            with pytest.raises(TimeoutError, match="timed out"):
                bounded_read(lambda index=index: reader(index), 0.01)
        with pytest.raises(TimeoutError, match="slots exhausted"):
            bounded_read(lambda: "must not be started", 0.01)
    finally:
        release.set()
        assert all(event.wait(2) for event in completed), (
            "owned reader threads must exit"
        )


def test_per_attempt_frequency_is_persistent_and_node_concurrency_is_one(tmp_path):
    agent = SimpleNamespace(diagnostic_output_dir=tmp_path)
    command = SimpleNamespace(
        operation=Op.COLLECT_HUNG_SUMMARY,
        parameters={"job_id": "job", "attempt_id": "attempt"},
    )
    with collection_lease(agent, command) as refused:
        assert refused is None
        with collection_lease(agent, command) as overlapping:
            assert overlapping == "NODE_COLLECTION_BUSY"
    with collection_lease(agent, command) as second:
        assert second is None, second
    with collection_lease(
        SimpleNamespace(diagnostic_output_dir=tmp_path), command
    ) as third:
        assert third == "TASK_COLLECTION_RATE_LIMIT", third
    command.parameters["attempt_id"] = "new-attempt"
    with collection_lease(agent, command) as fresh:
        assert fresh is None, fresh
