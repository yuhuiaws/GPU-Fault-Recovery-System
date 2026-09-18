from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from gpu_fault.node_agent.operations import hung_triage
from tests.node_agent._cov95_runtime_hung import COMPUTE_QUERY, GPU_QUERY, process
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import command, envelope
from tests.regional._cov95_runtime_support import Clock
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("failure", ["exit", "io", "timeout", "invalid-rows"])
def test_failed_compute_inventory_is_reported_as_missing_triage_evidence(
    node_factory: Callable[..., NodeActionExecutor], failure: str
) -> None:
    calls = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(argv)
        if COMPUTE_QUERY in argv:
            if failure == "io":
                raise OSError("synthetic inaccessible inventory")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 3)
            return subprocess.CompletedProcess(
                argv,
                1 if failure == "exit" else 0,
                "invalid\npid, GPU-a, python\n100, GPU-a\n",
                "synthetic query refusal",
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        runner=runner,
        python_stack_tool="",
    )
    result = agent.execute(envelope(command(WorkflowOperation.COLLECT_HUNG_TRIAGE)))
    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["ranks"] == []
    assert len(result.details["errors"]) == (0 if failure == "invalid-rows" else 1)
    assert result.details["read_only"] is True
    assert [argv[1] for argv in calls] == [COMPUTE_QUERY, GPU_QUERY]


@pytest.mark.parametrize(
    "rank_mapping", [[], {"node-a": {"100": 8}}, {"node-a": {"100": []}}]
)
@pytest.mark.parametrize("gpu_reply", ["exit", "io", "malformed", "valid"])
def test_incomplete_rank_and_gpu_samples_do_not_invent_health_or_hide_other_ranks(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    rank_mapping: Any,
    gpu_reply: str,
) -> None:
    first = process(tmp_path / "proc", 100)
    second = process(tmp_path / "proc", 101)
    (first / "environ").write_bytes(b"RANK=invalid\0LOCAL_RANK=invalid\0")
    (second / "environ").write_bytes(b"RANK=9\0LOCAL_RANK=1\0")
    tasks = first / "task"
    for tid, text in ((100, ""), (101, "101 (worker) X 0"), (102, "102 (worker) D 0")):
        target = tasks / str(tid)
        target.mkdir(parents=True)
        (target / "stat").write_text(text)
    (tasks / "103").mkdir()
    (first / "stat").write_text("malformed stat")

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        if COMPUTE_QUERY in argv:
            return subprocess.CompletedProcess(
                argv, 0, "100, GPU-a, python\n101, GPU-b, python\n", ""
            )
        if gpu_reply == "io":
            raise OSError("synthetic GPU query failure")
        return subprocess.CompletedProcess(
            argv,
            2 if gpu_reply == "exit" else 0,
            "truncated\nGPU-a, unavailable, 0, 0x0\n"
            + ("GPU-b, 12, 3, 0x1\n" if gpu_reply == "valid" else ""),
            "",
        )

    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        runner=runner,
        python_stack_tool="",
    )
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                gpu_uuids=[],
                parameters={"rank_by_pid_by_node": rank_mapping},
            )
        )
    )
    assert result.status is NodeActionStatus.SUCCEEDED
    a, b = result.details["ranks"]
    assert (a["pid"], a["rank"], a["local_rank"]) == (
        100,
        8 if rank_mapping == {"node-a": {"100": 8}} else None,
        None,
    )
    assert a["proc"]["thread_states"] == {"R": 0, "S": 0, "D": 1}
    assert a["proc"]["cpu_ticks_delta"] == 0
    assert a["gpu"] == {}
    assert (b["pid"], b["rank"], b["local_rank"]) == (101, 9, 1)
    assert b["gpu"] == (
        {
            "utilization_gpu_percent": 12.0,
            "utilization_memory_percent": 3.0,
            "clocks_throttle_reasons_active": "0x1",
        }
        if gpu_reply == "valid"
        else {}
    )


@pytest.mark.parametrize("failure", ["capture", "write", "vanished", "none"])
def test_stack_failure_is_isolated_from_sibling_rank_evidence(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    for pid in (100, 101):
        process(tmp_path / "proc", pid)
    real_write = Path.write_text
    real_is_file = Path.is_file

    def write(path: Path, *args: Any, **kwargs: Any) -> int:
        if failure == "write" and path.name.startswith("python-stack-100-"):
            raise PermissionError("synthetic stack destination failure")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: (
            False
            if failure == "vanished" and path.name.startswith("python-stack-100-")
            else real_is_file(path)
        ),
    )
    stack_text = (
        "Process 100: python train.py\nPython v3.12\nThread 1\n\n"
        "(only annotation)\n"
        + "\n".join(
            f"frame_{index} (/training/train.py:{index})" for index in range(16)
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        if COMPUTE_QUERY in argv:
            text = "100, GPU-a, python\n101, GPU-a, python\n"
        elif argv[0] == "/unit/py-spy":
            if argv[-1] == "100" and failure == "capture":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            text = stack_text if argv[-1] == "100" else "native_frame:12\n"
        else:
            text = ""
        return subprocess.CompletedProcess(argv, 0, text, "")

    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        python_stack_tool="/unit/py-spy",
        runner=runner,
    )
    result = agent.execute(envelope(command(WorkflowOperation.COLLECT_HUNG_TRIAGE)))
    assert result.status is NodeActionStatus.SUCCEEDED
    a, b = result.details["ranks"]
    assert b["python_stack"]["frames"] == ["native_frame"]
    assert b["python_stack"]["collective_frames"] is False
    if failure in {"capture", "write"}:
        assert "error" in a["python_stack"]
    elif failure == "vanished":
        assert a["python_stack"]["signature"] is None
        assert a["python_stack"]["frames"] == []
    else:
        assert len(a["python_stack"]["frames"]) == 12
        assert a["python_stack"]["frames"][0] == "/training/train.py:frame_0"
    assert result.details["read_only"] is True


@pytest.mark.parametrize("stack_enabled", [False, True])
def test_expired_initial_triage_budget_does_not_add_an_optional_sample_sleep(
    node_factory: Callable[..., NodeActionExecutor],
    monkeypatch: pytest.MonkeyPatch,
    stack_enabled: bool,
) -> None:
    clock = Clock(step=0)
    monkeypatch.setattr(hung_triage, "time", clock)
    sleeps = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        if COMPUTE_QUERY in argv:
            clock.value = 3
            return subprocess.CompletedProcess(argv, 0, "100, GPU-a, python\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        runner=runner,
        sleep=sleeps.append,
        python_stack_tool="/unit/py-spy" if stack_enabled else "",
    )
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                parameters={"triage_timeout_seconds": 2},
            )
        )
    )
    assert result.status is NodeActionStatus.SUCCEEDED
    assert len(result.details["ranks"]) == 1
    assert sleeps == []
    assert result.details["elapsed_seconds"] == 3
