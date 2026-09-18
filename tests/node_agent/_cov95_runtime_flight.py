from __future__ import annotations

import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionResult
from gpu_fault.node_agent.operations import flight_recorder, hung_triage
from tests.node_agent._support import command, envelope
from tests.regional._cov95_runtime_support import Clock


def collect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    node_factory: Callable[..., NodeActionExecutor],
    *,
    payload: Any = None,
    mode: str = "json",
) -> tuple[NodeActionResult, list[bytes]]:
    clock = Clock(step=0.1)
    monkeypatch.setattr(flight_recorder, "time", clock)
    monkeypatch.setattr(hung_triage, "time", clock)
    proc = tmp_path / "proc" / "100"
    root = proc / "root" / "tmp"
    root.mkdir(parents=True)
    base = "/tmp/pipe_{rank}" if mode == "template" else "/tmp/pipe_"
    pipe = root / ("pipe_7" if mode == "template" else "pipe_7.pipe")
    if mode == "traversal":
        base = "/../outside"
    if mode == "nonfifo":
        pipe.write_bytes(b"")
    else:
        os.mkfifo(pipe)
    if mode != "no-environment":
        (proc / "environ").write_bytes(
            (
                f"RANK=7\0LOCAL_RANK=0\0TORCH_NCCL_DEBUG_INFO_PIPE_FILE={base}\0"
                "TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/dump_\0IGNORED\0"
            ).encode()
        )
    dump = root / "dump_7.json"
    content = (
        b"cbuiltins\nstr\n."
        if mode == "global-pickle"
        else b"\x80invalid-pickle"
        if mode == "corrupt"
        else pickle.dumps(payload, protocol=2)
        if mode == "pickle"
        else json.dumps(payload).encode()
    )
    if mode == "unchanged":
        dump.write_bytes(content)
        os.utime(dump, ns=(1, 1))
    writes = []
    real_open, real_write, real_close = os.open, os.write, os.close
    fake_descriptor = 987654

    def open_pipe(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path) == pipe:
            assert flags == os.O_WRONLY | os.O_NONBLOCK
            if mode == "write-failure":
                raise OSError("synthetic absent FIFO reader")
            return fake_descriptor
        return real_open(path, flags, *args, **kwargs)

    def write_pipe(descriptor: int, data: bytes) -> int:
        if descriptor != fake_descriptor:
            return real_write(descriptor, data)
        writes.append(data)
        if mode not in {"unchanged", "no-dump"}:
            dump.write_bytes(content)
        return len(data)

    monkeypatch.setattr(os, "open", open_pipe)
    monkeypatch.setattr(os, "write", write_pipe)
    monkeypatch.setattr(
        os, "close", lambda fd: None if fd == fake_descriptor else real_close(fd)
    )
    commands = []

    def runner(argv: list[str], **kwargs: Any) -> CompletedProcess:
        commands.append(argv)
        text = (
            "invalid\n100, GPU-a, python\n101, GPU-b, ignored\n"
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv
            else "GPU-a, 0, 0, 0x0\n"
        )
        return CompletedProcess(argv, 0, stdout=text, stderr="")

    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        python_stack_tool="",
        runner=runner,
        sleep=clock.sleep,
    )
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                parameters={"triage_timeout_seconds": 2},
            )
        )
    )
    assert all(
        argv[0] == "nvidia-smi" and "--gpu-reset" not in argv for argv in commands
    ), "triage must use only the fake read-only query transport"
    return result, writes
