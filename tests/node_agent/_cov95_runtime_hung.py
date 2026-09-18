from __future__ import annotations

import json
import tarfile
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from tests.node_agent._support import command, envelope

COMPUTE_QUERY = "--query-compute-apps=pid,gpu_uuid,process_name"
GPU_QUERY = (
    "--query-gpu=uuid,utilization.gpu,utilization.memory,clocks_throttle_reasons.active"
)


def process(
    root: Path, pid: int, *, cgroup: str = "/training/job", name: str = "python"
) -> Path:
    directory = root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cgroup").write_text(f"0::{cgroup}\ninvalid\n0::/\n")
    (directory / "comm").write_text(name)
    (directory / "cmdline").write_bytes(f"{name}\0train.py\0".encode())
    (directory / "status").write_text("State:\tS\n")
    return directory


class BundleRunner:
    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> CompletedProcess:
        self.calls.append(argv)
        if COMPUTE_QUERY in argv:
            return CompletedProcess(argv, self.returncode, self.stdout, self.stderr)
        if argv[0] == "timeout":
            output = Path(argv[argv.index("-o") + 1])
            output.with_suffix(".trace").write_text("owned fake strace sample")
            return CompletedProcess(argv, 124, "", "")
        return CompletedProcess(argv, 0, "owned fake diagnostic output", "")


def capture_bundle(
    agent: NodeActionExecutor, **parameters: Any
) -> tuple[dict[str, Any], dict[str, bytes]]:
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                parameters={
                    "capture_process_state": True,
                    "strace_sample_count": 1,
                    "strace_duration_seconds": 1,
                    "strace_sample_interval_seconds": 0,
                    "pyspy_sample_interval_seconds": 0,
                    **parameters,
                },
            )
        )
    )
    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    path = Path(result.details["evidence_ref"].removeprefix("file://"))
    contents = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None, member.name
                contents[member.name.removeprefix("diagnostics/")] = stream.read()
    return json.loads(contents["manifest.json"]), contents
