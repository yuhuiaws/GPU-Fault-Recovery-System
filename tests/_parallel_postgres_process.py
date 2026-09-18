"""Real-process fixtures for local sharding; all PostgreSQL Docker calls are mocked."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.execution import deployment_deadline, recovery_active, run_command
from gpu_fault.admin.process_supervisor import (
    ProcessSupervisionLost,
    ensure_supervision_safe,
)
from gpu_fault.admin.release_postgres import isolated_postgres_allocation
from scripts import postgres_shard_worker as worker
from scripts import run_postgres_shards as runner
from scripts.postgres_shard_receipts import PostgresShardError, read_private_json
from tests.admin import _release_postgres_support as postgres_double
from tests.admin._release_postgres_support import CID, FakePostgresDocker
from tools.pytest_result_identity import source_identity

ROOT = Path(__file__).resolve().parents[1]
CASES = "tests/_parallel_postgres_cases.py"
LIFECYCLE_CASES = "tests/_parallel_postgres_lifecycle_cases.py"


class CleanupProcessDocker(FakePostgresDocker):
    def __init__(self, state: Path) -> None:
        super().__init__()
        self.state = state
        self.backend = state / "fake-backend.json"
        if self.backend.exists():
            self.containers = read_private_json(self.backend)

    def run(self, arguments: Sequence[str], **options: Any) -> str:
        if recovery_active():
            # Real nested owners must be allowed to start and finish during cleanup.
            write_json_atomic(self.state / "cleanup-started.json", {"pid": os.getpid()})
            run_command(
                [sys.executable, "-c", "import time; time.sleep(0.05)"],
                timeout_seconds=5,
            )
        result = super().run(arguments, **options)
        write_json_atomic(self.backend, self.containers)
        return result


def lifecycle_worker(mode: str, state: Path) -> int:
    postgres_double.CID = hashlib.sha256(state.name.encode()).hexdigest()
    with pytest.MonkeyPatch.context() as patch, worker.cancellation_scope():
        patch.setattr(worker, "CommandRunner", lambda: CleanupProcessDocker(state))
        return worker.run_worker(
            state=state,
            tests=(LIFECYCLE_CASES,),
            workers=8,
            index=int(state.name.removeprefix("shard-")),
            durations=0,
            identity=os.environ["PARALLEL_POSTGRES_IDENTITY"],
        )


def lifecycle_gate(mode: str, state: Path) -> int:
    jobs = []
    identity = source_identity(ROOT)
    for index in range(8):
        directory = state / f"shard-{index}"
        directory.mkdir(mode=0o700)
        command = (
            sys.executable,
            "-B",
            "-m",
            "tests._parallel_postgres_process",
            "lifecycle-worker",
            mode,
            str(directory),
        )
        environment = {
            **os.environ,
            "PARALLEL_POSTGRES_STATE": str(directory),
            "PARALLEL_POSTGRES_MODE": mode,
            "PARALLEL_POSTGRES_IDENTITY": identity,
        }
        jobs.append(runner.ShardJob(index, directory, command, environment))
    write_json_atomic(state / "group.json", {"pid": os.getpid()})
    failure = "unexpected-success"
    try:
        with (
            deployment_deadline(
                "nested local sharding regression", 35, recovery_seconds=10
            ),
            worker.cancellation_scope(request_only=True) as interrupted,
        ):
            try:
                runner.run_jobs(
                    jobs,
                    identity=identity,
                    tests=(LIFECYCLE_CASES,),
                    stress=("", ""),
                    started_after=datetime.now(timezone.utc),
                    interrupted=interrupted,
                )
            finally:
                for job in jobs:
                    worker.reconcile_allocation(
                        job.state, CleanupProcessDocker(job.state)
                    )
    except (Exception, KeyboardInterrupt, ProcessSupervisionLost) as error:
        failure = type(error).__name__
    started, stopped, cleaned = 0, 0, 0
    ownership = []
    for job in jobs:
        path = job.state / "test-running.json"
        if path.exists():
            pids = read_private_json(path)
            started += 1
            stopped += int(
                all(not Path(f"/proc/{pid}").exists() for pid in pids.values())
            )
        backend = job.state / "fake-backend.json"
        cleaned += int(backend.exists() and read_private_json(backend) == {})
        path = job.state / "release-postgres/ownership.json"
        if path.exists():
            value = read_private_json(path)
            ownership.append(
                {"phase": value["phase"], "lost": value["supervision_lost"]}
            )
    try:
        ensure_supervision_safe()
    except ProcessSupervisionLost:
        safe = False
    else:
        safe = True
    write_json_atomic(
        state / "result.json",
        {
            "failure": failure,
            "started": started,
            "stopped": stopped,
            "cleaned": cleaned,
            "ownership": ownership,
            "supervision_safe": safe,
        },
    )
    return 0


def wait_for(path: Path) -> None:
    until = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= until:
            raise RuntimeError("fixture did not become ready")
        time.sleep(0.02)


def leaf(mode: str, state: Path) -> int:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    write_json_atomic(state / "pids.json", {"leaf": os.getpid(), "child": child.pid})
    wait_for(state.parent / "shard-0/pids.json")
    wait_for(state.parent / "shard-1/pids.json")
    if mode == "failure":
        worker.record_failure(state / worker.FAILURE_FILE)
        return 1
    if mode == "exit":
        os.kill(os.getpid(), signal.SIGKILL)
    until = time.monotonic() + 30
    while not worker.cancellation_requested(state) and time.monotonic() < until:
        time.sleep(0.02)
    return 0


def group(mode: str, state: Path) -> int:
    jobs = []
    for index in range(2):
        directory = state / f"shard-{index}"
        directory.mkdir(mode=0o700)
        command = (
            sys.executable,
            "-B",
            "-m",
            "tests._parallel_postgres_process",
            "leaf",
            mode if index == 0 else "sleep",
            str(directory),
        )
        jobs.append(runner.ShardJob(index, directory, command, dict(os.environ)))
    write_json_atomic(state / "group.json", {"pid": os.getpid()})
    status = "unexpected-success"
    try:
        with (
            deployment_deadline("local test process group", 20, recovery_seconds=10),
            worker.cancellation_scope(request_only=True) as interrupted,
        ):
            runner.run_jobs(
                jobs,
                identity="a" * 64,
                tests=(CASES,),
                stress=("", ""),
                started_after=datetime.now(timezone.utc),
                interrupted=interrupted,
            )
    except KeyboardInterrupt:
        status = "interrupted"
    except PostgresShardError:
        status = "failed"
    pids = [
        pid
        for job in jobs
        for pid in read_private_json(job.state / "pids.json").values()
    ]
    stopped = all(not Path(f"/proc/{pid}").exists() for pid in pids)
    if stopped:
        runner.reconcile_jobs(jobs)
    write_json_atomic(state / "result.json", {"status": status, "stopped": stopped})
    return 0 if stopped and status != "unexpected-success" else 1


def main() -> int:
    os.umask(0o077)
    operation, mode, path = sys.argv[1:4]
    state = Path(path)
    if operation == "pytest":
        command = worker.serial_test_command(
            sys.executable, state=state, tests=(CASES,), durations=3
        )
        return int(pytest.main(command[4:]))
    if operation == "leaf":
        return leaf(mode, state)
    if operation == "group":
        return group(mode, state)
    if operation == "lifecycle-worker":
        return lifecycle_worker(mode, state)
    if operation == "lifecycle-gate":
        return lifecycle_gate(mode, state)
    if operation == "lifecycle-wrapper":
        result = run_command(
            [
                sys.executable,
                "-B",
                "-m",
                "tests._parallel_postgres_process",
                "lifecycle-gate",
                mode,
                str(state),
            ],
            timeout_seconds=50,
        )
        return int(result.returncode)
    if operation == "allocate-and-exit":
        docker = FakePostgresDocker()
        with isolated_postgres_allocation(
            docker, repository_root=ROOT, state_dir=state
        ):
            write_json_atomic(state / "fake-container.json", docker.containers[CID])
            os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("unknown fixture operation")


if __name__ == "__main__":
    raise SystemExit(main())
