"""Run local PostgreSQL tests on independent owned PG16 servers, never CI shards."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from gpu_fault.admin.atomic_json import write_json_atomic  # noqa: E402
from gpu_fault.admin.bootstrap_common import CommandRunner  # noqa: E402
from gpu_fault.admin.execution import (  # noqa: E402
    cleanup_deadline,
    deployment_deadline,
    remaining_timeout,
    run_command,
    run_driver,
)
from gpu_fault.admin.postgres_grant import (  # noqa: E402
    CONTAINER_ID,
    LOCAL_DOCKER_HOST,
    private_directory,
    private_write,
)
from gpu_fault.admin.process_supervisor import (  # noqa: E402
    ProcessSupervisionLost,
    ensure_supervision_safe,
    write_diagnostic,
)
from gpu_fault.admin.release_postgres import IMAGE_ID, POSTGRES_IMAGE  # noqa: E402
from scripts.postgres_shard_receipts import (  # noqa: E402
    PostgresShardError,
    ShardReceipt,
    describe_failure,
    exception_chain,
    read_failure,
    read_private_json,
    read_worker_failure,
    redact_output,
    validate_partition_union,
    validate_shard_receipt,
    validate_targets,
)
from scripts.postgres_shard_worker import (  # noqa: E402
    ALLOCATION_FILE,
    CANCEL_FILE,
    FAILURE_FILE,
    READY_FILE,
    RECEIPT_FILE,
    WORK_SECONDS,
    WORKER_FAILURE_FILE,
    cancellation_scope,
    reconcile_allocation,
    worker_environment,
)

STRESS_ENV = (
    "GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS",
    "GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS",
)
# A failed worker's verdict is followed by its last lines, the shape
# run_release_gates uses for a failing gate (GATE_TAIL_LINES); every byte
# passes the supervised-command redaction and the whole output stays private.
WORKER_TAIL_LINES = 20
WORKER_TAIL_CHARACTERS = 400
WORKER_LOG = "worker.log"
WORKER_LOG_CHARACTERS = 4 * 1024 * 1024


@dataclass(frozen=True)
class ShardJob:
    index: int
    state: Path
    command: tuple[str, ...]
    environment: dict[str, str]


def checked_identity(python: str, environment: Mapping[str, str]) -> str:
    result = run_command(
        [python, "-B", "-m", "scripts.postgres_shard_worker", "--identity-only"],
        environment=environment,
        cwd=ROOT,
        timeout_seconds=120,
    )
    identity: str = result.stdout.strip()
    if result.returncode or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise PostgresShardError(
            "could not bind local PostgreSQL tests to current source"
        )
    return identity


def failure_detail(evidence: Mapping[str, Any]) -> str:
    return (
        f"nodeid={evidence['nodeid'] or '<process>'} "
        f"phase={evidence['phase']} outcome={evidence['outcome']}"
    )


def exit_status(returncode: int) -> str:
    if returncode < 0:
        try:
            return f"was killed by {signal.Signals(-returncode).name}"
        except ValueError:
            pass
    return f"exited with status {returncode}"


def exit_phase(state: Path) -> str:
    """Where a worker stood when it exited, from the files its pytest leaves."""
    if not (state / READY_FILE).is_file():
        return "before pytest started"
    try:
        session = read_private_json(state / RECEIPT_FILE).get("session")
    except PostgresShardError:
        session = None
    if (
        isinstance(session, dict)
        and type(session.get("exitstatus")) is int
        and session["exitstatus"] == 0
    ):
        return "after pytest passed (post-test: allocation exit or cleanup)"
    return "after pytest started, without a passing pytest receipt"


def recorded_cause(state: Path) -> str:
    path = state / WORKER_FAILURE_FILE
    if not path.exists() and not path.is_symlink():
        return "the worker recorded no cause"
    try:
        record = read_worker_failure(path)
    except PostgresShardError:
        return "the worker's failure record is invalid"
    return f"worker phase {record['phase']}: {describe_failure(record['errors'])}"


def keep_worker_output(
    job: ShardJob, result: subprocess.CompletedProcess[str]
) -> tuple[Path | None, str]:
    """Redact a failed worker's stdout and stderr; keep all of it privately."""
    output = "".join(
        text if text.endswith("\n") else text + "\n"
        for text in (result.stdout, result.stderr)
        if text
    )
    redacted = redact_output(output, limit=WORKER_LOG_CHARACTERS)
    path = job.state / WORKER_LOG
    try:
        # Exclusive: a log written with the first failure is never replaced.
        private_write(path, redacted + "\n" if redacted else "")
    except (OSError, ValueError):
        return None, redacted
    return path, redacted


def worker_exit_failure(
    job: ShardJob, result: subprocess.CompletedProcess[str]
) -> PostgresShardError:
    """The worker's phase, its own recorded cause and its last redacted lines."""
    log, redacted = keep_worker_output(job, result)
    failure = job.state / FAILURE_FILE
    if failure.exists() or failure.is_symlink():
        # A pytest failure is named exactly as the running check names it.
        detail = failure_detail(read_failure(failure))
        summary = f"PostgreSQL shard {job.index} failed: {detail}"
    else:
        detail = f"{exit_status(result.returncode)} {exit_phase(job.state)}"
        summary = f"PostgreSQL shard {job.index} {detail}: {recorded_cause(job.state)}"
    write_diagnostic(
        f"postgres-shards: first failure shard={job.index} {detail}\n", final=True
    )
    kept = f"full output: {log}" if log is not None else "full output not kept"
    tail = [
        line
        if len(line) <= WORKER_TAIL_CHARACTERS
        else line[: WORKER_TAIL_CHARACTERS - 3] + "..."
        for line in redacted.splitlines()[-WORKER_TAIL_LINES:]
    ]
    if not tail:
        return PostgresShardError(f"{summary}\nshard {job.index} printed nothing")
    return PostgresShardError(
        "\n".join(
            [
                summary,
                f"last {len(tail)} lines of shard {job.index}, redacted ({kept}):",
                *(f"[shard-{job.index}] {line}" for line in tail),
            ]
        )
    )


def keep_failed_outputs(submitted: Mapping[Future[Any], ShardJob]) -> None:
    """After the drain, keep the output of every worker that exited unsuccessfully."""
    for future, job in submitted.items():
        if future.done() and not future.cancelled() and future.exception() is None:
            result = future.result()
            if result.returncode:
                keep_worker_output(job, result)


def run_jobs(
    jobs: Sequence[ShardJob],
    *,
    identity: str,
    tests: tuple[str, ...],
    stress: tuple[str, str],
    started_after: datetime,
    interrupted: threading.Event | None = None,
) -> list[ShardReceipt]:
    if interrupted is not None and interrupted.is_set():
        raise KeyboardInterrupt
    receipts: list[ShardReceipt] = []
    first_failure: int | None = None
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    started = time.monotonic()
    next_progress = started + 30
    submitted: dict[Future[Any], ShardJob] = {}
    try:
        submitted = {
            pool.submit(
                copy_context().run,
                run_driver,
                job.command,
                env=job.environment,
                cwd=ROOT,
                capture_output=True,
            ): job
            for job in jobs
        }
        pending = dict(submitted)
        while pending:
            ensure_supervision_safe()
            if interrupted is not None and interrupted.is_set():
                raise KeyboardInterrupt
            remaining_timeout(WORK_SECONDS)
            if time.monotonic() >= next_progress:
                write_diagnostic(
                    f"postgres-shards: running={len(pending)} "
                    f"elapsed={int(time.monotonic() - started)}s\n"
                )
                next_progress = time.monotonic() + 30
            for job in pending.values():
                failure = job.state / FAILURE_FILE
                if failure.exists() or failure.is_symlink():
                    evidence = read_failure(failure)
                    # -x finishes a failed test naturally; pytest does not stop on skips.
                    if evidence["outcome"] == "failed":
                        first_failure = job.index
                    detail = failure_detail(evidence)
                    write_diagnostic(
                        f"postgres-shards: first failure shard={job.index} {detail}\n",
                        final=True,
                    )
                    raise PostgresShardError(
                        f"PostgreSQL shard {job.index} failed: {detail}"
                    )
            finished, _running = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in finished:
                job = pending.pop(future)
                result = future.result()
                if result.returncode:
                    raise worker_exit_failure(job, result)
                receipt = validate_shard_receipt(
                    job.state / RECEIPT_FILE,
                    root=ROOT,
                    identity=identity,
                    tests=tests,
                    workers=len(jobs),
                    index=job.index,
                    stress=stress,
                    started_after=started_after,
                )
                if receipts and receipt.discovered != receipts[0].discovered:
                    raise PostgresShardError("PostgreSQL shards disagree on discovery")
                receipts.append(receipt)
                write_diagnostic(
                    f"postgres-shards: shard={job.index} passed "
                    f"tests={len(receipt.durations)} elapsed={receipt.wall_seconds:.2f}s\n"
                )
    except BaseException:
        # Cancelling lifecycle supervisors also kills the cleanup they subsequently
        # spawn. Workers instead stop their pytest leaf, then finish owned cleanup.
        for job in jobs:
            if job.index != first_failure:
                write_json_atomic(job.state / CANCEL_FILE, {"cancelled": True})
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        keep_failed_outputs(submitted)
    return receipts


def validate_independent_allocations(jobs: Sequence[ShardJob]) -> None:
    allocations = [read_private_json(job.state / ALLOCATION_FILE) for job in jobs]
    for item in allocations:
        if (
            type(item.get("schema_version")) is not int
            or item["schema_version"] != 1
            or not isinstance(item.get("container_id"), str)
            or CONTAINER_ID.fullmatch(item["container_id"]) is None
            or not isinstance(item.get("owner"), str)
            or re.fullmatch(r"[0-9a-f]{32}", item["owner"]) is None
            or item.get("name") != "gpu-fault-release-postgres-" + item["owner"]
            or not isinstance(item.get("image_id"), str)
            or IMAGE_ID.fullmatch(item["image_id"]) is None
            or item.get("image_reference") != POSTGRES_IMAGE
            or item.get("docker_host") != LOCAL_DOCKER_HOST
            or type(item.get("port")) is not int
            or not 1 <= item["port"] <= 65535
            or type(item.get("directory_inode")) is not int
            or type(item.get("directory_device")) is not int
            or item.get("create_attempted") is not True
            or item.get("supervision_lost") is not False
            or item.get("phase") != "READY"
        ):
            raise PostgresShardError("PostgreSQL allocation receipt is invalid")
    for field in ("container_id", "owner", "port", "directory_inode"):
        values = [item.get(field) for item in allocations]
        if any(value is None for value in values) or len(set(values)) != len(jobs):
            raise PostgresShardError(
                "PostgreSQL shards did not use independent allocations"
            )
    if len({item["image_id"] for item in allocations}) != 1:
        raise PostgresShardError("PostgreSQL shard allocation identity is inconsistent")


def reconcile_jobs(jobs: Sequence[ShardJob]) -> None:
    failures = False
    with cleanup_deadline("local PostgreSQL shard group cleanup", 120):
        for job in jobs:
            try:
                reconcile_allocation(job.state, CommandRunner())
            except ProcessSupervisionLost:
                raise
            except Exception:
                failures = True
    if failures:
        raise PostgresShardError(
            "one or more PostgreSQL shard cleanups are unconfirmed"
        )


def run_postgres_shards(
    python: str, *, workers: int, durations: int, tests: Sequence[str]
) -> Path:
    if type(workers) is not int or not 1 <= workers <= 16:
        raise PostgresShardError("PostgreSQL shard workers must be within 1..16")
    if type(durations) is not int or durations < 0:
        raise PostgresShardError("PostgreSQL durations must be a nonnegative integer")
    selected = validate_targets(ROOT, tests)
    base = worker_environment(os.environ, root=ROOT)
    stress = (base.get(STRESS_ENV[0], ""), base.get(STRESS_ENV[1], ""))
    jobs: list[ShardJob] = []
    directory: Path | None = None
    run_fields: dict[str, object] = {"kind": "local-postgres-shards"}
    with (
        deployment_deadline(
            "local PostgreSQL test shards", WORK_SECONDS, recovery_seconds=120
        ),
        cancellation_scope(request_only=True) as interrupted,
    ):
        try:
            directory = Path(
                tempfile.mkdtemp(prefix="gpu-fault-postgres-shards-")
            ).resolve()
            private_directory(directory)
            if directory.is_relative_to(ROOT):
                directory.rmdir()
                directory = None
                raise PostgresShardError(
                    "PostgreSQL shard state must stay outside the checkout"
                )
            home = directory / "home"
            home.mkdir(mode=0o700)
            base["HOME"] = str(home)
            identity = checked_identity(python, base)
            run_fields.update(
                source_identity=identity,
                workers=workers,
                tests=list(selected),
            )
            started_after = datetime.now(timezone.utc)
            if interrupted.is_set():
                raise KeyboardInterrupt
            started = time.monotonic()
            for index in range(workers):
                state = directory / f"shard-{index}"
                state.mkdir(mode=0o700)
                command = (
                    python,
                    "-B",
                    "-m",
                    "scripts.postgres_shard_worker",
                    "--state",
                    str(state),
                    "--identity",
                    identity,
                    "--workers",
                    str(workers),
                    "--index",
                    str(index),
                    "--durations",
                    str(durations),
                    "--tests",
                    *selected,
                )
                jobs.append(ShardJob(index, state, command, dict(base)))
            run_fields["status"] = "RUNNING"
            write_json_atomic(directory / "run.json", run_fields)
            print(
                f"postgres-shards: workers={workers} private receipts={directory}",
                flush=True,
            )
            try:
                receipts = run_jobs(
                    jobs,
                    identity=identity,
                    tests=selected,
                    stress=stress,
                    started_after=started_after,
                    interrupted=interrupted,
                )
            finally:
                # run_jobs drains all child owners before any parent reconciliation.
                reconcile_jobs(jobs)
            timings = validate_partition_union(receipts, workers=workers)
            validate_independent_allocations(jobs)
            if checked_identity(python, base) != identity:
                raise PostgresShardError(
                    "PostgreSQL source changed during sharded testing"
                )
            if interrupted.is_set():
                raise KeyboardInterrupt
            run_fields.update(
                status="COMPLETE",
                executed_tests=len(timings),
                elapsed_seconds=time.monotonic() - started,
            )
            write_json_atomic(directory / "run.json", run_fields)
            for nodeid, duration in sorted(
                timings.items(), key=lambda item: (-item[1], item[0])
            )[:durations]:
                print(f"postgres-shards: duration={duration:.3f}s {nodeid}")
            print(f"postgres-shards: PASS tests={len(timings)} workers={workers}")
            return directory
        except BaseException as error:
            if directory is not None:
                run_fields["status"] = (
                    "SUPERVISION_LOST"
                    if isinstance(error, ProcessSupervisionLost)
                    else "FAILED"
                )
                write_json_atomic(directory / "run.json", run_fields)
                print(
                    f"postgres-shards: failed; private reconciliation/receipts={directory}",
                    file=sys.stderr,
                )
            raise


def report_shard_failures(error: BaseException) -> None:
    """Print each shard failure in ``error``'s chain, the earliest first.

    Group cleanup runs in a ``finally``: when it fails too, its error replaces
    the worker's report, which stays the first cause and is printed first.
    """
    for failure in reversed(exception_chain(error, limit=8)):
        if isinstance(failure, PostgresShardError):
            for line in str(failure).splitlines() or [""]:
                print(f"postgres-shards: {line}", file=sys.stderr)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--durations", type=int, default=50)
    parser.add_argument("--tests", nargs="+", required=True)
    options = parser.parse_args(arguments)
    try:
        run_postgres_shards(
            options.python,
            workers=options.workers,
            durations=options.durations,
            tests=options.tests,
        )
    except KeyboardInterrupt:
        print(
            "postgres-shards: interrupted after owned children stopped", file=sys.stderr
        )
        return 130
    except PostgresShardError as error:
        report_shard_failures(error)
        return 2
    except (Exception, ProcessSupervisionLost) as error:
        report_shard_failures(error)
        print(
            "postgres-shards: failed; no test output or credentials are emitted",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
