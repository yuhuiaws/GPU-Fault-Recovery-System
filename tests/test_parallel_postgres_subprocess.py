from __future__ import annotations

import os
import re
import secrets
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.postgres_grant import ALLOCATION_ENV, POSTGRES_URL_ENV
from gpu_fault.admin.process_supervisor import run_owned_command
from gpu_fault.admin.release_postgres import ALLOCATION_DIRECTORY
from scripts import postgres_shard_worker as worker
from scripts.postgres_shard_receipts import (
    PostgresShardError,
    read_private_json,
    validate_partition_union,
    validate_shard_receipt,
)
from scripts.run_static_gates import isolated_gate_environment
from tests._parallel_postgres_process import CASES
from tests.admin._release_postgres_support import CID, FakePostgresDocker
from tools.pytest_case_reporter import (
    PARTITION_COUNT_ENV,
    PARTITION_INDEX_ENV,
    REPORT_ENV,
)
from tools.pytest_result_identity import source_identity

ROOT = Path(__file__).resolve().parents[1]
HELPER = "tests._parallel_postgres_process"


def environment() -> dict[str, str]:
    parent = {
        name: value
        for name, value in os.environ.items()
        if name not in {ALLOCATION_ENV, POSTGRES_URL_ENV}
        and not name.startswith(("PG", "POSTGRES_"))
    }
    return worker.worker_environment(isolated_gate_environment(parent), root=ROOT)


def run_pytest(
    state: Path, *, index: int, workers: int, mode: str = ""
) -> tuple[subprocess.CompletedProcess[str], str, datetime, str]:
    state.mkdir(mode=0o700)
    identity = source_identity(ROOT)
    started = datetime.now(timezone.utc)
    canary = secrets.token_urlsafe(32)
    env = environment()
    env.update(
        {
            REPORT_ENV: str(state / worker.RECEIPT_FILE),
            PARTITION_COUNT_ENV: str(workers),
            PARTITION_INDEX_ENV: str(index),
            worker.FAILURE_ENV: str(state / worker.FAILURE_FILE),
            "PARALLEL_POSTGRES_CASE_MODE": mode,
            "PARALLEL_POSTGRES_CANARY": canary,
        }
    )
    result = run_owned_command(
        [sys.executable, "-B", "-m", HELPER, "pytest", mode, str(state)],
        environment=env,
        cwd=ROOT,
        timeout=90,
    )
    return result, identity, started, canary


def validate(
    state: Path, identity: str, started: datetime, index: int, workers: int
) -> Any:
    return validate_shard_receipt(
        state / worker.RECEIPT_FILE,
        root=ROOT,
        identity=identity,
        tests=(CASES,),
        workers=workers,
        index=index,
        stress=("", ""),
        started_after=started,
    )


def test_real_reporter_proves_full_disjoint_serial_partitions(tmp_path: Path) -> None:
    receipts = []
    for index in range(2):
        state = tmp_path / str(index)
        result, identity, started, canary = run_pytest(state, index=index, workers=2)
        assert result.returncode == 0, "real serial pytest partition failed"
        receipts.append(validate(state, identity, started, index, 2))
        assert canary not in (state / worker.RECEIPT_FILE).read_text(), (
            "captured private test output entered the receipt"
        )
    durations = validate_partition_union(receipts, workers=2)
    assert len(durations) == 25, "partition union lost a real discovered test"
    assert sum(durations.values()) > 0, "real durations were not recorded"


def test_real_nested_pytest_is_not_partitioned_and_cannot_replace_outer_receipt(
    tmp_path: Path,
) -> None:
    state = tmp_path / "outer"
    result, identity, started, _canary = run_pytest(
        state, index=0, workers=1, mode="nested"
    )
    assert result.returncode == 0, "nested pytest inherited outer control variables"
    receipt = validate(state, identity, started, 0, 1)
    assert len(receipt.durations) == 25, "nested pytest overwrote the outer receipt"


@pytest.mark.parametrize(
    "mode",
    ["failure", "skip", "setup", "teardown", "collection-skip", "collection-error"],
)
def test_real_failure_or_skip_cannot_produce_a_passing_or_credential_leaking_receipt(
    tmp_path: Path, mode: str
) -> None:
    state = tmp_path / "failed"
    _result, identity, started, canary = run_pytest(
        state, index=0, workers=1, mode=mode
    )
    with pytest.raises(PostgresShardError):
        validate(state, identity, started, 0, 1)
    first = read_private_json(state / worker.FAILURE_FILE)
    expected_phase = (
        "collect"
        if mode.startswith("collection-")
        else mode
        if mode in {"setup", "teardown"}
        else "call"
    )
    # Every variant fails the same way, and the record names whichever ran
    # first. The nested run inherits GPU_FAULT_TEST_SHUFFLE_SEED on purpose
    # (worker_environment keeps the explicit test controls), so under
    # ``make test-shuffled`` that is not variant 0.
    nodeid = first.pop("nodeid", None)
    if expected_phase == "collect":
        assert nodeid == CASES, nodeid
    else:
        assert isinstance(nodeid, str) and re.fullmatch(
            re.escape(CASES) + r"::test_variant\[\d+\]", nodeid
        ), nodeid
    assert first == {
        "failed": True,
        "phase": expected_phase,
        "outcome": "skipped" if mode in {"skip", "collection-skip"} else "failed",
    }
    assert canary not in (state / worker.RECEIPT_FILE).read_text(), (
        "exception or skipped-test text leaked into a source-bound receipt"
    )


@pytest.mark.parametrize("mode", ["failure", "exit"])
def test_first_worker_failure_stops_and_reaps_detached_children(
    tmp_path: Path, mode: str
) -> None:
    tmp_path.chmod(0o700)
    result = run_owned_command(
        [sys.executable, "-B", "-m", HELPER, "group", mode, str(tmp_path)],
        environment=environment(),
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 0, "fail-fast process fixture failed"
    assert read_private_json(tmp_path / "result.json") == {
        "status": "failed",
        "stopped": True,
    }


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_repeated_parent_cancellation_drains_every_owned_child(
    tmp_path: Path, signum: signal.Signals
) -> None:
    tmp_path.chmod(0o700)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_owned_command,
            [sys.executable, "-B", "-m", HELPER, "group", "sleep", str(tmp_path)],
            environment=environment(),
            cwd=ROOT,
            timeout=30,
        )
        until = time.monotonic() + 15
        while not (tmp_path / "shard-1/pids.json").exists():
            assert time.monotonic() < until, "owned test children were not ready"
            assert not future.done(), "process fixture exited before cancellation"
            time.sleep(0.02)
        group = read_private_json(tmp_path / "group.json")
        descriptor = os.pidfd_open(group["pid"])
        try:
            signal.pidfd_send_signal(descriptor, signum)
            signal.pidfd_send_signal(descriptor, signum)
        finally:
            os.close(descriptor)
        # The fixture bounds itself (run_owned_command timeout=30 plus the
        # supervisor's termination grace); wait at least that long so a loaded
        # CI runner cannot make this assertion race the product's own deadline.
        result = future.result(timeout=45)
    assert result.returncode == 0, "cancellation process fixture failed"
    assert read_private_json(tmp_path / "result.json") == {
        "status": "interrupted",
        "stopped": True,
    }


def test_worker_dying_inside_real_allocation_context_is_reconciled_by_owner_record(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    result = run_owned_command(
        [
            sys.executable,
            "-B",
            "-m",
            HELPER,
            "allocate-and-exit",
            "crash",
            str(tmp_path),
        ],
        environment=environment(),
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == -signal.SIGKILL, (
        "fixture was not killed inside the allocation context"
    )
    directory = tmp_path / ALLOCATION_DIRECTORY
    assert directory.is_dir(), "interrupted allocator lost its reconciliation record"
    docker = FakePostgresDocker()
    docker.containers[CID] = read_private_json(tmp_path / "fake-container.json")
    worker.reconcile_allocation(tmp_path, docker)
    assert docker.removed == [CID], "parent reconciliation did not use the original CID"
    assert not directory.exists(), (
        "parent reconciliation retained confirmed credentials"
    )


@pytest.mark.parametrize("mode", ["failure", "skip"])
def test_eight_nested_workers_can_run_owned_cleanup_after_first_pytest_failure(
    tmp_path: Path, mode: str
) -> None:
    tmp_path.chmod(0o700)
    result = run_owned_command(
        [sys.executable, "-B", "-m", HELPER, "lifecycle-wrapper", mode, str(tmp_path)],
        environment=environment(),
        cwd=ROOT,
        timeout=60,
    )
    assert result.returncode == 0, "nested process harness failed"
    outcome = read_private_json(tmp_path / "result.json")
    assert outcome["started"] == 8, "failure was not inside eight active pytest workers"
    assert outcome["stopped"] == 8, "pytest descendants escaped their owners"
    assert outcome["supervision_safe"] is True
    assert outcome["cleaned"] == 8, "ordinary failure prevented owned database cleanup"
    assert outcome["ownership"] == [], (
        "ordinary cancellation retained lost-supervision state"
    )
    assert outcome["failure"] == "PostgresShardError"
    failed = tmp_path / "shard-3"
    first = read_private_json(failed / worker.FAILURE_FILE)
    assert first["phase"] == "call"
    assert first["outcome"] == ("failed" if mode == "failure" else "skipped")
    assert first["nodeid"].startswith(
        "tests/_parallel_postgres_lifecycle_cases.py::test_lifetime["
    ), "the original test failure was lost"
    if mode == "failure":
        receipt = read_private_json(failed / worker.RECEIPT_FILE)
        assert receipt["session"]["exitstatus"] == 1, (
            "failure cancellation erased the pytest receipt"
        )
    assert (
        "example private failure details"
        not in (failed / worker.FAILURE_FILE).read_text()
    )


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_eight_nested_workers_finish_owned_cleanup_after_repeated_signals(
    tmp_path: Path, signum: signal.Signals
) -> None:
    tmp_path.chmod(0o700)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_owned_command,
            [
                sys.executable,
                "-B",
                "-m",
                HELPER,
                "lifecycle-wrapper",
                "sleep",
                str(tmp_path),
            ],
            environment=environment(),
            cwd=ROOT,
            timeout=60,
        )
        until = time.monotonic() + 25
        while not all(
            (tmp_path / f"shard-{index}/test-running.json").exists()
            for index in range(8)
        ):
            assert time.monotonic() < until, (
                "eight nested pytest workers did not become ready"
            )
            assert not future.done(), "nested worker fixture exited before cancellation"
            time.sleep(0.02)
        descriptor = os.pidfd_open(read_private_json(tmp_path / "group.json")["pid"])
        try:
            signal.pidfd_send_signal(descriptor, signum)
            signal.pidfd_send_signal(descriptor, signum)
            until = time.monotonic() + 15
            while not all(
                (tmp_path / f"shard-{index}/cleanup-started.json").exists()
                for index in range(8)
            ):
                assert time.monotonic() < until, (
                    "owned allocation cleanup did not start"
                )
                assert not future.done(), "nested fixture exited before cleanup"
                time.sleep(0.02)
            signal.pidfd_send_signal(descriptor, signum)
        finally:
            os.close(descriptor)
        # Same rule: the fixture's own run_owned_command timeout is 60 seconds.
        result = future.result(timeout=75)
    assert result.returncode == 0, "nested cancellation harness failed"
    assert read_private_json(tmp_path / "result.json") == {
        "failure": "KeyboardInterrupt",
        "started": 8,
        "stopped": 8,
        "cleaned": 8,
        "ownership": [],
        "supervision_safe": True,
    }
