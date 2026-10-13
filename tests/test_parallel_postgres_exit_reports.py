"""What the shard runner reports when a worker exits without a passing receipt."""

from __future__ import annotations

import os
import secrets
import stat
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.postgres_grant import ALLOCATION_ENV, POSTGRES_URL_ENV
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts import postgres_shard_worker as worker
from scripts import run_postgres_shards as runner
from scripts.postgres_shard_receipts import (
    FAILURE_MESSAGE_CHARACTERS,
    PostgresShardError,
    failure_chain,
    read_private_json,
    read_worker_failure,
)
from scripts.run_static_gates import isolated_gate_environment
from tests._parallel_postgres_process import CASES
from tests.admin._release_postgres_support import FakePostgresDocker

ROOT = Path(__file__).resolve().parents[1]
HELPER = "tests._parallel_postgres_process"
# A stand-in worker: print the prepared lines and exit 3, or wait to be cancelled.
FAKE_WORKER = """
import sys, time
from pathlib import Path
state, mode = Path(sys.argv[1]), sys.argv[2]
if mode == "wait":
    until = time.monotonic() + 30
    while not (state / "cancel.json").exists() and time.monotonic() < until:
        time.sleep(0.02)
    print("stopped after cancellation", file=sys.stderr)
    raise SystemExit(130)
sys.stderr.write((state / "lines.txt").read_text())
raise SystemExit(3)
"""


def environment() -> dict[str, str]:
    parent = {
        name: value
        for name, value in os.environ.items()
        if name not in {ALLOCATION_ENV, POSTGRES_URL_ENV}
        and not name.startswith(("PG", "POSTGRES_"))
    }
    return worker.worker_environment(isolated_gate_environment(parent), root=ROOT)


def fake_job(
    root: Path, index: int, mode: str, lines: Sequence[str] = ()
) -> runner.ShardJob:
    root.chmod(0o700)
    state = root / f"shard-{index}"
    state.mkdir(mode=0o700)
    (state / "lines.txt").write_text("".join(f"{line}\n" for line in lines))
    command = (sys.executable, "-I", "-c", FAKE_WORKER, str(state), mode)
    return runner.ShardJob(index, state, command, environment())


def run(*jobs: runner.ShardJob) -> list[Any]:
    return runner.run_jobs(
        jobs,
        identity="a" * 64,
        tests=(CASES,),
        stress=("", ""),
        started_after=datetime.now(timezone.utc),
    )


def test_failed_worker_tail_is_redacted_and_its_whole_output_kept_private(
    tmp_path: Path,
) -> None:
    short, generated = secrets.token_hex(8), secrets.token_urlsafe(32)
    secret_lines = [
        f"PGPASSWORD={short}",
        f"connecting to postgresql://postgres:{short}@127.0.0.1:5432/postgres",
        f"127.0.0.1:54321:*:postgres:{generated}",
    ]
    lines = [f"worker line {number:02d}" for number in range(30)]
    lines[5:5] = secret_lines
    lines[-3:-3] = secret_lines
    job = fake_job(tmp_path, 0, "print", lines)
    with pytest.raises(PostgresShardError) as caught:
        run(job)
    message = str(caught.value)
    summary, header, *tail = message.splitlines()
    log = job.state / runner.WORKER_LOG
    assert summary == (
        "PostgreSQL shard 0 exited with status 3 before pytest started: "
        "the worker recorded no cause"
    )
    assert header == f"last 20 lines of shard 0, redacted (full output: {log}):"
    assert tail[0] == "[shard-0] worker line 13", tail
    assert tail[-1] == "[shard-0] worker line 29", tail
    assert len(tail) == runner.WORKER_TAIL_LINES
    assert tail[-6:-3] == [
        '[shard-0] PGPASSWORD="<redacted>"',
        "[shard-0] connecting to postgresql://<redacted>@127.0.0.1:5432/postgres",
        "[shard-0] 127.0.0.1:54321:*:postgres:<redacted>",
    ], "a password, Postgres URL or PGPASSFILE line was not redacted"
    assert short not in message and generated not in message
    info = log.lstat()
    assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600, (
        "the failed worker's output must be a private regular file"
    )
    kept = log.read_text()
    assert kept.splitlines()[0] == "worker line 00", "the whole output was not kept"
    assert kept.count("<redacted>") == 6, kept
    assert short not in kept and generated not in kept


def test_first_exit_still_cancels_and_drains_every_other_shard(tmp_path: Path) -> None:
    failing = fake_job(tmp_path, 0, "print", ["the cause"])
    waiting = fake_job(tmp_path, 1, "wait")
    with pytest.raises(PostgresShardError) as caught:
        run(failing, waiting)
    assert (
        str(caught.value)
        .splitlines()[0]
        .startswith("PostgreSQL shard 0 exited with status 3 ")
    ), "the first exit was not the reported failure"
    assert "[shard-0] the cause" in str(caught.value)
    assert read_private_json(waiting.state / worker.CANCEL_FILE) == {"cancelled": True}
    # Kept only after the drain, so it proves the other shard reached its own exit.
    assert (waiting.state / runner.WORKER_LOG).read_text() == (
        "stopped after cancellation\n"
    )


@pytest.mark.parametrize(
    "mode,phase,summary,cause",
    [
        (
            "allocation",
            "allocation",
            "PostgreSQL shard 0 exited with status 2 before pytest started: "
            "worker phase allocation: BootstrapError: command failed (3): docker",
            None,
        ),
        (
            # The ownership location is a temporary path, so only its frame is fixed.
            "cleanup",
            "post-test",
            "PostgreSQL shard 0 exited with status 2 after pytest passed "
            "(post-test: allocation exit or cleanup): worker phase post-test: "
            "PostgresCleanupError: local PostgreSQL cleanup unconfirmed; "
            "private ownership location: ",
            "; caused by BootstrapError: local PostgreSQL removal is unconfirmed",
        ),
        (
            "test-failure",
            None,
            f"PostgreSQL shard 0 failed: nodeid={CASES}::test_variant[0] "
            "phase=call outcome=failed",
            None,
        ),
    ],
    ids=["allocation", "cleanup", "test-failure"],
)
def test_real_worker_failure_names_its_phase(
    tmp_path: Path, mode: str, phase: str | None, summary: str, cause: str | None
) -> None:
    tmp_path.chmod(0o700)
    state = tmp_path / "shard-0"
    state.mkdir(mode=0o700)
    command = (sys.executable, "-B", "-m", HELPER, "exit-worker", mode, str(state))
    with pytest.raises(PostgresShardError) as caught:
        run(runner.ShardJob(0, state, command, environment()))
    first, *rest = str(caught.value).splitlines()
    if cause is None:
        assert first == summary
    else:
        assert first.startswith(summary) and first.endswith(cause), first
    if phase is not None:
        assert rest[-1].startswith(
            f"[shard-0] postgres-shards: worker failed during {phase}: "
        ), rest


@pytest.mark.parametrize("phase", ["preparation", "allocation", "post-test"])
def test_worker_records_its_own_non_test_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    tmp_path.chmod(0o700)
    canary = secrets.token_hex(8)
    docker = FakePostgresDocker()
    docker.ready, docker.ready_code = phase != "allocation", 3
    docker.keep_after_remove = phase == "post-test"

    def identify(_root: Path) -> str:
        if phase == "preparation":
            raise RuntimeError(
                f"cannot hash postgresql://postgres:{canary}@127.0.0.1:5432/x"
            )
        return "a" * 64

    def pytest_child(
        command: Sequence[str], **_options: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(worker, "CommandRunner", lambda: docker)
    monkeypatch.setattr(worker, "source_identity", identify)
    monkeypatch.setattr(worker, "run_owned_command", pytest_child)
    monkeypatch.setattr(os, "umask", lambda _mask: 0o022)
    arguments = ["--state", str(tmp_path), "--identity", "a" * 64, "--workers", "1"]
    arguments += ["--index", "0", "--durations", "0", "--tests", CASES]
    assert worker.main(arguments) == 2
    path = tmp_path / worker.WORKER_FAILURE_FILE
    record = read_private_json(path)
    assert record["phase"] == phase
    errors = [(item["type"], item["message"]) for item in record["errors"]]
    if phase == "preparation":
        assert errors == [
            ("RuntimeError", "cannot hash postgresql://<redacted>@127.0.0.1:5432/x")
        ]
    elif phase == "allocation":
        assert errors == [("BootstrapError", "command failed (3): docker")]
    else:
        assert [kind for kind, _message in errors] == [
            "PostgresCleanupError",
            "BootstrapError",
        ]
        assert errors[0][1].startswith(
            "local PostgreSQL cleanup unconfirmed; private ownership location: "
        ), errors
        assert errors[1][1] == "local PostgreSQL removal is unconfirmed"
    printed = capsys.readouterr().err
    assert f"postgres-shards: worker failed during {phase}: {errors[0][0]}: " in (
        printed
    )
    assert canary not in printed and canary not in path.read_text(), (
        "a credential reached the worker's failure record or output"
    )


def test_recorded_chain_is_bounded_redacted_and_accepted_by_the_runner(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    canary = secrets.token_hex(8)
    try:
        try:
            raise OSError("removal timed out\x7f " + " ".join(["word"] * 400))
        except OSError as cause:
            cause.add_note("retained for reconciliation")
            raise RuntimeError(f"cleanup of postgresql://u:{canary}@h/db") from cause
    except RuntimeError as error:
        errors = failure_chain(error)
    write_json_atomic(
        tmp_path / "record.json", {"phase": "post-test", "errors": errors}
    )
    record = read_worker_failure(tmp_path / "record.json")
    outer, cause = record["errors"]
    assert outer == {
        "type": "RuntimeError",
        "message": "cleanup of postgresql://<redacted>@h/db",
    }
    assert cause["type"] == "OSError"
    # Shortened in the middle: the start names the failure, the end its notes.
    assert cause["message"].startswith("removal timed out word word"), cause
    assert cause["message"].endswith("word word; retained for reconciliation"), cause
    assert len(cause["message"]) <= FAILURE_MESSAGE_CHARACTERS
    assert canary not in (tmp_path / "record.json").read_text()


@pytest.mark.parametrize("defect", ["shape", "phase", "mode"])
def test_invalid_worker_record_never_replaces_the_failure(
    tmp_path: Path, defect: str
) -> None:
    job = fake_job(tmp_path, 0, "print", ["the worker's own words"])
    record: dict[str, Any] = {
        "phase": "post-test",
        "errors": [{"type": "PostgresCleanupError", "message": "invented"}],
    }
    if defect == "shape":
        record["passed"] = True
    if defect == "phase":
        record["phase"] = "complete"
    path = job.state / worker.WORKER_FAILURE_FILE
    write_json_atomic(path, record)
    if defect == "mode":
        path.chmod(0o644)
    with pytest.raises(PostgresShardError) as caught:
        run(job)
    message = str(caught.value)
    assert message.splitlines()[0] == (
        "PostgreSQL shard 0 exited with status 3 before pytest started: "
        "the worker's failure record is invalid"
    )
    assert "invented" not in message
    assert "[shard-0] the worker's own words" in message


@pytest.mark.parametrize("final", ["shard-error", "supervision-lost"])
def test_cli_prints_every_line_and_keeps_an_earlier_shard_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], final: str
) -> None:
    def launch(_python: str, **_options: Any) -> Path:
        try:
            raise PostgresShardError(
                "PostgreSQL shard 8 exited with status 2\n"
                "last 1 lines of shard 8, redacted (full output: example):\n"
                "[shard-8] example cause"
            )
        except PostgresShardError:
            if final == "shard-error":
                raise PostgresShardError(
                    "one or more PostgreSQL shard cleanups are unconfirmed"
                )
            raise ProcessSupervisionLost("example supervision loss")

    monkeypatch.setattr(runner, "run_postgres_shards", launch)
    assert runner.main(["--tests", "tests/test_example.py"]) == 2
    assert capsys.readouterr().err.splitlines() == [
        "postgres-shards: PostgreSQL shard 8 exited with status 2",
        "postgres-shards: last 1 lines of shard 8, redacted (full output: example):",
        "postgres-shards: [shard-8] example cause",
        "postgres-shards: one or more PostgreSQL shard cleanups are unconfirmed"
        if final == "shard-error"
        else "postgres-shards: failed; no test output or credentials are emitted",
    ]
