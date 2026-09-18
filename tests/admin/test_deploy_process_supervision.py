from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gpu_fault.admin import execution, process_supervisor
from gpu_fault.admin.diagnostics import DriverDiagnostics


def wait_for(path: Path, *, timeout: float = 5) -> None:
    until = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < until, f"child did not create {path.name}"
        time.sleep(0.02)


def sleeper(ready: Path, marker: Path, *, ignore_term: bool = False) -> str:
    return (
        "import os,signal,time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if ignore_term else "")
        + f"Path({str(ready)!r}).write_text(str(os.getpid())); "
        "time.sleep(20); "
        f"Path({str(marker)!r}).touch()"
    )


@pytest.mark.parametrize("capture", [True, False])
def test_nested_runners_cannot_escape_parent_cleanup(tmp_path, capture):
    ready, marker = tmp_path / "ready", tmp_path / "survived"
    leaf = sleeper(ready, marker)
    parent = (
        "import sys; from gpu_fault.admin.execution import run_command; "
        f"run_command([sys.executable,'-c',{leaf!r}], capture=False, timeout_seconds=30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        execution.run_command(
            [sys.executable, "-c", parent], capture=capture, timeout_seconds=1.5
        )
    assert ready.exists(), "descendant was never started"
    assert not Path(f"/proc/{int(ready.read_text())}").exists(), (
        "nested descendant was not reaped before the command returned"
    )
    assert not marker.exists(), "timed-out descendant continued execution"


def test_successful_parent_also_reaps_detached_background_work(tmp_path):
    ready, marker = tmp_path / "ready", tmp_path / "survived"
    leaf = sleeper(ready, marker)
    parent = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{leaf!r}],start_new_session=True); "
        f"\nwhile not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        "os._exit(0)"
    )
    result = execution.run_command([sys.executable, "-c", parent], timeout_seconds=5)
    assert result.returncode == 0
    assert ready.exists(), "the detached descendant was never started"
    assert not Path(f"/proc/{int(ready.read_text())}").exists(), (
        "successful command returned with detached work still running"
    )
    assert not marker.exists(), "detached work outlived the command"


def test_timeout_cleanup_cannot_reap_another_concurrent_commands_children(tmp_path):
    ready, marker = tmp_path / "timed-out-ready", tmp_path / "timed-out-marker"
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(
            execution.run_command,
            [sys.executable, "-c", sleeper(ready, marker)],
            timeout_seconds=1.5,
        )
        healthy = pool.submit(
            execution.run_command,
            [sys.executable, "-c", "import time; time.sleep(2); print('completed')"],
            timeout_seconds=5,
        )
        with pytest.raises(subprocess.TimeoutExpired):
            failed.result(timeout=5)
        result = healthy.result(timeout=5)
    assert ready.exists(), "the failed command was not running concurrently"
    assert not marker.exists(), "timed-out work continued"
    assert result.returncode == 0 and result.stdout.strip() == "completed", (
        "cleanup signalled or reaped another command's process tree"
    )


def test_sigkill_escalation_reaps_a_new_session(tmp_path, monkeypatch):
    monkeypatch.setattr(process_supervisor, "TERMINATION_GRACE_SECONDS", 0.2)
    ready, marker = tmp_path / "ready", tmp_path / "survived"
    leaf = sleeper(ready, marker, ignore_term=True)
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{leaf!r}],start_new_session=True); "
        "time.sleep(20)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        execution.run_command([sys.executable, "-c", parent], timeout_seconds=1.5)
    assert ready.exists(), "the SIGTERM-resistant process was never started"
    assert not Path(f"/proc/{int(ready.read_text())}").exists(), (
        "SIGKILL did not reap the process outside the original process group"
    )
    assert not marker.exists(), "SIGTERM-resistant work continued"


@pytest.mark.parametrize("block_term", [False, True])
def test_supervisor_cleans_up_when_its_owner_is_killed(tmp_path, block_term):
    ready, marker = tmp_path / "ready", tmp_path / "survived"
    leaf = sleeper(ready, marker)
    owner = (
        "import signal,sys; from gpu_fault.admin.execution import run_command; "
        + (
            "signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM}); "
            if block_term
            else ""
        )
        + f"run_command([sys.executable,'-c',{leaf!r}],timeout_seconds=30)"
    )
    process = subprocess.Popen([sys.executable, "-c", owner])
    try:
        wait_for(ready)
        process.kill()
        process.wait(timeout=5)
        pid = int(ready.read_text())
        until = time.monotonic() + 5
        while Path(f"/proc/{pid}").exists():
            assert time.monotonic() < until, "owner death stranded a descendant"
            time.sleep(0.02)
        assert not marker.exists(), "owner death left executable background work"
    finally:
        process.kill()
        process.wait(timeout=5)


def test_outer_driver_preserves_recovery_after_normal_deadline(tmp_path):
    started, complete = tmp_path / "recovering", tmp_path / "recovered"
    child = f"""
import subprocess,sys,time
from pathlib import Path
from gpu_fault.admin.execution import run_command,recovery_deadline
try:
    run_command([sys.executable,'-c','import time; time.sleep(20)'])
except subprocess.TimeoutExpired:
    with recovery_deadline('test rollback'):
        Path({str(started)!r}).touch()
        time.sleep(0.6)
        Path({str(complete)!r}).touch()
    raise SystemExit(2)
"""
    with execution.deployment_deadline("test deploy", 1, recovery_seconds=2):
        result = execution.run_driver(
            [sys.executable, "-c", child], capture_output=True
        )
    assert result.returncode == 2, result.stderr
    assert started.exists(), "the child never entered recovery"
    assert complete.exists(), "an ancestor interrupted the independent recovery window"


def test_nested_recovery_cannot_extend_the_hard_stop(tmp_path):
    marker = tmp_path / "escaped"
    child = f"""
import sys
from gpu_fault.admin.execution import recovery_deadline,run_driver
with recovery_deadline('rollback'):
    run_driver([sys.executable,'-c',
        "import time; from pathlib import Path; time.sleep(20); Path({str(marker)!r}).touch()"])
"""
    started = time.monotonic()
    with execution.deployment_deadline("test deploy", 0.5, recovery_seconds=0.5):
        with pytest.raises(execution.DeploymentDeadlineExceeded):
            execution.run_driver([sys.executable, "-c", child], capture_output=True)
    assert time.monotonic() - started < 4, "nested driver extended the hard deadline"
    assert not marker.exists(), "work escaped the recovery hard stop"


def test_recovery_scope_reentry_cannot_reset_its_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    monkeypatch.delenv(execution.HARD_DEADLINE_ENV, raising=False)
    with execution.deployment_deadline("normal", 10, recovery_seconds=20):
        clock[0] = 111
        with execution.recovery_deadline("rollback"):
            assert execution.current_deadline().expires == 130
            clock[0] = 120
            with execution.recovery_deadline("nested rollback"):
                assert execution.current_deadline().expires == 130
        clock[0] = 131
        with pytest.raises(execution.DeploymentDeadlineExceeded):
            with execution.recovery_deadline("too late"):
                pytest.fail("expired recovery budget was extended")


@pytest.mark.parametrize("hard", ["0", "-1"])
def test_expired_hard_limit_cannot_be_replaced_with_a_new_reserve(monkeypatch, hard):
    monkeypatch.setenv(execution.HARD_DEADLINE_ENV, hard)
    with pytest.raises(execution.DeploymentDeadlineExceeded):
        with execution.recovery_deadline("expired"):
            pytest.fail("expired hard limit allowed recovery to start")
    with pytest.raises(execution.DeploymentDeadlineExceeded):
        with execution.cleanup_deadline("expired"):
            pytest.fail("expired hard limit allowed cleanup to start")


def test_json_driver_streams_progress_before_completion(tmp_path, capsys):
    done = tmp_path / "continue"
    line = "release-phase 2026-09-11T00:00:00Z schema-ready elapsed=1.0s total=1.0s"
    child = f"""
import sys,time
from pathlib import Path
print({line!r},file=sys.stderr,flush=True)
deadline=time.monotonic()+5
while not Path({str(done)!r}).exists() and time.monotonic()<deadline:
    time.sleep(0.01)
print('{{"healthy":true}}')
"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            execution.run_driver, [sys.executable, "-c", child], stdout=subprocess.PIPE
        )
        try:
            until = time.monotonic() + 3
            output = ""
            while line not in output:
                output += capsys.readouterr().err
                assert time.monotonic() < until, "driver progress stayed buffered"
                time.sleep(0.02)
            assert not future.done(), "progress appeared only after child exit"
        finally:
            done.touch()
        assert json.loads(future.result(timeout=5).stdout) == {"healthy": True}


def test_stream_preserves_multiline_redaction_context(capsys):
    diagnostic = DriverDiagnostics()
    for line in ('{"name":"execution-token",\n', '"value":"must-stay-private"}\n'):
        diagnostic.feed(line)
    assert not capsys.readouterr().err, "arbitrary JSON was streamed before redaction"
    diagnostic.finish()
    output = capsys.readouterr().err
    assert "must-stay-private" not in output
    assert "redacted" in output


def test_oversized_diagnostic_never_prints_an_untrusted_tail(capsys):
    diagnostic = DriverDiagnostics()
    diagnostic.maximum = 32
    diagnostic.feed("x" * 100 + "\n")
    diagnostic.feed("private-unlabelled-tail\n")
    diagnostic.finish()
    assert "private-unlabelled-tail" not in capsys.readouterr().err


def test_missing_command_keeps_oserror_contract():
    with pytest.raises(FileNotFoundError):
        execution.run_command(["/nonexistent/deploy-review-command"])


def test_closed_descriptor_is_rejected_before_starting_any_command(tmp_path):
    marker = tmp_path / "unexpected"
    with pytest.raises(OSError):
        execution.run_command(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ],
            pass_fds=(1_000_000_000,),
        )
    assert not marker.exists(), "invalid inherited descriptor started a command"
