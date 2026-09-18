from __future__ import annotations

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gpu_fault.admin import execution, process_supervisor

ROOT = Path(__file__).resolve().parents[2]


def command_tree(directory: Path) -> str:
    return f"""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path({str(directory)!r})
def identity(pid):
    fields=Path(f"/proc/{{pid}}/stat").read_text().rsplit(")",1)[1].split()
    return {{"pid":pid,"parent":int(fields[1]),"start":fields[19]}}
leaf_code="import time; time.sleep(30)"
detached=subprocess.Popen([sys.executable,"-c",leaf_code],start_new_session=True)
supervisor=identity(os.getppid())
guardian=identity(supervisor["parent"])
record={{"inner":supervisor,"outer":guardian,
        "leaf":identity(os.getpid()),"detached":identity(detached.pid)}}
(root/"identities.json").write_text(json.dumps(record))
(root/"ready").touch()
time.sleep(30)
(root/"late-mutation").touch()
"""


def wait_ready(directory: Path) -> dict[str, dict[str, int | str]]:
    until = time.monotonic() + 8
    while not (directory / "ready").exists():
        assert time.monotonic() < until, "isolated command did not start"
        time.sleep(0.02)
    return json.loads((directory / "identities.json").read_text())


def signal_fixture(identity: dict[str, int | str], signum: signal.Signals) -> None:
    pid = int(identity["pid"])
    descriptor = os.pidfd_open(pid)
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        assert fields[19] == identity["start"], "fixture PID was reused"
        signal.pidfd_send_signal(descriptor, signum)
    finally:
        os.close(descriptor)


def assert_children_reaped(identities: dict[str, dict[str, int | str]]) -> None:
    for name in ("leaf", "detached"):
        pid = identities[name]["pid"]
        assert not Path(f"/proc/{pid}").exists(), (
            f"{name} survived the command's reported completion"
        )


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("lost_owner", ["inner", "outer"])
def test_surviving_owner_reaps_the_tree_before_failure_returns(
    tmp_path: Path, capture: bool, lost_owner: str
) -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            execution.run_command,
            [sys.executable, "-c", command_tree(tmp_path)],
            capture=capture,
            timeout_seconds=15,
        )
        identities = wait_ready(tmp_path)
        assert identities["outer"]["parent"] == os.getpid(), (
            "refusing to signal a supervisor outside this test's command tree"
        )
        signal_fixture(identities[lost_owner], signal.SIGKILL)
        with pytest.raises(RuntimeError, match="surviving owner"):
            future.result(timeout=8)
    assert_children_reaped(identities)
    assert not (tmp_path / "late-mutation").exists(), (
        "failed command continued to mutate after reporting supervision loss"
    )


def test_owner_loss_does_not_cancel_another_commands_tree(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(
            execution.run_command,
            [sys.executable, "-c", command_tree(tmp_path)],
            timeout_seconds=15,
        )
        healthy = pool.submit(
            execution.run_command,
            [sys.executable, "-c", "import time; time.sleep(1); print('healthy')"],
            timeout_seconds=10,
        )
        identities = wait_ready(tmp_path)
        assert identities["outer"]["parent"] == os.getpid(), (
            "fixture guardian is not a child of this test"
        )
        signal_fixture(identities["inner"], signal.SIGKILL)
        with pytest.raises(RuntimeError, match="surviving owner"):
            failed.result(timeout=8)
        result = healthy.result(timeout=8)
    assert_children_reaped(identities)
    assert result.returncode == 0 and result.stdout.strip() == "healthy", (
        "failed command's guardian interfered with a concurrent command"
    )


@pytest.mark.parametrize("capture", [True, False])
def test_unproven_completion_cannot_start_compensation_or_another_command(
    tmp_path: Path, capture: bool
) -> None:
    # The outer test command owns the fixture even when both inner owners die,
    # so a failed assertion cannot leave an orphaned test process on the host.
    driver = f"""
import sys
from pathlib import Path
from gpu_fault.admin.execution import run_command
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault_release.regional_release_automatic_rollback import recover_failed_upgrade
root=Path({str(tmp_path)!r})
try:
    try:
        run_command([sys.executable,"-c",{command_tree(tmp_path)!r}],
                    capture={capture!r},timeout_seconds=15)
    except Exception:
        (root/"unsafe-compensation").touch()
        raise
except ProcessSupervisionLost:
    try:
        run_command([sys.executable,"-c","print('must not run')"])
    except ProcessSupervisionLost:
        (root/"subsequent-command-refused").touch()
    else:
        raise AssertionError("unproven tree allowed another command")
    try:
        recover_failed_upgrade(None,error=RuntimeError("ordinary secondary failure"),
            diff=None,plan=None,previous={{}},completed_phases=set(),
            completed_clusters=set(),registry_staged=False)
    except ProcessSupervisionLost:
        (root/"automatic-recovery-refused").touch()
    else:
        raise AssertionError("unproven tree allowed automatic recovery")
else:
    raise AssertionError("missing owner proofs were accepted")
"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            execution.run_command,
            [sys.executable, "-c", driver],
            environment={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
            },
            timeout_seconds=20,
        )
        identities = wait_ready(tmp_path)
        signal_fixture(identities["outer"], signal.SIGSTOP)
        signal_fixture(identities["inner"], signal.SIGKILL)
        signal_fixture(identities["outer"], signal.SIGKILL)
        result = future.result(timeout=8)
    assert result.returncode == 0, result.stderr
    assert_children_reaped(identities)
    assert (tmp_path / "subsequent-command-refused").exists(), (
        "the unproven-completion state was not retained"
    )
    assert (tmp_path / "automatic-recovery-refused").exists(), (
        "automatic rollback did not check the process safety state"
    )
    assert not (tmp_path / "unsafe-compensation").exists(), (
        "ordinary exception recovery accepted unproven process completion"
    )
    assert not (tmp_path / "late-mutation").exists(), (
        "outer owner did not drain the abandoned fixture"
    )


def test_loss_exception_is_not_an_ordinary_recoverable_failure() -> None:
    assert not issubclass(process_supervisor.ProcessSupervisionLost, Exception), (
        "unproven process completion must bypass ordinary automatic compensation"
    )


def test_captured_stdin_survives_incremental_supervisor_waiting() -> None:
    text = "fixture input\n" * 2048
    result = execution.run_command(
        [
            sys.executable,
            "-c",
            "import sys,time; time.sleep(.3); value=sys.stdin.read(); print(len(value))",
        ],
        input_text=text,
        timeout_seconds=8,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(len(text)), (
        "incremental communicate lost or resent command input"
    )
