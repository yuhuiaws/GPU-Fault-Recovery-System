from __future__ import annotations

import errno
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import run_owned_command

ROOT = Path(__file__).resolve().parents[2]

LEAF = """
import json
import os
import signal
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])

def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "parent": int(fields[1]), "start": fields[19]}

def resist_term(signum, frame):
    (root / "term-received").touch()

signal.signal(signal.SIGTERM, resist_term)
inner = identity(os.getppid())
guardian = identity(inner["parent"])
(root / "identities.json").write_text(json.dumps({
    "leaf": identity(os.getpid()), "inner": inner, "guardian": guardian,
}))
(root / "ready").touch()
deadline = time.monotonic() + 15
while not (root / "finish-leaf").exists() and time.monotonic() < deadline:
    time.sleep(.01)
with (root / "events.log").open("a") as events:
    events.write("leaf-final-mutation\\n")
"""

DRIVER_IMPORTS = """
import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

from gpu_fault.admin.operation_lock import site_operation_lock
from gpu_fault.admin.process_supervisor import (
    ProcessSupervisionLost, ensure_supervision_safe, run_owned_command,
)
"""

DRIVER_HELPERS = """
def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "parent": int(fields[1]), "start": fields[19]}

def still_present(item):
    try:
        return identity(item["pid"])["start"] == item["start"]
    except FileNotFoundError:
        return False

def wait_file(name, seconds=6, directory=None):
    directory = root if directory is None else directory
    deadline = time.monotonic() + seconds
    while not (directory / name).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"fixture did not reach {name}")
        time.sleep(.01)

def fixture(directory=None):
    directory = root if directory is None else directory
    wait_file("ready", directory=directory)
    value = json.loads((directory / "identities.json").read_text())
    assert value["guardian"]["parent"] == os.getpid(), (
        "refusing to signal a guardian outside this isolated driver"
    )
    return value

def send_fixture(item, signum):
    descriptor = os.pidfd_open(item["pid"])
    try:
        assert still_present(item), "fixture PID identity changed before signalling"
        signal.pidfd_send_signal(descriptor, signum)
    finally:
        os.close(descriptor)

def release_leaf(directory=None):
    directory = root if directory is None else directory
    # The caller may return only after this independent controller allows exit.
    # An early return is recorded before its marker releases the leaf.
    deadline = time.monotonic() + 1
    while not (directory / "owner-returned").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    (directory / "finish-leaf").touch()

def record_event(directory, event):
    with (directory / "events.log").open("a") as events:
        events.write(event + "\\n")

def observe_return(report):
    value = fixture()
    report["leaf_alive_at_return"] = still_present(value["leaf"])
    try:
        ensure_supervision_safe()
    except ProcessSupervisionLost:
        report["poisoned"] = True
    else:
        report["poisoned"] = False
    record_event(root, "owner-returned")
    (root / "owner-returned").touch()
"""

EXCEPTION_DRIVER = """
report = {}
controller_errors = []
injected = False
original_communicate = subprocess.Popen.communicate
initial_handler = signal.getsignal(signal.SIGINT)
initial_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
driver_identity = identity(os.getpid())

def pipe_failure(process, *args, **kwargs):
    global injected
    if not injected:
        value = fixture()
        assert process.pid == value["guardian"]["pid"], (
            "the pipe failure hook reached a different command"
        )
        send_fixture(value["guardian"], signal.SIGKILL)
        process.wait(timeout=6)
        injected = True
        (root / "fault-injected").touch()
        raise OSError(errno.EIO, "synthetic captured-pipe read failure")
    return original_communicate(process, *args, **kwargs)

def control():
    try:
        fixture()
        if scenario == "pipe-error":
            wait_file("fault-injected")
        else:
            send_fixture(driver_identity, signal.SIGINT)
            wait_file("term-received")
            send_fixture(driver_identity, signal.SIGINT)
            (root / "second-interrupt-sent").touch()
        release_leaf()
    except BaseException as error:
        controller_errors.append(type(error).__name__ + ": " + str(error))
        (root / "finish-leaf").touch()

controller = threading.Thread(target=control, daemon=True)
controller.start()
try:
    with site_operation_lock(root / "site", wait=False) as lock_fd:
        try:
            if scenario == "pipe-error":
                with patch.object(subprocess.Popen, "communicate", pipe_failure):
                    run_owned_command(
                        [sys.executable, "-c", leaf_program, str(root)],
                        timeout=10, capture=capture, pass_fds=(lock_fd,),
                    )
            else:
                run_owned_command(
                    [sys.executable, "-c", leaf_program, str(root)],
                    timeout=10, capture=capture, pass_fds=(lock_fd,),
                )
        except BaseException as error:
            report["exception"] = type(error).__name__
            report["errno"] = getattr(error, "errno", None)
        else:
            report["exception"] = None
        observe_return(report)
    report["lock_released_with_live_leaf"] = still_present(fixture()["leaf"])
finally:
    (root / "finish-leaf").touch()
    controller.join(timeout=8)
report.update(
    controller_finished=not controller.is_alive(),
    controller_errors=controller_errors,
    fault_injected=injected,
    second_interrupt_sent=(root / "second-interrupt-sent").exists(),
    signal_handler_restored=signal.getsignal(signal.SIGINT) == initial_handler,
    signal_mask_restored=(
        signal.pthread_sigmask(signal.SIG_BLOCK, set()) == initial_mask
    ),
)
(root / "result.json").write_text(json.dumps(report))
"""

WORKER_POOL_DRIVER = """
from concurrent.futures import ThreadPoolExecutor
from gpu_fault.admin import execution

report = {}
controller_errors = []
initial_handler = signal.getsignal(signal.SIGINT)
initial_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
driver_identity = identity(os.getpid())
directories = [root / f"worker-{index}" for index in range(2)]
cleanup_directories = [directory / "cleanup" for directory in directories]
for directory in [*directories, *cleanup_directories]:
    directory.mkdir(parents=True, exist_ok=True)

def worker(directory, lock_fd):
    observed = {}
    try:
        execution.run_command(
            [sys.executable, "-c", leaf_program, str(directory)],
            timeout_seconds=10, capture=capture, pass_fds=(lock_fd,),
        )
    except BaseException as error:
        observed["exception"] = type(error).__name__
    else:
        observed["exception"] = None
    observed["leaf_alive_at_return"] = still_present(fixture(directory)["leaf"])
    record_event(directory, "command-returned")
    (directory / "owner-returned").touch()
    if interrupt_phase == "cleanup":
        try:
            execution.run_command(
                [sys.executable, "-c", "raise SystemExit(91)"],
                timeout_seconds=3,
            )
        except KeyboardInterrupt:
            observed["forward_command_refused"] = True
        else:
            observed["forward_command_refused"] = False
        cleanup = directory / "cleanup"
        try:
            with execution.cleanup_deadline("fixture cleanup", seconds=6):
                completed = execution.run_command(
                    [sys.executable, "-c", leaf_program, str(cleanup)],
                    timeout_seconds=5, capture=capture, pass_fds=(lock_fd,),
                )
            observed["cleanup_exception"] = None
            observed["cleanup_returncode"] = completed.returncode
            observed["cleanup_alive_at_return"] = still_present(
                fixture(cleanup)["leaf"]
            )
            record_event(cleanup, "command-returned")
            (cleanup / "owner-returned").touch()
        except BaseException as error:
            observed["cleanup_exception"] = type(error).__name__
    return observed

def control():
    try:
        for directory in directories:
            fixture(directory)
        send_fixture(driver_identity, signal.SIGINT)
        for directory in directories:
            wait_file("term-received", directory=directory)
        pending = directories
        if interrupt_phase == "cleanup":
            for directory in directories:
                (directory / "finish-leaf").touch()
            for directory in cleanup_directories:
                fixture(directory)
            pending = cleanup_directories
        send_fixture(driver_identity, signal.SIGINT)
        (root / "second-interrupt-sent").touch()
        for directory in pending:
            release_leaf(directory)
    except BaseException as error:
        controller_errors.append(type(error).__name__ + ": " + str(error))
    finally:
        for directory in [*directories, *cleanup_directories]:
            (directory / "finish-leaf").touch()

controller = threading.Thread(target=control, daemon=True)
pool = ThreadPoolExecutor(max_workers=2)
futures = []
controller.start()
try:
    with site_operation_lock(root / "site", wait=False) as lock_fd:
        try:
            with execution.deployment_deadline(
                "worker operation", 16, recovery_seconds=4
            ):
                futures = [
                    pool.submit(worker, directory, lock_fd)
                    for directory in directories
                ]
                # Let the operation scope own the drain, not executor.shutdown().
                wait_file("second-interrupt-sent")
        except BaseException as error:
            report["exception"] = type(error).__name__
        else:
            report["exception"] = None
        observed_directories = [
            *directories,
            *(cleanup_directories if interrupt_phase == "cleanup" else []),
        ]
        report["live_at_operation_return"] = [
            str(directory.relative_to(root))
            for directory in observed_directories
            if (directory / "ready").exists()
            and still_present(fixture(directory)["leaf"])
        ]
        for directory in observed_directories:
            record_event(directory, "operation-returned")
        (root / "operation-returned").touch()
    report["lock_released_with_live_leaf"] = any(
        (directory / "ready").exists() and still_present(fixture(directory)["leaf"])
        for directory in observed_directories
    )
finally:
    for directory in [*directories, *cleanup_directories]:
        (directory / "finish-leaf").touch()
    controller.join(timeout=8)
    pool.shutdown(wait=True, cancel_futures=True)
report["workers"] = [future.result(timeout=1) for future in futures]
try:
    ensure_supervision_safe()
except ProcessSupervisionLost:
    report["poisoned"] = True
else:
    report["poisoned"] = False
report.update(
    controller_finished=not controller.is_alive(),
    controller_errors=controller_errors,
    second_interrupt_sent=(root / "second-interrupt-sent").exists(),
    cleanup_received_term=[
        (directory / "term-received").exists() for directory in cleanup_directories
    ],
    signal_handler_restored=signal.getsignal(signal.SIGINT) == initial_handler,
    signal_mask_restored=(
        signal.pthread_sigmask(signal.SIG_BLOCK, set()) == initial_mask
    ),
)
(root / "result.json").write_text(json.dumps(report))
"""

STDIO_DRIVER = """
for descriptor in closed_fds:
    os.close(descriptor)
report = {}
try:
    result = run_owned_command(
        [sys.executable, "-c",
         "import sys; print('captured stdout'); "
         "print('captured stderr', file=sys.stderr); sys.exit(7)"],
        timeout=6, capture=True,
    )
    report.update(
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        exception=None,
    )
except BaseException as error:
    report["exception"] = type(error).__name__ + ": " + str(error)
(root / "result.json").write_text(json.dumps(report))
"""

SIGCHLD_DRIVER = """
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
report = {}
try:
    result = run_owned_command(
        [sys.executable, "-c",
         "import sys; print('child completed'); sys.exit(" + str(exit_code) + ")"],
        timeout=6,
    )
    report.update(
        returncode=result.returncode, stdout=result.stdout, exception=None,
    )
except BaseException as error:
    report["exception"] = type(error).__name__ + ": " + str(error)
report["caller_sigchld_ignored"] = signal.getsignal(signal.SIGCHLD) == signal.SIG_IGN
(root / "result.json").write_text(json.dumps(report))
"""


def run_isolated_driver(tmp_path: Path, program: str) -> dict[str, Any]:
    # Only this driver's nested owners are faulted. The outer public API owns
    # every fixture descendant and reaps them even when a regression returns early.
    result = run_owned_command(
        [
            sys.executable,
            "-c",
            DRIVER_IMPORTS + f"\nroot = Path({str(tmp_path)!r})\n" + program,
        ],
        timeout=25,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    report_path = tmp_path / "result.json"
    assert report_path.is_file(), "isolated driver did not record its outcome"
    report = json.loads(report_path.read_text())
    assert isinstance(report, dict), "isolated driver returned a malformed outcome"
    for identities in sorted(tmp_path.rglob("identities.json")):
        for name, item in json.loads(identities.read_text()).items():
            path = Path(f"/proc/{item['pid']}/stat")
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                continue
            assert fields[19] != item["start"], (
                f"outer test owner failed to reap fixture {name}"
            )
    return dict(report)


@pytest.mark.parametrize("capture", [True, False])
def test_pipe_error_after_guardian_loss_waits_for_the_surviving_owner(
    tmp_path: Path, capture: bool
) -> None:
    report = run_isolated_driver(
        tmp_path,
        f"scenario = 'pipe-error'\ncapture = {capture!r}\nleaf_program = {LEAF!r}\n"
        + DRIVER_HELPERS
        + EXCEPTION_DRIVER,
    )
    assert report["controller_finished"], "fixture controller did not terminate"
    assert report["controller_errors"] == [], report["controller_errors"]
    assert report["fault_injected"], "the guardian-loss pipe fault never ran"
    assert report["exception"] == "OSError", report
    assert report["errno"] == errno.EIO, report
    assert not report["leaf_alive_at_return"], (
        "pipe failure returned before the surviving owner reaped its resistant leaf"
    )
    assert not report["lock_released_with_live_leaf"], (
        "site lock was released while the failed command could still mutate"
    )
    assert not report["poisoned"], "proven surviving-owner cleanup poisoned the driver"
    assert (tmp_path / "events.log").read_text().splitlines() == [
        "leaf-final-mutation",
        "owner-returned",
    ], "the leaf's final write did not precede ordinary failure returning"


@pytest.mark.parametrize("capture", [True, False])
def test_repeated_ctrl_c_cannot_release_the_site_lock_before_child_reaping(
    tmp_path: Path, capture: bool
) -> None:
    report = run_isolated_driver(
        tmp_path,
        f"scenario = 'interrupt'\ncapture = {capture!r}\nleaf_program = {LEAF!r}\n"
        + DRIVER_HELPERS
        + EXCEPTION_DRIVER,
    )
    assert report["controller_finished"], "fixture controller did not terminate"
    assert report["controller_errors"] == [], report["controller_errors"]
    assert report["second_interrupt_sent"], "cleanup was not interrupted a second time"
    assert report["exception"] == "KeyboardInterrupt", report
    assert not report["leaf_alive_at_return"], (
        "the second Ctrl-C bypassed command-tree completion proof"
    )
    assert not report["lock_released_with_live_leaf"], (
        "interrupted cleanup released the site lock with a live command"
    )
    assert report["signal_handler_restored"], (
        "cleanup changed the caller's SIGINT handler"
    )
    assert report["signal_mask_restored"], "cleanup changed the caller's signal mask"
    assert not report["poisoned"], "fully reaped interruption poisoned the driver"
    assert (tmp_path / "events.log").read_text().splitlines() == [
        "leaf-final-mutation",
        "owner-returned",
    ], "the leaf wrote after interruption returned to the caller"


@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("interrupt_phase", ["commands", "cleanup"])
def test_operation_scope_drains_worker_commands_across_repeated_interrupts(
    tmp_path: Path, capture: bool, interrupt_phase: str
) -> None:
    report = run_isolated_driver(
        tmp_path,
        f"capture = {capture!r}\ninterrupt_phase = {interrupt_phase!r}\n"
        f"leaf_program = {LEAF!r}\n" + DRIVER_HELPERS + WORKER_POOL_DRIVER,
    )
    assert report["controller_finished"], "worker controller did not terminate"
    assert report["controller_errors"] == [], report
    assert report["second_interrupt_sent"], "operation did not receive both interrupts"
    assert report["exception"] == "KeyboardInterrupt", report
    assert report["live_at_operation_return"] == [], (
        "the operation scope returned before all worker command trees were reaped"
    )
    assert not report["lock_released_with_live_leaf"], (
        "the operation released its site lock while a worker command could mutate"
    )
    assert report["signal_handler_restored"], "operation changed the SIGINT handler"
    assert report["signal_mask_restored"], "operation changed the caller's signal mask"
    assert not report["poisoned"], (
        "fully reaped worker cancellation poisoned the driver"
    )
    assert len(report["workers"]) == 2
    for index, worker in enumerate(report["workers"]):
        assert worker["exception"] == "KeyboardInterrupt", worker
        assert not worker["leaf_alive_at_return"], (
            "a worker returned before its resistant command was reaped"
        )
        directory = tmp_path / f"worker-{index}"
        events = (directory / "events.log").read_text().splitlines()
        assert sorted(events) == [
            "command-returned",
            "leaf-final-mutation",
            "operation-returned",
        ]
        assert events[0] == "leaf-final-mutation", (
            "a command wrote after its worker or operation scope returned"
        )
        if interrupt_phase == "cleanup":
            assert worker["forward_command_refused"], (
                "a cancelled operation admitted new forward work"
            )
            assert worker["cleanup_exception"] is None, worker
            assert worker["cleanup_returncode"] == 0
            assert not worker["cleanup_alive_at_return"], (
                "a recovery command returned without quiescence"
            )
            cleanup_events = (
                (directory / "cleanup" / "events.log").read_text().splitlines()
            )
            assert sorted(cleanup_events) == sorted(events)
            assert cleanup_events[0] == "leaf-final-mutation", (
                "a recovery command wrote after its worker or operation returned"
            )
    assert report["cleanup_received_term"] == [False, False], (
        "repeated interruption cancelled commands admitted for recovery"
    )


@pytest.mark.parametrize(
    "closed_fds",
    [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)],
    ids=[
        "stdin",
        "stdout",
        "stderr",
        "stdin-stdout",
        "stdin-stderr",
        "stdout-stderr",
        "all-stdio",
    ],
)
def test_closed_caller_stdio_cannot_alias_completion_report_descriptors(
    tmp_path: Path, closed_fds: tuple[int, ...]
) -> None:
    report = run_isolated_driver(
        tmp_path, f"closed_fds = {closed_fds!r}\n" + STDIO_DRIVER
    )
    assert report["exception"] is None, report
    assert report["returncode"] == 7, "stdio repair lost the actual command exit status"
    assert report["stdout"] == "captured stdout\n", (
        "completion reports replaced or contaminated captured stdout"
    )
    assert report["stderr"] == "captured stderr\n", (
        "completion reports replaced or contaminated captured stderr"
    )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_inherited_ignored_sigchld_preserves_child_exit_status_and_caller_policy(
    tmp_path: Path, exit_code: int
) -> None:
    report = run_isolated_driver(
        tmp_path, f"exit_code = {exit_code}\n" + SIGCHLD_DRIVER
    )
    assert report["exception"] is None, report
    assert report["returncode"] == exit_code, (
        "inherited SIGCHLD=SIG_IGN discarded the supervised command exit status"
    )
    assert report["stdout"] == "child completed\n"
    assert report["caller_sigchld_ignored"], (
        "supervision changed the caller's signal policy"
    )
