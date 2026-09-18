from __future__ import annotations

import errno
import io
import json
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpu_fault.admin import process_supervisor as supervisor
from tests.admin import test_supervisor_state_machine as kernel_support
from tests.admin._cov95_supervisor_support import (
    CompletionTransport,
    isolated_supervisor,
)
from tests.admin.test_supervisor_state_machine import (
    OWNER_PID,
    PRIMARY_PID,
    SUPERVISOR_PID,
)

kernel = kernel_support.kernel


@pytest.fixture
def completion(monkeypatch):
    module = isolated_supervisor(monkeypatch)
    transport = CompletionTransport()
    transport.install(module, monkeypatch)
    return module, transport


@pytest.mark.parametrize(
    "guardian,command,exception",
    [
        (
            {"status": "timeout", "returncode": 124},
            {"status": "exited", "returncode": 0},
            subprocess.TimeoutExpired,
        ),
        (
            {"status": "exited", "returncode": 0},
            {"status": "timeout", "returncode": 124},
            subprocess.TimeoutExpired,
        ),
        (
            {"status": "exited", "returncode": 9},
            {"status": "exited", "returncode": 0},
            RuntimeError,
        ),
        (
            {"status": "exited", "returncode": 0},
            {"status": "supervisor-error", "errno": errno.EACCES},
            PermissionError,
        ),
        (
            {"status": "exited", "returncode": 0},
            {"status": "supervisor-error", "errno": "invalid"},
            RuntimeError,
        ),
        (b"invalid-json", {"status": "exited", "returncode": 0}, RuntimeError),
        ([], {"status": "exited", "returncode": 0}, RuntimeError),
        (
            {"status": "exited", "returncode": True},
            {"status": "exited", "returncode": 0},
            RuntimeError,
        ),
        (b"x" * 5000, {"status": "exited", "returncode": 0}, RuntimeError),
        ({"status": "exited", "returncode": 0}, b"\xff", RuntimeError),
        ({"status": "exited", "returncode": 0}, {}, RuntimeError),
    ],
)
def test_owned_command_validates_independent_reports(
    completion, guardian, command, exception
):
    module, transport = completion
    transport.reports = [guardian, command]
    with pytest.raises(exception):
        module.run_owned_command(["fake-command"], timeout=5)
    module.ensure_supervision_safe()
    transport.assert_closed()


@pytest.mark.parametrize("reports", [[{}, {}], [b"", b""], [b"invalid", []]])
def test_loss_of_both_proofs_refuses_further_commands(completion, reports):
    module, transport = completion
    transport.reports = reports
    with pytest.raises(module.ProcessSupervisionLost, match="ownership was lost"):
        module.run_owned_command(["fake-command"], timeout=5)
    with pytest.raises(module.ProcessSupervisionLost, match="ownership was lost"):
        module.ensure_supervision_safe(allow_interrupted=True)
    with pytest.raises(module.ProcessSupervisionLost):
        module.run_owned_command(["must-not-start"], timeout=5)
    assert len(transport.calls) == 1
    transport.assert_closed()


@pytest.mark.parametrize("capture", [True, False])
def test_owned_command_success_closes_reports_and_preserves_exit(completion, capture):
    module, transport = completion
    result = module.run_owned_command(
        ["fake-command"], timeout=5, capture=capture, input_text="example-input"
    )
    assert result.returncode == 7
    assert result.stdout == ("output" if capture else "")
    assert result.stderr == ("diagnostic\n" if capture else "")
    transport.assert_closed()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_owned_command_rejects_invalid_timeout_before_spawn(completion, timeout):
    module, transport = completion
    with pytest.raises(ValueError, match="invalid command timeout"):
        module.run_owned_command(["fake"], timeout=timeout)
    assert not transport.calls, "invalid timeout started a command"


@pytest.mark.parametrize("expires", [float("nan"), float("inf"), float("-inf")])
def test_owned_command_rejects_invalid_absolute_deadline(completion, expires):
    module, transport = completion
    with pytest.raises(ValueError, match="invalid absolute"):
        module.run_owned_command(["fake"], timeout=5, expires_at=expires)
    assert not transport.calls, "invalid absolute deadline started a command"


def test_owned_command_expired_absolute_deadline_never_spawns(completion):
    module, transport = completion
    with pytest.raises(subprocess.TimeoutExpired):
        module.run_owned_command(["fake"], timeout=5, expires_at=0)
    assert not transport.calls, "expired absolute deadline started a command"


def test_spawn_failure_closes_all_report_writers(completion):
    module, transport = completion
    transport.spawn_error = OSError(errno.EMFILE, "modeled descriptor exhaustion")
    with pytest.raises(OSError, match="descriptor exhaustion"):
        module.run_owned_command(["fake"], timeout=5)
    transport.assert_closed()
    module.ensure_supervision_safe()


@pytest.mark.parametrize(
    "error", [OSError("pipe read failed"), InterruptedError("capture interrupted")]
)
def test_capture_error_still_requires_completion_and_closes_streams(completion, error):
    module, transport = completion
    transport.communicate_error = error
    with pytest.raises(type(error), match=str(error)):
        module.run_owned_command(["fake"], timeout=5)
    transport.assert_closed()
    module.ensure_supervision_safe()


def test_incremental_communicate_wait_does_not_lose_stdin(completion):
    module, transport = completion
    transport.communicate_error = subprocess.TimeoutExpired(["fake"], 0.1)
    result = module.run_owned_command(["fake"], timeout=5, input_text="example-input")
    assert result.returncode == 7
    transport.assert_closed()


def install_fork_model(kernel, monkeypatch, outcome):
    masks = []
    writes = []
    kernel.signal.SIG_BLOCK = signal.SIG_BLOCK
    kernel.signal.SIG_SETMASK = signal.SIG_SETMASK
    kernel.signal.pthread_sigmask = lambda how, mask: masks.append((how, mask)) or set()
    kernel.os.write = lambda descriptor, payload: writes.append((descriptor, payload))
    monkeypatch.setattr(
        supervisor, "threading", SimpleNamespace(active_count=lambda: 1)
    )

    def fork():
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome:
            kernel.spawn_calls.append((("guardian-fork",), ()))
            kernel.run_events()
        else:
            kernel.parent = SUPERVISOR_PID
        return outcome

    kernel.os.fork = fork
    return masks, writes


def test_guardian_fork_parent_restores_signal_mask_and_waits(kernel, monkeypatch):
    masks, writes = install_fork_model(kernel, monkeypatch, PRIMARY_PID)
    kernel.schedule(0, lambda: kernel.exit(PRIMARY_PID))
    result = supervisor.supervise(
        ["fake"],
        owner=OWNER_PID,
        expires=120,
        pass_fds=(),
        command_report_fd=90,
        guardian_report_fd=91,
    )
    assert result == {"status": "exited", "returncode": 0}
    assert masks == [
        (signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}),
        (signal.SIG_SETMASK, set()),
    ]
    assert writes == []


def test_guardian_fork_failure_restores_mask(kernel, monkeypatch):
    masks, writes = install_fork_model(
        kernel, monkeypatch, OSError(errno.EAGAIN, "fork failed")
    )
    with pytest.raises(OSError, match="fork failed"):
        supervisor.supervise(
            ["fake"],
            owner=OWNER_PID,
            expires=120,
            pass_fds=(),
            command_report_fd=90,
            guardian_report_fd=91,
        )
    assert masks[-1] == (signal.SIG_SETMASK, set())
    assert writes == [] and kernel.spawn_calls == []


@pytest.mark.parametrize("failed", [False, True])
def test_forked_owner_writes_its_own_report_and_exits(kernel, monkeypatch, failed):
    masks, writes = install_fork_model(kernel, monkeypatch, 0)
    kernel.pidfds[91] = (SUPERVISOR_PID, 1)
    kernel.schedule(0, lambda: kernel.exit(PRIMARY_PID, 3))
    if failed:
        original = kernel.os.fork

        def fork():
            value = original()
            kernel.parent = OWNER_PID
            return value

        kernel.os.fork = fork

    class ChildExit(BaseException):
        pass

    def exit_child(code):
        assert code == 0
        raise ChildExit

    kernel.os._exit = exit_child
    with pytest.raises(ChildExit):
        supervisor.supervise(
            ["fake"],
            owner=OWNER_PID,
            expires=120,
            pass_fds=(17,),
            command_report_fd=90,
            guardian_report_fd=91,
        )
    assert len(writes) == 1 and writes[0][0] == 90
    assert json.loads(writes[0][1]) == (
        {"status": "supervisor-error", "errno": errno.ESRCH}
        if failed
        else {"status": "exited", "returncode": 3}
    )
    assert 91 in kernel.closed
    if not failed:
        assert masks[-1] == (signal.SIG_SETMASK, set())


def test_guardian_requires_single_thread_and_its_report_descriptor(kernel, monkeypatch):
    with pytest.raises(ValueError, match="completion descriptor"):
        supervisor.supervise(
            ["fake"], owner=OWNER_PID, expires=120, pass_fds=(), command_report_fd=90
        )
    monkeypatch.setattr(
        supervisor, "threading", SimpleNamespace(active_count=lambda: 2)
    )
    with pytest.raises(OSError, match="single-threaded"):
        supervisor.supervise(
            ["fake"],
            owner=OWNER_PID,
            expires=120,
            pass_fds=(),
            command_report_fd=90,
            guardian_report_fd=91,
        )
    assert not kernel.spawn_calls, "invalid guardian configuration spawned a child"


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("separator", [False, True])
def test_supervisor_main_parses_and_emits_bounded_report(
    kernel, monkeypatch, failed, separator
):
    kernel.signal.SIGCHLD = signal.SIGCHLD
    kernel.signal.SIG_DFL = signal.SIG_DFL
    kernel.signal.SIG_UNBLOCK = signal.SIG_UNBLOCK
    masks = []
    kernel.signal.pthread_sigmask = lambda how, mask: masks.append((how, mask))
    writes = []
    kernel.os.write = lambda fd, payload: writes.append((fd, json.loads(payload)))
    kernel.pidfds[91] = (SUPERVISOR_PID, 1)
    if failed:
        kernel.prctl_failure = 36
    else:
        kernel.schedule(0, lambda: kernel.exit(PRIMARY_PID, 4))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervisor",
            "--owner",
            str(OWNER_PID),
            "--expires",
            "120",
            "--report-fd",
            "91",
            "--pass-fds",
            "17,19",
            *(["--"] if separator else []),
            "fake-command",
        ],
    )
    supervisor.main()
    assert writes == [
        (
            91,
            {"status": "supervisor-error", "errno": errno.EPERM}
            if failed
            else {"status": "exited", "returncode": 4},
        )
    ]
    assert masks == [(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})]
    assert 91 in kernel.closed


@pytest.mark.parametrize("kind", ["custom", "closed", "subclass"])
def test_diagnostic_writer_refuses_untrusted_stream_implementations(kind):
    class Custom:
        def write(self, _text):
            pytest.fail("untrusted writer was called")

    class CustomWrapper(io.TextIOWrapper):
        def write(self, _text):
            pytest.fail("custom text wrapper was called")

    if kind == "custom":
        stream = Custom()
    elif kind == "closed":
        stream = io.StringIO()
        stream.close()
    else:
        stream = CustomWrapper(io.BytesIO())
    assert supervisor.write_diagnostic("example", stream=stream) is False
