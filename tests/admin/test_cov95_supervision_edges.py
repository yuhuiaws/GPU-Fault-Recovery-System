from __future__ import annotations

import fcntl
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.admin._cov95_supervisor_support import (
    CompletionTransport,
    isolated_supervisor,
)


@pytest.fixture
def context(monkeypatch):
    module = isolated_supervisor(monkeypatch)
    transport = CompletionTransport()
    transport.install(module, monkeypatch)
    return module, transport


def live_guardian(context, monkeypatch, *, identity="owned"):
    module, transport = context
    transport.returncode = None
    descriptor = 100001
    closed = []
    signals = []
    system = SimpleNamespace(**vars(os))

    def open_pidfd(pid):
        assert pid == transport.pid, "registration requested an unrelated pidfd"
        if identity == "pidfd-failure":
            raise PermissionError("example pidfd unavailable")
        return descriptor

    def close(fd):
        if fd == descriptor:
            closed.append(fd)
        else:
            os.close(fd)

    def signal_owned(fd, number):
        assert fd == descriptor, "attempted to signal a host descriptor"
        signals.append(number)
        transport.returncode = 0

    def read_stat():
        if identity == "unreadable":
            raise PermissionError("example procfs unavailable")
        if identity == "malformed":
            return "invalid-stat"
        parent = os.getpid() + (identity == "foreign-parent")
        return f"{transport.pid} (example guardian) S {parent}"

    def path(value):
        if str(value) == f"/proc/{transport.pid}/stat":
            return SimpleNamespace(read_text=read_stat)
        return Path(value)

    system.pidfd_open = open_pidfd
    system.close = close
    monkeypatch.setattr(module, "os", system)
    monkeypatch.setattr(module, "Path", path)
    monkeypatch.setattr(module.signal, "pidfd_send_signal", signal_owned)
    communicate = transport.communicate

    def finished(input_text=None, timeout=None):
        transport.returncode = 0
        return communicate(input_text, timeout)

    monkeypatch.setattr(transport, "communicate", finished)
    return closed, signals


@pytest.mark.parametrize(
    "identity", ["owned", "pidfd-failure", "unreadable", "malformed", "foreign-parent"]
)
def test_guardian_registration_never_signals_an_unproven_pid(
    context, monkeypatch, identity
):
    module, transport = context
    closed, signals = live_guardian(context, monkeypatch, identity=identity)
    assert module.run_owned_command(["example"], timeout=5).returncode == 7
    assert closed == ([] if identity == "pidfd-failure" else [100001])
    assert signals == []
    transport.assert_closed()


@pytest.mark.parametrize("cleanup", [False, True])
def test_interruption_cancels_normal_guardian_but_preserves_cleanup_guardian(
    context, monkeypatch, cleanup
):
    module, transport = context
    closed, signals = live_guardian(context, monkeypatch)
    communicate = transport.communicate

    def interrupt(input_text=None, timeout=None):
        module.signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return communicate(input_text, timeout)

    monkeypatch.setattr(transport, "communicate", interrupt)
    with pytest.raises(KeyboardInterrupt, match="interruption requested"):
        module.run_owned_command(["example"], timeout=5, allow_interrupted=cleanup)
    assert signals == ([] if cleanup else [signal.SIGTERM])
    assert closed == [100001]
    transport.assert_closed()
    module.ensure_supervision_safe()


@pytest.mark.parametrize("interrupted_read", [False, True])
def test_completion_report_read_errors_preserve_independent_proof(
    context, monkeypatch, interrupted_read
):
    module, transport = context
    readers = []
    system = SimpleNamespace(**vars(os))
    once = False

    def pipe():
        pair = os.pipe()
        readers.append(pair[0])
        return pair

    def read(fd, size):
        nonlocal once
        assert fd in readers, "completion reader reached a non-owned descriptor"
        if fd == readers[0] and not once:
            once = True
            if interrupted_read:
                raise InterruptedError("example EINTR")
            raise OSError("example report read failure")
        return os.read(fd, size)

    system.pipe = pipe
    system.read = read
    monkeypatch.setattr(module, "os", system)
    if interrupted_read:
        assert module.run_owned_command(["example"], timeout=5).returncode == 7
    else:
        with pytest.raises(RuntimeError, match="surviving owner"):
            module.run_owned_command(["example"], timeout=5)
    module.ensure_supervision_safe()
    transport.assert_closed()


@pytest.mark.parametrize("fail_duplication", [False, True])
def test_report_descriptors_are_relocated_without_touching_host_stdio(
    context, monkeypatch, fail_duplication
):
    module, transport = context
    closed = []
    allocated = []
    calls = 0
    system = SimpleNamespace(**vars(os))

    def pipe():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("example no more report pipes")
        return 0, 1

    def duplicate(fd, operation, minimum):
        assert operation == fcntl.F_DUPFD_CLOEXEC and minimum == 3
        if fail_duplication and fd == 1:
            raise OSError("example report duplication failure")
        allocated.append(2001 + fd)
        return 2001 + fd

    system.pipe = pipe
    system.close = lambda fd: closed.append(fd)
    monkeypatch.setattr(module, "os", system)
    monkeypatch.setattr(
        module,
        "fcntl",
        SimpleNamespace(F_DUPFD_CLOEXEC=fcntl.F_DUPFD_CLOEXEC, fcntl=duplicate),
    )
    with pytest.raises(OSError, match="report"):
        module.run_owned_command(["example"], timeout=5)
    assert set(closed) == {0, 1, *allocated}
    assert transport.calls == []


@pytest.mark.parametrize("command_error", [False, True])
def test_diagnostic_reader_failure_is_reported_after_command_completion(
    context, command_error
):
    module, transport = context
    if command_error:
        transport.reports[1] = {"status": "supervisor-error", "errno": 13}

    def reject(_line):
        raise ValueError("example diagnostic consumer failure")

    with pytest.raises(PermissionError if command_error else RuntimeError) as error:
        module.run_owned_command(
            ["example"], timeout=5, diagnostics=SimpleNamespace(feed=reject)
        )
    if command_error:
        assert error.value.__notes__ == [
            "deployment diagnostic reader did not finish cleanly"
        ]
    else:
        assert str(error.value) == "deployment diagnostic reader did not finish cleanly"
    module.ensure_supervision_safe()
    transport.assert_closed()


def test_diagnostic_reader_success_keeps_stderr_out_of_communicate(context):
    module, transport = context
    lines = []
    result = module.run_owned_command(
        ["example"], timeout=5, diagnostics=SimpleNamespace(feed=lines.append)
    )
    assert result.returncode == 7
    assert result.stderr == ""
    assert lines == ["diagnostic\n"]
    transport.assert_closed()
