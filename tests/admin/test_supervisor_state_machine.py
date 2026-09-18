from __future__ import annotations

import ctypes
import errno
import io
import os
import signal
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from gpu_fault.admin import process_supervisor as supervisor

SUPERVISOR_PID = 1000
OWNER_PID = 900
PRIMARY_PID = 1001


@dataclass
class Process:
    pid: int
    parent: int
    start: int
    session: int
    ignore_term: bool = False
    status: int | None = None
    returncode: int | None = None

    def stat(self) -> str:
        fields = [
            "Z" if self.status is not None else "S",
            str(self.parent),
            str(self.pid),
            str(self.session),
            *(["0"] * 15),
            str(self.start),
        ]
        return f"{self.pid} (fake ) command) " + " ".join(fields)


class Prctl:
    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, option: int, value: int, arg3: int, arg4: int, arg5: int) -> int:
        assert (arg3, arg4, arg5) == (0, 0, 0)
        self.kernel.prctl_calls.append((option, value))
        if option == self.kernel.prctl_failure:
            return -1
        if option == 1 and self.kernel.stop_during_registration:
            self.kernel.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return 0


class ProcPath:
    def __init__(self, kernel: Kernel, value: str | PurePosixPath) -> None:
        self.kernel = kernel
        self.value = PurePosixPath(value)

    @property
    def name(self) -> str:
        return self.value.name

    def __truediv__(self, other: str) -> ProcPath:
        return ProcPath(self.kernel, self.value / other)

    def iterdir(self) -> Iterator[ProcPath]:
        assert self.value == PurePosixPath("/proc"), "unexpected filesystem access"
        if self.kernel.list_errors:
            raise self.kernel.list_errors.popleft()
        names = [
            "self",
            *(str(pid) for pid in self.kernel.processes),
            *self.kernel.extra_stats,
        ]
        return iter(ProcPath(self.kernel, self.value / name) for name in names)

    def read_text(self) -> str:
        assert self.value.parent.parent == PurePosixPath("/proc")
        assert self.name == "stat", "only virtual process stat files may be read"
        name = self.value.parent.name
        if name in self.kernel.extra_stats:
            value = self.kernel.extra_stats[name]
            if isinstance(value, OSError):
                raise value
            return value
        pid = int(name)
        overrides = self.kernel.stat_overrides.get(pid)
        if overrides:
            override = overrides.popleft()
            if isinstance(override, OSError):
                raise override
            if override is not None:
                return override
        if pid not in self.kernel.processes:
            raise FileNotFoundError(errno.ENOENT, "virtual process exited")
        return self.kernel.processes[pid].stat()


class Kernel:
    """Virtual waitable children and pidfds; no interface falls back to the OS."""

    def __init__(self) -> None:
        self.now = 100.0
        self.parent = OWNER_PID
        self.primary = Process(PRIMARY_PID, SUPERVISOR_PID, 10, SUPERVISOR_PID)
        self.processes = {PRIMARY_PID: self.primary}
        self.events: list[tuple[float, Callable[[], None]]] = []
        self.handlers: dict[int, Callable[[int, object], None]] = {}
        self.prctl_calls: list[tuple[int, int]] = []
        self.prctl_failure: int | None = None
        self.stop_during_registration = False
        self.spawn_calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
        self.wait_calls = 0
        self.reaped: list[int] = []
        self.adopted: list[int] = []
        self.pidfd_attempts: list[int] = []
        self.pidfds: dict[int, tuple[int, int]] = {}
        self.closed: list[int] = []
        self.next_fd = 50
        self.open_errors: dict[int, deque[OSError]] = {}
        self.send_errors: dict[int, deque[OSError]] = {}
        self.before_open: dict[int, Callable[[], None]] = {}
        self.before_send: dict[int, Callable[[], None]] = {}
        self.signal_attempts: list[tuple[int, int]] = []
        self.sent: list[tuple[int, int, float]] = []
        self.list_errors: deque[OSError] = deque()
        self.extra_stats: dict[str, str | OSError] = {}
        self.stat_overrides: dict[int, deque[str | OSError | None]] = {}
        self.stderr = io.StringIO()
        self.os = SimpleNamespace(
            getpid=lambda: SUPERVISOR_PID,
            getppid=lambda: self.parent,
            pidfd_open=self.pidfd_open,
            close=self.close,
            waitpid=self.waitpid,
            waitstatus_to_exitcode=os.waitstatus_to_exitcode,
            WNOHANG=os.WNOHANG,
        )
        self.signal = SimpleNamespace(
            SIGTERM=signal.SIGTERM,
            SIGINT=signal.SIGINT,
            SIGKILL=signal.SIGKILL,
            Signals=signal.Signals,
            signal=self.register_handler,
            pidfd_send_signal=self.pidfd_send_signal,
        )
        self.ctypes = SimpleNamespace(
            CDLL=self.library,
            c_int=ctypes.c_int,
            c_ulong=ctypes.c_ulong,
            get_errno=lambda: errno.EPERM,
        )
        self.sys = SimpleNamespace(platform="linux", stderr=self.stderr)

    def library(self, name: None, *, use_errno: bool) -> SimpleNamespace:
        assert name is None and use_errno
        return SimpleNamespace(prctl=Prctl(self))

    def register_handler(
        self, signum: int, handler: Callable[[int, object], None]
    ) -> None:
        self.handlers[signum] = handler

    def schedule(self, delay: float, event: Callable[[], None]) -> None:
        self.events.append((self.now + delay, event))

    def run_events(self) -> None:
        due = [event for when, event in self.events if when <= self.now]
        self.events = [(when, event) for when, event in self.events if when > self.now]
        for event in due:
            event()

    def sleep(self, seconds: float) -> None:
        assert 0 < seconds <= 1, "supervisor must keep polling during cleanup"
        self.now += 1
        assert self.now < 200, "virtual supervisor never finished cleanup"
        self.run_events()

    def spawn(
        self, arguments: Sequence[str], *, text: bool, pass_fds: Sequence[int]
    ) -> Process:
        assert text, "supervisor must preserve text-mode command execution"
        self.spawn_calls.append((tuple(arguments), tuple(pass_fds)))
        self.run_events()
        return self.primary

    def exit(self, pid: int, returncode: int = 0) -> None:
        self.processes[pid].status = -returncode if returncode < 0 else returncode << 8
        for child in self.processes.values():
            if child.parent == pid:
                child.parent = SUPERVISOR_PID
                self.adopted.append(child.pid)

    def waitpid(self, pid: int, options: int) -> tuple[int, int]:
        assert (pid, options) == (-1, os.WNOHANG)
        assert self.spawn_calls, "cannot reap before starting the virtual command"
        self.wait_calls += 1
        children = [
            process
            for process in self.processes.values()
            if process.parent == SUPERVISOR_PID
        ]
        for child in children:
            if child.status is not None:
                del self.processes[child.pid]
                self.reaped.append(child.pid)
                return child.pid, child.status
        if children:
            return 0, 0
        raise ChildProcessError(errno.ECHILD, "no virtual children")

    def pidfd_open(self, pid: int) -> int:
        self.pidfd_attempts.append(pid)
        if callback := self.before_open.pop(pid, None):
            callback()
        if failures := self.open_errors.get(pid):
            raise failures.popleft()
        if pid == SUPERVISOR_PID:
            start = 1
        elif pid in self.processes:
            start = self.processes[pid].start
        else:
            raise ProcessLookupError(errno.ESRCH, "virtual process exited")
        descriptor = self.next_fd
        self.next_fd += 1
        self.pidfds[descriptor] = (pid, start)
        return descriptor

    def close(self, descriptor: int) -> None:
        assert descriptor in self.pidfds, "closed an unknown or already closed pidfd"
        del self.pidfds[descriptor]
        self.closed.append(descriptor)

    def pidfd_send_signal(self, descriptor: int, signum: int) -> None:
        pid, start = self.pidfds[descriptor]
        self.signal_attempts.append((pid, signum))
        if callback := self.before_send.pop(pid, None):
            callback()
        if failures := self.send_errors.get(pid):
            raise failures.popleft()
        process = self.processes.get(pid)
        if process is None or process.start != start:
            raise ProcessLookupError(errno.ESRCH, "pidfd's process has exited")
        self.sent.append((pid, signum, self.now))
        if signum == signal.SIGKILL or not process.ignore_term:
            self.exit(pid, -signum)

    def reuse_pid(self, pid: int) -> None:
        old = self.processes[pid]
        assert old.parent == PRIMARY_PID, "reuse requires the old parent's reap"
        self.processes[pid] = Process(pid, 1, old.start + 1, pid)


@pytest.fixture
def kernel(monkeypatch: pytest.MonkeyPatch) -> Iterator[Kernel]:
    value = Kernel()
    real_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    # Replace module-local interfaces, not the shared OS modules pytest uses.
    monkeypatch.setattr(supervisor, "os", value.os)
    monkeypatch.setattr(supervisor, "signal", value.signal)
    monkeypatch.setattr(supervisor, "ctypes", value.ctypes)
    monkeypatch.setattr(supervisor, "sys", value.sys)
    monkeypatch.setattr(supervisor, "subprocess", SimpleNamespace(Popen=value.spawn))
    monkeypatch.setattr(
        supervisor,
        "time",
        SimpleNamespace(monotonic=lambda: value.now, sleep=value.sleep),
    )
    monkeypatch.setattr(supervisor, "Path", lambda path: ProcPath(value, path))
    yield value
    assert value.pidfds == {}, "supervisor leaked virtual pidfds"
    assert real_handlers == {
        signum: signal.getsignal(signum) for signum in real_handlers
    }, "in-process supervision changed pytest's signal handlers"


def run_supervisor(
    kernel: Kernel,
    *,
    expires: float = 120.0,
    grace: float = 2.0,
    arguments: Sequence[str] = ("fake-command",),
    pass_fds: Sequence[int] = (),
) -> dict[str, object]:
    return supervisor.supervise(
        arguments, owner=OWNER_PID, expires=expires, pass_fds=pass_fds, grace=grace
    )


@pytest.mark.parametrize("returncode", [0, 7, -signal.SIGSEGV])
def test_primary_exit_is_reaped_and_its_status_is_preserved(
    kernel: Kernel, returncode: int
) -> None:
    kernel.schedule(0, lambda: kernel.exit(PRIMARY_PID, returncode))

    result = run_supervisor(kernel, pass_fds=(17, 19))

    assert result == {"status": "exited", "returncode": returncode}
    assert kernel.spawn_calls == [(("fake-command",), (17, 19))]
    assert kernel.reaped == [PRIMARY_PID]
    assert kernel.sent == []
    assert (36, 1) in kernel.prctl_calls, "subreaper ownership was not requested"
    assert (1, signal.SIGTERM) in kernel.prctl_calls, "parent-death signal is missing"
    assert kernel.pidfd_attempts == [SUPERVISOR_PID]


def test_primary_exit_waits_for_detached_and_adopted_descendants(
    kernel: Kernel,
) -> None:
    kernel.processes[1002] = Process(1002, PRIMARY_PID, 20, 1002, ignore_term=True)
    kernel.processes[1003] = Process(1003, 1002, 30, 1003, ignore_term=True)
    kernel.processes[2000] = Process(2000, 1, 40, 2000)
    kernel.schedule(0, lambda: kernel.exit(PRIMARY_PID, 7))

    result = run_supervisor(kernel)

    assert result == {"status": "exited", "returncode": 7}
    assert kernel.adopted == [1002, 1003]
    assert kernel.reaped == [PRIMARY_PID, 1002, 1003]
    assert set(kernel.processes) == {2000}, "unrelated process was changed"
    assert kernel.sent == [
        (1002, signal.SIGTERM, 100.0),
        (1003, signal.SIGTERM, 100.0),
        (1002, signal.SIGKILL, 102.0),
        (1003, signal.SIGKILL, 102.0),
    ]


def test_deadline_sends_term_once_then_kill_after_the_grace(kernel: Kernel) -> None:
    kernel.primary.ignore_term = True

    result = run_supervisor(kernel, expires=102, grace=3)

    assert result == {"status": "timeout", "returncode": -signal.SIGKILL}
    assert kernel.sent == [
        (PRIMARY_PID, signal.SIGTERM, 102.0),
        (PRIMARY_PID, signal.SIGKILL, 105.0),
    ]
    assert kernel.reaped == [PRIMARY_PID]
    assert kernel.now == 106


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_stop_signal_cleans_up_before_the_deadline(
    kernel: Kernel, signum: signal.Signals
) -> None:
    kernel.schedule(1, lambda: kernel.handlers[signum](signum, None))

    result = run_supervisor(kernel)

    assert result == {"status": "timeout", "returncode": -signal.SIGTERM}
    assert kernel.sent == [(PRIMARY_PID, signal.SIGTERM, 101.0)]
    assert kernel.reaped == [PRIMARY_PID]
    assert kernel.now < 120


@pytest.mark.parametrize("stage", ["open", "send"])
@pytest.mark.parametrize("number", [errno.ESRCH, errno.EPERM, errno.EMFILE])
def test_failed_pidfd_operations_are_retried(
    kernel: Kernel, stage: str, number: int
) -> None:
    failures = kernel.open_errors if stage == "open" else kernel.send_errors
    failures[PRIMARY_PID] = deque([OSError(number, "transient virtual failure")])

    result = run_supervisor(kernel, expires=101, grace=10)

    assert result == {"status": "timeout", "returncode": -signal.SIGTERM}
    assert kernel.pidfd_attempts.count(PRIMARY_PID) == 2
    assert kernel.sent == [(PRIMARY_PID, signal.SIGTERM, 102.0)]
    assert kernel.reaped == [PRIMARY_PID]
    assert len(kernel.signal_attempts) == (1 if stage == "open" else 2)


@pytest.mark.parametrize("stage", ["open", "send"])
def test_pid_reuse_does_not_signal_an_unrelated_replacement(
    kernel: Kernel, stage: str
) -> None:
    kernel.primary.ignore_term = True
    kernel.processes[1002] = Process(1002, PRIMARY_PID, 20, 1002)
    hooks = kernel.before_open if stage == "open" else kernel.before_send
    hooks[1002] = lambda: kernel.reuse_pid(1002)

    result = run_supervisor(kernel, expires=101)

    assert result == {"status": "timeout", "returncode": -signal.SIGKILL}
    assert kernel.processes[1002].start == 21
    assert kernel.processes[1002].status is None
    assert kernel.processes[1002].parent == 1
    assert {pid for pid, _signum, _time in kernel.sent} == {PRIMARY_PID}
    assert kernel.reaped == [PRIMARY_PID]


@pytest.mark.parametrize(
    "bad_stat", [FileNotFoundError(errno.ENOENT, "transient stat failure"), "malformed"]
)
def test_failed_identity_read_is_retried_without_leaking_a_pidfd(
    kernel: Kernel, bad_stat: str | OSError
) -> None:
    kernel.stat_overrides[PRIMARY_PID] = deque([None, bad_stat])

    result = run_supervisor(kernel, expires=101, grace=10)

    assert result == {"status": "timeout", "returncode": -signal.SIGTERM}
    assert kernel.pidfd_attempts.count(PRIMARY_PID) == 2
    assert kernel.sent == [(PRIMARY_PID, signal.SIGTERM, 102.0)]


def test_proc_scan_failure_and_unreadable_entries_do_not_end_cleanup(
    kernel: Kernel,
) -> None:
    kernel.list_errors.append(PermissionError(errno.EACCES, "transient proc failure"))
    kernel.extra_stats = {
        "2000": FileNotFoundError(errno.ENOENT, "process disappeared"),
        "2001": "malformed",
        "2002": "2002 (bad parent) S invalid " + "0 " * 18,
    }

    result = run_supervisor(kernel, expires=101, grace=10)

    assert result == {"status": "timeout", "returncode": -signal.SIGTERM}
    assert kernel.sent == [(PRIMARY_PID, signal.SIGTERM, 102.0)]
    assert kernel.reaped == [PRIMARY_PID]


@pytest.mark.parametrize(
    "expires,grace",
    [
        (float("nan"), 2),
        (float("inf"), 2),
        (float("-inf"), 2),
        (120, 0),
        (120, -1),
        (120, 10.1),
        (120, float("nan")),
        (120, float("inf")),
    ],
)
def test_invalid_budgets_are_rejected_before_any_system_interface(
    kernel: Kernel, expires: float, grace: float
) -> None:
    with pytest.raises(ValueError, match="invalid command deadline"):
        run_supervisor(kernel, expires=expires, grace=grace)

    assert kernel.handlers == {}
    assert kernel.prctl_calls == []
    assert kernel.spawn_calls == []
    assert kernel.pidfd_attempts == []


@pytest.mark.parametrize("expires", [0.0, 99.0, 100.0])
def test_expired_deadline_does_not_spawn_or_reap(
    kernel: Kernel, expires: float
) -> None:
    result = run_supervisor(kernel, expires=expires)

    assert result == {"status": "timeout", "returncode": 124}
    assert kernel.spawn_calls == []
    assert kernel.wait_calls == 0
    assert kernel.sent == []


def test_owner_death_during_startup_is_rejected_before_spawn(kernel: Kernel) -> None:
    kernel.parent = 1

    with pytest.raises(OSError, match="owner exited before startup") as error:
        run_supervisor(kernel)

    assert error.value.errno == errno.ESRCH
    assert kernel.spawn_calls == []
    assert kernel.wait_calls == 0
    assert kernel.pidfd_attempts == []


def test_stop_during_registration_prevents_spawn(kernel: Kernel) -> None:
    kernel.stop_during_registration = True

    result = run_supervisor(kernel)

    assert result == {"status": "timeout", "returncode": 124}
    assert kernel.spawn_calls == []
    assert kernel.wait_calls == 0


@pytest.mark.parametrize("option", [36, 1])
def test_failed_process_ownership_registration_prevents_spawn(
    kernel: Kernel, option: int
) -> None:
    kernel.prctl_failure = option

    with pytest.raises(OSError, match="cannot establish process ownership") as error:
        run_supervisor(kernel)

    assert error.value.errno == errno.EPERM
    assert kernel.spawn_calls == []
    assert kernel.wait_calls == 0
    assert kernel.pidfd_attempts == []


@pytest.mark.parametrize("missing", ["linux", "pidfd_send_signal"])
def test_unsupported_platform_prevents_spawn(kernel: Kernel, missing: str) -> None:
    if missing == "linux":
        kernel.sys.platform = "darwin"
    else:
        del kernel.signal.pidfd_send_signal

    with pytest.raises(OSError, match="requires Linux pidfds") as error:
        run_supervisor(kernel)

    assert error.value.errno == errno.ENOSYS
    assert kernel.prctl_calls == []
    assert kernel.spawn_calls == []


def test_unavailable_pidfd_support_prevents_spawn(kernel: Kernel) -> None:
    kernel.open_errors[SUPERVISOR_PID] = deque(
        [OSError(errno.ENOSYS, "pidfds unavailable")]
    )

    with pytest.raises(OSError, match="pidfds unavailable"):
        run_supervisor(kernel)

    assert kernel.spawn_calls == []
    assert kernel.wait_calls == 0
    assert kernel.pidfd_attempts == [SUPERVISOR_PID]


@pytest.mark.parametrize("state", ["running", "cleanup"])
def test_progress_reports_state_without_command_arguments(
    kernel: Kernel, state: str
) -> None:
    if state == "cleanup":
        kernel.processes[1002] = Process(1002, PRIMARY_PID, 20, 1002, ignore_term=True)
        kernel.schedule(29, lambda: kernel.exit(PRIMARY_PID))
    else:
        kernel.schedule(32, lambda: kernel.exit(PRIMARY_PID))

    result = run_supervisor(
        kernel,
        expires=160,
        arguments=("/private/custom-command", "argument-not-for-logging"),
    )

    assert result == {"status": "exited", "returncode": 0}
    assert kernel.stderr.getvalue() == (
        f"deployment-wait command=command state={state} elapsed=30s remaining=30s\n"
    )


def test_closed_diagnostic_sink_cannot_abandon_a_live_process_tree(kernel: Kernel):
    class ClosedConsole(io.StringIO):
        def write(self, _text):
            raise BrokenPipeError("closed console")

    kernel.sys.stderr = ClosedConsole()
    kernel.schedule(32, lambda: kernel.exit(PRIMARY_PID))
    result = run_supervisor(kernel, expires=160)
    assert result == {"status": "exited", "returncode": 0}
    assert PRIMARY_PID in kernel.reaped, "diagnostic failure skipped child reaping"
