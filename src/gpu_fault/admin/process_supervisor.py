"""Own a command tree with two independent Linux subreapers and completion proofs.

The surviving owner adopts and reaps children if either supervisor dies. An
unproven tree is fatal, not an ordinary failure that could initiate compensation.
This module also runs under isolated Python and uses only the stdlib.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TextIO

if TYPE_CHECKING:
    from gpu_fault.admin.diagnostics import DriverDiagnostics

TERMINATION_GRACE_SECONDS = 10.0
POLL_SECONDS = 0.02
DIAGNOSTIC_JOIN_SECONDS = 2.0
REPORT_MAX_BYTES = 4096
_SUPERVISION_LOST = threading.Event()
_INTERRUPTED = threading.Event()
_GUARDIANS_LOCK = threading.RLock()
_GUARDIANS: dict[int, tuple[int | None, bool]] = {}
_INTERRUPTION_SCOPES = 0


class ProcessSupervisionLost(BaseException):
    """No surviving owner proved quiescence; ordinary recovery is unsafe."""


def ensure_supervision_safe(*, allow_interrupted: bool = False) -> None:
    if _SUPERVISION_LOST.is_set():
        raise ProcessSupervisionLost(
            "deployment process ownership was lost; further commands are refused"
        )
    if _INTERRUPTED.is_set() and not allow_interrupted:
        raise KeyboardInterrupt("deployment interruption requested")


def _above_stdio(descriptor: int) -> int:
    if descriptor > 2:
        return descriptor
    moved = int(fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3))
    os.close(descriptor)
    return moved


def _report_pipe() -> tuple[int, int]:
    read_fd, write_fd = os.pipe()
    try:
        read_fd = _above_stdio(read_fd)
        write_fd = _above_stdio(write_fd)
        return read_fd, write_fd
    except BaseException:
        for descriptor in (read_fd, write_fd):
            with contextlib.suppress(OSError):
                os.close(descriptor)
        raise


def _cancel_guardians() -> None:
    with _GUARDIANS_LOCK:
        for descriptor, cleanup in _GUARDIANS.values():
            if descriptor is not None and not cleanup:
                with contextlib.suppress(OSError):
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM)


@contextlib.contextmanager
def interruption_scope(*, wait_all: bool = False) -> Iterator[None]:
    """Request cancellation at safe boundaries; repeated SIGINT never unwinds cleanup."""
    global _INTERRUPTION_SCOPES
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if _INTERRUPTION_SCOPES:
        _INTERRUPTION_SCOPES += 1
        try:
            yield
        finally:
            _INTERRUPTION_SCOPES -= 1
        return
    _INTERRUPTED.clear()
    previous = signal.getsignal(signal.SIGINT)

    def interrupt(_signum: int, _frame: object) -> None:
        _INTERRUPTED.set()
        _cancel_guardians()

    signal.signal(signal.SIGINT, interrupt)
    _INTERRUPTION_SCOPES = 1
    failure: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        failure = exc
        raise
    finally:
        interrupted = _INTERRUPTED.is_set()
        try:
            if interrupted or wait_all:
                while True:
                    with _GUARDIANS_LOCK:
                        if not _GUARDIANS:
                            break
                    time.sleep(POLL_SECONDS)
            ensure_supervision_safe(allow_interrupted=True)
        finally:
            _INTERRUPTION_SCOPES = 0
            signal.signal(signal.SIGINT, previous)
            _INTERRUPTED.clear()
        if failure is None and interrupted:
            raise KeyboardInterrupt("deployment interruption requested")


def _register_guardian(process: subprocess.Popen[str], *, cleanup: bool) -> int | None:
    descriptor: int | None = None
    if process.poll() is None:
        try:
            descriptor = os.pidfd_open(process.pid)
            descriptor = _above_stdio(descriptor)
            fields = (
                Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
            )
            if int(fields[1]) != os.getpid() or process.poll() is not None:
                os.close(descriptor)
                descriptor = None
        except (OSError, ValueError, IndexError):
            if descriptor is not None:
                os.close(descriptor)
            descriptor = None
    with _GUARDIANS_LOCK:
        _GUARDIANS[id(process)] = (descriptor, cleanup)
        if _INTERRUPTED.is_set():
            _cancel_guardians()
    return descriptor


def _unregister_guardian(process: subprocess.Popen[str]) -> None:
    with _GUARDIANS_LOCK:
        descriptor, _cleanup = _GUARDIANS.pop(id(process), (None, False))
        if descriptor is not None:
            os.close(descriptor)


@dataclass
class _CompletionReport:
    descriptor: int
    data: bytearray = field(default_factory=bytearray)
    ended: bool = False
    invalid: bool = False

    def read(self) -> None:
        while not self.ended:
            try:
                chunk = os.read(self.descriptor, REPORT_MAX_BYTES)
            except BlockingIOError:
                return
            except InterruptedError:
                continue
            except OSError:
                self.invalid = True
                self.ended = True
                return
            if not chunk:
                self.ended = True
                return
            if len(self.data) + len(chunk) > REPORT_MAX_BYTES:
                self.invalid = True
            if not self.invalid:
                self.data.extend(chunk)

    def value(self) -> dict[str, object] | None:
        if not self.ended or self.invalid:
            return None
        try:
            value = json.loads(self.data)
        except (ValueError, UnicodeError):
            return None
        return value if isinstance(value, dict) else None

    def proves_quiescence(self) -> bool:
        value = self.value()
        return (
            value is not None
            and value.get("status") in {"exited", "timeout"}
            and type(value.get("returncode")) is int
        )


class _WaitableChild(Protocol):
    pid: int
    returncode: int | None


@dataclass
class _ForkedChild:
    pid: int
    returncode: int | None = None


def write_diagnostic(
    text: str, *, stream: TextIO | None = None, final: bool = False
) -> bool:
    """Best-effort sanitized stderr; pipe backpressure must never block supervision."""
    stream = sys.stderr if stream is None else stream
    descriptor: int | None = None
    owned: int | None = None
    try:
        # These standard in-memory implementations cannot wait on external IO.
        if isinstance(stream, io.StringIO):
            if type(stream).write is not io.StringIO.write:
                return False
            io.StringIO.write(stream, text)
            return True
        if isinstance(stream, io.TextIOWrapper):
            buffer = io.TextIOWrapper.buffer.__get__(stream)
            if type(buffer) is io.BytesIO:
                if type(stream).write is not io.TextIOWrapper.write:
                    return False
                io.TextIOWrapper.write(stream, text)
                io.TextIOWrapper.flush(stream)
                return True
        else:
            return False
        descriptor = stream.fileno()
        mode = os.fstat(descriptor).st_mode
        if stat.S_ISFIFO(mode) or stat.S_ISCHR(mode):
            # dup() would share O_NONBLOCK with every other writer. Reopening
            # procfs gives this diagnostic its own open-file description.
            owned = os.open(
                f"/proc/self/fd/{descriptor}",
                os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC,
            )
            descriptor = owned
        elif not (final and stat.S_ISREG(mode)):
            return False
        data = text.encode("utf-8")
        if len(data) > 4096:
            tail = data[-4000:].decode("utf-8", errors="ignore").encode("utf-8")
            data = b"[earlier diagnostic omitted]\n" + tail
        return os.write(descriptor, data) == len(data)
    except (OSError, ValueError):
        return False
    finally:
        if owned is not None:
            try:
                os.close(owned)
            except OSError:
                pass


def _configure_subreaper(owner: int) -> None:
    if sys.platform != "linux" or not hasattr(signal, "pidfd_send_signal"):
        raise OSError(errno.ENOSYS, "deployment supervision requires Linux pidfds")
    library = ctypes.CDLL(None, use_errno=True)
    prctl = library.prctl
    prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    prctl.restype = ctypes.c_int
    # PR_SET_CHILD_SUBREAPER and PR_SET_PDEATHSIG survive until this owner exits.
    for option, value in ((36, 1), (1, signal.SIGTERM)):
        if prctl(option, value, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "cannot establish process ownership")
    if os.getppid() != owner:
        raise OSError(errno.ESRCH, "deployment command owner exited before startup")
    descriptor = os.pidfd_open(os.getpid())
    os.close(descriptor)


def _descendants() -> dict[int, str]:
    parents: dict[int, tuple[int, str]] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(path.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    result: dict[int, str] = {}
    pending = {os.getpid()}
    while pending:
        children = {
            pid: identity
            for pid, (parent, identity) in parents.items()
            if parent in pending and pid not in result
        }
        result.update(children)
        pending = set(children)
    return result


def _signal_owned(pid: int, identity: str, signum: signal.Signals) -> bool:
    try:
        descriptor = os.pidfd_open(pid)
    except OSError:
        return False
    try:
        try:
            current = (
                Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        except (OSError, IndexError):
            return False
        if current == identity:
            signal.pidfd_send_signal(descriptor, signum)
            return True
    except OSError:
        return False
    finally:
        os.close(descriptor)
    return False


def _reap(primary: _WaitableChild) -> bool:
    """Return whether this supervisor still owns any children."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return False
        if pid == 0:
            return True
        if pid == primary.pid:
            primary.returncode = os.waitstatus_to_exitcode(status)


def _fork_owner(
    arguments: Sequence[str],
    *,
    expires: float,
    pass_fds: Sequence[int],
    grace: float,
    report_fd: int,
    guardian_report_fd: int,
) -> _WaitableChild:
    if threading.active_count() != 1:
        raise OSError(errno.EBUSY, "command guardian must be single-threaded")
    owner = os.getpid()
    blocked = {signal.SIGTERM, signal.SIGINT}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        pid = os.fork()
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        raise
    if pid:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        return _ForkedChild(pid)
    try:
        os.close(guardian_report_fd)
        # Fork only the isolated stdlib guardian, never the multithreaded CLI.
        # Pending termination stays blocked until the child's own handler and
        # parent-death signal are ready; no cancellation can be lost at handoff.
        try:
            result = supervise(
                arguments,
                owner=owner,
                expires=expires,
                pass_fds=pass_fds,
                grace=grace,
                restore_signal_mask=old_mask,
            )
        except OSError as exc:
            result = {"status": "supervisor-error", "errno": exc.errno or errno.EIO}
        os.write(report_fd, json.dumps(result, separators=(",", ":")).encode())
    finally:
        os._exit(0)


def supervise(
    arguments: Sequence[str],
    *,
    owner: int,
    expires: float,
    pass_fds: Sequence[int],
    grace: float = TERMINATION_GRACE_SECONDS,
    command_report_fd: int | None = None,
    guardian_report_fd: int | None = None,
    restore_signal_mask: set[int | signal.Signals] | None = None,
) -> dict[str, object]:
    if not math.isfinite(expires) or not 0 < grace <= 10:
        raise ValueError("invalid command deadline")
    stopped = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    _configure_subreaper(owner)
    if restore_signal_mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, restore_signal_mask)
    if stopped or time.monotonic() >= expires:
        return {"status": "timeout", "returncode": 124}
    primary: _WaitableChild
    if command_report_fd is None:
        primary = subprocess.Popen(list(arguments), text=True, pass_fds=tuple(pass_fds))
    else:
        if guardian_report_fd is None:
            raise ValueError("guardian completion descriptor is missing")
        primary = _fork_owner(
            arguments,
            expires=expires,
            pass_fds=pass_fds,
            grace=grace,
            report_fd=command_report_fd,
            guardian_report_fd=guardian_report_fd,
        )
    timed_out = False
    stopping_at: float | None = None
    signalled: set[tuple[int, str, signal.Signals]] = set()
    started = time.monotonic()
    next_report = started + 30
    while True:
        children = _reap(primary)
        now = time.monotonic()
        if stopped or now >= expires:
            timed_out = True
        if stopping_at is None and (timed_out or primary.returncode is not None):
            stopping_at = now
        if not children:
            break
        if now >= next_report:
            executable = Path(arguments[0]).name
            label = (
                executable
                if executable
                in {
                    "aws",
                    "kubectl",
                    "python",
                    "python3",
                    "python3.12",
                    "bash",
                    "sh",
                    "make",
                    "docker",
                }
                else "command"
            )
            if command_report_fd is None or stopping_at is not None:
                write_diagnostic(
                    f"deployment-wait command={label} "
                    f"state={'cleanup' if stopping_at is not None else 'running'} "
                    f"elapsed={int(now - started)}s "
                    f"remaining={max(0, int(expires - now))}s\n",
                )
            next_report = now + 30
        if stopping_at is not None:
            signum = signal.SIGKILL if now - stopping_at >= grace else signal.SIGTERM
            try:
                descendants = _descendants()
            except OSError:
                descendants = {}
            for pid, identity in descendants.items():
                entry = (pid, identity, signum)
                if entry not in signalled:
                    if _signal_owned(pid, identity, signum):
                        signalled.add(entry)
        time.sleep(POLL_SECONDS)
    return {
        "status": "timeout" if timed_out else "exited",
        "returncode": primary.returncode,
    }


def _drain(
    stream: TextIO, emit: Callable[[str], None], errors: list[Exception]
) -> None:
    try:
        while text := stream.readline(4096):
            emit(text)
    except Exception as exc:
        errors.append(exc)
    finally:
        stream.close()


def _supervisor_arguments(
    arguments: Sequence[str],
    *,
    owner: int,
    expires: float,
    report_fd: int,
    pass_fds: Sequence[int],
    grace: float,
    command_report_fd: int | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(Path(__file__).resolve()),
        "--owner",
        str(owner),
        "--expires",
        str(expires),
        "--report-fd",
        str(report_fd),
        "--grace",
        str(grace),
        "--pass-fds",
        ",".join(str(item) for item in pass_fds),
        *(
            ["--command-report-fd", str(command_report_fd)]
            if command_report_fd is not None
            else []
        ),
        "--",
        *arguments,
    ]


def _communicate_owned(
    process: subprocess.Popen[str],
    reports: Sequence[_CompletionReport],
    *,
    allow_interrupted: bool = False,
) -> tuple[str | None, str | None]:
    output = None
    while output is None or not all(report.ended for report in reports):
        ensure_supervision_safe(allow_interrupted=allow_interrupted)
        if output is None:
            try:
                output = process.communicate(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(POLL_SECONDS)
        for report in reports:
            report.read()
        # Only supervisors inherit these writers, never the actual command.
        # Check even while stdout is open: orphan output pipes cannot hide loss
        # of both owners and indefinitely block communicate().
        if all(report.ended for report in reports):
            _require_completion_proof(reports)
        ensure_supervision_safe(allow_interrupted=allow_interrupted)
    return output


def _require_completion_proof(reports: Sequence[_CompletionReport]) -> None:
    if not any(report.proves_quiescence() for report in reports):
        _SUPERVISION_LOST.set()
        raise ProcessSupervisionLost(
            "deployment supervisors did not prove command-tree completion; "
            "automatic compensation and further commands are refused"
        )


def _finish_owned_command(
    process: subprocess.Popen[str],
    reports: Sequence[_CompletionReport],
    descriptor: int | None,
) -> None:
    """Reap regardless of capture errors or which supervisor survived."""
    signalled = False
    while True:
        try:
            ended = process.poll() is not None
            if not ended and not signalled and descriptor is not None:
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                except OSError:
                    pass
                else:
                    signalled = True
            for report in reports:
                report.read()
            if ended and all(report.ended for report in reports):
                _require_completion_proof(reports)
                return
            time.sleep(POLL_SECONDS)
        except (KeyboardInterrupt, InterruptedError):
            # Even external/custom interruption must not abandon a live owner.
            continue
        except ProcessSupervisionLost:
            raise
        except BaseException as exc:
            _SUPERVISION_LOST.set()
            raise ProcessSupervisionLost(
                "command-tree cleanup could not establish completion"
            ) from exc


def _command_result(
    arguments: Sequence[str],
    *,
    reports: Sequence[_CompletionReport],
    timeout: float,
    stdout: str | None,
    stderr: str | None,
) -> subprocess.CompletedProcess[str]:
    guardian, command = (report.value() for report in reports)
    if (
        guardian is not None
        and reports[0].proves_quiescence()
        and guardian.get("status") == "timeout"
    ):
        raise subprocess.TimeoutExpired(list(arguments), timeout, stdout, stderr)
    if command is not None and command.get("status") == "supervisor-error":
        number = command.get("errno")
        if type(number) is int and reports[0].proves_quiescence():
            raise OSError(number, os.strerror(number), arguments[0])
    if (
        guardian is None
        or not reports[0].proves_quiescence()
        or command is None
        or not reports[1].proves_quiescence()
    ):
        raise RuntimeError(
            "deployment supervisor failed; the surviving owner stopped "
            "and reaped the command tree"
        )
    if command.get("status") == "timeout":
        raise subprocess.TimeoutExpired(list(arguments), timeout, stdout, stderr)
    if guardian.get("returncode") != 0:
        raise RuntimeError(
            "deployment supervisor exited abnormally after command-tree cleanup"
        )
    code = command["returncode"]
    assert type(code) is int, "validated completion must carry an integer exit code"
    return subprocess.CompletedProcess(
        list(arguments), code, stdout or "", stderr or ""
    )


def run_owned_command(
    arguments: Sequence[str],
    *,
    timeout: float,
    input_text: str | None = None,
    capture: bool = True,
    environment: Mapping[str, str] | None = None,
    cwd: os.PathLike[str] | None = None,
    pass_fds: Sequence[int] = (),
    diagnostics: DriverDiagnostics | None = None,
    expires_at: float | None = None,
    allow_interrupted: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Only return normally after an independent owner proves tree quiescence."""
    with interruption_scope():
        return _run_owned_command(
            arguments,
            timeout=timeout,
            input_text=input_text,
            capture=capture,
            environment=environment,
            cwd=cwd,
            pass_fds=pass_fds,
            diagnostics=diagnostics,
            expires_at=expires_at,
            allow_interrupted=allow_interrupted,
        )


def _run_owned_command(
    arguments: Sequence[str],
    *,
    timeout: float,
    input_text: str | None,
    capture: bool,
    environment: Mapping[str, str] | None,
    cwd: os.PathLike[str] | None,
    pass_fds: Sequence[int],
    diagnostics: DriverDiagnostics | None,
    expires_at: float | None,
    allow_interrupted: bool,
) -> subprocess.CompletedProcess[str]:
    ensure_supervision_safe(allow_interrupted=allow_interrupted)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid command timeout")
    expires = time.monotonic() + timeout
    if expires_at is not None:
        if not math.isfinite(expires_at):
            raise ValueError("invalid absolute command deadline")
        expires = min(expires, expires_at)
    if time.monotonic() >= expires:
        raise subprocess.TimeoutExpired(list(arguments), timeout)
    for descriptor in pass_fds:
        os.fstat(descriptor)
    descriptors: list[int] = []
    try:
        for _ in range(2):
            descriptors.extend(_report_pipe())
        reports = tuple(_CompletionReport(item) for item in descriptors[::2])
        for report in reports:
            os.set_blocking(report.descriptor, False)
        command = _supervisor_arguments(
            arguments,
            owner=os.getpid(),
            expires=expires,
            report_fd=descriptors[1],
            command_report_fd=descriptors[3],
            pass_fds=pass_fds,
            grace=TERMINATION_GRACE_SECONDS,
        )
        with contextlib.ExitStack() as inputs:
            stdin = None
            if input_text is not None:
                # Retried communicate(None) does not resume a partial stdin write.
                # An anonymous 0600 file keeps private input complete and gives EOF
                # independently of pipe capacity or child startup latency.
                stdin = inputs.enter_context(tempfile.TemporaryFile(mode="w+b"))
                stdin.write(input_text.encode("utf-8"))
                stdin.seek(0)
            if time.monotonic() >= expires:
                raise subprocess.TimeoutExpired(list(arguments), timeout)
            ensure_supervision_safe(allow_interrupted=allow_interrupted)
            process = subprocess.Popen(
                command,
                stdin=stdin,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture or diagnostics is not None else None,
                text=True,
                env=environment,
                cwd=cwd,
                start_new_session=True,
                pass_fds=(*pass_fds, *descriptors[1::2]),
            )
    except BaseException:
        for descriptor in descriptors[::2]:
            os.close(descriptor)
        raise
    finally:
        for descriptor in descriptors[1::2]:
            os.close(descriptor)
    reader: threading.Thread | None = None
    reader_errors: list[Exception] = []
    guardian_fd: int | None = None
    try:
        guardian_fd = _register_guardian(process, cleanup=allow_interrupted)
        ensure_supervision_safe(allow_interrupted=allow_interrupted)
        if diagnostics is not None:
            assert process.stderr is not None
            reader = threading.Thread(
                target=_drain,
                args=(process.stderr, diagnostics.feed, reader_errors),
                daemon=True,
            )
            reader.start()
            # communicate owns stdout, while the one reader streams stderr.
            process.stderr = None
        stdout, stderr = _communicate_owned(
            process, reports, allow_interrupted=allow_interrupted
        )
        return _command_result(
            arguments,
            reports=reports,
            timeout=timeout,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        error = sys.exc_info()[1]
        if reader is not None and reader.ident is not None:
            process.stderr = None
        try:
            _finish_owned_command(process, reports, guardian_fd)
        finally:
            try:
                for report in reports:
                    os.close(report.descriptor)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            finally:
                _unregister_guardian(process)
        if reader is not None and reader.ident is not None:
            reader.join(timeout=DIAGNOSTIC_JOIN_SECONDS)
            if reader.is_alive() or reader_errors:
                note = "deployment diagnostic reader did not finish cleanly"
                if error is not None:
                    error.add_note(note)
                else:
                    raise RuntimeError(note) from None


def main() -> None:
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", type=int, required=True)
    parser.add_argument("--expires", type=float, required=True)
    parser.add_argument("--report-fd", type=int, required=True)
    parser.add_argument("--command-report-fd", type=int)
    parser.add_argument("--grace", type=float, default=TERMINATION_GRACE_SECONDS)
    parser.add_argument("--pass-fds", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    pass_fds = tuple(int(item) for item in args.pass_fds.split(",") if item)
    try:
        result = supervise(
            command,
            owner=args.owner,
            expires=args.expires,
            pass_fds=pass_fds,
            grace=args.grace,
            command_report_fd=args.command_report_fd,
            guardian_report_fd=args.report_fd,
        )
    except OSError as exc:
        # An unexpected OS error could follow a successful spawn. This report
        # is never a completion proof; the other owner must establish quiescence.
        result = {"status": "supervisor-error", "errno": exc.errno or errno.EIO}
    os.write(args.report_fd, json.dumps(result, separators=(",", ":")).encode())
    os.close(args.report_fd)


if __name__ == "__main__":
    main()
