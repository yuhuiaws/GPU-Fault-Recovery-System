from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator, NamedTuple, Sequence, TypeVar

ADMIN_LOG_DIRECTORY = Path("logs")
ADMIN_LOG_ENVIRONMENT = "GPU_FAULT_ADMIN_LOG"
ADMIN_LOG_RETAINED = 100
# A count on its own bounds nothing. These logs tee whole `kubectl` and `aws`
# outputs, so one fleet rollout is tens of megabytes and a preflight report over a
# large fleet more, while a `status` run is kilobytes -- a hundred of the former is
# gigabytes in the directory that also holds the kubeconfigs and cluster tokens the
# next command needs, and a full state directory is exactly the failure this
# logging exists to explain.
ADMIN_LOG_MAX_BYTES = 512 * 1024 * 1024


class LogLimits(NamedTuple):
    retained: int
    max_bytes: int


# One ceiling for every command meant a hundred `status` runs evicted the one
# deploy log worth reading back (I3). Each kind now has its own directory under
# `logs/` and its own ceiling, so a read-only flood can only ever evict other
# read-only logs. Mutating logs are the ones that explain a failed rollout, so
# they get the larger count; a caller that names no kind is treated as mutating,
# because the safe mistake is keeping a `status` log too long, not losing a
# deploy log.
ADMIN_LOG_KIND_MUTATING = "mutating"
ADMIN_LOG_KIND_READONLY = "readonly"
ADMIN_LOG_LIMITS: dict[str, LogLimits] = {
    ADMIN_LOG_KIND_MUTATING: LogLimits(retained=200, max_bytes=ADMIN_LOG_MAX_BYTES),
    ADMIN_LOG_KIND_READONLY: LogLimits(
        retained=ADMIN_LOG_RETAINED, max_bytes=64 * 1024 * 1024
    ),
}


def admin_log_limits(kind: str) -> LogLimits:
    try:
        return ADMIN_LOG_LIMITS[kind]
    except KeyError:
        raise ValueError(
            f"unknown admin log kind {kind!r}; expected one of "
            f"{sorted(ADMIN_LOG_LIMITS)}"
        ) from None


def admin_log_directory(state_dir: Path, kind: str) -> Path:
    """Where logs of ``kind`` live under the state directory (not created)."""

    admin_log_limits(kind)
    return state_dir.expanduser().resolve() / ADMIN_LOG_DIRECTORY / kind


def _log_path(state_dir: Path, command: str, kind: str) -> Path:
    directory = admin_log_directory(state_dir, kind)
    # Both levels are created explicitly: ``parents=True`` applies ``mode`` to
    # the leaf only, and the root sits beside kubeconfigs and cluster tokens.
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.mkdir(mode=0o700, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{command}-{stamp}-{os.getpid()}.log"


def prune_admin_logs(
    directory: Path,
    *,
    retained: int = ADMIN_LOG_RETAINED,
    max_bytes: int = ADMIN_LOG_MAX_BYTES,
) -> None:
    """Keep the newest logs, bounded by count and by total size.

    The state directory also holds kubeconfigs and cluster tokens, so letting it
    grow without bound risks the failure this logging exists to diagnose.

    The newest file is never pruned for size: it is the log the running command is
    writing to, and a single run larger than the whole ceiling is still the run
    being diagnosed.
    """

    logs = sorted(
        (item for item in directory.glob("*.log") if item.is_file()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for stale in logs[retained:]:
        stale.unlink(missing_ok=True)
    used = 0
    for index, item in enumerate(logs[:retained]):
        try:
            used += item.stat().st_size
        except OSError:
            continue
        if index and used > max_bytes:
            item.unlink(missing_ok=True)


def announce(message: str) -> None:
    """Write to descriptor 2 directly so the tee cannot be bypassed.

    ``print`` goes through whatever ``sys.stderr`` currently is, and a wrapper
    that buffers in Python never reaches the descriptor the log is teed from --
    which would leave the log without the one line naming itself.
    """

    os.write(2, (message + "\n").encode("utf-8"))


# A deploy is a chain of wrappers -- outer CLI, staging driver, nested CLI,
# release driver, rollout -- and each used to add its own
# ``command failed (N): <argv0>`` on the way out. The operator read four lines
# naming a Python interpreter path while the cause sat hundreds of lines above.
# The rule now: a child that is one of our own drivers has already said why it
# failed, so its wrapper adds nothing and hands the exit status through; a
# foreign command (``aws``, ``kubectl``, a shell script) gets exactly one line
# naming what ran and its last words. The two attributes below carry that
# decision to the ``main()`` that finally prints, without changing the error
# types the layers in between catch.
OWN_DRIVER_NAMES = frozenset({"gpu-fault-admin", "make"})
FAILURE_EXIT_CODE_ATTRIBUTE = "child_exit_code"
FAILURE_REPORTED_ATTRIBUTE = "child_reported_failure"
COMMAND_LABEL_TOKENS = 6

_ErrorT = TypeVar("_ErrorT", bound=BaseException)


def is_own_driver(arguments: Sequence[str]) -> bool:
    """Whether ``arguments`` runs this repository's own CLI or a driver of it.

    Any Python interpreter counts: the only Python these wrappers run is our own
    modules and scripts, and every one of them reports its failure before it
    exits. ``make`` counts because it runs this Makefile and names the failed
    target itself.
    """

    if not arguments:
        return False
    name = Path(arguments[0]).name
    return name in OWN_DRIVER_NAMES or name.startswith("python")


def driver_name(arguments: Sequence[str]) -> str:
    """The unit an operator recognises in ``arguments``: module, script or target.

    ``/…/bundle-ceaac276/bin/python`` says nothing; ``gpu_fault_release.rollout``
    or ``release_deploy.py`` says which layer stopped.
    """

    name = Path(arguments[0]).name
    rest = list(arguments[1:])
    if name.startswith("python"):
        if "-m" in rest and rest.index("-m") + 1 < len(rest):
            return rest[rest.index("-m") + 1]
        for token in rest:
            if token.endswith(".py"):
                return Path(token).name
    elif name == "make":
        targets = [t for t in rest if "=" not in t and not t.startswith("-")]
        if targets:
            return f"make {targets[0]}"
    return name


def command_label(arguments: Sequence[str], *, sensitive: bool = False) -> str:
    """Enough of ``arguments`` to tell which step it was, never the whole argv."""

    if sensitive or not arguments:
        return Path(arguments[0]).name if arguments else "<no command>"
    shown = " ".join(arguments[:COMMAND_LABEL_TOKENS])
    return shown + (" ..." if len(arguments) > COMMAND_LABEL_TOKENS else "")


def last_output_line(text: str | None) -> str:
    """The last non-empty line of a captured stream, or the empty string."""

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def child_failure(
    error_type: type[_ErrorT],
    arguments: Sequence[str],
    returncode: int,
    *,
    detail: str = "",
    sensitive: bool = False,
) -> _ErrorT:
    """The exception a wrapper raises when a child process exits non-zero.

    For one of our own drivers the message is a bare status line that
    :func:`report_failure` will not print, and the child's exit status rides
    along so the outermost process exits with the first failing cause's code.
    For a foreign command the message is the one line the operator needs:
    ``command failed (N): <what ran>: <its last stderr line>``.
    """

    if is_own_driver(arguments):
        error = error_type(f"{driver_name(arguments)} exited with status {returncode}")
        setattr(error, FAILURE_REPORTED_ATTRIBUTE, True)
    else:
        message = f"command failed ({returncode}): "
        message += command_label(arguments, sensitive=sensitive)
        if detail:
            message += f": {detail}"
        error = error_type(message)
    setattr(error, FAILURE_EXIT_CODE_ATTRIBUTE, int(returncode))
    return error


def report_failure(
    program: str, error: BaseException, *, default_exit_code: int = 2
) -> int:
    """Print at most one line for ``error`` and return the exit status to use.

    A failure the child already reported passes through silently with the
    child's exit status. Anything else is printed as ``<program>: <error>``, with
    any notes the layers in between attached (``run_parallel`` names the task)
    on their own lines first, and exits with ``default_exit_code``.
    """

    if getattr(error, FAILURE_REPORTED_ATTRIBUTE, False):
        status = int(getattr(error, FAILURE_EXIT_CODE_ATTRIBUTE, default_exit_code))
        return status or default_exit_code
    for note in getattr(error, "__notes__", ()):
        print(f"{program}: {note}", file=sys.stderr, flush=True)
    print(f"{program}: {error}", file=sys.stderr, flush=True)
    return default_exit_code


def _pump(read_fd: int, passthrough_fd: int, handle: IO[bytes]) -> None:
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            return
        os.write(passthrough_fd, chunk)
        handle.write(chunk)
        handle.flush()


@contextmanager
def _teed(handle: IO[bytes]) -> Iterator[None]:
    """Duplicate everything written to fd 1 and 2 into ``handle``.

    The tee has to live at the file-descriptor level rather than on
    ``sys.stdout``, because almost everything an administrator needs to read
    back is produced by ``kubectl`` and ``aws`` subprocesses that inherit these
    descriptors. A Python-level wrapper would record the surrounding narration
    and lose the output that explains the failure.
    """

    for stream in (sys.stdout, sys.stderr):
        # Anything already buffered belongs to the console the caller was
        # watching, not to this log.
        stream.flush()
    saved = {stream: os.dup(stream) for stream in (1, 2)}
    pipes = {stream: os.pipe() for stream in (1, 2)}
    threads = [
        threading.Thread(
            target=_pump,
            args=(read_fd, saved[stream], handle),
            daemon=True,
        )
        for stream, (read_fd, _write_fd) in pipes.items()
    ]
    for thread in threads:
        thread.start()
    for stream, (_read_fd, write_fd) in pipes.items():
        os.dup2(write_fd, stream)
        os.close(write_fd)
    try:
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        # Restoring the saved descriptor closes the only remaining write end of
        # each pipe, which is what lets the pump threads see EOF and finish.
        for stream, original in saved.items():
            os.dup2(original, stream)
        # The saved descriptors stay open until the pumps have drained what is
        # still in the pipes: they are where the passthrough copy is written, so
        # closing them first loses the tail of the output from both the console
        # and the log -- which is exactly the part that explains a failure.
        for thread in threads:
            thread.join(timeout=5)
        for original in saved.values():
            os.close(original)
        for read_fd, _write_fd in pipes.values():
            os.close(read_fd)


@contextmanager
def command_log(
    state_dir: object,
    *,
    command: str,
    kind: str = ADMIN_LOG_KIND_MUTATING,
) -> Iterator[Path | None]:
    """Record this invocation's console output under the private state directory.

    An administrator command that fails mid-deployment prints the only copy of
    the reason it failed, and a preflight report names the failing check in the
    middle of thousands of lines. Without a file the operator has to have
    thought to redirect the run beforehand, which is exactly when they did not.

    A nested invocation -- the deploy path re-enters this CLI from the deploy
    host venv -- opens no log of its own: it inherits the descriptors the outer
    command already tees, so one operator action produces one file. It yields
    ``None`` like a command with no state directory, because only the process
    that opened the log owns it -- naming it on the way out is the outer
    command's job, and a nested one that did the same left the operator three
    identical ``full output in`` lines for one failure.

    ``kind`` selects the directory and the ceiling (``ADMIN_LOG_LIMITS``):
    ``readonly`` for commands that change nothing (``status``, diff, verify),
    ``mutating`` -- the default -- for everything else. An unknown kind is
    refused before anything is written.
    """

    limits = admin_log_limits(kind)
    if os.environ.get(ADMIN_LOG_ENVIRONMENT):
        yield None
        return
    if not isinstance(state_dir, Path):
        yield None
        return
    path = _log_path(state_dir, command, kind)
    with path.open("ab") as handle:
        path.chmod(0o600)
        prune_admin_logs(
            path.parent, retained=limits.retained, max_bytes=limits.max_bytes
        )
        # Logs written before the per-kind split sit flat in ``logs/``; keep
        # bounding them under the old ceiling so they age out rather than stay.
        prune_admin_logs(path.parent.parent)
        os.environ[ADMIN_LOG_ENVIRONMENT] = str(path)
        try:
            with _teed(handle):
                announce(f"gpu-fault-admin: logging to {path}")
                yield path
        finally:
            os.environ.pop(ADMIN_LOG_ENVIRONMENT, None)
