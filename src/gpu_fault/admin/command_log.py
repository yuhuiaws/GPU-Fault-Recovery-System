from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

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


def _log_path(state_dir: Path, command: str) -> Path:
    directory = state_dir.expanduser().resolve() / ADMIN_LOG_DIRECTORY
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
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
def command_log(state_dir: object, *, command: str) -> Iterator[Path | None]:
    """Record this invocation's console output under the private state directory.

    An administrator command that fails mid-deployment prints the only copy of
    the reason it failed, and a preflight report names the failing check in the
    middle of thousands of lines. Without a file the operator has to have
    thought to redirect the run beforehand, which is exactly when they did not.

    A nested invocation -- the deploy path re-enters this CLI from the deploy
    host venv -- appends to the log the outer command already opened, so one
    operator action produces one file.
    """

    inherited = os.environ.get(ADMIN_LOG_ENVIRONMENT)
    if inherited:
        yield Path(inherited)
        return
    if not isinstance(state_dir, Path):
        yield None
        return
    path = _log_path(state_dir, command)
    with path.open("ab") as handle:
        path.chmod(0o600)
        prune_admin_logs(path.parent)
        os.environ[ADMIN_LOG_ENVIRONMENT] = str(path)
        try:
            with _teed(handle):
                announce(f"gpu-fault-admin: logging to {path}")
                yield path
        finally:
            os.environ.pop(ADMIN_LOG_ENVIRONMENT, None)
