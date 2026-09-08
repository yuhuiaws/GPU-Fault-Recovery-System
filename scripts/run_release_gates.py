from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
# How much of the failing gate's output is repeated after the noise. Twenty
# lines hold a pytest summary or the static runner's own aggregate line with
# the failing check above it; the whole thing is still in the log.
GATE_TAIL_LINES = 20


class ReleaseGateError(RuntimeError):
    pass


def gate_parallelism() -> int:
    configured = os.getenv("GPU_FAULT_RELEASE_GATE_PARALLELISM", "").strip()
    if configured:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ReleaseGateError(
                "GPU_FAULT_RELEASE_GATE_PARALLELISM must be an integer"
            ) from exc
        if not 1 <= value <= 3:
            raise ReleaseGateError(
                "GPU_FAULT_RELEASE_GATE_PARALLELISM must be within 1..3"
            )
        return value
    return min(3, max(1, (os.cpu_count() or 1) // 4))


def _gate_environment(name: str, cache_root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPYCACHEPREFIX": str(cache_root / name / "pycache"),
        "PYTEST_ADDOPTS": (
            (
                os.environ.get("PYTEST_ADDOPTS", "").strip() + " "
                if os.environ.get("PYTEST_ADDOPTS", "").strip()
                else ""
            )
            + f"-o cache_dir={cache_root / name / 'pytest-cache'}"
        ),
    }


@dataclass(frozen=True)
class GateOutcome:
    """How one gate ended: its status, its last lines, and whether we stopped it."""

    name: str
    status: int
    tail: tuple[str, ...]
    cancelled: bool = False

    @property
    def failed(self) -> bool:
        return bool(self.status) and not self.cancelled


class GateGroup:
    """The gates of one parallel step, so the first failure can stop the rest.

    Before this, a failed static gate sat and waited while pytest and the
    PostgreSQL stress kept running for another two and a half minutes, and their
    progress dots buried the line that said what failed. Each gate now runs in
    its own process group so that stopping it stops ``make``'s children too.
    """

    def __init__(self) -> None:
        self.output_lock = threading.Lock()
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancelled: set[str] = set()
        self.first_failure: str | None = None
        self.interrupted: int | None = None

    def register(self, name: str, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes[name] = process
            if self.first_failure is not None:
                self._stop(name, process)

    def finished(self, name: str) -> bool:
        """Forget ``name``'s process; return whether we were the ones stopping it."""

        with self._lock:
            self._processes.pop(name, None)
            return name in self._cancelled

    @property
    def stopping(self) -> bool:
        return self.first_failure is not None

    def fail(self, name: str) -> None:
        """Record the first failure and stop every other gate still running."""

        with self._lock:
            if self.first_failure is not None:
                return
            self.first_failure = name
            for other, process in list(self._processes.items()):
                if other != name:
                    self._stop(other, process)

    def interrupt(self, signum: int) -> None:
        """Stop every gate because the operator did (Ctrl-C or a TERM)."""

        with self._lock:
            self.interrupted = signum
            for name, process in list(self._processes.items()):
                self._stop(name, process)

    def _stop(self, name: str, process: subprocess.Popen[str]) -> None:
        self._cancelled.add(name)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()


def _stream_gate(
    name: str,
    command: Sequence[str],
    *,
    environment: dict[str, str],
    output_lock: threading.Lock,
    group: GateGroup | None = None,
) -> GateOutcome:
    group = group or GateGroup()
    if group.stopping:
        with output_lock:
            print(
                f"release-gates: {name} not started after {group.first_failure} failed",
                file=sys.stderr,
                flush=True,
            )
        return GateOutcome(name, 0, (), cancelled=True)
    with output_lock:
        print(f"release-gates: starting {name}", file=sys.stderr, flush=True)
    process = subprocess.Popen(
        list(command),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        start_new_session=True,
    )
    group.register(name, process)
    tail: deque[str] = deque(maxlen=GATE_TAIL_LINES)
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line.rstrip("\n"))
        with output_lock:
            print(f"[{name}] {line}", end="", flush=True)
    returncode = process.wait()
    cancelled = group.finished(name)
    with output_lock:
        print(
            f"release-gates: {name} cancelled after {group.first_failure} failed"
            if cancelled
            else f"release-gates: {name} exited with status {returncode}",
            file=sys.stderr,
            flush=True,
        )
    return GateOutcome(name, returncode, tuple(tail), cancelled=cancelled)


def _report_first_failure(outcomes: Sequence[GateOutcome]) -> None:
    """Repeat the failing gate's last lines after every other gate's noise."""

    failed = [outcome for outcome in outcomes if outcome.failed]
    if not failed:
        return
    first = failed[0]
    stopped = [outcome.name for outcome in outcomes if outcome.cancelled]
    header = f"release-gates: first failing gate: {first.name} (status {first.status})"
    if stopped:
        header += "; stopped: " + ", ".join(sorted(stopped))
    lines = [header]
    if first.tail:
        lines.append(f"release-gates: last {len(first.tail)} lines of {first.name}:")
        lines.extend(f"[{first.name}] {line}" for line in first.tail)
    print("\n".join(lines), file=sys.stderr, flush=True)


def _run_parallel(
    commands: dict[str, list[str]],
    *,
    cache_root: Path,
    max_workers: int,
) -> dict[str, int]:
    """Run ``commands`` together; stop the rest at the first failure.

    Returns the gates that genuinely failed (name to status). Gates this runner
    stopped are reported on the console but are not failures of their own.
    """

    group = GateGroup()
    outcomes: list[GateOutcome] = []
    previous: dict[int, object] = {}
    # Ctrl-C reaches only the foreground process group, and the gates now run in
    # their own, so the interrupt has to be forwarded by hand or pytest keeps
    # running under a deploy the operator already gave up on.
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda received, _frame: group.interrupt(received))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _stream_gate,
                    name,
                    command,
                    environment=_gate_environment(name, cache_root),
                    output_lock=group.output_lock,
                    group=group,
                ): name
                for name, command in commands.items()
            }
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                if outcome.failed:
                    group.fail(outcome.name)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]
    if group.interrupted is not None:
        raise ReleaseGateError(f"interrupted by signal {group.interrupted}")
    _report_first_failure(outcomes)
    return {outcome.name: outcome.status for outcome in outcomes if outcome.failed}


def _raise_failures(failures: dict[str, int], *, description: str) -> None:
    if failures:
        raise ReleaseGateError(
            f"{description}: "
            + ", ".join(f"{name}={status}" for name, status in sorted(failures.items()))
        )


def run_release_gates(python: str) -> None:
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL", "").strip():
        raise ReleaseGateError("GPU_FAULT_TEST_POSTGRES_URL is required")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-release-gates-") as directory:
        cache_root = Path(directory)
        # The static gate is the one that fails on a fresh tree and it takes a
        # minute; the other two take nearly three. Running it alone first means
        # a lint or architecture failure costs one minute, not three, and its
        # verdict is the last thing on the screen rather than the first.
        failures = _run_parallel(
            {"static": ["make", "check-static", f"PYTHON={python}"]},
            cache_root=cache_root,
            max_workers=1,
        )
        _raise_failures(failures, description="static release gate failed")
        failures = _run_parallel(
            {
                "postgres": ["make", "test-postgres-stress", f"PYTHON={python}"],
                "pytest": ["make", "test-parallel-release", f"PYTHON={python}"],
            },
            cache_root=cache_root,
            max_workers=gate_parallelism(),
        )
        _raise_failures(failures, description="parallel release gate failed")
        subprocess.run(
            ["make", "artifact-check", f"PYTHON={python}"],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPYCACHEPREFIX": str(cache_root / "artifact"),
            },
            check=True,
        )
        subprocess.run(
            ["make", "python-cache-clean", f"PYTHON={python}"],
            cwd=ROOT,
            check=True,
        )


def run_check_gates(python: str) -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-fault-check-gates-") as directory:
        cache_root = Path(directory)
        subprocess.run(
            ["make", "check-static", f"PYTHON={python}"],
            cwd=ROOT,
            env=_gate_environment("static", cache_root),
            check=True,
        )
        failures = _run_parallel(
            {
                "artifact": ["make", "artifact-check", f"PYTHON={python}"],
                "pytest": ["make", "test-parallel-release", f"PYTHON={python}"],
            },
            cache_root=cache_root,
            max_workers=2,
        )
        _raise_failures(failures, description="parallel check tail failed")
        subprocess.run(
            ["make", "python-cache-clean", f"PYTHON={python}"],
            cwd=ROOT,
            check=True,
        )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mode", choices=("check", "release"), default="release")
    options = parser.parse_args(arguments)
    try:
        if options.mode == "check":
            run_check_gates(options.python)
        else:
            run_release_gates(options.python)
    except (OSError, ReleaseGateError, subprocess.SubprocessError) as exc:
        print(f"release-gates: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
