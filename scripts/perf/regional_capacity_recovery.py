"""Bounded teardown retries and the recovery of a teardown that gave up.

Every capacity suite cleans up through ``regional_capacity_suite.teardown``
(stop the run's resources, purge its data, deregister its synthetic clusters).
When the bounded retries are spent the run keeps its receipts, ``status.json``
records the cleanup error, and the log names the one command that finishes
the teardown from those receipts: ``regional_capacity_suite.py teardown`` with
the run's ``--suite-id`` and ``--run-dir``, which adopts them only while the
run's ``run.json`` still matches ``resume_identity``.

A termination signal ends a run through the same teardown (perf plan §8.4):
``abort_on_signals`` makes the first SIGINT, SIGTERM or SIGHUP before the
teardown raise ``RunTerminated`` where the run is, the runner records the run
aborted after its ``finally`` has torn it down, and a signal that arrives
while ``cleanup_signal_guard`` holds the teardown is recorded instead of
interrupting it -- it too ends the run aborted.
"""

from __future__ import annotations

import functools
import shlex
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ParamSpec

P = ParamSpec("P")

if __package__:
    from . import regional_capacity_registry as capacity_registry
    from .regional_capacity_results import write_status
else:
    import regional_capacity_registry as capacity_registry
    from regional_capacity_results import write_status

SUITE_SCRIPT = Path(__file__).resolve().parent / "regional_capacity_suite.py"
# Waits before teardown attempts 2, 3 and 4. A purge that meets a writer still
# touching the run's rows right after the run (live 2026-10-13: "capacity
# cleanup left run-owned data" once the action executors were gone) is one
# transaction and deletes nothing, so a later attempt starts from the same
# state; together the waits outlast the action suite's 120-s synthetic Agent
# lease, after which the control plane stops writing for the run's nodes.
TEARDOWN_RETRY_DELAYS_SECONDS = (15, 45, 90)


def retry_teardown(once: Callable[[], None], *, attempts: int) -> None:
    """Run one teardown attempt up to ``attempts`` times, waiting
    ``TEARDOWN_RETRY_DELAYS_SECONDS`` between them; raise the last error."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            once()
            return
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                delay = TEARDOWN_RETRY_DELAYS_SECONDS[
                    min(attempt, len(TEARDOWN_RETRY_DELAYS_SECONDS)) - 1
                ]
                detail = " ".join(str(exc).split())[-300:]
                capacity_registry.log(
                    f"teardown attempt {attempt}/{attempts} failed "
                    f"({type(exc).__name__}: {detail}); retrying in {delay} s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"capacity teardown failed after {attempts} attempts: {last_error}"
    ) from last_error


def resume_identity(
    *, suite_id: str, scope: str, identity: dict[str, str]
) -> dict[str, str]:
    """The ``run.json`` fields a later ``run``, ``purge`` or ``teardown`` must
    find unchanged before it adopts a run's receipts; every suite records them,
    so the recovery command can finish any suite's teardown."""

    return {
        "suite_id": suite_id,
        "registry_scope": scope,
        "control_namespace": capacity_registry.CONTROL_NAMESPACE,
        "dataplane_namespace": capacity_registry.NAMESPACE,
        "registry_secret": capacity_registry.REGISTRY_SECRET,
        "connection_secret": capacity_registry.CONNECTION_SECRET,
        **identity,
    }


def teardown_recovery_command(*, run_id: str, artifacts: Path, scope: str) -> str:
    """One shell line that finishes a run's teardown from its kept receipts,
    against the run's control plane, data-plane context, namespaces and
    registry Secrets, with this checkout's interpreter and harness."""

    environment = {
        "GPU_FAULT_CONTROL_KUBECONFIG": capacity_registry.CONTROL_KUBECONFIG,
        "GPU_FAULT_DATAPLANE_CONTEXT": capacity_registry.DATAPLANE_CONTEXT,
        "GPU_FAULT_PERF_CONTROL_NAMESPACE": capacity_registry.CONTROL_NAMESPACE,
        "GPU_FAULT_PERF_DATAPLANE_NAMESPACE": capacity_registry.NAMESPACE,
        "GPU_FAULT_PERF_REGISTRY_SECRET": capacity_registry.REGISTRY_SECRET,
        "GPU_FAULT_PERF_CONNECTION_SECRET": capacity_registry.CONNECTION_SECRET,
    }
    command = [
        sys.executable,
        str(SUITE_SCRIPT),
        "teardown",
        "--suite-id",
        run_id,
        "--run-dir",
        str(artifacts.absolute()),
    ]
    if scope == "live":
        command += [
            "--allow-live-registry",
            "--confirm-live-registry",
            capacity_registry.LIVE_REGISTRY_CONFIRMATION,
        ]
    return " ".join(
        [
            *(f"{name}={shlex.quote(value)}" for name, value in environment.items()),
            shlex.join(command),
        ]
    )


def write_aborted_status(
    artifacts: Path,
    failure: BaseException,
    *,
    cleanup_error: BaseException | None,
    run_id: str,
    scope: str,
) -> None:
    """Mark a failed run aborted. After a cleanup that gave up, ``status.json``
    also keeps the cleanup error and the command that finishes the teardown
    from the run's kept receipts, and the log names the same command."""

    fields: dict[str, str] = {}
    if cleanup_error is not None:
        command = teardown_recovery_command(
            run_id=run_id, artifacts=artifacts, scope=scope
        )
        capacity_registry.log(
            f"cleanup incomplete; the run's receipts stay in {artifacts}"
        )
        capacity_registry.log(f"recovery command: {command}")
        fields = {
            "cleanup_error": f"{type(cleanup_error).__name__}: {cleanup_error}",
            "recovery_command": command,
        }
    write_status(
        artifacts,
        status="aborted",
        reason=f"{type(failure).__name__}: {failure}",
        **fields,
    )


#: An operator's ^C, ``kill`` and a closed terminal.
TERMINATION_SIGNALS = tuple(
    value
    for name in ("SIGINT", "SIGTERM", "SIGHUP")
    if (value := getattr(signal, name, None)) is not None
)


class RunTerminated(BaseException):
    """A termination signal ended the run.

    Not an ``Exception`` on purpose: the harness retries kubectl reads and its
    teardown attempts on ``Exception``, so a signal raised as one would be
    swallowed by the very loop it interrupts. The runners take it in their
    ``except BaseException``, so the run still goes through its ``finally`` --
    the teardown -- and ``status.json`` records it aborted.
    """

    def __init__(self, signum: int, *, during_teardown: bool = False) -> None:
        where = " during the teardown" if during_teardown else ""
        super().__init__(f"received {signal.Signals(signum).name}{where}")
        self.signum = signum


class TerminationSignals:
    """The termination signals one run received. The first one before the
    teardown raises ``RunTerminated``; any later one, and every one once the
    teardown has begun, is only recorded, so nothing interrupts the cleanup
    the first signal started."""

    def __init__(self, *, raising: bool) -> None:
        self.raising = raising
        self.teardown = not raising
        self.received: list[int] = []
        self.during_teardown: list[int] = []

    def __call__(self, signum: int, _frame: object) -> None:
        raising, self.raising = self.raising, False
        self.received.append(signum)
        if self.teardown:
            self.during_teardown.append(signum)
        if raising:
            raise RunTerminated(signum)

    def abort(self, failure: BaseException | None) -> BaseException | None:
        """The run's failure once its teardown is over: ``failure``, else a
        signal that arrived during the teardown -- which did not interrupt it
        but still ends the run aborted -- else none."""

        if failure is not None or not self.during_teardown:
            return failure
        return RunTerminated(self.during_teardown[0], during_teardown=True)


# The run ``abort_on_signals`` guards right now; its teardown records signals.
_ACTIVE: list[TerminationSignals] = []


@contextmanager
def _handling(handler: TerminationSignals) -> Iterator[TerminationSignals]:
    """``handler`` for every termination signal that is not ignored on entry
    (``nohup`` or a non-interactive ``&`` start a process with SIGHUP or
    SIGINT ignored, and that stays so), then the previous handlers back."""

    previous: dict[int, Any] = {}
    try:
        for value in TERMINATION_SIGNALS:
            current = signal.getsignal(value)
            if current is not signal.SIG_IGN:
                previous[value] = current
                signal.signal(value, handler)
    except (OSError, ValueError):  # not the main thread: nothing to install
        for value, original in previous.items():
            signal.signal(value, original)
        previous.clear()
    try:
        yield handler
    finally:
        for value, original in previous.items():
            signal.signal(value, original)


def abort_on_signals(entry: Callable[P, int]) -> Callable[P, int]:
    """The runner function that owns a run's ``try``/``finally``: a termination
    signal ends it through its teardown instead of killing the process with
    the load generators, executors and synthetic registry left on the site.

    The runner records the run aborted and re-raises ``RunTerminated``; here
    that becomes the shell's exit status 128 + the signal number.
    """

    @functools.wraps(entry)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> int:
        with _handling(TerminationSignals(raising=True)) as signals:
            _ACTIVE.append(signals)
            try:
                return entry(*args, **kwargs)
            except RunTerminated as terminated:
                capacity_registry.log(f"run aborted: {terminated}")
                return 128 + terminated.signum
            finally:
                _ACTIVE.remove(signals)

    return guarded


@contextmanager
def cleanup_signal_guard() -> Iterator[TerminationSignals]:
    """Hold a run's bounded teardown against SIGINT, SIGTERM and SIGHUP: a
    signal from here on is recorded, never acted on, and ``abort(failure)``
    after the teardown turns it into the run's aborted failure. Inside
    ``abort_on_signals`` the run's own handler switches to recording for the
    rest of the run; outside it a recording handler holds the block and the
    previous handlers come back after it."""

    if _ACTIVE:
        signals = _ACTIVE[-1]
        signals.raising, signals.teardown = False, True
        yield signals
        return
    with _handling(TerminationSignals(raising=False)) as signals:
        yield signals
