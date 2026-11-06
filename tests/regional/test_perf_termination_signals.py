"""A termination signal ends a capacity run through its teardown.

The perf runners leave load generators, simulated executors, a synthetic
registry and a healthy fleet on the site while they run. Only SIGINT used to
reach the teardown (as a KeyboardInterrupt); SIGTERM and SIGHUP had no handler
and ended the process on the spot, and a second ^C could interrupt the very
cleanup the first one started. ``regional_capacity_recovery`` now holds the
three signals for a run: the first one before the teardown raises
``RunTerminated`` where the run is, every later one -- and every one once the
teardown has begun -- is only recorded, and a signal recorded during the
teardown still ends the run aborted.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts.perf import regional_capacity_recovery as recovery
from tests.regional._perf_caller_support import deliver_signal

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def default_dispositions() -> Iterator[None]:
    """Every test starts from, and must leave, the process's own handlers."""

    before = {value: signal.getsignal(value) for value in recovery.TERMINATION_SIGNALS}
    yield
    after = {value: signal.getsignal(value) for value in recovery.TERMINATION_SIGNALS}
    for value, handler in before.items():
        signal.signal(value, handler)
    assert after == before, "the guard must restore the handlers it replaced"


def guarded_run(
    events: list[str], *, during_run: int | None, during_teardown: int | None
):
    """A runner shaped like the capacity suites: try, except BaseException,
    the teardown in ``finally`` under ``cleanup_signal_guard``, then the status."""

    @recovery.abort_on_signals
    def run() -> int:
        failure: BaseException | None = None
        try:
            events.append("load")
            if during_run is not None:
                deliver_signal(during_run)
                events.append("load continued after the signal")
        except BaseException as exc:
            failure = exc
        finally:
            with recovery.cleanup_signal_guard() as late:
                events.append("teardown")
                if during_teardown is not None:
                    deliver_signal(during_teardown)
                events.append("teardown done")
            failure = late.abort(failure)
        if failure is not None:
            events.append(f"aborted: {failure}")
            raise failure
        events.append("ok")
        return 0

    return run


@pytest.mark.parametrize("signum", recovery.TERMINATION_SIGNALS)
def test_the_first_signal_before_the_teardown_aborts_through_it(signum: int) -> None:
    events: list[str] = []

    code = guarded_run(events, during_run=signum, during_teardown=None)()

    name = signal.Signals(signum).name
    assert events == [
        "load",
        "teardown",
        "teardown done",
        f"aborted: received {name}",
    ], "the signal must stop the load where it is and still run the whole teardown"
    assert code == 128 + signum, "the shell sees the signal's exit status"


def test_a_signal_during_the_teardown_finishes_it_and_aborts_the_run() -> None:
    events: list[str] = []

    code = guarded_run(events, during_run=None, during_teardown=signal.SIGHUP)()

    assert events == [
        "load",
        "teardown",
        "teardown done",
        "aborted: received SIGHUP during the teardown",
    ], "a signal must not interrupt the teardown, but it ends the run aborted"
    assert code == 128 + signal.SIGHUP, "the run that was signalled is not ok"


def test_a_second_signal_cannot_interrupt_the_cleanup_the_first_started() -> None:
    events: list[str] = []

    @recovery.abort_on_signals
    def run() -> int:
        try:
            deliver_signal(signal.SIGTERM)
        finally:
            # Cleanup outside the teardown guard (a context manager's exit on
            # the way out): a second signal there is recorded, not raised.
            deliver_signal(signal.SIGINT)
            events.append("context exit finished")
        return 0

    assert run() == 128 + signal.SIGTERM, "the first signal ends the run"
    assert events == ["context exit finished"], (
        "the second signal must not cut short the cleanup the first one started"
    )


def test_an_ignored_sighup_stays_ignored() -> None:
    """``nohup`` starts the runner with SIGHUP ignored; that choice stands."""

    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    seen: list[object] = []

    @recovery.abort_on_signals
    def run() -> int:
        seen.append(signal.getsignal(signal.SIGHUP))
        signal.raise_signal(signal.SIGHUP)
        return 0

    try:
        code = run()
        after = signal.getsignal(signal.SIGHUP)
    finally:
        signal.signal(signal.SIGHUP, previous)

    assert code == 0, "an ignored SIGHUP must not abort a nohup run"
    assert seen == [signal.SIG_IGN], "the guard must not take over an ignored signal"
    assert after is signal.SIG_IGN, "and SIGHUP is still ignored after the run"


def test_the_teardown_guard_alone_records_and_restores() -> None:
    """Outside a guarded run (a standalone helper), the teardown guard still
    holds its block: a signal is recorded for ``abort`` and never raised."""

    with recovery.cleanup_signal_guard() as late:
        assert all(
            signal.getsignal(value) is late for value in recovery.TERMINATION_SIGNALS
        ), "every termination signal is recorded while the teardown runs"
        deliver_signal(signal.SIGTERM)

    assert late.received == [signal.SIGTERM], "the signal was recorded"
    aborted = late.abort(None)
    assert isinstance(aborted, recovery.RunTerminated), "the run ends aborted"
    assert str(aborted) == "received SIGTERM during the teardown", str(aborted)
    earlier = RuntimeError("load observation failed")
    assert late.abort(earlier) is earlier, "an earlier failure stays the run's failure"


def test_a_real_sigterm_from_outside_reaches_the_teardown(tmp_path: Path) -> None:
    """The same contract with a real ``kill`` of a separate process: before the
    guard the default action ended the process with no ``finally`` at all."""

    script = tmp_path / "runner.py"
    events = tmp_path / "events.json"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, os, signal, sys, time
            sys.path.insert(0, {str(ROOT)!r})
            from scripts.perf import regional_capacity_recovery as recovery

            events = []

            @recovery.abort_on_signals
            def run():
                failure = None
                try:
                    events.append("load")
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(30)
                    events.append("load continued after the signal")
                except BaseException as exc:
                    failure = exc
                finally:
                    with recovery.cleanup_signal_guard() as late:
                        events.append("teardown")
                    failure = late.abort(failure)
                    with open({str(events)!r}, "w") as handle:
                        json.dump(events + [str(failure)], handle)
                if failure is not None:
                    raise failure
                return 0

            sys.exit(run())
            """
        )
    )

    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        },
    )

    assert completed.returncode == 128 + signal.SIGTERM, completed.stderr
    assert json.loads(events.read_text()) == ["load", "teardown", "received SIGTERM"], (
        "the killed runner must still run its teardown and record the signal"
    )
