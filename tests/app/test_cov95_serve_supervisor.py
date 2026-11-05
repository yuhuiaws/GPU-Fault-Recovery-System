"""``gpu_fault.app.serve.Supervisor``: uvicorn's multiprocess contract, in-process.

The supervisor loop runs here for real: ``Supervisor.run`` installs its
SIGINT/SIGTERM/SIGHUP handlers in this process and the tests invoke those
handlers the way the interpreter does when a signal arrives. Only the children
are fakes - ``ScriptedWorker`` (``tests/app/_cov95_serve_fakes.py``) answers
health checks from a per-spawn plan - so each test reads the exact order in
which workers were started, replaced, terminated and joined. The real
multi-process launch is ``tests/app/test_serve_reuse_port.py``.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from gpu_fault.app import serve
from tests.app._cov95_serve_fakes import (
    Fleet,
    LogRecords,
    WorkerPlan,
    record_launcher_logs,
    signal_handlers_restored,
    signals_on_call,
    uvicorn_logging_restored,
)

APP = "tests.app._reuseport_app:app"


@pytest.fixture(autouse=True)
def supervisor_globals(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(serve, "SUPERVISOR_TICK_SECONDS", 0.01)
    with signal_handlers_restored(), uvicorn_logging_restored():
        yield


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> LogRecords:
    return record_launcher_logs(monkeypatch)


def _run(
    monkeypatch: pytest.MonkeyPatch, fleet: Fleet, **settings: Any
) -> tuple[int, serve.Supervisor]:
    monkeypatch.setattr(serve, "WorkerProcess", fleet.worker)
    supervisor = serve.Supervisor(serve.ServerSettings(app=APP, workers=2, **settings))
    fleet.supervisor = supervisor
    code = supervisor.run()
    assert not fleet.runaway, "the scripted stop never reached the supervisor"
    return code, supervisor


def test_main_supervises_several_workers_and_stops_them_all_on_sigterm(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet(WorkerPlan(), WorkerPlan(on_alive=signals_on_call(1, signal.SIGTERM)))
    monkeypatch.setattr(serve, "WorkerProcess", fleet.worker)
    argv = [APP, "--workers", "2", "--port", "0", "--timeout-worker-healthcheck", "7"]
    assert serve.main(argv) == 0, "a graceful stop exits 0"
    assert not fleet.runaway, "SIGTERM must reach the supervisor"
    assert [(worker.index, worker.settings.workers) for worker in fleet.spawned] == [
        (0, 2),
        (1, 2),
    ]
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("terminate", "w0"),
        ("terminate", "w1"),
        ("join", "w0"),
        ("join", "w1"),
    ], "every worker is told to stop before the parent waits for any of them"
    timeouts = {t for worker in fleet.spawned for t in worker.health_timeouts}
    assert timeouts == {7}, "health checks use --timeout-worker-healthcheck"
    messages = logs.messages()
    assert messages[0] == (
        f"Started parent process [{os.getpid()}]: 2 workers, one SO_REUSEPORT "
        "socket each on 127.0.0.1:0"
    )
    assert "Received SIGTERM, exiting." in messages, messages
    assert messages[-1] == f"Stopping parent process [{os.getpid()}]"


def test_a_child_that_failed_to_start_stops_the_parent_with_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet(
        WorkerPlan(alive=(False,), exitcode=serve.STARTUP_FAILURE), WorkerPlan()
    )
    code, supervisor = _run(monkeypatch, fleet)
    assert code == serve.STARTUP_FAILURE, (
        "the Pod restarts and backs off instead of crash-looping one child"
    )
    assert supervisor.startup_failed is True
    assert len(fleet.spawned) == 2, "a start-up failure is never answered by a respawn"
    assert fleet.spawned[1].alive_calls == 0, "the parent stops at the first failure"
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("kill", "w0"),
        ("join", "w0"),
        ("terminate", "w0"),
        ("terminate", "w1"),
        ("join", "w0"),
        ("join", "w1"),
    ]
    assert logs.messages(logging.ERROR) == [
        f"Child process [{fleet.spawned[0].pid}] failed to start; stopping the parent."
    ]


def test_a_child_that_died_is_replaced_in_its_slot(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet(
        WorkerPlan(alive=(False,), exitcode=-signal.SIGKILL),
        WorkerPlan(),
        WorkerPlan(on_alive=signals_on_call(1, signal.SIGINT)),
    )
    code, supervisor = _run(monkeypatch, fleet)
    died, kept, replacement = fleet.spawned
    assert code == 0
    assert supervisor.startup_failed is False
    assert supervisor.workers == [replacement, kept], (
        "the replacement takes the dead worker's slot; its sibling is untouched"
    )
    assert replacement.index == 0, "the replacement runs as the same worker index"
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("kill", "w0"),
        ("join", "w0"),
        ("start", "w2"),
        ("terminate", "w2"),
        ("terminate", "w1"),
        ("join", "w2"),
        ("join", "w1"),
    ]
    messages = logs.messages()
    assert f"Child process [{died.pid}] died (exit -9)" in messages, messages
    assert "Received SIGINT, exiting." in messages, messages


def test_a_stop_that_lands_while_a_dead_child_is_reaped_spawns_no_replacement(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet()
    fleet.plans = [
        WorkerPlan(alive=(False,), on_join=fleet.stop_supervisor),
        WorkerPlan(),
    ]
    code, supervisor = _run(monkeypatch, fleet)
    assert code == 0, "a stop is not a start-up failure"
    assert len(fleet.spawned) == 2, "no replacement is started for a stopping parent"
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("kill", "w0"),
        ("join", "w0"),
        ("terminate", "w0"),
        ("terminate", "w1"),
        ("join", "w0"),
        ("join", "w1"),
    ]
    assert not [line for line in logs.messages() if "died" in line], logs.messages()


def test_sighup_restarts_the_workers_one_at_a_time_new_before_old(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet(
        WorkerPlan(on_alive=signals_on_call(1, signal.SIGHUP)),
        WorkerPlan(),
        WorkerPlan(on_alive=signals_on_call(1, signal.SIGTERM)),
        WorkerPlan(),
    )
    code, supervisor = _run(monkeypatch, fleet, timeout_worker_healthcheck=9)
    old_0, old_1, new_0, new_1 = fleet.spawned
    assert code == 0
    assert supervisor.workers == [new_0, new_1]
    assert (new_0.index, new_1.index) == (0, 1)
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("start", "w2"),
        ("wait", "w2"),
        ("terminate", "w0"),
        ("join", "w0"),
        ("start", "w3"),
        ("wait", "w3"),
        ("terminate", "w1"),
        ("join", "w1"),
        ("terminate", "w2"),
        ("terminate", "w3"),
        ("join", "w2"),
        ("join", "w3"),
    ], "each old worker stops only once its replacement serves"
    assert new_0.ready_args == (9, supervisor.should_exit)
    assert "Received SIGHUP, restarting workers one at a time." in logs.messages()


def test_a_replacement_that_is_not_ready_aborts_the_restart_and_keeps_the_old_worker(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet(
        WorkerPlan(on_alive=signals_on_call(1, signal.SIGHUP)),
        WorkerPlan(on_alive=signals_on_call(2, signal.SIGTERM)),
        WorkerPlan(ready=False),
    )
    code, supervisor = _run(monkeypatch, fleet)
    old_0, old_1, late = fleet.spawned
    assert code == 0
    assert supervisor.workers == [old_0, old_1], "the serving workers are kept"
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("start", "w2"),
        ("wait", "w2"),
        ("kill", "w2"),
        ("join", "w2"),
        ("terminate", "w0"),
        ("terminate", "w1"),
        ("join", "w0"),
        ("join", "w1"),
    ], "the rest of the restart is abandoned"
    assert logs.messages(logging.ERROR) == [
        f"New child process [{late.pid}] was not ready in time; keeping worker "
        f"[{old_0.pid}] and aborting the restart."
    ]


def test_a_replacement_abandoned_because_the_parent_stops_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    def stop_while_waiting(should_exit: threading.Event) -> None:
        should_exit.set()

    fleet = Fleet(
        WorkerPlan(on_alive=signals_on_call(1, signal.SIGHUP)),
        WorkerPlan(),
        WorkerPlan(ready=False, on_wait=stop_while_waiting),
    )
    code, supervisor = _run(monkeypatch, fleet)
    old_0, old_1, _abandoned = fleet.spawned
    assert code == 0
    assert supervisor.workers == [old_0, old_1]
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("start", "w2"),
        ("wait", "w2"),
        ("kill", "w2"),
        ("join", "w2"),
        ("terminate", "w0"),
        ("terminate", "w1"),
        ("join", "w0"),
        ("join", "w1"),
    ]
    assert logs.messages(logging.ERROR) == [], "a stop is not a failed restart"


def test_a_sighup_queued_behind_sigterm_restarts_nothing(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet(
        WorkerPlan(on_alive=signals_on_call(1, signal.SIGTERM, signal.SIGHUP)),
        WorkerPlan(),
    )
    code, _supervisor = _run(monkeypatch, fleet)
    assert code == 0
    assert len(fleet.spawned) == 2, "no replacement is spawned for a stopping parent"
    assert fleet.events == [
        ("start", "w0"),
        ("start", "w1"),
        ("terminate", "w0"),
        ("terminate", "w1"),
        ("join", "w0"),
        ("join", "w1"),
    ]
    messages = logs.messages()
    assert messages.index("Received SIGTERM, exiting.") < messages.index(
        "Received SIGHUP, restarting workers one at a time."
    ), messages


def test_a_signal_without_an_action_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    fleet = Fleet(
        WorkerPlan(
            on_alive=signals_on_call(
                1, signal.SIGUSR1, signal.SIGTERM, handler_of=signal.SIGTERM
            )
        ),
        WorkerPlan(),
    )
    code, supervisor = _run(monkeypatch, fleet)
    assert code == 0
    assert supervisor.workers == fleet.spawned, "no worker was replaced"
    assert [line for line in logs.messages() if line.startswith("Received")] == [
        "Received SIGTERM, exiting."
    ], "only SIGINT, SIGTERM and SIGHUP are acted on"
