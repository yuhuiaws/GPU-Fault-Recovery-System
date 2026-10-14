"""The integrated runner's event-driven validation telemetry session.

One ``kubectl exec -i`` session in an api-ha Pod runs the seed script's watch
mode for the whole workflow phase. Its program arrives length-prefixed on
stdin and the rest of stdin is the stop channel, so the runner can end the
session before teardown and a dying runner ends it too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_validation_telemetry as telemetry

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "scripts/perf/seed_regional_action_workflows.py"


class FakeStdin:
    def __init__(self, closed: threading.Event) -> None:
        self.closed = closed
        self.data = b""

    def write(self, data: bytes) -> int:
        self.data += data
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed.set()


class FakeSession:
    """A remote watch session: it reports only once its stdin closes."""

    def __init__(self, summary: dict[str, Any] | None, *, exits_early: bool) -> None:
        self.closed = threading.Event()
        self.stdin = FakeStdin(self.closed)
        self.summary = summary
        self.exits_early = exits_early
        self.returncode: int | None = None
        self.stderr: Iterator[bytes] = iter(
            [b"Traceback: lost the exec stream\n"] if exits_early else []
        )
        self.stdout = self._lines()
        self.killed = False

    def _lines(self) -> Iterator[bytes]:
        yield (json.dumps({"progress": {"polls": 1, "publishes": 0}}) + "\n").encode()
        if not self.exits_early:
            self.closed.wait(timeout=30)
        if self.summary is not None:
            yield (json.dumps(self.summary) + "\n").encode()
        self.returncode = 1 if self.exits_early else 0

    def wait(self, timeout: float | None = None) -> int:
        deadline = time.monotonic() + (timeout or 30)
        while self.returncode is None and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.returncode if self.returncode is not None else -9

    def kill(self) -> None:
        self.killed = True
        self.closed.set()


def session_summary(delays: list[float], ended_by: str = "stopped") -> dict[str, Any]:
    return {
        "mode": telemetry.WATCH_MODE,
        "run_id": "run-a",
        "ended_by": ended_by,
        "polls": 9,
        "publishes": len(delays),
        "nodes_watched": 4,
        "nodes_published": len(delays),
        "terminal_workflows": 0,
        "feed_truncated_polls": 0,
        "barrier_to_publish_seconds": delays,
    }


class Launcher:
    def __init__(self, sessions: list[FakeSession]) -> None:
        self.sessions = sessions
        self.argv: list[list[str]] = []
        self.launched: list[FakeSession] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeSession:
        assert kwargs == {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }, "the session needs all three pipes"
        self.argv.append(argv)
        session = self.sessions.pop(0)
        self.launched.append(session)
        return session


def control(*args: str, **_kwargs: Any) -> str:
    assert args == (
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ), "the watcher only looks up a Running api-ha Pod"
    return "api-a\n"


def watcher(launcher: Launcher, **overrides: Any) -> Any:
    return telemetry.ValidationTelemetryWatcher(
        run_id="run-a",
        clusters=32,
        workflows_per_cluster=4,
        deadline_seconds=2400,
        popen=launcher,
        control_fn=control,
        **overrides,
    )


def test_session_runs_the_seed_watch_mode_and_stops_when_stdin_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "CONTROL_KUBECONFIG", "/tmp/cpu.kubeconfig")
    monkeypatch.setattr(registry, "CONTROL_NAMESPACE", "gpu-fault-system")
    session = FakeSession(session_summary([0.4, 0.2, 1.1]), exits_early=False)
    launcher = Launcher([session])
    running = watcher(launcher)
    running.start()
    assert session.closed.wait(timeout=0.3) is False, (
        "the session must stay open until the runner stops it"
    )

    summary = running.stop()

    (argv,) = launcher.argv
    assert argv[:9] == [
        "kubectl",
        "--kubeconfig",
        "/tmp/cpu.kubeconfig",
        "-n",
        "gpu-fault-system",
        "exec",
        "-i",
        "api-a",
        "--",
    ]
    assert argv[-3:] == ["python3", "-c", telemetry.BOOTSTRAP]
    assert {
        "ACTION_RUN_ID=run-a",
        "ACTION_CLUSTERS=32",
        "ACTION_WORKFLOWS_PER_CLUSTER=4",
        f"ACTION_SEED_MODE={telemetry.WATCH_MODE}",
        "ACTION_WATCH_POLL_SECONDS=1.0",
        "ACTION_WATCH_DEADLINE_SECONDS=2400",
    } <= set(argv), argv
    program = SEED.read_bytes()
    assert session.stdin.data == str(len(program)).encode() + b"\n" + program, (
        "the seed program must arrive length-prefixed"
    )
    assert session.closed.is_set(), "stop() must close the session's stdin"
    assert summary["stop_confirmed"] is True
    assert summary["publishes"] == 3
    assert summary["restarts"] == 0
    assert summary["barrier_to_publish_seconds"] == {
        "count": 3,
        "p50": 0.4,
        "p95": 0.4,
        "p99": 0.4,
        "max": 1.1,
    }
    assert summary["sessions"][0]["ended_by"] == "stopped"
    assert "barrier_to_publish_seconds" not in summary["sessions"][0], (
        "per-session raw delays are folded into the run's percentiles"
    )


def test_a_lost_session_is_restarted_and_its_failure_recorded() -> None:
    lost = FakeSession(None, exits_early=True)
    replacement = FakeSession(session_summary([0.3]), exits_early=False)
    launcher = Launcher([lost, replacement])
    running = watcher(launcher, restart_pause_seconds=0.0)
    running.start()
    deadline = time.monotonic() + 10
    while len(launcher.launched) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    summary = running.stop()

    assert len(launcher.argv) == 2, "the lost session was not replaced"
    assert summary["restarts"] == 1
    assert summary["sessions"][0]["returncode"] == 1
    assert summary["sessions"][0]["stderr_tail"] == ["Traceback: lost the exec stream"]
    assert summary["sessions"][1]["ended_by"] == "stopped"
    assert summary["publishes"] == 1


def test_a_session_that_finished_on_its_own_is_not_restarted() -> None:
    finished = FakeSession(session_summary([0.2], "all-terminal"), exits_early=False)
    finished.closed.set()
    launcher = Launcher([finished])
    running = watcher(launcher)
    running.start()
    deadline = time.monotonic() + 10
    while finished.returncode is None and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)

    summary = running.stop()

    assert len(launcher.argv) == 1, "an all-terminal session must not be relaunched"
    assert summary["sessions"][0]["ended_by"] == "all-terminal"


def test_the_bootstrap_runs_the_real_seed_watch_mode_until_stdin_closes(
    tmp_path: Path,
) -> None:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        # The seed script imports gpu_fault.app, which the control-plane wheel
        # ships in the Pod; outside it (the deploy's release gate runs in a
        # deploy-host venv) only the checkout's src provides it.
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "HOME": str(tmp_path),
        "ACTION_RUN_ID": "run-a",
        "ACTION_CLUSTERS": "1",
        "ACTION_WORKFLOWS_PER_CLUSTER": "1",
        "ACTION_SEED_MODE": telemetry.WATCH_MODE,
        "ACTION_WATCH_POLL_SECONDS": "0.05",
    }
    process = subprocess.Popen(
        [sys.executable, "-c", telemetry.BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=environment,
    )
    assert process.stdin is not None
    process.stdin.write(telemetry.session_input(SEED.read_bytes()))
    process.stdin.flush()
    time.sleep(1.0)
    assert process.poll() is None, (
        "the watch session ended while its stdin was still open"
    )
    output, errors = process.communicate(timeout=120)  # closes stdin: EOF

    assert process.returncode == 0, errors.decode()
    summary = json.loads(output.decode().strip().splitlines()[-1])
    assert summary["mode"] == telemetry.WATCH_MODE
    assert summary["ended_by"] == "stopped", "stdin EOF is the stop signal"
    assert summary["publishes"] == 0
