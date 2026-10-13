"""Event-driven validation telemetry for the integrated workflow suite (§8.4).

Post-reset/reboot validation accepts only telemetry observed after the control
plane recorded the reset or reboot, and a synthetic node publishes nothing on
its own. The runner's 30 s refresh used to be the only publisher, so every such
workflow waited 5-30 s for it and the integrated percentiles measured the
refresh period (batch 80). This keeps one ``kubectl exec -i`` session of
``seed_regional_action_workflows.py`` in its ``integrated-validation-watch``
mode running in an api-ha Pod for the whole workflow phase: the session
publishes a node's healthy package right after the node's barrier is recorded,
stamped with the actual publish time. The periodic refresh stays as the
fallback.

The session's program arrives length-prefixed on stdin and the rest of stdin is
its stop channel: closing it ends the session (``stop``), and so does a runner
that dies, because the pipe closes with it. A session also ends on its own once
every run workflow is terminal, and at its deadline. A session that is lost
early (the exec stream dropped) is relaunched a bounded number of times.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import regional_capacity_registry as registry
else:
    import regional_capacity_registry as registry

SEED_SCRIPT = Path(__file__).resolve().parent / "seed_regional_action_workflows.py"
WATCH_MODE = "integrated-validation-watch"
#: ``python3 -`` reads stdin to its end before it runs anything; this reads
#: exactly the length-prefixed program and leaves the rest of stdin open.
BOOTSTRAP = (
    "import sys;stream=sys.stdin.buffer;size=int(stream.readline());"
    "source=stream.read(size).decode();"
    "exec(compile(source,'seed_regional_action_workflows.py','exec'),"
    "{'__name__':'__main__'})"
)
#: A session that ended like this is done; anything else was lost early.
FINAL_ENDINGS = frozenset({"all-terminal", "deadline"})
STDERR_TAIL_LINES = 20
STDERR_LINE_LIMIT = 300


def session_input(program: bytes) -> bytes:
    return str(len(program)).encode() + b"\n" + program


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * ratio)]


def delay_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


class ValidationTelemetryWatcher:
    """The runner's handle on the in-Pod watch session (start, then stop)."""

    def __init__(
        self,
        *,
        run_id: str,
        clusters: int,
        workflows_per_cluster: int,
        deadline_seconds: float,
        poll_seconds: float = 1.0,
        max_restarts: int = 3,
        restart_pause_seconds: float = 2.0,
        popen: Callable[..., Any] = subprocess.Popen,
        control_fn: Callable[..., str] | None = None,
    ) -> None:
        self.run_id = run_id
        self.clusters = clusters
        self.workflows_per_cluster = workflows_per_cluster
        self.deadline_seconds = deadline_seconds
        self.poll_seconds = poll_seconds
        self.max_restarts = max_restarts
        self.restart_pause_seconds = restart_pause_seconds
        self.popen = popen
        self.control = registry.control if control_fn is None else control_fn
        self.sessions: list[dict[str, Any]] = []
        self.delays: list[float] = []
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._process: Any = None
        self._thread: threading.Thread | None = None
        self.summary: dict[str, Any] = {}

    def __enter__(self) -> ValidationTelemetryWatcher:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.summary = self.stop()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._supervise, name="validation-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 60.0) -> dict[str, Any]:
        """End the session and return the run's event-driven evidence.

        Never raises: a session that does not end within ``timeout_seconds``
        has its local ``kubectl`` killed (which closes the remote stdin as
        well) and the summary says ``stop_confirmed: false``.
        """

        self._stopping.set()
        self._close_stdin()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_seconds)
        stop_confirmed = thread is None or not thread.is_alive()
        if not stop_confirmed:
            with self._lock:
                process = self._process
            if process is not None:
                try:
                    process.kill()
                except OSError:
                    pass
        return {
            "mode": "event-driven",
            "poll_seconds": self.poll_seconds,
            "stop_confirmed": stop_confirmed,
            "restarts": max(0, len(self.sessions) - 1),
            "publishes": sum(int(item.get("publishes", 0)) for item in self.sessions),
            "barrier_to_publish_seconds": delay_summary(self.delays),
            "sessions": list(self.sessions),
        }

    def _close_stdin(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.stdin is None:
            return
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass

    def _supervise(self) -> None:
        while not self._stopping.is_set():
            session = self._run_session()
            self.delays.extend(session.pop("barrier_to_publish_seconds", None) or [])
            self.sessions.append(session)
            if (
                self._stopping.is_set()
                or session.get("ended_by") in FINAL_ENDINGS
                or len(self.sessions) > self.max_restarts
            ):
                return
            self._stopping.wait(self.restart_pause_seconds)

    def _argv(self, pod: str) -> list[str]:
        return [
            "kubectl",
            "--kubeconfig",
            registry.CONTROL_KUBECONFIG,
            "-n",
            registry.CONTROL_NAMESPACE,
            "exec",
            "-i",
            pod,
            "--",
            "env",
            f"ACTION_RUN_ID={self.run_id}",
            f"ACTION_CLUSTERS={self.clusters}",
            f"ACTION_WORKFLOWS_PER_CLUSTER={self.workflows_per_cluster}",
            f"ACTION_SEED_MODE={WATCH_MODE}",
            f"ACTION_WATCH_POLL_SECONDS={self.poll_seconds}",
            f"ACTION_WATCH_DEADLINE_SECONDS={self.deadline_seconds:g}",
            "python3",
            "-c",
            BOOTSTRAP,
        ]

    def _run_session(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            pod = self.control(
                "get",
                "pod",
                "-l",
                "app=gpu-fault-api-ha",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ).strip()
            if not pod:
                raise RuntimeError("no Running api-ha Pod for the session")
            process = self.popen(
                self._argv(pod),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "returncode": None}
        with self._lock:
            self._process = process
        stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        reader = threading.Thread(
            target=_collect_lines,
            args=(process.stderr, stderr_tail),
            name="validation-telemetry-stderr",
            daemon=True,
        )
        reader.start()
        try:
            process.stdin.write(session_input(SEED_SCRIPT.read_bytes()))
            process.stdin.flush()
        except (OSError, ValueError):
            # The session died at once, or stop() closed its stdin first: the
            # remote bootstrap then reads no program and exits.
            pass
        if self._stopping.is_set():
            self._close_stdin()
        result: dict[str, Any] = {}
        for raw in process.stdout:
            document = _json_line(raw)
            if document.get("mode") == WATCH_MODE:
                result = document
        returncode = process.wait()
        reader.join(timeout=5)
        with self._lock:
            self._process = None
        session = {
            **result,
            "pod": pod,
            "returncode": returncode,
            "seconds": round(time.monotonic() - started, 3),
        }
        if returncode or not result:
            session["stderr_tail"] = list(stderr_tail)
        return session


def _json_line(raw: bytes) -> dict[str, Any]:
    try:
        document = json.loads(raw.decode(errors="replace"))
    except ValueError:
        return {}
    return document if isinstance(document, dict) else {}


def _collect_lines(stream: Iterable[bytes] | None, tail: deque[str]) -> None:
    for raw in stream or ():
        tail.append(raw.decode(errors="replace").rstrip()[:STDERR_LINE_LIMIT])


def start_validation_telemetry(**options: Any) -> ValidationTelemetryWatcher:
    """Start the run's watch session; leaving its ``with`` block stops it."""

    watcher = ValidationTelemetryWatcher(**options)
    watcher.start()
    return watcher
