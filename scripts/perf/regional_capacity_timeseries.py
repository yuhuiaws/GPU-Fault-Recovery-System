"""What the processor did second by second while a capacity run's backlog lasted.

The burst-50 regression analysis had two snapshots per run (``metrics-before``
and ``metrics-after``) and a drain probe on one Pod, so "were the workers busy
while the queue was full" had no answer. During the load and the drain each
control-plane Pod now runs one long-lived sampler: one ``kubectl exec -i``
session whose program reads the Pod's own ``/metrics`` once per second and
prints the sampled families as one JSON line (``SAMPLED_FAMILIES``; the Pod's
render already merges its processes -- in-flight and workers are Pod sums,
consumer cycles stay per process). The session's program arrives
length-prefixed on stdin and the rest of stdin is the stop channel, as for the
validation-telemetry watch; a session spawns one Python process per Pod
instead of one per sample. ``processor-timeseries.json`` keeps at most
``SAMPLE_LIMIT`` samples per Pod, and :func:`series_summary` reduces them to
the mean in-flight against the Pod's worker capacity while the backlog lasted.

The queue drain (:func:`queue_drain`, moved here from the suite) now starts the
moment the load Job completes and runs beside the after-load collection, so
``depth_at_load_stop`` and ``drain_seconds`` are measured from the load stop.
:func:`drain_reference` also gives the values the old placement (after the
metrics, cgroup, PostgreSQL and log collection, about 45 s later) would have
read, renamed ``*_after_collection``, so older runs stay comparable.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import regional_capacity_registry as registry
else:
    import regional_capacity_registry as registry

SAMPLED_FAMILIES = (
    "gpu_fault_processor_in_flight",
    "gpu_fault_processor_workers",
    "gpu_fault_processor_queue_depth",
    "gpu_fault_processor_consumer_cycles_total",
    "gpu_fault_processor_claimed_not_started",
    "gpu_fault_telemetry_spool_depth",
)
SAMPLE_INTERVAL_SECONDS = 1.0
#: One hour at one sample per second: a bound on the artifact, not a window.
SAMPLE_LIMIT = 3600
STOP_TIMEOUT_SECONDS = 30.0
BOOTSTRAP = (
    "import sys;stream=sys.stdin.buffer;size=int(stream.readline());"
    "source=stream.read(size).decode();"
    "exec(compile(source,'capacity_sampler.py','exec'),{'__name__':'__main__'})"
)
#: Runs in the Pod: the sampled families of its own /metrics once per interval.
SAMPLE_PROGRAM = r"""
import json, os, sys, threading, time, urllib.request
port = int(os.environ["SAMPLE_PORT"])
interval = float(os.environ["SAMPLE_INTERVAL"])
limit = int(os.environ["SAMPLE_LIMIT"])
families = frozenset(os.environ["SAMPLE_FAMILIES"].split(","))
stop = threading.Event()
def wait_for_eof():
    try:
        while sys.stdin.buffer.read(65536):
            pass
    finally:
        stop.set()
threading.Thread(target=wait_for_eof, daemon=True).start()
url = "http://127.0.0.1:%d/metrics" % port
taken = 0
while not stop.is_set() and taken < limit:
    started = time.time()
    try:
        body = urllib.request.urlopen(url, timeout=5).read().decode()
        values = {}
        for line in body.splitlines():
            name, _, value = line.rpartition(" ")
            if name.split("{", 1)[0] in families:
                try:
                    values[name] = float(value)
                except ValueError:
                    pass
        print(json.dumps({"t": round(started, 3), "v": values}), flush=True)
    except Exception as exc:
        print(json.dumps({"t": round(started, 3), "error": type(exc).__name__}), flush=True)
    taken += 1
    stop.wait(max(0.0, interval - (time.time() - started)))
"""


def metrics_port(pod: str) -> str:
    """The port a control-plane Pod serves ``/metrics`` on."""

    if "telemetry-spool-worker" in pod:
        return "8082"
    if "control-worker" in pod:
        return "8081"
    return "8080"


def session_input(program: str) -> bytes:
    encoded = program.encode()
    return str(len(encoded)).encode() + b"\n" + encoded


class _PodSession:
    def __init__(self, pod: str, process: Any, limit: int) -> None:
        self.pod = pod
        self.process = process
        self.limit = limit
        self.samples: list[dict[str, Any]] = []
        self.truncated = False
        self.error: str | None = None
        self.reader = threading.Thread(
            target=self._read, name=f"timeseries-{pod}", daemon=True
        )

    def _read(self) -> None:
        for raw in self.process.stdout or ():
            try:
                sample = json.loads(raw.decode(errors="replace"))
            except ValueError:
                continue
            if not isinstance(sample, dict):
                continue
            if len(self.samples) >= self.limit:
                self.truncated = True
                continue
            self.samples.append(sample)


class ProcessorTimeseries:
    """The run's per-Pod sampling sessions, from :meth:`start` to :meth:`stop`."""

    def __init__(
        self,
        pods: Iterable[str],
        *,
        interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
        limit: int = SAMPLE_LIMIT,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.pods = list(pods)
        self.interval_seconds = interval_seconds
        self.limit = limit
        self.popen = popen
        self.sessions: dict[str, _PodSession] = {}
        self.failed: dict[str, str] = {}

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
            f"SAMPLE_PORT={metrics_port(pod)}",
            f"SAMPLE_INTERVAL={self.interval_seconds:g}",
            f"SAMPLE_LIMIT={self.limit}",
            "SAMPLE_FAMILIES=" + ",".join(SAMPLED_FAMILIES),
            "python3",
            "-c",
            BOOTSTRAP,
        ]

    def start(self) -> ProcessorTimeseries:
        for pod in self.pods:
            try:
                process = self.popen(
                    self._argv(pod),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                process.stdin.write(session_input(SAMPLE_PROGRAM))
                process.stdin.flush()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.failed[pod] = f"{type(exc).__name__}: {exc}"
                continue
            session = _PodSession(pod, process, self.limit)
            session.reader.start()
            self.sessions[pod] = session
        return self

    def stop(self, *, timeout_seconds: float = STOP_TIMEOUT_SECONDS) -> None:
        """Close every session's stdin (its stop channel); kill what lingers."""

        for session in self.sessions.values():
            try:
                session.process.stdin.close()
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + timeout_seconds
        for session in self.sessions.values():
            session.reader.join(timeout=max(0.0, deadline - time.monotonic()))
            if session.reader.is_alive():
                session.error = "the session did not stop; its kubectl was killed"
                try:
                    session.process.kill()
                except OSError:
                    pass

    def document(self) -> dict[str, Any]:
        return {
            "interval_seconds": self.interval_seconds,
            "families": list(SAMPLED_FAMILIES),
            "limit_per_pod": self.limit,
            "pods": {
                pod: {
                    "port": metrics_port(pod),
                    "samples": session.samples,
                    "truncated": session.truncated,
                    "error": session.error,
                }
                for pod, session in sorted(self.sessions.items())
            },
            "failed": dict(sorted(self.failed.items())),
        }

    def summary(self) -> dict[str, Any]:
        pods = {
            pod: series_summary(session.samples)
            for pod, session in sorted(self.sessions.items())
        }
        busy = [item for item in pods.values() if item.get("capacity_workers")]
        return {
            "interval_seconds": self.interval_seconds,
            "pods": pods,
            "failed": sorted(self.failed),
            "consumers": {
                "pods": len(busy),
                "capacity_workers": sum(item["capacity_workers"] for item in busy),
                "mean_in_flight_while_backlog": round(
                    sum(item["mean_in_flight_while_backlog"] or 0.0 for item in busy),
                    3,
                ),
            },
        }


def _value(values: dict[str, Any], family: str) -> float | None:
    """A Pod-level family: its unlabelled sample, else the sum of its labels."""

    if family in values:
        return float(values[family])
    labelled = [
        float(value) for name, value in values.items() if name.startswith(family + "{")
    ]
    return sum(labelled) if labelled else None


def series_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """One Pod's series: mean in-flight against its worker capacity while the
    queue held a backlog, and its consumer loop's cycle rate meanwhile."""

    rows = [(float(item["t"]), item["v"]) for item in samples if "v" in item]
    backlog = [
        (t, values)
        for t, values in rows
        if (_value(values, "gpu_fault_processor_queue_depth") or 0.0) > 0
    ]
    in_flight = [
        value
        for _t, values in backlog
        if (value := _value(values, "gpu_fault_processor_in_flight")) is not None
    ]
    capacity = max(
        (_value(values, "gpu_fault_processor_workers") or 0.0 for _t, values in rows),
        default=0.0,
    )
    cycles = [
        (t, value)
        for t, values in backlog
        if (value := _value(values, "gpu_fault_processor_consumer_cycles_total"))
        is not None
    ]
    mean = round(statistics.mean(in_flight), 3) if in_flight else None
    rate = None
    if len(cycles) >= 2 and cycles[-1][0] > cycles[0][0]:
        rate = round((cycles[-1][1] - cycles[0][1]) / (cycles[-1][0] - cycles[0][0]), 3)
    return {
        "samples": len(samples),
        "errors": len(samples) - len(rows),
        "backlog_samples": len(backlog),
        "capacity_workers": int(capacity),
        "mean_in_flight_while_backlog": mean,
        "max_in_flight_while_backlog": max(in_flight) if in_flight else None,
        "utilization_while_backlog": (
            round(mean / capacity, 3) if mean is not None and capacity else None
        ),
        "consumer_cycles_per_second_while_backlog": rate,
    }


def queue_drain(
    pods: list[str],
    *,
    scrape: Callable[[list[str]], dict[str, str]],
    timeout_seconds: int = 900,
    interval_seconds: float = 1.0,
    target_queue_depth: float = 1.0,
    target_spool_depth: float = 1.0,
) -> dict:
    """Wait for the processor queue to return to the steady state.

    The acceptance gate is "back to steady state within 60s of the burst",
    so the sampling resolution has to be small compared to 60s. One
    ``/metrics`` scrape costs about 2s (kubectl exec), and the loop stops at
    the first empty sample, which means ``drain_seconds`` is an upper bound
    with roughly ``interval_seconds + 2s`` of granularity. Keep the interval
    at 1s: a 15s interval reported every fast drain as "about 2s" and every
    slow one as a multiple of 17s, which is not comparable across runs.
    """
    started = time.time()
    history = []
    probe = pods[0] if pods else None
    if probe is None:
        return {"samples": history}
    while time.time() - started < timeout_seconds:
        snapshot = scrape([probe]).get(probe, "")
        depth = None
        spool_depth = None
        oldest = None
        for line in snapshot.splitlines():
            if line.startswith("gpu_fault_processor_queue_depth "):
                depth = float(line.split()[-1])
            if line.startswith("gpu_fault_telemetry_spool_depth "):
                spool_depth = float(line.split()[-1])
            if line.startswith("gpu_fault_processor_queue_oldest_age_seconds "):
                oldest = float(line.split()[-1])
        history.append(
            {
                "epoch": time.time(),
                "elapsed_seconds": time.time() - started,
                "queue_depth": depth,
                "telemetry_spool_depth": spool_depth,
                "oldest_age_seconds": oldest,
            }
        )
        if (
            depth is not None
            and depth <= target_queue_depth
            and (spool_depth is None or spool_depth <= target_spool_depth)
        ):
            break
        time.sleep(interval_seconds)
    return {
        "started_at_epoch": started,
        "drain_seconds": history[-1]["elapsed_seconds"] if history else None,
        "sample_interval_seconds": interval_seconds,
        "depth_at_load_stop": history[0]["queue_depth"] if history else None,
        "telemetry_spool_depth_at_load_stop": (
            history[0]["telemetry_spool_depth"] if history else None
        ),
        "target_queue_depth": target_queue_depth,
        "target_spool_depth": target_spool_depth,
        "samples": history,
    }


def drain_reference(drain: dict, collected_at: float) -> dict:
    """The drain as the old placement would have read it, renamed.

    Before, the drain probe started only once the after-load collection had
    finished: its first sample was the depth then and its drain time counted
    from then. The new probe samples from the load stop, so both are read off
    its history at ``collected_at``.
    """

    history = drain.get("samples") or []
    if not history:
        return dict(drain)
    later = [item for item in history if item["epoch"] >= collected_at]
    first = later[0] if later else history[-1]
    return {
        **drain,
        "collection_finished_at_epoch": collected_at,
        "depth_after_collection": first["queue_depth"],
        "drain_seconds_after_collection": round(
            max(0.0, history[-1]["epoch"] - collected_at), 3
        ),
    }


class Background(threading.Thread):
    """One call on its own thread; :meth:`result` returns it or raises its error."""

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(daemon=True)
        self._call = (function, args, kwargs)
        self._value: Any = None
        self._error: BaseException | None = None

    def run(self) -> None:
        function, args, kwargs = self._call
        try:
            self._value = function(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised by result()
            self._error = exc

    def result(self, timeout: float | None = None) -> Any:
        self.join(timeout)
        if self.is_alive():
            raise TimeoutError("the background call did not finish")
        if self._error is not None:
            raise self._error
        return self._value
