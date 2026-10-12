#!/usr/bin/env python3
"""Local reproduction of the ingress liveness kill under a cold-connect burst.

The live 50-cluster integrated suite (50 load Pods x 256 blocking workers, a
new TCP connection per request, ~1,066 concurrent connections per uvicorn
worker process) was accepted in full by the api-ha tier - 0 non-2xx - and
then every api-ha Pod was killed by its liveness probe: ``/livez`` did not
answer inside 5 s twice in a row. The metrics that would have named the
mechanism are process-local and died with the processes, so this script
rebuilds the situation on one host and measures it.

It runs the real application (``gpu_fault.app:create_app`` in the ingress
role, regional mode, the rendered ingress environment, uvicorn with the
production flags) against the in-memory store with the admission flush
replaced by a stub that burns a configurable amount of CPU per item while
building the JSON rows - what the Aurora flush does before it waits on the
network - and drives it with ``clusters x workers`` closed-loop cold-connect
clients that send the live per-cluster event matrix. An external prober does
what the kubelet does (fresh connection, ``GET /livez``, 5 s timeout) once a
second, and a sampler inside every worker process records, four times a
second: event-loop lag at 10 ms resolution, the GIL acquisition wait of an
otherwise idle thread, every garbage collection's duration by generation,
RSS, runnable threads, the loop thread's share of the process CPU and the
batcher's backlog.

Nothing here touches a cluster or a database. Example::

    .venv/bin/python scripts/perf/ingress_event_loop_burst.py run \\
        --label slots8-decode32 --clusters 50 --workers-per-cluster 85 \\
        --flush-slots 8 --decode-workers 32 --flush-cpu-ms-per-item 8

``run`` prints one table per experiment and writes the raw samples under
``--out``; ``summarize <run-dir>`` re-prints a finished run. ``--launcher
uvicorn|serve`` compares uvicorn's shared listening socket with
``gpu_fault.app.serve``'s SO_REUSEPORT socket per worker (per-socket accept
queues and per-worker request shares are reported); ``--keep-alive`` makes
the load clients hold one connection per worker like a pooled collector.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import hashlib
import json
import os
import secrets
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).resolve().with_name("payload-templates.json")
GENERATED = REPO / "deploy" / "control-plane" / "regional" / "generated"
ENV_PREFIX = "INGRESS_BURST_"
PATHS = {
    "GPU_INVENTORY": "/v1/collector-events/gpu-inventory",
    "GPU_METRICS": "/v1/collector-events/gpu-metrics",
    "HOST_TELEMETRY": "/v1/collector-events/host-telemetry",
    "WORKLOAD_OBSERVATION": "/v1/workload-observations",
    "NVIDIA_KERNEL": "/v1/collector-events/nvidia-kernel",
    "FABRIC_MANAGER_LOG": "/v1/collector-events/fabric-manager",
}
CLOCK_TICK = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


# --------------------------------------------------------------------------
# The instrumented application (uvicorn factory target)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppSettings:
    sample_dir: Path
    sample_interval: float = 0.25
    flush_cpu_ms_per_item: float = 8.0
    flush_io_ms: float = 0.0
    gc_freeze: bool = False
    gc_threshold2: int = 0
    switch_interval: float = 0.0

    @classmethod
    def from_environment(cls) -> AppSettings:
        env = os.environ

        def read(name: str, default: str) -> str:
            return env.get(ENV_PREFIX + name, default)

        return cls(
            sample_dir=Path(read("SAMPLE_DIR", "/tmp/gpu-fault-ingress-burst/samples")),
            sample_interval=float(read("SAMPLE_INTERVAL", "0.25")),
            flush_cpu_ms_per_item=float(read("FLUSH_CPU_MS_PER_ITEM", "8")),
            flush_io_ms=float(read("FLUSH_IO_MS", "0")),
            gc_freeze=read("GC_FREEZE", "0") == "1",
            gc_threshold2=int(read("GC_THRESHOLD2", "0")),
            switch_interval=float(read("SWITCH_INTERVAL", "0")),
        )


def _row_unit(payload: Any) -> int:
    """One unit of the row building a flush does per item: serialise, hash."""

    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return len(hashlib.sha256(encoded).hexdigest())


def _calibrate_row_units(cpu_ms: float, sample_body: bytes) -> int:
    """How many row units burn ``cpu_ms`` of CPU for one ``sample_body``."""

    if cpu_ms <= 0:
        return 0
    payload = json.loads(sample_body)
    for _ in range(20):
        _row_unit(payload)
    rounds = 200
    started = time.thread_time()
    for _ in range(rounds):
        _row_unit(payload)
    unit_seconds = (time.thread_time() - started) / rounds
    return max(1, round(cpu_ms / 1000 / unit_seconds))


def install_flush_stub(store: Any, settings: AppSettings, sample_body: bytes) -> int:
    """Replace the store's batch admission with a CPU-burning accept-all.

    Returns the calibrated number of row units per item. The stub never
    calls the in-memory store: the live ingress keeps nothing of a request
    after the INSERT, and a growing in-memory queue would turn the GC
    measurement into a measurement of the stub.
    """

    units = _calibrate_row_units(settings.flush_cpu_ms_per_item, sample_body)
    io_seconds = settings.flush_io_ms / 1000

    def admit(requests: list[Any], **_kwargs: Any) -> list[tuple[Any, None]]:
        for request in requests:
            # ``ProcessorRequest`` keeps the body base64-encoded behind
            # ``body()``; decoding it is part of what the real flush pays.
            raw = request.body()
            payload = json.loads(raw) if raw else {}
            for _ in range(units):
                _row_unit(payload)
        if io_seconds > 0:
            time.sleep(io_seconds)
        return [(request, None) for request in requests]

    store.try_enqueue_processor_requests_batch = admit
    return units


class GcTimer:
    """Per-generation collection counts and durations via ``gc.callbacks``."""

    def __init__(self) -> None:
        self._started = 0.0
        self.lock = threading.Lock()
        self.window_count = [0, 0, 0]
        self.window_sum = [0.0, 0.0, 0.0]
        self.window_max = [0.0, 0.0, 0.0]
        self.total_count = [0, 0, 0]
        self.total_sum = [0.0, 0.0, 0.0]
        self.longest: list[tuple[float, int, float]] = []
        gc.callbacks.append(self._callback)

    def _callback(self, phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            self._started = time.perf_counter()
            return
        elapsed = time.perf_counter() - self._started
        generation = int(info.get("generation", 0))
        with self.lock:
            self.window_count[generation] += 1
            self.window_sum[generation] += elapsed
            self.window_max[generation] = max(self.window_max[generation], elapsed)
            self.total_count[generation] += 1
            self.total_sum[generation] += elapsed
            if generation == 2:
                self.longest.append((time.time(), generation, elapsed))
                self.longest.sort(key=lambda row: row[2], reverse=True)
                del self.longest[20:]

    def drain(self) -> dict[str, Any]:
        with self.lock:
            snapshot = {
                "gc_count": list(self.window_count),
                "gc_sum_ms": [value * 1000 for value in self.window_sum],
                "gc_max_ms": [value * 1000 for value in self.window_max],
                "gc_total_count": list(self.total_count),
                "gc_total_sum_ms": [value * 1000 for value in self.total_sum],
            }
            self.window_count = [0, 0, 0]
            self.window_sum = [0.0, 0.0, 0.0]
            self.window_max = [0.0, 0.0, 0.0]
            return snapshot


class WindowMax:
    def __init__(self) -> None:
        self.value = 0.0
        self.lock = threading.Lock()

    def observe(self, value: float) -> None:
        if value > self.value:
            with self.lock:
                self.value = max(self.value, value)

    def drain(self) -> float:
        with self.lock:
            value, self.value = self.value, 0.0
        return value


def _proc_stat_cpu(path: str) -> float:
    with open(path, "rb") as handle:
        fields = handle.read().rsplit(b")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / CLOCK_TICK


def _proc_stat_state(path: str) -> str:
    with open(path, "rb") as handle:
        return handle.read().rsplit(b")", 1)[1].split()[0].decode()


def _thread_states() -> tuple[int, int]:
    total = running = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            state = _proc_stat_state(f"/proc/self/task/{tid}/stat")
        except (OSError, IndexError, ValueError):
            # A thread that exited between the listing and the read.
            continue
        total += 1
        running += state == "R"
    return total, running


def listen_queues(port: int) -> tuple[list[int], int]:
    """(accept-queue depth of each listener on ``port``, established peers).

    ``/proc/net/tcp{,6}`` reports a LISTEN socket's rx_queue as the number
    of completed connections nobody has accept()ed yet - the queue the
    kubelet's probe connection joins behind a cold-connect storm. Under
    ``uvicorn --workers`` there is one socket; under ``gpu_fault.app.serve``
    one per worker, so the list is the per-worker split of the storm.
    """

    suffix = f":{port:04X}"
    queues: list[int] = []
    established = 0
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, encoding="ascii") as handle:
                next(handle)
                for line in handle:
                    fields = line.split()
                    if not fields[1].endswith(suffix):
                        continue
                    if fields[3] == "0A":
                        queues.append(int(fields[4].split(":")[1], 16))
                    elif fields[3] == "01":
                        established += 1
        except OSError:
            continue
    return queues, established


def listen_queue(port: int) -> tuple[int, int]:
    """(total accept-queue depth on ``port``, established peers)."""

    queues, established = listen_queues(port)
    return sum(queues), established


def _rss_mb() -> float:
    with open("/proc/self/statm", "rb") as handle:
        return int(handle.read().split()[1]) * PAGE_SIZE / 1_048_576


class ProcessSampler:
    """Writes one JSON line per window to ``<sample_dir>/proc-<pid>.jsonl``."""

    def __init__(self, inner: Any, settings: AppSettings, row_units: int) -> None:
        self.inner = inner
        self.settings = settings
        self.row_units = row_units
        self.gc_timer = GcTimer()
        self.loop_lag = WindowMax()
        self.gil_wait = WindowMax()
        self.served = 0
        self.livez_app = WindowMax()
        self.livez_count = 0
        self.stop = threading.Event()
        self._main_tid = os.getpid()

    def start(self) -> None:
        self.settings.sample_dir.mkdir(parents=True, exist_ok=True)
        asyncio.get_running_loop().create_task(
            self._fine_lag(), name="ingress-burst-fine-lag"
        )
        threading.Thread(
            target=self._gil_probe, name="ingress-burst-gil", daemon=True
        ).start()
        threading.Thread(
            target=self._sample, name="ingress-burst-sampler", daemon=True
        ).start()

    async def _fine_lag(self) -> None:
        loop = asyncio.get_running_loop()
        interval = 0.01
        while not self.stop.is_set():
            expected = loop.time() + interval
            await asyncio.sleep(interval)
            self.loop_lag.observe(max(0.0, loop.time() - expected))

    def _gil_probe(self) -> None:
        interval = 0.005
        while not self.stop.is_set():
            started = time.perf_counter()
            time.sleep(interval)
            self.gil_wait.observe(max(0.0, time.perf_counter() - started - interval))

    def _batcher_fields(self) -> dict[str, Any]:
        state = self.inner.state
        fields: dict[str, Any] = {}
        for name, prefix in (
            ("processor_admission_batcher", "adm"),
            ("fault_admission_batcher", "fault"),
        ):
            batcher = getattr(state, name, None)
            if batcher is None:
                continue
            fields[f"{prefix}_pending"] = batcher.pending_depth
            fields[f"{prefix}_in_flight"] = batcher.in_flight
            fields[f"{prefix}_limit"] = batcher.max_in_flight_flushes
            fields[f"{prefix}_flushes"] = batcher.flush_count
            fields[f"{prefix}_ewma_ms"] = batcher.round_seconds_ewma * 1000
        runtime = getattr(state, "runtime", None)
        if runtime is not None:
            fields["decode_in_flight"] = runtime.decode_io.in_flight
            fields["store_in_flight"] = runtime.store_io.in_flight
            count, total, maximum = runtime.event_loop_lag_snapshot()
            fields["app_lag_max_ms"] = maximum * 1000
            fields["app_lag_count"] = count
        return fields

    def _sample(self) -> None:
        pid = os.getpid()
        path = self.settings.sample_dir / f"proc-{pid}.jsonl"
        interval = self.settings.sample_interval
        last_wall = time.monotonic()
        last_proc = _proc_stat_cpu("/proc/self/stat")
        last_main = _proc_stat_cpu(f"/proc/self/task/{self._main_tid}/stat")
        last_served = 0
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "start",
                        "pid": pid,
                        "row_units": self.row_units,
                        "settings": {
                            **asdict(self.settings),
                            "sample_dir": str(self.settings.sample_dir),
                        },
                        "gc_threshold": gc.get_threshold(),
                        "switch_interval": sys.getswitchinterval(),
                        "decode_workers": os.getenv("GPU_FAULT_INGRESS_DECODE_WORKERS"),
                        "store_io_workers": os.getenv("GPU_FAULT_STORE_IO_WORKERS"),
                    }
                )
                + "\n"
            )
            errors = 0
            while not self.stop.wait(interval):
                now = time.monotonic()
                try:
                    proc = _proc_stat_cpu("/proc/self/stat")
                    main = _proc_stat_cpu(f"/proc/self/task/{self._main_tid}/stat")
                    threads, running = _thread_states()
                    batcher_fields = self._batcher_fields()
                except Exception as exc:  # noqa: BLE001 - a sampler must outlive a bad read
                    errors += 1
                    if errors <= 3:
                        print(
                            f"sampler pid={pid} skipped a window: {exc!r}", flush=True
                        )
                    continue
                served = self.served
                wall = max(1e-6, now - last_wall)
                sample = {
                    "t": time.time(),
                    "pid": pid,
                    "rss_mb": _rss_mb(),
                    "loop_lag_max_ms": self.loop_lag.drain() * 1000,
                    "gil_wait_max_ms": self.gil_wait.drain() * 1000,
                    "cpu_proc_cores": (proc - last_proc) / wall,
                    "cpu_main_cores": (main - last_main) / wall,
                    "threads": threads,
                    "threads_running": running,
                    "served": served - last_served,
                    "fds": len(os.listdir("/proc/self/fd")),
                    "livez_app_max_ms": self.livez_app.drain() * 1000,
                    "livez_count": self.livez_count,
                    "gc_counts_now": list(gc.get_count()),
                    "sampler_errors": errors,
                    **self.gc_timer.drain(),
                    **batcher_fields,
                }
                handle.write(json.dumps(sample) + "\n")
                handle.flush()
                last_wall, last_proc, last_main, last_served = now, proc, main, served
            handle.write(
                json.dumps(
                    {
                        "kind": "end",
                        "pid": pid,
                        "longest_gen2_ms": [
                            [stamp, ms * 1000] for stamp, _, ms in self.gc_timer.longest
                        ],
                        "gc_total_count": self.gc_timer.total_count,
                        "gc_total_sum_ms": [
                            value * 1000 for value in self.gc_timer.total_sum
                        ],
                        "tracked_objects": len(gc.get_objects()),
                    }
                )
                + "\n"
            )


class InstrumentedApp:
    """ASGI wrapper: starts the sampler once the inner lifespan is up."""

    def __init__(self, inner: Any, settings: AppSettings, row_units: int) -> None:
        self.inner = inner
        self.settings = settings
        self.sampler = ProcessSampler(inner, settings, row_units)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope.get("path") == "/livez":
            started = time.perf_counter()

            async def send_timed(message: Any) -> None:
                if message["type"] == "http.response.start":
                    self.sampler.livez_app.observe(time.perf_counter() - started)
                    self.sampler.livez_count += 1
                await send(message)

            await self.inner(scope, receive, send_timed)
            self.sampler.served += 1
            return
        if scope["type"] != "lifespan":
            await self.inner(scope, receive, send)
            if scope["type"] == "http":
                self.sampler.served += 1
            return

        async def send_wrapped(message: Any) -> None:
            if message["type"] == "lifespan.startup.complete":
                self._after_startup()
            elif message["type"] == "lifespan.shutdown.complete":
                self.sampler.stop.set()
            await send(message)

        await self.inner(scope, receive, send_wrapped)

    def _after_startup(self) -> None:
        if self.settings.gc_threshold2 > 0:
            gen0, gen1, _ = gc.get_threshold()
            gc.set_threshold(gen0, gen1, self.settings.gc_threshold2)
        if self.settings.gc_freeze:
            gc.collect()
            gc.freeze()
        if self.settings.switch_interval > 0:
            sys.setswitchinterval(self.settings.switch_interval)
        self.sampler.start()


def create_app() -> InstrumentedApp:
    """uvicorn ``--factory`` target; the ``run`` driver supplies the env."""

    from gpu_fault.app import ApplicationContext
    from gpu_fault.app import create_app as build_app

    settings = AppSettings.from_environment()
    context = ApplicationContext.from_environment()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    sample = json.dumps(templates["GPU_METRICS"], separators=(",", ":")).encode()
    row_units = install_flush_stub(context.store, settings, sample)
    inner = build_app(context)
    return InstrumentedApp(inner, settings, row_units)


# --------------------------------------------------------------------------
# Event matrix (the live suite's per-cluster shape) and the cold-connect load
# --------------------------------------------------------------------------


def cluster_events(
    templates: dict[str, Any],
    *,
    cluster_id: str,
    nodes: int,
    faults: int,
) -> list[tuple[str, bytes]]:
    """``(path, body)`` per event: three telemetry kinds per node plus faults."""

    now = datetime.now(timezone.utc).isoformat()
    events: list[tuple[str, bytes]] = []

    def stamped(kind: str, node_index: int, sequence: int) -> dict[str, Any]:
        value = copy.deepcopy(templates.get(kind) or {})
        value["cluster_id"] = cluster_id
        value["node_id"] = f"burst-node-{node_index:04d}"
        for name in ("observed_at", "collected_at", "ingested_at"):
            if name in value:
                value[name] = now
        for name in ("batch_id", "snapshot_id", "record_id"):
            if name in value:
                value[name] = f"burst-{kind.lower()}-{sequence}-{uuid4().hex[:8]}"
        if kind in {"GPU_METRICS", "HOST_TELEMETRY"}:
            value["affected_workload_ids"] = []
            value["workload_state"] = "IDLE"
            value["context_history"] = []
            value["edge_filter_reasons"] = ["health-summary"]
        return value

    sequence = 0
    for node_index in range(nodes):
        for kind in ("GPU_INVENTORY", "GPU_METRICS", "HOST_TELEMETRY"):
            sequence += 1
            body = json.dumps(
                stamped(kind, node_index, sequence), separators=(",", ":")
            ).encode()
            events.append((PATHS[kind], body))
    for index in range(faults):
        node_index = index % max(1, nodes)
        node_id = f"burst-node-{node_index:04d}"
        if index % 4 == 0:
            payload: dict[str, Any] = {
                "event_id": f"burst-xid-{cluster_id}-{index}-{uuid4().hex[:8]}",
                "cluster_id": cluster_id,
                "node_id": node_id,
                "observed_at": now,
                "source_event_time": now,
                "ingested_at": now,
                "event_source": "PERF_BURST",
                "xid": 63,
                "gpu_uuid": f"GPU-{uuid4()}",
                "product": "H100",
                "driver_branch": 575,
                "cuda_version": "12.9",
                "workload_state": "IDLE",
                "affected_workload_ids": [],
                "synthetic": True,
            }
            events.append((PATHS["NVIDIA_KERNEL"], json.dumps(payload).encode()))
        elif index % 4 == 1:
            payload = {
                "event_id": f"burst-sxid-{cluster_id}-{index}-{uuid4().hex[:8]}",
                "cluster_id": cluster_id,
                "node_id": node_id,
                "observed_at": now,
                "source_event_time": now,
                "ingested_at": now,
                "event_source": "PERF_BURST",
                "sxid": 11001,
                "classification": "NON_FATAL",
                "classification_source": "NVIDIA_FABRIC_MANAGER_CATALOG",
                "link_scope": "TRUNK",
                "link_scope_source": "TRUSTED_NVSWITCH_TOPOLOGY",
                "product": "H200",
                "fabric_partition": f"{cluster_id}/{node_id}/local-nvswitch",
                "participating_gpu_uuids": [f"GPU-{uuid4()}"],
                "workload_state": "IDLE",
                "affected_workload_ids": [],
                "synthetic": True,
            }
            events.append((PATHS["FABRIC_MANAGER_LOG"], json.dumps(payload).encode()))
        else:
            kind = "GPU_METRICS" if index % 4 == 2 else "HOST_TELEMETRY"
            sequence += 1
            payload = stamped(kind, node_index, sequence)
            payload["edge_filter_reasons"] = ["threshold:synthetic-priority-50"]
            payload["collection_errors"] = []
            events.append((PATHS[kind], json.dumps(payload).encode()))
    return events


def _request_bytes(
    path: str,
    body: bytes,
    *,
    host: str,
    cluster_id: str,
    token: str,
    keep_alive: bool = False,
) -> bytes:
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Authorization: Bearer {token}\r\n"
        f"X-GPU-Fault-Cluster-ID: {cluster_id}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: {'keep-alive' if keep_alive else 'close'}\r\n\r\n"
    )
    return head.encode() + body


async def _send_and_read(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request: bytes,
    timeout: float,
) -> int:
    """One request on an open connection; returns the status code."""

    writer.write(request)
    await asyncio.wait_for(writer.drain(), timeout)
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
    status = int(head.split(b" ", 2)[1])
    length = 0
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value.strip())
    if length:
        await asyncio.wait_for(reader.readexactly(length), timeout)
    return status


async def _http_exchange(
    host: str, port: int, request: bytes, *, bind: str | None, timeout: float
) -> int:
    """One cold connection, one request; returns the status code."""

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port, local_addr=(bind, 0) if bind else None),
        timeout,
    )
    try:
        return await _send_and_read(reader, writer, request, timeout)
    finally:
        writer.close()


@dataclass
class LoadStats:
    statuses: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0
    #: Connections opened; in keep-alive mode one per worker plus reconnects.
    connections: int = 0
    reconnects: int = 0


async def _cluster_workers(
    *,
    host: str,
    port: int,
    bind: str | None,
    cluster_id: str,
    token: str,
    events: list[tuple[str, bytes]],
    workers: int,
    timeout: float,
    stats: LoadStats,
    keep_alive: bool = False,
) -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    for path, body in events:
        queue.put_nowait(
            _request_bytes(
                path,
                body,
                host=f"{host}:{port}",
                cluster_id=cluster_id,
                token=token,
                keep_alive=keep_alive,
            )
        )

    def record(key: str, begin: float) -> None:
        stats.latencies_ms.append((time.perf_counter() - begin) * 1000)
        stats.statuses[key] = stats.statuses.get(key, 0) + 1

    async def cold_worker() -> None:
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            begin = time.perf_counter()
            try:
                stats.connections += 1
                status = await _http_exchange(
                    host, port, request, bind=bind, timeout=timeout
                )
                key = str(status)
            except Exception as exc:  # noqa: BLE001 - every failure is a sample
                key = type(exc).__name__
                stats.errors[key] = stats.errors.get(key, 0) + 1
            record(key, begin)

    async def keep_alive_worker() -> None:
        """One connection for the worker's whole run, reopened on a failure.

        What a collector process with a persistent pooled connection does:
        the handshake is paid once, every later request is a write and a
        read on the same socket pinned to the uvicorn worker that accepted
        it.
        """

        stream: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            begin = time.perf_counter()
            try:
                if stream is None:
                    stats.connections += 1
                    stream = await asyncio.wait_for(
                        asyncio.open_connection(
                            host, port, local_addr=(bind, 0) if bind else None
                        ),
                        timeout,
                    )
                status = await _send_and_read(stream[0], stream[1], request, timeout)
                key = str(status)
            except Exception as exc:  # noqa: BLE001 - every failure is a sample
                key = type(exc).__name__
                stats.errors[key] = stats.errors.get(key, 0) + 1
                if stream is not None:
                    stream[1].close()
                    stream = None
                    stats.reconnects += 1
            record(key, begin)
        if stream is not None:
            stream[1].close()

    worker = keep_alive_worker if keep_alive else cold_worker
    await asyncio.gather(*(worker() for _ in range(min(workers, len(events)))))


async def _load_main(args: argparse.Namespace) -> dict[str, Any]:
    templates = json.loads(Path(args.templates).read_text(encoding="utf-8"))
    if args.payload_scale > 1:
        for template in templates.values():
            if not isinstance(template, dict):
                continue
            lists = [key for key, value in template.items() if isinstance(value, list)]
            if lists:
                longest = max(lists, key=lambda key: len(json.dumps(template[key])))
                template[longest] = template[longest] * args.payload_scale
    tokens = json.loads(Path(args.tokens_file).read_text(encoding="utf-8"))
    clusters = [cluster for cluster in args.clusters.split(",") if cluster]
    plans = {
        cluster_id: cluster_events(
            templates,
            cluster_id=cluster_id,
            nodes=args.nodes_per_cluster,
            faults=args.faults_per_cluster,
        )
        for cluster_id in clusters
    }
    if args.start_at:
        await asyncio.sleep(max(0.0, args.start_at - time.time()))
    stats = LoadStats(started=time.time())
    await asyncio.gather(
        *(
            _cluster_workers(
                host=args.host,
                port=args.port,
                bind=args.bind or None,
                cluster_id=cluster_id,
                token=tokens[cluster_id],
                events=events,
                workers=args.workers_per_cluster,
                timeout=args.timeout,
                stats=stats,
                keep_alive=args.keep_alive,
            )
            for cluster_id, events in plans.items()
        )
    )
    stats.finished = time.time()
    ordered = sorted(stats.latencies_ms)

    def pct(ratio: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]

    wall = max(1e-6, stats.finished - stats.started)
    return {
        "clusters": clusters,
        "events": sum(len(events) for events in plans.values()),
        "requests": len(ordered),
        "statuses": stats.statuses,
        "errors": stats.errors,
        "started": stats.started,
        "finished": stats.finished,
        "wall_seconds": wall,
        "connection_mode": "keep-alive" if args.keep_alive else "cold-connect",
        "connections": stats.connections,
        "reconnects": stats.reconnects,
        "requests_per_second": len(ordered) / wall,
        "p50_ms": pct(0.5) if ordered else 0.0,
        "p95_ms": pct(0.95) if ordered else 0.0,
        "p99_ms": pct(0.99) if ordered else 0.0,
        "max_ms": ordered[-1] if ordered else 0.0,
    }


def run_load(args: argparse.Namespace) -> int:
    import uvloop

    result = uvloop.run(_load_main(args))
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------
# The kubelet-like liveness prober
# --------------------------------------------------------------------------


async def _probe_main(args: argparse.Namespace) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    request = (
        f"GET {args.path} HTTP/1.1\r\nHost: {args.host}:{args.port}\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    with Path(args.out).open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            queues, established = listen_queues(args.queue_port or args.port)
            queued = sum(queues)
            begin = time.perf_counter()
            stamp = time.time()
            try:
                status: int | str = await _http_exchange(
                    args.host, args.port, request, bind=None, timeout=args.timeout
                )
            except Exception as exc:  # noqa: BLE001 - a failed probe is the datum
                status = type(exc).__name__
            elapsed = time.perf_counter() - begin
            handle.write(
                json.dumps(
                    {
                        "t": stamp,
                        "status": status,
                        "ms": elapsed * 1000,
                        "accept_queue": queued,
                        "accept_queues": queues,
                        "established": established,
                    }
                )
                + "\n"
            )
            handle.flush()
            remaining = args.interval - (time.perf_counter() - begin)
            if remaining > 0:
                try:
                    await asyncio.wait_for(stop.wait(), remaining)
                except TimeoutError:
                    pass


def run_probe(args: argparse.Namespace) -> int:
    asyncio.run(_probe_main(args))
    return 0


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def _configmap_env() -> dict[str, str]:
    import yaml

    env: dict[str, str] = {}
    for path in sorted(GENERATED.glob("gpu-fault-api-ha-config-*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in (document.get("data") or {}).items():
            if key.startswith("GPU_FAULT_"):
                env[key] = str(value)
    return env


def server_environment(
    args: argparse.Namespace, run_dir: Path, clusters_json: str
) -> dict[str, str]:
    """The rendered ingress environment, minus Aurora/AWS, plus the experiment."""

    env = _configmap_env()
    for name in (
        "GPU_FAULT_STORE_URL_FILE",
        "GPU_FAULT_RDS_CA_BUNDLE",
        "GPU_FAULT_POSTGRES_FLEET_CONNECTION_BUDGET",
        # Falls back to GPU_FAULT_NODE_ACTION_SECRET, set below.
        "GPU_FAULT_NODE_ACTION_KEYS_DIR",
    ):
        env.pop(name, None)
    env.update(
        {
            "GPU_FAULT_ALLOW_EMAIL": "false",
            "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED": "false",
            "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true",
            "GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER": "false",
            "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER": "false",
            "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "hyperpod-v1",
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON": clusters_json,
            # The regional context refuses the in-memory store; the only
            # store calls left on the measured path are the per-second
            # registry refresh and the /metrics render, so sqlite will do.
            "GPU_FAULT_STORE_URL": f"sqlite:///{run_dir / 'store.db'}",
            "GPU_FAULT_EXECUTION_TOKEN": secrets.token_hex(24),
            "GPU_FAULT_PROCESSOR_REPLAY_SECRET": secrets.token_hex(24),
            "GPU_FAULT_NODE_ACTION_SECRET": secrets.token_hex(24),
            "GPU_FAULT_AGENT_REGISTRATION_SECRET": secrets.token_hex(24),
            "GPU_FAULT_SITE_ID": "ingress-burst-local",
            "POD_UID": f"ingress-burst-{uuid4().hex[:12]}",
            "GPU_FAULT_AWS_ACCOUNT_ID": "000000000000",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
            "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION": "2",
            "GPU_FAULT_PROCESS_METRICS_DIR": str(run_dir / "process-metrics"),
            "GPU_FAULT_INGRESS_WRITER_PROCESSES": str(args.uvicorn_workers),
            "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY": str(
                args.flush_slots * args.uvicorn_workers
            ),
            "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_FLOOR": str(
                (args.flush_floor or args.flush_slots) * args.uvicorn_workers
            ),
            "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY": str(
                max(1, args.flush_slots // 2) * args.uvicorn_workers
            ),
            "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY_FLOOR": str(
                max(1, (args.flush_floor or args.flush_slots) // 2)
                * args.uvicorn_workers
            ),
            "GPU_FAULT_INGRESS_DECODE_WORKERS": str(args.decode_workers),
            "GPU_FAULT_LIVENESS_PORT": str(args.liveness_port),
            "GPU_FAULT_GC_FREEZE_AFTER_STARTUP": "true" if args.gc_freeze else "false",
            **(
                {"GPU_FAULT_GC_GEN2_THRESHOLD": str(args.gc_threshold2)}
                if args.gc_threshold2
                else {}
            ),
            "GPU_FAULT_STORE_IO_WORKERS": str(args.store_io_workers),
            "AWS_REGION": "us-west-2",
            "AWS_DEFAULT_REGION": "us-west-2",
            ENV_PREFIX + "SAMPLE_DIR": str(run_dir / "samples"),
            ENV_PREFIX + "FLUSH_CPU_MS_PER_ITEM": str(args.flush_cpu_ms_per_item),
            ENV_PREFIX + "FLUSH_IO_MS": str(args.flush_io_ms),
            ENV_PREFIX + "GC_FREEZE": "1" if args.gc_freeze else "0",
            ENV_PREFIX + "GC_THRESHOLD2": str(args.gc_threshold2),
            ENV_PREFIX + "SWITCH_INTERVAL": str(args.switch_interval),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parent),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
        }
    )
    for name, value in args.env or []:
        env[name] = value
    return env


def _wait_ready(host: str, port: int, timeout: float) -> bool:
    import http.client

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            connection = http.client.HTTPConnection(host, port, timeout=2)
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def _taskset(cpus: str) -> list[str]:
    return ["taskset", "-c", cpus] if cpus else []


def _server_command(args: argparse.Namespace) -> list[str]:
    """The production flags, launched by uvicorn (one shared socket) or by
    ``gpu_fault.app.serve`` (one SO_REUSEPORT socket per worker)."""

    launcher = "uvicorn" if args.launcher == "uvicorn" else "gpu_fault.app.serve"
    flags = (
        f"--host {args.host} --port {args.port} --workers {args.uvicorn_workers} "
        "--loop uvloop --http httptools --backlog 8192 "
        f"--timeout-keep-alive {args.timeout_keep_alive} --no-proxy-headers "
        "--timeout-graceful-shutdown 60 --timeout-worker-healthcheck 20 "
        "--no-access-log --limit-concurrency 4096"
    )
    return [
        *_taskset(args.server_cpus),
        sys.executable,
        "-m",
        launcher,
        "ingress_event_loop_burst:create_app",
        "--factory",
        *flags.split(),
    ]


def _probe_command(args: argparse.Namespace, *, port: int, out: Path) -> list[str]:
    flags = (
        f"--host {args.host} --port {port} --queue-port {args.port} "
        f"--interval {args.probe_interval} --timeout {args.probe_timeout}"
    )
    prefix = [*_taskset(args.probe_cpus), sys.executable, __file__, "probe"]
    return [*prefix, *flags.split(), "--out", str(out)]


def _stop_process(process: subprocess.Popen[bytes], timeout: float) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()


def run_experiment(args: argparse.Namespace) -> int:
    label = args.label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    cluster_ids = [f"burst-cluster-{index:03d}" for index in range(args.clusters)]
    tokens = {cluster_id: secrets.token_hex(16) for cluster_id in cluster_ids}
    tokens_path = run_dir / "tokens.json"
    tokens_path.write_text(json.dumps(tokens), encoding="utf-8")
    tokens_path.chmod(0o600)
    clusters_json = json.dumps(
        [
            {
                "cluster_id": cluster_id,
                "region": "us-west-2",
                "hyperpod_cluster_name": f"hp-{cluster_id}",
                "eks_cluster_arn": (
                    f"arn:aws:eks:us-west-2:000000000000:cluster/{cluster_id}"
                ),
                "token": token,
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/8"],
            }
            for cluster_id, token in tokens.items()
        ]
    )
    env = server_environment(args, run_dir, clusters_json)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                name: value
                for name, value in vars(args).items()
                if name not in {"func", "env"}
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    # Four workers building the app at once race for the sqlite schema
    # ("database is locked" stops the launcher); create it once up front.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from gpu_fault.app import ApplicationContext; "
            "ApplicationContext.from_environment()",
        ],
        env=env,
        cwd=str(REPO),
        check=True,
        timeout=120,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    server_log = (run_dir / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        _server_command(args),
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        cwd=str(REPO),
    )
    probes: list[subprocess.Popen[bytes]] = []
    loads: list[subprocess.Popen[bytes]] = []
    try:
        if not _wait_ready(args.host, args.port, args.startup_timeout):
            print("server did not become ready; see", run_dir / "server.log")
            return 2
        time.sleep(args.warmup)
        probes.append(
            subprocess.Popen(
                _probe_command(
                    args,
                    port=args.probe_port or args.port,
                    out=run_dir / "probe.jsonl",
                )
            )
        )
        if args.probe_port and args.probe_port != args.port:
            # The service port's first-byte latency is the accept-queue wait
            # a client sees; the dedicated port is what the kubelet sees.
            probes.append(
                subprocess.Popen(
                    _probe_command(
                        args, port=args.port, out=run_dir / "probe-service.jsonl"
                    )
                )
            )
        start_at = time.time() + 5.0
        shards: list[list[str]] = [[] for _ in range(args.load_processes)]
        for index, cluster_id in enumerate(cluster_ids):
            shards[index % args.load_processes].append(cluster_id)
        for index, shard in enumerate(shards):
            if not shard:
                continue
            loads.append(
                subprocess.Popen(
                    [
                        *_taskset(args.load_cpus),
                        sys.executable,
                        __file__,
                        "load",
                        "--host",
                        args.host,
                        "--port",
                        str(args.port),
                        "--bind",
                        f"127.0.0.{index + 2}",
                        "--clusters",
                        ",".join(shard),
                        "--tokens-file",
                        str(tokens_path),
                        "--templates",
                        str(TEMPLATES),
                        "--workers-per-cluster",
                        str(args.workers_per_cluster),
                        "--nodes-per-cluster",
                        str(args.nodes_per_cluster),
                        "--faults-per-cluster",
                        str(args.faults_per_cluster),
                        "--timeout",
                        str(args.client_timeout),
                        "--payload-scale",
                        str(args.payload_scale),
                        "--start-at",
                        str(start_at),
                        "--out",
                        str(run_dir / f"load-{index}.json"),
                        *(["--keep-alive"] if args.keep_alive else []),
                    ]
                )
            )
        deadline = time.monotonic() + args.load_timeout
        for process in loads:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
        time.sleep(args.cooldown)
    finally:
        for probe in probes:
            _stop_process(probe, 15)
        _stop_process(server, 90)
        server_log.close()
    print(summarize(run_dir))
    return 0


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _longest_stall(probes: list[dict[str, Any]], threshold_ms: float) -> float:
    """Longest run of consecutive probes slower than the kubelet timeout, in s."""

    longest = current_start = 0.0
    previous_bad = False
    for probe in probes:
        bad = not isinstance(probe["status"], int) or probe["ms"] >= threshold_ms
        if bad and not previous_bad:
            current_start = probe["t"]
        if bad:
            longest = max(longest, probe["t"] + probe["ms"] / 1000 - current_start)
        previous_bad = bad
    return longest


def summarize(run_dir: Path) -> str:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    lines = [
        f"== {run_dir.name}: clusters={config['clusters']} "
        f"workers/cluster={config['workers_per_cluster']} "
        f"launcher={config.get('launcher', 'uvicorn')} "
        f"clients={'keep-alive' if config.get('keep_alive') else 'cold-connect'} "
        f"server-keep-alive={config.get('timeout_keep_alive', 5)} s "
        f"uvicorn-workers={config['uvicorn_workers']} "
        f"flush-slots={config['flush_slots']} "
        f"(floor {config['flush_floor'] or config['flush_slots']}) "
        f"decode={config['decode_workers']} store-io={config['store_io_workers']} "
        f"flush-cpu={config['flush_cpu_ms_per_item']} ms/item "
        f"flush-io={config['flush_io_ms']} ms gc-freeze={config['gc_freeze']} "
        f"gc-threshold2={config['gc_threshold2']} "
        f"switch-interval={config['switch_interval']}"
    ]
    loads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("load-*.json"))
    ]
    statuses: dict[str, int] = {}
    requests = 0
    started = min((load["started"] for load in loads), default=0.0)
    finished = max((load["finished"] for load in loads), default=0.0)
    for load in loads:
        requests += load["requests"]
        for key, count in load["statuses"].items():
            statuses[key] = statuses.get(key, 0) + count
    wall = max(1e-6, finished - started)
    if loads:
        p99 = max(load["p99_ms"] for load in loads)
        maximum = max(load["max_ms"] for load in loads)
        p50 = statistics.median(load["p50_ms"] for load in loads)
        connections = sum(load.get("connections", 0) for load in loads)
        reconnects = sum(load.get("reconnects", 0) for load in loads)
        lines.append(
            f"load: {requests} requests in {wall:.1f} s = {requests / wall:.0f} req/s; "
            f"statuses {json.dumps(statuses, sort_keys=True)}; "
            f"client p50 {p50:.0f} ms p99 {p99 / 1000:.2f} s max {maximum / 1000:.2f} s; "
            f"connections {connections} (reconnects {reconnects})"
        )
    for name, label in (
        ("probe.jsonl", "livez probe"),
        ("probe-service.jsonl", "service-port probe (accept-queue wait)"),
    ):
        probe_path = run_dir / name
        if not probe_path.exists():
            continue
        probes = _load_jsonl(probe_path)
        during = [probe for probe in probes if started <= probe["t"] <= finished + 5]
        if not during:
            continue
        timeout_ms = config["probe_timeout"] * 1000
        slow = [probe for probe in during if probe["ms"] >= timeout_ms]
        failed = [probe for probe in during if not isinstance(probe["status"], int)]
        stall = _longest_stall(during, timeout_ms)
        latencies = sorted(p["ms"] for p in during)
        p99_index = min(len(latencies) - 1, int((len(latencies) - 1) * 0.99))
        lines.append(
            f"{label} (1/s, {config['probe_timeout']} s timeout"
            f"{', dedicated port' if name == 'probe.jsonl' and config.get('probe_port') else ''}): "
            f"{len(during)} probes, max {latencies[-1] / 1000:.2f} s, "
            f"p99 {latencies[p99_index] / 1000:.2f} s, "
            f"p50 {statistics.median(latencies):.0f} ms, "
            f"{len(slow)} slower than the timeout, {len(failed)} errors, "
            f"longest stall {stall:.1f} s -> kubelet 5s/5s/2: "
            f"{'KILL' if stall >= 10 else 'survives'}; "
            f"10s/10s/3: {'KILL' if stall >= 30 else 'survives'}"
        )
        if name == "probe.jsonl":
            queue = [probe.get("accept_queue", 0) for probe in during]
            per_socket = [
                probe["accept_queues"]
                for probe in during
                if probe.get("accept_queues") and probe.get("accept_queue", 0) > 0
            ]
            lines.append(
                f"storm listener accept queue: max {max(queue)} / "
                f"median {statistics.median(queue):.0f} connections waiting "
                f"(established max {max(p.get('established', 0) for p in during)}); "
                f"{len(per_socket[0]) if per_socket else 1} listening socket(s)"
            )
            if per_socket and len(per_socket[0]) > 1:
                busiest = statistics.mean(max(q) for q in per_socket)
                quietest = statistics.mean(min(q) for q in per_socket)
                lines.append(
                    "per-socket accept queue while queued: mean busiest "
                    f"{busiest:.0f} / mean quietest {quietest:.0f} connections "
                    f"(ratio {busiest / max(quietest, 1):.2f})"
                )
    lines.append(
        "per process: lag=max event-loop lag (10 ms sampler) | gil=max GIL wait of an "
        "idle thread | gen2=max gen-2 GC pause / count | gc=total GC time | "
        "rss=max | run=max runnable threads | cpu=process cores avg (main thread "
        "share) | pend=max admission backlog | limit=max flush limit | ewma=max flush | "
        "fds=max open fds (accepted connections) | livez-app=max in-app /livez time"
    )
    served_by_pid: dict[str, int] = {}
    for path in sorted((run_dir / "samples").glob("proc-*.jsonl")):
        rows = _load_jsonl(path)
        start = next((row for row in rows if row.get("kind") == "start"), {})
        end = next((row for row in rows if row.get("kind") == "end"), {})
        samples = [
            row
            for row in rows
            if "kind" not in row and started - 1 <= row["t"] <= finished + 1
        ]
        if not samples:
            continue
        busy = [row for row in samples if row["served"] > 0] or samples
        cpu = statistics.mean(row["cpu_proc_cores"] for row in busy)
        main = statistics.mean(row["cpu_main_cores"] for row in busy)
        gen2_max = max(row["gc_max_ms"][2] for row in samples)
        gen2_count = sum(row["gc_count"][2] for row in samples)
        gc_total = sum(sum(row["gc_sum_ms"]) for row in samples)
        served_by_pid[path.stem.split("-")[1]] = sum(row["served"] for row in samples)
        lines.append(
            f"  pid {path.stem.split('-')[1]:>7}: "
            f"lag {max(row['loop_lag_max_ms'] for row in samples) / 1000:6.2f} s | "
            f"gil {max(row['gil_wait_max_ms'] for row in samples) / 1000:6.2f} s | "
            f"gen2 {gen2_max / 1000:5.2f} s /{gen2_count:3d} | "
            f"gc {gc_total / 1000:5.1f} s | "
            f"rss {max(row['rss_mb'] for row in samples):5.0f} MB | "
            f"run {max(row['threads_running'] for row in samples):3d}/"
            f"{max(row['threads'] for row in samples):3d} | "
            f"cpu {cpu:4.2f} ({main / max(cpu, 1e-9) * 100:3.0f}%) | "
            f"pend {max(row.get('adm_pending', 0) for row in samples):4d} | "
            f"limit {max(row.get('adm_limit', 0) for row in samples):2d} | "
            f"ewma {max(row.get('adm_ewma_ms', 0) for row in samples) / 1000:5.2f} s | "
            f"served {sum(row['served'] for row in samples):5d} | "
            f"fds {max(row.get('fds', 0) for row in samples):4d} | "
            f"livez-app {max(row.get('livez_app_max_ms', 0) for row in samples):5.0f} ms | "
            f"units {start.get('row_units')} | objects {end.get('tracked_objects')}"
        )
    if len(served_by_pid) > 1 and sum(served_by_pid.values()):
        total = sum(served_by_pid.values())
        shares = sorted(served_by_pid.values(), reverse=True)
        lines.append(
            "per-worker share of requests served: "
            + ", ".join(f"{count / total * 100:.1f}%" for count in shares)
            + f" (busiest/quietest {shares[0] / max(shares[-1], 1):.2f})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="drive one experiment: server, prober, load")
    run.add_argument("--label", default="")
    run.add_argument("--out", default="/tmp/gpu-fault-ingress-burst")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=18080)
    run.add_argument("--uvicorn-workers", type=int, default=4)
    run.add_argument(
        "--launcher",
        choices=("uvicorn", "serve"),
        default="serve",
        help="uvicorn: one shared listening socket; serve: one SO_REUSEPORT socket per worker",
    )
    run.add_argument("--timeout-keep-alive", type=int, default=5)
    run.add_argument(
        "--keep-alive",
        action="store_true",
        help="load clients hold one connection per worker",
    )
    run.add_argument("--clusters", type=int, default=50)
    run.add_argument("--workers-per-cluster", type=int, default=85)
    run.add_argument("--nodes-per-cluster", type=int, default=256)
    run.add_argument("--faults-per-cluster", type=int, default=62)
    run.add_argument("--load-processes", type=int, default=8)
    run.add_argument("--flush-slots", type=int, default=8)
    run.add_argument("--flush-floor", type=int, default=0)
    run.add_argument("--decode-workers", type=int, default=32)
    run.add_argument("--store-io-workers", type=int, default=28)
    run.add_argument("--flush-cpu-ms-per-item", type=float, default=8.0)
    run.add_argument("--flush-io-ms", type=float, default=0.0)
    run.add_argument("--gc-freeze", action="store_true")
    run.add_argument("--gc-threshold2", type=int, default=0)
    run.add_argument("--switch-interval", type=float, default=0.0)
    run.add_argument("--server-cpus", default="0-7")
    run.add_argument("--probe-cpus", default="8-9")
    run.add_argument("--load-cpus", default="16-63")
    run.add_argument(
        "--probe-port",
        type=int,
        default=0,
        help="probe this port instead of the storm port (dedicated liveness listener)",
    )
    run.add_argument(
        "--liveness-port",
        type=int,
        default=0,
        help="GPU_FAULT_LIVENESS_PORT for the server (0 = no dedicated listener)",
    )
    run.add_argument(
        "--payload-scale",
        type=int,
        default=1,
        help="repeat every template's longest list this many times",
    )
    run.add_argument("--probe-interval", type=float, default=1.0)
    run.add_argument("--probe-timeout", type=float, default=5.0)
    run.add_argument("--client-timeout", type=float, default=60.0)
    run.add_argument("--startup-timeout", type=float, default=120.0)
    run.add_argument("--warmup", type=float, default=3.0)
    run.add_argument("--cooldown", type=float, default=5.0)
    run.add_argument("--load-timeout", type=float, default=600.0)
    run.add_argument(
        "--env",
        nargs=2,
        action="append",
        metavar=("NAME", "VALUE"),
        help="extra server environment entry (repeatable)",
    )
    run.set_defaults(func=run_experiment)

    load = sub.add_parser("load", help="(internal) one cold-connect load process")
    load.add_argument("--host", default="127.0.0.1")
    load.add_argument("--port", type=int, required=True)
    load.add_argument("--bind", default="")
    load.add_argument("--clusters", required=True)
    load.add_argument("--tokens-file", required=True)
    load.add_argument("--templates", default=str(TEMPLATES))
    load.add_argument("--workers-per-cluster", type=int, default=85)
    load.add_argument("--nodes-per-cluster", type=int, default=256)
    load.add_argument("--faults-per-cluster", type=int, default=62)
    load.add_argument("--timeout", type=float, default=60.0)
    load.add_argument("--payload-scale", type=int, default=1)
    load.add_argument("--start-at", type=float, default=0.0)
    load.add_argument("--keep-alive", action="store_true")
    load.add_argument("--out", required=True)
    load.set_defaults(func=run_load)

    probe = sub.add_parser("probe", help="(internal) kubelet-like /livez prober")
    probe.add_argument("--host", default="127.0.0.1")
    probe.add_argument("--port", type=int, required=True)
    probe.add_argument("--path", default="/livez")
    probe.add_argument("--interval", type=float, default=1.0)
    probe.add_argument("--timeout", type=float, default=5.0)
    probe.add_argument(
        "--queue-port",
        type=int,
        default=0,
        help="listener whose accept queue is sampled (default: the probed port)",
    )
    probe.add_argument("--out", required=True)
    probe.set_defaults(func=run_probe)

    show = sub.add_parser("summarize", help="re-print a finished run")
    show.add_argument("run_dir")
    show.set_defaults(func=lambda args: print(summarize(Path(args.run_dir))) or 0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.func
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
