"""Process-level health facts for ``/metrics`` and the start-up GC policy.

When the 50-cluster cold-connect burst had every api-ha Pod killed by its
liveness probe, nothing recorded why: the app exposed no process memory, no
garbage-collection timing and nothing about the listening socket, and the
per-process counters it did have died with the processes. The local
reproduction (``scripts/perf/ingress_event_loop_burst.py``) found the three
facts that would have named the mechanism on the spot, so they are exported:

* the **accept queue** of the listening socket the Pod's uvicorn processes
  share. 4,250 cold connections against four processes held 3,600-3,900
  connections in the kernel's accept queue for the whole burst while the
  processes themselves held 550-740 each; the kubelet's probe connection
  joins that FIFO, so its latency was the queue depth over the accept rate
  (3.6 s at 1,090 req/s) while the request itself took under 0.11 s in-app;
* **garbage-collection pauses**. Every gen-2 collection stops the event loop
  for 0.15-0.6 s and was the longest single loop stall in every run (42-47
  of them in a 38 s burst, 8-9 s of GC time per process). ``gc.freeze()``
  after start-up took 26 % off that total without lengthening the pauses;
  raising the gen-2 threshold to 100 cut the collections to 4-5 but made
  each pause 2.5x longer - the wrong trade for an event loop, so the
  threshold stays at CPython's default unless a site sets it;
* **resident memory**, so a growing heap is visible before it is an OOMKill.

``GPU_FAULT_GC_FREEZE_AFTER_STARTUP`` (default off; the renderer sets it for
every control-plane role) moves the objects alive at the end of lifespan
start-up - routers, registries, pydantic schemas, ~100k objects - into the
permanent generation, where no later collection traverses them.
``GPU_FAULT_GC_GEN2_THRESHOLD`` (default CPython's 10) is exposed for an A/B,
with the measured trade-off above.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gpu_fault.env import env_bool

LOGGER = logging.getLogger(__name__)

GC_FREEZE_ENV = "GPU_FAULT_GC_FREEZE_AFTER_STARTUP"
GC_GEN2_THRESHOLD_ENV = "GPU_FAULT_GC_GEN2_THRESHOLD"
GENERATIONS = (0, 1, 2)
_TCP_LISTEN = "0A"
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


class GcTimer:
    """Collections and their durations per generation, from ``gc.callbacks``.

    One instance per process; ``install`` is idempotent so a test process
    that builds many apps does not stack callbacks. The callback runs in the
    thread that triggered the collection, under the GIL, so the bookkeeping
    needs no lock beyond the one that guards the snapshot.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = 0.0
        self.collections = [0, 0, 0]
        self.seconds = [0.0, 0.0, 0.0]
        self.max_seconds = [0.0, 0.0, 0.0]
        self.collected = [0, 0, 0]

    @property
    def installed(self) -> bool:
        return self._callback in gc.callbacks

    def install(self) -> None:
        if not self.installed:
            gc.callbacks.append(self._callback)

    def uninstall(self) -> None:
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)

    def _callback(self, phase: str, info: Mapping[str, Any]) -> None:
        if phase == "start":
            self._started = time.perf_counter()
            return
        elapsed = time.perf_counter() - self._started
        generation = int(info.get("generation", 0))
        if generation not in GENERATIONS:
            return
        with self._lock:
            self.collections[generation] += 1
            self.seconds[generation] += elapsed
            if elapsed > self.max_seconds[generation]:
                self.max_seconds[generation] = elapsed
            self.collected[generation] += int(info.get("collected", 0))

    def snapshot(self) -> dict[str, list[Any]]:
        with self._lock:
            return {
                "collections": list(self.collections),
                "seconds": list(self.seconds),
                "max_seconds": list(self.max_seconds),
                "collected": list(self.collected),
            }


_GC_TIMER = GcTimer()


def gc_timer() -> GcTimer:
    """The process's collection timer, installed on first use."""

    _GC_TIMER.install()
    return _GC_TIMER


def resident_memory_bytes() -> int:
    """This process's resident set size from ``/proc/self/statm``."""

    try:
        with open("/proc/self/statm", "rb") as handle:
            return int(handle.read().split()[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return 0


def _socket_inodes() -> set[str]:
    inodes: set[str] = set()
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return inodes
    for name in names:
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[8:-1])
    return inodes


def listen_accept_queues() -> dict[int, int]:
    """Port -> completed connections waiting in the accept queue.

    Only the listening sockets this process holds are reported - under
    uvicorn ``--workers`` that is the socket every worker in the Pod shares,
    so the Pod-level value is the maximum over processes, not the sum.
    ``/proc/net/tcp`` reports a LISTEN socket's ``rx_queue`` as
    ``sk_ack_backlog``: the connections the kernel finished the handshake
    for that no process has accept()ed yet. That is the queue a kubelet probe
    joins behind a cold-connect storm.
    """

    inodes = _socket_inodes()
    depths: dict[int, int] = {}
    if not inodes:
        return depths
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, encoding="ascii") as handle:
                next(handle, None)
                for line in handle:
                    fields = line.split()
                    if (
                        len(fields) < 10
                        or fields[3] != _TCP_LISTEN
                        or fields[9] not in inodes
                    ):
                        continue
                    port = int(fields[1].rsplit(":", 1)[1], 16)
                    depth = int(fields[4].split(":", 1)[1], 16)
                    depths[port] = depths.get(port, 0) + depth
        except (OSError, IndexError, ValueError):
            continue
    return depths


def somaxconn() -> int | None:
    """The network namespace's accept-queue cap, which bounds ``--backlog``.

    A Pod gets a namespace of its own with the kernel default (4096 since
    Linux 5.4), whatever the node's value is: ``uvicorn --backlog 8192`` is
    silently clamped to it, and once the queue is full new SYNs are dropped
    and the clients - the kubelet included - retransmit after one second.
    """

    try:
        with open("/proc/sys/net/core/somaxconn", encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class GcPolicy:
    freeze_after_startup: bool
    gen2_threshold: int

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> GcPolicy:
        env = os.environ if environ is None else environ
        # "10" is CPython's own gen-2 threshold; see the module docstring for
        # why a higher value is not the default.
        gen2 = int(env.get("GPU_FAULT_GC_GEN2_THRESHOLD", "10"))
        if gen2 < 0:
            raise ValueError(f"{GC_GEN2_THRESHOLD_ENV} cannot be negative")
        return cls(
            freeze_after_startup=env_bool(GC_FREEZE_ENV, False, environ=env),
            gen2_threshold=gen2,
        )

    def apply(self) -> dict[str, Any]:
        """Set the gen-2 threshold and freeze the start-up heap; report it."""

        gen0, gen1, gen2 = gc.get_threshold()
        if self.gen2_threshold != gen2:
            gc.set_threshold(gen0, gen1, self.gen2_threshold)
        frozen = 0
        if self.freeze_after_startup:
            gc.collect()
            gc.freeze()
            frozen = gc.get_freeze_count()
        return {
            "frozen_objects": frozen,
            "threshold": gc.get_threshold(),
        }


def apply_startup_gc_policy(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Called once the lifespan has started everything the process keeps."""

    gc_timer()
    policy = GcPolicy.from_environment(environ)
    applied = policy.apply()
    if policy.freeze_after_startup or policy.gen2_threshold != 10:
        LOGGER.info(
            "gc policy applied pid=%s frozen_objects=%s threshold=%s",
            os.getpid(),
            applied["frozen_objects"],
            applied["threshold"],
        )
    return applied


def render_process_health_metrics(lines: list[str]) -> None:
    """Append the process-health families to a ``/metrics`` render."""

    timer = gc_timer().snapshot()
    lines.extend(
        [
            "# HELP gpu_fault_process_resident_memory_bytes Resident set size of "
            "this uvicorn process.",
            "# TYPE gpu_fault_process_resident_memory_bytes gauge",
            f"gpu_fault_process_resident_memory_bytes {resident_memory_bytes()}",
            "# HELP gpu_fault_gc_collections_total Garbage collections completed "
            "in this process, by generation.",
            "# TYPE gpu_fault_gc_collections_total counter",
        ]
    )
    for generation in GENERATIONS:
        lines.append(
            f'gpu_fault_gc_collections_total{{generation="{generation}"}} '
            f"{timer['collections'][generation]}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_gc_collection_seconds_total Time this process spent "
            "inside garbage collections - the event loop is stopped for every "
            "one of them - by generation.",
            "# TYPE gpu_fault_gc_collection_seconds_total counter",
        ]
    )
    for generation in GENERATIONS:
        lines.append(
            f'gpu_fault_gc_collection_seconds_total{{generation="{generation}"}} '
            f"{timer['seconds'][generation]:.6f}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_gc_collection_seconds_max Longest single garbage "
            "collection since the process started, by generation.",
            "# TYPE gpu_fault_gc_collection_seconds_max gauge",
        ]
    )
    for generation in GENERATIONS:
        lines.append(
            f'gpu_fault_gc_collection_seconds_max{{generation="{generation}"}} '
            f"{timer['max_seconds'][generation]:.6f}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_gc_frozen_objects Objects moved to the permanent "
            "generation at start-up (GPU_FAULT_GC_FREEZE_AFTER_STARTUP).",
            "# TYPE gpu_fault_gc_frozen_objects gauge",
            f"gpu_fault_gc_frozen_objects {gc.get_freeze_count()}",
            "# HELP gpu_fault_listen_accept_queue_connections Completed connections "
            "waiting to be accepted on a listening port of this process; the "
            "Pod's uvicorn processes share the socket, so read the maximum.",
            "# TYPE gpu_fault_listen_accept_queue_connections gauge",
        ]
    )
    for port, depth in sorted(listen_accept_queues().items()):
        lines.append(
            f'gpu_fault_listen_accept_queue_connections{{port="{port}"}} {depth}'
        )
    cap = somaxconn()
    if cap is not None:
        lines.extend(
            [
                "# HELP gpu_fault_listen_somaxconn The network namespace's accept-"
                "queue cap (net.core.somaxconn), which bounds uvicorn --backlog.",
                "# TYPE gpu_fault_listen_somaxconn gauge",
                f"gpu_fault_listen_somaxconn {cap}",
            ]
        )
