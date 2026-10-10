"""Process-health facts and the start-up GC policy (``app.process_health``).

The three facts the 50-cluster liveness kill lacked: the listening socket's
accept queue (where the kubelet's probe waited), GC pause timing (the
longest single event-loop stall) and resident memory. ``gc.freeze()`` is the
one GC change that measured as a pure win; the gen-2 threshold stays at
CPython's default because raising it lengthened every pause.
"""

from __future__ import annotations

import gc
import socket

import pytest

from gpu_fault.app import process_health
from gpu_fault.app.process_health import GcPolicy, GcTimer


def test_the_gc_timer_counts_and_times_collections_once_installed() -> None:
    timer = GcTimer()
    assert not timer.installed, "a fresh timer is not in gc.callbacks"
    timer.install()
    callbacks = len(gc.callbacks)
    timer.install()
    assert timer.installed and len(gc.callbacks) == callbacks, "idempotent install"
    try:
        before = timer.snapshot()
        gc.collect()
        after = timer.snapshot()
        assert after["collections"][2] == before["collections"][2] + 1
        assert after["seconds"][2] > before["seconds"][2]
        assert after["max_seconds"][2] >= after["seconds"][2] - before["seconds"][2]
        assert after["collections"][0] == before["collections"][0], "gen 2 only"
    finally:
        timer.uninstall()
    assert not timer.installed, "uninstall removes the callback"


def test_the_module_timer_is_one_per_process() -> None:
    timer = process_health.gc_timer()
    assert timer is process_health.gc_timer() and timer.installed
    callbacks = len(gc.callbacks)
    process_health.gc_timer()
    assert len(gc.callbacks) == callbacks


def test_resident_memory_and_somaxconn_are_read_from_proc() -> None:
    assert process_health.resident_memory_bytes() > 10 * 1024 * 1024
    cap = process_health.somaxconn()
    assert cap is None or cap > 0


def test_the_accept_queue_counts_connections_nobody_accepted_yet() -> None:
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    port = int(server.getsockname()[1])
    clients = [socket.create_connection(("127.0.0.1", port)) for _ in range(3)]
    try:
        assert process_health.listen_accept_queues()[port] == 3
        accepted, _ = server.accept()
        accepted.close()
        assert process_health.listen_accept_queues()[port] == 2
    finally:
        for client in clients:
            client.close()
        server.close()
    assert port not in process_health.listen_accept_queues(), "closed listener"


def test_the_gc_policy_reads_its_two_settings_and_validates_them() -> None:
    default = GcPolicy.from_environment({})
    assert default == GcPolicy(freeze_after_startup=False, gen2_threshold=10)
    tuned = GcPolicy.from_environment(
        {
            "GPU_FAULT_GC_FREEZE_AFTER_STARTUP": "true",
            "GPU_FAULT_GC_GEN2_THRESHOLD": "25",
        }
    )
    assert tuned == GcPolicy(freeze_after_startup=True, gen2_threshold=25)
    with pytest.raises(ValueError, match="cannot be negative"):
        GcPolicy.from_environment({"GPU_FAULT_GC_GEN2_THRESHOLD": "-1"})
    with pytest.raises(ValueError):
        GcPolicy.from_environment({"GPU_FAULT_GC_FREEZE_AFTER_STARTUP": "maybe"})


def test_applying_the_policy_sets_the_threshold_and_freezes_the_heap() -> None:
    gen0, gen1, gen2 = gc.get_threshold()
    frozen_before = gc.get_freeze_count()
    try:
        applied = GcPolicy(freeze_after_startup=True, gen2_threshold=gen2 + 7).apply()
        assert gc.get_threshold() == (gen0, gen1, gen2 + 7)
        assert applied["threshold"] == (gen0, gen1, gen2 + 7)
        assert applied["frozen_objects"] > frozen_before
        # Objects allocated after the freeze (the report itself) are not in it.
        assert frozen_before < gc.get_freeze_count() <= applied["frozen_objects"] + 8
        # The default policy is a no-op: nothing frozen, threshold untouched.
        gc.set_threshold(gen0, gen1, gen2)
        gc.unfreeze()
        applied = GcPolicy(freeze_after_startup=False, gen2_threshold=gen2).apply()
        assert applied == {"frozen_objects": 0, "threshold": (gen0, gen1, gen2)}
    finally:
        gc.set_threshold(gen0, gen1, gen2)
        gc.unfreeze()


def test_the_metrics_render_names_every_process_health_family() -> None:
    lines: list[str] = []
    process_health.render_process_health_metrics(lines)
    families = {line.split()[2] for line in lines if line.startswith("# TYPE ")}
    assert {
        "gpu_fault_process_resident_memory_bytes",
        "gpu_fault_gc_collections_total",
        "gpu_fault_gc_collection_seconds_total",
        "gpu_fault_gc_collection_seconds_max",
        "gpu_fault_gc_frozen_objects",
        "gpu_fault_listen_accept_queue_connections",
    } <= families
    samples = [line for line in lines if not line.startswith("#")]
    assert any(
        line.startswith("gpu_fault_process_resident_memory_bytes ")
        and int(line.split()[1]) > 0
        for line in samples
    ), "RSS is rendered as a positive byte count"
    for generation in ("0", "1", "2"):
        assert any(
            line.startswith(
                f'gpu_fault_gc_collections_total{{generation="{generation}"}} '
            )
            for line in samples
        ), f"generation {generation} has a collections counter"
