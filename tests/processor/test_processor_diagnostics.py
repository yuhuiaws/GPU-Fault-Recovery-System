import asyncio

import pytest

from gpu_fault.async_store import AsyncStoreExecutor
from gpu_fault.processor_diagnostics import (
    ProcessorDiagnosticsPublisher,
    ProcessorReplayTracker,
    bind_processor_replay,
    process_runtime_snapshot,
    register_thread_dump_signal,
    report_processor_replay_phase,
    reset_processor_replay,
)


def test_replay_phase_context_reaches_store_io_thread() -> None:
    tracker = ProcessorReplayTracker()
    tracker.start(
        "request-a",
        owner_id="pod-a:1",
        path="/v1/collector-events/nvidia-kernel",
        lane_epoch=3,
        phase="handler_dispatch",
    )
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )

    async def scenario() -> None:
        token = bind_processor_replay(tracker, "request-a")
        try:
            await executor.run(report_processor_replay_phase, "kernel_evidence")
        finally:
            reset_processor_replay(token)

    try:
        asyncio.run(scenario())
    finally:
        executor.close()

    snapshot = tracker.snapshot()
    assert snapshot[0]["phase"] == "kernel_evidence"
    assert snapshot[0]["thread_name"].startswith("gpu-fault-store-io")


def test_process_runtime_snapshot_is_numeric() -> None:
    snapshot = process_runtime_snapshot()

    assert snapshot["pid"] > 0
    assert snapshot["vm_rss_bytes"] > 0
    assert snapshot["threads"] >= 1
    assert len(snapshot["gc_counts"]) == 3


def test_diagnostics_publisher_aggregates_process_files(tmp_path) -> None:
    publisher = ProcessorDiagnosticsPublisher(
        str(tmp_path),
        lambda: {
            "process": {"pid": 42},
            "in_flight_requests": [{"request_id": "request-a"}],
            "inbound_replay_requests": [],
        },
    )

    publisher.publish()

    documents = publisher.read_all()
    assert documents[0]["process"]["pid"] == 42
    assert documents[0]["in_flight_requests"][0]["request_id"] == "request-a"
    assert documents[0]["snapshot_age_seconds"] >= 0


def test_thread_dump_signal_rejects_unrelated_signal() -> None:
    with pytest.raises(ValueError):
        register_thread_dump_signal("SIGTERM")
