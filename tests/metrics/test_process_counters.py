"""The control-loop counters read the same on every process of a Pod (F-L1).

The control-worker runs several uvicorn processes; a counter kept as an
attribute is process-local and a scrape lands on an arbitrary process. Live
2026-09-08 (GF-REGIONAL-DESTR-018): the lease holder counted a lifetime
failure and every scrape of its Pod read 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event

from gpu_fault.app import ApplicationContext, create_app, process_counters
from tests._builders import asgi_client


def test_sharing_is_off_without_a_pod_identity_and_on_inside_a_pod() -> None:
    assert process_counters.counters_directory({}) is None
    assert process_counters.counters_directory({"POD_UID": "abc"}) == (
        process_counters.DEFAULT_ROOT / "abc"
    )
    assert process_counters.counters_directory(
        {"POD_UID": "abc", "GPU_FAULT_PROCESS_COUNTERS_DIR": "/tmp/x"}
    ) == Path("/tmp/x")
    for token in ("off", "OFF", "0", "none"):
        assert (
            process_counters.counters_directory(
                {"POD_UID": "abc", "GPU_FAULT_PROCESS_COUNTERS_DIR": token}
            )
            is None
        )


def _other_process() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_the_scrape_sums_live_processes_and_forgets_dead_ones(tmp_path: Path) -> None:
    live = _other_process()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    try:
        (tmp_path / f"{live.pid}.json").write_text(
            json.dumps({"lifetime_exceeded_total": 1, "only_theirs": 4})
        )
        (tmp_path / f"{dead.pid}.json").write_text(
            json.dumps({"lifetime_exceeded_total": 100})
        )
        (tmp_path / "not-a-pid.json").write_text("{}")
        (tmp_path / f"{live.pid + 1}.json").write_text("{not json")

        totals = process_counters.coherent_counters(
            {"lifetime_exceeded_total": 2, "dispatch_deferred_total": 0},
            directory=tmp_path,
        )

        assert totals == {
            "lifetime_exceeded_total": 3,
            "dispatch_deferred_total": 0,
            "only_theirs": 4,
        }
        assert not (tmp_path / f"{dead.pid}.json").exists(), (
            "a dead process's snapshot must not be counted after its restart"
        )
        published = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
        assert published == {"lifetime_exceeded_total": 2, "dispatch_deferred_total": 0}
    finally:
        live.kill()
        live.wait()


def test_an_unusable_directory_degrades_to_the_local_counters(tmp_path: Path) -> None:
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x")

    totals = process_counters.coherent_counters(
        {"lifetime_exceeded_total": 7}, directory=blocked / "inner"
    )

    assert totals == {"lifetime_exceeded_total": 7}


def test_the_publisher_writes_before_waiting_and_stops_on_the_event(
    tmp_path: Path,
) -> None:
    stop = Event()
    stop.set()
    values = iter([{"lifetime_exceeded_total": 5}])

    process_counters.publish_forever(lambda: next(values), stop, directory=tmp_path)

    assert json.loads((tmp_path / f"{os.getpid()}.json").read_text()) == {
        "lifetime_exceeded_total": 5
    }
    assert process_counters.publish_forever(lambda: {}, stop, directory=None) is None


def _scrape(app) -> str:
    import asyncio

    async def scenario():
        async with asgi_client(app) as client:
            return (await client.get("/metrics")).text

    return asyncio.run(scenario())


def test_metrics_read_the_whole_pod_not_the_answering_process(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-shared")
    monkeypatch.setenv("GPU_FAULT_PROCESS_COUNTERS_DIR", str(tmp_path))
    context = ApplicationContext(execution_token="processor-metrics-token-" + "z" * 32)
    app = create_app(context)
    # The sibling process that actually failed the workflow.
    sibling = _other_process()
    try:
        (tmp_path / f"{sibling.pid}.json").write_text(
            json.dumps(
                {
                    "lifetime_exceeded_total": 1,
                    "lifetime_record_only_total": 2,
                    "archive_withheld:incident has external successor": 3,
                }
            )
        )
        context.workflow_executor.lifetime_exceeded_total = 1

        text = _scrape(app)

        for line in (
            "gpu_fault_workflow_lifetime_exceeded_total 2",
            'gpu_fault_workflow_merge_record_only_total{reason="lifetime_exceeded"} 2',
            'gpu_fault_control_record_archive_withheld_total{reason="incident has external successor"} 3',
        ):
            assert line in text, line
        assert (tmp_path / f"{os.getpid()}.json").is_file(), (
            "the answering process must publish its own share for the others"
        )
    finally:
        sibling.kill()
        sibling.wait()


def test_the_snapshot_reads_every_counter_the_renderer_needs() -> None:
    context = ApplicationContext(execution_token="processor-metrics-token-" + "w" * 32)
    context.dispatcher.node_busy_timeouts_total = 2
    context.workflow_executor.lifetime_exceeded_total = 5

    snapshot = process_counters.control_loop_counter_snapshot(context)

    assert snapshot["dispatch_node_busy_timeouts_total"] == 2
    assert snapshot["lifetime_exceeded_total"] == 5
    assert snapshot["withdrawn_record_only_total"] == 0
    assert all(isinstance(value, int) for value in snapshot.values())
    assert process_counters.control_loop_counter_snapshot(object()) == {
        key: 0 for key, _, _ in process_counters.COUNTER_SOURCES
    }
