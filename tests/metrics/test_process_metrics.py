"""/metrics reads the same on every process of a multi-process Pod.

The control-worker runs several uvicorn processes; every in-memory family is
process-local and a scrape lands on an arbitrary process. Live 2026-09-08
(GF-REGIONAL-DESTR-018): the lease holder counted a lifetime failure and every
scrape of its Pod read 0. Each process now publishes its full render to a
shared directory and the answering process merges the live ones by the
strategy registered per family.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event

import pytest

from gpu_fault.app import ApplicationContext, create_app, process_metrics
from gpu_fault.app.metric_aggregation import (
    DEGRADED_METRIC,
    PROCESSES_METRIC,
    STRATEGIES,
    Strategy,
)
from tests._builders import asgi_client

SUM_FAMILY = "gpu_fault_workflow_lifetime_exceeded_total"
MAX_FAMILY = "gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds"
MIN_FAMILY = "gpu_fault_processor_healthy"
ANY_FAMILY = "gpu_fault_processor_queue_depth"
PER_PROCESS_FAMILY = "gpu_fault_processor_notification_shard"
SUMMARY_FAMILY = "gpu_fault_store_io_admission_wait_seconds"


def _lines(**values: str) -> list[str]:
    families = {
        SUM_FAMILY: "counter",
        MAX_FAMILY: "gauge",
        MIN_FAMILY: "gauge",
        ANY_FAMILY: "gauge",
        PER_PROCESS_FAMILY: "gauge",
    }
    out: list[str] = []
    for family, kind in families.items():
        out.append(f"# HELP {family} help for {family}.")
        out.append(f"# TYPE {family} {kind}")
        out.append(f"{family} {values[family]}")
    return out


def _rendered(slot: int, pid: int, **values: str) -> process_metrics.Rendered:
    rendered = process_metrics.parse_lines(_lines(**values))
    rendered.slot = slot
    rendered.pid = pid
    return rendered


def _value(lines: list[str], sample: str) -> str:
    matches = [line for line in lines if line.startswith(sample + " ")]
    assert len(matches) == 1, (sample, matches)
    return matches[0].split(" ", 1)[1]


def test_the_five_strategies_combine_two_processes_correctly() -> None:
    for family, strategy in (
        (SUM_FAMILY, Strategy.SUM),
        (MAX_FAMILY, Strategy.MAX),
        (MIN_FAMILY, Strategy.MIN),
        (ANY_FAMILY, Strategy.ANY),
        (PER_PROCESS_FAMILY, Strategy.PER_PROCESS),
    ):
        assert STRATEGIES[family] is strategy, family
    local = _rendered(
        2,
        100,
        **{
            SUM_FAMILY: "3",
            MAX_FAMILY: "1700000100.000",
            MIN_FAMILY: "1",
            ANY_FAMILY: "40",
            PER_PROCESS_FAMILY: "5",
        },
    )
    other = _rendered(
        0,
        200,
        **{
            SUM_FAMILY: "4",
            MAX_FAMILY: "1700000200.000",
            MIN_FAMILY: "0",
            ANY_FAMILY: "41",
            PER_PROCESS_FAMILY: "1",
        },
    )

    lines = process_metrics.aggregate(local, [other])

    assert _value(lines, SUM_FAMILY) == "7"
    assert _value(lines, MAX_FAMILY) == "1700000200.000"
    assert _value(lines, MIN_FAMILY) == "0"
    # ANY prefers the answering process's fresh render over a sibling's file.
    assert _value(lines, ANY_FAMILY) == "40"
    assert f'{PER_PROCESS_FAMILY}{{process="2"}} 5' in lines
    assert f'{PER_PROCESS_FAMILY}{{process="0"}} 1' in lines
    assert lines.count(f"# TYPE {SUM_FAMILY} counter") == 1, "one header per family"


def test_a_single_source_is_emitted_verbatim() -> None:
    """Sharing off, or a lone process: the render must not change shape."""

    source = [
        "# HELP gpu_fault_processor_healthy Whether this replica can claim.",
        "# TYPE gpu_fault_processor_healthy gauge",
        "gpu_fault_processor_healthy 1",
        "# HELP gpu_fault_store_io_admission_wait_seconds Time waiting.",
        "# TYPE gpu_fault_store_io_admission_wait_seconds summary",
        "gpu_fault_store_io_admission_wait_seconds_sum 0.250000",
        "gpu_fault_store_io_admission_wait_seconds_count 3",
        "gpu_fault_store_io_admission_wait_seconds_max 0.100000",
        'gpu_fault_store_io_rejections_total{reason="capacity"} 2',
    ]

    lines = process_metrics.pod_coherent_lines(source, environ={})

    assert lines[: len(source)] == source
    assert f"{PROCESSES_METRIC} 1" in lines
    assert f"{DEGRADED_METRIC} 0" in lines


def test_summaries_add_their_sums_and_counts_but_take_the_largest_max() -> None:
    local = process_metrics.parse_lines(
        [
            f"# TYPE {SUMMARY_FAMILY} summary",
            f"{SUMMARY_FAMILY}_sum 0.250000",
            f"{SUMMARY_FAMILY}_count 3",
            f"{SUMMARY_FAMILY}_max 0.100000",
            'gpu_fault_processor_request_processing_seconds_bucket{le="1"} 4',
            'gpu_fault_processor_request_processing_seconds_bucket{le="+Inf"} 6',
        ]
    )
    other = process_metrics.parse_lines(
        [
            f"# TYPE {SUMMARY_FAMILY} summary",
            f"{SUMMARY_FAMILY}_sum 0.500000",
            f"{SUMMARY_FAMILY}_count 1",
            f"{SUMMARY_FAMILY}_max 0.400000",
            'gpu_fault_processor_request_processing_seconds_bucket{le="1"} 1',
            'gpu_fault_processor_request_processing_seconds_bucket{le="+Inf"} 1',
        ]
    )
    local.slot, other.slot = 0, 1

    lines = process_metrics.aggregate(local, [other])

    assert _value(lines, f"{SUMMARY_FAMILY}_sum") == "0.750000"
    assert _value(lines, f"{SUMMARY_FAMILY}_count") == "4"
    assert _value(lines, f"{SUMMARY_FAMILY}_max") == "0.400000"
    assert 'gpu_fault_processor_request_processing_seconds_bucket{le="1"} 5' in lines
    assert 'gpu_fault_processor_request_processing_seconds_bucket{le="+Inf"} 7' in lines


def test_labelled_series_only_one_process_has_still_reach_the_scrape() -> None:
    local = process_metrics.parse_lines(
        ["# TYPE gpu_fault_control_record_archive_withheld_total counter"]
    )
    other = process_metrics.parse_lines(
        [
            "# TYPE gpu_fault_control_record_archive_withheld_total counter",
            'gpu_fault_control_record_archive_withheld_total{reason="open workflow"} 3',
            "# HELP gpu_fault_only_theirs_total Only the sibling has this family.",
            "# TYPE gpu_fault_only_theirs_total counter",
            "gpu_fault_only_theirs_total 9",
        ]
    )
    local.slot, other.slot = 0, 1

    lines = process_metrics.aggregate(local, [other])

    assert (
        'gpu_fault_control_record_archive_withheld_total{reason="open workflow"} 3'
        in lines
    )
    assert "gpu_fault_only_theirs_total 9" in lines
    assert (
        "# HELP gpu_fault_only_theirs_total Only the sibling has this family." in lines
    )


def test_sharing_is_off_without_a_pod_identity_and_on_inside_a_pod() -> None:
    assert process_metrics.metrics_directory({}) is None
    assert process_metrics.metrics_directory({"POD_UID": "abc"}) == (
        process_metrics.DEFAULT_ROOT / "abc"
    )
    assert process_metrics.metrics_directory(
        {"POD_UID": "abc", "GPU_FAULT_PROCESS_METRICS_DIR": "/tmp/x"}
    ) == Path("/tmp/x")
    for token in ("off", "OFF", "0", "none"):
        assert (
            process_metrics.metrics_directory(
                {"POD_UID": "abc", "GPU_FAULT_PROCESS_METRICS_DIR": token}
            )
            is None
        )


def _other_process() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _publish_as(directory: Path, pid: int, slot: int, lines: list[str]) -> None:
    rendered = process_metrics.parse_lines(lines)
    process_metrics.publish(directory, rendered, pid=pid, slot=slot)


def test_the_scrape_merges_live_processes_and_forgets_dead_ones(tmp_path: Path) -> None:
    live = _other_process()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    try:
        _publish_as(
            tmp_path, live.pid, 1, [f"# TYPE {SUM_FAMILY} counter", f"{SUM_FAMILY} 1"]
        )
        _publish_as(
            tmp_path, dead.pid, 2, [f"# TYPE {SUM_FAMILY} counter", f"{SUM_FAMILY} 100"]
        )
        (tmp_path / "not-a-pid.json").write_text("{}")
        (tmp_path / f"{live.pid + 1}.json").write_text("{not json")

        lines = process_metrics.pod_coherent_lines(
            [f"# TYPE {SUM_FAMILY} counter", f"{SUM_FAMILY} 2"], directory=tmp_path
        )

        assert _value(lines, SUM_FAMILY) == "3"
        assert f"{PROCESSES_METRIC} 2" in lines
        assert f"{DEGRADED_METRIC} 0" in lines
        assert not (tmp_path / f"{dead.pid}.json").exists(), (
            "a dead process's samples must not be counted after its restart"
        )
        own = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
        assert own["pid"] == os.getpid()
        assert own["samples"] == [[SUM_FAMILY, [], "2"]]
    finally:
        live.kill()
        live.wait()
        process_metrics.SLOTS.release()


def test_an_unusable_directory_degrades_to_the_local_render(tmp_path: Path) -> None:
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x")

    lines = process_metrics.pod_coherent_lines(
        [f"# TYPE {SUM_FAMILY} counter", f"{SUM_FAMILY} 7"], directory=blocked / "inner"
    )

    assert _value(lines, SUM_FAMILY) == "7"
    assert f"{DEGRADED_METRIC} 1" in lines
    assert f"{PROCESSES_METRIC} 1" in lines


def test_slots_are_stable_and_freed_by_a_dying_process(tmp_path: Path) -> None:
    registry = process_metrics.SlotRegistry()
    try:
        first = registry.slot(tmp_path)
        assert first == 0
        assert registry.slot(tmp_path) == 0, "a process keeps its slot"
        # Another process holding slot 0 forces a newcomer onto slot 1.
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import fcntl,sys,time;"
                f"h=open({str(tmp_path / 'slot-0.lock')!r},'a+');"
                "fcntl.flock(h.fileno(), fcntl.LOCK_EX);"
                "print('held', flush=True); time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
        )
        try:
            registry.release()
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == b"held"
            assert registry.slot(tmp_path) == 1
            holder.kill()
            holder.wait()
            registry.release()
            assert registry.slot(tmp_path) == 0, "a dead holder's slot is reclaimed"
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
    finally:
        registry.release()


def test_the_publisher_writes_before_waiting_and_stops_on_the_event(
    tmp_path: Path,
) -> None:
    stop = Event()
    stop.set()
    renders = iter([[f"# TYPE {SUM_FAMILY} counter", f"{SUM_FAMILY} 5"]])
    try:
        process_metrics.publish_forever(lambda: next(renders), stop, directory=tmp_path)
    finally:
        process_metrics.SLOTS.release()

    published = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
    assert published["samples"] == [[SUM_FAMILY, [], "5"]]
    assert published["slot"] == 0
    assert process_metrics.publish_forever(lambda: [], stop, directory=None) is None


def _scrape(app) -> str:
    async def scenario():
        async with asgi_client(app) as client:
            return (await client.get("/metrics")).text

    return asyncio.run(scenario())


def test_metrics_read_the_whole_pod_not_the_answering_process(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-shared")
    monkeypatch.setenv("GPU_FAULT_PROCESS_METRICS_DIR", str(tmp_path))
    context = ApplicationContext(execution_token="processor-metrics-token-" + "z" * 32)
    app = create_app(context)
    # The sibling process that actually failed the workflow.
    sibling = _other_process()
    try:
        _publish_as(
            tmp_path,
            sibling.pid,
            3,
            [
                "# TYPE gpu_fault_workflow_lifetime_exceeded_total counter",
                "gpu_fault_workflow_lifetime_exceeded_total 1",
                "# TYPE gpu_fault_workflow_merge_record_only_total counter",
                'gpu_fault_workflow_merge_record_only_total{reason="lifetime_exceeded"} 2',
                "# TYPE gpu_fault_control_record_archive_withheld_total counter",
                'gpu_fault_control_record_archive_withheld_total{reason="incident has external successor"} 3',
                "# TYPE gpu_fault_processor_healthy gauge",
                "gpu_fault_processor_healthy 0",
                "# TYPE gpu_fault_processor_notification_shard gauge",
                "gpu_fault_processor_notification_shard 7",
            ],
        )
        context.workflow_executor.lifetime_exceeded_total = 1

        text = _scrape(app)

        for line in (
            "gpu_fault_workflow_lifetime_exceeded_total 2",
            'gpu_fault_workflow_merge_record_only_total{reason="lifetime_exceeded"} 2',
            'gpu_fault_control_record_archive_withheld_total{reason="incident has external successor"} 3',
            "gpu_fault_processor_healthy 0",
            'gpu_fault_processor_notification_shard{process="3"} 7',
            f"{PROCESSES_METRIC} 2",
            f"{DEGRADED_METRIC} 0",
        ):
            assert line in text, line
        assert (tmp_path / f"{os.getpid()}.json").is_file(), (
            "the answering process must publish its own share for the others"
        )
        assert (tmp_path / "slot-0.lock").exists(), "the first process must hold slot 0"
    finally:
        sibling.kill()
        sibling.wait()
        process_metrics.SLOTS.release()


@pytest.mark.parametrize(
    ("line", "name", "labels", "value"),
    [
        ("gpu_fault_x 1", "gpu_fault_x", (), "1"),
        (
            'gpu_fault_x{a="1",b="two words"} 2.5',
            "gpu_fault_x",
            (("a", "1"), ("b", "two words")),
            "2.5",
        ),
        (
            'gpu_fault_x{reason="quote \\" and brace }"} +Inf',
            "gpu_fault_x",
            (("reason", 'quote \\" and brace }'),),
            "+Inf",
        ),
    ],
)
def test_sample_lines_round_trip_through_the_parser(line, name, labels, value) -> None:
    parsed = process_metrics.parse_lines([line])
    (sample,) = parsed.samples[name]
    assert (sample.name, sample.labels, sample.value) == (name, labels, value)
    assert sample.line() == line
