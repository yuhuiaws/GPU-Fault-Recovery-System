"""``gpu_fault_event_loop_lag_seconds`` is a histogram, not just a running max.

A high-water mark since process start cannot tell one garbage collection an
hour ago from a loop that stalls every second, and the kill the 50-cluster
burst produced was argued about for exactly that reason. The buckets resolve
ordinary scheduling, a gen-2 collection (0.15-0.6 s) and the stall a liveness
probe would die on.
"""

from __future__ import annotations

import asyncio

from gpu_fault.app import ApplicationContext, create_app, process_metrics
from gpu_fault.app.runtime import EVENT_LOOP_LAG_BUCKETS, EventLoopLag
from tests._builders import asgi_client, build_store


def test_observations_land_in_cumulative_buckets() -> None:
    lag = EventLoopLag()
    for value in (0.001, 0.004, 0.3, 7.0):
        lag.observe(value)
    rows = dict(lag.histogram())
    assert set(rows) == set(EVENT_LOOP_LAG_BUCKETS)
    assert rows[0.005] == 2
    assert rows[0.25] == 2
    assert rows[0.5] == 3
    assert rows[10.0] == 4, "7 s lands in the 10 s bucket"
    assert lag.snapshot() == (4, 7.305, 7.0)
    assert sum(lag.bucket_counts) == 4
    over = EventLoopLag()
    over.observe(42.0)
    assert sum(over.bucket_counts) == 0, "beyond the last bound: +Inf only"
    assert over.snapshot()[0] == 1


def test_metrics_render_the_lag_histogram_with_a_process_label(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-event-loop-lag-histogram")
    monkeypatch.setenv("GPU_FAULT_PROCESS_METRICS_DIR", "off")
    app = create_app(
        ApplicationContext(
            store=build_store(),
            execution_token="event-loop-lag-histogram-token-" + "x" * 32,
        )
    )

    async def scenario() -> str:
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.65)
            async with asgi_client(app) as client:
                response = await client.get("/metrics")
                assert response.status_code == 200, response.text
                return response.text

    rendered = process_metrics.parse_lines(asyncio.run(scenario()).splitlines())
    _, type_line = rendered.families["gpu_fault_event_loop_lag_seconds"]
    assert type_line == "# TYPE gpu_fault_event_loop_lag_seconds histogram"
    samples = rendered.samples["gpu_fault_event_loop_lag_seconds"]
    buckets = {
        dict(sample.labels)["le"]: float(sample.value)
        for sample in samples
        if sample.name == "gpu_fault_event_loop_lag_seconds_bucket"
    }
    assert set(buckets) == {f"{bound:g}" for bound in EVENT_LOOP_LAG_BUCKETS} | {"+Inf"}
    count = next(
        float(sample.value)
        for sample in samples
        if sample.name == "gpu_fault_event_loop_lag_seconds_count"
    )
    assert count >= 1 and buckets["+Inf"] == count
    assert buckets["10"] <= buckets["+Inf"]
    assert all(dict(sample.labels).get("process") == "0" for sample in samples), (
        "every series of the family carries the process slot"
    )
    names = {sample.name for sample in samples}
    assert "gpu_fault_event_loop_lag_seconds_max" in names
    assert "gpu_fault_event_loop_lag_seconds_sum" in names
    assert "gpu_fault_process_resident_memory_bytes" in rendered.families
    assert "gpu_fault_gc_collections_total" in rendered.families
