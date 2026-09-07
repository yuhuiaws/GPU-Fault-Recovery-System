"""The processor counter mode is exported, one series per known mode.

Store review 2026-09-07, item I. Production runs the processor admission
counters in ``dual`` mode: the single per-cluster row of
``gpu_fault_processor_queue_counts`` is still locked on every enqueue and the
16 priority shards are extra writes, not contention relief. Nothing exported
the mode, so nobody knew. ``gpu_fault_processor_counter_mode{mode=...}`` is 1
for the active mode and 0 for the others; only the Postgres store has the
accessor, so SQLite and memory stores emit no series at all.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.builtin_metric_contributors import control_loop_metric_lines
from tests._builders import asgi_client, build_store

HELP = (
    "# HELP gpu_fault_processor_counter_mode Processor admission counter mode; "
    "the 16 priority shards relieve counter-row contention only in partitioned "
    "mode; finalize requires an empty queue "
    "(gpu-fault-store-migrate --finalize-processor-counter-shards)"
)


def _lines(store: object) -> list[str]:
    runtime = SimpleNamespace(context=SimpleNamespace(store=store))
    return control_loop_metric_lines(runtime)


def test_the_active_mode_is_one_and_the_other_known_mode_is_zero() -> None:
    store = SimpleNamespace(processor_counter_mode=lambda: "dual")

    lines = _lines(store)

    assert HELP in lines
    assert "# TYPE gpu_fault_processor_counter_mode gauge" in lines
    assert 'gpu_fault_processor_counter_mode{mode="dual"} 1' in lines
    assert 'gpu_fault_processor_counter_mode{mode="partitioned"} 0' in lines


def test_partitioned_flips_the_series() -> None:
    lines = _lines(SimpleNamespace(processor_counter_mode=lambda: "partitioned"))

    assert 'gpu_fault_processor_counter_mode{mode="dual"} 0' in lines
    assert 'gpu_fault_processor_counter_mode{mode="partitioned"} 1' in lines


def test_an_unknown_mode_gets_its_own_series_instead_of_all_zeros() -> None:
    lines = _lines(SimpleNamespace(processor_counter_mode=lambda: "odd"))

    assert 'gpu_fault_processor_counter_mode{mode="dual"} 0' in lines
    assert 'gpu_fault_processor_counter_mode{mode="partitioned"} 0' in lines
    assert 'gpu_fault_processor_counter_mode{mode="odd"} 1' in lines


def test_a_store_without_the_accessor_emits_no_series() -> None:
    text = "\n".join(_lines(build_store()))

    assert "gpu_fault_processor_counter_mode" not in text, text


def test_the_memory_backed_metrics_endpoint_omits_the_family() -> None:
    app = create_app(ApplicationContext(store=build_store()))

    async def fetch() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    metrics = asyncio.run(fetch())

    assert "gpu_fault_processor_counter_mode" not in metrics


@pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)
def test_the_postgres_store_reports_dual_until_finalized() -> None:
    from tests.store._postgres_processor_claim_support import postgres_store_instance

    for store in postgres_store_instance():
        assert store.processor_counter_mode() == "dual"
        text = "\n".join(_lines(store))
        assert 'gpu_fault_processor_counter_mode{mode="dual"} 1' in text, text
        assert 'gpu_fault_processor_counter_mode{mode="partitioned"} 0' in text

        # The fixture truncates the queue, so the switch is allowed here.
        store.finalize_processor_counter_shards()
        assert store.processor_counter_mode() == "partitioned"
        text = "\n".join(_lines(store))
        assert 'gpu_fault_processor_counter_mode{mode="partitioned"} 1' in text
        assert 'gpu_fault_processor_counter_mode{mode="dual"} 0' in text

        store.restore_legacy_processor_counters()
        assert store.processor_counter_mode() == "dual"
