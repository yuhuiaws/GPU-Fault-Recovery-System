"""The workflow-dispatch lease and tick families on /metrics (b226).

A perf run could not tell whether the starved dispatch-lease holder lost its
lease mid-tick: no family said how long a tick ran or when the lease changed
hands. These render next to the other ``gpu_fault_workflow_dispatch_*``
families on every process, aggregate per process across a Pod (the held gauge
adds up to the Pod's number of holders) and pass the ADOT keep list.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from gpu_fault.app import ApplicationContext, create_app, process_metrics
from gpu_fault.app.dispatch_lease_metrics import dispatch_lease_metric_lines
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.execution.dispatch_lease import DispatchLease
from tests._builders import asgi_client, build_store
from tests.metrics._assertions import assert_samples

ADOT_CONTROL_PLANE = (
    Path(__file__).resolve().parents[2] / "deploy/observability/adot-control-plane.yaml"
)
TICK = "gpu_fault_workflow_dispatch_tick_seconds"
ACQUISITIONS = "gpu_fault_workflow_dispatch_lease_acquisitions_total"
LOSSES = "gpu_fault_workflow_dispatch_lease_losses_total"
HELD = "gpu_fault_workflow_dispatch_lease_held"
KEY = "workflow-dispatch"


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime.now(timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _holder() -> tuple[DispatchLease, Any]:
    """A holder that ran a 3-s and a 20-s tick, renewing at the start of each,
    on a store shared with nobody: its 15-s lease ran out during the second."""

    store = build_store()
    lease = DispatchLease(KEY, 15, clock=FakeClock())
    for seconds in (3, 20):
        assert lease.acquire(store, "executor-a"), "the holder lost its lease"
        with lease.holding_tick(lease.clock()):
            lease.clock.advance(seconds)
    return lease, store


def _render(lease: DispatchLease) -> list[str]:
    return dispatch_lease_metric_lines(SimpleNamespace(lease=lease))


def test_a_process_that_never_held_the_lease_renders_every_family_at_zero() -> None:
    for dispatcher in (None, object()):
        lines = dispatch_lease_metric_lines(dispatcher)
        parsed = process_metrics.parse_lines(lines)

        assert {TICK, ACQUISITIONS, LOSSES, HELD} <= set(parsed.families), lines
        assert (parsed.family_type(TICK), parsed.family_type(HELD)) == (
            "histogram",
            "gauge",
        )
        assert_samples(
            "\n".join(lines),
            (
                f'{TICK}_bucket{{le="0.5"}} 0',
                f'{TICK}_bucket{{le="15"}} 0',
                f'{TICK}_bucket{{le="+Inf"}} 0',
                f"{TICK}_count 0",
                f"{TICK}_sum 0",
                f"{ACQUISITIONS} 0",
                f'{LOSSES}{{when="mid_tick"}} 0',
                f'{LOSSES}{{when="tick_start"}} 0',
                f"{HELD} 0",
            ),
        )


def test_the_families_follow_the_holder_and_its_clock() -> None:
    lease, store = _holder()

    expired = _render(lease)
    assert lease.acquire(store, "executor-a"), "the lapsed lease was not re-taken"
    held = _render(lease)

    assert_samples(
        "\n".join(expired),
        (
            f'{TICK}_bucket{{le="2.5"}} 0',
            f'{TICK}_bucket{{le="5"}} 1',
            f'{TICK}_bucket{{le="15"}} 1',
            f'{TICK}_bucket{{le="30"}} 2',
            f'{TICK}_bucket{{le="120"}} 2',
            f'{TICK}_bucket{{le="+Inf"}} 2',
            f"{TICK}_count 2",
            f"{TICK}_sum 23",
            f"{ACQUISITIONS} 1",
            f"{HELD} 0",
        ),
    )
    assert_samples("\n".join(held), (f"{ACQUISITIONS} 2", f"{HELD} 1"))


def test_the_lease_families_aggregate_per_process_and_the_held_gauge_sums() -> None:
    assert {
        family: STRATEGIES.get(family) for family in (TICK, ACQUISITIONS, LOSSES, HELD)
    } == {
        TICK: Strategy.PER_PROCESS,
        ACQUISITIONS: Strategy.PER_PROCESS,
        LOSSES: Strategy.PER_PROCESS,
        HELD: Strategy.SUM,
    }
    lease, store = _holder()
    assert lease.acquire(store, "executor-a"), "the lapsed lease was not re-taken"
    holder = process_metrics.parse_lines(_render(lease))
    holder.slot, holder.pid = 0, 100
    idle = process_metrics.parse_lines(dispatch_lease_metric_lines(None))
    idle.slot, idle.pid = 1, 200

    merged = "\n".join(process_metrics.aggregate(idle, [holder]))

    assert_samples(
        merged,
        (
            f"{HELD} 1",
            f'{ACQUISITIONS}{{process="0"}} 2',
            f'{ACQUISITIONS}{{process="1"}} 0',
            f'{LOSSES}{{when="mid_tick",process="0"}} 0',
            f'{TICK}_bucket{{le="30",process="0"}} 2',
            f'{TICK}_bucket{{le="30",process="1"}} 0',
            f'{TICK}_count{{process="0"}} 2',
        ),
    )


def _adot_keep_filter() -> str:
    documents = yaml.safe_load_all(ADOT_CONTROL_PLANE.read_text(encoding="utf-8"))
    config = next(document for document in documents if document["kind"] == "ConfigMap")
    scrape = yaml.safe_load(config["data"]["collector.yaml"])["receivers"][
        "prometheus"
    ]["config"]["scrape_configs"][0]
    return next(
        item["regex"]
        for item in scrape["metric_relabel_configs"]
        if item["action"] == "keep" and item["source_labels"] == ["__name__"]
    )


def test_every_lease_and_tick_series_passes_the_adot_keep_list() -> None:
    """``gpu_fault_workflow_.+`` is what ships them to AMP, suffixes included."""

    keep = _adot_keep_filter()
    lease, _store = _holder()
    parsed = process_metrics.parse_lines(_render(lease))
    names = {
        sample.name
        for family in (TICK, ACQUISITIONS, LOSSES, HELD)
        for sample in parsed.samples[family]
    }

    assert {f"{TICK}_bucket", f"{TICK}_sum", f"{TICK}_count"} <= names, names
    assert sorted(name for name in names if not re.fullmatch(keep, name)) == []


def test_the_worker_scrape_renders_the_lease_families(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-dispatch-lease-metrics")
    monkeypatch.setenv("GPU_FAULT_PROCESS_METRICS_DIR", "off")
    context = ApplicationContext(
        execution_token="dispatch-lease-metrics-token-" + "x" * 32
    )
    app = create_app(context)
    lease, store = _holder()
    assert lease.acquire(store, "executor-a"), "the lapsed lease was not re-taken"
    context.dispatcher.lease = lease

    async def scrape() -> str:
        async with asgi_client(app) as client:
            return (await client.get("/metrics")).text

    text = asyncio.run(scrape())

    assert_samples(
        text,
        (
            f'{TICK}_count{{process="0"}} 2',
            f'{ACQUISITIONS}{{process="0"}} 2',
            f'{LOSSES}{{when="tick_start",process="0"}} 0',
            f"{HELD} 1",
        ),
    )
