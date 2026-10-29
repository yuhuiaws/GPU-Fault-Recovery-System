"""The processor queue caps admission enforces are exported next to the depths.

The queue depth alerts divide the depths by these gauges. They used to compare
with 7000 / 9000 / 700 -- 70 / 90 / 70 percent of the code defaults -- while
the regional manifests render caps of 65536 / 4096, so the warnings fired at
11-17 percent of the real caps and "exceeds 70 percent" was false. The
behaviour of the rules themselves is pinned with promtool in
``tests/test_alert_rules_promtool.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.app.process_metrics import aggregate, parse_lines
from tests._builders import asgi_client, build_store
from tests.metrics._assertions import assert_sample

ROOT = Path(__file__).resolve().parents[2]
DASHBOARDS = ROOT / "deploy/observability/dashboards"
LIMIT = "gpu_fault_processor_queue_depth_limit"
RESERVE = "gpu_fault_processor_queue_fault_reserved_depth"
CLUSTER_LIMIT = "gpu_fault_processor_cluster_queue_depth_limit"
CLUSTER_RESERVE = "gpu_fault_processor_cluster_queue_fault_reserved_depth"
#: What the regional manifests render (deploy/control-plane/regional/generated).
RENDERED = {
    "GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH": "65536",
    "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": "4096",
    "GPU_FAULT_PROCESSOR_FAULT_RESERVED_QUEUE_DEPTH": "8192",
    "GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH": "512",
}


def _scrape(app) -> str:
    async def scenario() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        pytest.param(
            RENDERED,
            {LIMIT: 65536, RESERVE: 8192, CLUSTER_LIMIT: 4096, CLUSTER_RESERVE: 512},
            id="regional-manifests",
        ),
        # Without the environment the reserves derive from the caps (2/5).
        pytest.param(
            {},
            {LIMIT: 10000, RESERVE: 4000, CLUSTER_LIMIT: 1000, CLUSTER_RESERVE: 400},
            id="code-defaults",
        ),
    ],
)
def test_the_gauges_are_the_caps_the_admission_batchers_enforce(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected: dict[str, int],
) -> None:
    for name in RENDERED:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    app = create_app(ApplicationContext(store=build_store()))

    text = _scrape(app)

    for family, value in expected.items():
        assert_sample(text, family, value)
        assert f"# TYPE {family} gauge" in text, family
    batcher = app.state.processor_admission_batcher
    assert {
        LIMIT: batcher.max_depth,
        RESERVE: batcher.reserved_fault_depth,
        CLUSTER_LIMIT: batcher.max_cluster_depth,
        CLUSTER_RESERVE: batcher.reserved_cluster_fault_depth,
    } == expected, "the gauges must read the values admission enforces"


def test_a_pod_of_four_processes_reports_one_cap_not_the_sum() -> None:
    """Every process enforces the whole cap against the same Store-wide count.

    The per-process flush limits are shares and sum; these are not, and a SUM
    would turn a 4096 cap into 16384 and the 70 percent warning into 17.5.
    """

    values = {LIMIT: 65536, RESERVE: 8192, CLUSTER_LIMIT: 4096, CLUSTER_RESERVE: 512}
    lines = [
        line
        for family, value in values.items()
        for line in (f"# TYPE {family} gauge", f"{family} {value}")
    ]
    peers = []
    for slot in range(4):
        peer = parse_lines(lines)
        peer.slot = slot
        peers.append(peer)

    merged = parse_lines(aggregate(peers[0], peers[1:]))

    for family, value in values.items():
        assert STRATEGIES[family] is Strategy.ANY, family
        assert [float(sample.value) for sample in merged.samples[family]] == [value]


def _panels(uid: str) -> dict[str, dict]:
    dashboard = json.loads((DASHBOARDS / f"{uid}.json").read_text(encoding="utf-8"))
    return {
        panel["title"]: panel for panel in dashboard["panels"] if panel["type"] != "row"
    }


def _steps(panel: dict) -> list[tuple[str, float | None]]:
    steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
    return [(step["color"], step["value"]) for step in steps]


@pytest.mark.parametrize(
    "uid", ["gpu-fault-overview", "gpu-fault-control-plane-capacity"]
)
def test_the_queue_panels_draw_shares_of_the_exported_caps(uid: str) -> None:
    panels = _panels(uid)
    regional = panels["Processor queue use"]
    cluster = panels["Processor queue use by cluster"]

    regional_exprs = " ".join(target["expr"] for target in regional["targets"])
    cluster_exprs = " ".join(target["expr"] for target in cluster["targets"])

    assert regional["fieldConfig"]["defaults"]["unit"] == "percentunit"
    assert _steps(regional) == [("green", None), ("orange", 0.7), ("red", 0.9)]
    assert LIMIT in regional_exprs and RESERVE in regional_exprs, regional_exprs
    assert cluster["fieldConfig"]["defaults"]["unit"] == "percentunit"
    assert _steps(cluster) == [("green", None), ("orange", 0.7)]
    assert CLUSTER_LIMIT in cluster_exprs and CLUSTER_RESERVE in cluster_exprs, (
        cluster_exprs
    )
    assert "GpuFaultProcessorClusterBacklogCritical" in cluster["description"]
    for title in ("Processor queue depth", "Processor queue depth by cluster"):
        # Requests, not shares: no ratio threshold may land on these.
        assert _steps(panels[title]) == [("green", None)], title
