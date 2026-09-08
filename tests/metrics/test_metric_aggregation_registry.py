"""Every family /metrics can render has a Pod-aggregation strategy, and the
registry names nothing the code no longer renders.

The ingress and control-worker Pods run four uvicorn processes behind one
port; :mod:`gpu_fault.app.process_metrics` merges every live process's render
before answering ADOT, and the merge rule per family is a fact about the
family's meaning that :mod:`gpu_fault.app.metric_aggregation` has to declare.
A family without a rule falls back to a type-based guess and logs; this test
turns that guess into a red run.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from gpu_fault.app import (
    ApplicationContext,
    create_app,
    metric_aggregation,
    process_metrics,
)
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from tests._builders import asgi_client, build_store

ROLES = ("ingress", "worker", "spool-worker")
APP_SOURCES = Path(__file__).resolve().parents[2] / "src" / "gpu_fault" / "app"


def _scrape(monkeypatch, role: str) -> str:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_WORKERS", "24")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", f"pod-aggregation-registry-{role}")
    monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL", raising=False)
    if role == "spool-worker":
        monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    app = create_app(
        ApplicationContext(
            store=build_store(),
            execution_token="metric-aggregation-registry-token-" + "x" * 32,
        )
    )

    async def run() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    return asyncio.run(run())


@pytest.fixture(scope="module")
def rendered_families(request) -> dict[str, set[str]]:
    """Family name -> roles that rendered it, from a real scrape per role."""

    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    monkeypatch.setenv("GPU_FAULT_PROCESS_METRICS_DIR", "off")
    monkeypatch.setenv(
        "GPU_FAULT_PROCESSOR_REPLAY_SECRET", "test-processor-replay-secret-" + "r" * 32
    )
    families: dict[str, set[str]] = {}
    for role in ROLES:
        parsed = process_metrics.parse_lines(_scrape(monkeypatch, role).splitlines())
        for family in parsed.families:
            families.setdefault(family, set()).add(role)
    return families


def test_every_rendered_family_has_a_registered_strategy(rendered_families) -> None:
    unregistered = sorted(
        family for family in rendered_families if family not in STRATEGIES
    )
    assert not unregistered, (
        "families rendered on /metrics without an aggregation strategy "
        f"(add them to gpu_fault.app.metric_aggregation): {unregistered}"
    )
    assert len(rendered_families) > 200, "the fixture did not render a full app"
    # The three roles between them exercise every registration path.
    assert {"ingress", "worker", "spool-worker"} == set().union(
        *rendered_families.values()
    )


def test_every_registered_strategy_is_a_known_one() -> None:
    for family, strategy in STRATEGIES.items():
        assert isinstance(strategy, Strategy), (family, strategy)
        assert family.startswith("gpu_fault_"), family


def _declared_in_source() -> set[str]:
    """Family names the app package still spells out somewhere.

    ``# TYPE <name>`` literals plus every quoted ``gpu_fault_*`` string, which
    covers the families built from tuples (pool statistics, credential
    counters, consumer liveness) and the summaries rendered by suffix.
    """

    declared: set[str] = set()
    for path in APP_SOURCES.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        declared.update(re.findall(r"# TYPE (gpu_fault_[a-z_]+)", text))
        declared.update(re.findall(r'"(gpu_fault_[a-z_]+)"', text))
    return declared


def test_registry_names_nothing_the_code_no_longer_renders(rendered_families) -> None:
    """A family deleted from a contributor must leave the registry with it.

    Conditionally rendered families (Postgres pool statistics, the Aurora
    refresher status file, the regional registry, the spare-health
    controller) do not show up in an in-memory render, so the check accepts a
    name the app package still declares as a literal.
    """

    declared = _declared_in_source() | set(rendered_families)
    declared |= set(metric_aggregation.MERGER_FAMILIES)
    stale = sorted(family for family in STRATEGIES if family not in declared)
    assert not stale, f"registry entries no source file declares any more: {stale}"


def test_alert_rules_agree_with_the_pod_level_strategy() -> None:
    """The rules that read a per-Pod flag with ``min by (pod)`` need the
    Pod's own value to be the minimum over its processes, and the ones that
    ``sum`` a counter need SUM. A drift here is a silent alert."""

    assert STRATEGIES["gpu_fault_processor_healthy"] is Strategy.MIN
    assert STRATEGIES["gpu_fault_processor_consumer_running"] is Strategy.MIN
    assert STRATEGIES["gpu_fault_telemetry_spool_consumer_running"] is Strategy.MIN
    assert (
        STRATEGIES["gpu_fault_processor_consumer_last_cycle_age_seconds"]
        is Strategy.MAX
    )
    assert STRATEGIES["gpu_fault_processor_active_consumer"] is Strategy.SUM
    assert STRATEGIES["gpu_fault_store_io_in_flight"] is Strategy.SUM
    assert STRATEGIES["gpu_fault_store_io_max_in_flight"] is Strategy.SUM
    assert STRATEGIES["gpu_fault_workflow_lifetime_exceeded_total"] is Strategy.SUM
    for family, strategy in STRATEGIES.items():
        if family.endswith("_timestamp_seconds"):
            assert strategy in (Strategy.MAX, Strategy.ANY), family
        if family.endswith("_total") and strategy is Strategy.PER_PROCESS:
            pytest.fail(f"a counter must not be exported per process: {family}")
