"""The shared detection budgets of the GF-GENERAL-LIVE harness, on the product's cadence.

Live 2026-10-25 (003 high_page_cache, release 13daa08b2227): the host collector
ships a sustained breach once, as the crossing's edge, then only on its health
summaries; the control plane mints the finding on the first later active sample
at least the rule's sustain after the first; the validation is judged on one
more summary. A budget counted from the load's start, sized from fixed numbers,
ended before that second summary. Every variant whose PASS needs a sustained
host rule now times detection from its oracle's crossing over
``sustained_detection_budget``; these tests pin the formula to the product's
rules and the site's cadence, and the shared crossing wait to its live shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.collectors.cadence import CadenceJitter
from gpu_fault.host_health import NodeHealthPolicy
from scripts.e2e.regional import general_live_common as common
from scripts.e2e.regional import general_live_host as host
from scripts.e2e.regional import general_live_memory as mem
from scripts.e2e.regional import general_live_network as net
from scripts.e2e.regional import general_live_utilization as util
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests._builders import build_store
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    store_poll_clock,
    store_script,
)

#: The live site: 15-s host collector interval, 300-s health summaries.
SITE = SimpleNamespace(host_summary=300, host_interval=15)
SLOW_SITE = SimpleNamespace(host_summary=600, host_interval=30)
#: The product's +-5 % phase move: two consecutive summaries are at most
#: 1.1 x the interval apart (``collectors/cadence.py``).
JITTER = CadenceJitter.from_environment({})
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def test_the_budget_is_sustain_plus_two_summaries_an_interval_and_the_margin() -> None:
    assert common.sustained_detection_budget(SITE, 120) == 120 + 2 * 300 + 15 + 180
    assert common.sustained_detection_budget(SITE, 60) == 60 + 2 * 300 + 15 + 180
    # A rule with no sustain mints on its edge batch: only the validation's
    # summary is still ahead.
    assert common.sustained_detection_budget(SITE, 0, summaries=1) == 300 + 15 + 180
    # Read from the node's collector environment, not from fixed numbers.
    assert common.sustained_detection_budget(SLOW_SITE, 120) == 120 + 2 * 600 + 30 + 180
    assert common.sustained_negative_window(SITE, 120) == 120 + 300 + 15 + 180
    assert common.sustained_negative_window(SLOW_SITE, 60) == 60 + 600 + 30 + 180


@pytest.mark.parametrize("sustain", [0.0, 60.0, 120.0, 300.0])
def test_the_budget_covers_two_jittered_summaries(sustain: float) -> None:
    # Each summary may come 1.1 x the cadence after the previous one; the
    # margin absorbs that on top of ingest and dispatch.
    for site in (SITE, SLOW_SITE):
        budget = common.sustained_detection_budget(site, sustain)
        assert budget >= sustain + 2 * JITTER.max_interval(site.host_summary) + (
            site.host_interval
        ), (site, sustain, budget)
        window = common.sustained_negative_window(site, sustain)
        assert window >= sustain + JITTER.max_interval(site.host_summary) + (
            site.host_interval
        ), (site, sustain, window)


def test_every_variant_budget_is_the_shared_formula_on_the_products_rule() -> None:
    rules = NodeHealthPolicy.METRIC_RULE_SUSTAIN_SECONDS
    # 001 cpu_usage_percent / load1_per_cpu and 002 memory_used_percent share
    # HOST_RESOURCE_SUSTAIN_SECONDS; 009 network_drops_delta has its own.
    assert host.SUSTAIN_SECONDS == rules["cpu_usage_percent"] == rules["load1_per_cpu"]
    assert mem.SUSTAIN_SECONDS == rules["memory_used_percent"]
    assert net.DROPS_SUSTAIN == rules["network_drops_delta"]
    assert host.detection_budget(SITE) == common.sustained_detection_budget(
        SITE, NodeHealthPolicy.HOST_RESOURCE_SUSTAIN_SECONDS
    )
    assert mem.detection_budget(SITE) == host.detection_budget(SITE) == 915
    assert net.drops_budget(SITE) == common.sustained_detection_budget(
        SITE, NodeHealthPolicy.NETWORK_DROPS_SUSTAIN_SECONDS
    )
    assert net.drops_budget(SITE) == 855
    # swap_used_percent has no sustain (host_health.py METRIC_RULES only).
    assert "swap_used_percent" not in rules, "the swap rule gained a sustain"
    assert mem.swap_detection_budget(SITE) == 495
    # 003 high_page_cache (commit 7e1c6c4) is the same formula on its 300-s rule.
    duration = NodeHealthPolicy(build_store()).page_cache_duration_seconds
    assert util.page_cache_hold(SITE) == common.sustained_detection_budget(
        SITE, duration
    )
    assert util.page_cache_hold(SITE) == 1095
    # The negative windows: the sustain plus one summary after the burst.
    assert host.quiet_window(SITE) == 615
    assert net.drops_quiet_window(SITE) == 555
    assert net.link_quiet_window(SITE) == 495


def test_the_new_budgets_outlast_the_fixed_ones_that_failed_live() -> None:
    # 001: 120 + 300 + 90 from the load's start; 002: 120 + 300 + 120 from the
    # crossing; 009: 60 + 300 + 90 from the generator's start. None reached the
    # summary the validation is judged on.
    assert host.detection_budget(SITE) > 120 + 300 + 90
    assert mem.detection_budget(SITE) > 120 + 300 + 120
    assert net.drops_budget(SITE) > 60 + 300 + 90


def test_cadence_record_names_the_budgets_and_their_source() -> None:
    record = common.cadence_record(
        SITE, sustain_seconds=120, detection_budget_seconds=915, ramp_budget_seconds=90
    )
    assert record == {
        "sustain_seconds": 120,
        "detection_budget_seconds": 915,
        "host_summary_seconds": 300,
        "host_interval_seconds": 15,
        "ramp_budget_seconds": 90,
    }
    assert "ramp_budget_seconds" not in common.cadence_record(
        SITE, sustain_seconds=0, detection_budget_seconds=495
    )


# --------------------------------------------------------------------------- #
# await_oracle_crossing: the runner's oracle decides when detection is timed
# --------------------------------------------------------------------------- #
def oracle_reads(*phases: list[float]) -> list[dict[str, Any]]:
    """One ``oracle-read`` answer per phase: the readings taken since the last read."""

    entries: list[dict[str, Any]] = []
    total = 0
    for values in phases:
        total += len(values)
        entries.append(
            {
                "records": [{"cpu_busy_percent": value} for value in values],
                "next_offset": total,
                "total": total,
            }
        )
    return entries


def crossing_context(tmp_path: Path, reads: list[dict[str, Any]]) -> Any:
    spec = next(spec for spec in host.VARIANTS if spec.variant == "cpu_busy")
    probe = FakeProbe({"oracle-start": {"unit": "oracle"}, "oracle-read": reads})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": {}})),
        probes={f"node@{NODE}": probe},
    )
    return store_poll_clock(ctx)


def crossing(ctx: Any) -> list[float]:
    oracle = common.Oracle(ctx, "cpu")
    oracle.start(600, cpu=True)
    return common.await_oracle_crossing(
        ctx,
        oracle,
        lambda record: record.get("cpu_busy_percent"),
        lambda values: bool(values) and values[-1] >= 98.0,
        timeout_seconds=120,
        name="cpu-ramp",
        condition="the load did not bring cpu_busy_percent to 98 %",
    )


def test_the_crossing_is_the_oracles_first_reading_at_the_threshold(
    tmp_path: Path,
) -> None:
    ctx = crossing_context(
        tmp_path, oracle_reads([40.0] * 3, [70.0] * 3, [98.5, 99.0], [99.0] * 10)
    )
    values = crossing(ctx)
    assert values[-1] == 99.0 and len(values) == 8, values
    timeline = json.loads((ctx.case_dir / "cpu-ramp-timeline.json").read_text())
    assert [entry["accepted"] for entry in timeline["entries"]] == [False, False, True]
    # Every reading the wait consumed is in the oracle evidence.
    lines = (ctx.case_dir / "oracle.jsonl").read_text().splitlines()
    assert len(lines) == 8, lines


def test_no_crossing_is_inconclusive_never_a_verdict(tmp_path: Path) -> None:
    ctx = crossing_context(tmp_path, oracle_reads([40.0] * 3, [71.2] * 3))
    with pytest.raises(VariantInconclusive) as raised:
        crossing(ctx)
    message = str(raised.value)
    assert message.startswith("condition not established: the load did not bring"), (
        message
    )
    assert "within 120 s" in message and "71.2" in message, message


def test_no_reading_at_all_is_inconclusive(tmp_path: Path) -> None:
    ctx = crossing_context(tmp_path, [{"records": [], "next_offset": 0, "total": 0}])
    with pytest.raises(VariantInconclusive, match=r"\(last \[\]\)"):
        crossing(ctx)
