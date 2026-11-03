"""COLLECT-014 establishes its fail-closed premise by a bounded wait.

Live 2026-11-03 (HyperPod, round 15) the single premise read found a
latest-metric row the collector had stopped refreshing 148 s older than the
back-dated event -- inside ingest's 600 s window -- and the case FAILED on the
refusal. The case now re-reads with a fresh event time until the premise
holds, posts on that read, and is INCONCLUSIVE when the bound ends.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from gpu_fault.app import ApplicationContext
from scripts.e2e.regional import collect014_premise as premise
from scripts.e2e.regional import run_collector_destructive as destructive
from tests.regional._collector_reset_support import reset_documents

START = datetime(2026, 11, 3, 6, 30, 7, tzinfo=timezone.utc)
AGE = destructive.FAIL_CLOSED_EVENT_AGE
FROZEN = "pcie_link_downtrain@GPU-a"


class Clock:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.at = START
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.at += timedelta(seconds=seconds)


def inventory(now: datetime, *frozen: datetime) -> dict[str, Any]:
    """The live shape: a 60 s snapshot, regularly refreshed rows, frozen rows."""

    rows = [(now - timedelta(seconds=15), f"gpu_temperature_c@GPU-{n}") for n in "ab"]
    rows += [(at, FROZEN) for at in frozen]
    return {
        "present": True,
        "observed_at": (now - timedelta(seconds=47)).isoformat(),
        "snapshot_max_age_seconds": 180.0,
        "legacy_max_age_seconds": 600.0,
        "legacy_observed_at": [at.isoformat() for at, _ in rows],
        "legacy_rows": [name for _, name in rows],
    }


def test_a_frozen_row_is_waited_out_and_posted_on_a_fresh_event_time() -> None:
    clock, lines = Clock(), []
    frozen = START - AGE - timedelta(seconds=148)  # the live offset

    result = premise.establish_fail_closed_premise(
        lambda: inventory(clock.now(), frozen),
        event_age=AGE,
        clock=clock,
        log=lines.append,
    )

    # 148 s older needs 660 s: 512 s more, so the 18th re-read (540 s) holds.
    assert result.refusal is None, result.refusal
    assert result.posted_at == START + timedelta(seconds=540)
    assert result.event_time == result.posted_at - AGE, "event not dated afresh"
    assert clock.sleeps == [30.0] * 18
    assert len(result.reads) == 19 and not result.reads[-1]["errors"], result.reads
    assert result.reads[0]["rows"] == [FROZEN], "the refusal names its row"
    assert "-148s from the back-dated event" in result.reads[0]["errors"][0]
    assert "waiting, re-reading every 30 s for at most 750 s" in lines[0], lines
    assert all("still does not hold" in line for line in lines[1:-1]), lines
    assert lines[-1].endswith("holds after 540 s; posting the SXID"), lines
    evidence = result.evidence()["fail_closed_premise"]
    assert evidence["held"] is True and evidence["waited_seconds"] == 540.0
    assert evidence["bound_seconds"] == 750.0


def test_a_premise_that_never_holds_within_the_bound_is_inconclusive() -> None:
    clock, lines = Clock(), []

    # A row rewritten 45 s after every event time never leaves the band.
    result = premise.establish_fail_closed_premise(
        lambda: inventory(clock.now(), clock.now() - AGE + timedelta(seconds=45)),
        event_age=AGE,
        clock=clock,
        log=lines.append,
    )

    assert result.refusal is not None, "a premise that never held was let through"
    assert result.refusal.startswith(
        "the fail-closed premise did not hold within 750 s (26 reads): "
    ), result.refusal
    assert f"(rows: {FROZEN})" in result.refusal, result.refusal
    assert sum(clock.sleeps) == 750.0, "the wait must end on the bound, not past it"
    assert result.posted_at == START + timedelta(seconds=750)
    assert lines[-1].startswith("read 26 at "), lines[-1]
    assert "INCONCLUSIVE" in lines[-1], lines[-1]


def test_the_premise_holding_at_once_posts_without_a_wait() -> None:
    clock, lines = Clock(), []
    result = premise.establish_fail_closed_premise(
        lambda: inventory(clock.now()), event_age=AGE, clock=clock, log=lines.append
    )
    assert result.refusal is None and clock.sleeps == [], (result, clock.sleeps)
    assert lines == [
        "read 1 at 2026-11-03T06:30:07+00:00, event 2026-11-03T06:22:07+00:00: "
        "holds at the first read; posting the SXID"
    ]


def test_no_wait_bound_without_the_products_max_ages() -> None:
    clock, lines = Clock(), []
    stale = {
        "present": True,
        "observed_at": (START - AGE + timedelta(seconds=45)).isoformat(),
        "legacy_observed_at": [],
    }
    result = premise.establish_fail_closed_premise(
        lambda: stale, event_age=AGE, clock=clock, log=lines.append
    )
    assert result.refusal is not None, "an unbounded premise must not be posted"
    assert result.refusal.endswith(
        "no wait bound: the read has no usable product max age"
    ), result.refusal
    assert clock.sleeps == [] and len(result.reads) == 1, clock.sleeps


@pytest.mark.parametrize(
    ("ages", "expected"),
    [
        ({"snapshot_max_age_seconds": 180.0, "legacy_max_age_seconds": 600.0}, 750),
        ({"snapshot_max_age_seconds": 180.0}, 330),
        ({"legacy_max_age_seconds": True}, None),
        ({}, None),
    ],
)
def test_the_wait_bound_is_the_whole_refusal_band(
    ages: dict[str, Any], expected: int | None
) -> None:
    bound = premise.premise_wait_bound(ages)
    assert (None if bound is None else bound.total_seconds()) == expected


def test_the_probe_names_each_latest_row_beside_its_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def row(name: str, gpu: str | None, at: datetime) -> SimpleNamespace:
        return SimpleNamespace(
            observed_at=at, sample=SimpleNamespace(canonical_name=name, gpu_uuid=gpu)
        )

    rows = [
        row("pcie_link_downtrain", "GPU-a", START - timedelta(seconds=628)),
        row("host_cpu", None, START),
        row("gpu_temperature_c", "GPU-b", START),
    ]
    store = SimpleNamespace(
        get_gpu_inventory_snapshot=lambda cluster, node: None,
        list_gpu_metrics_latest=lambda cluster, node: rows,
    )
    context = SimpleNamespace(
        store=store,
        gpu_inventory_max_age=timedelta(seconds=180),
        legacy_gpu_metrics_inventory_max_age=timedelta(seconds=600),
    )
    monkeypatch.setattr(ApplicationContext, "from_environment", lambda: context)
    monkeypatch.setattr(sys, "argv", ["probe", "cluster-a", "node-a"])
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(premise.GPU_INVENTORY_SNAPSHOT, "<gpu-inventory>", "exec"), {})
    read = json.loads(output.getvalue())

    assert read["legacy_observed_at"] == [
        (START - timedelta(seconds=628)).isoformat(),
        START.isoformat(),
    ]
    assert read["legacy_rows"] == [
        "pcie_link_downtrain@GPU-a",
        "gpu_temperature_c@GPU-b",
    ]
    assert read["present"] is False and read["legacy_max_age_seconds"] == 600.0


def _run_collect014(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reads: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Clock]:
    """``run_collect014`` with the premise reads scripted and the rest stubbed."""

    before, _after, _state = reset_documents("RESET_ALL_GPUS_NVSWITCHES")
    clock, posts, seen = Clock(), [], {}

    def wait_marker(marker: str, **kwargs: Any) -> dict[str, Any]:
        seen["observed_after"] = kwargs.get("observed_after")
        return {
            "workflows": [
                {
                    "status": "BLOCKED",
                    "blocked_reasons": [f"SXID 10003 {destructive.FAIL_CLOSED_REASON}"],
                    "official_action": "RESET_ALL_GPUS_AND_NVSWITCHES",
                }
            ]
        }

    monkeypatch.setattr(
        destructive,
        "post_fabric_event",
        lambda regional, payload: posts.append(payload),
    )
    monkeypatch.setattr(
        destructive,
        "run_full_reset_variant",
        lambda *a, sxid, **k: {
            "verdict": "PASS",
            "errors": [],
            "marker": f"marker-{sxid}",
            "state": {},
            "restore_workflows": [],
        },
    )
    monkeypatch.setattr(
        destructive, "wait_full_reset_stability", lambda *a: [{"host": before}]
    )
    result = destructive.run_collect014(
        SimpleNamespace(
            node="node-a",
            case_id="GF-REGIONAL-COLLECT-014",
            regional=SimpleNamespace(cluster_id="cluster-a"),
        ),
        SimpleNamespace(cpu_python=lambda script, *a, **k: reads(clock.now())),
        SimpleNamespace(execute=lambda *a, **k: before),
        SimpleNamespace(execute=lambda *a, **k: before, wait_marker=wait_marker),
        tmp_path,
        1,
        "profile-a",
        cleanup=Mock(restore=Mock(return_value=[])),
        clock=clock,
    )
    return result, posts, seen, clock


def test_collect014_waits_for_the_premise_then_posts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = [inventory(START, START - AGE - timedelta(seconds=148))]

    def reads(now: datetime) -> dict[str, Any]:
        # Stale first, then the frozen row is written again (fresh).
        return stale.pop() if stale else inventory(now, now - timedelta(seconds=5))

    result, posts, seen, clock = _run_collect014(tmp_path, monkeypatch, reads)

    assert result["verdict"] == "PASS", result["errors"]
    assert clock.sleeps == [30.0], clock.sleeps
    assert len(posts) == 1, "the SXID is posted once, after the wait"
    retried = START + timedelta(seconds=30)
    assert posts[0]["observed_at"] == (retried - AGE).isoformat(), posts[0]
    assert seen["observed_after"] == retried, seen
    record = result["fail_closed_premise"]
    assert record["held"] is True and len(record["reads"]) == 2, record
    assert record["reads"][0]["rows"] == [FROZEN], record["reads"][0]


def test_collect014_is_inconclusive_when_the_premise_never_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reads(now: datetime) -> dict[str, Any]:
        return inventory(now, now - AGE + timedelta(seconds=45))

    result, posts, _seen, clock = _run_collect014(tmp_path, monkeypatch, reads)

    assert result["verdict"] == "INCONCLUSIVE", result
    assert result["errors"] == [] and posts == [], (result["errors"], posts)
    assert "did not hold within 750 s" in result["inconclusive_reason"], result
    assert result["fail_closed"] is None and result["positive"] is None, result
    assert result["fail_closed_premise"]["held"] is False, result
    assert sum(clock.sleeps) == 750.0, clock.sleeps
